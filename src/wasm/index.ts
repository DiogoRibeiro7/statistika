/**
 * WebAssembly acceleration for hot-path operations.
 *
 * Provides near-native performance in environments where the Fortran
 * native addon is not available (browsers, serverless, etc.).
 *
 * Fallback chain: Native Fortran -> WASM -> Pure TypeScript
 *
 * Usage:
 * ```ts
 * import { loadWasm, getWasm } from './wasm';
 *
 * // Async initialization (call once at startup)
 * const wasm = await loadWasm();
 *
 * // Synchronous access after initialization
 * const mod = getWasm();
 * if (mod) {
 *   const result = mod.gammaLn(5.0);
 * }
 * ```
 */

/* eslint-disable @typescript-eslint/no-namespace */

// Minimal WebAssembly type declarations for Node.js environments
// where the DOM lib is not included in tsconfig.
declare namespace WebAssembly {
  interface Module {}
  interface Instance {
    readonly exports: Record<string, unknown>;
  }
  interface Memory {
    readonly buffer: ArrayBuffer;
  }
  interface ResultObject {
    instance: Instance;
    module: Module;
  }
  function instantiate(
    bytes: ArrayBuffer | Uint8Array,
    importObject?: Record<string, Record<string, unknown>>,
  ): Promise<ResultObject>;
}

/* eslint-enable @typescript-eslint/no-namespace */

import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { tsFallback } from "./fallback";

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

/**
 * Interface implemented by both the WASM module and the TypeScript fallback.
 */
export interface WasmModule {
  /** Log-gamma function: ln(Gamma(x)). */
  gammaLn(x: number): number;
  /** Gamma function: Gamma(x). */
  gamma(x: number): number;
  /** Error function: erf(x). */
  erf(x: number): number;
  /** Complementary error function: erfc(x) = 1 - erf(x). */
  erfc(x: number): number;
  /** Beta function: B(a, b) = Gamma(a)*Gamma(b)/Gamma(a+b). */
  betaFn(a: number, b: number): number;

  /**
   * Matrix multiplication: C = A * B.
   * Matrices are stored in row-major order as flat Float64Arrays.
   *
   * @param a - Matrix A (m x k), row-major
   * @param b - Matrix B (k x n), row-major
   * @param m - Number of rows in A
   * @param k - Shared dimension
   * @param n - Number of columns in B
   * @returns Result matrix C (m x n), row-major
   */
  matMul(
    a: Float64Array,
    b: Float64Array,
    m: number,
    k: number,
    n: number,
  ): Float64Array;

  /**
   * Solve a linear system A * x = b using LU decomposition.
   *
   * @param a - Square matrix A (n x n), row-major
   * @param b - Right-hand side vector (n)
   * @param n - System dimension
   * @returns Solution vector x (n)
   */
  solve(a: Float64Array, b: Float64Array, n: number): Float64Array;

  /**
   * Cholesky decomposition: A = L * L^T.
   * Only the lower triangle of A is read.
   *
   * @param a - Symmetric positive-definite matrix (n x n), row-major
   * @param n - Matrix dimension
   * @returns Lower-triangular factor L (n x n), row-major
   */
  cholesky(a: Float64Array, n: number): Float64Array;
}

// ---------------------------------------------------------------------------
// Module state
// ---------------------------------------------------------------------------

/** Whether WASM acceleration is available. */
export let hasWasm = false;

let wasmModule: WasmModule | null = null;
let loadPromise: Promise<WasmModule | null> | null = null;

// ---------------------------------------------------------------------------
// WASM loading
// ---------------------------------------------------------------------------

/**
 * Load the WASM module asynchronously.
 *
 * The result is cached; subsequent calls return the same module.
 * Returns `null` if the WASM binary is not found or instantiation fails.
 * In that case the TypeScript fallback (available via {@link tsFallback})
 * should be used instead.
 */
export async function loadWasm(): Promise<WasmModule | null> {
  if (wasmModule) return wasmModule;
  if (loadPromise) return loadPromise;

  loadPromise = doLoadWasm();
  return loadPromise;
}

/**
 * Get the WASM module synchronously.
 *
 * Returns `null` if {@link loadWasm} has not been called or WASM is
 * unavailable. If WASM is not loaded, returns the TypeScript fallback
 * so callers always get a usable module.
 */
export function getWasm(): WasmModule | null {
  return wasmModule;
}

/**
 * Get an accelerated module — WASM if available, otherwise the pure TS
 * fallback. This always returns a usable module.
 */
export function getAccelerated(): WasmModule {
  return wasmModule ?? tsFallback;
}

// Re-export the fallback for direct access
export { tsFallback } from "./fallback";

// ---------------------------------------------------------------------------
// Internal loader
// ---------------------------------------------------------------------------

async function doLoadWasm(): Promise<WasmModule | null> {
  try {
    const base = typeof __dirname !== "undefined" ? __dirname : process.cwd();
    const wasmPath = join(base, "stats.wasm");

    const wasmBytes = await readFile(wasmPath);
    const wasmImports = {
      env: {
        // C standard math functions needed by the WASM module
        log: Math.log,
        exp: Math.exp,
        sqrt: Math.sqrt,
        sin: Math.sin,
        cos: Math.cos,
        fabs: Math.abs,
        floor: Math.floor,
        ceil: Math.ceil,
        pow: Math.pow,
      },
    };

    const { instance } = await WebAssembly.instantiate(wasmBytes, wasmImports);
    const exports = instance.exports as Record<string, unknown>;
    const memory = exports.memory as WebAssembly.Memory;

    // Wrap the WASM exports into our WasmModule interface
    const mod: WasmModule = {
      gammaLn: exports.gammaLn as (x: number) => number,
      gamma: exports.gamma as (x: number) => number,
      erf: exports.erf as (x: number) => number,
      erfc: exports.erfc as (x: number) => number,
      betaFn: exports.betaFn as (a: number, b: number) => number,

      matMul(
        a: Float64Array,
        b: Float64Array,
        m: number,
        k: number,
        n: number,
      ): Float64Array {
        const wasmMalloc = exports.malloc as (size: number) => number;
        const wasmFree = exports.free as (ptr: number) => void;
        const wasmMatMul = exports.matMul as (
          aPtr: number,
          bPtr: number,
          cPtr: number,
          m: number,
          k: number,
          n: number,
        ) => void;

        const aBytes = a.byteLength;
        const bBytes = b.byteLength;
        const cBytes = m * n * 8;

        const aPtr = wasmMalloc(aBytes);
        const bPtr = wasmMalloc(bBytes);
        const cPtr = wasmMalloc(cBytes);

        const memView = new Float64Array(memory.buffer);
        memView.set(a, aPtr / 8);
        memView.set(b, bPtr / 8);

        wasmMatMul(aPtr, bPtr, cPtr, m, k, n);

        const result = new Float64Array(m * n);
        result.set(new Float64Array(memory.buffer, cPtr, m * n));

        wasmFree(aPtr);
        wasmFree(bPtr);
        wasmFree(cPtr);

        return result;
      },

      solve(a: Float64Array, b: Float64Array, n: number): Float64Array {
        const wasmMalloc = exports.malloc as (size: number) => number;
        const wasmFree = exports.free as (ptr: number) => void;
        const wasmSolve = exports.solve as (
          aPtr: number,
          bPtr: number,
          xPtr: number,
          n: number,
        ) => number;

        const aBytes = a.byteLength;
        const bBytes = b.byteLength;
        const xBytes = n * 8;

        const aPtr = wasmMalloc(aBytes);
        const bPtr = wasmMalloc(bBytes);
        const xPtr = wasmMalloc(xBytes);

        const memView = new Float64Array(memory.buffer);
        memView.set(a, aPtr / 8);
        memView.set(b, bPtr / 8);

        const info = wasmSolve(aPtr, bPtr, xPtr, n);
        if (info !== 0) {
          wasmFree(aPtr);
          wasmFree(bPtr);
          wasmFree(xPtr);
          throw new Error(`WASM solve failed with info = ${info} (singular matrix)`);
        }

        const result = new Float64Array(n);
        result.set(new Float64Array(memory.buffer, xPtr, n));

        wasmFree(aPtr);
        wasmFree(bPtr);
        wasmFree(xPtr);

        return result;
      },

      cholesky(a: Float64Array, n: number): Float64Array {
        const wasmMalloc = exports.malloc as (size: number) => number;
        const wasmFree = exports.free as (ptr: number) => void;
        const wasmCholesky = exports.cholesky as (
          aPtr: number,
          lPtr: number,
          n: number,
        ) => number;

        const aBytes = a.byteLength;
        const lBytes = n * n * 8;

        const aPtr = wasmMalloc(aBytes);
        const lPtr = wasmMalloc(lBytes);

        const memView = new Float64Array(memory.buffer);
        memView.set(a, aPtr / 8);

        const info = wasmCholesky(aPtr, lPtr, n);
        if (info !== 0) {
          wasmFree(aPtr);
          wasmFree(lPtr);
          throw new Error(
            `WASM cholesky failed with info = ${info} (not positive-definite)`,
          );
        }

        const result = new Float64Array(n * n);
        result.set(new Float64Array(memory.buffer, lPtr, n * n));

        wasmFree(aPtr);
        wasmFree(lPtr);

        return result;
      },
    };

    wasmModule = mod;
    hasWasm = true;
    return mod;
  } catch {
    // WASM not available — callers should use tsFallback
    wasmModule = null;
    hasWasm = false;
    return null;
  }
}
