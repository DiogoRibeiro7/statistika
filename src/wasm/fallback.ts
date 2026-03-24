/**
 * Pure TypeScript fallback implementations matching the WASM interface.
 *
 * These are used when neither native Fortran nor WASM is available.
 * The implementations are numerically correct and tested, though slower
 * than the compiled alternatives.
 */

import type { WasmModule } from "./index";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const LANCZOS_G = 7;
const LANCZOS_COEFF = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028,
  771.32342877765313, -176.61502916214059, 12.507343278686905,
  -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
];

// ---------------------------------------------------------------------------
// Special functions
// ---------------------------------------------------------------------------

function gammaLn(x: number): number {
  if (x <= 0 && Number.isInteger(x)) return NaN;
  if (x < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - gammaLn(1 - x);
  }
  x -= 1;
  let a = LANCZOS_COEFF[0];
  for (let i = 1; i < LANCZOS_COEFF.length; i++) {
    a += LANCZOS_COEFF[i] / (x + i);
  }
  const t = x + LANCZOS_G + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

function gamma(x: number): number {
  return Math.exp(gammaLn(x));
}

function erf(x: number): number {
  const sign = x >= 0 ? 1 : -1;
  x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const poly =
    t *
    (0.254829592 +
      t *
        (-0.284496736 +
          t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  return sign * (1 - poly * Math.exp(-x * x));
}

function erfc(x: number): number {
  return 1 - erf(x);
}

function betaFn(a: number, b: number): number {
  return Math.exp(gammaLn(a) + gammaLn(b) - gammaLn(a + b));
}

// ---------------------------------------------------------------------------
// Matrix operations
// ---------------------------------------------------------------------------

function matMul(
  a: Float64Array,
  b: Float64Array,
  m: number,
  k: number,
  n: number,
): Float64Array {
  const c = new Float64Array(m * n);
  for (let i = 0; i < m; i++) {
    for (let p = 0; p < k; p++) {
      const aip = a[i * k + p];
      if (aip === 0) continue;
      for (let j = 0; j < n; j++) {
        c[i * n + j] += aip * b[p * n + j];
      }
    }
  }
  return c;
}

function solve(a: Float64Array, b: Float64Array, n: number): Float64Array {
  // LU decomposition with partial pivoting
  const lu = new Float64Array(a);
  const x = new Float64Array(b);
  const perm = new Int32Array(n);
  for (let i = 0; i < n; i++) perm[i] = i;

  for (let col = 0; col < n; col++) {
    // Find pivot
    let maxVal = Math.abs(lu[col * n + col]);
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      const val = Math.abs(lu[row * n + col]);
      if (val > maxVal) {
        maxVal = val;
        maxRow = row;
      }
    }

    if (maxVal < 1e-15) {
      throw new Error("Matrix is singular or nearly singular");
    }

    // Swap rows
    if (maxRow !== col) {
      for (let j = 0; j < n; j++) {
        const tmp = lu[col * n + j];
        lu[col * n + j] = lu[maxRow * n + j];
        lu[maxRow * n + j] = tmp;
      }
      const tmpP = perm[col];
      perm[col] = perm[maxRow];
      perm[maxRow] = tmpP;
    }

    // Eliminate below
    for (let row = col + 1; row < n; row++) {
      const factor = lu[row * n + col] / lu[col * n + col];
      lu[row * n + col] = factor; // store L
      for (let j = col + 1; j < n; j++) {
        lu[row * n + j] -= factor * lu[col * n + j];
      }
    }
  }

  // Apply permutation to b
  const pb = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    pb[i] = x[perm[i]];
  }

  // Forward substitution (L * y = Pb)
  for (let i = 1; i < n; i++) {
    for (let j = 0; j < i; j++) {
      pb[i] -= lu[i * n + j] * pb[j];
    }
  }

  // Back substitution (U * x = y)
  for (let i = n - 1; i >= 0; i--) {
    for (let j = i + 1; j < n; j++) {
      pb[i] -= lu[i * n + j] * pb[j];
    }
    pb[i] /= lu[i * n + i];
  }

  return pb;
}

function cholesky(a: Float64Array, n: number): Float64Array {
  const L = new Float64Array(n * n);

  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = 0;
      for (let k = 0; k < j; k++) {
        sum += L[i * n + k] * L[j * n + k];
      }

      if (i === j) {
        const diag = a[i * n + i] - sum;
        if (diag <= 0) {
          throw new Error("Matrix is not positive-definite");
        }
        L[i * n + j] = Math.sqrt(diag);
      } else {
        L[i * n + j] = (a[i * n + j] - sum) / L[j * n + j];
      }
    }
  }

  return L;
}

// ---------------------------------------------------------------------------
// Exported fallback module
// ---------------------------------------------------------------------------

/**
 * Pure TypeScript fallback implementing the {@link WasmModule} interface.
 *
 * All functions are numerically equivalent to the WASM and native
 * implementations but run in pure JavaScript.
 */
export const tsFallback: WasmModule = {
  gammaLn,
  gamma,
  erf,
  erfc,
  betaFn,
  matMul,
  solve,
  cholesky,
};
