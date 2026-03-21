/**
 * Shared linear algebra utilities.
 *
 * When the native Fortran/LAPACK addon is available, matrix operations use
 * BLAS/LAPACK for production-grade speed and numerical precision.
 * Otherwise, pure TypeScript fallbacks are used automatically.
 */

import { nativeAddon } from "./native-addon";

// ── Native addon interface ──────────────────────────────────────────────

interface NativeLinalg {
  // LAPACK-backed operations
  matMul(A: number[][], B: number[][], m: number, k: number, n: number): number[][];
  solve(A: number[][], b: number[], n: number): { x: number[]; info: number };
  invert(A: number[][], n: number): { inv: number[][]; info: number };
  symEigen(A: number[][], n: number): {
    eigenvalues: number[];
    eigenvectors: number[][];
    info: number;
  };
  normalCdf(x: number): number;
}

let native: NativeLinalg | null = null;
try {
  if (nativeAddon) {
    // Verify that linalg functions are available (not just special functions).
    // The stub build sets info = -999 to signal no real LAPACK is linked.
    const addon = nativeAddon as unknown as NativeLinalg;
    const probe = addon.solve([[1, 0], [0, 1]], [1, 1], 2);
    if (probe && probe.info === 0) {
      native = addon;
    }
  }
} catch {
  // LAPACK probe failed — pure TypeScript fallback will be used.
}

/** Whether native LAPACK acceleration is active. */
export const hasNativeLinalg: boolean = native !== null;

// ── Types ───────────────────────────────────────────────────────────────

/** A matrix represented as an array of row arrays. */
export type Matrix = number[][];

// ── Pure TypeScript fallback implementations ────────────────────────────

function tsSolveLinearSystem(A: Matrix, b: number[]): number[] {
  const n = A.length;
  const aug: number[][] = A.map((row, i) => [...row, b[i]]);

  for (let col = 0; col < n; col++) {
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(aug[row][col]) > Math.abs(aug[maxRow][col])) {
        maxRow = row;
      }
    }
    [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];

    if (Math.abs(aug[col][col]) < 1e-12) {
      throw new Error("Singular matrix: features may be linearly dependent");
    }

    for (let row = col + 1; row < n; row++) {
      const factor = aug[row][col] / aug[col][col];
      for (let j = col; j <= n; j++) {
        aug[row][j] -= factor * aug[col][j];
      }
    }
  }

  const x = new Array(n).fill(0);
  for (let row = n - 1; row >= 0; row--) {
    x[row] = aug[row][n];
    for (let col = row + 1; col < n; col++) {
      x[row] -= aug[row][col] * x[col];
    }
    x[row] /= aug[row][row];
  }

  return x;
}

function tsInvertMatrix(M: Matrix): Matrix | null {
  const n = M.length;
  const aug: number[][] = M.map((row, i) => {
    const id = new Array(n).fill(0);
    id[i] = 1;
    return [...row, ...id];
  });

  for (let col = 0; col < n; col++) {
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(aug[row][col]) > Math.abs(aug[maxRow][col])) {
        maxRow = row;
      }
    }
    [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];

    const pivot = aug[col][col];
    if (Math.abs(pivot) < 1e-12) return null;

    for (let j = 0; j < 2 * n; j++) {
      aug[col][j] /= pivot;
    }

    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = aug[row][col];
      for (let j = 0; j < 2 * n; j++) {
        aug[row][j] -= factor * aug[col][j];
      }
    }
  }

  return aug.map((row) => row.slice(n));
}

function tsTranspose(A: Matrix): Matrix {
  const m = A.length;
  const n = A[0].length;
  const T: Matrix = Array.from({ length: n }, () => new Array(m));
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      T[j][i] = A[i][j];
    }
  }
  return T;
}

function tsMatMul(A: Matrix, B: Matrix): Matrix {
  const m = A.length;
  const n = B[0].length;
  const k = B.length;
  const C: Matrix = Array.from({ length: m }, () => new Array(n).fill(0));
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let l = 0; l < k; l++) {
        sum += A[i][l] * B[l][j];
      }
      C[i][j] = sum;
    }
  }
  return C;
}

function tsNormalCdf(x: number): number {
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;

  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x) / Math.SQRT2;
  const t = 1.0 / (1.0 + p * ax);
  const y = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-ax * ax);
  return 0.5 * (1.0 + sign * y);
}

/**
 * Eigenvalue decomposition of a symmetric matrix using the Jacobi method.
 * Returns eigenvalues (descending) and corresponding eigenvectors as columns.
 */
function tsSymmetricEigen(
  A: Matrix,
  maxIter = 200,
): { eigenvalues: number[]; eigenvectors: Matrix } {
  const n = A.length;
  const S: Matrix = A.map((row) => [...row]);
  const V: Matrix = Array.from({ length: n }, (_, i) => {
    const row = new Array(n).fill(0);
    row[i] = 1;
    return row;
  });

  for (let iter = 0; iter < maxIter; iter++) {
    let maxVal = 0;
    let p = 0;
    let q = 1;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (Math.abs(S[i][j]) > maxVal) {
          maxVal = Math.abs(S[i][j]);
          p = i;
          q = j;
        }
      }
    }

    if (maxVal < 1e-12) break;

    const theta =
      Math.abs(S[p][p] - S[q][q]) < 1e-15
        ? Math.PI / 4
        : 0.5 * Math.atan2(2 * S[p][q], S[p][p] - S[q][q]);

    const c = Math.cos(theta);
    const s = Math.sin(theta);

    const Spp = c * c * S[p][p] + 2 * s * c * S[p][q] + s * s * S[q][q];
    const Sqq = s * s * S[p][p] - 2 * s * c * S[p][q] + c * c * S[q][q];

    S[p][p] = Spp;
    S[q][q] = Sqq;
    S[p][q] = 0;
    S[q][p] = 0;

    for (let i = 0; i < n; i++) {
      if (i !== p && i !== q) {
        const Sip = c * S[i][p] + s * S[i][q];
        const Siq = -s * S[i][p] + c * S[i][q];
        S[i][p] = Sip;
        S[p][i] = Sip;
        S[i][q] = Siq;
        S[q][i] = Siq;
      }
    }

    for (let i = 0; i < n; i++) {
      const Vip = c * V[i][p] + s * V[i][q];
      const Viq = -s * V[i][p] + c * V[i][q];
      V[i][p] = Vip;
      V[i][q] = Viq;
    }
  }

  const eigenvalues = new Array(n);
  for (let i = 0; i < n; i++) eigenvalues[i] = S[i][i];

  const indices = Array.from({ length: n }, (_, i) => i);
  indices.sort((a, b) => eigenvalues[b] - eigenvalues[a]);

  const sortedValues = indices.map((i) => eigenvalues[i]);
  const sortedVectors: Matrix = Array.from({ length: n }, () => new Array(n));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      sortedVectors[i][j] = V[i][indices[j]];
    }
  }

  return { eigenvalues: sortedValues, eigenvectors: sortedVectors };
}

// ── Exported functions (native with TS fallback) ────────────────────────

/**
 * Solve a linear system Ax = b.
 *
 * Native: LAPACK DGESV (LU factorization with partial pivoting).
 * Fallback: Gaussian elimination with partial pivoting.
 */
export function solveLinearSystem(A: Matrix, b: number[]): number[] {
  if (native) {
    const n = A.length;
    const result = native.solve(A, b, n);
    if (result.info !== 0) {
      throw new Error("Singular matrix: features may be linearly dependent");
    }
    return result.x;
  }
  return tsSolveLinearSystem(A, b);
}

/**
 * Invert a square matrix.
 * Returns null if the matrix is singular.
 *
 * Native: LAPACK DGETRF + DGETRI (LU factorization).
 * Fallback: Gauss-Jordan elimination.
 */
export function invertMatrix(M: Matrix): Matrix | null {
  if (native) {
    const n = M.length;
    const result = native.invert(M, n);
    if (result.info !== 0) return null;
    return result.inv;
  }
  return tsInvertMatrix(M);
}

/**
 * Transpose a matrix.
 * Pure TypeScript (no LAPACK equivalent needed for simple transpose).
 */
export function transpose(A: Matrix): Matrix {
  return tsTranspose(A);
}

/**
 * Multiply two matrices: C = A * B.
 *
 * Native: BLAS DGEMM.
 * Fallback: Triple-nested loop O(m*n*k).
 */
export function matMul(A: Matrix, B: Matrix): Matrix {
  if (native) {
    const m = A.length;
    const k = B.length;
    const n = B[0].length;
    return native.matMul(A, B, m, k, n);
  }
  return tsMatMul(A, B);
}

/**
 * Standard normal CDF: P(Z <= x).
 *
 * Native: Fortran erfc-based (0.5 * erfc(-x / sqrt(2))), full double precision.
 * Fallback: Abramowitz & Stegun 26.2.17 rational approximation.
 */
export function normalCdf(x: number): number {
  if (native) return native.normalCdf(x);
  return tsNormalCdf(x);
}

/**
 * Eigenvalue decomposition of a symmetric matrix.
 * Returns eigenvalues in descending order and eigenvectors as columns.
 *
 * Native: LAPACK DSYEV (divide-and-conquer).
 * Fallback: Jacobi iteration.
 */
export function symmetricEigen(
  A: Matrix,
  maxIter?: number,
): { eigenvalues: number[]; eigenvectors: Matrix } {
  if (native) {
    const n = A.length;
    const result = native.symEigen(A, n);
    if (result.info !== 0) {
      // Fall back to TypeScript Jacobi if DSYEV fails
      return tsSymmetricEigen(A, maxIter);
    }
    // DSYEV returns eigenvalues in ascending order; reverse to descending
    const eigenvalues = result.eigenvalues.slice().reverse();
    // Reverse the column order of eigenvectors to match
    const eigenvectors: Matrix = Array.from({ length: n }, () => new Array(n));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        eigenvectors[i][j] = result.eigenvectors[i][n - 1 - j];
      }
    }
    return { eigenvalues, eigenvectors };
  }
  return tsSymmetricEigen(A, maxIter);
}

/**
 * Standard normal quantile function (Peter Acklam's rational approximation).
 */
export function normalQuantile(p: number): number {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  if (p === 0.5) return 0;

  const a = [
    -3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
    1.383577518672690e2, -3.066479806614716e1, 2.506628277459239e0,
  ];
  const b = [
    -5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
    6.680131188771972e1, -1.328068155288572e1,
  ];
  const c = [
    -7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838e0,
    -2.549732539343734e0, 4.374664141464968e0, 2.938163982698783e0,
  ];
  const d = [
    7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996e0,
    3.754408661907416e0,
  ];

  const pLow = 0.02425;
  const pHigh = 1 - pLow;

  let q: number;
  if (p < pLow) {
    q = Math.sqrt(-2 * Math.log(p));
    return (
      (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    );
  } else if (p <= pHigh) {
    q = p - 0.5;
    const r = q * q;
    return (
      ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) /
      (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    );
  } else {
    q = Math.sqrt(-2 * Math.log(1 - p));
    return -(
      (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    );
  }
}

/**
 * Seeded pseudo-random number generator (xorshift128+).
 */
export function createRng(seed: number): () => number {
  let s0 = seed | 0 || 1;
  let s1 = (seed * 2654435761) | 0 || 2;
  return () => {
    let a = s0;
    const b = s1;
    s0 = b;
    a ^= a << 23;
    a ^= a >> 17;
    a ^= b;
    a ^= b >> 26;
    s1 = a;
    return ((s0 + s1) >>> 0) / 4294967296;
  };
}

/**
 * Random sample of k indices from [0, n) without replacement (Fisher-Yates).
 */
export function randomSample(
  n: number,
  k: number,
  random: () => number,
): number[] {
  const indices = Array.from({ length: n }, (_, i) => i);
  for (let i = 0; i < k; i++) {
    const j = i + Math.floor(random() * (n - i));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  return indices.slice(0, k);
}
