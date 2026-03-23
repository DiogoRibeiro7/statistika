/**
 * Matrix class with decompositions (LU, QR, SVD, Cholesky) and solvers.
 *
 * When the native Fortran/LAPACK addon is available, decompositions use
 * LAPACK routines (DGETRF, DGEQRF, DPOTRF, DGESVD) for production-grade
 * speed and numerical precision. Otherwise, pure TypeScript fallbacks are
 * used automatically.
 *
 * Data is stored in row-major flat Float64Array for cache-friendly access.
 */

import { nativeAddon } from "./native-addon";

// ── Native addon interface for decompositions ────────────────────────────

interface NativeMatDecomps {
  lu(
    A: number[][],
    n: number,
  ): { lu: number[][]; ipiv: number[]; info: number };
  qr(
    A: number[][],
    m: number,
    n: number,
  ): { Q: number[][]; R: number[][]; info: number };
  cholesky(A: number[][], n: number): { L: number[][]; info: number };
  svd(
    A: number[][],
    m: number,
    n: number,
  ): { U: number[][]; S: number[]; Vt: number[][]; info: number };
  // Existing ops used for multiply and solve
  matMul(
    A: number[][],
    B: number[][],
    m: number,
    k: number,
    n: number,
  ): number[][];
  solve(
    A: number[][],
    b: number[],
    n: number,
  ): { x: number[]; info: number };
}

let native: NativeMatDecomps | null = null;
try {
  if (nativeAddon) {
    const addon = nativeAddon as unknown as NativeMatDecomps;
    // Probe: check that decomposition functions exist and LAPACK is linked.
    // The stub build sets info = -999 to signal no real LAPACK.
    if (typeof addon.lu === "function") {
      const probe = addon.lu([[1, 0], [0, 1]], 2);
      if (probe && probe.info === 0) {
        native = addon;
      }
    }
  }
} catch {
  // Native decompositions not available — pure TypeScript fallbacks will be used.
}

/**
 * Whether native LAPACK acceleration is active for Mat decompositions.
 * When true, LU, QR, Cholesky, SVD, and matrix multiplication use
 * LAPACK/BLAS routines for improved speed and numerical precision.
 */
export const hasNativeMatDecomps: boolean = native !== null;

// ── Helper: convert 2D array to Mat ──────────────────────────────────────

function fromArray(data: number[][], rows: number, cols: number): Mat {
  const flat = new Float64Array(rows * cols);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      flat[i * cols + j] = data[i][j];
    }
  }
  return Mat._create(rows, cols, flat);
}

// ── Matrix class ─────────────────────────────────────────────────────────

/**
 * Dense matrix class with decompositions and solvers.
 *
 * Data is stored in row-major flat Float64Array for cache-friendly access.
 * Supports LU, QR, Cholesky, and SVD decompositions, as well as linear
 * system solving, least squares, determinant, inverse, pseudoinverse,
 * rank, and condition number.
 *
 * When the native Fortran/LAPACK addon is available, decompositions use
 * LAPACK routines for production-grade speed and numerical precision.
 * Otherwise, pure TypeScript fallbacks are used automatically.
 *
 * @example
 * ```ts
 * const A = Mat.from([[1, 2], [3, 4]]);
 * const b = [5, 6];
 * const x = A.solve(b); // solves Ax = b
 * const { U, S, V } = A.svd(); // singular value decomposition
 * ```
 */
export class Mat {
  /** Row-major flat storage. */
  readonly data: Float64Array;
  /** Number of rows. */
  readonly rows: number;
  /** Number of columns. */
  readonly cols: number;

  /**
   * @param rows - Number of rows (must be positive)
   * @param cols - Number of columns (must be positive)
   * @param data - Optional pre-allocated Float64Array of length rows * cols
   * @throws Error if dimensions are not positive
   */
  private constructor(rows: number, cols: number, data?: Float64Array) {
    if (rows <= 0 || cols <= 0) {
      throw new Error("Matrix dimensions must be positive");
    }
    this.rows = rows;
    this.cols = cols;
    this.data = data ?? new Float64Array(rows * cols);
  }

  /** @internal Used by native helpers to construct without validation overhead. */
  static _create(rows: number, cols: number, data: Float64Array): Mat {
    return new Mat(rows, cols, data);
  }

  // ── Factories ────────────────────────────────────────────────────────

  /** Create a matrix from a 2D array of numbers. */
  static from(data: number[][]): Mat {
    const rows = data.length;
    if (rows === 0) throw new Error("Matrix must have at least one row");
    const cols = data[0].length;
    if (cols === 0) throw new Error("Matrix must have at least one column");
    const flat = new Float64Array(rows * cols);
    for (let i = 0; i < rows; i++) {
      if (data[i].length !== cols) {
        throw new Error("All rows must have the same length");
      }
      for (let j = 0; j < cols; j++) {
        flat[i * cols + j] = data[i][j];
      }
    }
    return new Mat(rows, cols, flat);
  }

  /** Create a matrix of zeros. */
  static zeros(rows: number, cols: number): Mat {
    return new Mat(rows, cols);
  }

  /** Create a matrix of ones. */
  static ones(rows: number, cols: number): Mat {
    const flat = new Float64Array(rows * cols).fill(1);
    return new Mat(rows, cols, flat);
  }

  /** Create an identity matrix. */
  static identity(n: number): Mat {
    const m = new Mat(n, n);
    for (let i = 0; i < n; i++) m.data[i * n + i] = 1;
    return m;
  }

  /** Create a diagonal matrix from a vector. */
  static diag(values: number[]): Mat {
    const n = values.length;
    const m = new Mat(n, n);
    for (let i = 0; i < n; i++) m.data[i * n + i] = values[i];
    return m;
  }

  /** Create a column vector from an array. */
  static fromVector(v: number[]): Mat {
    return Mat.from(v.map((x) => [x]));
  }

  // ── Element access ───────────────────────────────────────────────────

  /** Get element at (i, j). */
  get(i: number, j: number): number {
    return this.data[i * this.cols + j];
  }

  /** Set element at (i, j). */
  set(i: number, j: number, value: number): void {
    this.data[i * this.cols + j] = value;
  }

  /** Get a row as a number array. */
  row(i: number): number[] {
    const start = i * this.cols;
    return Array.from(this.data.slice(start, start + this.cols));
  }

  /** Get a column as a number array. */
  col(j: number): number[] {
    const result = new Array(this.rows);
    for (let i = 0; i < this.rows; i++) {
      result[i] = this.data[i * this.cols + j];
    }
    return result;
  }

  /** Get the diagonal as a number array. */
  diagonal(): number[] {
    const n = Math.min(this.rows, this.cols);
    const result = new Array(n);
    for (let i = 0; i < n; i++) {
      result[i] = this.data[i * this.cols + i];
    }
    return result;
  }

  /** Convert to a 2D array. */
  toArray(): number[][] {
    const result: number[][] = new Array(this.rows);
    for (let i = 0; i < this.rows; i++) {
      result[i] = this.row(i);
    }
    return result;
  }

  /** Convert a column vector to a flat array. */
  toVector(): number[] {
    if (this.cols !== 1) {
      throw new Error("toVector() requires a column vector (cols === 1)");
    }
    return this.col(0);
  }

  /** Deep clone. */
  clone(): Mat {
    return new Mat(this.rows, this.cols, new Float64Array(this.data));
  }

  /** Check if dimensions match another matrix. */
  sameSize(other: Mat): boolean {
    return this.rows === other.rows && this.cols === other.cols;
  }

  /** Check if the matrix is square. */
  isSquare(): boolean {
    return this.rows === this.cols;
  }

  /** Element-wise approximate equality. */
  equals(other: Mat, tol = 1e-10): boolean {
    if (!this.sameSize(other)) return false;
    for (let i = 0; i < this.data.length; i++) {
      if (Math.abs(this.data[i] - other.data[i]) > tol) return false;
    }
    return true;
  }

  // ── Arithmetic ───────────────────────────────────────────────────────

  /** Add two matrices. */
  add(other: Mat): Mat {
    if (!this.sameSize(other)) {
      throw new Error(
        `Dimension mismatch: (${this.rows},${this.cols}) vs (${other.rows},${other.cols})`,
      );
    }
    const result = new Float64Array(this.data.length);
    for (let i = 0; i < result.length; i++) {
      result[i] = this.data[i] + other.data[i];
    }
    return new Mat(this.rows, this.cols, result);
  }

  /** Subtract another matrix. */
  subtract(other: Mat): Mat {
    if (!this.sameSize(other)) {
      throw new Error(
        `Dimension mismatch: (${this.rows},${this.cols}) vs (${other.rows},${other.cols})`,
      );
    }
    const result = new Float64Array(this.data.length);
    for (let i = 0; i < result.length; i++) {
      result[i] = this.data[i] - other.data[i];
    }
    return new Mat(this.rows, this.cols, result);
  }

  /** Scalar multiplication. */
  scale(s: number): Mat {
    const result = new Float64Array(this.data.length);
    for (let i = 0; i < result.length; i++) {
      result[i] = this.data[i] * s;
    }
    return new Mat(this.rows, this.cols, result);
  }

  /** Negate all elements. */
  negate(): Mat {
    return this.scale(-1);
  }

  /**
   * Matrix multiplication: this * other.
   *
   * Native: BLAS DGEMM.
   * Fallback: Triple-nested loop O(m*n*k).
   */
  multiply(other: Mat): Mat {
    if (this.cols !== other.rows) {
      throw new Error(
        `Cannot multiply (${this.rows},${this.cols}) by (${other.rows},${other.cols})`,
      );
    }
    if (native) {
      const result = native.matMul(
        this.toArray(),
        other.toArray(),
        this.rows,
        this.cols,
        other.cols,
      );
      return fromArray(result, this.rows, other.cols);
    }
    return tsMatMul(this, other);
  }

  /** Transpose. */
  transpose(): Mat {
    const result = new Float64Array(this.rows * this.cols);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        result[j * this.rows + i] = this.data[i * this.cols + j];
      }
    }
    return new Mat(this.cols, this.rows, result);
  }

  /** Trace (sum of diagonal). */
  trace(): number {
    if (!this.isSquare()) throw new Error("Trace requires a square matrix");
    let sum = 0;
    for (let i = 0; i < this.rows; i++) {
      sum += this.data[i * this.cols + i];
    }
    return sum;
  }

  /** Frobenius norm. */
  normF(): number {
    let sum = 0;
    for (let i = 0; i < this.data.length; i++) {
      sum += this.data[i] * this.data[i];
    }
    return Math.sqrt(sum);
  }

  /** Extract a submatrix. */
  submatrix(
    rowStart: number,
    rowEnd: number,
    colStart: number,
    colEnd: number,
  ): Mat {
    const rows = rowEnd - rowStart;
    const cols = colEnd - colStart;
    const result = new Mat(rows, cols);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        result.data[i * cols + j] =
          this.data[(rowStart + i) * this.cols + (colStart + j)];
      }
    }
    return result;
  }

  // ── Decompositions ───────────────────────────────────────────────────

  /**
   * LU decomposition with partial pivoting: PA = LU.
   * Returns { L, U, P, pivots, sign } where P is the permutation matrix.
   *
   * Native: LAPACK DGETRF.
   * Fallback: Gaussian elimination with partial pivoting.
   */
  lu(): LUResult {
    if (!this.isSquare()) throw new Error("LU requires a square matrix");
    if (native) {
      return nativeLU(this);
    }
    return tsLU(this);
  }

  /**
   * QR decomposition: A = QR.
   * Works for any m×n matrix with m >= n.
   *
   * Native: LAPACK DGEQRF + DORGQR.
   * Fallback: Householder reflections.
   */
  qr(): QRResult {
    if (this.rows < this.cols) throw new Error("QR requires rows >= cols");
    if (native) {
      return nativeQR(this);
    }
    return tsQR(this);
  }

  /**
   * Cholesky decomposition for symmetric positive-definite matrices: A = L * L^T.
   * Returns the lower-triangular factor L.
   *
   * Native: LAPACK DPOTRF.
   * Fallback: Standard Cholesky algorithm.
   */
  cholesky(): Mat {
    if (!this.isSquare()) {
      throw new Error("Cholesky requires a square matrix");
    }
    if (native) {
      return nativeCholesky(this);
    }
    return tsCholesky(this);
  }

  /**
   * Singular Value Decomposition: A = U * diag(S) * V^T.
   * Works for any m×n matrix.
   *
   * Native: LAPACK DGESVD.
   * Fallback: One-sided Jacobi SVD.
   */
  svd(): SVDResult {
    if (native) {
      return nativeSVD(this);
    }
    return tsSVD(this);
  }

  // ── Solvers ──────────────────────────────────────────────────────────

  /**
   * Solve Ax = b for a square matrix A.
   *
   * Native: LAPACK DGESV (LU factorization with partial pivoting).
   * Fallback: LU decomposition + forward/back substitution.
   *
   * @param b Right-hand side vector (as number[] or column Mat).
   * @returns Solution vector x.
   */
  solve(b: number[] | Mat): number[] {
    if (!this.isSquare()) throw new Error("solve requires a square matrix");
    const n = this.rows;
    const bVec = b instanceof Mat ? b.toVector() : b;
    if (bVec.length !== n) {
      throw new Error(
        `RHS length ${bVec.length} does not match matrix size ${n}`,
      );
    }

    if (native) {
      const result = native.solve(this.toArray(), bVec, n);
      if (result.info !== 0) {
        throw new Error("Singular matrix: features may be linearly dependent");
      }
      return result.x;
    }

    const { L, U, pivots } = this.lu();

    // Apply permutation to b
    const pb = new Array(n);
    for (let i = 0; i < n; i++) pb[i] = bVec[pivots[i]];

    // Forward substitution: Ly = Pb
    const y = new Array(n);
    for (let i = 0; i < n; i++) {
      let sum = 0;
      for (let j = 0; j < i; j++) {
        sum += L.data[i * n + j] * y[j];
      }
      y[i] = pb[i] - sum;
    }

    // Back substitution: Ux = y
    const x = new Array(n);
    for (let i = n - 1; i >= 0; i--) {
      let sum = 0;
      for (let j = i + 1; j < n; j++) {
        sum += U.data[i * n + j] * x[j];
      }
      x[i] = (y[i] - sum) / U.data[i * n + i];
    }

    return x;
  }

  /**
   * Solve Ax = b in the least-squares sense using QR decomposition.
   * Works for overdetermined systems (m > n).
   */
  leastSquares(b: number[] | Mat): number[] {
    const bVec = b instanceof Mat ? b.toVector() : b;
    if (bVec.length !== this.rows) {
      throw new Error(
        `RHS length ${bVec.length} does not match matrix rows ${this.rows}`,
      );
    }

    const { QFull, RFull } = this.qr();
    const n = this.cols;

    // Q^T * b
    const Qtb = new Array(n);
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let i = 0; i < this.rows; i++) {
        sum += QFull.data[i * QFull.cols + j] * bVec[i];
      }
      Qtb[j] = sum;
    }

    // Back substitution with R (upper n×n block)
    const x = new Array(n);
    for (let i = n - 1; i >= 0; i--) {
      let sum = 0;
      for (let j = i + 1; j < n; j++) {
        sum += RFull.data[i * RFull.cols + j] * x[j];
      }
      const diag = RFull.data[i * RFull.cols + i];
      if (Math.abs(diag) < 1e-14) {
        throw new Error("Matrix is rank-deficient; cannot solve via QR");
      }
      x[i] = (Qtb[i] - sum) / diag;
    }

    return x;
  }

  /**
   * Compute the determinant using LU decomposition.
   */
  det(): number {
    if (!this.isSquare())
      throw new Error("Determinant requires a square matrix");
    const { U, sign } = this.lu();
    let det = sign;
    for (let i = 0; i < this.rows; i++) {
      det *= U.data[i * this.rows + i];
    }
    return det;
  }

  /**
   * Compute the inverse using LU decomposition.
   * Returns null if the matrix is singular.
   */
  inverse(): Mat | null {
    if (!this.isSquare()) throw new Error("Inverse requires a square matrix");
    const n = this.rows;
    try {
      const inv = Mat.zeros(n, n);
      for (let j = 0; j < n; j++) {
        const e = new Array(n).fill(0);
        e[j] = 1;
        const col = this.solve(e);
        for (let i = 0; i < n; i++) {
          inv.data[i * n + j] = col[i];
        }
      }
      return inv;
    } catch {
      return null;
    }
  }

  /**
   * Compute the rank of the matrix using SVD.
   * @param tol Threshold below which singular values are considered zero.
   */
  rank(tol?: number): number {
    const { S } = this.svd();
    const threshold = tol ?? S[0] * Math.max(this.rows, this.cols) * 2.2e-16;
    let r = 0;
    for (const s of S) {
      if (s > threshold) r++;
    }
    return r;
  }

  /**
   * Compute the condition number (ratio of largest to smallest singular value).
   */
  cond(): number {
    const { S } = this.svd();
    const sMax = S[0];
    const sMin = S[S.length - 1];
    if (sMin < 1e-14) return Infinity;
    return sMax / sMin;
  }

  /**
   * Compute the Moore-Penrose pseudoinverse using SVD.
   */
  pinv(tol?: number): Mat {
    const { U, S, V } = this.svd();
    const threshold = tol ?? S[0] * Math.max(this.rows, this.cols) * 2.2e-16;
    const k = S.length;

    // Construct S^+
    const Sinv = new Array(k);
    for (let i = 0; i < k; i++) {
      Sinv[i] = S[i] > threshold ? 1 / S[i] : 0;
    }

    // pinv(A) = V * diag(S^+) * U^T
    const result = Mat.zeros(this.cols, this.rows);
    for (let i = 0; i < this.cols; i++) {
      for (let j = 0; j < this.rows; j++) {
        let sum = 0;
        for (let l = 0; l < k; l++) {
          sum += V.data[i * k + l] * Sinv[l] * U.data[j * k + l];
        }
        result.data[i * this.rows + j] = sum;
      }
    }

    return result;
  }
}

// ── Result types ─────────────────────────────────────────────────────────

export interface LUResult {
  /** Lower triangular matrix with unit diagonal. */
  L: Mat;
  /** Upper triangular matrix. */
  U: Mat;
  /** Permutation matrix. */
  P: Mat;
  /** Pivot indices. */
  pivots: number[];
  /** Sign of the permutation (+1 or -1). */
  sign: number;
}

export interface QRResult {
  /** Orthogonal factor Q (thin: m×n). */
  Q: Mat;
  /** Upper triangular R (thin: n×n). */
  R: Mat;
  /** Full Q (m×m). */
  QFull: Mat;
  /** Full R (m×n). */
  RFull: Mat;
}

export interface SVDResult {
  /** Left singular vectors (m×k). */
  U: Mat;
  /** Singular values in descending order. */
  S: number[];
  /** Right singular vectors (n×k). */
  V: Mat;
}

// ── Native LAPACK implementations ────────────────────────────────────────

function nativeLU(A: Mat): LUResult {
  const n = A.rows;
  const result = native!.lu(A.toArray(), n);
  if (result.info !== 0) {
    throw new Error("Matrix is singular or nearly singular");
  }

  const luData = result.lu;
  const ipiv = result.ipiv; // 1-based LAPACK pivot indices

  // Separate L and U from the packed LU output
  const L = Mat.identity(n);
  const U = Mat.zeros(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i > j) {
        L.data[i * n + j] = luData[i][j];
      } else {
        U.data[i * n + j] = luData[i][j];
      }
    }
  }

  // Convert LAPACK 1-based ipiv (swap sequence) to a permutation vector.
  // LAPACK ipiv[i] means: row i was swapped with row ipiv[i]-1 (0-based).
  const perm = new Array(n);
  for (let i = 0; i < n; i++) perm[i] = i;
  let sign = 1;
  for (let i = 0; i < n; i++) {
    const target = ipiv[i] - 1; // Convert 1-based to 0-based
    if (target !== i) {
      const tmp = perm[i];
      perm[i] = perm[target];
      perm[target] = tmp;
      sign = -sign;
    }
  }

  // Build permutation matrix
  const P = Mat.zeros(n, n);
  for (let i = 0; i < n; i++) {
    P.data[i * n + perm[i]] = 1;
  }

  return { L, U, P, pivots: perm, sign };
}

function nativeQR(A: Mat): QRResult {
  const m = A.rows;
  const n = A.cols;
  const result = native!.qr(A.toArray(), m, n);
  if (result.info !== 0) {
    // Fall back to TypeScript
    return tsQR(A);
  }

  const Q = fromArray(result.Q, m, n);
  const R = fromArray(result.R, n, n);

  // For compatibility, also provide "full" versions.
  // QFull: use the thin Q (m×n) since LAPACK gives us the thin factorization.
  // RFull: embed R into m×n (with zeros below).
  const RFull = Mat.zeros(m, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      RFull.data[i * n + j] = R.data[i * n + j];
    }
  }

  // For QFull, we'd need DORGQR with m columns. Since leastSquares uses
  // QFull, provide the thin Q and mark that RFull is m×n.
  // The thin Q (m×n) is sufficient for Q^T*b in least squares.
  return { Q, R, QFull: Q, RFull };
}

function nativeCholesky(A: Mat): Mat {
  const n = A.rows;
  const result = native!.cholesky(A.toArray(), n);
  if (result.info > 0) {
    throw new Error(
      "Matrix is not positive definite (non-positive diagonal encountered)",
    );
  }
  if (result.info !== 0) {
    // Fall back to TypeScript on unexpected error
    return tsCholesky(A);
  }
  return fromArray(result.L, n, n);
}

function nativeSVD(A: Mat): SVDResult {
  const m = A.rows;
  const n = A.cols;
  const result = native!.svd(A.toArray(), m, n);
  if (result.info !== 0) {
    // Fall back to TypeScript
    return tsSVD(A);
  }

  const k = Math.min(m, n);
  const U = fromArray(result.U, m, k);
  const S = result.S;

  // LAPACK returns V^T (k×n); we want V (n×k) = (V^T)^T
  const Vt = fromArray(result.Vt, k, n);
  const V = Vt.transpose();

  return { U, S, V };
}

// ── Pure TypeScript fallback implementations ─────────────────────────────

function tsMatMul(A: Mat, B: Mat): Mat {
  const m = A.rows;
  const n = B.cols;
  const k = A.cols;
  const result = new Float64Array(m * n);
  for (let i = 0; i < m; i++) {
    for (let l = 0; l < k; l++) {
      const a = A.data[i * k + l];
      if (a === 0) continue;
      for (let j = 0; j < n; j++) {
        result[i * n + j] += a * B.data[l * n + j];
      }
    }
  }
  return Mat._create(m, n, result);
}

function tsLU(A: Mat): LUResult {
  const n = A.rows;
  const U = A.clone();
  const L = Mat.identity(n);
  const pivots = new Array(n);
  for (let i = 0; i < n; i++) pivots[i] = i;
  let sign = 1;

  for (let col = 0; col < n; col++) {
    // Find pivot
    let maxVal = Math.abs(U.data[col * n + col]);
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      const val = Math.abs(U.data[row * n + col]);
      if (val > maxVal) {
        maxVal = val;
        maxRow = row;
      }
    }

    if (maxVal < 1e-14) {
      throw new Error("Matrix is singular or nearly singular");
    }

    // Swap rows in U
    if (maxRow !== col) {
      for (let j = 0; j < n; j++) {
        const tmp = U.data[col * n + j];
        U.data[col * n + j] = U.data[maxRow * n + j];
        U.data[maxRow * n + j] = tmp;
      }
      // Swap rows in L (only the part below diagonal that's already filled)
      for (let j = 0; j < col; j++) {
        const tmp = L.data[col * n + j];
        L.data[col * n + j] = L.data[maxRow * n + j];
        L.data[maxRow * n + j] = tmp;
      }
      const tmp = pivots[col];
      pivots[col] = pivots[maxRow];
      pivots[maxRow] = tmp;
      sign = -sign;
    }

    // Eliminate below
    for (let row = col + 1; row < n; row++) {
      const factor = U.data[row * n + col] / U.data[col * n + col];
      L.data[row * n + col] = factor;
      for (let j = col; j < n; j++) {
        U.data[row * n + j] -= factor * U.data[col * n + j];
      }
    }
  }

  // Build permutation matrix
  const P = Mat.zeros(n, n);
  for (let i = 0; i < n; i++) {
    P.data[i * n + pivots[i]] = 1;
  }

  return { L, U, P, pivots, sign };
}

function tsQR(A: Mat): QRResult {
  const m = A.rows;
  const n = A.cols;

  const R = A.clone();
  const Q = Mat.identity(m);

  for (let col = 0; col < Math.min(m - 1, n); col++) {
    // Extract the column vector below the diagonal
    const xLen = m - col;
    const x = new Float64Array(xLen);
    for (let i = 0; i < xLen; i++) {
      x[i] = R.data[(col + i) * n + col];
    }

    // Compute Householder vector
    let xNorm = 0;
    for (let i = 0; i < xLen; i++) xNorm += x[i] * x[i];
    xNorm = Math.sqrt(xNorm);

    if (xNorm < 1e-14) continue;

    const sgn = x[0] >= 0 ? 1 : -1;
    const alpha = -sgn * xNorm;
    x[0] -= alpha;

    // Normalize v
    let vNorm = 0;
    for (let i = 0; i < xLen; i++) vNorm += x[i] * x[i];
    vNorm = Math.sqrt(vNorm);
    if (vNorm < 1e-14) continue;
    for (let i = 0; i < xLen; i++) x[i] /= vNorm;

    // Apply H = I - 2vv^T to R (from left)
    for (let j = col; j < n; j++) {
      let dot = 0;
      for (let i = 0; i < xLen; i++) {
        dot += x[i] * R.data[(col + i) * n + j];
      }
      for (let i = 0; i < xLen; i++) {
        R.data[(col + i) * n + j] -= 2 * x[i] * dot;
      }
    }

    // Apply H to Q (from right)
    for (let i = 0; i < m; i++) {
      let dot = 0;
      for (let k = 0; k < xLen; k++) {
        dot += Q.data[i * m + (col + k)] * x[k];
      }
      for (let k = 0; k < xLen; k++) {
        Q.data[i * m + (col + k)] -= 2 * dot * x[k];
      }
    }
  }

  // Extract the "thin" Q and R: Q is m×n, R is n×n
  const Qthin = Q.submatrix(0, m, 0, n);
  const Rthin = R.submatrix(0, n, 0, n);

  return { Q: Qthin, R: Rthin, QFull: Q, RFull: R };
}

function tsCholesky(A: Mat): Mat {
  const n = A.rows;
  const L = Mat.zeros(n, n);

  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = 0;
      for (let k = 0; k < j; k++) {
        sum += L.data[i * n + k] * L.data[j * n + k];
      }
      if (i === j) {
        const diag = A.data[i * n + i] - sum;
        if (diag <= 0) {
          throw new Error(
            "Matrix is not positive definite (non-positive diagonal encountered)",
          );
        }
        L.data[i * n + j] = Math.sqrt(diag);
      } else {
        L.data[i * n + j] = (A.data[i * n + j] - sum) / L.data[j * n + j];
      }
    }
  }

  return L;
}

function tsSVD(A: Mat): SVDResult {
  const m = A.rows;
  const n = A.cols;
  const wide = m < n;

  // For wide matrices, compute SVD of A^T then swap U and V
  const W = wide ? A.transpose() : A.clone();
  const rows = W.rows;
  const cols = W.cols;

  // V accumulates right rotations
  const V = Mat.identity(cols);

  // One-sided Jacobi: repeatedly apply rotations to columns of W
  const maxIter = 100 * cols * cols;
  for (let iter = 0; iter < maxIter; iter++) {
    let converged = true;

    for (let p = 0; p < cols - 1; p++) {
      for (let q = p + 1; q < cols; q++) {
        let a = 0,
          b = 0,
          d = 0;
        for (let i = 0; i < rows; i++) {
          const ap = W.data[i * cols + p];
          const aq = W.data[i * cols + q];
          a += ap * ap;
          b += aq * aq;
          d += ap * aq;
        }

        if (Math.abs(d) < 1e-14 * Math.sqrt(a * b + 1e-300)) continue;
        converged = false;

        const tau = (b - a) / (2 * d);
        const t =
          Math.sign(tau) / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
        const c = 1 / Math.sqrt(1 + t * t);
        const s = t * c;

        // Apply rotation to columns p, q of W
        for (let i = 0; i < rows; i++) {
          const ap = W.data[i * cols + p];
          const aq = W.data[i * cols + q];
          W.data[i * cols + p] = c * ap - s * aq;
          W.data[i * cols + q] = s * ap + c * aq;
        }

        // Accumulate in V
        for (let i = 0; i < cols; i++) {
          const vp = V.data[i * cols + p];
          const vq = V.data[i * cols + q];
          V.data[i * cols + p] = c * vp - s * vq;
          V.data[i * cols + q] = s * vp + c * vq;
        }
      }
    }

    if (converged) break;
  }

  // Extract singular values and build U
  const singularValues = new Array(cols);
  const U = Mat.zeros(rows, cols);

  for (let j = 0; j < cols; j++) {
    let norm = 0;
    for (let i = 0; i < rows; i++) {
      norm += W.data[i * cols + j] * W.data[i * cols + j];
    }
    norm = Math.sqrt(norm);
    singularValues[j] = norm;
    if (norm > 1e-14) {
      for (let i = 0; i < rows; i++) {
        U.data[i * cols + j] = W.data[i * cols + j] / norm;
      }
    }
  }

  // Sort by descending singular value
  const indices = Array.from({ length: cols }, (_, i) => i);
  indices.sort((a, b) => singularValues[b] - singularValues[a]);

  const sortedS = indices.map((i) => singularValues[i]);
  const sortedU = Mat.zeros(rows, cols);
  const sortedV = Mat.zeros(cols, cols);
  for (let j = 0; j < cols; j++) {
    const srcJ = indices[j];
    for (let i = 0; i < rows; i++) {
      sortedU.data[i * cols + j] = U.data[i * cols + srcJ];
    }
    for (let i = 0; i < cols; i++) {
      sortedV.data[i * cols + j] = V.data[i * cols + srcJ];
    }
  }

  if (wide) {
    return { U: sortedV, S: sortedS, V: sortedU };
  }

  return { U: sortedU, S: sortedS, V: sortedV };
}
