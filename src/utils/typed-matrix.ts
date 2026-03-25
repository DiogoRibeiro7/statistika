/**
 * Typed array matrix utilities for cache-friendly numerical computation.
 *
 * Provides efficient Float64Array-based matrix operations as free functions,
 * bridging the gap between the Mat class and code that still uses number[][].
 */

/** Create a Float64Array matrix from a 2D array. */
export function fromArray2D(data: number[][]): { data: Float64Array; rows: number; cols: number } {
  const rows = data.length;
  if (rows === 0) return { data: new Float64Array(0), rows: 0, cols: 0 };
  const cols = data[0].length;
  const flat = new Float64Array(rows * cols);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      flat[i * cols + j] = data[i][j];
    }
  }
  return { data: flat, rows, cols };
}

/** Convert a Float64Array (row-major) back to a 2D number[][] array. */
export function toArray2D(data: Float64Array, rows: number, cols: number): number[][] {
  const result: number[][] = [];
  for (let i = 0; i < rows; i++) {
    const row: number[] = [];
    for (let j = 0; j < cols; j++) {
      row.push(data[i * cols + j]);
    }
    result.push(row);
  }
  return result;
}

/**
 * Matrix multiply using Float64Array (row-major).
 * Multiplies (m x k) * (k x n) -> (m x n).
 */
export function matMulTyped(a: Float64Array, b: Float64Array, m: number, k: number, n: number): Float64Array {
  const result = new Float64Array(m * n);
  for (let i = 0; i < m; i++) {
    for (let p = 0; p < k; p++) {
      const aVal = a[i * k + p];
      for (let j = 0; j < n; j++) {
        result[i * n + j] += aVal * b[p * n + j];
      }
    }
  }
  return result;
}

/** Transpose a (rows x cols) matrix stored as Float64Array row-major. */
export function transposeTyped(a: Float64Array, rows: number, cols: number): Float64Array {
  const result = new Float64Array(rows * cols);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      result[j * rows + i] = a[i * cols + j];
    }
  }
  return result;
}

/** Dot product of two Float64Arrays of equal length. */
export function dotTyped(a: Float64Array, b: Float64Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

/** Matrix-vector multiply: (rows x cols) matrix times (cols) vector -> (rows) vector. */
export function matVecMulTyped(a: Float64Array, x: Float64Array, rows: number, cols: number): Float64Array {
  const result = new Float64Array(rows);
  for (let i = 0; i < rows; i++) {
    let sum = 0;
    for (let j = 0; j < cols; j++) {
      sum += a[i * cols + j] * x[j];
    }
    result[i] = sum;
  }
  return result;
}

/** Element-wise addition of two Float64Arrays. */
export function addTyped(a: Float64Array, b: Float64Array): Float64Array {
  const result = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) {
    result[i] = a[i] + b[i];
  }
  return result;
}

/** Element-wise scalar multiplication. */
export function scaleTyped(a: Float64Array, scalar: number): Float64Array {
  const result = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) {
    result[i] = a[i] * scalar;
  }
  return result;
}

/** Create an n x n identity matrix as a flat Float64Array. */
export function eyeTyped(n: number): Float64Array {
  const result = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    result[i * n + i] = 1;
  }
  return result;
}

/** Frobenius norm of a matrix stored as Float64Array. */
export function frobeniusNormTyped(a: Float64Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * a[i];
  }
  return Math.sqrt(sum);
}
