import { Dataset } from "./types";
import { mean, variance } from "./utils/descriptive";

/**
 * Euclidean distance between two vectors.
 */
export function euclidean(a: Dataset, b: Dataset): number {
  assertSameLength(a, b);
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += (a[i] - b[i]) ** 2;
  return Math.sqrt(sum);
}

/**
 * Manhattan (city-block / L1) distance between two vectors.
 */
export function manhattan(a: Dataset, b: Dataset): number {
  assertSameLength(a, b);
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += Math.abs(a[i] - b[i]);
  return sum;
}

/**
 * Chebyshev (L∞) distance between two vectors.
 */
export function chebyshev(a: Dataset, b: Dataset): number {
  assertSameLength(a, b);
  let max = 0;
  for (let i = 0; i < a.length; i++) max = Math.max(max, Math.abs(a[i] - b[i]));
  return max;
}

/**
 * Minkowski distance between two vectors.
 *
 * @param p - The order of the Minkowski metric (p >= 1)
 */
export function minkowski(a: Dataset, b: Dataset, p: number): number {
  assertSameLength(a, b);
  if (p < 1) throw new Error("Minkowski order p must be >= 1");
  if (p === Infinity) return chebyshev(a, b);

  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += Math.abs(a[i] - b[i]) ** p;
  return Math.pow(sum, 1 / p);
}

/**
 * Cosine similarity between two vectors.
 *
 * Returns a value between -1 and 1, where 1 means identical direction,
 * 0 means orthogonal, and -1 means opposite direction.
 */
export function cosineSimilarity(a: Dataset, b: Dataset): number {
  assertSameLength(a, b);
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  if (denom === 0) throw new Error("Cannot compute cosine similarity for zero vectors");
  return dot / denom;
}

/**
 * Cosine distance between two vectors (1 - cosine similarity).
 */
export function cosineDistance(a: Dataset, b: Dataset): number {
  return 1 - cosineSimilarity(a, b);
}

/**
 * Jaccard index for two binary sets represented as arrays.
 *
 * Measures the overlap between two sets: |A ∩ B| / |A ∪ B|.
 *
 * @param a - First set of elements
 * @param b - Second set of elements
 */
export function jaccardIndex(a: number[], b: number[]): number {
  const setA = new Set(a);
  const setB = new Set(b);
  let intersection = 0;
  for (const v of setA) {
    if (setB.has(v)) intersection++;
  }
  const union = new Set([...a, ...b]).size;
  if (union === 0) return 1; // Both empty sets
  return intersection / union;
}

/**
 * Jaccard distance (1 - Jaccard index).
 */
export function jaccardDistance(a: number[], b: number[]): number {
  return 1 - jaccardIndex(a, b);
}

/**
 * Mahalanobis distance of a point from a distribution.
 *
 * Uses the inverse covariance matrix to account for correlations and
 * scale differences between dimensions.
 *
 * @param point - The point to measure
 * @param data - Dataset as array of observation vectors (each row is an observation)
 */
export function mahalanobis(point: Dataset, data: Dataset[]): number {
  if (data.length < 2) throw new Error("Need at least 2 observations");
  const p = point.length;
  if (data.some((row) => row.length !== p)) {
    throw new Error("All observations must have the same dimensionality as the point");
  }

  // Compute means
  const means = new Array<number>(p).fill(0);
  for (const row of data) {
    for (let j = 0; j < p; j++) means[j] += row[j];
  }
  for (let j = 0; j < p; j++) means[j] /= data.length;

  // Compute covariance matrix
  const cov = Array.from({ length: p }, () => new Array<number>(p).fill(0));
  for (const row of data) {
    for (let i = 0; i < p; i++) {
      for (let j = 0; j < p; j++) {
        cov[i][j] += (row[i] - means[i]) * (row[j] - means[j]);
      }
    }
  }
  for (let i = 0; i < p; i++) {
    for (let j = 0; j < p; j++) {
      cov[i][j] /= data.length - 1;
    }
  }

  // Invert covariance matrix (Gauss-Jordan for small matrices)
  const inv = invertMatrix(cov);

  // Compute (x - μ)' * Σ^{-1} * (x - μ)
  const diff = point.map((v, i) => v - means[i]);
  let result = 0;
  for (let i = 0; i < p; i++) {
    for (let j = 0; j < p; j++) {
      result += diff[i] * inv[i][j] * diff[j];
    }
  }

  return Math.sqrt(result);
}

/**
 * Compute a pairwise distance matrix for a set of vectors.
 *
 * @param vectors - Array of vectors
 * @param metric - Distance function (default: euclidean)
 */
export function distanceMatrix(
  vectors: Dataset[],
  metric: (a: Dataset, b: Dataset) => number = euclidean,
): number[][] {
  const n = vectors.length;
  const matrix = Array.from({ length: n }, () => new Array<number>(n).fill(0));

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const d = metric(vectors[i], vectors[j]);
      matrix[i][j] = d;
      matrix[j][i] = d;
    }
  }

  return matrix;
}

// ── Helpers ─────────────────────────────────────────────────────────────

function assertSameLength(a: Dataset, b: Dataset): void {
  if (a.length === 0) throw new Error("Vectors must not be empty");
  if (a.length !== b.length) {
    throw new Error(`Vectors must have the same length (got ${a.length} and ${b.length})`);
  }
}

function invertMatrix(matrix: number[][]): number[][] {
  const n = matrix.length;
  // Augment with identity
  const aug = matrix.map((row, i) => {
    const r = [...row];
    for (let j = 0; j < n; j++) r.push(i === j ? 1 : 0);
    return r;
  });

  for (let col = 0; col < n; col++) {
    // Find pivot
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(aug[row][col]) > Math.abs(aug[maxRow][col])) maxRow = row;
    }
    [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];

    const pivot = aug[col][col];
    if (Math.abs(pivot) < 1e-12) {
      throw new Error("Covariance matrix is singular and cannot be inverted");
    }

    // Scale pivot row
    for (let j = 0; j < 2 * n; j++) aug[col][j] /= pivot;

    // Eliminate column
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = aug[row][col];
      for (let j = 0; j < 2 * n; j++) aug[row][j] -= factor * aug[col][j];
    }
  }

  return aug.map((row) => row.slice(n));
}
