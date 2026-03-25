import { Dataset } from "./types";
import { mean, variance } from "./utils/descriptive";
import { invertMatrix } from "./utils/linalg";
import { pairwiseEuclidean as nativePairwiseEuclidean } from "./utils/native-stats";

/**
 * Euclidean distance between two vectors.
 *
 * @param a - First vector
 * @param b - Second vector
 * @returns The L2 (Euclidean) distance between a and b
 * @throws If vectors are empty or have different lengths
 *
 * @example
 * ```ts
 * euclidean([1, 2, 3], [4, 5, 6]); // ~5.196
 * ```
 */
export function euclidean(a: Dataset, b: Dataset): number {
  assertSameLength(a, b);
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += (a[i] - b[i]) ** 2;
  return Math.sqrt(sum);
}

/**
 * Manhattan (city-block / L1) distance between two vectors.
 *
 * d(a, b) = sum_i |a_i - b_i|
 *
 * @param a - First vector
 * @param b - Second vector
 * @returns The L1 (Manhattan) distance between a and b (non-negative)
 * @throws {Error} If vectors are empty or have different lengths
 *
 * @example
 * ```ts
 * manhattan([1, 2, 3], [4, 5, 6]); // 9
 * ```
 */
export function manhattan(a: Dataset, b: Dataset): number {
  assertSameLength(a, b);
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += Math.abs(a[i] - b[i]);
  return sum;
}

/**
 * Chebyshev (L-infinity) distance between two vectors.
 *
 * d(a, b) = max_i |a_i - b_i|
 *
 * @param a - First vector
 * @param b - Second vector
 * @returns The L-infinity (Chebyshev) distance between a and b (non-negative)
 * @throws {Error} If vectors are empty or have different lengths
 *
 * @example
 * ```ts
 * chebyshev([1, 2, 3], [4, 6, 5]); // 4 (max of |1-4|, |2-6|, |3-5|)
 * ```
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
 * d(a, b) = ( sum_i |a_i - b_i|^p )^{1/p}
 *
 * Special cases: p=1 gives Manhattan distance, p=2 gives Euclidean distance,
 * p=Infinity gives Chebyshev distance.
 *
 * @param a - First vector
 * @param b - Second vector
 * @param p - The order of the Minkowski metric (p >= 1, or Infinity)
 * @returns The Minkowski distance of order p between a and b (non-negative)
 * @throws {Error} If vectors are empty or have different lengths
 * @throws {Error} If p < 1 or is not finite (except Infinity)
 *
 * @example
 * ```ts
 * minkowski([1, 2], [4, 6], 2); // same as euclidean
 * minkowski([1, 2], [4, 6], 1); // same as manhattan
 * ```
 */
export function minkowski(a: Dataset, b: Dataset, p: number): number {
  assertSameLength(a, b);
  if (p < 1) throw new Error("Minkowski order p must be >= 1");
  if (!Number.isFinite(p)) {
    if (p === Infinity) return chebyshev(a, b);
    throw new Error("Minkowski order p must be finite or Infinity");
  }

  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += Math.abs(a[i] - b[i]) ** p;
  return Math.pow(sum, 1 / p);
}

/**
 * Cosine similarity between two vectors.
 *
 * Returns a value between -1 and 1, where 1 means identical direction,
 * 0 means orthogonal, and -1 means opposite direction.
 *
 * @param a - First vector
 * @param b - Second vector
 * @returns The cosine similarity in the range [-1, 1]
 * @throws If vectors are empty, have different lengths, or either is a zero vector
 *
 * @example
 * ```ts
 * cosineSimilarity([1, 0], [0, 1]); // 0 (orthogonal)
 * cosineSimilarity([1, 2], [2, 4]); // 1 (same direction)
 * ```
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
 *
 * @param a - First vector
 * @param b - Second vector
 * @returns The cosine distance in the range [0, 2]
 * @throws If vectors are empty, have different lengths, or either is a zero vector
 *
 * @example
 * ```ts
 * cosineDistance([1, 0], [0, 1]); // 1 (orthogonal vectors)
 * ```
 */
export function cosineDistance(a: Dataset, b: Dataset): number {
  return 1 - cosineSimilarity(a, b);
}

/**
 * Jaccard index for two binary sets represented as arrays.
 *
 * Measures the overlap between two sets: |A intersection B| / |A union B|.
 *
 * @param a - First set of elements
 * @param b - Second set of elements
 * @returns The Jaccard index in the range [0, 1]
 * @throws Never (returns 1 for two empty sets)
 *
 * @example
 * ```ts
 * jaccardIndex([1, 2, 3], [2, 3, 4]); // 0.5 (2 shared out of 4 unique)
 * ```
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
 *
 * @param a - First set of elements
 * @param b - Second set of elements
 * @returns The Jaccard distance in the range [0, 1]
 * @throws Never
 *
 * @example
 * ```ts
 * jaccardDistance([1, 2, 3], [2, 3, 4]); // 0.5
 * ```
 */
export function jaccardDistance(a: number[], b: number[]): number {
  return 1 - jaccardIndex(a, b);
}

/**
 * Mahalanobis distance of a point from a distribution.
 *
 * Computes d = sqrt( (x - mu)' * Sigma^{-1} * (x - mu) ) where mu is the
 * sample mean vector and Sigma is the sample covariance matrix of the data.
 *
 * This distance accounts for correlations and scale differences between
 * dimensions, unlike Euclidean distance which treats all dimensions equally.
 *
 * @param point - The point to measure (length p)
 * @param data - Dataset as array of observation vectors (n x p, each row is an observation)
 * @returns The Mahalanobis distance (non-negative scalar)
 * @throws {Error} If data has fewer than 2 observations
 * @throws {Error} If observations have inconsistent dimensionality
 * @throws {Error} If n <= p (not enough observations for covariance estimation)
 * @throws {Error} If the covariance matrix is singular
 * @throws {Error} If any value is not finite
 *
 * @example
 * ```ts
 * const data = [[1, 0], [0, 1], [-1, 0], [0, -1]];
 * const dist = mahalanobis([2, 0], data);
 * console.log(dist); // distance accounting for the data's covariance structure
 * ```
 */
export function mahalanobis(point: Dataset, data: Dataset[]): number {
  if (data.length < 2) throw new Error("Need at least 2 observations");
  const p = point.length;
  const n = data.length;

  if (n <= p) {
    throw new Error(
      `Need more observations than dimensions: n=${n} must be > p=${p}`,
    );
  }

  if (data.some((row) => row.length !== p)) {
    throw new Error("All observations must have the same dimensionality as the point");
  }

  // NaN / Infinity guards
  for (let j = 0; j < p; j++) {
    if (!Number.isFinite(point[j])) {
      throw new Error(`point[${j}] is not finite`);
    }
  }
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < p; j++) {
      if (!Number.isFinite(data[i][j])) {
        throw new Error(`data[${i}][${j}] is not finite`);
      }
    }
  }

  // Compute means
  const means = new Array<number>(p).fill(0);
  for (const row of data) {
    for (let j = 0; j < p; j++) means[j] += row[j];
  }
  for (let j = 0; j < p; j++) means[j] /= n;

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
      cov[i][j] /= n - 1;
    }
  }

  // Invert covariance matrix (uses LAPACK when available)
  const inv = invertMatrix(cov);
  if (inv === null) {
    throw new Error("Covariance matrix is singular and cannot be inverted");
  }

  // Compute (x - mu)' * Sigma^{-1} * (x - mu)
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
 * @returns A symmetric n x n matrix where entry [i][j] is the distance between vectors[i] and vectors[j]
 * @throws If any call to the metric function throws
 *
 * @example
 * ```ts
 * const m = distanceMatrix([[0, 0], [1, 0], [0, 1]]);
 * console.log(m[0][1]); // 1 (Euclidean distance)
 * ```
 */
export function distanceMatrix(
  vectors: Dataset[],
  metric: (a: Dataset, b: Dataset) => number = euclidean,
): number[][] {
  const n = vectors.length;

  // Use Fortran-accelerated pairwise Euclidean when the default metric is used
  if (metric === euclidean && n > 0) {
    return nativePairwiseEuclidean(vectors);
  }

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

// -- Helpers -----------------------------------------------------------------

function assertSameLength(a: Dataset, b: Dataset): void {
  if (a.length === 0) throw new Error("Vectors must not be empty");
  if (a.length !== b.length) {
    throw new Error(`Vectors must have the same length (got ${a.length} and ${b.length})`);
  }
}
