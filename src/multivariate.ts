/**
 * Multivariate statistics: PCA, factor analysis, k-means, hierarchical clustering.
 */

import { transpose, matMul, createRng, symmetricEigen, type Matrix } from "./utils/linalg";

// ── Types ───────────────────────────────────────────────────────────────

export interface PCAResult {
  /** Eigenvalues in descending order. */
  eigenvalues: number[];
  /** Eigenvectors (principal components) as column vectors; components[i] is the i-th PC. */
  components: number[][];
  /** Proportion of variance explained by each component. */
  explainedVariance: number[];
  /** Cumulative proportion of variance explained. */
  cumulativeVariance: number[];
  /** Projected data in the PC space (scores). */
  scores: Matrix;
  /** Number of original features. */
  nFeatures: number;
  /** Number of observations. */
  nObservations: number;
}

export interface FactorAnalysisResult {
  /** Factor loadings matrix (nFeatures x nFactors). */
  loadings: Matrix;
  /** Uniquenesses (specific variances) for each variable. */
  uniquenesses: number[];
  /** Communalities for each variable. */
  communalities: number[];
  /** Proportion of variance explained by each factor. */
  explainedVariance: number[];
  /** Number of factors extracted. */
  nFactors: number;
}

export interface KMeansResult {
  /** Cluster assignments for each observation (0-indexed). */
  assignments: number[];
  /** Cluster centroids (k x p). */
  centroids: Matrix;
  /** Number of iterations until convergence. */
  iterations: number;
  /** Within-cluster sum of squares for each cluster. */
  wcss: number[];
  /** Total within-cluster sum of squares. */
  totalWCSS: number;
}

export interface HierarchicalClusterResult {
  /** Cluster assignments for each observation (0-indexed). */
  assignments: number[];
  /** Merge history: each entry [i, j, distance] records which clusters merged. */
  merges: [number, number, number][];
  /** Number of clusters. */
  nClusters: number;
}

// ── Internal helpers ────────────────────────────────────────────────────

/** Compute column means of a matrix. */
function colMeans(X: Matrix): number[] {
  const n = X.length;
  const p = X[0].length;
  const means = new Array(p).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < p; j++) {
      means[j] += X[i][j];
    }
  }
  for (let j = 0; j < p; j++) means[j] /= n;
  return means;
}

/** Center a matrix (subtract column means). Returns new matrix and means. */
function centerMatrix(X: Matrix): { centered: Matrix; means: number[] } {
  const means = colMeans(X);
  const n = X.length;
  const p = X[0].length;
  const centered: Matrix = Array.from({ length: n }, () => new Array(p));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < p; j++) {
      centered[i][j] = X[i][j] - means[j];
    }
  }
  return { centered, means };
}

/** Compute covariance matrix from a centered data matrix. */
function covarianceMatrix(centered: Matrix): Matrix {
  const n = centered.length;
  const p = centered[0].length;
  const Ct = transpose(centered);
  const cov = matMul(Ct, centered);
  for (let i = 0; i < p; i++) {
    for (let j = 0; j < p; j++) {
      cov[i][j] /= n - 1;
    }
  }
  return cov;
}

// ── PCA ─────────────────────────────────────────────────────────────────

/**
 * Principal Component Analysis.
 *
 * Reduces dimensionality by finding orthogonal directions of maximum variance.
 * Uses eigendecomposition of the covariance matrix (LAPACK DSYEV when
 * available, Jacobi iteration fallback).
 *
 * @param data - Data matrix (n observations x p features)
 * @param options - Configuration options
 * @param options.nComponents - Number of components to retain (default: all)
 * @param options.center - Whether to center the data (default: true)
 * @param options.scale - Whether to standardize to unit variance (default: false)
 */
export function pca(
  data: Matrix,
  options: { nComponents?: number; center?: boolean; scale?: boolean } = {},
): PCAResult {
  const n = data.length;
  const p = data[0].length;
  if (n < 2) throw new Error("Need at least 2 observations");
  if (p < 1) throw new Error("Need at least 1 feature");

  const shouldCenter = options.center !== false;
  const shouldScale = options.scale === true;
  const nComp = options.nComponents ?? p;

  if (nComp < 1 || nComp > p) {
    throw new Error(`nComponents must be between 1 and ${p}`);
  }

  // Center and optionally scale
  let X: Matrix;
  if (shouldCenter) {
    const { centered } = centerMatrix(data);
    X = centered;
  } else {
    X = data.map((row) => [...row]);
  }

  if (shouldScale) {
    const stds = new Array(p);
    for (let j = 0; j < p; j++) {
      let sum2 = 0;
      for (let i = 0; i < n; i++) sum2 += X[i][j] ** 2;
      stds[j] = Math.sqrt(sum2 / (n - 1));
      if (stds[j] < 1e-15) stds[j] = 1;
    }
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < p; j++) {
        X[i][j] /= stds[j];
      }
    }
  }

  // Covariance matrix and eigen decomposition
  const cov = covarianceMatrix(X);
  const { eigenvalues, eigenvectors } = symmetricEigen(cov);

  // Clip small negative eigenvalues from numerical error
  for (let i = 0; i < eigenvalues.length; i++) {
    if (eigenvalues[i] < 0) eigenvalues[i] = 0;
  }

  const totalVar = eigenvalues.reduce((s, v) => s + v, 0);
  const explainedVariance = eigenvalues.slice(0, nComp).map((v) => (totalVar > 0 ? v / totalVar : 0));

  const cumulativeVariance: number[] = [];
  let cumSum = 0;
  for (const ev of explainedVariance) {
    cumSum += ev;
    cumulativeVariance.push(cumSum);
  }

  // Extract components (eigenvectors as column vectors)
  const components: number[][] = [];
  for (let k = 0; k < nComp; k++) {
    const comp = new Array(p);
    for (let j = 0; j < p; j++) {
      comp[j] = eigenvectors[j][k];
    }
    components.push(comp);
  }

  // Project data onto components
  const W: Matrix = Array.from({ length: p }, (_, i) => {
    const row = new Array(nComp);
    for (let k = 0; k < nComp; k++) row[k] = eigenvectors[i][k];
    return row;
  });
  const scores = matMul(X, W);

  return {
    eigenvalues: eigenvalues.slice(0, nComp),
    components,
    explainedVariance,
    cumulativeVariance,
    scores,
    nFeatures: p,
    nObservations: n,
  };
}

// ── Factor Analysis ─────────────────────────────────────────────────────

/**
 * Factor Analysis via principal axis factoring.
 *
 * Extracts latent factors that explain the shared variance among observed
 * variables. Uses iterative principal axis factoring with communality
 * estimation.
 *
 * @param data - Data matrix (n observations x p features)
 * @param nFactors - Number of factors to extract
 * @param options - Configuration options
 * @param options.maxIter - Maximum iterations (default: 100)
 * @param options.tol - Convergence tolerance (default: 1e-6)
 */
export function factorAnalysis(
  data: Matrix,
  nFactors: number,
  options: { maxIter?: number; tol?: number } = {},
): FactorAnalysisResult {
  const n = data.length;
  const p = data[0].length;
  if (n < 2) throw new Error("Need at least 2 observations");
  if (nFactors < 1 || nFactors > p) {
    throw new Error(`nFactors must be between 1 and ${p}`);
  }

  const maxIter = options.maxIter ?? 100;
  const tol = options.tol ?? 1e-6;

  const { centered } = centerMatrix(data);
  const cov = covarianceMatrix(centered);

  // Initial communality estimates: squared multiple correlations (use R^2 approximation)
  // Approximate with 1 - 1/diag(inv(cor)), but simpler: use max abs off-diagonal correlation
  const communalities = new Array(p);
  for (let j = 0; j < p; j++) {
    let maxCor = 0;
    for (let k = 0; k < p; k++) {
      if (k !== j) {
        const denom = Math.sqrt(cov[j][j] * cov[k][k]);
        if (denom > 0) maxCor = Math.max(maxCor, Math.abs(cov[j][k] / denom));
      }
    }
    communalities[j] = maxCor * maxCor;
  }

  let loadings: Matrix = [];

  for (let iter = 0; iter < maxIter; iter++) {
    // Reduced correlation matrix: replace diagonal with communalities
    const reduced: Matrix = cov.map((row) => [...row]);
    for (let j = 0; j < p; j++) {
      reduced[j][j] = communalities[j];
    }

    // Eigen decomposition
    const { eigenvalues, eigenvectors } = symmetricEigen(reduced);

    // Extract factor loadings: L = V * sqrt(lambda) for top nFactors
    loadings = Array.from({ length: p }, () => new Array(nFactors));
    for (let i = 0; i < p; i++) {
      for (let k = 0; k < nFactors; k++) {
        const ev = Math.max(0, eigenvalues[k]);
        loadings[i][k] = eigenvectors[i][k] * Math.sqrt(ev);
      }
    }

    // Update communalities
    const newCommunalities = new Array(p);
    let maxDiff = 0;
    for (let j = 0; j < p; j++) {
      let h2 = 0;
      for (let k = 0; k < nFactors; k++) {
        h2 += loadings[j][k] ** 2;
      }
      // Clamp to [0, cov[j][j]]
      newCommunalities[j] = Math.min(h2, cov[j][j]);
      maxDiff = Math.max(maxDiff, Math.abs(newCommunalities[j] - communalities[j]));
      communalities[j] = newCommunalities[j];
    }

    if (maxDiff < tol) break;
  }

  // Compute uniquenesses and explained variance
  const uniquenesses = new Array(p);
  for (let j = 0; j < p; j++) {
    uniquenesses[j] = cov[j][j] - communalities[j];
    if (uniquenesses[j] < 0) uniquenesses[j] = 0;
  }

  const totalVar = cov.reduce((s, row, i) => s + row[i], 0);
  const explainedVariance = new Array(nFactors);
  for (let k = 0; k < nFactors; k++) {
    let factorVar = 0;
    for (let j = 0; j < p; j++) {
      factorVar += loadings[j][k] ** 2;
    }
    explainedVariance[k] = totalVar > 0 ? factorVar / totalVar : 0;
  }

  return {
    loadings,
    uniquenesses,
    communalities,
    explainedVariance,
    nFactors,
  };
}

// ── K-Means Clustering ──────────────────────────────────────────────────

/** Squared Euclidean distance between two vectors. */
function sqDist(a: number[], b: number[]): number {
  let d = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i];
    d += diff * diff;
  }
  return d;
}

/**
 * K-Means clustering.
 *
 * Partitions n observations into k clusters by minimizing within-cluster
 * sum of squares. Uses k-means++ initialization for better convergence.
 *
 * @param data - Data matrix (n observations x p features)
 * @param k - Number of clusters
 * @param options - Configuration options
 * @param options.maxIter - Maximum iterations (default: 300)
 * @param options.tol - Convergence tolerance for centroid movement (default: 1e-6)
 * @param options.seed - Random seed for reproducibility
 * @param options.nInit - Number of initializations, best result kept (default: 10)
 */
export function kMeans(
  data: Matrix,
  k: number,
  options: { maxIter?: number; tol?: number; seed?: number; nInit?: number } = {},
): KMeansResult {
  const n = data.length;
  const p = data[0].length;
  if (n < k) throw new Error("Number of observations must be >= k");
  if (k < 1) throw new Error("k must be at least 1");

  const maxIter = options.maxIter ?? 300;
  const tolSq = (options.tol ?? 1e-6) ** 2;
  const nInit = options.nInit ?? 10;
  const baseSeed = options.seed ?? Math.floor(Math.random() * 1e9);

  let bestResult: KMeansResult | null = null;

  for (let init = 0; init < nInit; init++) {
    const rng = createRng(baseSeed + init);

    // K-means++ initialization
    const centroids: Matrix = [];
    const firstIdx = Math.floor(rng() * n);
    centroids.push([...data[firstIdx]]);

    const minDists = new Array(n).fill(Infinity);
    for (let c = 1; c < k; c++) {
      // Update min distances
      for (let i = 0; i < n; i++) {
        const d = sqDist(data[i], centroids[c - 1]);
        if (d < minDists[i]) minDists[i] = d;
      }
      const total = minDists.reduce((s, d) => s + d, 0);
      if (total === 0) {
        // All remaining points coincide with existing centroids
        centroids.push([...data[Math.floor(rng() * n)]]);
        continue;
      }
      // Weighted random selection
      let r = rng() * total;
      let selected = 0;
      for (let i = 0; i < n; i++) {
        r -= minDists[i];
        if (r <= 0) {
          selected = i;
          break;
        }
      }
      centroids.push([...data[selected]]);
    }

    // Iterate
    const assignments = new Array(n).fill(0);
    let iterations = 0;

    for (let iter = 0; iter < maxIter; iter++) {
      iterations++;

      // Assign each point to nearest centroid
      for (let i = 0; i < n; i++) {
        let minD = Infinity;
        let best = 0;
        for (let c = 0; c < k; c++) {
          const d = sqDist(data[i], centroids[c]);
          if (d < minD) {
            minD = d;
            best = c;
          }
        }
        assignments[i] = best;
      }

      // Recompute centroids
      const counts = new Array(k).fill(0);
      const newCentroids: Matrix = Array.from({ length: k }, () => new Array(p).fill(0));
      for (let i = 0; i < n; i++) {
        const c = assignments[i];
        counts[c]++;
        for (let j = 0; j < p; j++) {
          newCentroids[c][j] += data[i][j];
        }
      }

      let maxShift = 0;
      for (let c = 0; c < k; c++) {
        if (counts[c] > 0) {
          for (let j = 0; j < p; j++) {
            newCentroids[c][j] /= counts[c];
          }
        } else {
          // Empty cluster: reinitialize to a random point
          const idx = Math.floor(rng() * n);
          for (let j = 0; j < p; j++) {
            newCentroids[c][j] = data[idx][j];
          }
        }
        maxShift = Math.max(maxShift, sqDist(centroids[c], newCentroids[c]));
        centroids[c] = newCentroids[c];
      }

      if (maxShift < tolSq) break;
    }

    // Compute WCSS
    const wcss = new Array(k).fill(0);
    for (let i = 0; i < n; i++) {
      wcss[assignments[i]] += sqDist(data[i], centroids[assignments[i]]);
    }
    const totalWCSS = wcss.reduce((s, v) => s + v, 0);

    if (bestResult === null || totalWCSS < bestResult.totalWCSS) {
      bestResult = { assignments: [...assignments], centroids, iterations, wcss, totalWCSS };
    }
  }

  return bestResult!;
}

// ── Hierarchical Clustering ─────────────────────────────────────────────

export type Linkage = "single" | "complete" | "average";

/**
 * Agglomerative hierarchical clustering.
 *
 * Builds a hierarchy of clusters by successively merging the closest pair.
 * Supports single, complete, and average linkage.
 *
 * @param data - Data matrix (n observations x p features)
 * @param nClusters - Desired number of clusters
 * @param options - Configuration options
 * @param options.linkage - Linkage method: "single", "complete", "average" (default: "complete")
 */
export function hierarchicalClustering(
  data: Matrix,
  nClusters: number,
  options: { linkage?: Linkage } = {},
): HierarchicalClusterResult {
  const n = data.length;
  if (n < 1) throw new Error("Need at least 1 observation");
  if (nClusters < 1 || nClusters > n) {
    throw new Error(`nClusters must be between 1 and ${n}`);
  }

  const linkage = options.linkage ?? "complete";

  // Compute full distance matrix
  const dist: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const d = Math.sqrt(sqDist(data[i], data[j]));
      dist[i][j] = d;
      dist[j][i] = d;
    }
  }

  // Each observation starts as its own cluster
  const clusters: Set<number>[] = Array.from({ length: n }, (_, i) => new Set([i]));
  const active = new Set<number>(Array.from({ length: n }, (_, i) => i));
  const merges: [number, number, number][] = [];

  // Merge until we reach the desired number of clusters
  while (active.size > nClusters) {
    // Find the closest pair of active clusters
    let minDist = Infinity;
    let mergeA = -1;
    let mergeB = -1;

    const activeArr = Array.from(active);
    for (let ai = 0; ai < activeArr.length; ai++) {
      for (let bi = ai + 1; bi < activeArr.length; bi++) {
        const a = activeArr[ai];
        const b = activeArr[bi];
        const d = clusterDistance(clusters[a], clusters[b], dist, linkage);
        if (d < minDist) {
          minDist = d;
          mergeA = a;
          mergeB = b;
        }
      }
    }

    // Merge B into A
    merges.push([mergeA, mergeB, minDist]);
    for (const idx of clusters[mergeB]) {
      clusters[mergeA].add(idx);
    }
    active.delete(mergeB);
  }

  // Build assignments
  const assignments = new Array(n);
  let clusterIdx = 0;
  for (const cIdx of active) {
    for (const obsIdx of clusters[cIdx]) {
      assignments[obsIdx] = clusterIdx;
    }
    clusterIdx++;
  }

  return { assignments, merges, nClusters };
}

function clusterDistance(
  a: Set<number>,
  b: Set<number>,
  dist: number[][],
  linkage: Linkage,
): number {
  if (linkage === "single") {
    let min = Infinity;
    for (const i of a) {
      for (const j of b) {
        if (dist[i][j] < min) min = dist[i][j];
      }
    }
    return min;
  }

  if (linkage === "complete") {
    let max = 0;
    for (const i of a) {
      for (const j of b) {
        if (dist[i][j] > max) max = dist[i][j];
      }
    }
    return max;
  }

  // average
  let sum = 0;
  let count = 0;
  for (const i of a) {
    for (const j of b) {
      sum += dist[i][j];
      count++;
    }
  }
  return sum / count;
}
