import { Dataset } from "./types";
import { mean, variance } from "./utils/descriptive";
import { euclidean } from "./distance";
import { createRng } from "./utils/linalg";

/**
 * Result of t-SNE dimensionality reduction.
 *
 * Contains the 2D embedding coordinates, the final KL divergence
 * (measuring how well the embedding preserves the original neighborhood
 * structure), and the number of iterations performed.
 */
export interface TSNEResult {
  /** Embedded 2D coordinates. */
  embedding: number[][];
  /** Final KL divergence. */
  klDivergence: number;
  /** Number of iterations actually performed. */
  iterations: number;
}

/**
 * t-SNE (t-distributed Stochastic Neighbor Embedding) for nonlinear
 * dimensionality reduction.
 *
 * Reduces high-dimensional data to 2D by minimizing the KL divergence
 * between pairwise probability distributions in the original and embedded
 * spaces. Uses a Student-t distribution with 1 degree of freedom in the
 * low-dimensional space to model heavy tails and avoid the crowding problem.
 *
 * Uses the exact O(n^2) algorithm with adaptive learning rate gains,
 * momentum, and early stopping if KL divergence stops improving.
 *
 * @param data - Data matrix (n observations x p features)
 * @param options - Configuration options
 * @param options.perplexity - Perplexity parameter controlling neighborhood size
 *   (default: min(30, floor(n/3)))
 * @param options.learningRate - Gradient descent learning rate (default: 200)
 * @param options.iterations - Maximum number of iterations (default: 500)
 * @param options.seed - Random seed for reproducibility (optional)
 * @returns A {@link TSNEResult} with the 2D embedding, final KL divergence, and iteration count
 * @throws {Error} If fewer than 4 observations
 * @throws {Error} If any data value is NaN or Infinity
 *
 * @example
 * ```ts
 * const data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]];
 * const result = tsne(data, { perplexity: 2, seed: 42 });
 * console.log(result.embedding); // [[x1, y1], [x2, y2], ...]
 * ```
 */
export function tsne(
  data: number[][],
  options: {
    perplexity?: number;
    learningRate?: number;
    iterations?: number;
    seed?: number;
  } = {},
): TSNEResult {
  const n = data.length;
  if (n < 4) throw new Error("Need at least 4 observations");

  // NaN/Infinity guard
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < data[i].length; j++) {
      if (!Number.isFinite(data[i][j])) {
        throw new Error("Data must not contain NaN or Infinity");
      }
    }
  }

  const perplexity = options.perplexity ?? Math.min(30, Math.floor(n / 3));
  const learningRate = options.learningRate ?? 200;
  const iterations = options.iterations ?? 500;
  const rng = options.seed != null ? createRng(options.seed) : Math.random;

  // Compute pairwise distances
  const dist = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => euclidean(data[i], data[j])),
  );

  // Compute P (symmetrized conditional probabilities)
  const P = computeJointProbabilities(dist, perplexity, n);

  // Initialize embedding randomly
  const Y = Array.from({ length: n }, () => [
    (rng() - 0.5) * 0.01,
    (rng() - 0.5) * 0.01,
  ]);

  // Gradient descent
  const gains = Array.from({ length: n }, () => [1, 1]);
  const yIncs = Array.from({ length: n }, () => [0, 0]);
  const momentum = 0.5;
  const finalMomentum = 0.8;
  const momentumSwitch = 250;

  let klDiv = 0;
  let prevKlDiv = Infinity;
  let noImprovementCount = 0;
  const earlyStopPatience = 50;
  const earlyStopTolerance = 1e-7;
  let actualIterations = iterations;

  for (let iter = 0; iter < iterations; iter++) {
    // Compute Q (Student-t with 1 degree of freedom)
    const Q = Array.from({ length: n }, () => new Array<number>(n).fill(0));
    let qSum = 0;

    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const d = (Y[i][0] - Y[j][0]) ** 2 + (Y[i][1] - Y[j][1]) ** 2;
        const q = 1 / (1 + d);
        Q[i][j] = q;
        Q[j][i] = q;
        qSum += 2 * q;
      }
    }

    // Normalize Q
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        Q[i][j] = Math.max(Q[i][j] / qSum, 1e-12);
      }
    }

    // Compute gradients
    const grad = Array.from({ length: n }, () => [0, 0]);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const d = (Y[i][0] - Y[j][0]) ** 2 + (Y[i][1] - Y[j][1]) ** 2;
        const mult = 4 * (P[i][j] - Q[i][j]) / (1 + d);
        grad[i][0] += mult * (Y[i][0] - Y[j][0]);
        grad[i][1] += mult * (Y[i][1] - Y[j][1]);
      }
    }

    // Update embedding with adaptive gains
    const mom = iter < momentumSwitch ? momentum : finalMomentum;
    for (let i = 0; i < n; i++) {
      for (let d = 0; d < 2; d++) {
        const sign = Math.sign(grad[i][d]) !== Math.sign(yIncs[i][d]);
        gains[i][d] = sign ? gains[i][d] + 0.2 : gains[i][d] * 0.8;
        gains[i][d] = Math.max(gains[i][d], 0.01);
        yIncs[i][d] = mom * yIncs[i][d] - learningRate * gains[i][d] * grad[i][d];
        Y[i][d] += yIncs[i][d];
      }
    }

    // Center
    const meanY = [0, 0];
    for (let i = 0; i < n; i++) {
      meanY[0] += Y[i][0];
      meanY[1] += Y[i][1];
    }
    meanY[0] /= n;
    meanY[1] /= n;
    for (let i = 0; i < n; i++) {
      Y[i][0] -= meanY[0];
      Y[i][1] -= meanY[1];
    }

    // KL divergence (compute periodically for early stopping, and on last iteration)
    if (iter % 10 === 0 || iter === iterations - 1) {
      klDiv = 0;
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          if (i !== j && P[i][j] > 1e-12) {
            klDiv += P[i][j] * Math.log(P[i][j] / Q[i][j]);
          }
        }
      }

      // Early stopping check
      if (iter > 0) {
        if (prevKlDiv - klDiv < earlyStopTolerance) {
          noImprovementCount++;
        } else {
          noImprovementCount = 0;
        }
        if (noImprovementCount >= earlyStopPatience / 10) {
          actualIterations = iter + 1;
          break;
        }
      }
      prevKlDiv = klDiv;
    }
  }

  return { embedding: Y, klDivergence: klDiv, iterations: actualIterations };
}

/**
 * Silhouette score for evaluating clustering quality.
 *
 * Measures how similar each point is to its own cluster vs. the nearest
 * other cluster. Returns values in [-1, 1]: higher is better.
 *
 * @param data - Data matrix (n x p)
 * @param labels - Cluster labels for each observation
 * @returns Mean silhouette score across all observations (in [-1, 1])
 * @throws {Error} If data and labels have different lengths
 * @throws {Error} If fewer than 2 observations
 * @throws {Error} If fewer than 2 distinct clusters
 * @throws {Error} If any data value is NaN or Infinity
 *
 * @example
 * ```ts
 * const data = [[0, 0], [1, 0], [0, 1], [10, 10], [11, 10], [10, 11]];
 * const labels = [0, 0, 0, 1, 1, 1];
 * const score = silhouetteScore(data, labels);
 * console.log(score); // close to 1.0 (well-separated clusters)
 * ```
 */
export function silhouetteScore(data: number[][], labels: number[]): number {
  const n = data.length;
  if (n !== labels.length) throw new Error("Data and labels must have same length");
  if (n < 2) throw new Error("Need at least 2 observations");

  // NaN/Infinity guard
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < data[i].length; j++) {
      if (!Number.isFinite(data[i][j])) {
        throw new Error("Data must not contain NaN or Infinity");
      }
    }
  }

  const uniqueLabels = [...new Set(labels)];
  if (uniqueLabels.length < 2) throw new Error("Need at least 2 clusters");

  const scores = new Array<number>(n);

  for (let i = 0; i < n; i++) {
    // a(i) = mean distance to same-cluster points
    let aDist = 0;
    let aCount = 0;

    // b(i) = min mean distance to other-cluster points
    const clusterDists = new Map<number, { sum: number; count: number }>();

    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const d = euclidean(data[i], data[j]);

      if (labels[j] === labels[i]) {
        aDist += d;
        aCount++;
      } else {
        if (!clusterDists.has(labels[j])) {
          clusterDists.set(labels[j], { sum: 0, count: 0 });
        }
        const cd = clusterDists.get(labels[j])!;
        cd.sum += d;
        cd.count++;
      }
    }

    const a = aCount > 0 ? aDist / aCount : 0;

    let b = Infinity;
    for (const [, cd] of clusterDists) {
      const avgDist = cd.sum / cd.count;
      if (avgDist < b) b = avgDist;
    }

    scores[i] = b === Infinity ? 0 : (b - a) / Math.max(a, b);
  }

  return mean(scores);
}

/**
 * Silhouette scores per observation.
 *
 * @param data - Data matrix (n x p)
 * @param labels - Cluster labels for each observation
 * @returns Array of silhouette scores, one per observation (each in [-1, 1])
 * @throws {Error} If data and labels have different lengths
 * @throws {Error} If any data value is NaN or Infinity
 */
export function silhouetteScores(data: number[][], labels: number[]): number[] {
  const n = data.length;
  if (n !== labels.length) throw new Error("Data and labels must have same length");

  // NaN/Infinity guard
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < data[i].length; j++) {
      if (!Number.isFinite(data[i][j])) {
        throw new Error("Data must not contain NaN or Infinity");
      }
    }
  }

  const scores = new Array<number>(n);

  for (let i = 0; i < n; i++) {
    let aDist = 0;
    let aCount = 0;
    const clusterDists = new Map<number, { sum: number; count: number }>();

    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const d = euclidean(data[i], data[j]);

      if (labels[j] === labels[i]) {
        aDist += d;
        aCount++;
      } else {
        if (!clusterDists.has(labels[j])) {
          clusterDists.set(labels[j], { sum: 0, count: 0 });
        }
        const cd = clusterDists.get(labels[j])!;
        cd.sum += d;
        cd.count++;
      }
    }

    const a = aCount > 0 ? aDist / aCount : 0;
    let b = Infinity;
    for (const [, cd] of clusterDists) {
      const avgDist = cd.sum / cd.count;
      if (avgDist < b) b = avgDist;
    }

    scores[i] = b === Infinity ? 0 : (b - a) / Math.max(a, b);
  }

  return scores;
}

/**
 * Davies-Bouldin index for cluster validation.
 *
 * Lower values indicate better clustering. Measures the average similarity
 * ratio of each cluster with its most similar cluster.
 *
 * @param data - Data matrix (n x p)
 * @param labels - Cluster labels
 * @returns The Davies-Bouldin index (non-negative; lower is better)
 * @throws {Error} If fewer than 2 clusters
 * @throws {Error} If any data value is NaN or Infinity
 */
export function daviesBouldinIndex(data: number[][], labels: number[]): number {
  const n = data.length;
  const uniqueLabels = [...new Set(labels)];
  const k = uniqueLabels.length;
  if (k < 2) throw new Error("Need at least 2 clusters");

  // NaN/Infinity guard
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < data[i].length; j++) {
      if (!Number.isFinite(data[i][j])) {
        throw new Error("Data must not contain NaN or Infinity");
      }
    }
  }

  // Compute cluster centroids and scatter
  const centroids = new Map<number, number[]>();
  const scatters = new Map<number, number>();

  for (const label of uniqueLabels) {
    const members = data.filter((_, i) => labels[i] === label);
    const p = data[0].length;
    const centroid = new Array<number>(p).fill(0);
    for (const m of members) {
      for (let j = 0; j < p; j++) centroid[j] += m[j];
    }
    for (let j = 0; j < p; j++) centroid[j] /= members.length;
    centroids.set(label, centroid);

    // Scatter = average distance to centroid
    const scatter = mean(members.map((m) => euclidean(m, centroid)));
    scatters.set(label, scatter);
  }

  // Compute DB index
  let dbSum = 0;
  for (const li of uniqueLabels) {
    let maxRatio = 0;
    for (const lj of uniqueLabels) {
      if (li === lj) continue;
      const dist = euclidean(centroids.get(li)!, centroids.get(lj)!);
      if (dist === 0) continue;
      const ratio = (scatters.get(li)! + scatters.get(lj)!) / dist;
      if (ratio > maxRatio) maxRatio = ratio;
    }
    dbSum += maxRatio;
  }

  return dbSum / k;
}

/**
 * Adjusted Rand Index for comparing two clusterings.
 *
 * Measures agreement between two cluster label assignments,
 * adjusted for chance. Returns values in [-1, 1]: 1 = perfect agreement.
 *
 * @param labels1 - First set of cluster labels
 * @param labels2 - Second set of cluster labels
 * @returns The Adjusted Rand Index (in [-1, 1]; 1 = perfect agreement, 0 = random)
 * @throws {Error} If label arrays have different lengths
 * @throws {Error} If fewer than 2 observations
 *
 * @example
 * ```ts
 * const truth  = [0, 0, 0, 1, 1, 1];
 * const pred   = [0, 0, 1, 1, 1, 1];
 * const ari = adjustedRandIndex(truth, pred);
 * console.log(ari); // 0.444... (partial agreement)
 *
 * const perfect = adjustedRandIndex(truth, truth);
 * console.log(perfect); // 1.0
 * ```
 */
export function adjustedRandIndex(labels1: number[], labels2: number[]): number {
  const n = labels1.length;
  if (n !== labels2.length) throw new Error("Label arrays must have same length");
  if (n < 2) throw new Error("Need at least 2 observations");

  const unique1 = [...new Set(labels1)];
  const unique2 = [...new Set(labels2)];

  // Contingency table
  const table = new Map<string, number>();
  for (let i = 0; i < n; i++) {
    const key = `${labels1[i]},${labels2[i]}`;
    table.set(key, (table.get(key) ?? 0) + 1);
  }

  // Row and column sums
  const rowSums = new Map<number, number>();
  const colSums = new Map<number, number>();
  for (const l1 of unique1) rowSums.set(l1, 0);
  for (const l2 of unique2) colSums.set(l2, 0);

  for (let i = 0; i < n; i++) {
    rowSums.set(labels1[i], (rowSums.get(labels1[i]) ?? 0) + 1);
    colSums.set(labels2[i], (colSums.get(labels2[i]) ?? 0) + 1);
  }

  // Compute ARI using C(n,2) combinations
  const choose2 = (x: number) => (x * (x - 1)) / 2;

  let sumNij2 = 0;
  for (const [, v] of table) sumNij2 += choose2(v);

  let sumA = 0;
  for (const [, v] of rowSums) sumA += choose2(v);

  let sumB = 0;
  for (const [, v] of colSums) sumB += choose2(v);

  const totalC2 = choose2(n);
  const expected = (sumA * sumB) / totalC2;
  const maxIndex = 0.5 * (sumA + sumB);

  if (maxIndex === expected) return 1;

  return (sumNij2 - expected) / (maxIndex - expected);
}

// ── Helpers ─────────────────────────────────────────────────────────────

function computeJointProbabilities(
  dist: number[][],
  perplexity: number,
  n: number,
): number[][] {
  const P = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  const logPerplexity = Math.log(perplexity);

  for (let i = 0; i < n; i++) {
    // Binary search for sigma
    let low = 1e-10;
    let high = 1e4;
    let sigma = 1;

    for (let iter = 0; iter < 50; iter++) {
      sigma = (low + high) / 2;
      let sumP = 0;
      let entropy = 0;

      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const p = Math.exp(-(dist[i][j] ** 2) / (2 * sigma * sigma));
        sumP += p;
      }

      if (sumP === 0) sumP = 1e-10;

      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const p = Math.exp(-(dist[i][j] ** 2) / (2 * sigma * sigma)) / sumP;
        if (p > 0) entropy -= p * Math.log(p);
      }

      if (entropy > logPerplexity) high = sigma;
      else low = sigma;
    }

    // Set conditional probabilities
    let sumP = 0;
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      P[i][j] = Math.exp(-(dist[i][j] ** 2) / (2 * sigma * sigma));
      sumP += P[i][j];
    }
    for (let j = 0; j < n; j++) {
      P[i][j] /= Math.max(sumP, 1e-10);
    }
  }

  // Symmetrize
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const sym = (P[i][j] + P[j][i]) / (2 * n);
      P[i][j] = Math.max(sym, 1e-12);
      P[j][i] = P[i][j];
    }
  }

  return P;
}

// ── DBSCAN ──────────────────────────────────────────────────────────────

/**
 * Result of DBSCAN clustering.
 */
export interface DBSCANResult {
  /** Cluster labels for each point (-1 = noise). */
  labels: number[];
  /** Number of clusters found. */
  nClusters: number;
  /** Number of noise points. */
  nNoise: number;
  /** Indices of core points. */
  corePoints: number[];
}

/**
 * DBSCAN (Density-Based Spatial Clustering of Applications with Noise).
 *
 * Finds clusters of arbitrary shape by grouping points that are closely
 * packed together and marking points in low-density regions as noise.
 * Does not require specifying the number of clusters in advance.
 *
 * @param data - Data matrix (n observations x p features)
 * @param options - Configuration options
 * @param options.epsilon - Neighborhood radius (default: 0.5)
 * @param options.minPoints - Minimum neighbors to form a core point (default: 5)
 * @param options.distanceMetric - Distance function (default: euclidean)
 * @returns A {@link DBSCANResult} with cluster labels, counts, and core point indices
 * @throws {Error} If fewer than 1 observation
 * @throws {Error} If epsilon <= 0 or minPoints < 1
 *
 * @example
 * ```ts
 * const data = [[0,0],[0.1,0],[0,0.1],[10,10],[10.1,10],[10,10.1]];
 * const result = dbscan(data, { epsilon: 1, minPoints: 2 });
 * console.log(result.labels); // [0,0,0,1,1,1]
 * console.log(result.nNoise); // 0
 * ```
 */
export function dbscan(
  data: number[][],
  options: {
    epsilon?: number;
    minPoints?: number;
    distanceMetric?: (a: number[], b: number[]) => number;
  } = {},
): DBSCANResult {
  const n = data.length;
  if (n < 1) throw new Error("Need at least 1 observation");

  const epsilon = options.epsilon ?? 0.5;
  const minPoints = options.minPoints ?? 5;
  const dist = options.distanceMetric ?? euclidean;

  if (epsilon <= 0) throw new Error("epsilon must be positive");
  if (minPoints < 1) throw new Error("minPoints must be at least 1");

  const labels = new Array<number>(n).fill(-2); // -2 = unvisited
  const corePoints: number[] = [];
  let clusterId = 0;

  // Region query: find all points within epsilon of point index
  function regionQuery(idx: number): number[] {
    const neighbors: number[] = [];
    for (let j = 0; j < n; j++) {
      if (dist(data[idx], data[j]) <= epsilon) {
        neighbors.push(j);
      }
    }
    return neighbors;
  }

  for (let i = 0; i < n; i++) {
    if (labels[i] !== -2) continue; // already visited

    const neighbors = regionQuery(i);

    if (neighbors.length < minPoints) {
      labels[i] = -1; // noise (may be reclaimed later)
      continue;
    }

    // Core point - start a new cluster
    corePoints.push(i);
    labels[i] = clusterId;

    const seed = new Set(neighbors);
    seed.delete(i);
    const queue = [...seed];

    while (queue.length > 0) {
      const q = queue.shift()!;

      if (labels[q] === -1) {
        // Was noise, reclaim into this cluster
        labels[q] = clusterId;
      }

      if (labels[q] !== -2) continue; // already assigned to a cluster
      labels[q] = clusterId;

      const qNeighbors = regionQuery(q);
      if (qNeighbors.length >= minPoints) {
        corePoints.push(q);
        for (const nb of qNeighbors) {
          if (labels[nb] === -2 || labels[nb] === -1) {
            if (!seed.has(nb)) {
              seed.add(nb);
              queue.push(nb);
            }
          }
        }
      }
    }

    clusterId++;
  }

  // Mark any remaining unvisited as noise
  for (let i = 0; i < n; i++) {
    if (labels[i] === -2) labels[i] = -1;
  }

  const nNoise = labels.filter((l) => l === -1).length;

  return {
    labels,
    nClusters: clusterId,
    nNoise,
    corePoints: [...new Set(corePoints)],
  };
}

// ── OPTICS ──────────────────────────────────────────────────────────────

/**
 * Result of OPTICS clustering.
 */
export interface OPTICSResult {
  /** Point indices in OPTICS ordering. */
  ordering: number[];
  /** Reachability distances in ordering. */
  reachabilityDistances: number[];
  /** Core distances for each point. */
  coreDistances: number[];
  /** Extracted cluster labels. */
  labels: number[];
  /** Number of clusters found. */
  nClusters: number;
}

/**
 * OPTICS (Ordering Points To Identify the Clustering Structure).
 *
 * Produces an augmented ordering of the data along with reachability
 * distances that can be used to extract clusters at varying density
 * thresholds. Generalizes DBSCAN.
 *
 * @param data - Data matrix (n observations x p features)
 * @param options - Configuration options
 * @param options.epsilon - Maximum neighborhood radius (default: Infinity)
 * @param options.minPoints - Minimum neighbors for core points (default: 5)
 * @param options.distanceMetric - Distance function (default: euclidean)
 * @param options.extractionEpsilon - Epsilon for flat cluster extraction (default: same as epsilon, or Infinity means auto)
 * @returns An {@link OPTICSResult} with ordering, reachability, core distances, and labels
 * @throws {Error} If fewer than 1 observation
 *
 * @example
 * ```ts
 * const data = [[0,0],[0.1,0],[0,0.1],[10,10],[10.1,10],[10,10.1]];
 * const result = optics(data, { minPoints: 2, extractionEpsilon: 1 });
 * console.log(result.ordering);
 * console.log(result.reachabilityDistances);
 * ```
 */
export function optics(
  data: number[][],
  options: {
    epsilon?: number;
    minPoints?: number;
    distanceMetric?: (a: number[], b: number[]) => number;
    extractionEpsilon?: number;
  } = {},
): OPTICSResult {
  const n = data.length;
  if (n < 1) throw new Error("Need at least 1 observation");

  const epsilon = options.epsilon ?? Infinity;
  const minPoints = options.minPoints ?? 5;
  const dist = options.distanceMetric ?? euclidean;

  const reachability = new Array<number>(n).fill(Infinity);
  const coreDist = new Array<number>(n).fill(Infinity);
  const processed = new Array<boolean>(n).fill(false);
  const ordering: number[] = [];

  // Precompute distance matrix
  const distMatrix = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => dist(data[i], data[j])),
  );

  // Compute core distances
  for (let i = 0; i < n; i++) {
    const dists = distMatrix[i]
      .map((d, j) => ({ d, j }))
      .filter(({ d }) => d <= epsilon)
      .sort((a, b) => a.d - b.d);
    if (dists.length >= minPoints) {
      coreDist[i] = dists[minPoints - 1].d;
    }
  }

  function getNeighbors(idx: number): number[] {
    const neighbors: number[] = [];
    for (let j = 0; j < n; j++) {
      if (distMatrix[idx][j] <= epsilon) {
        neighbors.push(j);
      }
    }
    return neighbors;
  }

  function update(idx: number, neighbors: number[], seeds: number[]): void {
    const cd = coreDist[idx];
    for (const nb of neighbors) {
      if (processed[nb]) continue;
      const newReach = Math.max(cd, distMatrix[idx][nb]);
      if (newReach < reachability[nb]) {
        reachability[nb] = newReach;
        if (!seeds.includes(nb)) {
          seeds.push(nb);
        }
      }
    }
  }

  for (let i = 0; i < n; i++) {
    if (processed[i]) continue;
    processed[i] = true;
    ordering.push(i);

    if (coreDist[i] === Infinity) continue;

    const neighbors = getNeighbors(i);
    const seeds: number[] = [];
    update(i, neighbors, seeds);

    while (seeds.length > 0) {
      // Pick seed with smallest reachability
      let minIdx = 0;
      for (let s = 1; s < seeds.length; s++) {
        if (reachability[seeds[s]] < reachability[seeds[minIdx]]) {
          minIdx = s;
        }
      }
      const current = seeds[minIdx];
      seeds.splice(minIdx, 1);

      if (processed[current]) continue;
      processed[current] = true;
      ordering.push(current);

      if (coreDist[current] === Infinity) continue;

      const nbrs = getNeighbors(current);
      update(current, nbrs, seeds);
    }
  }

  // Build reachability distances in ordering sequence
  const reachInOrder = ordering.map((idx) => reachability[idx]);
  const coreInOrder = ordering.map((idx) => coreDist[idx]);

  // Extract flat clusters using extractionEpsilon
  const extractEps = options.extractionEpsilon ?? (epsilon === Infinity ? autoExtractEpsilon(reachInOrder) : epsilon);
  const labels = new Array<number>(n).fill(-1);
  let clusterId = -1;

  for (let i = 0; i < ordering.length; i++) {
    const pointIdx = ordering[i];
    if (reachInOrder[i] > extractEps) {
      // Check if this is a steep down (start of cluster)
      if (coreDist[pointIdx] <= extractEps) {
        clusterId++;
        labels[pointIdx] = clusterId;
      }
      // else noise
    } else {
      labels[pointIdx] = clusterId >= 0 ? clusterId : -1;
    }
  }

  const nClusters = clusterId + 1;

  return {
    ordering,
    reachabilityDistances: reachInOrder,
    coreDistances: coreInOrder,
    labels,
    nClusters,
  };
}

/** Auto-select extraction epsilon as the mean of finite reachability values. */
function autoExtractEpsilon(reachInOrder: number[]): number {
  const finite = reachInOrder.filter((r) => Number.isFinite(r) && r > 0);
  if (finite.length === 0) return 1;
  return mean(finite);
}

// ── Spectral Clustering ─────────────────────────────────────────────────

/**
 * Result of spectral clustering.
 */
export interface SpectralClusteringResult {
  /** Cluster labels for each point. */
  labels: number[];
  /** Number of clusters. */
  nClusters: number;
  /** Eigenvector embedding used for clustering. */
  embedding: number[][];
}

/**
 * Spectral Clustering via normalized graph Laplacian eigenvectors.
 *
 * Builds a similarity graph from the data using an RBF kernel, computes
 * the normalized Laplacian, extracts the k smallest eigenvectors, and
 * applies k-means to the resulting embedding.
 *
 * @param data - Data matrix (n observations x p features)
 * @param k - Number of clusters
 * @param options - Configuration options
 * @param options.sigma - RBF kernel width (default: 1.0)
 * @param options.graphType - Graph type: "knn" or "epsilon" (default: "knn")
 * @param options.nNeighbors - Number of neighbors for knn graph (default: 10)
 * @param options.epsilon - Radius for epsilon graph (default: 1.0)
 * @param options.maxKmeansIter - Maximum k-means iterations (default: 100)
 * @param options.seed - Random seed for k-means initialization
 * @returns A {@link SpectralClusteringResult} with labels, cluster count, and embedding
 * @throws {Error} If fewer than k observations
 * @throws {Error} If k < 2
 *
 * @example
 * ```ts
 * const data = [[0,0],[1,0],[0,1],[10,10],[11,10],[10,11]];
 * const result = spectralClustering(data, 2, { sigma: 1.0 });
 * console.log(result.labels);
 * ```
 */
export function spectralClustering(
  data: number[][],
  k: number,
  options: {
    sigma?: number;
    graphType?: "knn" | "epsilon";
    nNeighbors?: number;
    epsilon?: number;
    maxKmeansIter?: number;
    seed?: number;
  } = {},
): SpectralClusteringResult {
  const n = data.length;
  if (k < 2) throw new Error("k must be at least 2");
  if (n < k) throw new Error("Need at least k observations");

  const sigma = options.sigma ?? 1.0;
  const graphType = options.graphType ?? "knn";
  const nNeighbors = options.nNeighbors ?? 10;
  const epsRadius = options.epsilon ?? 1.0;
  const maxKmeansIter = options.maxKmeansIter ?? 100;
  const rng = options.seed != null ? createRng(options.seed) : Math.random;

  // Step 1: Build affinity matrix W using RBF kernel
  const W = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  const twoSigmaSq = 2 * sigma * sigma;

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const d = euclidean(data[i], data[j]);
      const w = Math.exp(-(d * d) / twoSigmaSq);
      W[i][j] = w;
      W[j][i] = w;
    }
  }

  // Apply graph sparsification
  if (graphType === "knn") {
    const mask = Array.from({ length: n }, () => new Array<boolean>(n).fill(false));
    for (let i = 0; i < n; i++) {
      // Find k-nearest neighbors by sorting affinities
      const neighbors = Array.from({ length: n }, (_, j) => ({ j, w: W[i][j] }))
        .filter(({ j }) => j !== i)
        .sort((a, b) => b.w - a.w)
        .slice(0, nNeighbors);
      for (const { j } of neighbors) {
        mask[i][j] = true;
        mask[j][i] = true; // mutual
      }
    }
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (!mask[i][j]) W[i][j] = 0;
      }
    }
  } else {
    // epsilon-neighborhood
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const d = euclidean(data[i], data[j]);
        if (d > epsRadius) {
          W[i][j] = 0;
          W[j][i] = 0;
        }
      }
    }
  }

  // Step 2: Compute degree matrix D and normalized Laplacian L = I - D^(-1/2) W D^(-1/2)
  const degree = new Array<number>(n).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      degree[i] += W[i][j];
    }
  }

  const dInvSqrt = degree.map((d) => (d > 0 ? 1 / Math.sqrt(d) : 0));

  // L_norm = I - D^(-1/2) W D^(-1/2)
  // We want the k smallest eigenvectors of L_norm, which correspond to
  // the k largest eigenvectors of D^(-1/2) W D^(-1/2).
  // Compute M = D^(-1/2) W D^(-1/2) and find its k largest eigenvectors.
  const M = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      M[i][j] = dInvSqrt[i] * W[i][j] * dInvSqrt[j];
    }
  }

  // Step 3: Find k largest eigenvectors of M using power iteration with deflation
  const eigenvectors: number[][] = [];

  // Copy M for deflation
  const Mwork = M.map((row) => [...row]);

  for (let e = 0; e < k; e++) {
    // Power iteration to find largest eigenvector of Mwork
    let v = new Array<number>(n);
    for (let i = 0; i < n; i++) v[i] = rng() - 0.5;

    // Normalize
    let norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
    if (norm > 0) v = v.map((x) => x / norm);

    for (let iter = 0; iter < 300; iter++) {
      // Multiply Mwork * v
      const newV = new Array<number>(n).fill(0);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          newV[i] += Mwork[i][j] * v[j];
        }
      }

      norm = Math.sqrt(newV.reduce((s, x) => s + x * x, 0));
      if (norm < 1e-15) break;
      const scaled = newV.map((x) => x / norm);

      // Check convergence
      let diff = 0;
      for (let i = 0; i < n; i++) diff += (scaled[i] - v[i]) ** 2;
      v = scaled;
      if (diff < 1e-12) break;
    }

    eigenvectors.push(v);

    // Deflate: Mwork = Mwork - eigenvalue * v * v^T
    // eigenvalue ≈ v^T Mwork v
    const Mv = new Array<number>(n).fill(0);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        Mv[i] += Mwork[i][j] * v[j];
      }
    }
    let eigenvalue = 0;
    for (let i = 0; i < n; i++) eigenvalue += v[i] * Mv[i];

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        Mwork[i][j] -= eigenvalue * v[i] * v[j];
      }
    }
  }

  // Step 4: Build embedding matrix (n x k) and row-normalize
  const embedding = Array.from({ length: n }, (_, i) =>
    eigenvectors.map((ev) => ev[i]),
  );

  // Row-normalize the embedding
  for (let i = 0; i < n; i++) {
    const rowNorm = Math.sqrt(embedding[i].reduce((s, x) => s + x * x, 0));
    if (rowNorm > 0) {
      for (let d = 0; d < k; d++) {
        embedding[i][d] /= rowNorm;
      }
    }
  }

  // Step 5: K-means on the embedding
  const labels = kmeansHelper(embedding, k, maxKmeansIter, rng);

  return {
    labels,
    nClusters: k,
    embedding,
  };
}

/** Simple k-means clustering helper for internal use. */
function kmeansHelper(
  data: number[][],
  k: number,
  maxIter: number,
  rng: () => number,
): number[] {
  const n = data.length;
  const dim = data[0].length;

  // k-means++ initialization
  const centroids: number[][] = [];
  const firstIdx = Math.floor(rng() * n);
  centroids.push([...data[firstIdx]]);

  for (let c = 1; c < k; c++) {
    const dists = data.map((point) => {
      let minD = Infinity;
      for (const cen of centroids) {
        const d = euclidean(point, cen);
        if (d < minD) minD = d;
      }
      return minD * minD;
    });
    const total = dists.reduce((s, d) => s + d, 0);
    if (total === 0) {
      centroids.push([...data[Math.floor(rng() * n)]]);
      continue;
    }
    let r = rng() * total;
    for (let i = 0; i < n; i++) {
      r -= dists[i];
      if (r <= 0) {
        centroids.push([...data[i]]);
        break;
      }
      if (i === n - 1) centroids.push([...data[i]]);
    }
  }

  const labels = new Array<number>(n).fill(0);

  for (let iter = 0; iter < maxIter; iter++) {
    // Assign
    let changed = false;
    for (let i = 0; i < n; i++) {
      let bestC = 0;
      let bestD = Infinity;
      for (let c = 0; c < k; c++) {
        const d = euclidean(data[i], centroids[c]);
        if (d < bestD) {
          bestD = d;
          bestC = c;
        }
      }
      if (labels[i] !== bestC) {
        labels[i] = bestC;
        changed = true;
      }
    }

    if (!changed) break;

    // Update centroids
    for (let c = 0; c < k; c++) {
      const members = data.filter((_, i) => labels[i] === c);
      if (members.length === 0) continue;
      for (let d = 0; d < dim; d++) {
        centroids[c][d] = mean(members.map((m) => m[d]));
      }
    }
  }

  return labels;
}

// ── UMAP ────────────────────────────────────────────────────────────────

/**
 * Result of UMAP dimensionality reduction.
 */
export interface UMAPResult {
  /** Embedded low-dimensional coordinates. */
  embedding: number[][];
  /** Number of output dimensions. */
  nComponents: number;
  /** Number of optimization iterations performed. */
  iterations: number;
}

/**
 * UMAP (Uniform Manifold Approximation and Projection) for dimensionality reduction.
 *
 * A simplified implementation that builds a fuzzy simplicial set from
 * k-nearest neighbor distances, then optimizes a low-dimensional embedding
 * using stochastic gradient descent with attractive and repulsive forces.
 *
 * @param data - Data matrix (n observations x p features)
 * @param options - Configuration options
 * @param options.nNeighbors - Number of nearest neighbors (default: 15)
 * @param options.minDist - Minimum distance in embedding (default: 0.1)
 * @param options.nComponents - Number of output dimensions (default: 2)
 * @param options.iterations - Number of optimization iterations (default: 200)
 * @param options.seed - Random seed for reproducibility
 * @param options.learningRate - SGD learning rate (default: 1.0)
 * @returns A {@link UMAPResult} with the embedding, component count, and iteration count
 * @throws {Error} If fewer than nNeighbors + 1 observations
 * @throws {Error} If any data value is NaN or Infinity
 *
 * @example
 * ```ts
 * const data = [[0,0],[1,0],[0,1],[10,10],[11,10],[10,11]];
 * const result = umap(data, { nNeighbors: 3, nComponents: 2, seed: 42 });
 * console.log(result.embedding); // 2D coordinates
 * ```
 */
export function umap(
  data: number[][],
  options: {
    nNeighbors?: number;
    minDist?: number;
    nComponents?: number;
    iterations?: number;
    seed?: number;
    learningRate?: number;
  } = {},
): UMAPResult {
  const n = data.length;
  const nNeighbors = options.nNeighbors ?? 15;
  const minDist = options.minDist ?? 0.1;
  const nComponents = options.nComponents ?? 2;
  const iterations = options.iterations ?? 200;
  const learningRate = options.learningRate ?? 1.0;
  const rng = options.seed != null ? createRng(options.seed) : Math.random;

  if (n < nNeighbors + 1) {
    throw new Error(`Need at least ${nNeighbors + 1} observations for nNeighbors=${nNeighbors}`);
  }

  // NaN/Infinity guard
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < data[i].length; j++) {
      if (!Number.isFinite(data[i][j])) {
        throw new Error("Data must not contain NaN or Infinity");
      }
    }
  }

  // Step 1: Compute pairwise distances and k-nearest neighbors
  const distMatrix = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => euclidean(data[i], data[j])),
  );

  // kNN for each point (sorted by distance, excluding self)
  const knnIndices: number[][] = [];
  const knnDists: number[][] = [];

  for (let i = 0; i < n; i++) {
    const neighbors = Array.from({ length: n }, (_, j) => ({ j, d: distMatrix[i][j] }))
      .filter(({ j }) => j !== i)
      .sort((a, b) => a.d - b.d)
      .slice(0, nNeighbors);
    knnIndices.push(neighbors.map(({ j }) => j));
    knnDists.push(neighbors.map(({ d }) => d));
  }

  // Step 2: Compute fuzzy simplicial set (membership strengths)
  // Smooth kNN distances: find sigma_i such that sum of exp(-(d - rho)/sigma) = log2(nNeighbors)
  const sigmas = new Array<number>(n);
  const rhos = new Array<number>(n);
  const target = Math.log2(nNeighbors);

  for (let i = 0; i < n; i++) {
    rhos[i] = knnDists[i][0]; // distance to nearest neighbor

    // Binary search for sigma
    let lo = 1e-10;
    let hi = 1000;
    let mid = 1;

    for (let iter = 0; iter < 64; iter++) {
      mid = (lo + hi) / 2;
      let sum = 0;
      for (let j = 0; j < knnDists[i].length; j++) {
        const dij = Math.max(knnDists[i][j] - rhos[i], 0);
        sum += Math.exp(-dij / mid);
      }
      if (sum > target) {
        hi = mid;
      } else {
        lo = mid;
      }
      if (Math.abs(sum - target) < 1e-5) break;
    }
    sigmas[i] = mid;
  }

  // Build fuzzy graph: weight(i,j) = exp(-(d(i,j) - rho_i) / sigma_i)
  // Symmetrize: w_sym = w_ij + w_ji - w_ij * w_ji
  const graph = new Map<string, number>();

  for (let i = 0; i < n; i++) {
    for (let k = 0; k < knnIndices[i].length; k++) {
      const j = knnIndices[i][k];
      const dij = Math.max(knnDists[i][k] - rhos[i], 0);
      const w = Math.exp(-dij / sigmas[i]);
      const key = `${i},${j}`;
      graph.set(key, w);
    }
  }

  // Symmetrize
  const symGraph: Array<{ i: number; j: number; w: number }> = [];
  const visited = new Set<string>();

  for (const [key, wij] of graph) {
    const [iStr, jStr] = key.split(",");
    const i = parseInt(iStr);
    const j = parseInt(jStr);
    const pairKey = i < j ? `${i},${j}` : `${j},${i}`;
    if (visited.has(pairKey)) continue;
    visited.add(pairKey);

    const wji = graph.get(`${j},${i}`) ?? 0;
    const wSym = wij + wji - wij * wji;
    if (wSym > 0) {
      symGraph.push({ i, j, w: wSym });
    }
  }

  // Step 3: Initialize embedding randomly
  const embedding = Array.from({ length: n }, () => {
    const point = new Array<number>(nComponents);
    for (let d = 0; d < nComponents; d++) {
      point[d] = (rng() - 0.5) * 20;
    }
    return point;
  });

  // Precompute a and b parameters for the smooth approximation:
  // 1/(1 + a*d^(2b)) ≈ 1 if d <= minDist, decaying otherwise
  // Use a simple curve fit: a ≈ 1, b adjusted for minDist
  const { a: paramA, b: paramB } = fitAB(minDist);

  // Step 4: Optimize with SGD
  const nEdges = symGraph.length;
  const nNegSamples = 5;

  for (let iter = 0; iter < iterations; iter++) {
    const alpha = learningRate * (1 - iter / iterations);

    for (let e = 0; e < nEdges; e++) {
      const { i, j, w } = symGraph[e];

      // Attractive force
      let distSq = 0;
      for (let d = 0; d < nComponents; d++) {
        distSq += (embedding[i][d] - embedding[j][d]) ** 2;
      }
      distSq = Math.max(distSq, 1e-10);

      const gradCoeff = (-2 * paramA * paramB * Math.pow(distSq, paramB - 1)) /
        (1 + paramA * Math.pow(distSq, paramB));

      for (let d = 0; d < nComponents; d++) {
        const grad = gradCoeff * (embedding[i][d] - embedding[j][d]);
        const clampedGrad = Math.max(-4, Math.min(4, grad));
        embedding[i][d] -= alpha * w * clampedGrad;
        embedding[j][d] += alpha * w * clampedGrad;
      }

      // Repulsive forces (negative sampling)
      for (let neg = 0; neg < nNegSamples; neg++) {
        const k = Math.floor(rng() * n);
        if (k === i) continue;

        let negDistSq = 0;
        for (let d = 0; d < nComponents; d++) {
          negDistSq += (embedding[i][d] - embedding[k][d]) ** 2;
        }
        negDistSq = Math.max(negDistSq, 1e-10);

        const repGradCoeff = (2 * paramB) /
          ((0.001 + negDistSq) * (1 + paramA * Math.pow(negDistSq, paramB)));

        for (let d = 0; d < nComponents; d++) {
          const grad = repGradCoeff * (embedding[i][d] - embedding[k][d]);
          const clampedGrad = Math.max(-4, Math.min(4, grad));
          embedding[i][d] += alpha * clampedGrad;
        }
      }
    }
  }

  return {
    embedding,
    nComponents,
    iterations,
  };
}

/** Fit the a, b parameters for UMAP's smooth distance approximation. */
function fitAB(minDist: number): { a: number; b: number } {
  // Approximate: for the curve 1/(1 + a * d^(2b)),
  // we want it to be ~1 at d=minDist and decay after.
  // Simple analytical approximation:
  if (minDist <= 0) return { a: 1, b: 1 };

  // Use a grid search over reasonable parameter ranges
  let bestA = 1;
  let bestB = 1;
  let bestError = Infinity;

  // Sample distances for fitting
  const testDists = Array.from({ length: 30 }, (_, i) => (i + 1) * 0.1);

  for (let aCandidate = 0.1; aCandidate <= 10; aCandidate += 0.2) {
    for (let bCandidate = 0.2; bCandidate <= 3; bCandidate += 0.1) {
      let error = 0;
      for (const d of testDists) {
        const predicted = 1 / (1 + aCandidate * Math.pow(d, 2 * bCandidate));
        const target = d <= minDist ? 1 : Math.exp(-(d - minDist));
        error += (predicted - target) ** 2;
      }
      if (error < bestError) {
        bestError = error;
        bestA = aCandidate;
        bestB = bCandidate;
      }
    }
  }

  return { a: bestA, b: bestB };
}
