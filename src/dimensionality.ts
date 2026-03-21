import { Dataset } from "./types";
import { mean, variance } from "./utils/descriptive";
import { euclidean } from "./distance";
import { createRng } from "./utils/linalg";

/**
 * t-SNE (t-distributed Stochastic Neighbor Embedding) result.
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
 * t-SNE for nonlinear dimensionality reduction.
 *
 * Reduces high-dimensional data to 2D while preserving local structure.
 * Uses the exact O(n^2) algorithm suitable for small-to-medium datasets.
 * Includes early stopping if KL divergence stops improving.
 *
 * @param data - Data matrix (n observations x p features)
 * @param options - Configuration
 * @returns TSNEResult containing the 2D embedding, final KL divergence, and iteration count
 * @throws {Error} If fewer than 4 observations
 * @throws {Error} If any data value is NaN or Infinity
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
