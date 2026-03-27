/**
 * Spatial statistics.
 *
 * - **Moran's I** — global spatial autocorrelation.
 * - **Geary's C** — local spatial autocorrelation contrast.
 * - **Empirical variogram** — semivariance as a function of distance.
 * - **Variogram models** — spherical, exponential, gaussian, linear.
 * - **Ordinary kriging** — spatial interpolation with uncertainty.
 * - **Distance matrix** — Euclidean distance between point pairs.
 */

import { solveLinearSystem } from "./utils/linalg";
import { mean } from "./utils/descriptive";

// ── Types ─────────────────────────────────────────────────────────────────

/**
 * A 2-D or n-D spatial coordinate represented as an array of numbers.
 * For example, [x, y] for 2D or [x, y, z] for 3D.
 */
export type SpatialPoint = number[];

/**
 * Spatial weights matrix (row-standardised or binary).
 * W[i][j] represents the spatial relationship between locations i and j.
 */
export type SpatialWeights = number[][];

/**
 * Result of a Moran's I spatial autocorrelation test.
 */
export interface MoranResult {
  /** Moran's I statistic (range roughly −1 to +1). */
  I: number;
  /** Expected I under H₀ (no spatial autocorrelation): −1/(n−1). */
  expectedI: number;
  /** Variance of I under randomisation assumption. */
  varianceI: number;
  /** z-score: (I − E[I]) / √Var(I). */
  zScore: number;
  /** Two-sided p-value from normal approximation. */
  pValue: number;
}

/**
 * Result of a Geary's C spatial autocorrelation test.
 */
export interface GearyResult {
  /** Geary's C statistic (0 = perfect positive, 1 = no autocorrelation, >1 = negative). */
  C: number;
  /** Expected C under H₀: 1. */
  expectedC: number;
  /** z-score. */
  zScore: number;
  /** Two-sided p-value. */
  pValue: number;
}

/**
 * A single bin of the empirical variogram.
 */
export interface VariogramBin {
  /** Average distance (lag) for this bin. */
  distance: number;
  /** Semivariance γ(h). */
  semivariance: number;
  /** Number of point pairs in this bin. */
  count: number;
}

/**
 * A fitted variogram model with parameters and evaluation function.
 */
export interface VariogramModel {
  /** Model type. */
  type: "spherical" | "exponential" | "gaussian" | "linear";
  /** Nugget (discontinuity at origin). */
  nugget: number;
  /** Partial sill (sill − nugget). */
  sill: number;
  /** Range parameter. */
  range: number;
  /** Evaluate the model at distance h. */
  evaluate: (h: number) => number;
}

/**
 * Result of an ordinary kriging interpolation.
 */
export interface KrigingResult {
  /** Predicted values at query points. */
  predictions: number[];
  /** Kriging variance (uncertainty) at each query point. */
  variances: number[];
}

// ── Distance Matrix ───────────────────────────────────────────────────────

/**
 * Compute pairwise Euclidean distance matrix.
 *
 * D[i][j] = ||points[i] - points[j]||_2. The matrix is symmetric with
 * zeros on the diagonal.
 *
 * @param points - Array of spatial coordinates (n points, each d-dimensional)
 * @returns Symmetric n x n distance matrix
 *
 * @example
 * ```ts
 * const pts = [[0, 0], [3, 4]];
 * spatialDistanceMatrix(pts); // [[0, 5], [5, 0]]
 * ```
 */
export function spatialDistanceMatrix(points: SpatialPoint[]): number[][] {
  const n = points.length;
  const D: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      let sum = 0;
      for (let d = 0; d < points[i].length; d++) {
        sum += (points[i][d] - points[j][d]) ** 2;
      }
      const dist = Math.sqrt(sum);
      D[i][j] = dist;
      D[j][i] = dist;
    }
  }
  return D;
}

/**
 * Build binary spatial weights from a distance threshold.
 *
 * W[i][j] = 1 if dist(i,j) <= threshold and i != j, 0 otherwise.
 *
 * @param points - Array of spatial coordinates
 * @param threshold - Maximum distance for two points to be considered neighbors
 * @returns Binary spatial weights matrix (n x n)
 */
export function distanceBandWeights(
  points: SpatialPoint[],
  threshold: number,
): SpatialWeights {
  const n = points.length;
  const D = spatialDistanceMatrix(points);
  const W: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i !== j && D[i][j] <= threshold) W[i][j] = 1;
    }
  }
  return W;
}

/**
 * Build k-nearest-neighbour spatial weights.
 *
 * W[i][j] = 1 if j is one of the k nearest neighbors of i, 0 otherwise.
 * Note: the resulting matrix may be asymmetric.
 *
 * @param points - Array of spatial coordinates
 * @param k - Number of nearest neighbors
 * @returns Spatial weights matrix (n x n)
 * @throws Error if k >= number of points
 */
export function knnWeights(points: SpatialPoint[], k: number): SpatialWeights {
  const n = points.length;
  if (k >= n) throw new Error(`Invalid parameter 'k': expected less than number of points (${n}), received ${k}`);
  const D = spatialDistanceMatrix(points);
  const W: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));

  for (let i = 0; i < n; i++) {
    const neighbours = Array.from({ length: n }, (_, j) => j)
      .filter((j) => j !== i)
      .sort((a, b) => D[i][a] - D[i][b])
      .slice(0, k);
    for (const j of neighbours) W[i][j] = 1;
  }
  return W;
}

// ── Moran's I ─────────────────────────────────────────────────────────────

/**
 * Global Moran's I for spatial autocorrelation.
 *
 * I = (n / S0) * (sum_ij w_ij (x_i - x_bar)(x_j - x_bar)) / (sum_i (x_i - x_bar)^2)
 *
 * where S0 = sum_ij w_ij. Values near +1 indicate positive spatial autocorrelation
 * (clustering), values near -1 indicate negative (dispersion), and values near
 * E[I] = -1/(n-1) indicate random spatial pattern.
 *
 * @param values - Observed values at each location (length n)
 * @param W - Spatial weights matrix (n x n)
 * @returns MoranResult with I statistic, expected value, variance, z-score, and p-value
 * @throws Error if fewer than 3 observations
 * @throws Error if weights matrix size does not match values length
 *
 * @example
 * ```ts
 * const values = [1, 2, 3, 4, 5];
 * const W = distanceBandWeights([[0,0],[1,0],[2,0],[3,0],[4,0]], 1.5);
 * const result = moranI(values, W);
 * // result.I — Moran's I statistic
 * // result.pValue — significance of spatial autocorrelation
 * ```
 */
export function moranI(values: number[], W: SpatialWeights): MoranResult {
  const n = values.length;
  if (n < 3) throw new Error(`Invalid parameter 'values': expected at least 3 observations, received ${n}`);
  if (W.length !== n) throw new Error(`Invalid parameter 'W': expected length ${n} to match values, received ${W.length}`);

  const xBar = mean(values);
  const dev = values.map((v) => v - xBar);

  let S0 = 0;
  let numSum = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      S0 += W[i][j];
      numSum += W[i][j] * dev[i] * dev[j];
    }
  }

  let denomSum = 0;
  for (let i = 0; i < n; i++) denomSum += dev[i] * dev[i];

  if (denomSum === 0 || S0 === 0) {
    return { I: 0, expectedI: -1 / (n - 1), varianceI: 0, zScore: 0, pValue: 1 };
  }

  const I = (n / S0) * (numSum / denomSum);
  const expectedI = -1 / (n - 1);

  // Variance under randomisation assumption
  let S1 = 0;
  let S2 = 0;
  for (let i = 0; i < n; i++) {
    let rowSum = 0;
    let colSum = 0;
    for (let j = 0; j < n; j++) {
      S1 += (W[i][j] + W[j][i]) ** 2;
      rowSum += W[i][j];
      colSum += W[j][i];
    }
    S2 += (rowSum + colSum) ** 2;
  }
  S1 /= 2;

  const n2 = n * n;
  const S0sq = S0 * S0;
  const varianceI =
    (n2 * S1 - n * S2 + 3 * S0sq) / (S0sq * (n2 - 1)) - expectedI * expectedI;

  const zScore = varianceI > 0 ? (I - expectedI) / Math.sqrt(varianceI) : 0;
  const pValue = 2 * (1 - normalCdfApprox(Math.abs(zScore)));

  return { I, expectedI, varianceI, zScore, pValue };
}

// ── Geary's C ─────────────────────────────────────────────────────────────

/**
 * Geary's C for spatial autocorrelation.
 *
 * C = ((n-1) / (2 * S0)) * (sum_ij w_ij (x_i - x_j)^2) / (sum_i (x_i - x_bar)^2)
 *
 * Values near 0 indicate strong positive autocorrelation, near 1 indicates
 * no autocorrelation, and values greater than 1 indicate negative autocorrelation.
 *
 * @param values - Observed values at each location (length n)
 * @param W - Spatial weights matrix (n x n)
 * @returns GearyResult with C statistic, expected value, z-score, and p-value
 * @throws Error if fewer than 3 observations
 */
export function gearyC(values: number[], W: SpatialWeights): GearyResult {
  const n = values.length;
  if (n < 3) throw new Error(`Invalid parameter 'values': expected at least 3 observations, received ${n}`);

  const xBar = mean(values);
  let S0 = 0;
  let numSum = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      S0 += W[i][j];
      numSum += W[i][j] * (values[i] - values[j]) ** 2;
    }
  }

  let denomSum = 0;
  for (let i = 0; i < n; i++) denomSum += (values[i] - xBar) ** 2;

  if (denomSum === 0 || S0 === 0) {
    return { C: 1, expectedC: 1, zScore: 0, pValue: 1 };
  }

  const C = ((n - 1) / (2 * S0)) * (numSum / denomSum);

  // Approximate variance (simplified)
  const varianceC = (2 * S0 + (S0 * S0) / n) / (3 * S0 * S0 / (n * (n - 1)));
  const se = Math.sqrt(Math.max(0, varianceC - 1)) / Math.sqrt(n);
  const zScore = se > 0 ? (C - 1) / se : 0;
  const pValue = 2 * (1 - normalCdfApprox(Math.abs(zScore)));

  return { C, expectedC: 1, zScore, pValue };
}

// ── Variograms ────────────────────────────────────────────────────────────

/**
 * Compute the empirical (semi)variogram.
 *
 * gamma(h) = (1 / (2 * |N(h)|)) * sum_{(i,j) in N(h)} (z(s_i) - z(s_j))^2
 *
 * where N(h) is the set of point pairs at distance approximately h.
 *
 * @param points - Spatial coordinates (n x d)
 * @param values - Observed values at each point (length n)
 * @param nBins - Number of distance bins (default 15)
 * @param maxDist - Maximum distance to consider (default: half the max pairwise distance)
 * @returns Array of VariogramBin objects with distance, semivariance, and count
 * @throws Error if points and values have different lengths
 * @throws Error if fewer than 3 observations
 *
 * @example
 * ```ts
 * const points = [[0,0], [1,0], [2,0], [3,0], [0,1], [1,1]];
 * const values = [1, 2, 4, 8, 2, 3];
 * const bins = empiricalVariogram(points, values, 10);
 * // bins[i].distance — average lag distance for bin i
 * // bins[i].semivariance — estimated semivariance at that lag
 * ```
 */
export function empiricalVariogram(
  points: SpatialPoint[],
  values: number[],
  nBins = 15,
  maxDist?: number,
): VariogramBin[] {
  const n = points.length;
  if (n !== values.length) throw new Error(`Invalid parameter 'values': expected length ${n} to match points, received ${values.length}`);
  if (n < 3) throw new Error(`Invalid parameter 'points': expected at least 3 observations, received ${n}`);

  const D = spatialDistanceMatrix(points);

  // Find max distance
  let maxD = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (D[i][j] > maxD) maxD = D[i][j];
    }
  }
  const cutoff = maxDist ?? maxD / 2;
  const binWidth = cutoff / nBins;

  const bins: VariogramBin[] = [];
  for (let b = 0; b < nBins; b++) {
    const lo = b * binWidth;
    const hi = (b + 1) * binWidth;
    let sumSV = 0;
    let sumDist = 0;
    let count = 0;

    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (D[i][j] >= lo && D[i][j] < hi) {
          sumSV += (values[i] - values[j]) ** 2;
          sumDist += D[i][j];
          count++;
        }
      }
    }

    if (count > 0) {
      bins.push({
        distance: sumDist / count,
        semivariance: sumSV / (2 * count),
        count,
      });
    }
  }

  return bins;
}

/**
 * Fit a variogram model to empirical variogram bins.
 *
 * Uses weighted least squares (WLS) on the empirical bins to estimate
 * nugget, sill, and range parameters for the chosen model type.
 *
 * @param bins - Empirical variogram bins from {@link empiricalVariogram}
 * @param type - Model type: "spherical", "exponential", "gaussian", or "linear" (default "spherical")
 * @returns VariogramModel with fitted parameters and an evaluate function
 * @throws Error if fewer than 2 variogram bins
 */
export function fitVariogramModel(
  bins: VariogramBin[],
  type: VariogramModel["type"] = "spherical",
): VariogramModel {
  if (bins.length < 2) throw new Error(`Invalid parameter 'bins': expected at least 2 variogram bins, received ${bins.length}`);

  // Initial estimates
  const maxSV = Math.max(...bins.map((b) => b.semivariance));
  const maxDist = Math.max(...bins.map((b) => b.distance));
  let nugget = Math.max(0, bins[0].semivariance * 0.1);
  let sill = maxSV - nugget;
  let range = maxDist * 0.6;

  // Weighted least squares optimisation (simple grid search + refinement)
  const modelFn = getModelFn(type);

  for (let iter = 0; iter < 3; iter++) {
    let bestCost = Infinity;
    let bestParams = { nugget, sill, range };

    const nSteps = 20;
    for (let ni = 0; ni <= nSteps; ni++) {
      for (let si = 1; si <= nSteps; si++) {
        for (let ri = 1; ri <= nSteps; ri++) {
          const nug = nugget + (ni / nSteps - 0.5) * nugget * (iter === 0 ? 2 : 0.5);
          const s = sill * (si / nSteps) * 2;
          const r = range * (ri / nSteps) * 2;
          if (nug < 0 || s <= 0 || r <= 0) continue;

          let cost = 0;
          for (const bin of bins) {
            const pred = modelFn(bin.distance, nug, s, r);
            cost += bin.count * (bin.semivariance - pred) ** 2;
          }

          if (cost < bestCost) {
            bestCost = cost;
            bestParams = { nugget: nug, sill: s, range: r };
          }
        }
      }
    }

    nugget = bestParams.nugget;
    sill = bestParams.sill;
    range = bestParams.range;
  }

  return {
    type,
    nugget,
    sill,
    range,
    evaluate: (h: number) => modelFn(h, nugget, sill, range),
  };
}

// ── Kriging ───────────────────────────────────────────────────────────────

/**
 * Ordinary kriging -- spatial prediction with uncertainty.
 *
 * Solves the kriging system for each query point to find optimal weights
 * that minimize prediction variance subject to unbiasedness:
 *   [C  1] [lambda]   [c ]
 *   [1' 0] [mu    ] = [1 ]
 *
 * where C is the covariance matrix of observed points, c is the covariance
 * vector between observed points and the query point, lambda are kriging
 * weights, and mu is the Lagrange multiplier.
 *
 * @param points - Known observation locations (n x d)
 * @param values - Known values at observation locations (length n)
 * @param queryPoints - Locations to predict at (m x d)
 * @param model - Fitted variogram model from {@link fitVariogramModel}
 * @returns KrigingResult with predictions and kriging variances at each query point
 * @throws Error if points and values have different lengths
 * @throws Error if fewer than 2 observations
 *
 * @example
 * ```ts
 * const points = [[0,0], [1,0], [0,1], [1,1]];
 * const values = [1, 2, 3, 4];
 * const bins = empiricalVariogram(points, values);
 * const model = fitVariogramModel(bins);
 * const result = ordinaryKriging(points, values, [[0.5, 0.5]], model);
 * // result.predictions[0] — interpolated value at (0.5, 0.5)
 * // result.variances[0] — kriging variance (uncertainty)
 * ```
 */
export function ordinaryKriging(
  points: SpatialPoint[],
  values: number[],
  queryPoints: SpatialPoint[],
  model: VariogramModel,
): KrigingResult {
  const n = points.length;
  if (n !== values.length) throw new Error(`Invalid parameter 'values': expected length ${n} to match points, received ${values.length}`);
  if (n < 2) throw new Error(`Invalid parameter 'points': expected at least 2 observations, received ${n}`);

  const totalSill = model.nugget + model.sill;

  // Build covariance matrix C (n+1 × n+1) with Lagrange multiplier row/col
  const D = spatialDistanceMatrix(points);
  const size = n + 1;
  const C: number[][] = Array.from({ length: size }, () => new Array(size).fill(0));

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      // Covariance = sill - variogram(h)
      C[i][j] = totalSill - model.evaluate(D[i][j]);
    }
    C[i][n] = 1;
    C[n][i] = 1;
  }
  C[n][n] = 0;

  const predictions: number[] = [];
  const variances: number[] = [];

  for (const qp of queryPoints) {
    // Build covariance vector c
    const rhs = new Array(size);
    for (let i = 0; i < n; i++) {
      let dist = 0;
      for (let d = 0; d < points[i].length; d++) {
        dist += (points[i][d] - qp[d]) ** 2;
      }
      dist = Math.sqrt(dist);
      rhs[i] = totalSill - model.evaluate(dist);
    }
    rhs[n] = 1; // Lagrange constraint

    // Solve the kriging system
    const lambda = solveLinearSystem(C, rhs);

    // Prediction: z* = Σ λᵢ zᵢ
    let pred = 0;
    for (let i = 0; i < n; i++) {
      pred += lambda[i] * values[i];
    }
    predictions.push(pred);

    // Kriging variance: σ² = sill - Σ λᵢ cᵢ - μ
    let kVar = totalSill;
    for (let i = 0; i < n; i++) {
      kVar -= lambda[i] * rhs[i];
    }
    kVar -= lambda[n]; // Lagrange multiplier
    variances.push(Math.max(0, kVar));
  }

  return { predictions, variances };
}

// ── Internal Helpers ──────────────────────────────────────────────────────

function getModelFn(
  type: VariogramModel["type"],
): (h: number, nugget: number, sill: number, range: number) => number {
  switch (type) {
    case "spherical":
      return (h, nugget, sill, range) => {
        if (h === 0) return 0;
        if (h >= range) return nugget + sill;
        const r = h / range;
        return nugget + sill * (1.5 * r - 0.5 * r * r * r);
      };
    case "exponential":
      return (h, nugget, sill, range) => {
        if (h === 0) return 0;
        return nugget + sill * (1 - Math.exp(-3 * h / range));
      };
    case "gaussian":
      return (h, nugget, sill, range) => {
        if (h === 0) return 0;
        return nugget + sill * (1 - Math.exp(-3 * (h / range) ** 2));
      };
    case "linear":
      return (h, nugget, sill, range) => {
        if (h === 0) return 0;
        return nugget + sill * Math.min(h / range, 1);
      };
  }
}

function normalCdfApprox(x: number): number {
  if (x < -8) return 0;
  if (x > 8) return 1;
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const d = 0.3989422804014327; // 1/sqrt(2*pi)
  const p =
    d * Math.exp(-0.5 * x * x) *
    (t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))));
  return x > 0 ? 1 - p : p;
}
