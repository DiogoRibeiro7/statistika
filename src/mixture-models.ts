import { Dataset } from "./types";
import { mean, variance } from "./utils/descriptive";
import { createRng } from "./utils/linalg";

/**
 * Result of fitting a 1D Gaussian Mixture Model via the EM algorithm.
 *
 * Contains the estimated mixture parameters (weights, means, variances),
 * soft and hard cluster assignments, model selection criteria (AIC/BIC),
 * and convergence information.
 */
export interface GMMResult {
  /** Number of components. */
  k: number;
  /** Mixture weights (sum to 1). */
  weights: number[];
  /** Component means. */
  means: number[];
  /** Component variances. */
  variances: number[];
  /** Log-likelihood of the fitted model. */
  logLikelihood: number;
  /** BIC (Bayesian Information Criterion). */
  bic: number;
  /** AIC (Akaike Information Criterion). */
  aic: number;
  /** Number of iterations to convergence. */
  iterations: number;
  /** Soft cluster assignments (n x k matrix of responsibilities). */
  responsibilities: number[][];
  /** Hard cluster labels for each observation. */
  labels: number[];
}

/**
 * Fit a 1D Gaussian Mixture Model using the EM algorithm.
 *
 * @param data - Input dataset (must not contain NaN or Infinity values)
 * @param k - Number of mixture components
 * @param options - Configuration
 * @returns The fitted GMM including weights, means, variances, responsibilities, and model selection criteria
 * @throws {Error} If there are fewer observations than components
 * @throws {Error} If k is less than 1
 * @throws {Error} If data contains NaN or Infinity values
 *
 * @example
 * ```ts
 * // Fit a 2-component mixture to bimodal data
 * const data = [1.0, 1.2, 1.1, 5.0, 5.1, 4.9];
 * const result = gaussianMixture(data, 2, { seed: 42 });
 * // result.means  — approximately [1.1, 5.0]
 * // result.labels — cluster assignment for each observation
 * ```
 */
export function gaussianMixture(
  data: Dataset,
  k: number,
  options: {
    maxIterations?: number;
    tol?: number;
    seed?: number;
  } = {},
): GMMResult {
  const n = data.length;
  if (n < k) throw new Error("Need at least k observations");
  if (k < 1) throw new Error("k must be at least 1");

  // NaN / Infinity guard
  for (let i = 0; i < n; i++) {
    if (!Number.isFinite(data[i])) {
      throw new Error(`Data contains non-finite value at index ${i}`);
    }
  }

  const maxIterations = options.maxIterations ?? 100;
  const tol = options.tol ?? 1e-6;
  const rng = options.seed != null ? createRng(options.seed) : Math.random;

  // Initialize with k-means++ style
  const sorted = [...data].sort((a, b) => a - b);
  const weights = new Array<number>(k).fill(1 / k);
  const means = new Array<number>(k);
  const variances = new Array<number>(k);
  const overallVar = variance(data);

  // Spread initial means across the data range
  for (let j = 0; j < k; j++) {
    means[j] = sorted[Math.floor(((j + 0.5) / k) * n)];
    variances[j] = overallVar;
  }

  const responsibilities = Array.from({ length: n }, () => new Array<number>(k).fill(0));
  let logLik = -Infinity;

  let iter = 0;
  for (; iter < maxIterations; iter++) {
    // E-step: compute responsibilities
    for (let i = 0; i < n; i++) {
      let totalDensity = 0;
      for (let j = 0; j < k; j++) {
        const d = gaussianPdf(data[i], means[j], variances[j]);
        responsibilities[i][j] = weights[j] * d;
        totalDensity += responsibilities[i][j];
      }
      if (totalDensity > 0) {
        for (let j = 0; j < k; j++) {
          responsibilities[i][j] /= totalDensity;
        }
      }
    }

    // M-step: update parameters
    for (let j = 0; j < k; j++) {
      let nk = 0;
      let sumX = 0;
      for (let i = 0; i < n; i++) {
        nk += responsibilities[i][j];
        sumX += responsibilities[i][j] * data[i];
      }

      if (nk < 1e-10) continue;

      means[j] = sumX / nk;
      let sumVar = 0;
      for (let i = 0; i < n; i++) {
        sumVar += responsibilities[i][j] * (data[i] - means[j]) ** 2;
      }
      variances[j] = Math.max(sumVar / nk, 1e-10); // floor to avoid degenerate solutions
      weights[j] = nk / n;
    }

    // Compute log-likelihood
    let newLogLik = 0;
    for (let i = 0; i < n; i++) {
      let density = 0;
      for (let j = 0; j < k; j++) {
        density += weights[j] * gaussianPdf(data[i], means[j], variances[j]);
      }
      newLogLik += Math.log(Math.max(density, 1e-300));
    }

    // Convergence check: if log-likelihood is NaN, break early
    if (Number.isNaN(newLogLik)) {
      console.warn(
        `gaussianMixture: log-likelihood became NaN at iteration ${iter + 1}; stopping early`,
      );
      break;
    }

    if (Math.abs(newLogLik - logLik) < tol) {
      logLik = newLogLik;
      iter++;
      break;
    }
    logLik = newLogLik;
  }

  // Assign hard labels
  const labels = responsibilities.map((row) => {
    let maxIdx = 0;
    for (let j = 1; j < k; j++) {
      if (row[j] > row[maxIdx]) maxIdx = j;
    }
    return maxIdx;
  });

  // Model selection criteria
  const nParams = 3 * k - 1; // k means + k variances + (k-1) weights
  const bic = -2 * logLik + nParams * Math.log(n);
  const aic = -2 * logLik + 2 * nParams;

  return {
    k,
    weights,
    means,
    variances,
    logLikelihood: logLik,
    bic,
    aic,
    iterations: iter,
    responsibilities,
    labels,
  };
}

/**
 * Select the optimal number of components using BIC.
 *
 * Fits GMMs with 1 to maxK components and returns the one with the
 * lowest BIC score.
 *
 * @param data - Input dataset
 * @param maxK - Maximum number of components to try
 * @param options - Configuration passed to gaussianMixture
 * @returns Object containing the best k, all fitted results, and BIC values
 * @throws {Error} If maxK is less than 1
 * @throws {Error} If data contains NaN or Infinity values
 */
export function selectComponents(
  data: Dataset,
  maxK: number,
  options: { maxIterations?: number; tol?: number; seed?: number } = {},
): { bestK: number; results: GMMResult[]; bicValues: number[] } {
  if (maxK < 1) throw new Error("maxK must be at least 1");

  const results: GMMResult[] = [];
  const bicValues: number[] = [];

  for (let k = 1; k <= Math.min(maxK, data.length); k++) {
    const result = gaussianMixture(data, k, options);
    results.push(result);
    bicValues.push(result.bic);
  }

  let bestIdx = 0;
  for (let i = 1; i < bicValues.length; i++) {
    if (bicValues[i] < bicValues[bestIdx]) bestIdx = i;
  }

  return { bestK: bestIdx + 1, results, bicValues };
}

function gaussianPdf(x: number, mu: number, sigma2: number): number {
  return Math.exp(-0.5 * (x - mu) ** 2 / sigma2) / Math.sqrt(2 * Math.PI * sigma2);
}
