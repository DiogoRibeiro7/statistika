/**
 * Robust regression methods for outlier-resistant linear fitting.
 *
 * Provides two approaches:
 * - **Huber M-estimator**: Uses IRLS with the Huber loss function, which is
 *   quadratic for small residuals and linear for large residuals.
 * - **RANSAC**: Random Sample Consensus, which repeatedly fits models to
 *   random subsets and selects the one with the most inlier support.
 */

import { mean } from "../utils/descriptive";
import { solveLinearSystem, randomSample } from "../utils/linalg";
import { weightedCrossProducts } from "../utils/native-stats";

// ---- Huber Regression ----

/**
 * Result of a Huber robust regression fit.
 */
export interface HuberRegressionResult {
  /** Full coefficient vector including intercept as the first element. */
  coefficients: number[];
  /** Intercept (constant) term of the fitted model. */
  intercept: number;
  /** Slope coefficients (one per predictor, excludes intercept). */
  slopes: number[];
  /** Robust scale estimate based on the Median Absolute Deviation (MAD). */
  scale: number;
  /** Number of IRLS iterations performed until convergence or limit. */
  iterations: number;
  /**
   * Predicts the response for a given input.
   * @param x - A single predictor value (simple regression) or feature vector (multiple regression).
   * @returns The predicted response value.
   */
  predict: (x: number | number[]) => number;
}

/**
 * Fits a Huber M-estimator regression using Iteratively Reweighted Least Squares (IRLS).
 *
 * The Huber loss function is defined as:
 *
 *   L(u) = u^2 / 2           if |u| <= delta
 *   L(u) = delta * |u| - delta^2 / 2  if |u| > delta
 *
 * This provides a smooth transition between quadratic loss (for small residuals)
 * and linear loss (for large residuals / outliers). The scale is estimated
 * robustly via MAD (Median Absolute Deviation) / 0.6745.
 *
 * Accepts either a 1D array (simple regression) or a 2D matrix (multiple regression).
 * An intercept column is prepended internally.
 *
 * @param X - Predictor values: a numeric array of length n for simple regression,
 *   or an (n x p) matrix for multiple regression.
 * @param y - Response variable (length n).
 * @param delta - Huber threshold controlling the transition from quadratic to
 *   linear loss. Default is 1.345, which yields 95% asymptotic efficiency
 *   at the normal distribution.
 * @param maxIterations - Maximum number of IRLS iterations (default 50).
 * @param tolerance - Convergence tolerance on the maximum absolute change
 *   in coefficients (default 1e-6).
 * @returns A {@link HuberRegressionResult} with coefficients, slopes, scale estimate,
 *   iteration count, and a `predict` function.
 * @throws {Error} If `X` and `y` have different lengths.
 * @throws {Error} If fewer than 2 observations are provided.
 * @throws {Error} If `delta` is not positive.
 *
 * @example
 * ```ts
 * const x = [1, 2, 3, 4, 100]; // outlier at index 4
 * const y = [2, 4, 6, 8, 200];
 * const result = huberRegression(x, y);
 * result.predict(5); // robust prediction, less affected by the outlier
 * ```
 */
export function huberRegression(
  X: number[] | number[][],
  y: number[],
  delta = 1.345,
  maxIterations = 50,
  tolerance = 1e-6,
): HuberRegressionResult {
  // Normalize input: convert simple array to matrix with intercept
  const isSimple = typeof X[0] === "number";
  const n = y.length;

  let designMatrix: number[][];
  if (isSimple) {
    const xArr = X as number[];
    if (xArr.length !== n) throw new Error(`Invalid parameters 'X', 'y': expected same length, received X.length=${xArr.length}, y.length=${n}`);
    // Add intercept column
    designMatrix = xArr.map((x) => [1, x]);
  } else {
    const xMat = X as number[][];
    if (xMat.length !== n) throw new Error(`Invalid parameters 'X', 'y': expected same length, received X.length=${xMat.length}, y.length=${n}`);
    // Add intercept column
    designMatrix = xMat.map((row) => [1, ...row]);
  }

  if (n < 2) throw new Error(`Invalid parameter 'y': expected at least 2 observations, received ${n}`);
  if (delta <= 0) throw new Error(`Invalid parameter 'delta': expected a positive number, received ${delta}`);

  const p = designMatrix[0].length;

  // Initial OLS estimate
  let beta = olsSolve(designMatrix, y);

  let iterations = 0;
  let scale = 1;

  for (let iter = 0; iter < maxIterations; iter++) {
    iterations = iter + 1;

    // Compute residuals
    const residuals = new Array(n);
    for (let i = 0; i < n; i++) {
      let yHat = 0;
      for (let j = 0; j < p; j++) yHat += designMatrix[i][j] * beta[j];
      residuals[i] = y[i] - yHat;
    }

    // Robust scale estimate (MAD)
    const absResiduals = residuals.map(Math.abs);
    absResiduals.sort((a, b) => a - b);
    const medianAbsRes =
      n % 2 === 1
        ? absResiduals[Math.floor(n / 2)]
        : (absResiduals[n / 2 - 1] + absResiduals[n / 2]) / 2;
    scale = medianAbsRes / 0.6745; // MAD scaled to match normal std dev
    if (scale < 1e-10) scale = 1e-10; // Prevent division by zero

    // Compute Huber weights
    const weights = new Array(n);
    for (let i = 0; i < n; i++) {
      const u = Math.abs(residuals[i]) / scale;
      weights[i] = u <= delta ? 1 : delta / u;
    }

    // Weighted least squares
    const newBeta = wlsSolve(designMatrix, y, weights);

    // Check convergence
    let maxChange = 0;
    for (let j = 0; j < p; j++) {
      maxChange = Math.max(maxChange, Math.abs(newBeta[j] - beta[j]));
    }

    beta = newBeta;
    if (maxChange < tolerance) break;
  }

  const intercept = beta[0];
  const slopes = beta.slice(1);

  return {
    coefficients: beta,
    intercept,
    slopes,
    scale,
    iterations,
    predict: (x: number | number[]) => {
      if (typeof x === "number") {
        return intercept + slopes[0] * x;
      }
      let result = intercept;
      for (let j = 0; j < slopes.length; j++) result += slopes[j] * x[j];
      return result;
    },
  };
}

// ---- RANSAC Regression ----

/**
 * Result of a RANSAC (Random Sample Consensus) regression fit.
 */
export interface RANSACResult {
  /** Full coefficient vector including intercept as the first element. */
  coefficients: number[];
  /** Intercept (constant) term of the fitted model. */
  intercept: number;
  /** Slope coefficients (one per predictor, excludes intercept). */
  slopes: number[];
  /** 0-based indices of observations identified as inliers by the best model. */
  inlierIndices: number[];
  /** Total number of inlier observations in the best model. */
  nInliers: number;
  /** Number of random trials performed. */
  iterations: number;
  /**
   * Predicts the response for a given input.
   * @param x - A single predictor value (simple regression) or feature vector (multiple regression).
   * @returns The predicted response value.
   */
  predict: (x: number | number[]) => number;
}

/**
 * Fits a RANSAC (Random Sample Consensus) robust regression model.
 *
 * The algorithm proceeds by:
 * 1. Randomly sampling `minSamples` observations.
 * 2. Fitting an OLS model to the subsample.
 * 3. Counting inliers (observations with |residual| <= residualThreshold).
 * 4. Repeating for `maxTrials` iterations, keeping the model with the most inliers.
 * 5. Refitting OLS on all inliers of the best model.
 *
 * Accepts either a 1D array (simple regression) or a 2D matrix (multiple regression).
 * An intercept column is prepended internally.
 *
 * @param X - Predictor values: a numeric array of length n for simple regression,
 *   or an (n x p) matrix for multiple regression.
 * @param y - Response variable (length n).
 * @param residualThreshold - Maximum absolute residual for an observation to be
 *   classified as an inlier. If omitted, defaults to 3 * MAD / 0.6745 of the
 *   OLS residuals.
 * @param maxTrials - Maximum number of random sampling trials (default 100).
 * @param minSamples - Number of observations to sample per trial. Defaults to
 *   the number of parameters (including intercept).
 * @param random - Random number generator function returning values in [0, 1),
 *   useful for reproducibility. Defaults to `Math.random`.
 * @returns A {@link RANSACResult} with coefficients, inlier information, and
 *   a `predict` function.
 * @throws {Error} If `X` and `y` have different lengths.
 * @throws {Error} If there are not enough observations for the minimum sample size.
 *
 * @example
 * ```ts
 * const x = [1, 2, 3, 4, 5, 100]; // outlier at index 5
 * const y = [2, 4, 6, 8, 10, 999];
 * const result = ransacRegression(x, y);
 * result.nInliers; // 5 (the outlier is excluded)
 * result.predict(6); // ~12
 * ```
 */
export function ransacRegression(
  X: number[] | number[][],
  y: number[],
  residualThreshold?: number,
  maxTrials = 100,
  minSamples?: number,
  random: () => number = Math.random,
): RANSACResult {
  const isSimple = typeof X[0] === "number";
  const n = y.length;

  let designMatrix: number[][];
  if (isSimple) {
    const xArr = X as number[];
    if (xArr.length !== n) throw new Error(`Invalid parameters 'X', 'y': expected same length, received X.length=${xArr.length}, y.length=${n}`);
    designMatrix = xArr.map((x) => [1, x]);
  } else {
    const xMat = X as number[][];
    if (xMat.length !== n) throw new Error(`Invalid parameters 'X', 'y': expected same length, received X.length=${xMat.length}, y.length=${n}`);
    designMatrix = xMat.map((row) => [1, ...row]);
  }

  const p = designMatrix[0].length;
  const samplesPerTrial = minSamples ?? p;

  if (n < samplesPerTrial) {
    throw new Error(`Invalid parameter 'X': expected at least ${samplesPerTrial} observations for RANSAC, received ${n}`);
  }

  // Default threshold: MAD of OLS residuals * 3
  if (residualThreshold === undefined) {
    const olsBeta = olsSolve(designMatrix, y);
    const olsResiduals = y.map((yi, i) => {
      let yHat = 0;
      for (let j = 0; j < p; j++) yHat += designMatrix[i][j] * olsBeta[j];
      return Math.abs(yi - yHat);
    });
    olsResiduals.sort((a, b) => a - b);
    const mad =
      n % 2 === 1
        ? olsResiduals[Math.floor(n / 2)]
        : (olsResiduals[n / 2 - 1] + olsResiduals[n / 2]) / 2;
    residualThreshold = 3 * mad / 0.6745;
    if (residualThreshold < 1e-10) residualThreshold = 1;
  }

  let bestInliers: number[] = [];
  let bestBeta: number[] = new Array(p).fill(0);
  let bestNInliers = 0;
  let iterations = 0;

  for (let trial = 0; trial < maxTrials; trial++) {
    iterations = trial + 1;

    // Random subsample
    const indices = randomSample(n, samplesPerTrial, random);
    const subX = indices.map((i) => designMatrix[i]);
    const subY = indices.map((i) => y[i]);

    // Fit model to subsample
    const beta = olsSolve(subX, subY);

    // Find inliers
    const inliers: number[] = [];
    for (let i = 0; i < n; i++) {
      let yHat = 0;
      for (let j = 0; j < p; j++) yHat += designMatrix[i][j] * beta[j];
      if (Math.abs(y[i] - yHat) <= residualThreshold!) {
        inliers.push(i);
      }
    }

    if (inliers.length > bestNInliers) {
      bestNInliers = inliers.length;
      bestInliers = inliers;
      bestBeta = beta;
    }

    // Early termination if we have enough inliers
    if (bestNInliers > 0.8 * n) break;
  }

  // Refit on all inliers
  if (bestInliers.length >= p) {
    const inlierX = bestInliers.map((i) => designMatrix[i]);
    const inlierY = bestInliers.map((i) => y[i]);
    bestBeta = olsSolve(inlierX, inlierY);
  }

  const intercept = bestBeta[0];
  const slopes = bestBeta.slice(1);

  return {
    coefficients: bestBeta,
    intercept,
    slopes,
    inlierIndices: bestInliers,
    nInliers: bestInliers.length,
    iterations,
    predict: (x: number | number[]) => {
      if (typeof x === "number") {
        return intercept + slopes[0] * x;
      }
      let result = intercept;
      for (let j = 0; j < slopes.length; j++) result += slopes[j] * x[j];
      return result;
    },
  };
}

// ---- Helper Functions ----

/**
 * Solves ordinary least squares: beta = (X^T X)^{-1} X^T y.
 *
 * @param X - Design matrix of shape (n x p).
 * @param y - Response vector of length n.
 * @returns The OLS coefficient vector of length p.
 */
function olsSolve(X: number[][], y: number[]): number[] {
  const n = X.length;
  const ones = new Array(n).fill(1);
  const { XtWX, XtWz } = weightedCrossProducts(X, ones, y);
  return solveLinearSystem(XtWX, XtWz);
}

/**
 * Solves weighted least squares: beta = (X^T W X)^{-1} X^T W y,
 * where W = diag(w) is the diagonal weight matrix.
 *
 * @param X - Design matrix of shape (n x p).
 * @param y - Response vector of length n.
 * @param w - Weight vector of length n (non-negative).
 * @returns The WLS coefficient vector of length p.
 */
function wlsSolve(X: number[][], y: number[], w: number[]): number[] {
  const { XtWX, XtWz } = weightedCrossProducts(X, w, y);
  return solveLinearSystem(XtWX, XtWz);
}

