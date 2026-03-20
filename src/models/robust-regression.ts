/**
 * Robust regression: Huber and RANSAC methods for outlier-resistant fitting.
 */

import { mean } from "../utils/descriptive";
import { solveLinearSystem, randomSample } from "../utils/linalg";

// ---- Huber Regression ----

/** Result of a Huber robust regression. */
export interface HuberRegressionResult {
  /** Regression coefficients (including intercept as first element) */
  coefficients: number[];
  /** Intercept */
  intercept: number;
  /** Slope(s) */
  slopes: number[];
  /** Robust scale estimate */
  scale: number;
  /** Number of IRLS iterations */
  iterations: number;
  /** Predict y for given x */
  predict: (x: number | number[]) => number;
}

/**
 * Huber M-estimator regression.
 *
 * Uses Iteratively Reweighted Least Squares (IRLS) with the Huber
 * loss function, which is quadratic for small residuals and linear
 * for large residuals, providing robustness to outliers.
 *
 * @param X - Predictor matrix (n x p) or vector (n x 1 for simple regression)
 * @param y - Response variable
 * @param delta - Huber threshold (default 1.345 for 95% efficiency at normal)
 * @param maxIterations - Maximum IRLS iterations (default 50)
 * @param tolerance - Convergence tolerance (default 1e-6)
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
    if (xArr.length !== n) throw new Error("X and y must have the same length");
    // Add intercept column
    designMatrix = xArr.map((x) => [1, x]);
  } else {
    const xMat = X as number[][];
    if (xMat.length !== n) throw new Error("X and y must have the same length");
    // Add intercept column
    designMatrix = xMat.map((row) => [1, ...row]);
  }

  if (n < 2) throw new Error("Need at least 2 observations");
  if (delta <= 0) throw new Error("delta must be positive");

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

/** Result of a RANSAC regression. */
export interface RANSACResult {
  /** Regression coefficients (including intercept as first element) */
  coefficients: number[];
  /** Intercept */
  intercept: number;
  /** Slope(s) */
  slopes: number[];
  /** Indices of inlier observations */
  inlierIndices: number[];
  /** Number of inliers */
  nInliers: number;
  /** Number of iterations run */
  iterations: number;
  /** Predict y for given x */
  predict: (x: number | number[]) => number;
}

/**
 * RANSAC (Random Sample Consensus) robust regression.
 *
 * Repeatedly fits models to random subsets, identifies inliers, and
 * selects the model with the most inlier support.
 *
 * @param X - Predictor(s)
 * @param y - Response variable
 * @param residualThreshold - Max residual to be considered an inlier
 * @param maxTrials - Maximum number of random trials (default 100)
 * @param minSamples - Minimum samples per trial (default p+1)
 * @param random - Random number generator for reproducibility
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
    if (xArr.length !== n) throw new Error("X and y must have the same length");
    designMatrix = xArr.map((x) => [1, x]);
  } else {
    const xMat = X as number[][];
    if (xMat.length !== n) throw new Error("X and y must have the same length");
    designMatrix = xMat.map((row) => [1, ...row]);
  }

  const p = designMatrix[0].length;
  const samplesPerTrial = minSamples ?? p;

  if (n < samplesPerTrial) {
    throw new Error("Not enough observations for RANSAC");
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
 * Solve ordinary least squares: beta = (X'X)^{-1} X'y
 */
function olsSolve(X: number[][], y: number[]): number[] {
  const n = X.length;
  const p = X[0].length;

  // X'X
  const XtX: number[][] = [];
  for (let j = 0; j < p; j++) {
    XtX[j] = new Array(p).fill(0);
    for (let k = 0; k < p; k++) {
      for (let i = 0; i < n; i++) {
        XtX[j][k] += X[i][j] * X[i][k];
      }
    }
  }

  // X'y
  const Xty = new Array(p).fill(0);
  for (let j = 0; j < p; j++) {
    for (let i = 0; i < n; i++) {
      Xty[j] += X[i][j] * y[i];
    }
  }

  // Solve via Cholesky or direct inverse for small p
  return solveLinearSystem(XtX, Xty);
}

/**
 * Solve weighted least squares: beta = (X'WX)^{-1} X'Wy
 */
function wlsSolve(X: number[][], y: number[], w: number[]): number[] {
  const n = X.length;
  const p = X[0].length;

  const XtWX: number[][] = [];
  for (let j = 0; j < p; j++) {
    XtWX[j] = new Array(p).fill(0);
    for (let k = 0; k < p; k++) {
      for (let i = 0; i < n; i++) {
        XtWX[j][k] += X[i][j] * w[i] * X[i][k];
      }
    }
  }

  const XtWy = new Array(p).fill(0);
  for (let j = 0; j < p; j++) {
    for (let i = 0; i < n; i++) {
      XtWy[j] += X[i][j] * w[i] * y[i];
    }
  }

  return solveLinearSystem(XtWX, XtWy);
}

