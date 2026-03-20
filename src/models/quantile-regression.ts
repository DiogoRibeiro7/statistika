import { Dataset } from "../types";
import { mean } from "../utils/descriptive";

/**
 * Result of a quantile regression fit.
 */
export interface QuantileRegressionResult {
  tau: number;
  coefficients: number[];
  intercept: number;
  predict: (x: number[]) => number;
}

/**
 * Simple quantile regression result (single predictor).
 */
export interface SimpleQuantileRegressionResult {
  tau: number;
  slope: number;
  intercept: number;
  predict: (x: number) => number;
}

/**
 * Simple quantile regression for a single predictor.
 * Minimizes the "check" (pinball) loss via iteratively reweighted least squares.
 *
 * @param x - Predictor values
 * @param y - Response values
 * @param tau - Quantile to estimate (0 < tau < 1), default 0.5 (median)
 * @param maxIterations - Maximum IRLS iterations
 * @param tolerance - Convergence tolerance
 */
export function quantileRegression(
  x: Dataset,
  y: Dataset,
  tau = 0.5,
  maxIterations = 100,
  tolerance = 1e-6,
): SimpleQuantileRegressionResult {
  validateInputs(x, y, tau);

  // Build design matrix [[1, x_i], ...]
  const X = x.map((xi) => [1, xi]);
  const beta = irlsQuantile(X, y, tau, maxIterations, tolerance);

  return {
    tau,
    intercept: beta[0],
    slope: beta[1],
    predict: (xVal: number) => beta[0] + beta[1] * xVal,
  };
}

/**
 * Multiple quantile regression for multiple predictors.
 * Minimizes the check (pinball) loss via IRLS.
 *
 * @param X - Feature matrix (n × p)
 * @param y - Response values
 * @param tau - Quantile to estimate (0 < tau < 1)
 * @param maxIterations - Maximum IRLS iterations
 * @param tolerance - Convergence tolerance
 */
export function multipleQuantileRegression(
  X: number[][],
  y: Dataset,
  tau = 0.5,
  maxIterations = 100,
  tolerance = 1e-6,
): QuantileRegressionResult {
  if (X.length !== y.length) {
    throw new Error("X and y must have the same length");
  }
  if (X.length < 2) throw new Error("Need at least 2 observations");
  if (tau <= 0 || tau >= 1) throw new Error("tau must be in (0, 1)");

  const p = X[0].length;
  for (const row of X) {
    if (row.length !== p) throw new Error("All rows must have the same length");
  }

  // Add intercept column
  const Xaug = X.map((row) => [1, ...row]);
  const beta = irlsQuantile(Xaug, y, tau, maxIterations, tolerance);

  return {
    tau,
    intercept: beta[0],
    coefficients: beta.slice(1),
    predict: (xNew: number[]) => {
      if (xNew.length !== p) {
        throw new Error(`Expected ${p} features, got ${xNew.length}`);
      }
      let result = beta[0];
      for (let j = 0; j < p; j++) {
        result += beta[j + 1] * xNew[j];
      }
      return result;
    },
  };
}

/**
 * IRLS (Iteratively Reweighted Least Squares) for quantile regression.
 * Solves: min_beta sum rho_tau(y_i - x_i'beta)
 * where rho_tau(u) = u * (tau - I(u < 0))
 *
 * Uses the Hunter-Lange MM algorithm approach.
 */
function irlsQuantile(
  X: number[][],
  y: Dataset,
  tau: number,
  maxIterations: number,
  tolerance: number,
): number[] {
  const n = X.length;
  const p = X[0].length;

  // Initialize with OLS estimate
  let beta = olsInitialize(X, y);
  const epsilon = 1e-6; // small constant to avoid division by zero

  for (let iter = 0; iter < maxIterations; iter++) {
    // Compute residuals and weights
    const weights = new Array(n);
    for (let i = 0; i < n; i++) {
      let fitted = 0;
      for (let j = 0; j < p; j++) {
        fitted += X[i][j] * beta[j];
      }
      const residual = y[i] - fitted;
      const absResidual = Math.abs(residual) + epsilon;

      if (residual >= 0) {
        weights[i] = tau / absResidual;
      } else {
        weights[i] = (1 - tau) / absResidual;
      }
    }

    // Solve weighted least squares: (X'WX)beta = X'Wy
    const XtWX: number[][] = Array.from({ length: p }, () =>
      new Array(p).fill(0),
    );
    const XtWy = new Array(p).fill(0);

    for (let i = 0; i < n; i++) {
      const w = weights[i];
      for (let j = 0; j < p; j++) {
        XtWy[j] += w * X[i][j] * y[i];
        for (let k = 0; k < p; k++) {
          XtWX[j][k] += w * X[i][j] * X[i][k];
        }
      }
    }

    const newBeta = solveSystem(XtWX, XtWy);

    // Check convergence
    let maxChange = 0;
    for (let j = 0; j < p; j++) {
      maxChange = Math.max(maxChange, Math.abs(newBeta[j] - beta[j]));
    }
    beta = newBeta;

    if (maxChange < tolerance) break;
  }

  return beta;
}

/**
 * Initialize with OLS solution.
 */
function olsInitialize(X: number[][], y: Dataset): number[] {
  const n = X.length;
  const p = X[0].length;

  const XtX: number[][] = Array.from({ length: p }, () =>
    new Array(p).fill(0),
  );
  const Xty = new Array(p).fill(0);

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < p; j++) {
      Xty[j] += X[i][j] * y[i];
      for (let k = 0; k < p; k++) {
        XtX[j][k] += X[i][j] * X[i][k];
      }
    }
  }

  return solveSystem(XtX, Xty);
}

/**
 * Gaussian elimination with partial pivoting.
 */
function solveSystem(A: number[][], b: number[]): number[] {
  const n = A.length;
  const aug = A.map((row, i) => [...row, b[i]]);

  for (let col = 0; col < n; col++) {
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(aug[row][col]) > Math.abs(aug[maxRow][col])) {
        maxRow = row;
      }
    }
    [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];

    if (Math.abs(aug[col][col]) < 1e-12) {
      throw new Error("Singular matrix in quantile regression");
    }

    for (let row = col + 1; row < n; row++) {
      const factor = aug[row][col] / aug[col][col];
      for (let j = col; j <= n; j++) {
        aug[row][j] -= factor * aug[col][j];
      }
    }
  }

  const x = new Array(n).fill(0);
  for (let row = n - 1; row >= 0; row--) {
    x[row] = aug[row][n];
    for (let col = row + 1; col < n; col++) {
      x[row] -= aug[row][col] * x[col];
    }
    x[row] /= aug[row][row];
  }
  return x;
}

function validateInputs(x: Dataset, y: Dataset, tau: number): void {
  if (x.length !== y.length) {
    throw new Error("x and y must have the same length");
  }
  if (x.length < 2) throw new Error("Need at least 2 observations");
  if (tau <= 0 || tau >= 1) throw new Error("tau must be in (0, 1)");
}
