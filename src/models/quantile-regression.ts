import { Dataset } from "../types";
import { mean } from "../utils/descriptive";

/**
 * Result of a multiple quantile regression fit.
 */
export interface QuantileRegressionResult {
  /** The quantile level that was estimated (between 0 and 1). */
  tau: number;
  /** Regression coefficients for each predictor variable. */
  coefficients: number[];
  /** Intercept term of the fitted model. */
  intercept: number;
  /**
   * Predicts the conditional quantile for a new feature vector.
   * @param x - Feature vector of length p.
   * @returns The predicted quantile value.
   */
  predict: (x: number[]) => number;
}

/**
 * Result of a simple (single-predictor) quantile regression fit.
 */
export interface SimpleQuantileRegressionResult {
  /** The quantile level that was estimated (between 0 and 1). */
  tau: number;
  /** Slope coefficient for the single predictor. */
  slope: number;
  /** Intercept term of the fitted model. */
  intercept: number;
  /**
   * Predicts the conditional quantile for a new x value.
   * @param x - A single predictor value.
   * @returns The predicted quantile value.
   */
  predict: (x: number) => number;
}

/**
 * Fits a simple (single-predictor) quantile regression model.
 *
 * Minimizes the check (pinball) loss function:
 *
 *   rho_tau(u) = u * (tau - I(u < 0))
 *
 * via iteratively reweighted least squares (IRLS) using the Hunter-Lange
 * MM algorithm approach.
 *
 * @param x - Predictor values (length n).
 * @param y - Response values (length n).
 * @param tau - Quantile to estimate, must be in (0, 1). Default is 0.5 (median regression).
 * @param maxIterations - Maximum number of IRLS iterations (default 100).
 * @param tolerance - Convergence tolerance on coefficient changes (default 1e-6).
 * @returns A {@link SimpleQuantileRegressionResult} with slope, intercept, and `predict` function.
 * @throws {Error} If `x` and `y` have different lengths.
 * @throws {Error} If fewer than 2 observations are provided.
 * @throws {Error} If `tau` is not in the open interval (0, 1).
 *
 * @example
 * ```ts
 * const result = quantileRegression([1, 2, 3, 4, 5], [2, 4, 5, 4, 5], 0.5);
 * result.predict(3); // predicted median of y at x=3
 * ```
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
 * Fits a multiple quantile regression model for several predictors.
 *
 * Minimizes the check (pinball) loss rho_tau(u) = u * (tau - I(u < 0))
 * via iteratively reweighted least squares (IRLS). An intercept column
 * is prepended to X internally.
 *
 * @param X - Feature matrix of shape (n x p).
 * @param y - Response values (length n).
 * @param tau - Quantile to estimate, must be in (0, 1). Default is 0.5.
 * @param maxIterations - Maximum number of IRLS iterations (default 100).
 * @param tolerance - Convergence tolerance on coefficient changes (default 1e-6).
 * @returns A {@link QuantileRegressionResult} with coefficients, intercept, and `predict` function.
 * @throws {Error} If `X` and `y` have different lengths.
 * @throws {Error} If fewer than 2 observations are provided.
 * @throws {Error} If `tau` is not in the open interval (0, 1).
 * @throws {Error} If rows of `X` have inconsistent lengths.
 *
 * @example
 * ```ts
 * const X = [[1, 2], [3, 4], [5, 6]];
 * const y = [3, 7, 11];
 * const result = multipleQuantileRegression(X, y, 0.75);
 * result.predict([4, 5]); // predicted 75th percentile at the given features
 * ```
 */
export function multipleQuantileRegression(
  X: number[][],
  y: Dataset,
  tau = 0.5,
  maxIterations = 100,
  tolerance = 1e-6,
): QuantileRegressionResult {
  if (X.length !== y.length) {
    throw new Error(`Invalid parameters 'X', 'y': expected same length, received X.length=${X.length}, y.length=${y.length}`);
  }
  if (X.length < 2) throw new Error(`Invalid parameter 'X': expected at least 2 observations, received ${X.length}`);
  if (tau <= 0 || tau >= 1) throw new Error(`Invalid parameter 'tau': expected a value in (0, 1), received ${tau}`);

  const p = X[0].length;
  for (const row of X) {
    if (row.length !== p) throw new Error(`Invalid parameter 'X': expected all rows to have ${p} columns, received ${row.length}`);
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
        throw new Error(`Invalid parameter 'xNew': expected ${p} features, received ${xNew.length}`);
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
 * IRLS (Iteratively Reweighted Least Squares) solver for quantile regression.
 *
 * Solves: min_beta sum_i rho_tau(y_i - x_i' beta)
 * where rho_tau(u) = u * (tau - I(u < 0)) is the check/pinball loss.
 *
 * Uses the Hunter-Lange MM (majorization-minimization) algorithm, which
 * constructs quadratic upper bounds and iteratively solves weighted least
 * squares subproblems with weights w_i = tau / |r_i| for r_i >= 0 and
 * w_i = (1 - tau) / |r_i| for r_i < 0.
 *
 * @param X - Design matrix of shape (n x p), including intercept column if desired.
 * @param y - Response values (length n).
 * @param tau - Quantile level in (0, 1).
 * @param maxIterations - Maximum number of IRLS iterations.
 * @param tolerance - Convergence tolerance.
 * @returns The fitted coefficient vector of length p.
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
 * Computes an initial OLS (ordinary least squares) estimate for the IRLS algorithm.
 *
 * Solves (X^T X) beta = X^T y via Gaussian elimination.
 *
 * @param X - Design matrix of shape (n x p).
 * @param y - Response vector of length n.
 * @returns The OLS coefficient vector of length p.
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
 * Solves the linear system A*x = b using Gaussian elimination with partial pivoting.
 *
 * @param A - Square coefficient matrix of shape (n x n).
 * @param b - Right-hand side vector of length n.
 * @returns The solution vector x of length n.
 * @throws {Error} If the matrix is singular (pivot element below 1e-12).
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

/**
 * Validates inputs for simple quantile regression.
 *
 * @param x - Predictor values.
 * @param y - Response values.
 * @param tau - Quantile level.
 * @throws {Error} If `x` and `y` differ in length, have fewer than 2 elements, or `tau` is not in (0, 1).
 */
function validateInputs(x: Dataset, y: Dataset, tau: number): void {
  if (x.length !== y.length) {
    throw new Error(`Invalid parameters 'x', 'y': expected same length, received x.length=${x.length}, y.length=${y.length}`);
  }
  if (x.length < 2) throw new Error(`Invalid parameter 'x': expected at least 2 observations, received ${x.length}`);
  if (tau <= 0 || tau >= 1) throw new Error(`Invalid parameter 'tau': expected a value in (0, 1), received ${tau}`);
}
