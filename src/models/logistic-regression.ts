import { LogisticRegressionResult } from "../types";
import { solveLinearSystem } from "../utils/linalg";
import { weightedCrossProducts } from "../utils/native-stats";

/**
 * Computes the logistic (sigmoid) function: sigma(z) = 1 / (1 + exp(-z)).
 *
 * Clamps the input to [-500, 500] to prevent numerical overflow.
 *
 * @param z - Input value.
 * @returns A probability in the range (0, 1).
 */
function sigmoid(z: number): number {
  if (z > 500) return 1;
  if (z < -500) return 0;
  return 1 / (1 + Math.exp(-z));
}

/**
 * Fits a binary logistic regression model using Iteratively Reweighted
 * Least Squares (IRLS), also known as Fisher scoring.
 *
 * Models the probability P(y=1|x) = sigma(intercept + x'*beta) where
 * sigma is the logistic function. The algorithm iteratively computes
 * Newton-Raphson updates by solving H * delta = gradient, where H is the
 * Hessian (X^T W X) and the gradient is X^T (y - mu).
 *
 * @param X - Feature matrix of shape (n x p), where n is the number of
 *   observations and p is the number of features.
 * @param y - Binary response vector (each element must be 0 or 1).
 * @param options - Optional configuration.
 * @param options.maxIterations - Maximum number of IRLS iterations (default 100).
 * @param options.tolerance - Convergence tolerance on the maximum absolute
 *   change in coefficients (default 1e-8).
 * @returns A {@link LogisticRegressionResult} containing fitted coefficients,
 *   intercept, iteration count, and a `predict` function that returns
 *   P(y=1|x) for a new feature vector.
 * @throws {Error} If `X` and `y` have different numbers of observations.
 * @throws {Error} If fewer than 2 observations are provided.
 * @throws {Error} If feature vectors have inconsistent lengths.
 * @throws {Error} If any response value is not 0 or 1.
 * @throws {Error} If the Hessian becomes singular during iteration.
 *
 * @example
 * ```ts
 * const X = [[0.5, 1.0], [1.5, 2.0], [2.5, 0.5], [3.5, 1.5]];
 * const y = [0, 0, 1, 1];
 * const model = logisticRegression(X, y);
 * const prob = model.predict([2.0, 1.0]); // predicted probability of class 1
 * ```
 */
export function logisticRegression(
  X: number[][],
  y: number[],
  options: { maxIterations?: number; tolerance?: number } = {},
): LogisticRegressionResult {
  const { maxIterations = 100, tolerance = 1e-8 } = options;

  const n = X.length;
  if (n !== y.length) {
    throw new Error(`Invalid parameters 'X', 'y': expected same number of observations, received X.length=${n}, y.length=${y.length}`);
  }
  if (n < 2) {
    throw new Error(`Invalid parameter 'X': expected at least 2 observations, received ${n}`);
  }

  const p = X[0].length;
  for (let i = 0; i < n; i++) {
    if (X[i].length !== p) {
      throw new Error(`Invalid parameter 'X[${i}]': expected ${p} features, received ${X[i].length}`);
    }
  }
  for (let i = 0; i < n; i++) {
    if (y[i] !== 0 && y[i] !== 1) {
      throw new Error(`Invalid parameter 'y[${i}]': expected binary (0 or 1), received ${y[i]}`);
    }
  }

  // beta = [intercept, coeff1, ..., coeffp]
  const cols = p + 1;
  const beta = new Array(cols).fill(0);
  const augmentedX = X.map((row) => [1, ...row]);
  const zeroResponse = new Array(n).fill(0);

  let iterations = 0;

  for (let iter = 0; iter < maxIterations; iter++) {
    iterations = iter + 1;

    // Compute predictions and working weights
    const mu = new Array(n); // predicted probabilities
    const w = new Array(n); // diagonal weights

    for (let i = 0; i < n; i++) {
      let z = beta[0];
      for (let j = 0; j < p; j++) {
        z += beta[j + 1] * X[i][j];
      }
      mu[i] = sigmoid(z);
      w[i] = Math.max(mu[i] * (1 - mu[i]), 1e-12); // avoid zero weights
    }

    // Compute gradient: X^T (y - mu)
    const gradient = new Array(cols).fill(0);
    for (let i = 0; i < n; i++) {
      const residual = y[i] - mu[i];
      gradient[0] += residual;
      for (let j = 0; j < p; j++) {
        gradient[j + 1] += X[i][j] * residual;
      }
    }

    // Compute Hessian: X^T W X using native acceleration when available.
    const { XtWX: H } = weightedCrossProducts(augmentedX, w, zeroResponse);

    // Solve H * delta = gradient (Newton step)
    const delta = solveLinearSystem(H, gradient);

    // Update beta
    let maxDelta = 0;
    for (let j = 0; j < cols; j++) {
      beta[j] += delta[j];
      maxDelta = Math.max(maxDelta, Math.abs(delta[j]));
    }

    if (maxDelta < tolerance) break;
  }

  const intercept = beta[0];
  const coefficients = beta.slice(1);

  return {
    coefficients,
    intercept,
    iterations,
    predict: (x: number[]) => {
      if (x.length !== p) {
        throw new Error(`Invalid parameter 'x': Expected ${p} features, received ${x.length}`);
      }
      let z = intercept;
      for (let j = 0; j < p; j++) {
        z += coefficients[j] * x[j];
      }
      return sigmoid(z);
    },
  };
}

