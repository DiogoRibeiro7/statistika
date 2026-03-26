import { LogisticRegressionResult } from "../types";

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

    // Compute Hessian: -X^T W X
    const H: number[][] = Array.from({ length: cols }, () =>
      new Array(cols).fill(0),
    );
    for (let i = 0; i < n; i++) {
      const row = [1, ...X[i]];
      for (let j = 0; j < cols; j++) {
        for (let k = j; k < cols; k++) {
          H[j][k] += row[j] * row[k] * w[i];
          if (k !== j) H[k][j] = H[j][k];
        }
      }
    }

    // Solve H * delta = gradient (Newton step)
    const aug = H.map((row, i) => [...row, gradient[i]]);
    const delta = solveAugmented(aug, cols);

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
        throw new Error(`Invalid parameter 'x': expected ${p} features, received ${x.length}`);
      }
      let z = intercept;
      for (let j = 0; j < p; j++) {
        z += coefficients[j] * x[j];
      }
      return sigmoid(z);
    },
  };
}

/**
 * Solves the linear system A*x = b via Gaussian elimination with partial pivoting,
 * given the augmented matrix [A|b].
 *
 * @param aug - Augmented matrix of shape (n x (n+1)), modified in place.
 * @param n - Number of unknowns (rows/columns of A).
 * @returns The solution vector x of length n.
 * @throws {Error} If the matrix is singular (pivot element near zero).
 */
function solveAugmented(aug: number[][], n: number): number[] {
  for (let col = 0; col < n; col++) {
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(aug[row][col]) > Math.abs(aug[maxRow][col])) {
        maxRow = row;
      }
    }
    [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];

    if (Math.abs(aug[col][col]) < 1e-12) {
      throw new Error("Singular matrix in logistic regression");
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
