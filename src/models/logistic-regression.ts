import { LogisticRegressionResult } from "../types";

function sigmoid(z: number): number {
  if (z > 500) return 1;
  if (z < -500) return 0;
  return 1 / (1 + Math.exp(-z));
}

/**
 * Binary logistic regression using iteratively reweighted least squares (IRLS).
 *
 * @param X - Array of feature vectors (n observations x p features)
 * @param y - Binary response variable (0 or 1)
 * @param options - Optional settings
 * @param options.maxIterations - Maximum IRLS iterations (default 100)
 * @param options.tolerance - Convergence tolerance (default 1e-8)
 * @returns LogisticRegressionResult with coefficients, intercept, and predict function
 */
export function logisticRegression(
  X: number[][],
  y: number[],
  options: { maxIterations?: number; tolerance?: number } = {},
): LogisticRegressionResult {
  const { maxIterations = 100, tolerance = 1e-8 } = options;

  const n = X.length;
  if (n !== y.length) {
    throw new Error("X and y must have the same number of observations");
  }
  if (n < 2) {
    throw new Error("Must have at least 2 observations");
  }

  const p = X[0].length;
  for (let i = 0; i < n; i++) {
    if (X[i].length !== p) {
      throw new Error("All feature vectors must have the same length");
    }
  }
  for (let i = 0; i < n; i++) {
    if (y[i] !== 0 && y[i] !== 1) {
      throw new Error("Response variable must be binary (0 or 1)");
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
        throw new Error(`Expected ${p} features, got ${x.length}`);
      }
      let z = intercept;
      for (let j = 0; j < p; j++) {
        z += coefficients[j] * x[j];
      }
      return sigmoid(z);
    },
  };
}

/** Solve augmented matrix [A|b] via Gaussian elimination with partial pivoting. */
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
