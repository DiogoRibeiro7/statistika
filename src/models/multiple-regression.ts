import { Dataset, MultipleRegressionResult } from "../types";
import { mean } from "../utils/descriptive";

/**
 * Solve a linear system Ax = b using Gaussian elimination with partial pivoting.
 * A is modified in place. Returns the solution vector x.
 */
function solveLinearSystem(A: number[][], b: number[]): number[] {
  const n = A.length;
  const aug: number[][] = A.map((row, i) => [...row, b[i]]);

  for (let col = 0; col < n; col++) {
    // Partial pivoting
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(aug[row][col]) > Math.abs(aug[maxRow][col])) {
        maxRow = row;
      }
    }
    [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];

    if (Math.abs(aug[col][col]) < 1e-12) {
      throw new Error("Singular matrix: features may be linearly dependent");
    }

    // Eliminate below
    for (let row = col + 1; row < n; row++) {
      const factor = aug[row][col] / aug[col][col];
      for (let j = col; j <= n; j++) {
        aug[row][j] -= factor * aug[col][j];
      }
    }
  }

  // Back substitution
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
 * Multiple linear regression using ordinary least squares (normal equations).
 *
 * @param X - Array of feature vectors, each of length p (n observations x p features)
 * @param y - Response variable (n observations)
 * @returns MultipleRegressionResult with coefficients, intercept, R², and predict function
 */
export function multipleRegression(
  X: number[][],
  y: Dataset,
): MultipleRegressionResult {
  const n = X.length;
  if (n !== y.length) {
    throw new Error("X and y must have the same number of observations");
  }
  if (n < 2) {
    throw new Error("Must have at least 2 observations");
  }

  const p = X[0].length;
  if (p === 0) {
    throw new Error("Feature vectors must not be empty");
  }
  for (let i = 1; i < n; i++) {
    if (X[i].length !== p) {
      throw new Error("All feature vectors must have the same length");
    }
  }
  if (n <= p) {
    throw new Error(
      "Number of observations must exceed number of features",
    );
  }

  // Build design matrix with intercept column: [1, x1, x2, ..., xp]
  const cols = p + 1;
  // Compute X^T * X (cols x cols) and X^T * y (cols x 1)
  const XtX: number[][] = Array.from({ length: cols }, () =>
    new Array(cols).fill(0),
  );
  const Xty: number[] = new Array(cols).fill(0);

  for (let i = 0; i < n; i++) {
    const row = [1, ...X[i]];
    for (let j = 0; j < cols; j++) {
      Xty[j] += row[j] * y[i];
      for (let k = 0; k < cols; k++) {
        XtX[j][k] += row[j] * row[k];
      }
    }
  }

  // Solve normal equations: (X^T X) beta = X^T y
  const beta = solveLinearSystem(XtX, Xty);
  const intercept = beta[0];
  const coefficients = beta.slice(1);

  // Compute R²
  const yMean = mean(y);
  let ssTot = 0;
  let ssRes = 0;
  for (let i = 0; i < n; i++) {
    let predicted = intercept;
    for (let j = 0; j < p; j++) {
      predicted += coefficients[j] * X[i][j];
    }
    ssRes += (y[i] - predicted) ** 2;
    ssTot += (y[i] - yMean) ** 2;
  }
  const rSquared = 1 - ssRes / ssTot;

  return {
    coefficients,
    intercept,
    rSquared,
    predict: (x: number[]) => {
      if (x.length !== p) {
        throw new Error(`Expected ${p} features, got ${x.length}`);
      }
      let result = intercept;
      for (let j = 0; j < p; j++) {
        result += coefficients[j] * x[j];
      }
      return result;
    },
  };
}
