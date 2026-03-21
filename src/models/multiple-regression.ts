import { Dataset, MultipleRegressionResult } from "../types";
import { mean } from "../utils/descriptive";
import { solveLinearSystem } from "../utils/linalg";

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
