import { Dataset, MultipleRegressionResult } from "../types";
import { mean } from "../utils/descriptive";
import { solveLinearSystem } from "../utils/linalg";

/**
 * Fits a multiple linear regression model using ordinary least squares (OLS)
 * via the normal equations.
 *
 * Solves the system (X^T X) beta = X^T y where the design matrix includes
 * an intercept column prepended automatically. The coefficient of determination
 * R^2 = 1 - SS_res / SS_tot is computed on the training data.
 *
 * @param X - Feature matrix of shape (n x p), where n is the number of
 *   observations and p is the number of predictor variables.
 * @param y - Response vector of length n.
 * @returns A {@link MultipleRegressionResult} containing the fitted coefficients
 *   (one per feature), intercept, R^2, and a `predict` function for new observations.
 * @throws {Error} If `X` and `y` have different numbers of observations.
 * @throws {Error} If fewer than 2 observations are provided.
 * @throws {Error} If feature vectors are empty or have inconsistent lengths.
 * @throws {Error} If the number of observations does not exceed the number of features
 *   (the system would be underdetermined).
 *
 * @example
 * ```ts
 * const X = [[1, 2], [3, 4], [5, 6], [7, 8]];
 * const y = [3, 7, 11, 15];
 * const model = multipleRegression(X, y);
 * model.predict([9, 10]); // predicted value for a new observation
 * ```
 */
export function multipleRegression(
  X: number[][],
  y: Dataset,
): MultipleRegressionResult {
  const n = X.length;
  if (n !== y.length) {
    throw new Error(`Invalid parameters 'X', 'y': expected same number of observations, received X.length=${n}, y.length=${y.length}`);
  }
  if (n < 2) {
    throw new Error(`Invalid parameter 'X': expected at least 2 observations, received ${n}`);
  }

  const p = X[0].length;
  if (p === 0) {
    throw new Error(`Invalid parameter 'X': expected non-empty feature vectors, received 0 features`);
  }
  for (let i = 1; i < n; i++) {
    if (X[i].length !== p) {
      throw new Error(`Invalid parameter 'X[${i}]': expected ${p} features with the same length, received ${X[i].length}`);
    }
  }
  if (n <= p) {
    throw new Error(
      `Invalid parameters 'X', 'y': expected more observations than features (n must exceed p), received n=${n}, p=${p}`,
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
        throw new Error(`Invalid parameter 'x': Expected ${p} features, received ${x.length}`);
      }
      let result = intercept;
      for (let j = 0; j < p; j++) {
        result += coefficients[j] * x[j];
      }
      return result;
    },
  };
}
