import { Dataset, LinearRegressionResult } from "../types";
import { mean } from "../utils/descriptive";

/**
 * Performs simple (univariate) ordinary least squares linear regression.
 *
 * Fits the model y = slope * x + intercept by minimizing the sum of
 * squared residuals. The slope is computed as:
 *
 *   slope = sum((xi - x_mean)(yi - y_mean)) / sum((xi - x_mean)^2)
 *
 * and the intercept as: intercept = y_mean - slope * x_mean.
 *
 * The coefficient of determination R^2 = 1 - SS_res / SS_tot measures
 * the proportion of variance explained by the linear model.
 *
 * @param x - Independent variable values (length n).
 * @param y - Dependent variable values (length n).
 * @returns A {@link LinearRegressionResult} containing slope, intercept, R^2,
 *   and a `predict` function for computing fitted values.
 * @throws {Error} If `x` and `y` have different lengths.
 * @throws {Error} If the datasets have fewer than 2 data points.
 *
 * @example
 * ```ts
 * const result = linearRegression([1, 2, 3], [2, 4, 6]);
 * result.slope;     // 2
 * result.intercept;  // 0
 * result.predict(4); // 8
 * ```
 */
export function linearRegression(
  x: Dataset,
  y: Dataset,
): LinearRegressionResult {
  if (x.length !== y.length) {
    throw new Error(`Invalid parameters 'x', 'y': expected same length, received x.length=${x.length}, y.length=${y.length}`);
  }
  if (x.length < 2) {
    throw new Error(`Invalid parameter 'x': expected at least 2 data points, received ${x.length}`);
  }

  const n = x.length;
  const xMean = mean(x);
  const yMean = mean(y);

  let ssXY = 0;
  let ssXX = 0;
  let ssTot = 0;

  for (let i = 0; i < n; i++) {
    ssXY += (x[i] - xMean) * (y[i] - yMean);
    ssXX += (x[i] - xMean) ** 2;
    ssTot += (y[i] - yMean) ** 2;
  }

  const slope = ssXY / ssXX;
  const intercept = yMean - slope * xMean;

  const ssRes = y.reduce((acc, yi, i) => {
    const predicted = slope * x[i] + intercept;
    return acc + (yi - predicted) ** 2;
  }, 0);

  const rSquared = 1 - ssRes / ssTot;

  return {
    slope,
    intercept,
    rSquared,
    predict: (xVal: number) => slope * xVal + intercept,
  };
}
