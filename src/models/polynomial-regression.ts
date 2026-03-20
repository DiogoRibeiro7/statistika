import { Dataset, PolynomialRegressionResult } from "../types";
import { multipleRegression } from "./multiple-regression";

/**
 * Polynomial regression of a given degree.
 *
 * Fits y = a0 + a1*x + a2*x² + ... + ad*x^d using OLS via the
 * multiple regression solver on the Vandermonde matrix.
 *
 * @param x - Independent variable values
 * @param y - Dependent variable values
 * @param degree - Polynomial degree (1 = linear, 2 = quadratic, etc.)
 * @returns PolynomialRegressionResult with coefficients, R², and predict function
 */
export function polynomialRegression(
  x: Dataset,
  y: Dataset,
  degree: number,
): PolynomialRegressionResult {
  if (!Number.isInteger(degree) || degree < 1) {
    throw new Error("Degree must be a positive integer");
  }
  if (x.length !== y.length) {
    throw new Error("x and y must have the same length");
  }
  if (x.length <= degree) {
    throw new Error("Number of data points must exceed the polynomial degree");
  }

  // Build Vandermonde feature matrix: [x, x², ..., x^d]
  const X = x.map((xi) => {
    const row: number[] = [];
    let power = xi;
    for (let d = 1; d <= degree; d++) {
      row.push(power);
      power *= xi;
    }
    return row;
  });

  const result = multipleRegression(X, y);

  // coefficients[0] = intercept (a0), coefficients[1] = a1, ..., coefficients[d] = ad
  const coefficients = [result.intercept, ...result.coefficients];

  return {
    coefficients,
    degree,
    rSquared: result.rSquared,
    predict: (xVal: number) => {
      let yVal = coefficients[0];
      let power = xVal;
      for (let d = 1; d <= degree; d++) {
        yVal += coefficients[d] * power;
        power *= xVal;
      }
      return yVal;
    },
  };
}
