import { Dataset, PolynomialRegressionResult } from "../types";
import { multipleRegression } from "./multiple-regression";

/**
 * Fits a polynomial regression of a specified degree using ordinary least squares.
 *
 * Constructs the Vandermonde feature matrix [x, x^2, ..., x^d] and delegates to
 * {@link multipleRegression} to solve:
 *
 *   y = a0 + a1*x + a2*x^2 + ... + ad*x^d
 *
 * The returned `coefficients` array has length `degree + 1`, where
 * `coefficients[0]` is the intercept (a0) and `coefficients[k]` is the
 * coefficient for x^k.
 *
 * @param x - Independent variable values (length n).
 * @param y - Dependent variable values (length n).
 * @param degree - Polynomial degree (positive integer; 1 = linear, 2 = quadratic, etc.).
 * @returns A {@link PolynomialRegressionResult} containing the polynomial coefficients,
 *   degree, R^2, and a `predict` function for evaluating the fitted polynomial.
 * @throws {Error} If `degree` is not a positive integer.
 * @throws {Error} If `x` and `y` have different lengths.
 * @throws {Error} If the number of data points does not exceed the polynomial degree.
 *
 * @example
 * ```ts
 * const result = polynomialRegression([1, 2, 3, 4], [1, 4, 9, 16], 2);
 * // Fits y = x^2, so coefficients ~ [0, 0, 1]
 * result.predict(5); // ~25
 * ```
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
