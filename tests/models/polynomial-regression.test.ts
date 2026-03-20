import { polynomialRegression } from "../../src/models";

describe("polynomialRegression", () => {
  it("fits a perfect quadratic relationship", () => {
    // y = 1 + 2x + 3x²
    const x = [-2, -1, 0, 1, 2, 3];
    const y = x.map((xi) => 1 + 2 * xi + 3 * xi * xi);
    const result = polynomialRegression(x, y, 2);

    expect(result.coefficients[0]).toBeCloseTo(1, 6); // intercept
    expect(result.coefficients[1]).toBeCloseTo(2, 6); // linear
    expect(result.coefficients[2]).toBeCloseTo(3, 6); // quadratic
    expect(result.rSquared).toBeCloseTo(1, 8);
    expect(result.degree).toBe(2);
  });

  it("predicts values correctly", () => {
    const x = [0, 1, 2, 3, 4, 5];
    const y = x.map((xi) => 5 - 2 * xi + xi * xi);
    const result = polynomialRegression(x, y, 2);

    expect(result.predict(6)).toBeCloseTo(5 - 12 + 36, 4);
    expect(result.predict(0)).toBeCloseTo(5, 4);
  });

  it("degree 1 matches linear regression", () => {
    const x = [1, 2, 3, 4, 5];
    const y = [2.1, 3.9, 6.2, 7.8, 10.1];
    const result = polynomialRegression(x, y, 1);

    expect(result.coefficients).toHaveLength(2); // intercept + slope
    expect(result.rSquared).toBeGreaterThan(0.99);
    expect(result.degree).toBe(1);
  });

  it("fits a cubic", () => {
    // y = x³
    const x = [-3, -2, -1, 0, 1, 2, 3];
    const y = x.map((xi) => xi ** 3);
    const result = polynomialRegression(x, y, 3);

    expect(result.coefficients[0]).toBeCloseTo(0, 4); // intercept
    expect(result.coefficients[3]).toBeCloseTo(1, 4); // cubic term
    expect(result.rSquared).toBeCloseTo(1, 6);
  });

  it("throws for non-positive degree", () => {
    expect(() => polynomialRegression([1, 2, 3], [1, 2, 3], 0)).toThrow(
      "positive integer",
    );
    expect(() => polynomialRegression([1, 2, 3], [1, 2, 3], -1)).toThrow(
      "positive integer",
    );
  });

  it("throws when x and y have different lengths", () => {
    expect(() => polynomialRegression([1, 2], [1], 1)).toThrow("same length");
  });

  it("throws when n <= degree", () => {
    expect(() => polynomialRegression([1, 2], [1, 2], 2)).toThrow("exceed");
  });
});
