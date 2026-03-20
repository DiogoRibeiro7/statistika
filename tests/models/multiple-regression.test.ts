import { multipleRegression } from "../../src/models";

describe("multipleRegression", () => {
  it("fits a perfect linear relationship with two features", () => {
    // y = 2*x1 + 3*x2 + 1
    const X = [
      [1, 0],
      [0, 1],
      [1, 1],
      [2, 1],
      [1, 2],
    ];
    const y = X.map(([x1, x2]) => 2 * x1 + 3 * x2 + 1);
    const result = multipleRegression(X, y);

    expect(result.coefficients[0]).toBeCloseTo(2, 8);
    expect(result.coefficients[1]).toBeCloseTo(3, 8);
    expect(result.intercept).toBeCloseTo(1, 8);
    expect(result.rSquared).toBeCloseTo(1, 8);
  });

  it("predicts values correctly", () => {
    const X = [
      [1, 5],
      [2, 3],
      [3, 1],
      [4, 6],
      [5, 2],
    ];
    const y = X.map(([x1, x2]) => 1.5 * x1 + 2.5 * x2 + 0.5);
    const result = multipleRegression(X, y);

    expect(result.predict([6, 7])).toBeCloseTo(1.5 * 6 + 2.5 * 7 + 0.5, 6);
  });

  it("handles noisy data with reasonable R²", () => {
    const X = [
      [1, 3],
      [2, 1],
      [3, 4],
      [4, 2],
      [5, 6],
      [6, 5],
    ];
    const y = [8.1, 5.9, 12.2, 9.8, 18.1, 16.9];
    const result = multipleRegression(X, y);

    expect(result.rSquared).toBeGreaterThan(0.95);
  });

  it("throws when X and y have different lengths", () => {
    expect(() =>
      multipleRegression(
        [[1], [2]],
        [1, 2, 3],
      ),
    ).toThrow("same number of observations");
  });

  it("throws when fewer than 2 observations", () => {
    expect(() => multipleRegression([[1, 2]], [1])).toThrow("at least 2");
  });

  it("throws when feature vectors have different lengths", () => {
    expect(() =>
      multipleRegression(
        [[1, 2], [3]],
        [1, 2],
      ),
    ).toThrow("same length");
  });

  it("throws when n <= p", () => {
    expect(() =>
      multipleRegression(
        [[1, 2, 3], [4, 5, 6]],
        [1, 2],
      ),
    ).toThrow("exceed");
  });

  it("predict throws on wrong feature count", () => {
    const X = [
      [1, 5],
      [3, 2],
      [5, 4],
      [2, 6],
    ];
    const y = [12, 8, 14, 15];
    const result = multipleRegression(X, y);

    expect(() => result.predict([1])).toThrow("Expected 2 features");
  });
});
