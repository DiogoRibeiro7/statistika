import { logisticRegression } from "../../src/models";

describe("logisticRegression", () => {
  it("learns a linearly separable dataset", () => {
    // Simple 1D: low values → 0, high values → 1
    const X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]];
    const y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1];

    const result = logisticRegression(X, y);

    // Coefficient should be positive (higher x → higher probability)
    expect(result.coefficients[0]).toBeGreaterThan(0);

    // Predictions for clearly low and high values
    expect(result.predict([1])).toBeLessThan(0.3);
    expect(result.predict([10])).toBeGreaterThan(0.7);
  });

  it("handles 2D features", () => {
    const X = [
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
      [2, 0],
      [0, 2],
      [2, 2],
      [3, 3],
      [3, 2],
      [2, 3],
    ];
    const y = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1];

    const result = logisticRegression(X, y);

    expect(result.coefficients).toHaveLength(2);
    expect(result.predict([0, 0])).toBeLessThan(0.5);
    expect(result.predict([3, 3])).toBeGreaterThan(0.5);
  });

  it("returns iterations count", () => {
    const X = [[1], [2], [3], [4], [5], [6]];
    const y = [0, 0, 0, 1, 1, 1];

    const result = logisticRegression(X, y);

    expect(result.iterations).toBeGreaterThan(0);
    expect(result.iterations).toBeLessThanOrEqual(100);
  });

  it("respects maxIterations option", () => {
    const X = [[1], [2], [3], [4], [5], [6]];
    const y = [0, 0, 0, 1, 1, 1];

    const result = logisticRegression(X, y, { maxIterations: 5 });

    expect(result.iterations).toBeLessThanOrEqual(5);
  });

  it("throws when X and y have different lengths", () => {
    expect(() => logisticRegression([[1], [2]], [0])).toThrow(
      "same number of observations",
    );
  });

  it("throws for non-binary response", () => {
    expect(() => logisticRegression([[1], [2]], [0, 2])).toThrow("binary");
  });

  it("throws for fewer than 2 observations", () => {
    expect(() => logisticRegression([[1]], [0])).toThrow("at least 2");
  });

  it("predict throws on wrong feature count", () => {
    const X = [[1, 5], [3, 2], [6, 1], [8, 7], [2, 8], [7, 3]];
    const y = [0, 0, 0, 1, 1, 1];
    const result = logisticRegression(X, y);

    expect(() => result.predict([1])).toThrow("Expected 2 features");
  });
});
