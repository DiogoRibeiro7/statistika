import {
  quantileRegression,
  multipleQuantileRegression,
} from "../src/models/quantile-regression";

describe("quantileRegression (simple)", () => {
  // y = 2x + 1 with no noise => all quantiles should give same result
  const x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  const y = x.map((xi) => 2 * xi + 1);

  it("fits the median (tau=0.5) on a perfect linear relationship", () => {
    const result = quantileRegression(x, y, 0.5);

    expect(result.tau).toBe(0.5);
    expect(result.slope).toBeCloseTo(2, 1);
    expect(result.intercept).toBeCloseTo(1, 1);
    expect(result.predict(5)).toBeCloseTo(11, 0);
  });

  it("fits lower quantile (tau=0.25) on perfect data", () => {
    const result = quantileRegression(x, y, 0.25);
    expect(result.slope).toBeCloseTo(2, 1);
    expect(result.intercept).toBeCloseTo(1, 1);
  });

  it("fits upper quantile (tau=0.75) on perfect data", () => {
    const result = quantileRegression(x, y, 0.75);
    expect(result.slope).toBeCloseTo(2, 1);
    expect(result.intercept).toBeCloseTo(1, 1);
  });

  it("upper quantile is above lower quantile for data with noise", () => {
    // Add asymmetric noise
    const yNoisy = [3, 4, 7, 10, 9, 14, 13, 18, 17, 25];

    const q25 = quantileRegression(x, yNoisy, 0.25);
    const q50 = quantileRegression(x, yNoisy, 0.5);
    const q75 = quantileRegression(x, yNoisy, 0.75);

    // At x=5, predictions should be ordered
    expect(q25.predict(5)).toBeLessThanOrEqual(q50.predict(5) + 1);
    expect(q50.predict(5)).toBeLessThanOrEqual(q75.predict(5) + 1);
  });

  it("defaults to median regression (tau=0.5)", () => {
    const result = quantileRegression(x, y);
    expect(result.tau).toBe(0.5);
  });

  it("throws on mismatched lengths", () => {
    expect(() => quantileRegression([1, 2], [1])).toThrow();
  });

  it("throws on insufficient data", () => {
    expect(() => quantileRegression([1], [1])).toThrow();
  });

  it("throws on invalid tau", () => {
    expect(() => quantileRegression(x, y, 0)).toThrow();
    expect(() => quantileRegression(x, y, 1)).toThrow();
    expect(() => quantileRegression(x, y, -0.5)).toThrow();
    expect(() => quantileRegression(x, y, 1.5)).toThrow();
  });
});

describe("multipleQuantileRegression", () => {
  it("fits median with multiple predictors", () => {
    // y = 3*x1 + 2*x2 + 1
    const X = [
      [1, 1],
      [2, 1],
      [3, 2],
      [4, 2],
      [5, 3],
      [6, 3],
      [7, 4],
      [8, 4],
    ];
    const y = X.map(([x1, x2]) => 3 * x1 + 2 * x2 + 1);

    const result = multipleQuantileRegression(X, y, 0.5);

    expect(result.tau).toBe(0.5);
    expect(result.coefficients[0]).toBeCloseTo(3, 0);
    expect(result.coefficients[1]).toBeCloseTo(2, 0);
    expect(result.intercept).toBeCloseTo(1, 0);
  });

  it("predict function works correctly", () => {
    // Non-collinear features
    const X = [
      [1, 5], [2, 3], [3, 7], [4, 2], [5, 8],
      [6, 1], [7, 6], [8, 4], [9, 9], [10, 3],
      [1, 2], [3, 8], [5, 4], [7, 1], [9, 6],
    ];
    const y = X.map(([x1, x2]) => 2 * x1 - x2 + 5);

    const result = multipleQuantileRegression(X, y, 0.5);
    const predicted = result.predict([10, 5]);
    const expected = 2 * 10 - 5 + 5;

    expect(predicted).toBeCloseTo(expected, 0);
  });

  it("throws on mismatched dimensions", () => {
    expect(() =>
      multipleQuantileRegression([[1, 2], [3]], [1, 2], 0.5),
    ).toThrow();
  });
});
