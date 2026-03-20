import { linearRegression } from "../../src/models";

describe("linearRegression", () => {
  it("fits a perfect linear relationship", () => {
    const x = [1, 2, 3, 4, 5];
    const y = [2, 4, 6, 8, 10];
    const result = linearRegression(x, y);

    expect(result.slope).toBeCloseTo(2);
    expect(result.intercept).toBeCloseTo(0);
    expect(result.rSquared).toBeCloseTo(1);
  });

  it("predicts values correctly", () => {
    const x = [1, 2, 3, 4, 5];
    const y = [2, 4, 6, 8, 10];
    const result = linearRegression(x, y);

    expect(result.predict(6)).toBeCloseTo(12);
    expect(result.predict(0)).toBeCloseTo(0);
  });

  it("throws when x and y have different lengths", () => {
    expect(() => linearRegression([1, 2], [1])).toThrow();
  });

  it("throws when fewer than 2 data points", () => {
    expect(() => linearRegression([1], [1])).toThrow();
  });
});
