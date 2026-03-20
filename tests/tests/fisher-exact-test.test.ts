import { fisherExactTest } from "../../src/tests";

describe("fisherExactTest", () => {
  it("computes correct p-value for the classic tea-tasting example", () => {
    // Lady tasting tea: [[3, 1], [1, 3]]
    const result = fisherExactTest([[3, 1], [1, 3]]);

    // Known p-value ≈ 0.4857
    expect(result.pValue).toBeCloseTo(0.4857, 2);
    expect(result.rejected).toBe(false);
  });

  it("detects a strong association", () => {
    // Strong association: [[10, 0], [0, 10]]
    const result = fisherExactTest([[10, 0], [0, 10]]);

    expect(result.pValue).toBeLessThan(0.001);
    expect(result.rejected).toBe(true);
    expect(result.statistic).toBe(Infinity); // odds ratio
  });

  it("returns p-value of 1 for perfectly balanced table", () => {
    const result = fisherExactTest([[5, 5], [5, 5]]);

    expect(result.pValue).toBeCloseTo(1, 2);
    expect(result.rejected).toBe(false);
  });

  it("handles zero cells", () => {
    const result = fisherExactTest([[5, 0], [0, 5]]);

    expect(result.pValue).toBeLessThan(0.05);
    expect(result.rejected).toBe(true);
  });

  it("returns finite odds ratio for non-zero cells", () => {
    const result = fisherExactTest([[3, 2], [1, 4]]);

    expect(result.statistic).toBeCloseTo(6, 8); // (3*4)/(2*1)
    expect(result.degreesOfFreedom).toBe(0);
  });

  it("computes known p-value for small table", () => {
    // [[1, 9], [11, 3]] — known Fisher's p ≈ 0.0014
    const result = fisherExactTest([[1, 9], [11, 3]]);

    expect(result.pValue).toBeLessThan(0.01);
    expect(result.rejected).toBe(true);
  });

  it("throws for negative values", () => {
    expect(() => fisherExactTest([[-1, 2], [3, 4]])).toThrow(
      "non-negative integers",
    );
  });

  it("throws for non-integer values", () => {
    expect(() => fisherExactTest([[1.5, 2], [3, 4]])).toThrow(
      "non-negative integers",
    );
  });

  it("throws for empty table", () => {
    expect(() => fisherExactTest([[0, 0], [0, 0]])).toThrow(
      "at least one observation",
    );
  });

  it("handles asymmetric tables", () => {
    const result = fisherExactTest([[8, 2], [1, 5]]);

    expect(result.pValue).toBeLessThan(0.05);
    expect(result.rejected).toBe(true);
  });
});
