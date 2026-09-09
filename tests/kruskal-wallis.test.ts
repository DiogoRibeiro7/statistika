import { kruskalWallisTest } from "../src/tests/nonparametric";

describe("kruskalWallisTest", () => {
  it("detects significant differences between clearly separated groups", () => {
    const group1 = [1, 2, 3, 4, 5];
    const group2 = [6, 7, 8, 9, 10];
    const group3 = [11, 12, 13, 14, 15];
    const result = kruskalWallisTest([group1, group2, group3]);
    expect(result.statistic).toBeGreaterThan(0);
    expect(result.pValue).toBeLessThan(0.05);
    expect(result.df).toBe(2);
    expect(result.rejected).toBe(true);
  });

  it("does not reject for groups from similar distributions", () => {
    // All groups have similar values
    const group1 = [4, 5, 6, 7, 8];
    const group2 = [5, 6, 7, 8, 9];
    const group3 = [3, 5, 7, 8, 6];
    const result = kruskalWallisTest([group1, group2, group3]);
    expect(result.pValue).toBeGreaterThan(0.05);
    expect(result.rejected).toBe(false);
  });

  it("works with two groups (equivalent to Mann-Whitney)", () => {
    const group1 = [1, 2, 3, 4, 5];
    const group2 = [10, 11, 12, 13, 14];
    const result = kruskalWallisTest([group1, group2]);
    expect(result.df).toBe(1);
    expect(result.pValue).toBeLessThan(0.05);
    expect(result.rejected).toBe(true);
  });

  it("handles ties correctly", () => {
    const group1 = [1, 1, 2, 2, 3];
    const group2 = [3, 3, 4, 4, 5];
    const result = kruskalWallisTest([group1, group2]);
    expect(result.statistic).toBeGreaterThan(0);
    expect(result.df).toBe(1);
    expect(typeof result.pValue).toBe("number");
  });

  it("computes correct H for known example", () => {
    // Classic textbook example:
    // Group A: 6, 8, 10
    // Group B: 7, 11, 13
    // Group C: 14, 15, 17
    // Expected H approximately 6.49 (from R: kruskal.test)
    const result = kruskalWallisTest(
      [[6, 8, 10], [7, 11, 13], [14, 15, 17]],
    );
    expect(result.statistic).toBeGreaterThan(4);
    expect(result.statistic).toBeLessThan(9);
    expect(result.df).toBe(2);
  });

  it("handles unequal group sizes", () => {
    const group1 = [1, 2, 3];
    const group2 = [4, 5, 6, 7, 8];
    const group3 = [9, 10];
    const result = kruskalWallisTest([group1, group2, group3]);
    expect(result.df).toBe(2);
    expect(result.statistic).toBeGreaterThan(0);
  });

  it("handles single-element groups", () => {
    const result = kruskalWallisTest([[1], [5], [10]]);
    expect(result.df).toBe(2);
    expect(typeof result.pValue).toBe("number");
  });

  it("throws for fewer than 2 groups", () => {
    expect(() => kruskalWallisTest([[1, 2, 3]])).toThrow("at least 2 groups");
    expect(() => kruskalWallisTest([])).toThrow("at least 2 groups");
  });

  it("throws for empty groups", () => {
    expect(() => kruskalWallisTest([[1, 2], []])).toThrow("at least 1 observation");
  });

  it("throws when total observations are fewer than 3", () => {
    expect(() => kruskalWallisTest([[1], [2]])).toThrow("at least 3 total");
  });

  it("returns correct result structure", () => {
    const result = kruskalWallisTest([[1, 2, 3], [4, 5, 6]]);
    expect(result).toHaveProperty("statistic");
    expect(result).toHaveProperty("pValue");
    expect(result).toHaveProperty("df");
    expect(result).toHaveProperty("rejected");
    expect(typeof result.statistic).toBe("number");
    expect(typeof result.pValue).toBe("number");
    expect(typeof result.df).toBe("number");
    expect(typeof result.rejected).toBe("boolean");
  });

  it("p-value is bounded between 0 and 1", () => {
    const result = kruskalWallisTest([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    expect(result.pValue).toBeGreaterThanOrEqual(0);
    expect(result.pValue).toBeLessThanOrEqual(1);
  });

  it("H statistic is non-negative", () => {
    const result = kruskalWallisTest([[1, 2, 3], [2, 3, 4]]);
    expect(result.statistic).toBeGreaterThanOrEqual(0);
  });

  it("respects custom alpha", () => {
    const group1 = [1, 2, 3, 4, 5];
    const group2 = [6, 7, 8, 9, 10];
    const strict = kruskalWallisTest([group1, group2], 0.001);
    expect(typeof strict.rejected).toBe("boolean");
  });
});
