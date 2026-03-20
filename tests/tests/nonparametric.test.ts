import { mannWhitneyU, wilcoxonSignedRank } from "../../src/tests";

describe("mannWhitneyU", () => {
  it("detects a significant difference between shifted distributions", () => {
    const data1 = [1, 2, 3, 4, 5, 6, 7, 8];
    const data2 = [9, 10, 11, 12, 13, 14, 15, 16];
    const result = mannWhitneyU(data1, data2);

    expect(result.statistic).toBe(0);
    expect(result.pValue).toBeLessThan(0.01);
    expect(result.rejected).toBe(true);
  });

  it("fails to reject for similar distributions", () => {
    const data1 = [1, 3, 5, 7, 9, 11];
    const data2 = [2, 4, 6, 8, 10, 12];
    const result = mannWhitneyU(data1, data2);

    expect(result.pValue).toBeGreaterThan(0.05);
    expect(result.rejected).toBe(false);
  });

  it("handles ties correctly", () => {
    const data1 = [1, 2, 2, 3, 4];
    const data2 = [2, 3, 3, 4, 5];
    const result = mannWhitneyU(data1, data2);

    expect(result.statistic).toBeGreaterThanOrEqual(0);
    expect(result.pValue).toBeGreaterThanOrEqual(0);
    expect(result.pValue).toBeLessThanOrEqual(1);
  });

  it("returns degreesOfFreedom = 0", () => {
    const result = mannWhitneyU([1, 2, 3], [4, 5, 6]);
    expect(result.degreesOfFreedom).toBe(0);
  });

  it("throws when a sample is empty", () => {
    expect(() => mannWhitneyU([], [1, 2, 3])).toThrow("at least 1");
    expect(() => mannWhitneyU([1, 2], [])).toThrow("at least 1");
  });

  it("handles equal samples", () => {
    const data = [3, 5, 7, 9, 11];
    const result = mannWhitneyU(data, [...data]);

    expect(result.pValue).toBeGreaterThan(0.05);
    expect(result.rejected).toBe(false);
  });

  it("works with different sample sizes", () => {
    const data1 = [1, 2, 3];
    const data2 = [10, 11, 12, 13, 14, 15];
    const result = mannWhitneyU(data1, data2);

    expect(result.pValue).toBeLessThan(0.05);
    expect(result.rejected).toBe(true);
  });
});

describe("wilcoxonSignedRank", () => {
  it("detects a significant systematic shift", () => {
    const before = [10, 12, 14, 16, 18, 20, 22, 24];
    const after = [15, 17, 19, 21, 23, 25, 27, 29];
    const result = wilcoxonSignedRank(before, after);

    expect(result.statistic).toBe(0);
    expect(result.pValue).toBeLessThan(0.05);
    expect(result.rejected).toBe(true);
  });

  it("fails to reject for no systematic difference", () => {
    const data1 = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28];
    const data2 = [11, 11, 15, 15, 19, 19, 23, 23, 27, 27];
    const result = wilcoxonSignedRank(data1, data2);

    expect(result.pValue).toBeGreaterThan(0.05);
    expect(result.rejected).toBe(false);
  });

  it("handles zero differences gracefully", () => {
    const data = [1, 2, 3, 4, 5, 6, 7, 8];
    const result = wilcoxonSignedRank(data, [...data]);

    expect(result.statistic).toBe(0);
    expect(result.pValue).toBe(1);
    expect(result.rejected).toBe(false);
  });

  it("returns degreesOfFreedom = 0", () => {
    const result = wilcoxonSignedRank([1, 2, 3, 4], [5, 6, 7, 8]);
    expect(result.degreesOfFreedom).toBe(0);
  });

  it("throws for mismatched lengths", () => {
    expect(() => wilcoxonSignedRank([1, 2, 3], [1, 2])).toThrow("equal length");
  });

  it("throws for fewer than 2 pairs", () => {
    expect(() => wilcoxonSignedRank([1], [2])).toThrow("at least 2");
  });

  it("handles ties in absolute differences", () => {
    const data1 = [10, 20, 30, 40, 50];
    const data2 = [8, 18, 28, 38, 48]; // all diffs = 2
    const result = wilcoxonSignedRank(data1, data2);

    expect(result.statistic).toBe(0);
    expect(result.pValue).toBeLessThan(0.05);
  });
});
