import { oneWayAnova } from "../../src/tests/anova";

describe("one-way ANOVA", () => {
  it("does not reject when group means are similar", () => {
    const g1 = [5.1, 4.9, 5.0, 5.2, 4.8];
    const g2 = [5.0, 5.1, 4.9, 5.0, 5.2];
    const g3 = [4.9, 5.0, 5.1, 4.8, 5.2];
    const result = oneWayAnova([g1, g2, g3]);
    expect(result.rejected).toBe(false);
    expect(result.pValue).toBeGreaterThan(0.05);
    expect(result.dfBetween).toBe(2);
    expect(result.dfWithin).toBe(12);
  });

  it("rejects when group means differ significantly", () => {
    const g1 = [10, 11, 12, 10, 11];
    const g2 = [20, 21, 22, 20, 21];
    const g3 = [30, 31, 32, 30, 31];
    const result = oneWayAnova([g1, g2, g3]);
    expect(result.rejected).toBe(true);
    expect(result.pValue).toBeLessThan(0.001);
    expect(result.fStatistic).toBeGreaterThan(10);
  });

  it("SS between + SS within = SS total", () => {
    const groups = [
      [3, 5, 7],
      [10, 12, 14],
      [20, 22, 24],
    ];
    const result = oneWayAnova(groups);

    // Compute SS total manually
    const all = groups.flat();
    const grandMean = all.reduce((a, b) => a + b, 0) / all.length;
    const ssTotal = all.reduce((acc, v) => acc + (v - grandMean) ** 2, 0);

    expect(result.ssBetween + result.ssWithin).toBeCloseTo(ssTotal, 8);
  });

  it("F = MSbetween / MSwithin", () => {
    const groups = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    const result = oneWayAnova(groups);
    expect(result.fStatistic).toBeCloseTo(result.msBetween / result.msWithin, 10);
  });

  it("throws for fewer than 2 groups", () => {
    expect(() => oneWayAnova([[1, 2, 3]])).toThrow();
  });

  it("throws for empty group", () => {
    expect(() => oneWayAnova([[1, 2], []])).toThrow();
  });
});
