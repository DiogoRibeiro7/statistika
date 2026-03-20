import { Geometric } from "../../../src/distributions/discrete/geometric";

describe("Geometric distribution", () => {
  const g = new Geometric(0.5);

  it("has correct mean and variance", () => {
    expect(g.mean()).toBe(1);
    expect(g.variance()).toBe(2);
  });

  it("pmf(0) = p", () => {
    expect(g.pmf(0)).toBe(0.5);
  });

  it("cdf(0) = p", () => {
    expect(g.cdf(0)).toBe(0.5);
  });

  it("pmf decays geometrically", () => {
    expect(g.pmf(1)).toBeCloseTo(0.25, 8);
    expect(g.pmf(2)).toBeCloseTo(0.125, 8);
  });

  it("quantile and cdf are consistent", () => {
    for (const p of [0.3, 0.5, 0.8]) {
      const k = g.quantile(p);
      expect(g.cdf(k)).toBeGreaterThanOrEqual(p);
    }
  });
});
