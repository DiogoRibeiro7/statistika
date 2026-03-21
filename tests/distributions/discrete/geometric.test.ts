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

  it("stdDev is sqrt(variance)", () => {
    expect(g.stdDev()).toBeCloseTo(Math.sqrt(g.variance()), 8);
  });

  it("sf(k) = 1 - cdf(k)", () => {
    expect(g.sf(1)).toBeCloseTo(1 - g.cdf(1), 10);
  });

  it("cdf returns 0 for negative k", () => {
    expect(g.cdf(-1)).toBe(0);
  });

  it("pmf returns 0 for non-integer and negative k", () => {
    expect(g.pmf(-1)).toBe(0);
    expect(g.pmf(1.5)).toBe(0);
  });

  it("quantile edge cases", () => {
    expect(g.quantile(0)).toBe(0);
    expect(g.quantile(1)).toBe(Infinity);
    expect(() => g.quantile(-0.1)).toThrow();
    expect(() => g.quantile(1.1)).toThrow();
  });

  it("sample returns non-negative integers", () => {
    for (let i = 0; i < 50; i++) {
      const s = g.sample();
      expect(s).toBeGreaterThanOrEqual(0);
      expect(Number.isInteger(s)).toBe(true);
    }
  });

  it("sampleN returns correct number of samples", () => {
    expect(g.sampleN(10).length).toBe(10);
  });

  it("throws on invalid p", () => {
    expect(() => new Geometric(0)).toThrow();
    expect(() => new Geometric(1.5)).toThrow();
    expect(() => new Geometric(-0.1)).toThrow();
  });
});
