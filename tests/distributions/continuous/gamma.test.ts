import { GammaDistribution } from "../../../src/distributions/continuous/gamma";

describe("Gamma distribution", () => {
  const g = new GammaDistribution(2, 1);

  it("has correct mean and variance", () => {
    expect(g.mean()).toBe(2);
    expect(g.variance()).toBe(2);
  });

  it("pdf(0) = 0 for shape > 1", () => {
    expect(g.pdf(0)).toBe(0);
  });

  it("cdf(0) = 0", () => {
    expect(g.cdf(0)).toBe(0);
  });

  it("cdf matches known values", () => {
    // Gamma(1, 1) = Exponential(1), so CDF at 1 = 1 - e^-1
    const exp = new GammaDistribution(1, 1);
    expect(exp.cdf(1)).toBeCloseTo(1 - Math.exp(-1), 6);
  });

  it("quantile and cdf are inverses", () => {
    for (const p of [0.1, 0.5, 0.9]) {
      expect(g.cdf(g.quantile(p))).toBeCloseTo(p, 4);
    }
  });

  it("throws on non-positive parameters", () => {
    expect(() => new GammaDistribution(0, 1)).toThrow();
    expect(() => new GammaDistribution(1, 0)).toThrow();
  });
});
