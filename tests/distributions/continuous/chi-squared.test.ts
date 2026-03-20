import { ChiSquared } from "../../../src/distributions/continuous/chi-squared";

describe("Chi-Squared distribution", () => {
  const chi2 = new ChiSquared(3);

  it("has correct mean and variance", () => {
    expect(chi2.mean()).toBe(3);
    expect(chi2.variance()).toBe(6);
  });

  it("cdf(0) = 0", () => {
    expect(chi2.cdf(0)).toBe(0);
  });

  it("quantile and cdf are inverses", () => {
    for (const p of [0.1, 0.5, 0.9]) {
      expect(chi2.cdf(chi2.quantile(p))).toBeCloseTo(p, 4);
    }
  });

  it("Chi-Squared(2) CDF matches 1-exp(-x/2)", () => {
    const chi2_2 = new ChiSquared(2);
    expect(chi2_2.cdf(2)).toBeCloseTo(1 - Math.exp(-1), 4);
  });

  it("stdDev is sqrt(variance)", () => {
    expect(chi2.stdDev()).toBeCloseTo(Math.sqrt(6), 8);
  });

  it("sf(x) = 1 - cdf(x)", () => {
    expect(chi2.sf(3)).toBeCloseTo(1 - chi2.cdf(3), 10);
  });

  it("pdf is non-negative", () => {
    for (const x of [0.5, 1, 3, 5, 10]) {
      expect(chi2.pdf(x)).toBeGreaterThanOrEqual(0);
    }
  });

  it("sample returns positive values", () => {
    for (let i = 0; i < 50; i++) {
      expect(chi2.sample()).toBeGreaterThan(0);
    }
  });

  it("sampleN returns correct number of samples", () => {
    expect(chi2.sampleN(10).length).toBe(10);
  });

  it("throws on invalid k", () => {
    expect(() => new ChiSquared(0)).toThrow();
    expect(() => new ChiSquared(-1)).toThrow();
    expect(() => new ChiSquared(1.5)).toThrow();
  });
});
