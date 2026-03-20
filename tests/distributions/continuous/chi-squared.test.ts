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
});
