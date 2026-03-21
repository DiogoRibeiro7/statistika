import { Bernoulli } from "../../../src/distributions/discrete/bernoulli";

describe("Bernoulli distribution", () => {
  const b = new Bernoulli(0.3);

  it("has correct mean and variance", () => {
    expect(b.mean()).toBe(0.3);
    expect(b.variance()).toBeCloseTo(0.21, 8);
  });

  it("pmf sums to 1", () => {
    expect(b.pmf(0) + b.pmf(1)).toBeCloseTo(1, 10);
  });

  it("pmf(0) = 1-p, pmf(1) = p", () => {
    expect(b.pmf(0)).toBeCloseTo(0.7, 10);
    expect(b.pmf(1)).toBeCloseTo(0.3, 10);
  });

  it("cdf is correct", () => {
    expect(b.cdf(-1)).toBe(0);
    expect(b.cdf(0)).toBeCloseTo(0.7, 10);
    expect(b.cdf(0.5)).toBeCloseTo(0.7, 10);
    expect(b.cdf(1)).toBe(1);
  });

  it("quantile is correct", () => {
    expect(b.quantile(0.5)).toBe(0);
    expect(b.quantile(0.8)).toBe(1);
  });

  it("stdDev is sqrt(variance)", () => {
    expect(b.stdDev()).toBeCloseTo(Math.sqrt(0.21), 8);
  });

  it("sf(k) = 1 - cdf(k)", () => {
    expect(b.sf(0)).toBeCloseTo(1 - b.cdf(0), 10);
  });

  it("sample returns 0 or 1", () => {
    for (let i = 0; i < 50; i++) {
      const s = b.sample();
      expect(s === 0 || s === 1).toBe(true);
    }
  });

  it("sampleN returns correct number of samples", () => {
    expect(b.sampleN(10).length).toBe(10);
  });
});
