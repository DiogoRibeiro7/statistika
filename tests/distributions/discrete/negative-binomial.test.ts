import { NegativeBinomial } from "../../../src/distributions/discrete/negative-binomial";

describe("Negative Binomial distribution", () => {
  const nb = new NegativeBinomial(5, 0.4);

  it("has correct name", () => {
    expect(nb.name).toBe("NegBin(5, 0.4)");
  });

  it("mean = r(1-p)/p", () => {
    expect(nb.mean()).toBeCloseTo(5 * 0.6 / 0.4, 10);
  });

  it("variance = r(1-p)/p^2", () => {
    expect(nb.variance()).toBeCloseTo(5 * 0.6 / 0.16, 10);
  });

  it("pmf sums to ~1", () => {
    let sum = 0;
    for (let k = 0; k < 100; k++) sum += nb.pmf(k);
    expect(sum).toBeCloseTo(1, 4);
  });

  it("pmf(k) = 0 for negative or non-integer k", () => {
    expect(nb.pmf(-1)).toBe(0);
    expect(nb.pmf(1.5)).toBe(0);
  });

  it("cdf is consistent with pmf sum", () => {
    let sum = 0;
    for (let k = 0; k <= 10; k++) {
      sum += nb.pmf(k);
      expect(nb.cdf(k)).toBeCloseTo(sum, 8);
    }
  });

  it("quantile inverts cdf", () => {
    for (const p of [0.1, 0.5, 0.9]) {
      const k = nb.quantile(p);
      expect(nb.cdf(k)).toBeGreaterThanOrEqual(p);
      if (k > 0) expect(nb.cdf(k - 1)).toBeLessThan(p);
    }
  });

  it("special case: Geometric when r=1", () => {
    const geo = new NegativeBinomial(1, 0.5);
    expect(geo.mean()).toBeCloseTo(1, 10);
    expect(geo.pmf(0)).toBeCloseTo(0.5, 10);
    expect(geo.pmf(1)).toBeCloseTo(0.25, 10);
  });

  it("throws on invalid parameters", () => {
    expect(() => new NegativeBinomial(0, 0.5)).toThrow();
    expect(() => new NegativeBinomial(3, 0)).toThrow();
    expect(() => new NegativeBinomial(3, 1.5)).toThrow();
  });
});
