import { Binomial } from "../../../src/distributions/discrete/binomial";

describe("Binomial distribution", () => {
  const b = new Binomial(10, 0.5);

  it("has correct mean and variance", () => {
    expect(b.mean()).toBe(5);
    expect(b.variance()).toBe(2.5);
  });

  it("pmf is symmetric for p=0.5", () => {
    expect(b.pmf(3)).toBeCloseTo(b.pmf(7), 8);
  });

  it("pmf sums to 1", () => {
    let sum = 0;
    for (let k = 0; k <= 10; k++) sum += b.pmf(k);
    expect(sum).toBeCloseTo(1, 8);
  });

  it("cdf(10) = 1", () => {
    expect(b.cdf(10)).toBeCloseTo(1, 8);
  });

  it("cdf(4) ~ 0.3770", () => {
    expect(b.cdf(4)).toBeCloseTo(0.377, 2);
  });

  it("quantile(0.5) = 5 for symmetric binomial", () => {
    expect(b.quantile(0.5)).toBe(5);
  });

  it("handles edge cases p=0 and p=1", () => {
    const b0 = new Binomial(5, 0);
    expect(b0.pmf(0)).toBe(1);
    const b1 = new Binomial(5, 1);
    expect(b1.pmf(5)).toBe(1);
  });

  it("stdDev is sqrt(variance)", () => {
    expect(b.stdDev()).toBeCloseTo(Math.sqrt(2.5), 8);
  });

  it("sf(k) = 1 - cdf(k)", () => {
    expect(b.sf(5)).toBeCloseTo(1 - b.cdf(5), 10);
  });

  it("cdf returns 0 for negative k", () => {
    expect(b.cdf(-1)).toBe(0);
  });

  it("pmf returns 0 for non-integer and out-of-range k", () => {
    expect(b.pmf(-1)).toBe(0);
    expect(b.pmf(11)).toBe(0);
    expect(b.pmf(1.5)).toBe(0);
  });

  it("quantile edge cases", () => {
    expect(b.quantile(0)).toBe(0);
    expect(b.quantile(1)).toBe(10);
    expect(() => b.quantile(-0.1)).toThrow();
    expect(() => b.quantile(1.1)).toThrow();
  });

  it("sample returns integers in [0, n]", () => {
    for (let i = 0; i < 50; i++) {
      const s = b.sample();
      expect(s).toBeGreaterThanOrEqual(0);
      expect(s).toBeLessThanOrEqual(10);
      expect(Number.isInteger(s)).toBe(true);
    }
  });

  it("sampleN returns correct number of samples", () => {
    expect(b.sampleN(10).length).toBe(10);
  });

  it("throws on invalid parameters", () => {
    expect(() => new Binomial(0, 0.5)).toThrow();
    expect(() => new Binomial(1.5, 0.5)).toThrow();
    expect(() => new Binomial(5, -0.1)).toThrow();
    expect(() => new Binomial(5, 1.1)).toThrow();
  });
});
