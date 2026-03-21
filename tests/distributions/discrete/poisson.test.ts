import { Poisson } from "../../../src/distributions/discrete/poisson";

describe("Poisson distribution", () => {
  const p = new Poisson(3);

  it("has correct mean and variance", () => {
    expect(p.mean()).toBe(3);
    expect(p.variance()).toBe(3);
  });

  it("pmf(0) = e^-lambda", () => {
    expect(p.pmf(0)).toBeCloseTo(Math.exp(-3), 8);
  });

  it("pmf sums approximately to 1", () => {
    let sum = 0;
    for (let k = 0; k <= 30; k++) sum += p.pmf(k);
    expect(sum).toBeCloseTo(1, 8);
  });

  it("cdf is non-decreasing", () => {
    let prev = 0;
    for (let k = 0; k <= 20; k++) {
      const curr = p.cdf(k);
      expect(curr).toBeGreaterThanOrEqual(prev);
      prev = curr;
    }
  });

  it("quantile and cdf are consistent", () => {
    expect(p.quantile(0.5)).toBe(3);
  });

  it("quantile edge cases", () => {
    expect(p.quantile(0)).toBe(0);
    expect(p.quantile(1)).toBe(Infinity);
    expect(() => p.quantile(-0.1)).toThrow();
    expect(() => p.quantile(1.1)).toThrow();
  });

  it("pmf returns 0 for non-integer and negative k", () => {
    expect(p.pmf(-1)).toBe(0);
    expect(p.pmf(1.5)).toBe(0);
  });

  it("cdf returns 0 for negative k", () => {
    expect(p.cdf(-1)).toBe(0);
  });

  it("stdDev is sqrt(variance)", () => {
    expect(p.stdDev()).toBeCloseTo(Math.sqrt(3), 8);
  });

  it("sf(k) = 1 - cdf(k)", () => {
    expect(p.sf(2)).toBeCloseTo(1 - p.cdf(2), 10);
    expect(p.sf(5)).toBeCloseTo(1 - p.cdf(5), 10);
  });

  it("sample returns non-negative integers", () => {
    for (let i = 0; i < 50; i++) {
      const s = p.sample();
      expect(s).toBeGreaterThanOrEqual(0);
      expect(Number.isInteger(s)).toBe(true);
    }
  });

  it("sampleN returns correct number of samples", () => {
    expect(p.sampleN(10).length).toBe(10);
    expect(p.sampleN(100).length).toBe(100);
  });

  it("sample mean converges to theoretical mean", () => {
    const samples = p.sampleN(5000);
    const sampleMean = samples.reduce((a, b) => a + b, 0) / samples.length;
    expect(sampleMean).toBeCloseTo(p.mean(), 0);
  });

  it("sample works for large lambda (inverse transform path)", () => {
    const largeLambda = new Poisson(50);
    const s = largeLambda.sample();
    expect(s).toBeGreaterThanOrEqual(0);
    expect(Number.isInteger(s)).toBe(true);
  });

  it("throws on non-positive lambda", () => {
    expect(() => new Poisson(0)).toThrow();
    expect(() => new Poisson(-1)).toThrow();
  });
});
