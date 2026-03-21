import { Exponential } from "../../../src/distributions/continuous/exponential";

describe("Exponential distribution", () => {
  const exp1 = new Exponential(1);
  const exp2 = new Exponential(2);

  it("has correct mean and variance", () => {
    expect(exp1.mean()).toBe(1);
    expect(exp1.variance()).toBe(1);
    expect(exp2.mean()).toBe(0.5);
    expect(exp2.variance()).toBe(0.25);
  });

  it("pdf(0) = lambda", () => {
    expect(exp1.pdf(0)).toBe(1);
    expect(exp2.pdf(0)).toBe(2);
  });

  it("cdf(0) = 0", () => {
    expect(exp1.cdf(0)).toBe(0);
  });

  it("cdf(1) = 1 - e^-1 for lambda=1", () => {
    expect(exp1.cdf(1)).toBeCloseTo(1 - Math.exp(-1), 8);
  });

  it("quantile and cdf are inverses", () => {
    for (const p of [0.1, 0.5, 0.9]) {
      expect(exp1.cdf(exp1.quantile(p))).toBeCloseTo(p, 8);
    }
  });

  it("sf = 1 - cdf", () => {
    expect(exp1.sf(1)).toBeCloseTo(Math.exp(-1), 8);
  });

  it("stdDev is sqrt(variance)", () => {
    expect(exp1.stdDev()).toBeCloseTo(1, 8);
    expect(exp2.stdDev()).toBeCloseTo(0.5, 8);
  });

  it("pdf is 0 for negative x", () => {
    expect(exp1.pdf(-1)).toBe(0);
  });

  it("cdf is 0 for negative x", () => {
    expect(exp1.cdf(-1)).toBe(0);
  });

  it("quantile edge cases", () => {
    expect(exp1.quantile(0)).toBeCloseTo(0, 10);
    expect(exp1.quantile(1)).toBe(Infinity);
    expect(() => exp1.quantile(-0.1)).toThrow();
    expect(() => exp1.quantile(1.1)).toThrow();
  });

  it("sample returns non-negative values", () => {
    for (let i = 0; i < 50; i++) {
      expect(exp1.sample()).toBeGreaterThanOrEqual(0);
    }
  });

  it("sampleN returns correct number of samples", () => {
    expect(exp1.sampleN(10).length).toBe(10);
  });

  it("throws on non-positive lambda", () => {
    expect(() => new Exponential(0)).toThrow();
    expect(() => new Exponential(-1)).toThrow();
  });
});
