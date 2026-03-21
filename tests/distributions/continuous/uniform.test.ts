import { Uniform } from "../../../src/distributions/continuous/uniform";

describe("Uniform distribution", () => {
  const u = new Uniform(0, 1);
  const u2 = new Uniform(2, 8);

  it("has correct mean and variance", () => {
    expect(u.mean()).toBe(0.5);
    expect(u.variance()).toBeCloseTo(1 / 12, 8);
    expect(u2.mean()).toBe(5);
    expect(u2.variance()).toBeCloseTo(3, 8);
  });

  it("pdf is constant on [a,b]", () => {
    expect(u.pdf(0.5)).toBe(1);
    expect(u.pdf(-1)).toBe(0);
    expect(u2.pdf(5)).toBeCloseTo(1 / 6, 8);
  });

  it("cdf is linear on [a,b]", () => {
    expect(u.cdf(0)).toBe(0);
    expect(u.cdf(0.5)).toBe(0.5);
    expect(u.cdf(1)).toBe(1);
  });

  it("quantile is inverse of cdf", () => {
    for (const p of [0, 0.25, 0.5, 0.75, 1]) {
      expect(u.cdf(u.quantile(p))).toBeCloseTo(p, 8);
    }
  });

  it("stdDev is sqrt(variance)", () => {
    expect(u.stdDev()).toBeCloseTo(Math.sqrt(1 / 12), 8);
  });

  it("sf(x) = 1 - cdf(x)", () => {
    expect(u.sf(0.3)).toBeCloseTo(0.7, 10);
  });

  it("cdf returns 0 below a and 1 above b", () => {
    expect(u.cdf(-1)).toBe(0);
    expect(u.cdf(2)).toBe(1);
  });

  it("quantile throws on out-of-range p", () => {
    expect(() => u.quantile(-0.1)).toThrow();
    expect(() => u.quantile(1.1)).toThrow();
  });

  it("sample returns values in [a, b]", () => {
    for (let i = 0; i < 50; i++) {
      const s = u.sample();
      expect(s).toBeGreaterThanOrEqual(0);
      expect(s).toBeLessThanOrEqual(1);
    }
  });

  it("sampleN returns correct number of samples", () => {
    expect(u.sampleN(10).length).toBe(10);
  });

  it("throws when a >= b", () => {
    expect(() => new Uniform(1, 1)).toThrow();
    expect(() => new Uniform(2, 1)).toThrow();
  });
});
