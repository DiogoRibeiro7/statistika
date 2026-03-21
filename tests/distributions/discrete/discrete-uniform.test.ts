import { DiscreteUniform } from "../../../src/distributions/discrete/discrete-uniform";

describe("DiscreteUniform distribution", () => {
  const du = new DiscreteUniform(1, 6);

  it("has correct mean", () => {
    expect(du.mean()).toBe(3.5);
  });

  it("has correct variance", () => {
    expect(du.variance()).toBeCloseTo(35 / 12, 8);
  });

  it("pmf = 1/6 for k in [1,6]", () => {
    for (let k = 1; k <= 6; k++) {
      expect(du.pmf(k)).toBeCloseTo(1 / 6, 10);
    }
  });

  it("pmf = 0 outside [1,6]", () => {
    expect(du.pmf(0)).toBe(0);
    expect(du.pmf(7)).toBe(0);
  });

  it("cdf(6) = 1", () => {
    expect(du.cdf(6)).toBe(1);
  });

  it("cdf(3) = 3/6", () => {
    expect(du.cdf(3)).toBeCloseTo(0.5, 8);
  });

  it("cdf returns 0 for values below a", () => {
    expect(du.cdf(0)).toBe(0);
  });

  it("stdDev is sqrt(variance)", () => {
    expect(du.stdDev()).toBeCloseTo(Math.sqrt(du.variance()), 8);
  });

  it("sf(k) = 1 - cdf(k)", () => {
    expect(du.sf(3)).toBeCloseTo(1 - du.cdf(3), 10);
  });

  it("quantile edge cases", () => {
    expect(du.quantile(0)).toBe(1);
    expect(du.quantile(1)).toBe(6);
    expect(() => du.quantile(-0.1)).toThrow();
    expect(() => du.quantile(1.1)).toThrow();
  });

  it("quantile returns values in [a, b]", () => {
    for (const p of [0.1, 0.25, 0.5, 0.75, 0.9]) {
      const k = du.quantile(p);
      expect(k).toBeGreaterThanOrEqual(1);
      expect(k).toBeLessThanOrEqual(6);
      expect(Number.isInteger(k)).toBe(true);
    }
  });

  it("sample returns values in [a, b]", () => {
    for (let i = 0; i < 50; i++) {
      const s = du.sample();
      expect(s).toBeGreaterThanOrEqual(1);
      expect(s).toBeLessThanOrEqual(6);
      expect(Number.isInteger(s)).toBe(true);
    }
  });

  it("sampleN returns correct number of samples", () => {
    expect(du.sampleN(10).length).toBe(10);
  });

  it("throws on non-integer parameters", () => {
    expect(() => new DiscreteUniform(1.5, 6)).toThrow();
    expect(() => new DiscreteUniform(1, 6.5)).toThrow();
  });

  it("throws when a >= b", () => {
    expect(() => new DiscreteUniform(6, 6)).toThrow();
    expect(() => new DiscreteUniform(7, 3)).toThrow();
  });

  it("pmf returns 0 for non-integer k", () => {
    expect(du.pmf(1.5)).toBe(0);
  });
});
