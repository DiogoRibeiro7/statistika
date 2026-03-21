import { Hypergeometric } from "../../../src/distributions/discrete/hypergeometric";

describe("Hypergeometric distribution", () => {
  // Classic urn: 50 balls, 10 red, draw 5
  const h = new Hypergeometric(50, 10, 5);

  it("has correct name", () => {
    expect(h.name).toBe("Hypergeometric(50, 10, 5)");
  });

  it("mean = nK/N", () => {
    expect(h.mean()).toBeCloseTo(5 * 10 / 50, 10);
  });

  it("variance matches formula", () => {
    const expected = (5 * 10 * 40 * 45) / (50 * 50 * 49);
    expect(h.variance()).toBeCloseTo(expected, 8);
  });

  it("pmf sums to 1", () => {
    let sum = 0;
    for (let k = 0; k <= 5; k++) sum += h.pmf(k);
    expect(sum).toBeCloseTo(1, 10);
  });

  it("pmf = 0 outside valid range", () => {
    expect(h.pmf(-1)).toBe(0);
    expect(h.pmf(6)).toBe(0);
    expect(h.pmf(1.5)).toBe(0);
  });

  it("cdf is consistent with pmf sum", () => {
    let sum = 0;
    for (let k = 0; k <= 5; k++) {
      sum += h.pmf(k);
      expect(h.cdf(k)).toBeCloseTo(sum, 8);
    }
  });

  it("quantile inverts cdf", () => {
    for (const p of [0.1, 0.5, 0.9]) {
      const k = h.quantile(p);
      expect(h.cdf(k)).toBeGreaterThanOrEqual(p);
    }
  });

  it("edge case: all drawn are successes", () => {
    // N=5, K=5, n=5: must get 5
    const h2 = new Hypergeometric(5, 5, 5);
    expect(h2.pmf(5)).toBeCloseTo(1, 10);
    expect(h2.mean()).toBeCloseTo(5, 10);
  });

  it("edge case: no successes possible", () => {
    // N=10, K=0, n=5: must get 0
    const h3 = new Hypergeometric(10, 0, 5);
    expect(h3.pmf(0)).toBeCloseTo(1, 10);
  });

  it("throws on invalid parameters", () => {
    expect(() => new Hypergeometric(-1, 5, 3)).toThrow();
    expect(() => new Hypergeometric(10, 15, 3)).toThrow();
    expect(() => new Hypergeometric(10, 5, 15)).toThrow();
  });

  it("stdDev is sqrt(variance)", () => {
    expect(h.stdDev()).toBeCloseTo(Math.sqrt(h.variance()), 8);
  });

  it("sf(k) = 1 - cdf(k)", () => {
    expect(h.sf(1)).toBeCloseTo(1 - h.cdf(1), 10);
  });

  it("cdf returns 0 below lower bound", () => {
    expect(h.cdf(-1)).toBe(0);
  });

  it("quantile edge cases", () => {
    expect(h.quantile(0)).toBe(0);
    expect(h.quantile(1)).toBe(5);
    expect(() => h.quantile(-0.1)).toThrow();
    expect(() => h.quantile(1.1)).toThrow();
  });

  it("sample returns integers in valid range", () => {
    for (let i = 0; i < 50; i++) {
      const s = h.sample();
      expect(s).toBeGreaterThanOrEqual(0);
      expect(s).toBeLessThanOrEqual(5);
      expect(Number.isInteger(s)).toBe(true);
    }
  });

  it("sampleN returns correct number of samples", () => {
    expect(h.sampleN(10).length).toBe(10);
  });

  it("throws on non-integer parameters", () => {
    expect(() => new Hypergeometric(10.5, 5, 3)).toThrow();
    expect(() => new Hypergeometric(10, 5.5, 3)).toThrow();
    expect(() => new Hypergeometric(10, 5, 3.5)).toThrow();
  });
});
