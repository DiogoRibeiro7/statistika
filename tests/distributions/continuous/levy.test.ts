import { Levy } from "../../../src/distributions/continuous/levy";

describe("Levy distribution", () => {
  const lev = new Levy(0, 1); // standard Levy

  it("has correct name", () => {
    expect(lev.name).toBe("Levy(0, 1)");
  });

  it("mean is Infinity", () => {
    expect(lev.mean()).toBe(Infinity);
  });

  it("variance is Infinity", () => {
    expect(lev.variance()).toBe(Infinity);
  });

  it("stdDev is Infinity", () => {
    expect(lev.stdDev()).toBe(Infinity);
  });

  it("pdf is 0 for x <= mu", () => {
    expect(lev.pdf(0)).toBe(0);
    expect(lev.pdf(-1)).toBe(0);
  });

  it("pdf at known values", () => {
    // f(1; 0, 1) = sqrt(1/(2*pi)) * exp(-1/2) / 1^(3/2)
    const expected = Math.sqrt(1 / (2 * Math.PI)) * Math.exp(-0.5);
    expect(lev.pdf(1)).toBeCloseTo(expected, 10);
  });

  it("pdf approaches 0 for large x", () => {
    expect(lev.pdf(1000)).toBeLessThan(0.001);
  });

  it("cdf is 0 for x <= mu", () => {
    expect(lev.cdf(0)).toBe(0);
    expect(lev.cdf(-5)).toBe(0);
  });

  it("cdf increases with x", () => {
    expect(lev.cdf(1)).toBeLessThan(lev.cdf(2));
    expect(lev.cdf(2)).toBeLessThan(lev.cdf(10));
  });

  it("cdf approaches 1 for large x", () => {
    expect(lev.cdf(1e6)).toBeGreaterThan(0.999);
  });

  it("cdf and quantile are inverses", () => {
    for (const p of [0.1, 0.25, 0.5, 0.75, 0.9]) {
      expect(lev.cdf(lev.quantile(p))).toBeCloseTo(p, 5);
    }
  });

  it("quantile(0) = mu", () => {
    expect(lev.quantile(0)).toBe(0);
  });

  it("quantile(1) = Infinity", () => {
    expect(lev.quantile(1)).toBe(Infinity);
  });

  it("quantile throws on out-of-range p", () => {
    expect(() => lev.quantile(-0.1)).toThrow();
    expect(() => lev.quantile(1.1)).toThrow();
  });

  it("works with non-standard parameters", () => {
    const lev2 = new Levy(2, 3);
    expect(lev2.name).toBe("Levy(2, 3)");
    expect(lev2.pdf(2)).toBe(0);
    expect(lev2.cdf(2)).toBe(0);
    expect(lev2.pdf(5)).toBeGreaterThan(0);
    expect(lev2.cdf(5)).toBeGreaterThan(0);
    expect(lev2.quantile(0)).toBe(2);
  });

  it("throws on non-positive c", () => {
    expect(() => new Levy(0, 0)).toThrow();
    expect(() => new Levy(0, -1)).toThrow();
  });

  it("sf(x) = 1 - cdf(x)", () => {
    expect(lev.sf(2)).toBeCloseTo(1 - lev.cdf(2), 10);
  });

  it("sample returns values > mu", () => {
    for (let i = 0; i < 50; i++) {
      expect(lev.sample()).toBeGreaterThan(0);
    }
  });

  it("sampleN returns correct number of samples", () => {
    expect(lev.sampleN(10).length).toBe(10);
  });
});
