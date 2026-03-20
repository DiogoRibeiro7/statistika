import { Cauchy } from "../../../src/distributions/continuous/cauchy";

describe("Cauchy distribution", () => {
  const c = new Cauchy(0, 1); // standard Cauchy

  it("has correct name", () => {
    expect(c.name).toBe("Cauchy(0, 1)");
  });

  it("mean and variance are NaN (undefined)", () => {
    expect(c.mean()).toBeNaN();
    expect(c.variance()).toBeNaN();
  });

  it("pdf at x0 = 1 / (pi * gamma)", () => {
    expect(c.pdf(0)).toBeCloseTo(1 / Math.PI, 10);
  });

  it("pdf is symmetric about x0", () => {
    expect(c.pdf(2)).toBeCloseTo(c.pdf(-2), 10);
    expect(c.pdf(5)).toBeCloseTo(c.pdf(-5), 10);
  });

  it("cdf(x0) = 0.5", () => {
    expect(c.cdf(0)).toBeCloseTo(0.5, 10);
  });

  it("cdf and quantile are inverses", () => {
    for (const p of [0.1, 0.25, 0.5, 0.75, 0.9]) {
      expect(c.cdf(c.quantile(p))).toBeCloseTo(p, 10);
    }
  });

  it("quantile(0) = -Inf, quantile(1) = +Inf", () => {
    expect(c.quantile(0)).toBe(-Infinity);
    expect(c.quantile(1)).toBe(Infinity);
  });

  it("works with non-standard parameters", () => {
    const c2 = new Cauchy(3, 2);
    expect(c2.cdf(3)).toBeCloseTo(0.5, 10);
    expect(c2.pdf(3)).toBeCloseTo(1 / (Math.PI * 2), 10);
  });

  it("throws on non-positive gamma", () => {
    expect(() => new Cauchy(0, 0)).toThrow();
    expect(() => new Cauchy(0, -1)).toThrow();
  });

  it("stdDev is NaN (undefined)", () => {
    expect(c.stdDev()).toBeNaN();
  });

  it("sf(x) = 1 - cdf(x)", () => {
    expect(c.sf(1)).toBeCloseTo(1 - c.cdf(1), 10);
  });

  it("quantile throws on out-of-range p", () => {
    expect(() => c.quantile(-0.1)).toThrow();
    expect(() => c.quantile(1.1)).toThrow();
  });

  it("sample returns finite values", () => {
    for (let i = 0; i < 50; i++) {
      expect(isFinite(c.sample())).toBe(true);
    }
  });

  it("sampleN returns correct number of samples", () => {
    expect(c.sampleN(10).length).toBe(10);
  });
});
