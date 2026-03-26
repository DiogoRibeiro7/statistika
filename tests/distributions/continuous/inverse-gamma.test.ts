import { InverseGamma } from "../../../src/distributions/continuous/inverse-gamma";

describe("InverseGamma distribution", () => {
  const ig = new InverseGamma(3, 2); // alpha=3, beta=2

  it("has correct name", () => {
    expect(ig.name).toBe("InverseGamma(3, 2)");
  });

  it("throws on non-positive alpha", () => {
    expect(() => new InverseGamma(0, 1)).toThrow();
    expect(() => new InverseGamma(-1, 1)).toThrow();
  });

  it("throws on non-positive beta", () => {
    expect(() => new InverseGamma(1, 0)).toThrow();
    expect(() => new InverseGamma(1, -2)).toThrow();
  });

  it("mean is beta/(alpha-1) for alpha > 1", () => {
    // alpha=3, beta=2 => mean = 2/(3-1) = 1
    expect(ig.mean()).toBeCloseTo(1, 10);
  });

  it("mean is Infinity for alpha <= 1", () => {
    const ig1 = new InverseGamma(1, 1);
    expect(ig1.mean()).toBe(Infinity);
    const ig05 = new InverseGamma(0.5, 1);
    expect(ig05.mean()).toBe(Infinity);
  });

  it("variance is beta^2/((alpha-1)^2*(alpha-2)) for alpha > 2", () => {
    // alpha=3, beta=2 => var = 4 / (4 * 1) = 1
    expect(ig.variance()).toBeCloseTo(1, 10);
  });

  it("variance is Infinity for alpha <= 2", () => {
    const ig2 = new InverseGamma(2, 1);
    expect(ig2.variance()).toBe(Infinity);
    const ig15 = new InverseGamma(1.5, 1);
    expect(ig15.variance()).toBe(Infinity);
  });

  it("pdf is 0 for x <= 0", () => {
    expect(ig.pdf(0)).toBe(0);
    expect(ig.pdf(-1)).toBe(0);
  });

  it("pdf at known values", () => {
    // For InverseGamma(3, 2), pdf(1) = (2^3/Gamma(3)) * 1^(-4) * exp(-2)
    // = 8/2 * 1 * exp(-2) = 4 * exp(-2)
    const expected = 4 * Math.exp(-2);
    expect(ig.pdf(1)).toBeCloseTo(expected, 10);
  });

  it("pdf is positive for x > 0", () => {
    expect(ig.pdf(0.1)).toBeGreaterThan(0);
    expect(ig.pdf(1)).toBeGreaterThan(0);
    expect(ig.pdf(10)).toBeGreaterThan(0);
  });

  it("pdf approaches 0 for large x", () => {
    expect(ig.pdf(100)).toBeLessThan(0.001);
  });

  it("cdf is 0 for x <= 0", () => {
    expect(ig.cdf(0)).toBe(0);
    expect(ig.cdf(-5)).toBe(0);
  });

  it("cdf increases with x", () => {
    expect(ig.cdf(0.5)).toBeLessThan(ig.cdf(1));
    expect(ig.cdf(1)).toBeLessThan(ig.cdf(2));
    expect(ig.cdf(2)).toBeLessThan(ig.cdf(5));
  });

  it("cdf approaches 1 for large x", () => {
    expect(ig.cdf(1e6)).toBeGreaterThan(0.999);
  });

  it("cdf and quantile are inverses", () => {
    for (const p of [0.1, 0.25, 0.5, 0.75, 0.9]) {
      expect(ig.cdf(ig.quantile(p))).toBeCloseTo(p, 4);
    }
  });

  it("quantile(0) = 0", () => {
    expect(ig.quantile(0)).toBe(0);
  });

  it("quantile(1) = Infinity", () => {
    expect(ig.quantile(1)).toBe(Infinity);
  });

  it("quantile throws on out-of-range p", () => {
    expect(() => ig.quantile(-0.1)).toThrow();
    expect(() => ig.quantile(1.1)).toThrow();
  });

  it("sf(x) = 1 - cdf(x)", () => {
    expect(ig.sf(1)).toBeCloseTo(1 - ig.cdf(1), 10);
    expect(ig.sf(2)).toBeCloseTo(1 - ig.cdf(2), 10);
  });

  it("sample returns positive values", () => {
    for (let i = 0; i < 50; i++) {
      expect(ig.sample()).toBeGreaterThan(0);
    }
  });

  it("sampleN returns correct number of samples", () => {
    expect(ig.sampleN(10).length).toBe(10);
  });

  it("sample mean converges to theoretical mean", () => {
    const samples = ig.sampleN(5000);
    const sampleMean = samples.reduce((a, b) => a + b, 0) / samples.length;
    // mean = 1, allow generous tolerance for sampling
    expect(sampleMean).toBeCloseTo(1, 0);
  });

  it("works with non-standard parameters", () => {
    const ig2 = new InverseGamma(5, 10);
    expect(ig2.name).toBe("InverseGamma(5, 10)");
    // mean = 10/(5-1) = 2.5
    expect(ig2.mean()).toBeCloseTo(2.5, 10);
    // var = 100 / (16*3) = 100/48
    expect(ig2.variance()).toBeCloseTo(100 / 48, 10);
    expect(ig2.pdf(1)).toBeGreaterThan(0);
    expect(ig2.cdf(1)).toBeGreaterThan(0);
  });

  it("handles alpha close to boundary values", () => {
    const igSmall = new InverseGamma(0.01, 1);
    expect(igSmall.mean()).toBe(Infinity);
    expect(igSmall.variance()).toBe(Infinity);
    expect(igSmall.pdf(1)).toBeGreaterThan(0);
  });
});
