import { Frechet } from "../../../src/distributions/continuous/frechet";

describe("Fréchet distribution", () => {
  const f = new Frechet(2, 1, 0);

  it("pdf is 0 for x <= m", () => {
    expect(f.pdf(0)).toBe(0);
    expect(f.pdf(-1)).toBe(0);
  });

  it("cdf is 0 for x <= m", () => {
    expect(f.cdf(0)).toBe(0);
    expect(f.cdf(-1)).toBe(0);
  });

  it("cdf(1) = exp(-1)", () => {
    expect(f.cdf(1)).toBeCloseTo(Math.exp(-1), 8);
  });

  it("mean is Infinity for alpha <= 1", () => {
    const f1 = new Frechet(1, 1, 0);
    expect(f1.mean()).toBe(Infinity);
  });

  it("variance is Infinity for alpha <= 2", () => {
    expect(f.variance()).toBe(Infinity);
    const f3 = new Frechet(3, 1, 0);
    expect(f3.variance()).toBeGreaterThan(0);
    expect(isFinite(f3.variance())).toBe(true);
  });

  it("quantile and cdf are inverses", () => {
    for (const p of [0.1, 0.5, 0.9]) {
      expect(f.cdf(f.quantile(p))).toBeCloseTo(p, 8);
    }
  });

  it("quantile(0) = m", () => {
    expect(f.quantile(0)).toBe(0);
  });

  it("throws on non-positive alpha", () => {
    expect(() => new Frechet(0, 1, 0)).toThrow();
  });

  it("throws on non-positive s", () => {
    expect(() => new Frechet(2, 0, 0)).toThrow();
  });

  it("mean for alpha > 1", () => {
    const f3 = new Frechet(3, 1, 0);
    expect(f3.mean()).toBeGreaterThan(0);
    expect(isFinite(f3.mean())).toBe(true);
  });

  it("stdDev is sqrt(variance) for alpha > 2", () => {
    const f3 = new Frechet(3, 1, 0);
    expect(f3.stdDev()).toBeCloseTo(Math.sqrt(f3.variance()), 8);
  });

  it("sf(x) = 1 - cdf(x)", () => {
    expect(f.sf(1)).toBeCloseTo(1 - f.cdf(1), 10);
  });

  it("quantile(1) = Infinity", () => {
    expect(f.quantile(1)).toBe(Infinity);
  });

  it("quantile throws on out-of-range p", () => {
    expect(() => f.quantile(-0.1)).toThrow();
    expect(() => f.quantile(1.1)).toThrow();
  });

  it("sample returns values > m", () => {
    for (let i = 0; i < 50; i++) {
      expect(f.sample()).toBeGreaterThan(0);
    }
  });

  it("sampleN returns correct number of samples", () => {
    expect(f.sampleN(10).length).toBe(10);
  });

  it("works with non-zero location", () => {
    const f4 = new Frechet(2, 1, 5);
    expect(f4.pdf(4)).toBe(0);
    expect(f4.cdf(4)).toBe(0);
    expect(f4.quantile(0)).toBe(5);
  });
});
