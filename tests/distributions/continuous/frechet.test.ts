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
});
