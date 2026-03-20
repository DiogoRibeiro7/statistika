import { Weibull } from "../../../src/distributions/continuous/weibull";

describe("Weibull distribution", () => {
  const w = new Weibull(1, 1); // Exponential(1)

  it("Weibull(1,1) is Exponential(1)", () => {
    expect(w.mean()).toBeCloseTo(1, 6);
    expect(w.cdf(1)).toBeCloseTo(1 - Math.exp(-1), 8);
    expect(w.pdf(0)).toBeCloseTo(1, 8);
  });

  it("cdf(0) = 0", () => {
    expect(w.cdf(0)).toBe(0);
  });

  it("pdf is 0 for negative x", () => {
    expect(w.pdf(-1)).toBe(0);
  });

  it("quantile and cdf are inverses", () => {
    const w2 = new Weibull(2, 3);
    for (const p of [0.1, 0.5, 0.9]) {
      expect(w2.cdf(w2.quantile(p))).toBeCloseTo(p, 10);
    }
  });

  it("quantile(0) = 0", () => {
    expect(w.quantile(0)).toBe(0);
  });

  it("Weibull(2, lambda) has known mean", () => {
    const w2 = new Weibull(2, 1);
    // mean = lambda * Gamma(1 + 1/k) = Gamma(1.5) = sqrt(pi)/2
    expect(w2.mean()).toBeCloseTo(Math.sqrt(Math.PI) / 2, 6);
  });

  it("throws on non-positive parameters", () => {
    expect(() => new Weibull(0, 1)).toThrow();
    expect(() => new Weibull(1, 0)).toThrow();
  });
});
