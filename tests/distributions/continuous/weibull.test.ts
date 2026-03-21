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

  it("has correct variance", () => {
    const w2 = new Weibull(2, 1);
    expect(w2.variance()).toBeGreaterThan(0);
    expect(isFinite(w2.variance())).toBe(true);
  });

  it("stdDev is sqrt(variance)", () => {
    const w2 = new Weibull(2, 3);
    expect(w2.stdDev()).toBeCloseTo(Math.sqrt(w2.variance()), 8);
  });

  it("sf(x) = 1 - cdf(x)", () => {
    expect(w.sf(1)).toBeCloseTo(1 - w.cdf(1), 10);
  });

  it("cdf is 0 for negative x", () => {
    expect(w.cdf(-1)).toBe(0);
  });

  it("quantile(1) = Infinity", () => {
    expect(w.quantile(1)).toBe(Infinity);
  });

  it("quantile throws on out-of-range p", () => {
    expect(() => w.quantile(-0.1)).toThrow();
    expect(() => w.quantile(1.1)).toThrow();
  });

  it("pdf edge cases at x=0", () => {
    // k=1: pdf(0) = 1/lambda
    expect(new Weibull(1, 2).pdf(0)).toBeCloseTo(0.5, 8);
    // k<1: pdf(0) = Infinity
    expect(new Weibull(0.5, 1).pdf(0)).toBe(Infinity);
    // k>1: pdf(0) = 0
    expect(new Weibull(2, 1).pdf(0)).toBe(0);
  });

  it("sample returns non-negative values", () => {
    const w2 = new Weibull(2, 3);
    for (let i = 0; i < 50; i++) {
      expect(w2.sample()).toBeGreaterThanOrEqual(0);
    }
  });

  it("sampleN returns correct number of samples", () => {
    expect(w.sampleN(10).length).toBe(10);
  });

  it("sample mean converges to theoretical mean", () => {
    const w2 = new Weibull(2, 3);
    const samples = w2.sampleN(5000);
    const sampleMean = samples.reduce((a, b) => a + b, 0) / samples.length;
    expect(sampleMean).toBeCloseTo(w2.mean(), 0);
  });
});
