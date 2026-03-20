import { Pareto } from "../../../src/distributions/continuous/pareto";

describe("Pareto distribution", () => {
  const p = new Pareto(3, 1); // alpha=3, xm=1

  it("has correct name", () => {
    expect(p.name).toBe("Pareto(3, 1)");
  });

  it("mean = alpha * xm / (alpha - 1)", () => {
    expect(p.mean()).toBeCloseTo(3 / 2, 10);
  });

  it("mean is Infinity when alpha <= 1", () => {
    expect(new Pareto(1).mean()).toBe(Infinity);
    expect(new Pareto(0.5).mean()).toBe(Infinity);
  });

  it("variance matches formula", () => {
    // xm^2 * alpha / ((alpha-1)^2 * (alpha-2))
    expect(p.variance()).toBeCloseTo(3 / (4 * 1), 10);
  });

  it("variance is Infinity when alpha <= 2", () => {
    expect(new Pareto(2).variance()).toBe(Infinity);
  });

  it("pdf = 0 for x < xm", () => {
    expect(p.pdf(0.5)).toBe(0);
  });

  it("pdf at xm = alpha / xm", () => {
    expect(p.pdf(1)).toBeCloseTo(3, 10);
  });

  it("cdf(xm) = 0", () => {
    expect(p.cdf(1)).toBeCloseTo(0, 10);
  });

  it("cdf and quantile are inverses", () => {
    for (const prob of [0.1, 0.5, 0.9, 0.99]) {
      expect(p.cdf(p.quantile(prob))).toBeCloseTo(prob, 10);
    }
  });

  it("quantile(0) = xm", () => {
    expect(p.quantile(0)).toBe(1);
  });

  it("throws on invalid parameters", () => {
    expect(() => new Pareto(0)).toThrow();
    expect(() => new Pareto(-1)).toThrow();
    expect(() => new Pareto(2, 0)).toThrow();
  });

  it("stdDev is sqrt(variance)", () => {
    expect(p.stdDev()).toBeCloseTo(Math.sqrt(p.variance()), 8);
  });

  it("sf(x) = 1 - cdf(x)", () => {
    expect(p.sf(2)).toBeCloseTo(1 - p.cdf(2), 10);
  });

  it("quantile(1) = Infinity", () => {
    expect(p.quantile(1)).toBe(Infinity);
  });

  it("quantile throws on out-of-range p", () => {
    expect(() => p.quantile(-0.1)).toThrow();
    expect(() => p.quantile(1.1)).toThrow();
  });

  it("sample returns values >= xm", () => {
    for (let i = 0; i < 50; i++) {
      expect(p.sample()).toBeGreaterThanOrEqual(1);
    }
  });

  it("sampleN returns correct number of samples", () => {
    expect(p.sampleN(10).length).toBe(10);
  });
});
