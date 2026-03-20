import { FDistribution } from "../../../src/distributions/continuous/f-distribution";

describe("F distribution", () => {
  const f = new FDistribution(5, 10);

  it("has correct name", () => {
    expect(f.name).toBe("F(5, 10)");
  });

  it("mean = d2 / (d2 - 2) for d2 > 2", () => {
    expect(f.mean()).toBeCloseTo(10 / 8, 8);
  });

  it("mean is NaN when d2 <= 2", () => {
    expect(new FDistribution(5, 2).mean()).toBeNaN();
  });

  it("variance matches formula for d2 > 4", () => {
    const expected = (2 * 100 * (5 + 10 - 2)) / (5 * 64 * 6);
    expect(f.variance()).toBeCloseTo(expected, 6);
  });

  it("variance is NaN when d2 <= 4", () => {
    expect(new FDistribution(3, 4).variance()).toBeNaN();
  });

  it("pdf(0) = 0 and pdf > 0 for x > 0", () => {
    expect(f.pdf(0)).toBe(0);
    expect(f.pdf(-1)).toBe(0);
    expect(f.pdf(1)).toBeGreaterThan(0);
  });

  it("cdf(0) = 0", () => {
    expect(f.cdf(0)).toBe(0);
  });

  it("cdf is monotonically increasing", () => {
    let prev = 0;
    for (const x of [0.5, 1, 2, 5, 10]) {
      const c = f.cdf(x);
      expect(c).toBeGreaterThan(prev);
      prev = c;
    }
  });

  it("cdf and quantile are inverses", () => {
    for (const p of [0.1, 0.25, 0.5, 0.75, 0.9]) {
      expect(f.cdf(f.quantile(p))).toBeCloseTo(p, 4);
    }
  });

  it("throws on invalid parameters", () => {
    expect(() => new FDistribution(0, 5)).toThrow();
    expect(() => new FDistribution(5, -1)).toThrow();
    expect(() => new FDistribution(1.5, 3)).toThrow();
  });

  it("stdDev is sqrt(variance)", () => {
    expect(f.stdDev()).toBeCloseTo(Math.sqrt(f.variance()), 8);
  });

  it("sf(x) = 1 - cdf(x)", () => {
    expect(f.sf(1)).toBeCloseTo(1 - f.cdf(1), 10);
  });

  it("quantile edge cases", () => {
    expect(f.quantile(0)).toBe(0);
    expect(f.quantile(1)).toBe(Infinity);
    expect(() => f.quantile(-0.1)).toThrow();
    expect(() => f.quantile(1.1)).toThrow();
  });

  it("sample returns positive values", () => {
    for (let i = 0; i < 50; i++) {
      expect(f.sample()).toBeGreaterThan(0);
    }
  });

  it("sampleN returns correct number of samples", () => {
    expect(f.sampleN(10).length).toBe(10);
  });

  it("sample mean converges to theoretical mean", () => {
    const samples = f.sampleN(5000);
    const sampleMean = samples.reduce((a, b) => a + b, 0) / samples.length;
    expect(sampleMean).toBeCloseTo(f.mean(), 0);
  });
});
