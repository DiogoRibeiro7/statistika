import { GammaDistribution } from "../../../src/distributions/continuous/gamma";

describe("Gamma distribution", () => {
  const g = new GammaDistribution(2, 1);

  // ---- Constructor & Name ----
  it("has correct name", () => {
    expect(g.name).toBe("Gamma(2, 1)");
  });

  it("throws on non-positive shape", () => {
    expect(() => new GammaDistribution(0, 1)).toThrow();
    expect(() => new GammaDistribution(-1, 1)).toThrow();
  });

  it("throws on non-positive rate", () => {
    expect(() => new GammaDistribution(1, 0)).toThrow();
    expect(() => new GammaDistribution(1, -2)).toThrow();
  });

  // ---- Mean & Variance ----
  it("has correct mean and variance", () => {
    expect(g.mean()).toBe(2);
    expect(g.variance()).toBe(2);
  });

  it("mean = shape/rate for various parameters", () => {
    const g2 = new GammaDistribution(5, 2);
    expect(g2.mean()).toBeCloseTo(2.5, 8);
  });

  it("variance = shape/rate^2", () => {
    const g2 = new GammaDistribution(5, 2);
    expect(g2.variance()).toBeCloseTo(1.25, 8);
  });

  // ---- StdDev ----
  it("stdDev is sqrt(variance)", () => {
    expect(g.stdDev()).toBeCloseTo(Math.sqrt(g.variance()), 8);
  });

  // ---- PDF ----
  it("pdf(x) = 0 for x < 0", () => {
    expect(g.pdf(-1)).toBe(0);
    expect(g.pdf(-0.001)).toBe(0);
  });

  it("pdf(0) = 0 for shape > 1", () => {
    expect(g.pdf(0)).toBe(0);
  });

  it("pdf(0) = rate for shape = 1 (exponential)", () => {
    const exp = new GammaDistribution(1, 3);
    expect(exp.pdf(0)).toBe(3);
  });

  it("pdf(0) = Infinity for shape < 1", () => {
    const g2 = new GammaDistribution(0.5, 1);
    expect(g2.pdf(0)).toBe(Infinity);
  });

  it("pdf is positive for x > 0", () => {
    for (const x of [0.1, 0.5, 1, 2, 5, 10]) {
      expect(g.pdf(x)).toBeGreaterThan(0);
    }
  });

  it("pdf integrates to approximately 1", () => {
    // Trapezoidal rule over [0, 20]
    const n = 2000;
    const upper = 20;
    let sum = 0;
    for (let i = 0; i <= n; i++) {
      const x = (i / n) * upper;
      const w = i === 0 || i === n ? 0.5 : 1;
      sum += w * g.pdf(x);
    }
    sum *= upper / n;
    expect(sum).toBeCloseTo(1, 2);
  });

  it("pdf matches known values for exponential (shape=1, rate=1)", () => {
    const exp = new GammaDistribution(1, 1);
    expect(exp.pdf(1)).toBeCloseTo(Math.exp(-1), 6);
    expect(exp.pdf(2)).toBeCloseTo(Math.exp(-2), 6);
  });

  // ---- CDF ----
  it("cdf(0) = 0", () => {
    expect(g.cdf(0)).toBe(0);
  });

  it("cdf(x) = 0 for x < 0", () => {
    expect(g.cdf(-1)).toBe(0);
  });

  it("cdf is monotonically increasing", () => {
    let prev = 0;
    for (const x of [0.5, 1, 2, 3, 5, 10]) {
      const c = g.cdf(x);
      expect(c).toBeGreaterThan(prev);
      prev = c;
    }
  });

  it("cdf approaches 1 for large x", () => {
    expect(g.cdf(20)).toBeGreaterThan(0.999);
  });

  it("cdf matches known values for exponential", () => {
    const exp = new GammaDistribution(1, 1);
    expect(exp.cdf(1)).toBeCloseTo(1 - Math.exp(-1), 6);
    expect(exp.cdf(2)).toBeCloseTo(1 - Math.exp(-2), 6);
  });

  // ---- SF ----
  it("sf(x) = 1 - cdf(x)", () => {
    for (const x of [0.5, 1, 2, 5]) {
      expect(g.sf(x)).toBeCloseTo(1 - g.cdf(x), 10);
    }
  });

  // ---- Quantile ----
  it("quantile and cdf are inverses", () => {
    for (const p of [0.1, 0.25, 0.5, 0.75, 0.9]) {
      expect(g.cdf(g.quantile(p))).toBeCloseTo(p, 4);
    }
  });

  it("quantile(0) = 0", () => {
    expect(g.quantile(0)).toBe(0);
  });

  it("quantile(1) = Infinity", () => {
    expect(g.quantile(1)).toBe(Infinity);
  });

  it("quantile throws on out-of-range p", () => {
    expect(() => g.quantile(-0.1)).toThrow();
    expect(() => g.quantile(1.1)).toThrow();
  });

  // ---- Sample ----
  it("sample returns non-negative values", () => {
    for (let i = 0; i < 100; i++) {
      expect(g.sample()).toBeGreaterThanOrEqual(0);
    }
  });

  it("sample mean converges to theoretical mean (shape >= 1)", () => {
    const samples = g.sampleN(5000);
    const sampleMean = samples.reduce((a, b) => a + b, 0) / samples.length;
    expect(sampleMean).toBeCloseTo(g.mean(), 0);
  });

  it("sample works with shape < 1", () => {
    const g2 = new GammaDistribution(0.5, 1);
    const samples = g2.sampleN(3000);
    const sampleMean = samples.reduce((a, b) => a + b, 0) / samples.length;
    expect(sampleMean).toBeCloseTo(g2.mean(), 0);

    // All should be positive
    for (const s of samples) {
      expect(s).toBeGreaterThan(0);
    }
  });

  it("sample works with different rate", () => {
    const g2 = new GammaDistribution(3, 2);
    const samples = g2.sampleN(3000);
    const sampleMean = samples.reduce((a, b) => a + b, 0) / samples.length;
    expect(sampleMean).toBeCloseTo(g2.mean(), 0);
  });

  // ---- SampleN ----
  it("sampleN returns correct number of samples", () => {
    expect(g.sampleN(10).length).toBe(10);
    expect(g.sampleN(100).length).toBe(100);
  });

  // ---- Edge cases ----
  it("works with shape = 1 (exponential distribution)", () => {
    const exp = new GammaDistribution(1, 2);
    expect(exp.mean()).toBeCloseTo(0.5, 8);
    expect(exp.variance()).toBeCloseTo(0.25, 8);
  });

  it("works with large shape parameter", () => {
    const g2 = new GammaDistribution(100, 1);
    expect(g2.mean()).toBe(100);
    expect(g2.variance()).toBe(100);
    // CDF at mean should be near 0.5
    expect(g2.cdf(100)).toBeCloseTo(0.5, 0);
  });

  it("works with large rate parameter", () => {
    const g2 = new GammaDistribution(2, 100);
    expect(g2.mean()).toBeCloseTo(0.02, 8);
    expect(g2.sample()).toBeGreaterThan(0);
  });
});
