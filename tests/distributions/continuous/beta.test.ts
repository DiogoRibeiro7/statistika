import { BetaDistribution } from "../../../src/distributions/continuous/beta";

describe("Beta distribution", () => {
  const beta = new BetaDistribution(2, 5);

  // ---- Constructor & Name ----
  it("has correct name", () => {
    expect(beta.name).toBe("Beta(2, 5)");
  });

  it("throws on non-positive alpha", () => {
    expect(() => new BetaDistribution(0, 1)).toThrow();
    expect(() => new BetaDistribution(-1, 1)).toThrow();
  });

  it("throws on non-positive beta", () => {
    expect(() => new BetaDistribution(1, 0)).toThrow();
    expect(() => new BetaDistribution(1, -2)).toThrow();
  });

  // ---- Mean ----
  it("has correct mean", () => {
    expect(beta.mean()).toBeCloseTo(2 / 7, 8);
  });

  it("mean of Beta(1,1) is 0.5", () => {
    expect(new BetaDistribution(1, 1).mean()).toBeCloseTo(0.5, 8);
  });

  // ---- Variance ----
  it("has correct variance", () => {
    // Var = alpha*beta / ((alpha+beta)^2 * (alpha+beta+1))
    const expected = (2 * 5) / (7 * 7 * 8);
    expect(beta.variance()).toBeCloseTo(expected, 8);
  });

  it("variance of Beta(1,1) is 1/12", () => {
    expect(new BetaDistribution(1, 1).variance()).toBeCloseTo(1 / 12, 8);
  });

  // ---- StdDev ----
  it("stdDev is sqrt(variance)", () => {
    expect(beta.stdDev()).toBeCloseTo(Math.sqrt(beta.variance()), 8);
  });

  // ---- PDF ----
  it("pdf is 0 outside [0,1]", () => {
    expect(beta.pdf(-0.1)).toBe(0);
    expect(beta.pdf(1.1)).toBe(0);
  });

  it("pdf at x=0 returns 0 when alpha > 1", () => {
    expect(beta.pdf(0)).toBe(0);
  });

  it("pdf at x=0 returns beta when alpha = 1", () => {
    const b = new BetaDistribution(1, 3);
    expect(b.pdf(0)).toBe(3);
  });

  it("pdf at x=0 returns Infinity when alpha < 1", () => {
    const b = new BetaDistribution(0.5, 2);
    expect(b.pdf(0)).toBe(Infinity);
  });

  it("pdf at x=1 returns 0 when beta > 1", () => {
    expect(beta.pdf(1)).toBe(0);
  });

  it("pdf at x=1 returns alpha when beta = 1", () => {
    const b = new BetaDistribution(3, 1);
    expect(b.pdf(1)).toBe(3);
  });

  it("pdf at x=1 returns Infinity when beta < 1", () => {
    const b = new BetaDistribution(2, 0.5);
    expect(b.pdf(1)).toBe(Infinity);
  });

  it("pdf is positive in the interior", () => {
    for (const x of [0.1, 0.3, 0.5, 0.7, 0.9]) {
      expect(beta.pdf(x)).toBeGreaterThan(0);
    }
  });

  it("pdf integrates to approximately 1 (numerical check)", () => {
    // Trapezoidal rule
    const n = 1000;
    let sum = 0;
    for (let i = 0; i <= n; i++) {
      const x = i / n;
      const w = i === 0 || i === n ? 0.5 : 1;
      sum += w * beta.pdf(x);
    }
    sum /= n;
    expect(sum).toBeCloseTo(1, 2);
  });

  // ---- CDF ----
  it("cdf(0) = 0 and cdf(1) = 1", () => {
    expect(beta.cdf(0)).toBe(0);
    expect(beta.cdf(1)).toBe(1);
  });

  it("cdf is monotonically increasing", () => {
    let prev = 0;
    for (const x of [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]) {
      const c = beta.cdf(x);
      expect(c).toBeGreaterThanOrEqual(prev);
      prev = c;
    }
  });

  it("cdf values below 0 and above 1", () => {
    expect(beta.cdf(-1)).toBe(0);
    expect(beta.cdf(2)).toBe(1);
  });

  // ---- SF ----
  it("sf(x) = 1 - cdf(x)", () => {
    for (const x of [0.1, 0.3, 0.5, 0.7, 0.9]) {
      expect(beta.sf(x)).toBeCloseTo(1 - beta.cdf(x), 10);
    }
  });

  // ---- Beta(1,1) is Uniform(0,1) ----
  it("Beta(1,1) is Uniform(0,1)", () => {
    const uniform = new BetaDistribution(1, 1);
    expect(uniform.pdf(0.5)).toBeCloseTo(1, 8);
    expect(uniform.cdf(0.5)).toBeCloseTo(0.5, 8);
    expect(uniform.quantile(0.25)).toBeCloseTo(0.25, 4);
  });

  // ---- Quantile ----
  it("quantile and cdf are inverses", () => {
    for (const p of [0.1, 0.25, 0.5, 0.75, 0.9]) {
      expect(beta.cdf(beta.quantile(p))).toBeCloseTo(p, 4);
    }
  });

  it("quantile(0) = 0 and quantile(1) = 1", () => {
    expect(beta.quantile(0)).toBe(0);
    expect(beta.quantile(1)).toBe(1);
  });

  it("quantile throws on out-of-range p", () => {
    expect(() => beta.quantile(-0.1)).toThrow();
    expect(() => beta.quantile(1.1)).toThrow();
  });

  // ---- Sample ----
  it("sample returns values in [0, 1]", () => {
    for (let i = 0; i < 100; i++) {
      const s = beta.sample();
      expect(s).toBeGreaterThanOrEqual(0);
      expect(s).toBeLessThanOrEqual(1);
    }
  });

  it("sample mean converges to theoretical mean", () => {
    const samples = beta.sampleN(5000);
    const sampleMean = samples.reduce((a, b) => a + b, 0) / samples.length;
    expect(sampleMean).toBeCloseTo(beta.mean(), 1);
  });

  // ---- SampleN ----
  it("sampleN returns correct number of samples", () => {
    expect(beta.sampleN(10).length).toBe(10);
    expect(beta.sampleN(100).length).toBe(100);
  });

  // ---- Various alpha/beta combinations ----
  it("works with alpha < 1, beta < 1", () => {
    const b = new BetaDistribution(0.5, 0.5);
    expect(b.mean()).toBeCloseTo(0.5, 8);
    expect(b.pdf(0.5)).toBeGreaterThan(0);
    expect(b.cdf(0.5)).toBeCloseTo(0.5, 4);
  });

  it("works with large alpha and beta", () => {
    const b = new BetaDistribution(100, 100);
    expect(b.mean()).toBeCloseTo(0.5, 8);
    // Very concentrated around 0.5
    expect(b.cdf(0.45)).toBeLessThan(0.1);
    expect(b.cdf(0.55)).toBeGreaterThan(0.9);
  });

  it("works with asymmetric parameters", () => {
    const b = new BetaDistribution(0.5, 5);
    expect(b.mean()).toBeCloseTo(0.5 / 5.5, 4);
    expect(b.pdf(0.01)).toBeGreaterThan(0);
    const s = b.sample();
    expect(s).toBeGreaterThanOrEqual(0);
    expect(s).toBeLessThanOrEqual(1);
  });
});
