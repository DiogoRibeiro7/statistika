import { TruncatedNormal } from "../../../src/distributions/continuous/truncated-normal";

describe("TruncatedNormal distribution", () => {
  const tn = new TruncatedNormal(0, 1, -1, 1); // mu=0, sigma=1, a=-1, b=1

  it("has correct name", () => {
    expect(tn.name).toBe("TruncatedNormal(0, 1, -1, 1)");
  });

  it("throws on non-positive sigma", () => {
    expect(() => new TruncatedNormal(0, 0, -1, 1)).toThrow();
    expect(() => new TruncatedNormal(0, -1, -1, 1)).toThrow();
  });

  it("throws when a >= b", () => {
    expect(() => new TruncatedNormal(0, 1, 1, 1)).toThrow();
    expect(() => new TruncatedNormal(0, 1, 2, 1)).toThrow();
  });

  it("mean is 0 by symmetry for symmetric truncation around mu", () => {
    expect(tn.mean()).toBeCloseTo(0, 5);
  });

  it("variance is less than untruncated normal variance", () => {
    // Truncation reduces variance; untruncated N(0,1) has variance 1
    expect(tn.variance()).toBeLessThan(1);
    expect(tn.variance()).toBeGreaterThan(0);
  });

  it("pdf is 0 outside [a, b]", () => {
    expect(tn.pdf(-1.5)).toBe(0);
    expect(tn.pdf(1.5)).toBe(0);
    expect(tn.pdf(-2)).toBe(0);
    expect(tn.pdf(2)).toBe(0);
  });

  it("pdf is positive inside [a, b]", () => {
    expect(tn.pdf(0)).toBeGreaterThan(0);
    expect(tn.pdf(0.5)).toBeGreaterThan(0);
    expect(tn.pdf(-0.5)).toBeGreaterThan(0);
  });

  it("pdf is symmetric for symmetric truncation", () => {
    expect(tn.pdf(0.5)).toBeCloseTo(tn.pdf(-0.5), 10);
    expect(tn.pdf(0.9)).toBeCloseTo(tn.pdf(-0.9), 10);
  });

  it("pdf integrates to approximately 1 (numerical check)", () => {
    // Rough numerical integration using trapezoidal rule
    const n = 1000;
    const h = 2 / n; // from -1 to 1
    let sum = 0;
    for (let i = 0; i <= n; i++) {
      const x = -1 + i * h;
      const w = i === 0 || i === n ? 0.5 : 1;
      sum += w * tn.pdf(x);
    }
    sum *= h;
    expect(sum).toBeCloseTo(1, 2);
  });

  it("cdf is 0 at a", () => {
    expect(tn.cdf(-1)).toBe(0);
  });

  it("cdf is 1 at b", () => {
    expect(tn.cdf(1)).toBe(1);
  });

  it("cdf is 0.5 at mu for symmetric truncation", () => {
    expect(tn.cdf(0)).toBeCloseTo(0.5, 5);
  });

  it("cdf increases with x inside [a, b]", () => {
    expect(tn.cdf(-0.5)).toBeLessThan(tn.cdf(0));
    expect(tn.cdf(0)).toBeLessThan(tn.cdf(0.5));
  });

  it("cdf and quantile are inverses", () => {
    for (const p of [0.1, 0.25, 0.5, 0.75, 0.9]) {
      const q = tn.quantile(p);
      expect(tn.cdf(q)).toBeCloseTo(p, 2);
    }
  });

  it("quantile(0) = a", () => {
    expect(tn.quantile(0)).toBe(-1);
  });

  it("quantile(1) = b", () => {
    expect(tn.quantile(1)).toBe(1);
  });

  it("quantile throws on out-of-range p", () => {
    expect(() => tn.quantile(-0.1)).toThrow();
    expect(() => tn.quantile(1.1)).toThrow();
  });

  it("sample returns values in [a, b]", () => {
    for (let i = 0; i < 100; i++) {
      const s = tn.sample();
      expect(s).toBeGreaterThanOrEqual(-1);
      expect(s).toBeLessThanOrEqual(1);
    }
  });

  it("sampleN returns correct number of samples", () => {
    expect(tn.sampleN(10).length).toBe(10);
  });

  it("sample mean converges to theoretical mean", () => {
    const samples = tn.sampleN(5000);
    const sampleMean = samples.reduce((a, b) => a + b, 0) / samples.length;
    expect(sampleMean).toBeCloseTo(tn.mean(), 0);
  });

  it("works with non-standard parameters", () => {
    const tn2 = new TruncatedNormal(5, 2, 3, 7);
    expect(tn2.name).toBe("TruncatedNormal(5, 2, 3, 7)");
    expect(tn2.pdf(5)).toBeGreaterThan(0);
    expect(tn2.pdf(2)).toBe(0);
    expect(tn2.pdf(8)).toBe(0);
    expect(tn2.cdf(3)).toBe(0);
    expect(tn2.cdf(7)).toBe(1);
  });

  it("approaches normal distribution with wide truncation bounds", () => {
    const tnWide = new TruncatedNormal(0, 1, -100, 100);
    // Should be very close to standard normal
    const normalPdf0 = 1 / Math.sqrt(2 * Math.PI);
    expect(tnWide.pdf(0)).toBeCloseTo(normalPdf0, 5);
    expect(tnWide.cdf(0)).toBeCloseTo(0.5, 5);
    expect(tnWide.mean()).toBeCloseTo(0, 5);
    expect(tnWide.variance()).toBeCloseTo(1, 2);
  });

  it("handles one-sided truncation (lower bound only)", () => {
    const tnLower = new TruncatedNormal(0, 1, 0, Infinity);
    // Half-normal: mean = sigma * sqrt(2/pi) = sqrt(2/pi)
    expect(tnLower.mean()).toBeCloseTo(Math.sqrt(2 / Math.PI), 2);
    expect(tnLower.pdf(-0.5)).toBe(0);
    expect(tnLower.pdf(0.5)).toBeGreaterThan(0);
    expect(tnLower.cdf(0)).toBe(0);
  });

  it("handles one-sided truncation (upper bound only)", () => {
    const tnUpper = new TruncatedNormal(0, 1, -Infinity, 0);
    // Symmetric to half-normal
    expect(tnUpper.mean()).toBeCloseTo(-Math.sqrt(2 / Math.PI), 2);
    expect(tnUpper.pdf(0.5)).toBe(0);
    expect(tnUpper.pdf(-0.5)).toBeGreaterThan(0);
    expect(tnUpper.cdf(0)).toBe(1);
  });

  it("sf(x) = 1 - cdf(x)", () => {
    expect(tn.sf(0.3)).toBeCloseTo(1 - tn.cdf(0.3), 10);
  });
});
