import { LogNormal } from "../../../src/distributions/continuous/log-normal";

describe("LogNormal distribution", () => {
  const ln = new LogNormal(0, 1);

  it("has correct mean", () => {
    expect(ln.mean()).toBeCloseTo(Math.exp(0.5), 8);
  });

  it("pdf(0) = 0", () => {
    expect(ln.pdf(0)).toBe(0);
  });

  it("pdf is 0 for negative x", () => {
    expect(ln.pdf(-1)).toBe(0);
  });

  it("cdf(0) = 0", () => {
    expect(ln.cdf(0)).toBe(0);
  });

  it("cdf(1) = 0.5 for LogNormal(0,1)", () => {
    // Because log(1) = 0, and Normal(0,1).cdf(0) = 0.5
    expect(ln.cdf(1)).toBeCloseTo(0.5, 6);
  });

  it("quantile and cdf are inverses", () => {
    for (const p of [0.1, 0.5, 0.9]) {
      expect(ln.cdf(ln.quantile(p))).toBeCloseTo(p, 4);
    }
  });

  it("has correct variance", () => {
    // Var = (e^(sigma^2) - 1) * e^(2*mu + sigma^2)
    const expected = (Math.exp(1) - 1) * Math.exp(1);
    expect(ln.variance()).toBeCloseTo(expected, 6);
  });

  it("stdDev is sqrt(variance)", () => {
    expect(ln.stdDev()).toBeCloseTo(Math.sqrt(ln.variance()), 8);
  });

  it("sf(x) = 1 - cdf(x)", () => {
    expect(ln.sf(1)).toBeCloseTo(1 - ln.cdf(1), 10);
    expect(ln.sf(2)).toBeCloseTo(1 - ln.cdf(2), 10);
  });

  it("cdf is negative for negative x", () => {
    expect(ln.cdf(-1)).toBe(0);
  });

  it("quantile edge cases", () => {
    expect(ln.quantile(0)).toBe(0);
    expect(ln.quantile(1)).toBe(Infinity);
    expect(() => ln.quantile(-0.1)).toThrow();
    expect(() => ln.quantile(1.1)).toThrow();
  });

  it("sample returns positive values", () => {
    for (let i = 0; i < 50; i++) {
      expect(ln.sample()).toBeGreaterThan(0);
    }
  });

  it("sampleN returns correct number of samples", () => {
    expect(ln.sampleN(10).length).toBe(10);
  });

  it("sample mean converges to theoretical mean", () => {
    const samples = ln.sampleN(5000);
    const sampleMean = samples.reduce((a, b) => a + b, 0) / samples.length;
    expect(sampleMean).toBeCloseTo(ln.mean(), 0);
  });

  it("throws on non-positive sigma", () => {
    expect(() => new LogNormal(0, 0)).toThrow();
    expect(() => new LogNormal(0, -1)).toThrow();
  });

  it("works with non-standard parameters", () => {
    const ln2 = new LogNormal(1, 0.5);
    expect(ln2.mean()).toBeCloseTo(Math.exp(1 + 0.125), 6);
    expect(ln2.pdf(1)).toBeGreaterThan(0);
  });
});
