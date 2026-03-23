import { ZeroInflatedPoisson } from "../../../src/distributions/discrete/zero-inflated-poisson";

describe("ZeroInflatedPoisson distribution", () => {
  const zip = new ZeroInflatedPoisson(3, 0.3);

  it("has correct name", () => {
    expect(zip.name).toBe("ZIP(3, 0.3)");
  });

  it("mean = (1-pi) * lambda", () => {
    expect(zip.mean()).toBeCloseTo(0.7 * 3, 10);
  });

  it("variance = (1-pi) * lambda * (1 + pi * lambda)", () => {
    expect(zip.variance()).toBeCloseTo(0.7 * 3 * (1 + 0.3 * 3), 10);
  });

  it("pmf(0) includes zero-inflation", () => {
    const p0 = zip.pmf(0);
    // P(0) = pi + (1-pi)*exp(-lambda)
    const expected = 0.3 + 0.7 * Math.exp(-3);
    expect(p0).toBeCloseTo(expected, 10);
  });

  it("pmf for k > 0 is (1-pi) * Poisson(k)", () => {
    for (const k of [1, 2, 5]) {
      const poissonPmf = Math.exp(k * Math.log(3) - 3 - logFact(k));
      expect(zip.pmf(k)).toBeCloseTo(0.7 * poissonPmf, 8);
    }
  });

  it("pmf sums to ~1", () => {
    let sum = 0;
    for (let k = 0; k < 30; k++) sum += zip.pmf(k);
    expect(sum).toBeCloseTo(1, 4);
  });

  it("pmf returns 0 for negative or non-integer k", () => {
    expect(zip.pmf(-1)).toBe(0);
    expect(zip.pmf(1.5)).toBe(0);
  });

  it("cdf is consistent with pmf sum", () => {
    let sum = 0;
    for (let k = 0; k <= 10; k++) {
      sum += zip.pmf(k);
      expect(zip.cdf(k)).toBeCloseTo(sum, 6);
    }
  });

  it("cdf returns 0 for negative k", () => {
    expect(zip.cdf(-1)).toBe(0);
  });

  it("quantile inverts cdf", () => {
    for (const p of [0.1, 0.5, 0.9]) {
      const k = zip.quantile(p);
      expect(zip.cdf(k)).toBeGreaterThanOrEqual(p - 1e-10);
      if (k > 0) expect(zip.cdf(k - 1)).toBeLessThan(p + 1e-10);
    }
  });

  it("quantile edge cases", () => {
    expect(zip.quantile(0)).toBe(0);
    expect(zip.quantile(1)).toBe(Infinity);
    expect(() => zip.quantile(-0.1)).toThrow();
    expect(() => zip.quantile(1.1)).toThrow();
  });

  it("stdDev is sqrt(variance)", () => {
    expect(zip.stdDev()).toBeCloseTo(Math.sqrt(zip.variance()), 8);
  });

  it("sf(k) = 1 - cdf(k)", () => {
    expect(zip.sf(3)).toBeCloseTo(1 - zip.cdf(3), 10);
  });

  it("sample returns non-negative integers", () => {
    for (let i = 0; i < 50; i++) {
      const s = zip.sample();
      expect(s).toBeGreaterThanOrEqual(0);
      expect(Number.isInteger(s)).toBe(true);
    }
  });

  it("sampleN returns correct number of samples", () => {
    expect(zip.sampleN(10).length).toBe(10);
  });

  it("produces more zeros than standard Poisson", () => {
    const samples = zip.sampleN(1000);
    const nZeros = samples.filter((x) => x === 0).length;
    // With pi=0.3, we expect substantially more zeros
    expect(nZeros / 1000).toBeGreaterThan(0.25);
  });

  it("pi=0 reduces to standard Poisson", () => {
    const poisson = new ZeroInflatedPoisson(3, 0);
    expect(poisson.pmf(0)).toBeCloseTo(Math.exp(-3), 10);
    expect(poisson.mean()).toBeCloseTo(3, 10);
    expect(poisson.variance()).toBeCloseTo(3, 10);
  });

  it("throws on invalid parameters", () => {
    expect(() => new ZeroInflatedPoisson(0, 0.3)).toThrow("positive");
    expect(() => new ZeroInflatedPoisson(3, -0.1)).toThrow("pi");
    expect(() => new ZeroInflatedPoisson(3, 1)).toThrow("pi");
  });
});

function logFact(n: number): number {
  let s = 0;
  for (let i = 2; i <= n; i++) s += Math.log(i);
  return s;
}
