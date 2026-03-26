import { Laplace } from "../../../src/distributions/continuous/laplace";

describe("Laplace distribution", () => {
  const lap = new Laplace(0, 1); // standard Laplace

  it("has correct name", () => {
    expect(lap.name).toBe("Laplace(0, 1)");
  });

  it("throws on non-positive b", () => {
    expect(() => new Laplace(0, 0)).toThrow();
    expect(() => new Laplace(0, -1)).toThrow();
  });

  it("mean equals mu", () => {
    expect(lap.mean()).toBe(0);
    const lap2 = new Laplace(3, 2);
    expect(lap2.mean()).toBe(3);
  });

  it("variance equals 2*b^2", () => {
    expect(lap.variance()).toBeCloseTo(2, 10);
    const lap2 = new Laplace(0, 3);
    expect(lap2.variance()).toBeCloseTo(18, 10);
  });

  it("stdDev equals sqrt(2)*b", () => {
    expect(lap.stdDev()).toBeCloseTo(Math.sqrt(2), 10);
  });

  it("pdf at mu equals 1/(2b)", () => {
    // f(0; 0, 1) = 1/(2*1) * exp(0) = 0.5
    expect(lap.pdf(0)).toBeCloseTo(0.5, 10);
  });

  it("pdf is symmetric around mu", () => {
    expect(lap.pdf(1)).toBeCloseTo(lap.pdf(-1), 10);
    expect(lap.pdf(2)).toBeCloseTo(lap.pdf(-2), 10);
  });

  it("pdf at known values", () => {
    // f(1; 0, 1) = (1/2) * exp(-1)
    const expected = 0.5 * Math.exp(-1);
    expect(lap.pdf(1)).toBeCloseTo(expected, 10);
  });

  it("pdf approaches 0 for extreme values", () => {
    expect(lap.pdf(20)).toBeLessThan(0.001);
    expect(lap.pdf(-20)).toBeLessThan(0.001);
  });

  it("cdf at mu equals 0.5", () => {
    expect(lap.cdf(0)).toBeCloseTo(0.5, 10);
  });

  it("cdf at known values", () => {
    // For x < mu: F(x) = 0.5 * exp((x-mu)/b)
    // F(-1; 0, 1) = 0.5 * exp(-1)
    expect(lap.cdf(-1)).toBeCloseTo(0.5 * Math.exp(-1), 10);
    // For x >= mu: F(x) = 1 - 0.5 * exp(-(x-mu)/b)
    // F(1; 0, 1) = 1 - 0.5 * exp(-1)
    expect(lap.cdf(1)).toBeCloseTo(1 - 0.5 * Math.exp(-1), 10);
  });

  it("cdf increases with x", () => {
    expect(lap.cdf(-2)).toBeLessThan(lap.cdf(-1));
    expect(lap.cdf(-1)).toBeLessThan(lap.cdf(0));
    expect(lap.cdf(0)).toBeLessThan(lap.cdf(1));
  });

  it("cdf approaches 0 for large negative x", () => {
    expect(lap.cdf(-100)).toBeLessThan(0.001);
  });

  it("cdf approaches 1 for large positive x", () => {
    expect(lap.cdf(100)).toBeGreaterThan(0.999);
  });

  it("cdf and quantile are inverses", () => {
    for (const p of [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]) {
      expect(lap.cdf(lap.quantile(p))).toBeCloseTo(p, 5);
    }
  });

  it("quantile(0) = -Infinity", () => {
    expect(lap.quantile(0)).toBe(-Infinity);
  });

  it("quantile(1) = Infinity", () => {
    expect(lap.quantile(1)).toBe(Infinity);
  });

  it("quantile(0.5) = mu", () => {
    expect(lap.quantile(0.5)).toBeCloseTo(0, 10);
  });

  it("quantile throws on out-of-range p", () => {
    expect(() => lap.quantile(-0.1)).toThrow();
    expect(() => lap.quantile(1.1)).toThrow();
  });

  it("sf(x) = 1 - cdf(x)", () => {
    expect(lap.sf(1)).toBeCloseTo(1 - lap.cdf(1), 10);
    expect(lap.sf(-1)).toBeCloseTo(1 - lap.cdf(-1), 10);
  });

  it("sample returns finite values", () => {
    for (let i = 0; i < 50; i++) {
      expect(Number.isFinite(lap.sample())).toBe(true);
    }
  });

  it("sampleN returns correct number of samples", () => {
    expect(lap.sampleN(10).length).toBe(10);
  });

  it("sample mean converges to mu", () => {
    const samples = lap.sampleN(5000);
    const sampleMean = samples.reduce((a, b) => a + b, 0) / samples.length;
    expect(sampleMean).toBeCloseTo(0, 0);
  });

  it("works with non-standard parameters", () => {
    const lap2 = new Laplace(5, 2);
    expect(lap2.name).toBe("Laplace(5, 2)");
    expect(lap2.mean()).toBe(5);
    expect(lap2.variance()).toBeCloseTo(8, 10);
    expect(lap2.pdf(5)).toBeCloseTo(1 / (2 * 2), 10); // 0.25
    expect(lap2.cdf(5)).toBeCloseTo(0.5, 10);
    expect(lap2.quantile(0.5)).toBeCloseTo(5, 5);
  });
});
