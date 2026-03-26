import { Rayleigh } from "../../../src/distributions/continuous/rayleigh";

describe("Rayleigh distribution", () => {
  const ray = new Rayleigh(1); // sigma=1

  it("has correct name", () => {
    expect(ray.name).toBe("Rayleigh(1)");
  });

  it("throws on non-positive sigma", () => {
    expect(() => new Rayleigh(0)).toThrow();
    expect(() => new Rayleigh(-1)).toThrow();
  });

  it("mean is sigma * sqrt(pi/2)", () => {
    const expected = Math.sqrt(Math.PI / 2);
    expect(ray.mean()).toBeCloseTo(expected, 10);
  });

  it("variance is (4-pi)/2 * sigma^2", () => {
    const expected = (4 - Math.PI) / 2;
    expect(ray.variance()).toBeCloseTo(expected, 10);
  });

  it("stdDev is sqrt(variance)", () => {
    expect(ray.stdDev()).toBeCloseTo(Math.sqrt(ray.variance()), 10);
  });

  it("pdf is 0 for x < 0", () => {
    expect(ray.pdf(-1)).toBe(0);
    expect(ray.pdf(-0.001)).toBe(0);
  });

  it("pdf at x=0 is 0", () => {
    expect(ray.pdf(0)).toBe(0);
  });

  it("pdf at known values", () => {
    // f(1; 1) = (1/1) * exp(-1/2) = exp(-0.5)
    const expected = Math.exp(-0.5);
    expect(ray.pdf(1)).toBeCloseTo(expected, 10);
  });

  it("pdf has correct mode at sigma", () => {
    // Mode of Rayleigh(sigma) = sigma
    // PDF should be highest near x=sigma
    const pdfAtMode = ray.pdf(1);
    expect(pdfAtMode).toBeGreaterThan(ray.pdf(0.5));
    expect(pdfAtMode).toBeGreaterThan(ray.pdf(2));
  });

  it("pdf approaches 0 for large x", () => {
    expect(ray.pdf(10)).toBeLessThan(0.001);
  });

  it("cdf is 0 for x < 0", () => {
    expect(ray.cdf(-1)).toBe(0);
  });

  it("cdf at known values", () => {
    // F(1; 1) = 1 - exp(-1/2)
    const expected = 1 - Math.exp(-0.5);
    expect(ray.cdf(1)).toBeCloseTo(expected, 10);
  });

  it("cdf increases with x", () => {
    expect(ray.cdf(0.5)).toBeLessThan(ray.cdf(1));
    expect(ray.cdf(1)).toBeLessThan(ray.cdf(2));
    expect(ray.cdf(2)).toBeLessThan(ray.cdf(5));
  });

  it("cdf approaches 1 for large x", () => {
    expect(ray.cdf(10)).toBeGreaterThan(0.999);
  });

  it("cdf and quantile are inverses", () => {
    for (const p of [0.1, 0.25, 0.5, 0.75, 0.9]) {
      expect(ray.cdf(ray.quantile(p))).toBeCloseTo(p, 10);
    }
  });

  it("quantile(0) = 0", () => {
    expect(ray.quantile(0)).toBe(0);
  });

  it("quantile(1) = Infinity", () => {
    expect(ray.quantile(1)).toBe(Infinity);
  });

  it("quantile at known values", () => {
    // Q(0.5; 1) = sqrt(-2*ln(0.5)) = sqrt(2*ln(2))
    const expected = Math.sqrt(2 * Math.log(2));
    expect(ray.quantile(0.5)).toBeCloseTo(expected, 10);
  });

  it("quantile throws on out-of-range p", () => {
    expect(() => ray.quantile(-0.1)).toThrow();
    expect(() => ray.quantile(1.1)).toThrow();
  });

  it("sf(x) = 1 - cdf(x)", () => {
    expect(ray.sf(1)).toBeCloseTo(1 - ray.cdf(1), 10);
    expect(ray.sf(2)).toBeCloseTo(1 - ray.cdf(2), 10);
  });

  it("sample returns non-negative values", () => {
    for (let i = 0; i < 50; i++) {
      expect(ray.sample()).toBeGreaterThanOrEqual(0);
    }
  });

  it("sampleN returns correct number of samples", () => {
    expect(ray.sampleN(10).length).toBe(10);
  });

  it("sample mean converges to theoretical mean", () => {
    const samples = ray.sampleN(5000);
    const sampleMean = samples.reduce((a, b) => a + b, 0) / samples.length;
    expect(sampleMean).toBeCloseTo(ray.mean(), 0);
  });

  it("works with non-standard parameters", () => {
    const ray2 = new Rayleigh(3);
    expect(ray2.name).toBe("Rayleigh(3)");
    // mean = 3 * sqrt(pi/2)
    expect(ray2.mean()).toBeCloseTo(3 * Math.sqrt(Math.PI / 2), 10);
    // variance = (4-pi)/2 * 9
    expect(ray2.variance()).toBeCloseTo((4 - Math.PI) / 2 * 9, 10);
    expect(ray2.pdf(3)).toBeGreaterThan(0);
    expect(ray2.cdf(3)).toBeGreaterThan(0);
    expect(ray2.cdf(3)).toBeLessThan(1);
  });
});
