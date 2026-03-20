import { GPD } from "../../../src/distributions/continuous/gpd";

describe("GPD distribution", () => {
  describe("Exponential case (xi = 0)", () => {
    const gpd = new GPD(0, 1, 0);

    it("cdf matches Exponential: 1 - exp(-x)", () => {
      expect(gpd.cdf(1)).toBeCloseTo(1 - Math.exp(-1), 8);
      expect(gpd.cdf(0)).toBe(0);
    });

    it("pdf(0) = 1/sigma", () => {
      expect(gpd.pdf(0)).toBeCloseTo(1, 8);
    });

    it("mean = sigma / (1 - xi) = 1", () => {
      expect(gpd.mean()).toBeCloseTo(1, 10);
    });

    it("quantile and cdf are inverses", () => {
      for (const p of [0.1, 0.5, 0.9]) {
        expect(gpd.cdf(gpd.quantile(p))).toBeCloseTo(p, 10);
      }
    });
  });

  describe("Heavy tail (xi > 0)", () => {
    const gpd = new GPD(0, 1, 0.5);

    it("mean = sigma / (1 - xi) = 2", () => {
      expect(gpd.mean()).toBeCloseTo(2, 10);
    });

    it("variance is Infinity for xi >= 0.5", () => {
      expect(gpd.variance()).toBe(Infinity);
    });

    it("cdf and quantile are inverses", () => {
      for (const p of [0.1, 0.5, 0.9]) {
        expect(gpd.cdf(gpd.quantile(p))).toBeCloseTo(p, 8);
      }
    });
  });

  describe("Bounded tail (xi < 0)", () => {
    const gpd = new GPD(0, 1, -0.5);

    it("has upper bound at -sigma/xi = 2", () => {
      expect(gpd.cdf(2)).toBe(1);
      expect(gpd.cdf(3)).toBe(1);
    });

    it("quantile(1) = upper bound", () => {
      expect(gpd.quantile(1)).toBe(2);
    });

    it("cdf and quantile are inverses", () => {
      for (const p of [0.1, 0.5, 0.9]) {
        expect(gpd.cdf(gpd.quantile(p))).toBeCloseTo(p, 8);
      }
    });
  });

  it("throws on non-positive sigma", () => {
    expect(() => new GPD(0, 0, 0)).toThrow();
  });

  describe("coverage for edge cases", () => {
    it("mean is Infinity for xi >= 1", () => {
      expect(new GPD(0, 1, 1).mean()).toBe(Infinity);
    });

    it("variance for xi < 0.5", () => {
      const gpd = new GPD(0, 1, 0.2);
      expect(gpd.variance()).toBeGreaterThan(0);
      expect(isFinite(gpd.variance())).toBe(true);
    });

    it("stdDev is sqrt(variance)", () => {
      const gpd = new GPD(0, 1, 0);
      expect(gpd.stdDev()).toBeCloseTo(Math.sqrt(gpd.variance()), 8);
    });

    it("sf(x) = 1 - cdf(x)", () => {
      const gpd = new GPD(0, 1, 0);
      expect(gpd.sf(1)).toBeCloseTo(1 - gpd.cdf(1), 10);
    });

    it("pdf returns 0 below mu", () => {
      const gpd = new GPD(0, 1, 0.5);
      expect(gpd.pdf(-1)).toBe(0);
    });

    it("cdf returns 0 below mu", () => {
      const gpd = new GPD(0, 1, 0.5);
      expect(gpd.cdf(-1)).toBe(0);
    });

    it("quantile(0) = mu", () => {
      expect(new GPD(5, 1, 0).quantile(0)).toBe(5);
    });

    it("quantile(1) = Infinity for xi >= 0", () => {
      expect(new GPD(0, 1, 0).quantile(1)).toBe(Infinity);
      expect(new GPD(0, 1, 0.5).quantile(1)).toBe(Infinity);
    });

    it("quantile throws on out-of-range p", () => {
      const gpd = new GPD(0, 1, 0);
      expect(() => gpd.quantile(-0.1)).toThrow();
      expect(() => gpd.quantile(1.1)).toThrow();
    });

    it("sample returns values >= mu", () => {
      const gpd = new GPD(0, 1, 0);
      for (let i = 0; i < 50; i++) {
        expect(gpd.sample()).toBeGreaterThanOrEqual(0);
      }
    });

    it("sampleN returns correct number of samples", () => {
      expect(new GPD(0, 1, 0).sampleN(10).length).toBe(10);
    });
  });
});
