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
});
