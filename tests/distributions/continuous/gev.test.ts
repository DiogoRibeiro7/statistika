import { GEV } from "../../../src/distributions/continuous/gev";

describe("GEV distribution", () => {
  describe("Gumbel case (xi = 0)", () => {
    const gev = new GEV(0, 1, 0);

    it("cdf matches Gumbel: exp(-exp(-x))", () => {
      expect(gev.cdf(0)).toBeCloseTo(Math.exp(-1), 8);
      expect(gev.cdf(1)).toBeCloseTo(Math.exp(-Math.exp(-1)), 8);
    });

    it("mean = Euler-Mascheroni constant", () => {
      expect(gev.mean()).toBeCloseTo(0.5772, 3);
    });

    it("variance = pi^2/6", () => {
      expect(gev.variance()).toBeCloseTo(Math.PI ** 2 / 6, 6);
    });

    it("quantile and cdf are inverses", () => {
      for (const p of [0.1, 0.5, 0.9]) {
        expect(gev.cdf(gev.quantile(p))).toBeCloseTo(p, 8);
      }
    });
  });

  describe("Fréchet case (xi > 0)", () => {
    const gev = new GEV(0, 1, 0.5);

    it("has lower bound at mu - sigma/xi", () => {
      expect(gev.cdf(-2)).toBe(0);
      expect(gev.pdf(-3)).toBe(0);
    });

    it("quantile(0) = lower bound", () => {
      expect(gev.quantile(0)).toBe(-2);
    });

    it("cdf and quantile are inverses", () => {
      for (const p of [0.1, 0.5, 0.9]) {
        expect(gev.cdf(gev.quantile(p))).toBeCloseTo(p, 6);
      }
    });
  });

  describe("Reversed Weibull case (xi < 0)", () => {
    const gev = new GEV(0, 1, -0.5);

    it("has upper bound at mu - sigma/xi", () => {
      expect(gev.cdf(2)).toBe(1);
    });

    it("quantile(1) = upper bound", () => {
      expect(gev.quantile(1)).toBe(2);
    });

    it("cdf and quantile are inverses", () => {
      for (const p of [0.1, 0.5, 0.9]) {
        expect(gev.cdf(gev.quantile(p))).toBeCloseTo(p, 6);
      }
    });
  });

  it("throws on non-positive sigma", () => {
    expect(() => new GEV(0, 0, 0)).toThrow();
  });

  describe("coverage for edge cases", () => {
    it("mean is Infinity for xi >= 1", () => {
      expect(new GEV(0, 1, 1).mean()).toBe(Infinity);
    });

    it("variance is Infinity for xi >= 0.5", () => {
      expect(new GEV(0, 1, 0.5).variance()).toBe(Infinity);
    });

    it("stdDev is sqrt(variance) for Gumbel", () => {
      const gev = new GEV(0, 1, 0);
      expect(gev.stdDev()).toBeCloseTo(Math.sqrt(gev.variance()), 8);
    });

    it("sf(x) = 1 - cdf(x)", () => {
      const gev = new GEV(0, 1, 0);
      expect(gev.sf(0)).toBeCloseTo(1 - gev.cdf(0), 10);
    });

    it("quantile(0) = -Infinity for Gumbel", () => {
      expect(new GEV(0, 1, 0).quantile(0)).toBe(-Infinity);
    });

    it("quantile(1) = Infinity for Gumbel and Fréchet", () => {
      expect(new GEV(0, 1, 0).quantile(1)).toBe(Infinity);
      expect(new GEV(0, 1, 0.5).quantile(1)).toBe(Infinity);
    });

    it("pdf returns 0 outside support for xi > 0", () => {
      const gev = new GEV(0, 1, 0.5);
      expect(gev.pdf(-10)).toBe(0);
    });

    it("sample returns finite values", () => {
      const gev = new GEV(0, 1, 0);
      for (let i = 0; i < 50; i++) {
        expect(isFinite(gev.sample())).toBe(true);
      }
    });

    it("sampleN returns correct number of samples", () => {
      expect(new GEV(0, 1, 0).sampleN(10).length).toBe(10);
    });

    it("Fréchet variance with xi > 0 and < 0.5", () => {
      const gev = new GEV(0, 1, 0.3);
      expect(gev.variance()).toBeGreaterThan(0);
      expect(isFinite(gev.variance())).toBe(true);
    });

    it("Reversed Weibull mean for xi < 0", () => {
      const gev = new GEV(0, 1, -0.5);
      expect(isFinite(gev.mean())).toBe(true);
    });
  });
});
