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
});
