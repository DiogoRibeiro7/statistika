import { Gumbel } from "../../../src/distributions/continuous/gumbel";

describe("Gumbel distribution", () => {
  const g = new Gumbel(0, 1);

  it("has correct mean (Euler-Mascheroni)", () => {
    expect(g.mean()).toBeCloseTo(0.5772, 3);
  });

  it("has correct variance (pi^2/6)", () => {
    expect(g.variance()).toBeCloseTo(Math.PI ** 2 / 6, 8);
  });

  it("cdf(0) = exp(-1) ~ 0.3679", () => {
    expect(g.cdf(0)).toBeCloseTo(Math.exp(-1), 8);
  });

  it("pdf is non-negative", () => {
    for (const x of [-5, -1, 0, 1, 5, 10]) {
      expect(g.pdf(x)).toBeGreaterThanOrEqual(0);
    }
  });

  it("quantile and cdf are inverses", () => {
    for (const p of [0.01, 0.1, 0.5, 0.9, 0.99]) {
      expect(g.cdf(g.quantile(p))).toBeCloseTo(p, 10);
    }
  });

  it("quantile(0.5) is the median", () => {
    const median = -Math.log(Math.log(2));
    expect(g.quantile(0.5)).toBeCloseTo(median, 10);
  });

  it("custom location and scale", () => {
    const g2 = new Gumbel(5, 2);
    expect(g2.mean()).toBeCloseTo(5 + 2 * 0.5772, 3);
    expect(g2.variance()).toBeCloseTo(4 * Math.PI ** 2 / 6, 6);
  });

  it("throws on non-positive beta", () => {
    expect(() => new Gumbel(0, 0)).toThrow();
    expect(() => new Gumbel(0, -1)).toThrow();
  });
});
