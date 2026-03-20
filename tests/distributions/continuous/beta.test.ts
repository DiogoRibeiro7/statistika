import { BetaDistribution } from "../../../src/distributions/continuous/beta";

describe("Beta distribution", () => {
  const beta = new BetaDistribution(2, 5);

  it("has correct mean", () => {
    expect(beta.mean()).toBeCloseTo(2 / 7, 8);
  });

  it("pdf is 0 outside [0,1]", () => {
    expect(beta.pdf(-0.1)).toBe(0);
    expect(beta.pdf(1.1)).toBe(0);
  });

  it("cdf(0) = 0 and cdf(1) = 1", () => {
    expect(beta.cdf(0)).toBe(0);
    expect(beta.cdf(1)).toBe(1);
  });

  it("Beta(1,1) is Uniform(0,1)", () => {
    const uniform = new BetaDistribution(1, 1);
    expect(uniform.pdf(0.5)).toBeCloseTo(1, 8);
    expect(uniform.cdf(0.5)).toBeCloseTo(0.5, 8);
  });

  it("quantile and cdf are inverses", () => {
    for (const p of [0.1, 0.5, 0.9]) {
      expect(beta.cdf(beta.quantile(p))).toBeCloseTo(p, 4);
    }
  });
});
