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
});
