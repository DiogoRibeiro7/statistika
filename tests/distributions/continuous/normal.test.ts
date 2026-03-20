import { Normal } from "../../../src/distributions/continuous/normal";

describe("Normal distribution", () => {
  const std = new Normal(0, 1);
  const custom = new Normal(5, 2);

  it("has correct mean and variance", () => {
    expect(std.mean()).toBe(0);
    expect(std.variance()).toBe(1);
    expect(custom.mean()).toBe(5);
    expect(custom.variance()).toBe(4);
  });

  it("pdf at mean equals 1/(sigma*sqrt(2pi))", () => {
    expect(std.pdf(0)).toBeCloseTo(1 / Math.sqrt(2 * Math.PI), 8);
  });

  it("pdf is symmetric", () => {
    expect(std.pdf(1)).toBeCloseTo(std.pdf(-1), 10);
  });

  it("cdf(0) = 0.5 for standard normal", () => {
    expect(std.cdf(0)).toBeCloseTo(0.5, 8);
  });

  it("cdf(-inf) ~ 0 and cdf(inf) ~ 1", () => {
    expect(std.cdf(-10)).toBeCloseTo(0, 5);
    expect(std.cdf(10)).toBeCloseTo(1, 5);
  });

  it("quantile(0.5) = mean", () => {
    expect(std.quantile(0.5)).toBeCloseTo(0, 2);
    expect(custom.quantile(0.5)).toBeCloseTo(5, 2);
  });

  it("quantile(0.975) ~ 1.96 for standard normal", () => {
    expect(std.quantile(0.975)).toBeCloseTo(1.96, 1);
  });

  it("cdf and quantile are inverses", () => {
    for (const p of [0.1, 0.25, 0.5, 0.75, 0.9]) {
      expect(std.cdf(std.quantile(p))).toBeCloseTo(p, 2);
    }
  });

  it("sample returns numbers", () => {
    const samples = std.sampleN(100);
    expect(samples).toHaveLength(100);
    samples.forEach((s) => expect(typeof s).toBe("number"));
  });

  it("throws on non-positive sigma", () => {
    expect(() => new Normal(0, 0)).toThrow();
    expect(() => new Normal(0, -1)).toThrow();
  });
});
