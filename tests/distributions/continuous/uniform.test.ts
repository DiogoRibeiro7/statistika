import { Uniform } from "../../../src/distributions/continuous/uniform";

describe("Uniform distribution", () => {
  const u = new Uniform(0, 1);
  const u2 = new Uniform(2, 8);

  it("has correct mean and variance", () => {
    expect(u.mean()).toBe(0.5);
    expect(u.variance()).toBeCloseTo(1 / 12, 8);
    expect(u2.mean()).toBe(5);
    expect(u2.variance()).toBeCloseTo(3, 8);
  });

  it("pdf is constant on [a,b]", () => {
    expect(u.pdf(0.5)).toBe(1);
    expect(u.pdf(-1)).toBe(0);
    expect(u2.pdf(5)).toBeCloseTo(1 / 6, 8);
  });

  it("cdf is linear on [a,b]", () => {
    expect(u.cdf(0)).toBe(0);
    expect(u.cdf(0.5)).toBe(0.5);
    expect(u.cdf(1)).toBe(1);
  });

  it("quantile is inverse of cdf", () => {
    for (const p of [0, 0.25, 0.5, 0.75, 1]) {
      expect(u.cdf(u.quantile(p))).toBeCloseTo(p, 8);
    }
  });
});
