import { StudentT } from "../../../src/distributions/continuous/student-t";

describe("Student-t distribution", () => {
  const t5 = new StudentT(5);
  const t30 = new StudentT(30);

  it("has correct mean", () => {
    expect(t5.mean()).toBe(0);
  });

  it("mean is NaN for nu <= 1", () => {
    expect(new StudentT(1).mean()).toBeNaN();
  });

  it("variance = nu/(nu-2) for nu > 2", () => {
    expect(t5.variance()).toBeCloseTo(5 / 3, 8);
  });

  it("pdf is symmetric", () => {
    expect(t5.pdf(1)).toBeCloseTo(t5.pdf(-1), 10);
  });

  it("cdf(0) = 0.5", () => {
    expect(t5.cdf(0)).toBeCloseTo(0.5, 6);
  });

  it("t(30) approaches standard normal", () => {
    expect(t30.cdf(1.96)).toBeCloseTo(0.975, 1);
  });

  it("quantile and cdf are approximate inverses", () => {
    for (const p of [0.1, 0.5, 0.9]) {
      expect(t5.cdf(t5.quantile(p))).toBeCloseTo(p, 2);
    }
  });

  it("variance is Infinity for 1 < nu <= 2", () => {
    expect(new StudentT(1.5).variance()).toBe(Infinity);
    expect(new StudentT(2).variance()).toBe(Infinity);
  });

  it("variance is NaN for nu <= 1", () => {
    expect(new StudentT(0.5).variance()).toBeNaN();
  });

  it("stdDev is sqrt(variance)", () => {
    expect(t5.stdDev()).toBeCloseTo(Math.sqrt(t5.variance()), 8);
  });

  it("sf(x) = 1 - cdf(x)", () => {
    expect(t5.sf(1)).toBeCloseTo(1 - t5.cdf(1), 10);
  });

  it("quantile edge cases", () => {
    expect(t5.quantile(0)).toBe(-Infinity);
    expect(t5.quantile(1)).toBe(Infinity);
    expect(() => t5.quantile(-0.1)).toThrow();
    expect(() => t5.quantile(1.1)).toThrow();
  });

  it("quantile uses symmetry for p < 0.5", () => {
    expect(t5.quantile(0.25)).toBeCloseTo(-t5.quantile(0.75), 4);
  });

  it("sample returns finite values", () => {
    for (let i = 0; i < 50; i++) {
      expect(isFinite(t5.sample())).toBe(true);
    }
  });

  it("sampleN returns correct number of samples", () => {
    expect(t5.sampleN(10).length).toBe(10);
  });

  it("throws on non-positive nu", () => {
    expect(() => new StudentT(0)).toThrow();
    expect(() => new StudentT(-1)).toThrow();
  });
});
