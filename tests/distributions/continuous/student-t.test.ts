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
});
