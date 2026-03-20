import { DiscreteUniform } from "../../../src/distributions/discrete/discrete-uniform";

describe("DiscreteUniform distribution", () => {
  const du = new DiscreteUniform(1, 6);

  it("has correct mean", () => {
    expect(du.mean()).toBe(3.5);
  });

  it("has correct variance", () => {
    expect(du.variance()).toBeCloseTo(35 / 12, 8);
  });

  it("pmf = 1/6 for k in [1,6]", () => {
    for (let k = 1; k <= 6; k++) {
      expect(du.pmf(k)).toBeCloseTo(1 / 6, 10);
    }
  });

  it("pmf = 0 outside [1,6]", () => {
    expect(du.pmf(0)).toBe(0);
    expect(du.pmf(7)).toBe(0);
  });

  it("cdf(6) = 1", () => {
    expect(du.cdf(6)).toBe(1);
  });

  it("cdf(3) = 3/6", () => {
    expect(du.cdf(3)).toBeCloseTo(0.5, 8);
  });
});
