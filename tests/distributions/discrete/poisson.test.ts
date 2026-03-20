import { Poisson } from "../../../src/distributions/discrete/poisson";

describe("Poisson distribution", () => {
  const p = new Poisson(3);

  it("has correct mean and variance", () => {
    expect(p.mean()).toBe(3);
    expect(p.variance()).toBe(3);
  });

  it("pmf(0) = e^-lambda", () => {
    expect(p.pmf(0)).toBeCloseTo(Math.exp(-3), 8);
  });

  it("pmf sums approximately to 1", () => {
    let sum = 0;
    for (let k = 0; k <= 30; k++) sum += p.pmf(k);
    expect(sum).toBeCloseTo(1, 8);
  });

  it("cdf is non-decreasing", () => {
    let prev = 0;
    for (let k = 0; k <= 20; k++) {
      const curr = p.cdf(k);
      expect(curr).toBeGreaterThanOrEqual(prev);
      prev = curr;
    }
  });

  it("quantile and cdf are consistent", () => {
    expect(p.quantile(0.5)).toBe(3);
  });
});
