import { Binomial } from "../../../src/distributions/discrete/binomial";

describe("Binomial distribution", () => {
  const b = new Binomial(10, 0.5);

  it("has correct mean and variance", () => {
    expect(b.mean()).toBe(5);
    expect(b.variance()).toBe(2.5);
  });

  it("pmf is symmetric for p=0.5", () => {
    expect(b.pmf(3)).toBeCloseTo(b.pmf(7), 8);
  });

  it("pmf sums to 1", () => {
    let sum = 0;
    for (let k = 0; k <= 10; k++) sum += b.pmf(k);
    expect(sum).toBeCloseTo(1, 8);
  });

  it("cdf(10) = 1", () => {
    expect(b.cdf(10)).toBeCloseTo(1, 8);
  });

  it("cdf(4) ~ 0.3770", () => {
    expect(b.cdf(4)).toBeCloseTo(0.377, 2);
  });

  it("quantile(0.5) = 5 for symmetric binomial", () => {
    expect(b.quantile(0.5)).toBe(5);
  });

  it("handles edge cases p=0 and p=1", () => {
    const b0 = new Binomial(5, 0);
    expect(b0.pmf(0)).toBe(1);
    const b1 = new Binomial(5, 1);
    expect(b1.pmf(5)).toBe(1);
  });
});
