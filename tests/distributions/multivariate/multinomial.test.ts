import { Multinomial } from "../../../src/distributions/multivariate/multinomial";

describe("Multinomial distribution", () => {
  const m = new Multinomial(10, [0.2, 0.3, 0.5]);

  it("has correct name", () => {
    expect(m.name).toBe("Multinomial(10, k=3)");
  });

  it("mean = n * p", () => {
    const mu = m.mean();
    expect(mu[0]).toBeCloseTo(2, 10);
    expect(mu[1]).toBeCloseTo(3, 10);
    expect(mu[2]).toBeCloseTo(5, 10);
  });

  it("variance = n * p * (1-p)", () => {
    const v = m.variance();
    expect(v[0]).toBeCloseTo(10 * 0.2 * 0.8, 10);
    expect(v[1]).toBeCloseTo(10 * 0.3 * 0.7, 10);
    expect(v[2]).toBeCloseTo(10 * 0.5 * 0.5, 10);
  });

  it("covariance(i, j) = -n * p_i * p_j for i != j", () => {
    expect(m.covariance(0, 1)).toBeCloseTo(-10 * 0.2 * 0.3, 10);
    expect(m.covariance(0, 0)).toBeCloseTo(10 * 0.2 * 0.8, 10);
  });

  it("pmf of valid outcome", () => {
    // P(2, 3, 5) for Multinomial(10, [0.2, 0.3, 0.5])
    const p = m.pmf([2, 3, 5]);
    expect(p).toBeGreaterThan(0);
    expect(p).toBeLessThanOrEqual(1);
  });

  it("pmf returns 0 for counts not summing to n", () => {
    expect(m.pmf([1, 1, 1])).toBe(0); // sum = 3 != 10
  });

  it("pmf returns 0 for negative counts", () => {
    expect(m.pmf([-1, 5, 6])).toBe(0);
  });

  it("pmf returns 0 for non-integer counts", () => {
    expect(m.pmf([2.5, 3, 4.5])).toBe(0);
  });

  it("logPmf throws for wrong length", () => {
    expect(() => m.logPmf([1, 2])).toThrow("length");
  });

  it("sample returns valid counts", () => {
    for (let i = 0; i < 20; i++) {
      const s = m.sample();
      expect(s.length).toBe(3);
      const sum = s.reduce((a, b) => a + b, 0);
      expect(sum).toBe(10);
      for (const c of s) {
        expect(c).toBeGreaterThanOrEqual(0);
        expect(Number.isInteger(c)).toBe(true);
      }
    }
  });

  it("sampleN returns correct count", () => {
    expect(m.sampleN(5).length).toBe(5);
  });

  it("logLikelihood is sum of logPmf", () => {
    const data = [[2, 3, 5], [4, 2, 4], [1, 5, 4]];
    const ll = m.logLikelihood(data);
    const expected = data.reduce((s, x) => s + m.logPmf(x), 0);
    expect(ll).toBeCloseTo(expected, 10);
  });

  it("throws on invalid parameters", () => {
    expect(() => new Multinomial(0, [0.5, 0.5])).toThrow("positive integer");
    expect(() => new Multinomial(5, [0.5])).toThrow("at least 2");
    expect(() => new Multinomial(5, [-0.1, 1.1])).toThrow("non-negative");
    expect(() => new Multinomial(5, [0.3, 0.3])).toThrow("sum to 1");
  });

  it("handles zero-probability category", () => {
    const m2 = new Multinomial(5, [0.5, 0, 0.5]);
    expect(m2.pmf([3, 0, 2])).toBeGreaterThan(0);
    expect(m2.pmf([2, 1, 2])).toBe(0); // positive count for zero-prob category
  });
});
