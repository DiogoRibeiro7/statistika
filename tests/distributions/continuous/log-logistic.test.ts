import { LogLogistic } from "../../../src/distributions/continuous/log-logistic";

describe("LogLogistic distribution", () => {
  const ll = new LogLogistic(1, 2); // alpha=1, beta=2

  it("has correct name", () => {
    expect(ll.name).toBe("LogLogistic(1, 2)");
  });

  it("throws on non-positive alpha", () => {
    expect(() => new LogLogistic(0, 1)).toThrow();
    expect(() => new LogLogistic(-1, 1)).toThrow();
  });

  it("throws on non-positive beta", () => {
    expect(() => new LogLogistic(1, 0)).toThrow();
    expect(() => new LogLogistic(1, -1)).toThrow();
  });

  it("mean is alpha * pi/beta / sin(pi/beta) for beta > 1", () => {
    // alpha=1, beta=2 => mean = 1 * (pi/2) / sin(pi/2) = pi/2
    const expected = Math.PI / 2;
    expect(ll.mean()).toBeCloseTo(expected, 10);
  });

  it("mean is Infinity for beta <= 1", () => {
    const ll1 = new LogLogistic(1, 1);
    expect(ll1.mean()).toBe(Infinity);
    const ll05 = new LogLogistic(1, 0.5);
    expect(ll05.mean()).toBe(Infinity);
  });

  it("variance is finite for beta > 2", () => {
    const ll3 = new LogLogistic(1, 3);
    const v = ll3.variance();
    expect(Number.isFinite(v)).toBe(true);
    expect(v).toBeGreaterThan(0);
  });

  it("variance is Infinity for beta <= 2", () => {
    expect(ll.variance()).toBe(Infinity);
    const ll1 = new LogLogistic(1, 1);
    expect(ll1.variance()).toBe(Infinity);
  });

  it("pdf is 0 for x < 0", () => {
    expect(ll.pdf(-1)).toBe(0);
    expect(ll.pdf(-0.001)).toBe(0);
  });

  it("pdf at x=0 with beta > 1 is 0", () => {
    expect(ll.pdf(0)).toBe(0);
  });

  it("pdf at x=0 with beta=1 returns 1/alpha", () => {
    const ll1 = new LogLogistic(2, 1);
    expect(ll1.pdf(0)).toBeCloseTo(0.5, 10);
  });

  it("pdf at known values", () => {
    // For LogLogistic(1, 2): f(1) = (2/1)*(1/1)^1 / (1+1)^2 = 2/4 = 0.5
    expect(ll.pdf(1)).toBeCloseTo(0.5, 10);
  });

  it("pdf is positive for x > 0", () => {
    expect(ll.pdf(0.5)).toBeGreaterThan(0);
    expect(ll.pdf(1)).toBeGreaterThan(0);
    expect(ll.pdf(10)).toBeGreaterThan(0);
  });

  it("cdf is 0 for x <= 0", () => {
    expect(ll.cdf(0)).toBe(0);
    expect(ll.cdf(-1)).toBe(0);
  });

  it("cdf at median equals 0.5", () => {
    // Median of LogLogistic(alpha, beta) = alpha
    expect(ll.cdf(1)).toBeCloseTo(0.5, 10);
  });

  it("cdf at known values", () => {
    // F(2; 1, 2) = 2^2 / (1 + 2^2) = 4/5 = 0.8
    expect(ll.cdf(2)).toBeCloseTo(0.8, 10);
  });

  it("cdf increases with x", () => {
    expect(ll.cdf(0.5)).toBeLessThan(ll.cdf(1));
    expect(ll.cdf(1)).toBeLessThan(ll.cdf(2));
  });

  it("cdf approaches 1 for large x", () => {
    expect(ll.cdf(1000)).toBeGreaterThan(0.999);
  });

  it("cdf and quantile are inverses", () => {
    for (const p of [0.1, 0.25, 0.5, 0.75, 0.9]) {
      expect(ll.cdf(ll.quantile(p))).toBeCloseTo(p, 10);
    }
  });

  it("quantile(0) = 0", () => {
    expect(ll.quantile(0)).toBe(0);
  });

  it("quantile(1) = Infinity", () => {
    expect(ll.quantile(1)).toBe(Infinity);
  });

  it("quantile(0.5) = alpha (median)", () => {
    expect(ll.quantile(0.5)).toBeCloseTo(1, 10);
  });

  it("quantile at known values", () => {
    // Q(0.8; 1, 2) = 1 * (0.8/0.2)^(1/2) = sqrt(4) = 2
    expect(ll.quantile(0.8)).toBeCloseTo(2, 10);
  });

  it("quantile throws on out-of-range p", () => {
    expect(() => ll.quantile(-0.1)).toThrow();
    expect(() => ll.quantile(1.1)).toThrow();
  });

  it("sf(x) = 1 - cdf(x)", () => {
    expect(ll.sf(1)).toBeCloseTo(1 - ll.cdf(1), 10);
    expect(ll.sf(2)).toBeCloseTo(1 - ll.cdf(2), 10);
  });

  it("sample returns non-negative values", () => {
    for (let i = 0; i < 50; i++) {
      expect(ll.sample()).toBeGreaterThanOrEqual(0);
    }
  });

  it("sampleN returns correct number of samples", () => {
    expect(ll.sampleN(10).length).toBe(10);
  });

  it("works with non-standard parameters", () => {
    const ll2 = new LogLogistic(3, 5);
    expect(ll2.name).toBe("LogLogistic(3, 5)");
    expect(ll2.quantile(0.5)).toBeCloseTo(3, 10);
    expect(ll2.cdf(3)).toBeCloseTo(0.5, 10);
    expect(ll2.pdf(3)).toBeGreaterThan(0);
  });
});
