import { Dirichlet } from "../../../src/distributions/multivariate/dirichlet";

describe("Dirichlet", () => {
  const alpha = [2, 3, 5];
  const dir = new Dirichlet(alpha);

  it("has correct name and dimension", () => {
    expect(dir.name).toBe("Dirichlet(dim=3)");
    expect(dir.dim).toBe(3);
  });

  it("throws for fewer than 2 dimensions", () => {
    expect(() => new Dirichlet([1])).toThrow("at least 2");
  });

  it("throws for non-positive alpha", () => {
    expect(() => new Dirichlet([1, 0])).toThrow("positive");
    expect(() => new Dirichlet([1, -1])).toThrow("positive");
  });

  it("mean sums to 1", () => {
    const m = dir.mean();
    const sum = m.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 10);
  });

  it("mean components are alpha_i / sum(alpha)", () => {
    const m = dir.mean();
    expect(m[0]).toBeCloseTo(0.2, 10);
    expect(m[1]).toBeCloseTo(0.3, 10);
    expect(m[2]).toBeCloseTo(0.5, 10);
  });

  it("variance components are positive", () => {
    const v = dir.variance();
    expect(v.every((vi) => vi > 0)).toBe(true);
  });

  it("pdf is positive on the simplex", () => {
    expect(dir.pdf([0.2, 0.3, 0.5])).toBeGreaterThan(0);
  });

  it("logPdf is consistent with pdf", () => {
    const x = [0.2, 0.3, 0.5];
    expect(dir.logPdf(x)).toBeCloseTo(Math.log(dir.pdf(x)), 8);
  });

  it("pdf returns 0 for points outside simplex", () => {
    expect(dir.pdf([0.5, 0.5, 0.5])).toBe(0); // sums to 1.5
    expect(dir.pdf([-0.1, 0.6, 0.5])).toBe(0); // negative
  });

  it("logPdf throws for wrong dimension", () => {
    expect(() => dir.logPdf([0.5, 0.5])).toThrow();
  });

  it("samples sum to approximately 1", () => {
    const s = dir.sample();
    expect(s).toHaveLength(3);
    const sum = s.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 6);
  });

  it("all sample components are in (0, 1)", () => {
    const s = dir.sample();
    for (const v of s) {
      expect(v).toBeGreaterThan(0);
      expect(v).toBeLessThan(1);
    }
  });

  it("sampleN returns correct count", () => {
    const samples = dir.sampleN(50);
    expect(samples).toHaveLength(50);
  });

  it("empirical mean is close to theoretical for large n", () => {
    const samples = dir.sampleN(5000);
    const empiricalMean = [0, 0, 0];
    for (const s of samples) {
      for (let i = 0; i < 3; i++) empiricalMean[i] += s[i];
    }
    for (let i = 0; i < 3; i++) empiricalMean[i] /= 5000;
    const m = dir.mean();
    for (let i = 0; i < 3; i++) {
      expect(empiricalMean[i]).toBeCloseTo(m[i], 1);
    }
  });

  it("logLikelihood works", () => {
    const x = [0.2, 0.3, 0.5];
    const ll = dir.logLikelihood([x, x]);
    expect(ll).toBeCloseTo(2 * dir.logPdf(x), 8);
  });

  it("symmetric Dirichlet (uniform on simplex) when all alpha = 1", () => {
    const uniform = new Dirichlet([1, 1, 1]);
    // PDF should be constant on the simplex = (k-1)! = 2! = 2
    expect(uniform.pdf([1 / 3, 1 / 3, 1 / 3])).toBeCloseTo(2, 5);
    expect(uniform.pdf([0.1, 0.2, 0.7])).toBeCloseTo(2, 5);
  });
});
