import { MultivariateT } from "../../../src/distributions/multivariate/multivariate-t";

describe("MultivariateT", () => {
  const mu = [0, 0];
  const sigma = [
    [1, 0.5],
    [0.5, 1],
  ];
  const df = 5;
  const mvt = new MultivariateT(mu, sigma, df);

  it("has correct name and dimension", () => {
    expect(mvt.name).toBe("MultivariateT(dim=2, df=5)");
    expect(mvt.dim).toBe(2);
  });

  it("throws for empty mu", () => {
    expect(() => new MultivariateT([], [[1]], 5)).toThrow();
  });

  it("throws for non-positive df", () => {
    expect(() => new MultivariateT([0], [[1]], 0)).toThrow();
    expect(() => new MultivariateT([0], [[1]], -1)).toThrow();
  });

  it("throws for non-square sigma", () => {
    expect(() => new MultivariateT([0, 0], [[1, 0]], 5)).toThrow("square");
  });

  it("throws for sigma dimension mismatch with mu", () => {
    expect(() => new MultivariateT([0, 0], [[1]], 5)).toThrow();
  });

  it("throws for non-positive-definite sigma", () => {
    expect(
      () =>
        new MultivariateT(
          [0, 0],
          [
            [-1, 0],
            [0, 1],
          ],
          5,
        ),
    ).toThrow("positive definite");
  });

  it("mean equals mu for df > 1", () => {
    const m = mvt.mean();
    expect(m).toEqual([0, 0]);
  });

  it("mean returns a copy of mu", () => {
    const m = mvt.mean();
    m[0] = 999;
    expect(mvt.mean()[0]).toBe(0);
  });

  it("mean throws for df <= 1", () => {
    const mvt1 = new MultivariateT([0, 0], sigma, 1);
    expect(() => mvt1.mean()).toThrow("Invalid state 'df'");
    const mvt05 = new MultivariateT([0, 0], sigma, 0.5);
    expect(() => mvt05.mean()).toThrow("Invalid state 'df'");
  });

  it("covariance is (df/(df-2)) * sigma for df > 2", () => {
    const cov = mvt.covariance();
    const factor = df / (df - 2); // 5/3
    expect(cov[0][0]).toBeCloseTo(factor * sigma[0][0], 10);
    expect(cov[0][1]).toBeCloseTo(factor * sigma[0][1], 10);
    expect(cov[1][0]).toBeCloseTo(factor * sigma[1][0], 10);
    expect(cov[1][1]).toBeCloseTo(factor * sigma[1][1], 10);
  });

  it("covariance throws for df <= 2", () => {
    const mvt2 = new MultivariateT([0, 0], sigma, 2);
    expect(() => mvt2.covariance()).toThrow("Invalid state 'df'");
    const mvt1 = new MultivariateT([0, 0], sigma, 1.5);
    expect(() => mvt1.covariance()).toThrow("Invalid state 'df'");
  });

  it("logPdf returns finite value at the mode", () => {
    const lp = mvt.logPdf([0, 0]);
    expect(Number.isFinite(lp)).toBe(true);
  });

  it("pdf is consistent with logPdf", () => {
    const x = [1, -0.5];
    expect(mvt.pdf(x)).toBeCloseTo(Math.exp(mvt.logPdf(x)), 10);
  });

  it("pdf is maximum at mu", () => {
    const pdfAtMu = mvt.pdf([0, 0]);
    expect(pdfAtMu).toBeGreaterThan(mvt.pdf([1, 0]));
    expect(pdfAtMu).toBeGreaterThan(mvt.pdf([0, 1]));
    expect(pdfAtMu).toBeGreaterThan(mvt.pdf([1, 1]));
  });

  it("pdf decreases with distance from mu", () => {
    expect(mvt.pdf([0.5, 0])).toBeGreaterThan(mvt.pdf([1, 0]));
    expect(mvt.pdf([1, 0])).toBeGreaterThan(mvt.pdf([2, 0]));
    expect(mvt.pdf([2, 0])).toBeGreaterThan(mvt.pdf([5, 0]));
  });

  it("pdf is positive everywhere", () => {
    expect(mvt.pdf([0, 0])).toBeGreaterThan(0);
    expect(mvt.pdf([10, -10])).toBeGreaterThan(0);
    expect(mvt.pdf([-5, 3])).toBeGreaterThan(0);
  });

  it("logPdf throws for wrong dimension", () => {
    expect(() => mvt.logPdf([1])).toThrow();
    expect(() => mvt.logPdf([1, 2, 3])).toThrow();
  });

  it("pdf for identity sigma at origin matches known formula", () => {
    // For MultivariateT(0, I, df=5) in 2D at x=(0,0):
    // f(0) = Gamma(7/2) / (Gamma(5/2) * 5*pi * 1) * 1
    // = Gamma(3.5) / (Gamma(2.5) * 5*pi)
    const mvtId = new MultivariateT([0, 0], [[1, 0], [0, 1]], 5);
    // Gamma(3.5) = 2.5 * Gamma(2.5), so ratio = 2.5
    // f(0) = 2.5 / (5*pi) = 0.5/pi
    const expected = 0.5 / Math.PI;
    expect(mvtId.pdf([0, 0])).toBeCloseTo(expected, 5);
  });

  it("sample returns correct dimension", () => {
    const s = mvt.sample();
    expect(s).toHaveLength(2);
    expect(typeof s[0]).toBe("number");
    expect(typeof s[1]).toBe("number");
  });

  it("sampleN returns correct count", () => {
    const samples = mvt.sampleN(20);
    expect(samples).toHaveLength(20);
    for (const s of samples) {
      expect(s).toHaveLength(2);
    }
  });

  it("empirical mean converges to mu", () => {
    const n = 5000;
    const samples = mvt.sampleN(n);
    const empiricalMean = [0, 0];
    for (const s of samples) {
      empiricalMean[0] += s[0];
      empiricalMean[1] += s[1];
    }
    empiricalMean[0] /= n;
    empiricalMean[1] /= n;
    expect(empiricalMean[0]).toBeCloseTo(0, 0);
    expect(empiricalMean[1]).toBeCloseTo(0, 0);
  });

  it("works for 1d case", () => {
    const mvt1d = new MultivariateT([3], [[4]], 10);
    expect(mvt1d.dim).toBe(1);
    expect(mvt1d.mean()).toEqual([3]);
    const cov = mvt1d.covariance();
    expect(cov[0][0]).toBeCloseTo(10 / 8 * 4, 10); // df/(df-2) * 4 = 5
    const s = mvt1d.sample();
    expect(s).toHaveLength(1);
    expect(typeof s[0]).toBe("number");
  });

  it("works for 3d case", () => {
    const mu3 = [1, 2, 3];
    const sigma3 = [
      [3, 0.5, 0.1],
      [0.5, 2, 0.3],
      [0.1, 0.3, 1],
    ];
    const mvt3d = new MultivariateT(mu3, sigma3, 10);
    expect(mvt3d.dim).toBe(3);
    expect(mvt3d.name).toBe("MultivariateT(dim=3, df=10)");

    const m = mvt3d.mean();
    expect(m).toEqual([1, 2, 3]);

    const s = mvt3d.sample();
    expect(s).toHaveLength(3);

    const lp = mvt3d.logPdf([1, 2, 3]);
    expect(Number.isFinite(lp)).toBe(true);
  });

  it("has heavier tails than multivariate normal (low df)", () => {
    // With low df, the pdf at distant points should be relatively higher
    // than what a normal would give
    const mvtLowDf = new MultivariateT([0, 0], [[1, 0], [0, 1]], 2);
    const pdfAtOrigin = mvtLowDf.pdf([0, 0]);
    const pdfFar = mvtLowDf.pdf([5, 0]);
    const ratio = pdfFar / pdfAtOrigin;
    // For normal, this ratio would be exp(-12.5) ~ 3.7e-6
    // For t with df=2, it should be significantly larger
    expect(ratio).toBeGreaterThan(1e-5);
  });

  it("with non-zero mu, pdf peaks at mu", () => {
    const mvtShifted = new MultivariateT([3, -2], [[1, 0], [0, 1]], 5);
    const pdfAtMu = mvtShifted.pdf([3, -2]);
    expect(pdfAtMu).toBeGreaterThan(mvtShifted.pdf([0, 0]));
    expect(pdfAtMu).toBeGreaterThan(mvtShifted.pdf([3, 0]));
  });

  it("sample values are finite", () => {
    for (let i = 0; i < 50; i++) {
      const s = mvt.sample();
      expect(Number.isFinite(s[0])).toBe(true);
      expect(Number.isFinite(s[1])).toBe(true);
    }
  });
});
