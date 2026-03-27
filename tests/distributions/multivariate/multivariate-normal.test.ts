import { MultivariateNormal } from "../../../src/distributions/multivariate/multivariate-normal";

describe("MultivariateNormal", () => {
  const mean2d = [1, 2];
  const cov2d = [
    [1, 0.5],
    [0.5, 2],
  ];
  const mvn = new MultivariateNormal(mean2d, cov2d);

  it("has correct name and dimension", () => {
    expect(mvn.name).toBe("MultivariateNormal(dim=2)");
    expect(mvn.dim).toBe(2);
  });

  it("throws for mismatched dimensions", () => {
    expect(() => new MultivariateNormal([1, 2], [[1]])).toThrow();
    expect(() => new MultivariateNormal([], [])).toThrow();
  });

  it("throws for non-positive-definite covariance", () => {
    expect(
      () =>
        new MultivariateNormal(
          [0, 0],
          [
            [-1, 0],
            [0, 1],
          ],
        ),
    ).toThrow("positive definite");
  });

  it("pdf is positive at the mean", () => {
    expect(mvn.pdf(mean2d)).toBeGreaterThan(0);
  });

  it("logPdf at mean is consistent with pdf", () => {
    expect(mvn.logPdf(mean2d)).toBeCloseTo(Math.log(mvn.pdf(mean2d)), 10);
  });

  it("pdf decreases away from mean", () => {
    const atMean = mvn.pdf(mean2d);
    const away = mvn.pdf([10, 10]);
    expect(atMean).toBeGreaterThan(away);
  });

  it("logPdf throws for wrong dimension", () => {
    expect(() => mvn.logPdf([1])).toThrow();
  });

  it("samples have correct dimension", () => {
    const s = mvn.sample();
    expect(s).toHaveLength(2);
    expect(s.every(Number.isFinite)).toBe(true);
  });

  it("sampleN returns correct count", () => {
    const samples = mvn.sampleN(100);
    expect(samples).toHaveLength(100);
    expect(samples.every((s) => s.length === 2)).toBe(true);
  });

  it("sample mean is close to true mean for large n", () => {
    const samples = mvn.sampleN(5000);
    const empiricalMean = [0, 0];
    for (const s of samples) {
      empiricalMean[0] += s[0];
      empiricalMean[1] += s[1];
    }
    empiricalMean[0] /= samples.length;
    empiricalMean[1] /= samples.length;
    expect(empiricalMean[0]).toBeCloseTo(1, 0);
    expect(empiricalMean[1]).toBeCloseTo(2, 0);
  });

  it("logLikelihood works on multiple observations", () => {
    const ll = mvn.logLikelihood([mean2d, mean2d]);
    expect(ll).toBeCloseTo(2 * mvn.logPdf(mean2d), 10);
  });

  it("works for 1d case", () => {
    const mvn1d = new MultivariateNormal([0], [[1]]);
    // Should match standard normal pdf at 0
    expect(mvn1d.pdf([0])).toBeCloseTo(1 / Math.sqrt(2 * Math.PI), 6);
  });

  it("works for 3d case", () => {
    const mvn3d = new MultivariateNormal(
      [0, 0, 0],
      [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
    );
    expect(mvn3d.dim).toBe(3);
    const s = mvn3d.sample();
    expect(s).toHaveLength(3);
  });
});
