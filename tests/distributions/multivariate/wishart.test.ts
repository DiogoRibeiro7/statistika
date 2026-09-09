import { Wishart } from "../../../src/distributions/multivariate/wishart";

describe("Wishart", () => {
  const scale = [
    [2, 0.5],
    [0.5, 1],
  ];
  const df = 5;
  const w = new Wishart(df, scale);

  it("has correct name and dimension", () => {
    expect(w.name).toBe("Wishart(df=5, dim=2)");
    expect(w.dim).toBe(2);
  });

  it("throws for df < dim", () => {
    expect(() => new Wishart(1, scale)).toThrow("Degrees of freedom");
  });

  it("throws for non-square scale matrix", () => {
    expect(() => new Wishart(3, [[1, 0]])).toThrow("square");
  });

  it("throws for non-positive-definite scale", () => {
    expect(
      () =>
        new Wishart(2, [
          [-1, 0],
          [0, 1],
        ]),
    ).toThrow("positive definite");
  });

  it("mean is df * scale", () => {
    const m = w.mean();
    expect(m[0][0]).toBeCloseTo(df * scale[0][0], 10);
    expect(m[0][1]).toBeCloseTo(df * scale[0][1], 10);
    expect(m[1][0]).toBeCloseTo(df * scale[1][0], 10);
    expect(m[1][1]).toBeCloseTo(df * scale[1][1], 10);
  });

  it("logPdf returns finite value for positive-definite matrix", () => {
    const X = [
      [5, 1],
      [1, 3],
    ];
    const lp = w.logPdf(X);
    expect(Number.isFinite(lp)).toBe(true);
  });

  it("pdf is consistent with logPdf", () => {
    const X = [
      [5, 1],
      [1, 3],
    ];
    expect(w.pdf(X)).toBeCloseTo(Math.exp(w.logPdf(X)), 10);
  });

  it("logPdf returns -Infinity for non-positive-definite X", () => {
    const X = [
      [-1, 0],
      [0, 1],
    ];
    expect(w.logPdf(X)).toBe(-Infinity);
  });

  it("logPdf throws for wrong dimensions", () => {
    expect(() => w.logPdf([[1]])).toThrow();
  });

  it("sample returns a symmetric matrix", () => {
    const S = w.sample();
    expect(S).toHaveLength(2);
    expect(S[0]).toHaveLength(2);
    expect(S[0][1]).toBeCloseTo(S[1][0], 10);
  });

  it("sample is positive definite (positive diagonal)", () => {
    const S = w.sample();
    expect(S[0][0]).toBeGreaterThan(0);
    expect(S[1][1]).toBeGreaterThan(0);
    // Determinant should be positive
    const det = S[0][0] * S[1][1] - S[0][1] * S[1][0];
    expect(det).toBeGreaterThan(0);
  });

  it("sampleN returns correct count", () => {
    const samples = w.sampleN(20);
    expect(samples).toHaveLength(20);
  });

  it("empirical mean is close to theoretical for large n", () => {
    const samples = w.sampleN(2000);
    const empiricalMean = [
      [0, 0],
      [0, 0],
    ];
    for (const S of samples) {
      for (let i = 0; i < 2; i++) {
        for (let j = 0; j < 2; j++) {
          empiricalMean[i][j] += S[i][j];
        }
      }
    }
    const expected = w.mean();
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        empiricalMean[i][j] /= 2000;
        expect(empiricalMean[i][j]).toBeCloseTo(expected[i][j], 0);
      }
    }
  });

  it("works for 1d case (chi-squared)", () => {
    const w1d = new Wishart(10, [[2]]);
    const S = w1d.sample();
    expect(S).toHaveLength(1);
    expect(S[0][0]).toBeGreaterThan(0);
  });
});
