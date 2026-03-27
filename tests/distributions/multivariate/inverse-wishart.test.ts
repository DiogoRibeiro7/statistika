import { InverseWishart } from "../../../src/distributions/multivariate/inverse-wishart";

describe("InverseWishart", () => {
  const scale = [
    [2, 0.5],
    [0.5, 1],
  ];
  const df = 6;
  const iw = new InverseWishart(df, scale);

  it("has correct name and dimension", () => {
    expect(iw.name).toBe("InverseWishart(df=6, dim=2)");
    expect(iw.dim).toBe(2);
  });

  it("throws for df <= dim - 1", () => {
    expect(() => new InverseWishart(1, scale)).toThrow();
    expect(() => new InverseWishart(0, scale)).toThrow();
  });

  it("throws for non-square scale matrix", () => {
    expect(() => new InverseWishart(3, [[1, 0]])).toThrow("square");
  });

  it("throws for non-positive-definite scale", () => {
    expect(
      () =>
        new InverseWishart(5, [
          [-1, 0],
          [0, 1],
        ]),
    ).toThrow("positive definite");
  });

  it("throws for empty scale matrix", () => {
    expect(() => new InverseWishart(5, [])).toThrow();
  });

  it("mean is Psi / (df - p - 1) for df > p + 1", () => {
    // df=6, p=2, denom = 6-2-1 = 3
    const m = iw.mean();
    expect(m[0][0]).toBeCloseTo(scale[0][0] / 3, 10);
    expect(m[0][1]).toBeCloseTo(scale[0][1] / 3, 10);
    expect(m[1][0]).toBeCloseTo(scale[1][0] / 3, 10);
    expect(m[1][1]).toBeCloseTo(scale[1][1] / 3, 10);
  });

  it("mean throws for df <= p + 1", () => {
    const iwSmallDf = new InverseWishart(3, scale); // df=3, p=2, need df > 3
    expect(() => iwSmallDf.mean()).toThrow("expected df");
  });

  it("logPdf returns finite value for positive-definite matrix", () => {
    const X = [
      [3, 0.2],
      [0.2, 2],
    ];
    const lp = iw.logPdf(X);
    expect(Number.isFinite(lp)).toBe(true);
  });

  it("pdf is consistent with logPdf", () => {
    const X = [
      [3, 0.2],
      [0.2, 2],
    ];
    expect(iw.pdf(X)).toBeCloseTo(Math.exp(iw.logPdf(X)), 10);
  });

  it("logPdf returns -Infinity for non-positive-definite X", () => {
    const X = [
      [-1, 0],
      [0, 1],
    ];
    expect(iw.logPdf(X)).toBe(-Infinity);
  });

  it("logPdf throws for wrong dimensions", () => {
    expect(() => iw.logPdf([[1]])).toThrow();
  });

  it("pdf is positive for valid positive-definite matrices", () => {
    const X = [
      [5, 1],
      [1, 3],
    ];
    expect(iw.pdf(X)).toBeGreaterThan(0);
  });

  it("sample returns a symmetric matrix", () => {
    const S = iw.sample();
    expect(S).toHaveLength(2);
    expect(S[0]).toHaveLength(2);
    expect(S[0][1]).toBeCloseTo(S[1][0], 10);
  });

  it("sample is positive definite (positive diagonal and determinant)", () => {
    const S = iw.sample();
    expect(S[0][0]).toBeGreaterThan(0);
    expect(S[1][1]).toBeGreaterThan(0);
    const det = S[0][0] * S[1][1] - S[0][1] * S[1][0];
    expect(det).toBeGreaterThan(0);
  });

  it("sampleN returns correct count", () => {
    const samples = iw.sampleN(20);
    expect(samples).toHaveLength(20);
  });

  it("empirical mean is close to theoretical for large n", () => {
    const n = 3000;
    const samples = iw.sampleN(n);
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
    const expected = iw.mean();
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        empiricalMean[i][j] /= n;
        expect(empiricalMean[i][j]).toBeCloseTo(expected[i][j], 0);
      }
    }
  });

  it("works for 1d case", () => {
    const iw1d = new InverseWishart(5, [[2]]);
    expect(iw1d.dim).toBe(1);
    const S = iw1d.sample();
    expect(S).toHaveLength(1);
    expect(S[0][0]).toBeGreaterThan(0);
    // mean = 2 / (5 - 1 - 1) = 2/3
    const m = iw1d.mean();
    expect(m[0][0]).toBeCloseTo(2 / 3, 10);
  });

  it("works for 3d case", () => {
    const scale3 = [
      [3, 0.5, 0.1],
      [0.5, 2, 0.3],
      [0.1, 0.3, 1],
    ];
    const iw3d = new InverseWishart(8, scale3);
    expect(iw3d.dim).toBe(3);
    expect(iw3d.name).toBe("InverseWishart(df=8, dim=3)");

    const S = iw3d.sample();
    expect(S).toHaveLength(3);
    expect(S[0]).toHaveLength(3);

    // Symmetry check
    expect(S[0][1]).toBeCloseTo(S[1][0], 10);
    expect(S[0][2]).toBeCloseTo(S[2][0], 10);
    expect(S[1][2]).toBeCloseTo(S[2][1], 10);

    // Positive diagonal
    expect(S[0][0]).toBeGreaterThan(0);
    expect(S[1][1]).toBeGreaterThan(0);
    expect(S[2][2]).toBeGreaterThan(0);
  });

  it("logPdf is higher for matrices closer to the mode", () => {
    // For InverseWishart, the mode is Psi / (df + p + 1)
    // df=6, p=2: mode = scale / 9
    const mode = scale.map((row) => row.map((v) => v / 9));
    const farFromMode = [
      [10, 0],
      [0, 10],
    ];
    expect(iw.logPdf(mode)).toBeGreaterThan(iw.logPdf(farFromMode));
  });

  it("pdf values differ for different positive-definite matrices", () => {
    const X1 = [
      [2, 0],
      [0, 2],
    ];
    const X2 = [
      [5, 0],
      [0, 5],
    ];
    expect(iw.pdf(X1)).not.toBeCloseTo(iw.pdf(X2), 5);
  });
});
