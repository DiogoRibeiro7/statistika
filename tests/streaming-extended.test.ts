import {
  OnlineSkewnessKurtosis,
  OnlineCorrelation,
  OnlineCovarianceMatrix,
} from "../src/streaming";

// Simple seeded PRNG (mulberry32) for reproducible tests
function mulberry32(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Batch helpers
function batchMean(data: number[]): number {
  return data.reduce((a, b) => a + b, 0) / data.length;
}

function batchVariance(data: number[]): number {
  const m = batchMean(data);
  return data.reduce((a, v) => a + (v - m) ** 2, 0) / (data.length - 1);
}

function batchSkewness(data: number[]): number {
  const n = data.length;
  const m = batchMean(data);
  const m2 = data.reduce((a, v) => a + (v - m) ** 2, 0);
  const m3 = data.reduce((a, v) => a + (v - m) ** 3, 0);
  if (m2 === 0) return 0;
  // Population skewness: G1 = sqrt(n) * M3 / M2^1.5
  const g1 = (Math.sqrt(n) * m3) / Math.pow(m2, 1.5);
  // Adjusted Fisher-Pearson: g1 * sqrt(n*(n-1)) / (n-2)
  return (g1 * Math.sqrt(n * (n - 1))) / (n - 2);
}

function batchExcessKurtosis(data: number[]): number {
  const n = data.length;
  const m = batchMean(data);
  const m2 = data.reduce((a, v) => a + (v - m) ** 2, 0);
  const m4 = data.reduce((a, v) => a + (v - m) ** 4, 0);
  if (m2 === 0) return 0;
  const kurtPop = (n * m4) / (m2 * m2);
  return ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * kurtPop - 3 * (n - 1));
}

function batchCovariance(xs: number[], ys: number[]): number {
  const mx = batchMean(xs);
  const my = batchMean(ys);
  const n = xs.length;
  let c = 0;
  for (let i = 0; i < n; i++) c += (xs[i] - mx) * (ys[i] - my);
  return c / (n - 1);
}

function batchCorrelation(xs: number[], ys: number[]): number {
  const cov = batchCovariance(xs, ys);
  const sx = Math.sqrt(batchVariance(xs));
  const sy = Math.sqrt(batchVariance(ys));
  if (sx === 0 || sy === 0) return 0;
  return cov / (sx * sy);
}

describe("Extended Streaming Statistics", () => {
  describe("OnlineSkewnessKurtosis", () => {
    it("computes correct mean and variance", () => {
      const sk = new OnlineSkewnessKurtosis();
      sk.pushAll([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
      expect(sk.count).toBe(10);
      expect(sk.mean).toBeCloseTo(5.5);
      expect(sk.variance).toBeCloseTo(9.1667, 3);
    });

    it("matches batch skewness for N=10000", () => {
      const rng = mulberry32(42);
      const data = Array.from({ length: 10000 }, () => rng());
      const sk = new OnlineSkewnessKurtosis();
      sk.pushAll(data);

      const expected = batchSkewness(data);
      expect(sk.skewness).toBeCloseTo(expected, 5);
    });

    it("matches batch kurtosis for N=10000", () => {
      const rng = mulberry32(123);
      const data = Array.from({ length: 10000 }, () => rng());
      const sk = new OnlineSkewnessKurtosis();
      sk.pushAll(data);

      const expected = batchExcessKurtosis(data);
      expect(sk.kurtosis).toBeCloseTo(expected, 5);
    });

    it("matches batch mean and variance for N=10000", () => {
      const rng = mulberry32(77);
      const data = Array.from({ length: 10000 }, () => rng() * 100 - 50);
      const sk = new OnlineSkewnessKurtosis();
      sk.pushAll(data);

      expect(sk.mean).toBeCloseTo(batchMean(data), 8);
      expect(sk.variance).toBeCloseTo(batchVariance(data), 5);
    });

    it("computes near-zero skewness for symmetric distribution", () => {
      // Symmetric data around 0
      const data: number[] = [];
      for (let i = 1; i <= 5000; i++) {
        data.push(i);
        data.push(-i);
      }
      const sk = new OnlineSkewnessKurtosis();
      sk.pushAll(data);
      expect(Math.abs(sk.skewness)).toBeLessThan(0.01);
    });

    it("computes positive skewness for right-skewed data", () => {
      const rng = mulberry32(99);
      // Exponential-like: -ln(U)
      const data = Array.from({ length: 10000 }, () => -Math.log(rng()));
      const sk = new OnlineSkewnessKurtosis();
      sk.pushAll(data);
      expect(sk.skewness).toBeGreaterThan(0.5);
    });

    it("returns 0 skewness for identical values", () => {
      const sk = new OnlineSkewnessKurtosis();
      sk.pushAll([5, 5, 5, 5, 5]);
      expect(sk.skewness).toBe(0);
    });

    it("returns 0 kurtosis for identical values", () => {
      const sk = new OnlineSkewnessKurtosis();
      sk.pushAll([5, 5, 5, 5, 5]);
      expect(sk.kurtosis).toBe(0);
    });

    it("throws on no observations for mean", () => {
      const sk = new OnlineSkewnessKurtosis();
      expect(() => sk.mean).toThrow("expected at least 1 observation");
    });

    it("throws on fewer than 2 for variance", () => {
      const sk = new OnlineSkewnessKurtosis();
      sk.push(1);
      expect(() => sk.variance).toThrow("at least 2");
    });

    it("throws on fewer than 3 for skewness", () => {
      const sk = new OnlineSkewnessKurtosis();
      sk.pushAll([1, 2]);
      expect(() => sk.skewness).toThrow("at least 3");
    });

    it("throws on fewer than 4 for kurtosis", () => {
      const sk = new OnlineSkewnessKurtosis();
      sk.pushAll([1, 2, 3]);
      expect(() => sk.kurtosis).toThrow("at least 4");
    });

    it("throws on NaN push", () => {
      const sk = new OnlineSkewnessKurtosis();
      expect(() => sk.push(NaN)).toThrow("NaN");
    });

    it("resets correctly", () => {
      const sk = new OnlineSkewnessKurtosis();
      sk.pushAll([1, 2, 3, 4, 5]);
      sk.reset();
      expect(sk.count).toBe(0);
      expect(() => sk.mean).toThrow("expected at least 1 observation");
    });
  });

  describe("OnlineCorrelation", () => {
    it("computes correlation for perfectly correlated data", () => {
      const corr = new OnlineCorrelation();
      for (let i = 0; i < 10; i++) corr.push(i, 2 * i + 3);
      expect(corr.correlation).toBeCloseTo(1);
    });

    it("computes correlation for negatively correlated data", () => {
      const corr = new OnlineCorrelation();
      for (let i = 0; i < 10; i++) corr.push(i, -3 * i + 10);
      expect(corr.correlation).toBeCloseTo(-1);
    });

    it("matches batch correlation for N=10000", () => {
      const rng = mulberry32(55);
      const xs = Array.from({ length: 10000 }, () => rng());
      const ys = xs.map((x) => 2 * x + 0.5 * (rng() - 0.5));

      const corr = new OnlineCorrelation();
      corr.pushAll(xs, ys);

      expect(corr.correlation).toBeCloseTo(batchCorrelation(xs, ys), 8);
      expect(corr.meanX).toBeCloseTo(batchMean(xs), 8);
      expect(corr.meanY).toBeCloseTo(batchMean(ys), 8);
      expect(corr.covariance).toBeCloseTo(batchCovariance(xs, ys), 8);
    });

    it("computes near-zero correlation for uncorrelated data", () => {
      const corr = new OnlineCorrelation();
      const xs = [1, 0, -1, 0, 1, 0, -1, 0];
      const ys = [0, 1, 0, -1, 0, 1, 0, -1];
      corr.pushAll(xs, ys);
      expect(Math.abs(corr.correlation)).toBeLessThan(0.1);
    });

    it("returns 0 correlation for identical values", () => {
      const corr = new OnlineCorrelation();
      corr.pushAll([5, 5, 5], [5, 5, 5]);
      expect(corr.correlation).toBe(0);
    });

    it("throws on no observations for meanX", () => {
      const corr = new OnlineCorrelation();
      expect(() => corr.meanX).toThrow("expected at least 1 observation");
    });

    it("throws on no observations for meanY", () => {
      const corr = new OnlineCorrelation();
      expect(() => corr.meanY).toThrow("expected at least 1 observation");
    });

    it("throws on fewer than 2 for correlation", () => {
      const corr = new OnlineCorrelation();
      corr.push(1, 2);
      expect(() => corr.correlation).toThrow("at least 2");
    });

    it("throws on fewer than 2 for covariance", () => {
      const corr = new OnlineCorrelation();
      corr.push(1, 2);
      expect(() => corr.covariance).toThrow("at least 2");
    });

    it("throws on NaN push", () => {
      const corr = new OnlineCorrelation();
      expect(() => corr.push(NaN, 1)).toThrow("NaN");
      expect(() => corr.push(1, NaN)).toThrow("NaN");
    });

    it("throws on mismatched array lengths", () => {
      const corr = new OnlineCorrelation();
      expect(() => corr.pushAll([1, 2], [1])).toThrow("same length");
    });

    it("resets correctly", () => {
      const corr = new OnlineCorrelation();
      corr.pushAll([1, 2, 3], [4, 5, 6]);
      corr.reset();
      expect(corr.count).toBe(0);
      expect(() => corr.meanX).toThrow("expected at least 1 observation");
    });
  });

  describe("OnlineCovarianceMatrix", () => {
    it("computes correct covariance matrix for 2D data", () => {
      const mat = new OnlineCovarianceMatrix(2);
      // x = [1,2,3,4,5], y = [2,4,6,8,10]
      mat.pushAll([
        [1, 2],
        [2, 4],
        [3, 6],
        [4, 8],
        [5, 10],
      ]);
      const cov = mat.covarianceMatrix;
      // Var(x)=2.5, Var(y)=10, Cov(x,y)=5
      expect(cov[0][0]).toBeCloseTo(2.5);
      expect(cov[1][1]).toBeCloseTo(10);
      expect(cov[0][1]).toBeCloseTo(5);
      expect(cov[1][0]).toBeCloseTo(5); // symmetric
    });

    it("computes correct correlation matrix for 2D data", () => {
      const mat = new OnlineCovarianceMatrix(2);
      mat.pushAll([
        [1, 2],
        [2, 4],
        [3, 6],
        [4, 8],
        [5, 10],
      ]);
      const corr = mat.correlationMatrix;
      expect(corr[0][0]).toBeCloseTo(1);
      expect(corr[1][1]).toBeCloseTo(1);
      expect(corr[0][1]).toBeCloseTo(1); // perfect correlation
      expect(corr[1][0]).toBeCloseTo(1);
    });

    it("matches batch computation for N=10000 with 3 variables", () => {
      const rng = mulberry32(88);
      const N = 10000;
      const rows: number[][] = [];
      const cols: number[][] = [[], [], []];

      for (let i = 0; i < N; i++) {
        const x = rng();
        const y = 2 * x + 0.1 * rng();
        const z = -x + 0.5 * rng();
        rows.push([x, y, z]);
        cols[0].push(x);
        cols[1].push(y);
        cols[2].push(z);
      }

      const mat = new OnlineCovarianceMatrix(3);
      mat.pushAll(rows);

      // Verify means
      const means = mat.means;
      expect(means[0]).toBeCloseTo(batchMean(cols[0]), 8);
      expect(means[1]).toBeCloseTo(batchMean(cols[1]), 8);
      expect(means[2]).toBeCloseTo(batchMean(cols[2]), 8);

      // Verify covariance matrix entries against batch
      const cov = mat.covarianceMatrix;
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          const batchCov = batchCovariance(cols[i], cols[j]);
          expect(cov[i][j]).toBeCloseTo(batchCov, 5);
        }
      }

      // Verify correlation matrix entries against batch
      const corr = mat.correlationMatrix;
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 3; j++) {
          if (i === j) {
            expect(corr[i][j]).toBeCloseTo(1);
          } else {
            const batchCorr = batchCorrelation(cols[i], cols[j]);
            expect(corr[i][j]).toBeCloseTo(batchCorr, 5);
          }
        }
      }
    });

    it("covariance matrix is symmetric", () => {
      const rng = mulberry32(12);
      const mat = new OnlineCovarianceMatrix(4);
      for (let i = 0; i < 100; i++) {
        mat.push([rng(), rng(), rng(), rng()]);
      }
      const cov = mat.covarianceMatrix;
      for (let i = 0; i < 4; i++) {
        for (let j = 0; j < 4; j++) {
          expect(cov[i][j]).toBeCloseTo(cov[j][i], 10);
        }
      }
    });

    it("correlation matrix diagonal is 1", () => {
      const rng = mulberry32(34);
      const mat = new OnlineCovarianceMatrix(3);
      for (let i = 0; i < 50; i++) {
        mat.push([rng(), rng(), rng()]);
      }
      const corr = mat.correlationMatrix;
      for (let i = 0; i < 3; i++) {
        expect(corr[i][i]).toBeCloseTo(1);
      }
    });

    it("returns 0 correlation for zero-variance variables", () => {
      const mat = new OnlineCovarianceMatrix(2);
      mat.pushAll([
        [5, 1],
        [5, 2],
        [5, 3],
      ]);
      const corr = mat.correlationMatrix;
      // x has zero variance
      expect(corr[0][1]).toBe(0);
      expect(corr[1][0]).toBe(0);
    });

    it("throws on invalid dimension", () => {
      expect(() => new OnlineCovarianceMatrix(0)).toThrow("positive integer");
      expect(() => new OnlineCovarianceMatrix(-1)).toThrow("positive integer");
      expect(() => new OnlineCovarianceMatrix(1.5)).toThrow("positive integer");
    });

    it("throws on wrong vector length", () => {
      const mat = new OnlineCovarianceMatrix(3);
      expect(() => mat.push([1, 2])).toThrow("Expected 3 values, got 2");
    });

    it("throws on NaN value", () => {
      const mat = new OnlineCovarianceMatrix(2);
      expect(() => mat.push([1, NaN])).toThrow("NaN");
    });

    it("throws on no observations for means", () => {
      const mat = new OnlineCovarianceMatrix(2);
      expect(() => mat.means).toThrow("expected at least 1 observation");
    });

    it("throws on fewer than 2 for covarianceMatrix", () => {
      const mat = new OnlineCovarianceMatrix(2);
      mat.push([1, 2]);
      expect(() => mat.covarianceMatrix).toThrow("at least 2");
    });

    it("throws on fewer than 2 for correlationMatrix", () => {
      const mat = new OnlineCovarianceMatrix(2);
      mat.push([1, 2]);
      expect(() => mat.correlationMatrix).toThrow("at least 2");
    });

    it("works with dimension 1", () => {
      const mat = new OnlineCovarianceMatrix(1);
      mat.pushAll([[1], [2], [3], [4], [5]]);
      expect(mat.means).toEqual([3]);
      expect(mat.covarianceMatrix[0][0]).toBeCloseTo(2.5);
      expect(mat.correlationMatrix[0][0]).toBeCloseTo(1);
    });

    it("resets correctly", () => {
      const mat = new OnlineCovarianceMatrix(2);
      mat.pushAll([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      mat.reset();
      expect(mat.count).toBe(0);
      expect(() => mat.means).toThrow("expected at least 1 observation");
    });

    it("dim getter returns correct dimension", () => {
      const mat = new OnlineCovarianceMatrix(5);
      expect(mat.dim).toBe(5);
    });
  });
});
