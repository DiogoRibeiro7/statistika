import { OnlineStats, OnlineCovariance, OnlineQuantile } from "../src/streaming";

describe("Streaming/Online Statistics", () => {
  describe("OnlineStats", () => {
    it("computes correct mean and variance", () => {
      const stats = new OnlineStats();
      stats.pushAll([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
      expect(stats.count).toBe(10);
      expect(stats.mean).toBeCloseTo(5.5);
      expect(stats.variance).toBeCloseTo(9.1667, 3);
      expect(stats.stdDev).toBeCloseTo(Math.sqrt(9.1667), 2);
      expect(stats.min).toBe(1);
      expect(stats.max).toBe(10);
    });

    it("matches batch computation", () => {
      const data = [3.5, 7.2, 1.8, 9.1, 4.4, 6.6, 2.3, 8.9, 5.0, 7.7];
      const stats = new OnlineStats();
      stats.pushAll(data);

      const batchMean = data.reduce((a, b) => a + b, 0) / data.length;
      const batchVar =
        data.reduce((a, v) => a + (v - batchMean) ** 2, 0) / (data.length - 1);

      expect(stats.mean).toBeCloseTo(batchMean, 10);
      expect(stats.variance).toBeCloseTo(batchVar, 10);
    });

    it("merges two instances correctly", () => {
      const a = new OnlineStats();
      a.pushAll([1, 2, 3, 4, 5]);

      const b = new OnlineStats();
      b.pushAll([6, 7, 8, 9, 10]);

      a.merge(b);
      expect(a.count).toBe(10);
      expect(a.mean).toBeCloseTo(5.5);
      expect(a.min).toBe(1);
      expect(a.max).toBe(10);
    });

    it("resets correctly", () => {
      const stats = new OnlineStats();
      stats.pushAll([1, 2, 3]);
      stats.reset();
      expect(stats.count).toBe(0);
      expect(() => stats.mean).toThrow("expected at least 1 observation");
    });
  });

  describe("OnlineCovariance", () => {
    it("computes correlation for perfectly correlated data", () => {
      const cov = new OnlineCovariance();
      for (let i = 0; i < 10; i++) cov.push(i, 2 * i);
      expect(cov.correlation).toBeCloseTo(1);
    });

    it("computes correlation for negatively correlated data", () => {
      const cov = new OnlineCovariance();
      for (let i = 0; i < 10; i++) cov.push(i, -i);
      expect(cov.correlation).toBeCloseTo(-1);
    });

    it("computes near-zero correlation for uncorrelated data", () => {
      const cov = new OnlineCovariance();
      // Orthogonal signals
      const xs = [1, 0, -1, 0, 1, 0, -1, 0];
      const ys = [0, 1, 0, -1, 0, 1, 0, -1];
      cov.pushAll(xs, ys);
      expect(Math.abs(cov.correlation)).toBeLessThan(0.1);
    });

    it("throws on insufficient data", () => {
      const cov = new OnlineCovariance();
      cov.push(1, 2);
      expect(() => cov.covariance).toThrow("at least 2");
    });
  });

  describe("OnlineQuantile", () => {
    it("estimates median accurately", () => {
      const q = new OnlineQuantile(0.5);
      const data = Array.from({ length: 1000 }, (_, i) => i);
      q.pushAll(data);
      // Median of 0..999 should be ~499.5
      expect(q.estimate).toBeCloseTo(499.5, -1); // within ~10
    });

    it("estimates 90th percentile", () => {
      const q = new OnlineQuantile(0.9);
      const data = Array.from({ length: 1000 }, (_, i) => i);
      q.pushAll(data);
      expect(q.estimate).toBeCloseTo(900, -1); // within ~10
    });

    it("works with small datasets", () => {
      const q = new OnlineQuantile(0.5);
      q.pushAll([1, 3, 5]);
      expect(q.estimate).toBe(3);
    });

    it("throws on invalid quantile", () => {
      expect(() => new OnlineQuantile(0)).toThrow("Invalid parameter 'quantile'");
      expect(() => new OnlineQuantile(1)).toThrow("Invalid parameter 'quantile'");
    });

    it("throws on no observations", () => {
      const q = new OnlineQuantile(0.5);
      expect(() => q.estimate).toThrow("expected at least 1 observation");
    });

    it("throws on NaN push", () => {
      const q = new OnlineQuantile(0.5);
      expect(() => q.push(NaN)).toThrow("NaN");
    });

    it("works with exactly 5 observations", () => {
      const q = new OnlineQuantile(0.5);
      q.pushAll([5, 3, 1, 4, 2]);
      expect(q.estimate).toBe(3); // median of sorted [1,2,3,4,5]
    });

    it("estimates 10th percentile", () => {
      const q = new OnlineQuantile(0.1);
      const data = Array.from({ length: 1000 }, (_, i) => i);
      q.pushAll(data);
      expect(q.estimate).toBeCloseTo(100, -1.5);
    });

    it("count tracks total observations", () => {
      const q = new OnlineQuantile(0.5);
      q.push(1);
      q.push(2);
      q.pushAll([3, 4, 5]);
      expect(q.count).toBe(5);
    });
  });

  describe("OnlineStats edge cases", () => {
    it("throws on NaN push", () => {
      const stats = new OnlineStats();
      expect(() => stats.push(NaN)).toThrow("NaN");
    });

    it("throws on NaN in pushAll", () => {
      const stats = new OnlineStats();
      expect(() => stats.pushAll([1, NaN, 3])).toThrow("NaN");
    });

    it("throws on mean with no observations", () => {
      const stats = new OnlineStats();
      expect(() => stats.mean).toThrow("expected at least 1 observation");
    });

    it("throws on variance with fewer than 2 observations", () => {
      const stats = new OnlineStats();
      stats.push(5);
      expect(() => stats.variance).toThrow("at least 2");
    });

    it("throws on min with no observations", () => {
      const stats = new OnlineStats();
      expect(() => stats.min).toThrow("expected at least 1 observation");
    });

    it("throws on max with no observations", () => {
      const stats = new OnlineStats();
      expect(() => stats.max).toThrow("expected at least 1 observation");
    });

    it("computes populationVariance", () => {
      const stats = new OnlineStats();
      stats.pushAll([2, 4, 6, 8, 10]);
      // population variance of [2,4,6,8,10] with mean=6: ((16+4+0+4+16)/5) = 8
      expect(stats.populationVariance).toBeCloseTo(8);
    });

    it("throws on populationVariance with no observations", () => {
      const stats = new OnlineStats();
      expect(() => stats.populationVariance).toThrow("expected at least 1 observation");
    });

    it("merge with empty OnlineStats does nothing", () => {
      const a = new OnlineStats();
      a.pushAll([1, 2, 3]);
      const b = new OnlineStats();
      a.merge(b);
      expect(a.count).toBe(3);
      expect(a.mean).toBeCloseTo(2);
    });

    it("merge into empty OnlineStats copies data", () => {
      const a = new OnlineStats();
      const b = new OnlineStats();
      b.pushAll([10, 20, 30]);
      a.merge(b);
      expect(a.count).toBe(3);
      expect(a.mean).toBeCloseTo(20);
      expect(a.min).toBe(10);
      expect(a.max).toBe(30);
    });

    it("push one at a time matches pushAll", () => {
      const data = [3, 7, 11, 15, 19];
      const s1 = new OnlineStats();
      for (const v of data) s1.push(v);
      const s2 = new OnlineStats();
      s2.pushAll(data);
      expect(s1.mean).toBeCloseTo(s2.mean, 10);
      expect(s1.variance).toBeCloseTo(s2.variance, 10);
    });

    it("stdDev is sqrt of variance", () => {
      const stats = new OnlineStats();
      stats.pushAll([1, 2, 3, 4, 5]);
      expect(stats.stdDev).toBeCloseTo(Math.sqrt(stats.variance), 10);
    });
  });

  describe("OnlineCovariance edge cases", () => {
    it("throws on NaN push", () => {
      const cov = new OnlineCovariance();
      expect(() => cov.push(NaN, 1)).toThrow("NaN");
      expect(() => cov.push(1, NaN)).toThrow("NaN");
    });

    it("throws on meanX with no observations", () => {
      const cov = new OnlineCovariance();
      expect(() => cov.meanX).toThrow("expected at least 1 observation");
    });

    it("throws on meanY with no observations", () => {
      const cov = new OnlineCovariance();
      expect(() => cov.meanY).toThrow("expected at least 1 observation");
    });

    it("throws on correlation with fewer than 2 observations", () => {
      const cov = new OnlineCovariance();
      cov.push(1, 2);
      expect(() => cov.correlation).toThrow("at least 2");
    });

    it("correlation returns 0 for zero variance", () => {
      const cov = new OnlineCovariance();
      cov.pushAll([5, 5, 5], [5, 5, 5]);
      expect(cov.correlation).toBe(0);
    });

    it("pushAll throws on mismatched lengths", () => {
      const cov = new OnlineCovariance();
      expect(() => cov.pushAll([1, 2], [1])).toThrow("same length");
    });

    it("reset clears all state", () => {
      const cov = new OnlineCovariance();
      cov.pushAll([1, 2, 3], [4, 5, 6]);
      cov.reset();
      expect(cov.count).toBe(0);
      expect(() => cov.meanX).toThrow("expected at least 1 observation");
    });

    it("computes correct covariance and means", () => {
      const cov = new OnlineCovariance();
      cov.pushAll([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]);
      expect(cov.meanX).toBeCloseTo(3);
      expect(cov.meanY).toBeCloseTo(6);
      expect(cov.covariance).toBeCloseTo(5); // sample covariance
    });
  });
});
