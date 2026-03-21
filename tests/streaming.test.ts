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
      expect(() => stats.mean).toThrow("No observations");
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
      expect(() => new OnlineQuantile(0)).toThrow("between 0 and 1");
      expect(() => new OnlineQuantile(1)).toThrow("between 0 and 1");
    });
  });
});
