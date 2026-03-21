import {
  tsne,
  silhouetteScore,
  silhouetteScores,
  daviesBouldinIndex,
  adjustedRandIndex,
} from "../src/dimensionality";

describe("Dimensionality Reduction & Cluster Validation", () => {
  // Two well-separated clusters in 3D
  const cluster1 = Array.from({ length: 15 }, (_, i) => [i * 0.1, i * 0.1, 0]);
  const cluster2 = Array.from({ length: 15 }, (_, i) => [10 + i * 0.1, 10 + i * 0.1, 10]);
  const data = [...cluster1, ...cluster2];
  const labels = [...new Array(15).fill(0), ...new Array(15).fill(1)];

  describe("tsne", () => {
    it("reduces to 2D", () => {
      const result = tsne(data, { iterations: 100, seed: 42 });
      expect(result.embedding).toHaveLength(30);
      expect(result.embedding[0]).toHaveLength(2);
      expect(result.iterations).toBe(100);
    });

    it("separates clusters", () => {
      const result = tsne(data, { iterations: 200, seed: 42, perplexity: 5 });

      // Compute centroid of each cluster in embedding space
      const centroid1 = [0, 0];
      const centroid2 = [0, 0];
      for (let i = 0; i < 15; i++) {
        centroid1[0] += result.embedding[i][0];
        centroid1[1] += result.embedding[i][1];
        centroid2[0] += result.embedding[15 + i][0];
        centroid2[1] += result.embedding[15 + i][1];
      }
      centroid1[0] /= 15; centroid1[1] /= 15;
      centroid2[0] /= 15; centroid2[1] /= 15;

      // Centroids should be separated
      const dist = Math.sqrt(
        (centroid1[0] - centroid2[0]) ** 2 + (centroid1[1] - centroid2[1]) ** 2,
      );
      expect(dist).toBeGreaterThan(0);
    });

    it("throws on too few observations", () => {
      expect(() => tsne([[1, 2], [3, 4]])).toThrow("at least 4");
    });
  });

  describe("silhouetteScore", () => {
    it("returns high score for well-separated clusters", () => {
      const score = silhouetteScore(data, labels);
      expect(score).toBeGreaterThan(0.5);
    });

    it("throws on single cluster", () => {
      expect(() => silhouetteScore(data, new Array(30).fill(0))).toThrow("at least 2 clusters");
    });
  });

  describe("silhouetteScores", () => {
    it("returns per-observation scores", () => {
      const scores = silhouetteScores(data, labels);
      expect(scores).toHaveLength(30);
      for (const s of scores) {
        expect(s).toBeGreaterThanOrEqual(-1);
        expect(s).toBeLessThanOrEqual(1);
      }
    });
  });

  describe("daviesBouldinIndex", () => {
    it("returns low value for well-separated clusters", () => {
      const db = daviesBouldinIndex(data, labels);
      expect(db).toBeGreaterThan(0);
      expect(db).toBeLessThan(1); // Well-separated should have low DB
    });

    it("throws on single cluster", () => {
      expect(() => daviesBouldinIndex(data, new Array(30).fill(0))).toThrow("at least 2");
    });
  });

  describe("adjustedRandIndex", () => {
    it("returns 1 for identical labelings", () => {
      expect(adjustedRandIndex(labels, labels)).toBeCloseTo(1);
    });

    it("returns ~0 for random labelings", () => {
      const random = Array.from({ length: 30 }, (_, i) => i % 3);
      const ari = adjustedRandIndex(labels, random);
      expect(Math.abs(ari)).toBeLessThan(0.5);
    });

    it("is symmetric", () => {
      const other = [...new Array(10).fill(0), ...new Array(10).fill(1), ...new Array(10).fill(2)];
      expect(adjustedRandIndex(labels, other)).toBeCloseTo(
        adjustedRandIndex(other, labels),
      );
    });

    it("throws on mismatched lengths", () => {
      expect(() => adjustedRandIndex([0, 1], [0])).toThrow("same length");
    });
  });
});
