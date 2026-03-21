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

    it("throws on fewer than 2 observations", () => {
      expect(() => adjustedRandIndex([0], [0])).toThrow("at least 2");
    });
  });

  describe("tsne edge cases", () => {
    it("throws on NaN/Infinity in data", () => {
      expect(() => tsne([[1, 2], [3, NaN], [5, 6], [7, 8]])).toThrow("NaN or Infinity");
      expect(() => tsne([[1, Infinity], [3, 4], [5, 6], [7, 8]])).toThrow("NaN or Infinity");
    });

    it("returns klDivergence as a number", () => {
      const result = tsne(data, { iterations: 50, seed: 42 });
      expect(typeof result.klDivergence).toBe("number");
      expect(result.klDivergence).toBeGreaterThanOrEqual(0);
    });

    it("accepts custom perplexity and learningRate", () => {
      const result = tsne(data, { iterations: 50, seed: 42, perplexity: 3, learningRate: 100 });
      expect(result.embedding).toHaveLength(30);
    });
  });

  describe("silhouetteScore edge cases", () => {
    it("throws on mismatched data/labels lengths", () => {
      expect(() => silhouetteScore([[0, 0], [1, 1]], [0])).toThrow("same length");
    });

    it("throws on fewer than 2 observations", () => {
      expect(() => silhouetteScore([[0, 0]], [0])).toThrow("at least 2");
    });

    it("throws on NaN in data", () => {
      expect(() => silhouetteScore([[NaN, 0], [1, 1], [2, 2]], [0, 0, 1])).toThrow("NaN or Infinity");
    });
  });

  describe("silhouetteScores edge cases", () => {
    it("throws on mismatched data/labels lengths", () => {
      expect(() => silhouetteScores([[0, 0]], [0, 1])).toThrow("same length");
    });

    it("throws on NaN in data", () => {
      expect(() => silhouetteScores([[1, NaN], [2, 3]], [0, 1])).toThrow("NaN or Infinity");
    });
  });

  describe("daviesBouldinIndex edge cases", () => {
    it("throws on NaN in data", () => {
      expect(() => daviesBouldinIndex([[NaN, 0], [1, 1], [2, 2]], [0, 0, 1])).toThrow("NaN or Infinity");
    });

    it("produces higher value for overlapping clusters", () => {
      const overlapping = Array.from({ length: 20 }, (_, i) => [i * 0.1, i * 0.1]);
      const labelsOverlap = overlapping.map((_, i) => (i % 2 === 0 ? 0 : 1));
      const dbOverlap = daviesBouldinIndex(overlapping, labelsOverlap);
      const dbSeparated = daviesBouldinIndex(data, labels);
      expect(dbOverlap).toBeGreaterThan(dbSeparated);
    });
  });

  describe("tsne additional", () => {
    it("embedding is centered near zero", () => {
      const result = tsne(data, { iterations: 50, seed: 42 });
      const meanX = result.embedding.reduce((s, p) => s + p[0], 0) / result.embedding.length;
      const meanY = result.embedding.reduce((s, p) => s + p[1], 0) / result.embedding.length;
      expect(Math.abs(meanX)).toBeLessThan(1);
      expect(Math.abs(meanY)).toBeLessThan(1);
    });

    it("throws on exactly 3 observations", () => {
      expect(() => tsne([[1, 2], [3, 4], [5, 6]])).toThrow("at least 4");
    });

    it("handles minimal 4-point dataset", () => {
      const smallData = [[0, 0], [1, 0], [0, 1], [1, 1]];
      const result = tsne(smallData, { iterations: 50, seed: 42 });
      expect(result.embedding).toHaveLength(4);
      expect(result.embedding[0]).toHaveLength(2);
    });
  });

  describe("silhouetteScore and silhouetteScores consistency", () => {
    it("mean of per-observation scores equals overall score", () => {
      const scores = silhouetteScores(data, labels);
      const meanScore = scores.reduce((a, b) => a + b, 0) / scores.length;
      const overall = silhouetteScore(data, labels);
      expect(meanScore).toBeCloseTo(overall, 5);
    });
  });

  describe("adjustedRandIndex additional", () => {
    it("returns value in [-1, 1] range", () => {
      const l1 = [0, 0, 1, 1, 2, 2];
      const l2 = [1, 1, 0, 0, 2, 2];
      const ari = adjustedRandIndex(l1, l2);
      expect(ari).toBeGreaterThanOrEqual(-1);
      expect(ari).toBeLessThanOrEqual(1);
    });

    it("handles two observations", () => {
      const ari = adjustedRandIndex([0, 1], [0, 1]);
      expect(ari).toBeCloseTo(1);
    });

    it("returns 1 for permuted identical labels", () => {
      // Same partition but different label values
      const l1 = [0, 0, 0, 1, 1, 1];
      const l2 = [5, 5, 5, 9, 9, 9];
      expect(adjustedRandIndex(l1, l2)).toBeCloseTo(1);
    });
  });

  describe("daviesBouldinIndex additional", () => {
    it("throws on Infinity in data", () => {
      expect(() => daviesBouldinIndex([[Infinity, 0], [1, 1], [2, 2]], [0, 0, 1])).toThrow("NaN or Infinity");
    });

    it("returns non-negative value", () => {
      const db = daviesBouldinIndex(data, labels);
      expect(db).toBeGreaterThanOrEqual(0);
    });
  });
});
