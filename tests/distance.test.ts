import {
  euclidean,
  manhattan,
  chebyshev,
  minkowski,
  cosineSimilarity,
  cosineDistance,
  jaccardIndex,
  jaccardDistance,
  mahalanobis,
  distanceMatrix,
} from "../src/distance";

describe("Distance & Similarity Metrics", () => {
  describe("euclidean", () => {
    it("computes distance correctly", () => {
      expect(euclidean([0, 0], [3, 4])).toBeCloseTo(5);
      expect(euclidean([1, 2, 3], [1, 2, 3])).toBeCloseTo(0);
    });

    it("throws on different lengths", () => {
      expect(() => euclidean([1, 2], [1])).toThrow("same length");
    });
  });

  describe("manhattan", () => {
    it("computes L1 distance", () => {
      expect(manhattan([0, 0], [3, 4])).toBeCloseTo(7);
    });
  });

  describe("chebyshev", () => {
    it("computes L∞ distance", () => {
      expect(chebyshev([0, 0], [3, 4])).toBeCloseTo(4);
    });
  });

  describe("minkowski", () => {
    it("reduces to euclidean for p=2", () => {
      expect(minkowski([0, 0], [3, 4], 2)).toBeCloseTo(5);
    });

    it("reduces to manhattan for p=1", () => {
      expect(minkowski([0, 0], [3, 4], 1)).toBeCloseTo(7);
    });

    it("reduces to chebyshev for p=Infinity", () => {
      expect(minkowski([0, 0], [3, 4], Infinity)).toBeCloseTo(4);
    });

    it("throws on p < 1", () => {
      expect(() => minkowski([0], [1], 0.5)).toThrow("p must be >= 1");
    });
  });

  describe("cosineSimilarity", () => {
    it("returns 1 for identical directions", () => {
      expect(cosineSimilarity([1, 2, 3], [2, 4, 6])).toBeCloseTo(1);
    });

    it("returns 0 for orthogonal vectors", () => {
      expect(cosineSimilarity([1, 0], [0, 1])).toBeCloseTo(0);
    });

    it("returns -1 for opposite directions", () => {
      expect(cosineSimilarity([1, 0], [-1, 0])).toBeCloseTo(-1);
    });
  });

  describe("cosineDistance", () => {
    it("equals 1 - similarity", () => {
      expect(cosineDistance([1, 0], [0, 1])).toBeCloseTo(1);
      expect(cosineDistance([1, 2], [2, 4])).toBeCloseTo(0);
    });
  });

  describe("jaccardIndex", () => {
    it("computes set overlap", () => {
      expect(jaccardIndex([1, 2, 3], [2, 3, 4])).toBeCloseTo(0.5); // 2/4
    });

    it("returns 1 for identical sets", () => {
      expect(jaccardIndex([1, 2, 3], [1, 2, 3])).toBeCloseTo(1);
    });

    it("returns 0 for disjoint sets", () => {
      expect(jaccardIndex([1, 2], [3, 4])).toBeCloseTo(0);
    });

    it("returns 1 for two empty sets", () => {
      expect(jaccardIndex([], [])).toBeCloseTo(1);
    });
  });

  describe("jaccardDistance", () => {
    it("equals 1 - index", () => {
      expect(jaccardDistance([1, 2, 3], [2, 3, 4])).toBeCloseTo(0.5);
    });
  });

  describe("mahalanobis", () => {
    it("computes distance for simple case", () => {
      const data = [
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1],
      ];
      const dist = mahalanobis([2, 0], data);
      expect(dist).toBeGreaterThan(0);
    });

    it("throws on too few observations", () => {
      expect(() => mahalanobis([1], [[1]])).toThrow("at least 2");
    });
  });

  describe("distanceMatrix", () => {
    it("produces symmetric matrix", () => {
      const vectors = [[0, 0], [1, 0], [0, 1]];
      const matrix = distanceMatrix(vectors);
      expect(matrix.length).toBe(3);
      for (let i = 0; i < 3; i++) {
        expect(matrix[i][i]).toBeCloseTo(0);
        for (let j = 0; j < 3; j++) {
          expect(matrix[i][j]).toBeCloseTo(matrix[j][i]);
        }
      }
    });

    it("uses custom metric", () => {
      const vectors = [[0, 0], [3, 4]];
      const matrix = distanceMatrix(vectors, manhattan);
      expect(matrix[0][1]).toBeCloseTo(7);
    });
  });
});
