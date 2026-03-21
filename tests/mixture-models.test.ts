import { gaussianMixture, selectComponents } from "../src/mixture-models";

describe("Mixture Models", () => {
  // Generate data from two well-separated Gaussians
  const data: number[] = [];
  // Cluster 1: centered at 0
  for (let i = 0; i < 50; i++) data.push(i * 0.1 - 2.5);
  // Cluster 2: centered at 10
  for (let i = 0; i < 50; i++) data.push(i * 0.1 + 7.5);

  describe("gaussianMixture", () => {
    it("fits a 2-component model", () => {
      const result = gaussianMixture(data, 2, { seed: 42 });
      expect(result.k).toBe(2);
      expect(result.weights).toHaveLength(2);
      expect(result.means).toHaveLength(2);
      expect(result.variances).toHaveLength(2);

      // Weights should sum to 1
      const weightSum = result.weights.reduce((a, b) => a + b, 0);
      expect(weightSum).toBeCloseTo(1);

      // Means should be near 0 and 10 (or their estimates)
      const sortedMeans = [...result.means].sort((a, b) => a - b);
      expect(sortedMeans[0]).toBeLessThan(5);
      expect(sortedMeans[1]).toBeGreaterThan(5);
    });

    it("produces valid responsibilities", () => {
      const result = gaussianMixture(data, 2, { seed: 42 });
      for (const row of result.responsibilities) {
        const sum = row.reduce((a, b) => a + b, 0);
        expect(sum).toBeCloseTo(1);
        for (const v of row) {
          expect(v).toBeGreaterThanOrEqual(0);
          expect(v).toBeLessThanOrEqual(1);
        }
      }
    });

    it("assigns hard labels", () => {
      const result = gaussianMixture(data, 2, { seed: 42 });
      expect(result.labels).toHaveLength(data.length);
      const uniqueLabels = new Set(result.labels);
      expect(uniqueLabels.size).toBe(2);
    });

    it("computes BIC and AIC", () => {
      const result = gaussianMixture(data, 2, { seed: 42 });
      expect(typeof result.bic).toBe("number");
      expect(typeof result.aic).toBe("number");
      expect(result.bic).toBeGreaterThan(result.aic); // BIC penalizes more
    });

    it("fits a 1-component model", () => {
      const result = gaussianMixture(data, 1);
      expect(result.k).toBe(1);
      expect(result.weights).toEqual([1]);
      expect(result.labels.every((l) => l === 0)).toBe(true);
    });

    it("throws on invalid input", () => {
      expect(() => gaussianMixture([], 1)).toThrow("at least k");
      expect(() => gaussianMixture([1], 0)).toThrow("at least 1");
    });
  });

  describe("selectComponents", () => {
    it("selects k=2 for bimodal data", () => {
      const result = selectComponents(data, 4, { seed: 42 });
      expect(result.bestK).toBe(2);
      expect(result.results).toHaveLength(4);
      expect(result.bicValues).toHaveLength(4);
    });

    it("selects k=1 for unimodal data", () => {
      const unimodal = Array.from({ length: 50 }, (_, i) => i * 0.1);
      const result = selectComponents(unimodal, 3, { seed: 42 });
      expect(result.bestK).toBe(1);
    });
  });
});
