import {
  entropy,
  entropyFromData,
  jointEntropy,
  conditionalEntropy,
  mutualInformation,
  normalizedMutualInformation,
  klDivergence,
  jsDivergence,
  crossEntropy,
} from "../src/information-theory";

describe("Information Theory", () => {
  describe("entropy", () => {
    it("returns 1 bit for fair coin", () => {
      expect(entropy([0.5, 0.5])).toBeCloseTo(1);
    });

    it("returns 0 for certain outcome", () => {
      expect(entropy([1, 0, 0])).toBeCloseTo(0);
    });

    it("returns log2(n) for uniform distribution", () => {
      expect(entropy([0.25, 0.25, 0.25, 0.25])).toBeCloseTo(2);
    });

    it("supports natural log base", () => {
      expect(entropy([0.5, 0.5], Math.E)).toBeCloseTo(Math.LN2);
    });

    it("throws on invalid distribution", () => {
      expect(() => entropy([0.5, 0.3])).toThrow("sum to 1");
      expect(() => entropy([-0.5, 1.5])).toThrow("non-negative");
    });
  });

  describe("entropyFromData", () => {
    it("computes entropy from raw data", () => {
      // Uniform over 4 values → 2 bits
      const data = [1, 2, 3, 4, 1, 2, 3, 4];
      expect(entropyFromData(data)).toBeCloseTo(2);
    });

    it("returns 0 for constant data", () => {
      expect(entropyFromData([5, 5, 5])).toBeCloseTo(0);
    });
  });

  describe("jointEntropy", () => {
    it("equals sum of individual entropies for independent variables", () => {
      const x = [1, 1, 2, 2, 1, 1, 2, 2];
      const y = [1, 2, 1, 2, 1, 2, 1, 2];
      const hxy = jointEntropy(x, y);
      const hx = entropyFromData(x);
      const hy = entropyFromData(y);
      expect(hxy).toBeCloseTo(hx + hy, 1);
    });

    it("equals individual entropy for perfectly dependent variables", () => {
      const x = [1, 2, 3, 1, 2, 3];
      expect(jointEntropy(x, x)).toBeCloseTo(entropyFromData(x));
    });
  });

  describe("conditionalEntropy", () => {
    it("equals 0 when Y is determined by X", () => {
      const x = [1, 2, 3, 1, 2, 3];
      expect(conditionalEntropy(x, x)).toBeCloseTo(0);
    });
  });

  describe("mutualInformation", () => {
    it("equals 0 for independent variables", () => {
      const x = [1, 1, 2, 2, 1, 1, 2, 2];
      const y = [1, 2, 1, 2, 1, 2, 1, 2];
      expect(mutualInformation(x, y)).toBeCloseTo(0, 1);
    });

    it("equals entropy for perfectly dependent variables", () => {
      const x = [1, 2, 3, 1, 2, 3];
      expect(mutualInformation(x, x)).toBeCloseTo(entropyFromData(x));
    });
  });

  describe("normalizedMutualInformation", () => {
    it("returns 1 for identical variables", () => {
      const x = [1, 2, 3, 1, 2, 3];
      expect(normalizedMutualInformation(x, x)).toBeCloseTo(1);
    });

    it("returns ~0 for independent variables", () => {
      const x = [1, 1, 2, 2, 1, 1, 2, 2];
      const y = [1, 2, 1, 2, 1, 2, 1, 2];
      expect(normalizedMutualInformation(x, y)).toBeCloseTo(0, 1);
    });
  });

  describe("klDivergence", () => {
    it("returns 0 for identical distributions", () => {
      expect(klDivergence([0.5, 0.5], [0.5, 0.5])).toBeCloseTo(0);
    });

    it("returns positive value for different distributions", () => {
      expect(klDivergence([0.9, 0.1], [0.5, 0.5])).toBeGreaterThan(0);
    });

    it("is not symmetric", () => {
      const p = [0.9, 0.1];
      const q = [0.5, 0.5];
      expect(klDivergence(p, q)).not.toBeCloseTo(klDivergence(q, p));
    });

    it("throws when q=0 where p>0", () => {
      expect(() => klDivergence([0.5, 0.5], [1, 0])).toThrow("undefined");
    });
  });

  describe("jsDivergence", () => {
    it("returns 0 for identical distributions", () => {
      expect(jsDivergence([0.5, 0.5], [0.5, 0.5])).toBeCloseTo(0);
    });

    it("is symmetric", () => {
      const p = [0.9, 0.1];
      const q = [0.5, 0.5];
      expect(jsDivergence(p, q)).toBeCloseTo(jsDivergence(q, p));
    });

    it("is bounded between 0 and 1 (base 2)", () => {
      const result = jsDivergence([1, 0], [0, 1]);
      expect(result).toBeGreaterThanOrEqual(0);
      expect(result).toBeLessThanOrEqual(1);
    });
  });

  describe("crossEntropy", () => {
    it("equals entropy when p=q", () => {
      const p = [0.5, 0.5];
      expect(crossEntropy(p, p)).toBeCloseTo(entropy(p));
    });

    it("is greater than entropy when p≠q", () => {
      const p = [0.7, 0.3];
      const q = [0.5, 0.5];
      expect(crossEntropy(p, q)).toBeGreaterThan(entropy(p));
    });
  });
});
