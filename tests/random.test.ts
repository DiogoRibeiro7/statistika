import {
  SeededRng,
  haltonSequence,
  haltonSequenceND,
  latinHypercube,
} from "../src/random";

describe("Random Number Generation", () => {
  describe("SeededRng", () => {
    it("produces reproducible sequences", () => {
      const rng1 = new SeededRng(42);
      const rng2 = new SeededRng(42);
      for (let i = 0; i < 100; i++) {
        expect(rng1.next()).toBe(rng2.next());
      }
    });

    it("generates values in [0, 1)", () => {
      const rng = new SeededRng(123);
      for (let i = 0; i < 1000; i++) {
        const v = rng.next();
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThan(1);
      }
    });

    it("generates integers in range", () => {
      const rng = new SeededRng(42);
      for (let i = 0; i < 100; i++) {
        const v = rng.nextInt(1, 6);
        expect(v).toBeGreaterThanOrEqual(1);
        expect(v).toBeLessThanOrEqual(6);
      }
    });

    it("generates normal variates", () => {
      const rng = new SeededRng(42);
      const samples = Array.from({ length: 1000 }, () => rng.nextNormal());
      const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
      expect(mean).toBeCloseTo(0, 0);
    });

    it("shuffles array", () => {
      const rng = new SeededRng(42);
      const arr = [1, 2, 3, 4, 5];
      rng.shuffle(arr);
      // Should contain same elements
      expect(arr.sort((a, b) => a - b)).toEqual([1, 2, 3, 4, 5]);
    });

    it("chooses k items without replacement", () => {
      const rng = new SeededRng(42);
      const result = rng.choose([1, 2, 3, 4, 5], 3);
      expect(result).toHaveLength(3);
      expect(new Set(result).size).toBe(3); // all unique
    });

    it("throws on invalid choose", () => {
      const rng = new SeededRng(42);
      expect(() => rng.choose([1, 2], 3)).toThrow("cannot exceed");
    });
  });

  describe("haltonSequence", () => {
    it("generates values in (0, 1)", () => {
      const seq = haltonSequence(2, 100);
      expect(seq).toHaveLength(100);
      for (const v of seq) {
        expect(v).toBeGreaterThan(0);
        expect(v).toBeLessThan(1);
      }
    });

    it("produces known values for base 2", () => {
      const seq = haltonSequence(2, 4);
      expect(seq[0]).toBeCloseTo(0.5);
      expect(seq[1]).toBeCloseTo(0.25);
      expect(seq[2]).toBeCloseTo(0.75);
      expect(seq[3]).toBeCloseTo(0.125);
    });

    it("has better coverage than random", () => {
      const seq = haltonSequence(2, 100);
      // Divide [0,1] into 10 bins; Halton should cover most
      const bins = new Array(10).fill(0);
      for (const v of seq) bins[Math.min(9, Math.floor(v * 10))]++;
      const covered = bins.filter((b) => b > 0).length;
      expect(covered).toBe(10);
    });

    it("throws on invalid base", () => {
      expect(() => haltonSequence(1, 10)).toThrow("integer >= 2");
    });
  });

  describe("haltonSequenceND", () => {
    it("generates multi-dimensional points", () => {
      const points = haltonSequenceND(3, 50);
      expect(points).toHaveLength(50);
      expect(points[0]).toHaveLength(3);
      for (const p of points) {
        for (const v of p) {
          expect(v).toBeGreaterThan(0);
          expect(v).toBeLessThan(1);
        }
      }
    });
  });

  describe("latinHypercube", () => {
    it("generates stratified samples", () => {
      const samples = latinHypercube(2, 100, 42);
      expect(samples).toHaveLength(100);
      expect(samples[0]).toHaveLength(2);

      // All values should be in [0, 1)
      for (const s of samples) {
        for (const v of s) {
          expect(v).toBeGreaterThanOrEqual(0);
          expect(v).toBeLessThan(1);
        }
      }
    });

    it("covers strata uniformly", () => {
      const samples = latinHypercube(1, 10, 42);
      // Each stratum [0,0.1), [0.1,0.2), ... should have exactly one sample
      const strata = samples.map((s) => Math.floor(s[0] * 10));
      const unique = new Set(strata);
      expect(unique.size).toBe(10);
    });

    it("throws on dimensions < 1", () => {
      expect(() => latinHypercube(0, 10)).toThrow("at least 1");
    });

    it("throws on n < 1", () => {
      expect(() => latinHypercube(2, 0)).toThrow("at least 1");
    });
  });

  describe("SeededRng edge cases", () => {
    it("throws on NaN seed", () => {
      expect(() => new SeededRng(NaN)).toThrow("finite");
    });

    it("throws on Infinity seed", () => {
      expect(() => new SeededRng(Infinity)).toThrow("finite");
    });

    it("nextInt throws on non-finite min", () => {
      const rng = new SeededRng(42);
      expect(() => rng.nextInt(NaN, 5)).toThrow("finite");
    });

    it("nextInt throws on non-finite max", () => {
      const rng = new SeededRng(42);
      expect(() => rng.nextInt(1, Infinity)).toThrow("finite");
    });

    it("nextNormal throws on non-finite mu", () => {
      const rng = new SeededRng(42);
      expect(() => rng.nextNormal(NaN, 1)).toThrow("finite");
    });

    it("nextNormal throws on non-finite sigma", () => {
      const rng = new SeededRng(42);
      expect(() => rng.nextNormal(0, Infinity)).toThrow("finite");
    });

    it("nextNormal with custom mu and sigma", () => {
      const rng = new SeededRng(42);
      const samples = Array.from({ length: 5000 }, () => rng.nextNormal(10, 2));
      const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
      expect(Math.abs(mean - 10)).toBeLessThan(1);
    });

    it("sample generates n values", () => {
      const rng = new SeededRng(42);
      const result = rng.sample(5);
      expect(result).toHaveLength(5);
      for (const v of result) {
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThan(1);
      }
    });

    it("sample throws on non-positive n", () => {
      const rng = new SeededRng(42);
      expect(() => rng.sample(0)).toThrow("positive integer");
      expect(() => rng.sample(-1)).toThrow("positive integer");
    });

    it("sample throws on non-integer n", () => {
      const rng = new SeededRng(42);
      expect(() => rng.sample(1.5)).toThrow("positive integer");
    });

    it("choose with k=0 returns empty array", () => {
      const rng = new SeededRng(42);
      const result = rng.choose([1, 2, 3], 0);
      expect(result).toEqual([]);
    });

    it("choose with k < 0 throws", () => {
      const rng = new SeededRng(42);
      expect(() => rng.choose([1, 2, 3], -1)).toThrow("non-negative");
    });

    it("choose with k = array.length returns all elements", () => {
      const rng = new SeededRng(42);
      const result = rng.choose([1, 2, 3], 3);
      expect(result).toHaveLength(3);
      expect(result.sort((a, b) => a - b)).toEqual([1, 2, 3]);
    });

    it("shuffle returns the same array reference", () => {
      const rng = new SeededRng(42);
      const arr = [1, 2, 3];
      const result = rng.shuffle(arr);
      expect(result).toBe(arr);
    });

    it("shuffle of empty array works", () => {
      const rng = new SeededRng(42);
      const arr: number[] = [];
      rng.shuffle(arr);
      expect(arr).toEqual([]);
    });

    it("shuffle of single element works", () => {
      const rng = new SeededRng(42);
      const arr = [42];
      rng.shuffle(arr);
      expect(arr).toEqual([42]);
    });

    it("different seeds produce different sequences", () => {
      const rng1 = new SeededRng(1);
      const rng2 = new SeededRng(2);
      const vals1 = Array.from({ length: 10 }, () => rng1.next());
      const vals2 = Array.from({ length: 10 }, () => rng2.next());
      expect(vals1).not.toEqual(vals2);
    });
  });

  describe("haltonSequence edge cases", () => {
    it("throws on non-prime base", () => {
      expect(() => haltonSequence(4, 10)).toThrow("prime");
    });

    it("throws on non-integer base", () => {
      expect(() => haltonSequence(2.5, 10)).toThrow("integer >= 2");
    });

    it("throws on n < 1", () => {
      expect(() => haltonSequence(2, 0)).toThrow("at least 1");
    });

    it("supports skip parameter", () => {
      const seq1 = haltonSequence(2, 5);
      const seq2 = haltonSequence(2, 5, 3);
      // seq2 should start where seq1[3..] would be
      expect(seq2[0]).toBeCloseTo(seq1[3]);
    });

    it("produces known values for base 3", () => {
      const seq = haltonSequence(3, 3);
      expect(seq[0]).toBeCloseTo(1 / 3);
      expect(seq[1]).toBeCloseTo(2 / 3);
      expect(seq[2]).toBeCloseTo(1 / 9);
    });
  });

  describe("haltonSequenceND edge cases", () => {
    it("throws on dimensions < 1", () => {
      expect(() => haltonSequenceND(0, 10)).toThrow("at least 1");
    });

    it("throws on n < 1", () => {
      expect(() => haltonSequenceND(2, 0)).toThrow("at least 1");
    });

    it("1D matches haltonSequence with base 2", () => {
      const nd = haltonSequenceND(1, 5);
      const seq = haltonSequence(2, 5);
      for (let i = 0; i < 5; i++) {
        expect(nd[i][0]).toBeCloseTo(seq[i]);
      }
    });

    it("supports skip parameter", () => {
      const nd = haltonSequenceND(2, 5, 10);
      expect(nd).toHaveLength(5);
      expect(nd[0]).toHaveLength(2);
    });
  });
});
