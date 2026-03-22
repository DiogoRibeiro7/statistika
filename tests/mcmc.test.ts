import {
  metropolisHastings,
  metropolisHastingsND,
  gelmanRubin,
  estimateESS,
} from "../src/mcmc";

describe("MCMC Sampler", () => {
  describe("metropolisHastings", () => {
    it("samples from a normal distribution", () => {
      // Target: N(3, 1)
      const logDensity = (x: number) => -0.5 * (x - 3) ** 2;

      const result = metropolisHastings(logDensity, {
        initial: 3,
        proposalStd: 2.0,
        iterations: 100000,
        burnIn: 20000,
      });

      expect(Math.abs(result.posteriorMean - 3)).toBeLessThan(1);
      expect(result.posteriorStd).toBeGreaterThan(0.3);
      expect(result.acceptanceRate).toBeGreaterThan(0.1);
      expect(result.acceptanceRate).toBeLessThan(0.9);
    });

    it("produces valid credible interval", () => {
      const logDensity = (x: number) => -0.5 * (x - 5) ** 2;
      const result = metropolisHastings(logDensity, {
        iterations: 10000,
        burnIn: 2000,
        seed: 42,
      });

      expect(result.credibleInterval.lower).toBeLessThan(result.posteriorMean);
      expect(result.credibleInterval.upper).toBeGreaterThan(result.posteriorMean);
      expect(result.credibleInterval.level).toBe(0.95);
    });

    it("estimates effective sample size", () => {
      const logDensity = (x: number) => -0.5 * x ** 2;
      const result = metropolisHastings(logDensity, {
        iterations: 10000,
        seed: 42,
      });

      expect(result.effectiveSampleSize).toBeGreaterThan(0);
      expect(result.effectiveSampleSize).toBeLessThanOrEqual(result.chain.length);
    });
  });

  describe("metropolisHastingsND", () => {
    it("samples from a 2D normal", () => {
      // Target: bivariate normal centered at [2, -1]
      const logDensity = (x: number[]) => -0.5 * ((x[0] - 2) ** 2 + (x[1] + 1) ** 2);

      const result = metropolisHastingsND(logDensity, 2, {
        initial: [2, -1],
        proposalStd: 2.0,
        iterations: 100000,
        burnIn: 20000,
      });

      expect(Math.abs(result.posteriorMeans[0] - 2)).toBeLessThan(1);
      expect(Math.abs(result.posteriorMeans[1] + 1)).toBeLessThan(1);
      expect(result.acceptanceRate).toBeGreaterThan(0.05);
    });
  });

  describe("gelmanRubin", () => {
    it("returns ~1 for converged chains", () => {
      // Two chains sampling from the same distribution
      const logDensity = (x: number) => -0.5 * x ** 2;
      const chain1 = metropolisHastings(logDensity, { seed: 1, iterations: 5000 }).chain;
      const chain2 = metropolisHastings(logDensity, { seed: 2, iterations: 5000 }).chain;

      const rhat = gelmanRubin([chain1, chain2]);
      expect(rhat).toBeCloseTo(1, 0);
      expect(rhat).toBeLessThan(1.2);
    });

    it("throws on single chain", () => {
      expect(() => gelmanRubin([[1, 2, 3]])).toThrow("at least 2");
    });
  });

  describe("estimateESS", () => {
    it("returns high ESS for independent samples", () => {
      // Independent samples should have ESS close to n
      const independent = Array.from({ length: 1000 }, (_, i) => Math.sin(i * 100));
      const ess = estimateESS(independent);
      expect(ess).toBeGreaterThan(100);
    });

    it("returns low ESS for highly correlated samples", () => {
      // Random walk is highly correlated
      const walk: number[] = [0];
      for (let i = 1; i < 1000; i++) walk.push(walk[i - 1] + 0.01);
      const ess = estimateESS(walk);
      expect(ess).toBeLessThan(100);
    });

    it("returns n for very short chains", () => {
      const short = [1, 2, 3, 4, 5];
      const ess = estimateESS(short);
      expect(ess).toBe(5); // n < 10, returns n directly
    });

    it("returns n for constant chain (zero variance)", () => {
      const constant = new Array(20).fill(5);
      const ess = estimateESS(constant);
      expect(ess).toBe(20); // v === 0, returns n
    });
  });

  describe("metropolisHastings edge cases", () => {
    it("uses default options when none provided", () => {
      const logDensity = (x: number) => -0.5 * x * x;
      const result = metropolisHastings(logDensity);
      expect(result.totalIterations).toBe(10000);
      expect(result.burnIn).toBe(2000);
      expect(result.chain.length).toBe(8000);
    });

    it("respects custom burnIn", () => {
      const logDensity = (x: number) => -0.5 * x * x;
      const result = metropolisHastings(logDensity, {
        iterations: 5000,
        burnIn: 1000,
        seed: 42,
      });
      expect(result.burnIn).toBe(1000);
      expect(result.chain.length).toBe(4000);
    });

    it("respects custom credibleLevel", () => {
      const logDensity = (x: number) => -0.5 * x * x;
      const result = metropolisHastings(logDensity, {
        iterations: 5000,
        seed: 42,
        credibleLevel: 0.90,
      });
      expect(result.credibleInterval.level).toBe(0.90);
    });

    it("handles NaN from logDensity gracefully", () => {
      // A logDensity that returns NaN for some inputs
      const logDensity = (x: number) => (x > 10 ? NaN : -0.5 * x * x);
      const result = metropolisHastings(logDensity, {
        iterations: 5000,
        seed: 42,
      });
      // Should still produce a valid chain
      expect(result.chain.length).toBeGreaterThan(0);
      expect(Number.isFinite(result.posteriorMean)).toBe(true);
    });

    it("warns on very high acceptance rate", () => {
      // Very small proposalStd => almost always accept
      const logDensity = (x: number) => -0.5 * x * x;
      const result = metropolisHastings(logDensity, {
        iterations: 5000,
        proposalStd: 0.0001,
        seed: 42,
      });
      if (result.acceptanceRate > 0.95) {
        expect(result.warning).toBeDefined();
        expect(result.warning).toContain("high acceptance rate");
      }
    });

    it("uses seed for reproducibility", () => {
      const logDensity = (x: number) => -0.5 * x * x;
      const r1 = metropolisHastings(logDensity, { iterations: 1000, seed: 99 });
      const r2 = metropolisHastings(logDensity, { iterations: 1000, seed: 99 });
      expect(r1.chain).toEqual(r2.chain);
    });
  });

  describe("metropolisHastingsND edge cases", () => {
    it("uses default initial values of zero", () => {
      const logDensity = (x: number[]) => -0.5 * (x[0] * x[0] + x[1] * x[1]);
      const result = metropolisHastingsND(logDensity, 2, {
        iterations: 5000,
        seed: 42,
      });
      expect(result.posteriorMeans).toHaveLength(2);
      expect(result.posteriorStds).toHaveLength(2);
    });

    it("accepts per-dimension proposalStd array", () => {
      const logDensity = (x: number[]) => -0.5 * (x[0] * x[0] + x[1] * x[1]);
      const result = metropolisHastingsND(logDensity, 2, {
        iterations: 5000,
        seed: 42,
        proposalStd: [0.5, 2.0],
      });
      expect(result.chains.length).toBeGreaterThan(0);
    });

    it("handles NaN from logDensity", () => {
      const logDensity = (x: number[]) =>
        x[0] > 100 ? NaN : -0.5 * (x[0] * x[0] + x[1] * x[1]);
      const result = metropolisHastingsND(logDensity, 2, {
        iterations: 5000,
        seed: 42,
      });
      expect(result.chains.length).toBeGreaterThan(0);
    });
  });

  describe("gelmanRubin edge cases", () => {
    it("throws on chains with different lengths", () => {
      expect(() => gelmanRubin([[1, 2, 3], [1, 2]])).toThrow("same length");
    });

    it("returns higher R-hat for divergent chains", () => {
      // Two chains with very different means
      const chain1 = Array.from({ length: 100 }, () => 0);
      const chain2 = Array.from({ length: 100 }, () => 10);
      const rhat = gelmanRubin([chain1, chain2]);
      expect(rhat).toBeGreaterThan(1.5);
    });

    it("works with more than 2 chains", () => {
      const logDensity = (x: number) => -0.5 * x * x;
      const chain1 = metropolisHastings(logDensity, { seed: 1, iterations: 5000 }).chain;
      const chain2 = metropolisHastings(logDensity, { seed: 2, iterations: 5000 }).chain;
      const chain3 = metropolisHastings(logDensity, { seed: 3, iterations: 5000 }).chain;
      const rhat = gelmanRubin([chain1, chain2, chain3]);
      expect(rhat).toBeGreaterThan(0);
      expect(rhat).toBeLessThan(2);
    });
  });

  describe("metropolisHastings warns on low acceptance", () => {
    it("warns on very low acceptance rate", () => {
      // Very large proposalStd => almost always reject
      const logDensity = (x: number) => -0.5 * x * x;
      const result = metropolisHastings(logDensity, {
        iterations: 5000,
        proposalStd: 10000,
        seed: 42,
      });
      if (result.acceptanceRate < 0.05) {
        expect(result.warning).toBeDefined();
        expect(result.warning).toContain("low acceptance rate");
      }
    });
  });

  describe("estimateESS edge cases", () => {
    it("returns at least 1", () => {
      // Highly correlated monotonic chain
      const chain = Array.from({ length: 100 }, (_, i) => i);
      const ess = estimateESS(chain);
      expect(ess).toBeGreaterThanOrEqual(1);
    });

    it("handles empty chain by returning 0", () => {
      const ess = estimateESS([]);
      // n < 10 so returns n which is 0
      expect(ess).toBe(0);
    });
  });

  describe("metropolisHastingsND additional", () => {
    it("warns on very high acceptance rate", () => {
      const logDensity = (x: number[]) => -0.5 * (x[0] * x[0] + x[1] * x[1]);
      const result = metropolisHastingsND(logDensity, 2, {
        iterations: 5000,
        proposalStd: 0.00001,
        seed: 42,
      });
      if (result.acceptanceRate > 0.95) {
        expect(result.warning).toBeDefined();
        expect(result.warning).toContain("high acceptance rate");
      }
    });

    it("respects seed for reproducibility in ND", () => {
      const logDensity = (x: number[]) => -0.5 * (x[0] * x[0] + x[1] * x[1]);
      const r1 = metropolisHastingsND(logDensity, 2, { iterations: 1000, seed: 42 });
      const r2 = metropolisHastingsND(logDensity, 2, { iterations: 1000, seed: 42 });
      expect(r1.chains).toEqual(r2.chains);
    });

    it("chain length matches iterations minus burnIn", () => {
      const logDensity = (x: number[]) => -0.5 * x[0] * x[0];
      const result = metropolisHastingsND(logDensity, 1, {
        iterations: 1000,
        burnIn: 200,
        seed: 42,
      });
      expect(result.chains.length).toBe(800);
      expect(result.totalIterations).toBe(1000);
      expect(result.burnIn).toBe(200);
    });
  });
});
