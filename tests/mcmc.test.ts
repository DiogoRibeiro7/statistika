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
        proposalStd: 1.5,
        iterations: 50000,
        burnIn: 10000,
      });

      expect(result.posteriorMean).toBeCloseTo(3, 0);
      expect(result.posteriorStd).toBeCloseTo(1, 0);
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
        proposalStd: 1.5,
        iterations: 50000,
        burnIn: 10000,
      });

      expect(result.posteriorMeans[0]).toBeCloseTo(2, 0);
      expect(result.posteriorMeans[1]).toBeCloseTo(-1, 0);
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
  });
});
