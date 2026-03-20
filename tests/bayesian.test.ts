import {
  betaBinomial,
  normalNormal,
  gammaPoisson,
  metropolisHastings,
  bayesFactor,
} from "../src/bayesian";

describe("Bayesian Methods", () => {
  // ---- Beta-Binomial ----
  describe("betaBinomial", () => {
    it("uniform prior with no data returns prior", () => {
      const result = betaBinomial(0, 0);
      expect(result.posteriorAlpha).toBe(1);
      expect(result.posteriorBeta).toBe(1);
      expect(result.posteriorMean).toBeCloseTo(0.5, 8);
    });

    it("updates posterior correctly", () => {
      const result = betaBinomial(7, 10, 1, 1);
      expect(result.posteriorAlpha).toBe(8);
      expect(result.posteriorBeta).toBe(4);
      expect(result.posteriorMean).toBeCloseTo(8 / 12, 8);
    });

    it("informative prior shifts posterior", () => {
      // Strong prior: Beta(50, 50) => prior mean = 0.5
      const result = betaBinomial(8, 10, 50, 50);
      // Posterior mean should be between 0.5 (prior) and 0.8 (data)
      expect(result.posteriorMean).toBeGreaterThan(0.5);
      expect(result.posteriorMean).toBeLessThan(0.8);
    });

    it("credible interval contains posterior mean", () => {
      const result = betaBinomial(30, 50);
      expect(result.credibleInterval[0]).toBeLessThan(result.posteriorMean);
      expect(result.credibleInterval[1]).toBeGreaterThan(result.posteriorMean);
    });

    it("posterior variance decreases with more data", () => {
      const small = betaBinomial(3, 10);
      const large = betaBinomial(30, 100);
      expect(large.posteriorVariance).toBeLessThan(small.posteriorVariance);
    });

    it("throws on invalid inputs", () => {
      expect(() => betaBinomial(-1, 10)).toThrow();
      expect(() => betaBinomial(11, 10)).toThrow();
      expect(() => betaBinomial(5, 10, 0, 1)).toThrow();
    });
  });

  // ---- Normal-Normal ----
  describe("normalNormal", () => {
    it("with vague prior, posterior approaches data mean", () => {
      const data = [5.1, 4.9, 5.2, 5.0, 4.8];
      const result = normalNormal(data, 1, 0, 10000);
      const dataMean = data.reduce((a, b) => a + b, 0) / data.length;
      expect(result.posteriorMean).toBeCloseTo(dataMean, 1);
    });

    it("with strong prior, posterior is pulled toward prior", () => {
      const data = [10, 10, 10];
      const result = normalNormal(data, 1, 0, 0.01); // very tight prior at 0
      expect(result.posteriorMean).toBeLessThan(5);
    });

    it("posterior variance is smaller than prior variance", () => {
      const data = [1, 2, 3, 4, 5];
      const result = normalNormal(data, 1, 0, 10);
      expect(result.posteriorVariance).toBeLessThan(result.priorVariance);
    });

    it("credible interval brackets posterior mean", () => {
      const data = [3, 4, 5, 6, 7];
      const result = normalNormal(data, 2);
      expect(result.credibleInterval[0]).toBeLessThan(result.posteriorMean);
      expect(result.credibleInterval[1]).toBeGreaterThan(result.posteriorMean);
    });

    it("throws on empty data", () => {
      expect(() => normalNormal([], 1)).toThrow();
    });

    it("throws on non-positive variance", () => {
      expect(() => normalNormal([1], 0)).toThrow();
      expect(() => normalNormal([1], 1, 0, 0)).toThrow();
    });
  });

  // ---- Gamma-Poisson ----
  describe("gammaPoisson", () => {
    it("updates posterior correctly", () => {
      const data = [3, 5, 4, 6, 2];
      const result = gammaPoisson(data, 1, 1);
      // posterior alpha = 1 + sum(data) = 21
      // posterior beta = 1 + n = 6
      expect(result.posteriorAlpha).toBe(21);
      expect(result.posteriorBeta).toBe(6);
      expect(result.posteriorMean).toBeCloseTo(21 / 6, 8);
    });

    it("posterior variance matches Gamma formula", () => {
      const data = [2, 3, 4];
      const result = gammaPoisson(data, 2, 2);
      const expectedVariance =
        result.posteriorAlpha / (result.posteriorBeta * result.posteriorBeta);
      expect(result.posteriorVariance).toBeCloseTo(expectedVariance, 8);
    });

    it("credible interval is non-negative", () => {
      const data = [0, 1, 0, 0, 1];
      const result = gammaPoisson(data);
      expect(result.credibleInterval[0]).toBeGreaterThanOrEqual(0);
    });

    it("throws on non-integer data", () => {
      expect(() => gammaPoisson([1.5])).toThrow();
    });

    it("throws on negative data", () => {
      expect(() => gammaPoisson([-1])).toThrow();
    });

    it("throws on empty data", () => {
      expect(() => gammaPoisson([])).toThrow();
    });
  });

  // ---- Metropolis-Hastings ----
  describe("metropolisHastings", () => {
    it("samples from a normal distribution", () => {
      // Target: N(5, 1)
      const logPosterior = (x: number) => -0.5 * (x - 5) ** 2;

      // Use a simple LCG for reproducibility
      let seed = 42;
      const random = () => {
        seed = (seed * 1103515245 + 12345) & 0x7fffffff;
        return seed / 0x7fffffff;
      };

      const result = metropolisHastings(logPosterior, {
        nSamples: 5000,
        burnIn: 1000,
        proposalStd: 1,
        initial: 0,
        random,
      });

      expect(result.samples.length).toBe(5000);
      expect(result.mean).toBeCloseTo(5, 0);
      expect(result.std).toBeCloseTo(1, 0);
      expect(result.acceptanceRate).toBeGreaterThan(0.1);
      expect(result.acceptanceRate).toBeLessThan(0.9);
    });

    it("credible interval contains true mean", () => {
      const logPosterior = (x: number) => -0.5 * (x - 3) ** 2;

      let seed = 99;
      const random = () => {
        seed = (seed * 1103515245 + 12345) & 0x7fffffff;
        return seed / 0x7fffffff;
      };

      const result = metropolisHastings(logPosterior, {
        nSamples: 3000,
        burnIn: 500,
        random,
      });

      expect(result.credibleInterval[0]).toBeLessThan(3);
      expect(result.credibleInterval[1]).toBeGreaterThan(3);
    });

    it("thinning reduces sample count appropriately", () => {
      const logPosterior = (x: number) => -x * x;

      const result = metropolisHastings(logPosterior, {
        nSamples: 100,
        burnIn: 0,
        thin: 2,
      });

      expect(result.samples.length).toBe(100);
    });

    it("throws on invalid parameters", () => {
      const lp = (x: number) => -x * x;
      expect(() =>
        metropolisHastings(lp, { nSamples: 0 }),
      ).toThrow();
      expect(() =>
        metropolisHastings(lp, { nSamples: 10, thin: 0 }),
      ).toThrow();
      expect(() =>
        metropolisHastings(lp, { nSamples: 10, proposalStd: 0 }),
      ).toThrow();
    });
  });

  // ---- Bayes Factor ----
  describe("bayesFactor", () => {
    it("returns 1 for identical models", () => {
      const logLik = [-10, -11, -10.5, -10.2];
      const result = bayesFactor(logLik, logLik);
      expect(result.bayesFactor).toBeCloseTo(1, 4);
      expect(result.logBayesFactor).toBeCloseTo(0, 4);
    });

    it("favors model with higher likelihoods", () => {
      const logLik1 = [-5, -5.1, -4.9, -5.2]; // better
      const logLik2 = [-10, -10.1, -9.9, -10.2]; // worse
      const result = bayesFactor(logLik1, logLik2);
      expect(result.bayesFactor).toBeGreaterThan(1);
    });

    it("provides an interpretation string", () => {
      const logLik1 = [-5, -5, -5];
      const logLik2 = [-100, -100, -100];
      const result = bayesFactor(logLik1, logLik2);
      expect(typeof result.interpretation).toBe("string");
      expect(result.interpretation.length).toBeGreaterThan(0);
    });

    it("throws on empty arrays", () => {
      expect(() => bayesFactor([], [-1])).toThrow();
      expect(() => bayesFactor([-1], [])).toThrow();
    });
  });
});
