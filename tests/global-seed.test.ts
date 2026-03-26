import {
  setGlobalSeed,
  getGlobalRng,
  resetGlobalSeed,
  SeededRng,
} from "../src/random";
import { Normal } from "../src/distributions/continuous/normal";
import { GammaDistribution } from "../src/distributions/continuous/gamma";
import { Poisson } from "../src/distributions/discrete/poisson";
import { Bernoulli } from "../src/distributions/discrete/bernoulli";
import { MultivariateNormal } from "../src/distributions/multivariate/multivariate-normal";

afterEach(() => {
  // Always clean up global state between tests
  resetGlobalSeed();
});

describe("Global Seed Management", () => {
  describe("setGlobalSeed / getGlobalRng / resetGlobalSeed", () => {
    it("getGlobalRng returns null when no seed is set", () => {
      expect(getGlobalRng()).toBeNull();
    });

    it("getGlobalRng returns a function after setGlobalSeed", () => {
      setGlobalSeed(42);
      const rng = getGlobalRng();
      expect(rng).toBeInstanceOf(Function);
      expect(typeof rng!()).toBe("number");
    });

    it("resetGlobalSeed clears the global RNG", () => {
      setGlobalSeed(42);
      expect(getGlobalRng()).not.toBeNull();
      resetGlobalSeed();
      expect(getGlobalRng()).toBeNull();
    });

    it("setGlobalSeed throws on non-finite seed", () => {
      expect(() => setGlobalSeed(NaN)).toThrow("finite");
      expect(() => setGlobalSeed(Infinity)).toThrow("finite");
    });
  });

  describe("deterministic results with global seed", () => {
    it("same seed produces identical Normal samples", () => {
      setGlobalSeed(42);
      const d1 = new Normal(0, 1);
      const samples1 = d1.sampleN(20);

      setGlobalSeed(42);
      const d2 = new Normal(0, 1);
      const samples2 = d2.sampleN(20);

      expect(samples1).toEqual(samples2);
    });

    it("same seed produces identical Gamma samples", () => {
      setGlobalSeed(99);
      const d1 = new GammaDistribution(2, 1);
      const samples1 = d1.sampleN(20);

      setGlobalSeed(99);
      const d2 = new GammaDistribution(2, 1);
      const samples2 = d2.sampleN(20);

      expect(samples1).toEqual(samples2);
    });

    it("same seed produces identical Poisson samples (discrete)", () => {
      setGlobalSeed(123);
      const d1 = new Poisson(5);
      const samples1 = d1.sampleN(20);

      setGlobalSeed(123);
      const d2 = new Poisson(5);
      const samples2 = d2.sampleN(20);

      expect(samples1).toEqual(samples2);
    });

    it("same seed produces identical Bernoulli samples (discrete)", () => {
      setGlobalSeed(7);
      const d1 = new Bernoulli(0.3);
      const samples1 = d1.sampleN(50);

      setGlobalSeed(7);
      const d2 = new Bernoulli(0.3);
      const samples2 = d2.sampleN(50);

      expect(samples1).toEqual(samples2);
    });

    it("same seed produces identical MultivariateNormal samples", () => {
      setGlobalSeed(42);
      const d1 = new MultivariateNormal([0, 0], [[1, 0.5], [0.5, 1]]);
      const samples1 = d1.sampleN(10);

      setGlobalSeed(42);
      const d2 = new MultivariateNormal([0, 0], [[1, 0.5], [0.5, 1]]);
      const samples2 = d2.sampleN(10);

      expect(samples1).toEqual(samples2);
    });
  });

  describe("different seeds produce different results", () => {
    it("different seeds give different Normal samples", () => {
      setGlobalSeed(1);
      const d1 = new Normal(0, 1);
      const samples1 = d1.sampleN(10);

      setGlobalSeed(2);
      const d2 = new Normal(0, 1);
      const samples2 = d2.sampleN(10);

      expect(samples1).not.toEqual(samples2);
    });

    it("different seeds give different Gamma samples", () => {
      setGlobalSeed(100);
      const d1 = new GammaDistribution(3, 2);
      const samples1 = d1.sampleN(10);

      setGlobalSeed(200);
      const d2 = new GammaDistribution(3, 2);
      const samples2 = d2.sampleN(10);

      expect(samples1).not.toEqual(samples2);
    });
  });

  describe("resetGlobalSeed returns to non-deterministic behaviour", () => {
    it("after reset, distributions no longer share a seeded RNG", () => {
      setGlobalSeed(42);
      resetGlobalSeed();

      // After reset, two distributions created independently should use
      // Math.random, which (almost certainly) produces different sequences.
      // We test that getGlobalRng() is null; functional non-determinism
      // can't be asserted reliably, but we verify the mechanism.
      expect(getGlobalRng()).toBeNull();
    });
  });

  describe("explicit rng parameter overrides global seed", () => {
    it("explicit rng on Normal overrides the global seed", () => {
      setGlobalSeed(42);

      // Create a distribution with an explicit RNG
      const explicitRng = new SeededRng(999);
      const dExplicit = new Normal(0, 1, () => explicitRng.next());
      const samplesExplicit = dExplicit.sampleN(10);

      // Create a distribution using the global seed
      setGlobalSeed(42);
      const dGlobal = new Normal(0, 1);
      const samplesGlobal = dGlobal.sampleN(10);

      // They should differ because the explicit RNG uses seed 999, not 42
      expect(samplesExplicit).not.toEqual(samplesGlobal);

      // And the explicit RNG should be reproducible on its own
      const explicitRng2 = new SeededRng(999);
      const dExplicit2 = new Normal(0, 1, () => explicitRng2.next());
      const samplesExplicit2 = dExplicit2.sampleN(10);
      expect(samplesExplicit).toEqual(samplesExplicit2);
    });

    it("explicit rng on GammaDistribution overrides the global seed", () => {
      setGlobalSeed(42);

      const explicitRng = new SeededRng(777);
      const dExplicit = new GammaDistribution(2, 1, () => explicitRng.next());
      const samplesExplicit = dExplicit.sampleN(10);

      setGlobalSeed(42);
      const dGlobal = new GammaDistribution(2, 1);
      const samplesGlobal = dGlobal.sampleN(10);

      expect(samplesExplicit).not.toEqual(samplesGlobal);
    });

    it("explicit rng on Poisson (discrete) overrides the global seed", () => {
      setGlobalSeed(42);

      const explicitRng = new SeededRng(555);
      const dExplicit = new Poisson(5, () => explicitRng.next());
      const samplesExplicit = dExplicit.sampleN(20);

      setGlobalSeed(42);
      const dGlobal = new Poisson(5);
      const samplesGlobal = dGlobal.sampleN(20);

      expect(samplesExplicit).not.toEqual(samplesGlobal);
    });
  });

  describe("global seed works across multiple distributions", () => {
    it("sequence of distribution constructions is deterministic", () => {
      setGlobalSeed(42);
      const norm = new Normal(0, 1);
      const s1 = norm.sample();
      const gamma = new GammaDistribution(2, 1);
      const s2 = gamma.sample();

      setGlobalSeed(42);
      const norm2 = new Normal(0, 1);
      const s3 = norm2.sample();
      const gamma2 = new GammaDistribution(2, 1);
      const s4 = gamma2.sample();

      expect(s1).toBe(s3);
      expect(s2).toBe(s4);
    });
  });
});
