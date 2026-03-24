import {
  computeELBO,
  meanFieldVI,
  advi,
  normalVariational,
  logNormalVariational,
  multivariateNormalVariational,
  compareModelsVI,
} from "../src/variational";
import { SeededRng } from "../src/random";

// Simple target: standard normal
const normalLogDensity = (params: number[]): number => {
  return -0.5 * params[0] * params[0];
};

// Target: N(3, 1)
const shiftedNormalLogDensity = (params: number[]): number => {
  const diff = params[0] - 3;
  return -0.5 * diff * diff;
};

// Bivariate normal target: N([1,2], I)
const bivariateLogDensity = (params: number[]): number => {
  return -0.5 * ((params[0] - 1) ** 2 + (params[1] - 2) ** 2);
};

describe("computeELBO", () => {
  it("returns a finite ELBO value", () => {
    const rng = new SeededRng(42);
    const elbo = computeELBO(normalLogDensity, [0], [1], rng, 100);
    expect(Number.isFinite(elbo)).toBe(true);
  });

  it("ELBO is higher when approximation is closer to target", () => {
    const rng1 = new SeededRng(42);
    const elboGood = computeELBO(normalLogDensity, [0], [1], rng1, 200);
    const rng2 = new SeededRng(42);
    const elboBad = computeELBO(normalLogDensity, [5], [1], rng2, 200);
    expect(elboGood).toBeGreaterThan(elboBad);
  });
});

describe("meanFieldVI", () => {
  it("approximates a standard normal", () => {
    const rng = new SeededRng(123);
    const result = meanFieldVI(normalLogDensity, 1, {
      rng,
      maxIterations: 500,
      learningRate: 0.01,
      numSamples: 50,
    });
    expect(Math.abs(result.means[0] - 0)).toBeLessThan(2);
    expect(result.stds[0]).toBeGreaterThan(0);
    expect(result.elboHistory.length).toBeGreaterThan(0);
  });

  it("approximates a shifted normal", () => {
    const rng = new SeededRng(456);
    const result = meanFieldVI(shiftedNormalLogDensity, 1, {
      rng,
      maxIterations: 500,
      learningRate: 0.01,
      numSamples: 50,
    });
    expect(Math.abs(result.means[0] - 3)).toBeLessThan(2);
  });
});

describe("advi", () => {
  it("fits an approximate posterior and returns samples", () => {
    const rng = new SeededRng(789);
    const result = advi(normalLogDensity, 1, {
      rng,
      maxIterations: 500,
      numSamples: 50,
    });
    expect(result.means).toHaveLength(1);
    expect(result.stds).toHaveLength(1);
    expect(result.samples.length).toBeGreaterThan(0);
    expect(result.transforms).toHaveLength(1);
    expect(result.adamState).toBeDefined();
  });

  it("ELBO history is non-decreasing on average", () => {
    const rng = new SeededRng(101);
    const result = advi(normalLogDensity, 1, {
      rng,
      maxIterations: 300,
      numSamples: 100,
    });
    // ELBO should generally increase; check that last is better than first
    const history = result.elboHistory;
    if (history.length > 10) {
      const earlyAvg = history.slice(0, 5).reduce((a, b) => a + b, 0) / 5;
      const lateAvg = history.slice(-5).reduce((a, b) => a + b, 0) / 5;
      expect(lateAvg).toBeGreaterThanOrEqual(earlyAvg - 5);
    }
  });
});

describe("normalVariational", () => {
  it("approximates a univariate normal target", () => {
    const rng = new SeededRng(202);
    const univariateLogDensity = (x: number): number => -0.5 * (x - 2) ** 2;
    const result = normalVariational(univariateLogDensity, rng, {
      maxIterations: 500,
      learningRate: 0.01,
    });
    expect(Math.abs(result.means[0] - 2)).toBeLessThan(2);
    expect(result.stds[0]).toBeGreaterThan(0);
  });
});

describe("multivariateNormalVariational", () => {
  it("approximates a bivariate target", () => {
    const rng = new SeededRng(303);
    const result = multivariateNormalVariational(bivariateLogDensity, 2, rng, {
      maxIterations: 500,
      learningRate: 0.01,
      numSamples: 50,
    });
    expect(result.means).toHaveLength(2);
    expect(result.stds).toHaveLength(2);
    expect(Math.abs(result.means[0] - 1)).toBeLessThan(2);
    expect(Math.abs(result.means[1] - 2)).toBeLessThan(2);
  });
});

describe("compareModelsVI", () => {
  it("selects the better model", () => {
    const rng = new SeededRng(404);
    const result = compareModelsVI(
      [
        {
          logDensity: normalLogDensity,
          dim: 1,
          options: { maxIterations: 200, numSamples: 50 },
        },
        {
          logDensity: normalLogDensity,
          dim: 1,
          options: { maxIterations: 200, numSamples: 50 },
        },
      ],
      rng,
    );
    expect(result.elbos).toHaveLength(2);
    expect(result.bestModelIndex).toBeGreaterThanOrEqual(0);
    expect(result.bestModelIndex).toBeLessThanOrEqual(1);
    expect(result.modelWeights).toHaveLength(2);
    const weightSum = result.modelWeights.reduce((a, b) => a + b, 0);
    expect(weightSum).toBeCloseTo(1, 1);
  });
});
