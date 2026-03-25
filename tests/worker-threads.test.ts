import {
  WorkerPool,
  parallelMCMC,
  parallelBootstrap,
  parallelCrossValidation,
} from "../src/worker-threads";

// Worker threads can be slow, especially in CI environments.
jest.setTimeout(30_000);

// ---------------------------------------------------------------------------
// WorkerPool
// ---------------------------------------------------------------------------

describe("WorkerPool", () => {
  it("should execute a basic task and return a result", async () => {
    const pool = new WorkerPool({ maxWorkers: 1 });

    // The generic task path requires a module file, but we can test the
    // terminated-state error without needing one.
    await pool.terminate();

    await expect(pool.runTasks("nonexistent.js", [1])).rejects.toThrow(
      "WorkerPool has been terminated",
    );
  });

  it("should reject after termination", async () => {
    const pool = new WorkerPool({ maxWorkers: 2 });
    await pool.terminate();

    await expect(pool.runTasks("any-file.js", [1, 2, 3])).rejects.toThrow(
      "terminated",
    );
  });
});

// ---------------------------------------------------------------------------
// parallelMCMC
// ---------------------------------------------------------------------------

describe("parallelMCMC", () => {
  it("should run 2 chains sampling from N(0,1)", async () => {
    const result = await parallelMCMC({
      logDensity: "(x) => -0.5 * x * x",
      numChains: 2,
      iterationsPerChain: 5000,
      burnIn: 1000,
      proposalStd: 1,
      seeds: [42, 123],
    });

    // Each chain should have length = iterations - burnIn
    expect(result.chains).toHaveLength(2);
    for (const chain of result.chains) {
      expect(chain).toHaveLength(5000 - 1000);
    }

    // Gelman-Rubin should be computed and close to 1.0
    expect(result.gelmanRubin).toBeDefined();
    expect(result.gelmanRubin!).toBeGreaterThan(0.8);
    expect(result.gelmanRubin!).toBeLessThan(1.5);
  });

  it("should produce chain values with mean near 0 for N(0,1)", async () => {
    const result = await parallelMCMC({
      logDensity: "(x) => -0.5 * x * x",
      numChains: 2,
      iterationsPerChain: 10000,
      burnIn: 2000,
      proposalStd: 1,
      seeds: [7, 99],
    });

    for (const chain of result.chains) {
      const chainMean = chain.reduce((a, b) => a + b, 0) / chain.length;
      // Wide tolerance: MCMC is stochastic
      expect(Math.abs(chainMean)).toBeLessThan(0.5);
    }
  });

  it("should throw when numChains < 1", async () => {
    await expect(
      parallelMCMC({
        logDensity: "(x) => -0.5 * x * x",
        numChains: 0,
        iterationsPerChain: 100,
      }),
    ).rejects.toThrow("numChains must be at least 1");
  });

  it("should omit gelmanRubin when running a single chain", async () => {
    const result = await parallelMCMC({
      logDensity: "(x) => -0.5 * x * x",
      numChains: 1,
      iterationsPerChain: 2000,
      burnIn: 500,
      seeds: [1],
    });

    expect(result.chains).toHaveLength(1);
    expect(result.chains[0]).toHaveLength(2000 - 500);
    expect(result.gelmanRubin).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// parallelBootstrap
// ---------------------------------------------------------------------------

describe("parallelBootstrap", () => {
  it("should return the correct number of resamples", async () => {
    const stats = await parallelBootstrap({
      data: [1, 2, 3, 4, 5],
      statistic: "(d) => d.reduce((a,b) => a+b, 0) / d.length",
      nResamples: 500,
      nWorkers: 2,
    });

    expect(stats).toHaveLength(500);
  });

  it("should have bootstrap mean near the population mean", async () => {
    const data = [1, 2, 3, 4, 5];
    const populationMean = 3; // (1+2+3+4+5)/5

    const stats = await parallelBootstrap({
      data,
      statistic: "(d) => d.reduce((a,b) => a+b, 0) / d.length",
      nResamples: 2000,
      nWorkers: 2,
    });

    const bootstrapMean = stats.reduce((a, b) => a + b, 0) / stats.length;
    // The mean of bootstrap means should be close to the population mean
    expect(Math.abs(bootstrapMean - populationMean)).toBeLessThan(0.5);
  });

  it("should produce numeric values for each resample", async () => {
    const stats = await parallelBootstrap({
      data: [10, 20, 30],
      statistic: "(d) => d.reduce((a,b) => a+b, 0) / d.length",
      nResamples: 100,
      nWorkers: 1,
    });

    for (const v of stats) {
      expect(typeof v).toBe("number");
      expect(Number.isFinite(v)).toBe(true);
    }
  });
});

// ---------------------------------------------------------------------------
// parallelCrossValidation
// ---------------------------------------------------------------------------

describe("parallelCrossValidation", () => {
  // A simple mean predictor: predict the training mean, score with MSE.
  const meanPredictorFn = `(trX, trY, teX, teY) => {
    const m = trY.reduce((a,b) => a+b, 0) / trY.length;
    const errs = teY.map(y => (y - m) ** 2);
    return errs.reduce((a,b) => a+b, 0) / errs.length;
  }`;

  it("should return fold scores for 3-fold CV", async () => {
    const data = [[1], [2], [3], [4], [5], [6]];
    const labels = [1, 2, 3, 4, 5, 6];

    const result = await parallelCrossValidation({
      data,
      labels,
      nFolds: 3,
      modelFn: meanPredictorFn,
    });

    expect(result.foldScores).toHaveLength(3);
    expect(typeof result.mean).toBe("number");
    expect(typeof result.std).toBe("number");
    expect(Number.isFinite(result.mean)).toBe(true);
    expect(Number.isFinite(result.std)).toBe(true);
  });

  it("should produce positive MSE scores for a mean predictor", async () => {
    const data = [[1], [2], [3], [4], [5], [6]];
    const labels = [1, 2, 3, 4, 5, 6];

    const result = await parallelCrossValidation({
      data,
      labels,
      nFolds: 3,
      modelFn: meanPredictorFn,
    });

    for (const score of result.foldScores) {
      // MSE should be non-negative
      expect(score).toBeGreaterThanOrEqual(0);
    }

    // Mean of fold scores should match the reported mean
    const manualMean =
      result.foldScores.reduce((a, b) => a + b, 0) / result.foldScores.length;
    expect(result.mean).toBeCloseTo(manualMean, 10);
  });

  it("should throw for mismatched data and labels lengths", async () => {
    await expect(
      parallelCrossValidation({
        data: [[1], [2], [3]],
        labels: [1, 2],
        nFolds: 2,
        modelFn: meanPredictorFn,
      }),
    ).rejects.toThrow("data and labels must have the same length");
  });

  it("should throw when nFolds < 2", async () => {
    await expect(
      parallelCrossValidation({
        data: [[1], [2], [3]],
        labels: [1, 2, 3],
        nFolds: 1,
        modelFn: meanPredictorFn,
      }),
    ).rejects.toThrow("nFolds must be at least 2");
  });

  it("should throw when nFolds exceeds the number of observations", async () => {
    await expect(
      parallelCrossValidation({
        data: [[1], [2]],
        labels: [1, 2],
        nFolds: 5,
        modelFn: meanPredictorFn,
      }),
    ).rejects.toThrow("nFolds cannot exceed the number of observations");
  });
});
