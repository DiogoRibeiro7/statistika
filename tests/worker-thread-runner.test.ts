/**
 * Tests for src/worker-thread-runner.ts
 *
 * Since the runner is a side-effect module (no exports) designed to execute
 * inside worker threads, we test it by spawning actual Worker instances that
 * load the runner, sending task messages via workerData, and asserting on
 * the messages posted back.
 */

import { Worker } from "node:worker_threads";
import { join } from "node:path";
import { writeFileSync, unlinkSync, mkdirSync, existsSync } from "node:fs";

jest.setTimeout(30_000);

// ---------------------------------------------------------------------------
// Helper: resolve the runner script
// ---------------------------------------------------------------------------

function runnerPath(): string {
  return join(__dirname, "..", "src", "worker-thread-runner.ts");
}

/**
 * Spawn a worker with the given workerData and return the first message it
 * posts back. Rejects on worker-level errors.
 */
function runWorker<T = unknown>(
  workerData: Record<string, unknown>,
): Promise<{ success: boolean; result?: T; error?: string }> {
  return new Promise((resolve, reject) => {
    const worker = new Worker(runnerPath(), { workerData });
    worker.on("message", (msg) => {
      worker.terminate();
      resolve(msg);
    });
    worker.on("error", (err) => {
      worker.terminate();
      reject(err);
    });
  });
}

// ---------------------------------------------------------------------------
// MCMC task
// ---------------------------------------------------------------------------

describe("worker-thread-runner: MCMC task", () => {
  it("should run MCMC and return a chain of correct length", async () => {
    const msg = await runWorker<number[]>({
      type: "mcmc",
      iterations: 1000,
      burnIn: 200,
      proposalStd: 1.0,
      seed: 42,
      logDensitySource: "(x) => -0.5 * x * x",
    });

    expect(msg.success).toBe(true);
    expect(msg.result).toBeDefined();
    expect(msg.result).toHaveLength(1000 - 200);
    for (const v of msg.result!) {
      expect(typeof v).toBe("number");
      expect(Number.isFinite(v)).toBe(true);
    }
  });

  it("should produce chain mean near 0 for standard normal", async () => {
    const msg = await runWorker<number[]>({
      type: "mcmc",
      iterations: 10000,
      burnIn: 2000,
      proposalStd: 1.0,
      seed: 7,
      logDensitySource: "(x) => -0.5 * x * x",
    });

    expect(msg.success).toBe(true);
    const chain = msg.result!;
    const chainMean = chain.reduce((a, b) => a + b, 0) / chain.length;
    expect(Math.abs(chainMean)).toBeLessThan(0.5);
  });

  it("should produce chain mean near target for shifted normal", async () => {
    const msg = await runWorker<number[]>({
      type: "mcmc",
      iterations: 10000,
      burnIn: 2000,
      proposalStd: 1.0,
      seed: 99,
      logDensitySource: "(x) => -0.5 * (x - 5) * (x - 5)",
    });

    expect(msg.success).toBe(true);
    const chain = msg.result!;
    const chainMean = chain.reduce((a, b) => a + b, 0) / chain.length;
    expect(Math.abs(chainMean - 5)).toBeLessThan(1.0);
  });

  it("should return zero-length chain when burnIn equals iterations", async () => {
    const msg = await runWorker<number[]>({
      type: "mcmc",
      iterations: 500,
      burnIn: 500,
      proposalStd: 1.0,
      seed: 1,
      logDensitySource: "(x) => -0.5 * x * x",
    });

    expect(msg.success).toBe(true);
    expect(msg.result).toHaveLength(0);
  });

  it("should handle NaN-producing log density gracefully", async () => {
    // A density that returns NaN for negative values
    const msg = await runWorker<number[]>({
      type: "mcmc",
      iterations: 500,
      burnIn: 100,
      proposalStd: 1.0,
      seed: 42,
      logDensitySource: "(x) => x < 0 ? NaN : -x",
    });

    expect(msg.success).toBe(true);
    expect(msg.result).toBeDefined();
    expect(msg.result).toHaveLength(400);
    // All returned values should be finite
    for (const v of msg.result!) {
      expect(Number.isFinite(v)).toBe(true);
    }
  });

  it("should produce deterministic results for the same seed", async () => {
    const taskData = {
      type: "mcmc",
      iterations: 500,
      burnIn: 100,
      proposalStd: 1.0,
      seed: 12345,
      logDensitySource: "(x) => -0.5 * x * x",
    };

    const msg1 = await runWorker<number[]>(taskData);
    const msg2 = await runWorker<number[]>(taskData);

    expect(msg1.success).toBe(true);
    expect(msg2.success).toBe(true);
    expect(msg1.result).toEqual(msg2.result);
  });

  it("should propagate error for invalid log density source", async () => {
    const msg = await runWorker({
      type: "mcmc",
      iterations: 100,
      burnIn: 10,
      proposalStd: 1.0,
      seed: 1,
      logDensitySource: "this is not valid javascript %%%",
    });

    expect(msg.success).toBe(false);
    expect(msg.error).toBeDefined();
    expect(typeof msg.error).toBe("string");
  });
});

// ---------------------------------------------------------------------------
// Bootstrap task
// ---------------------------------------------------------------------------

describe("worker-thread-runner: Bootstrap task", () => {
  it("should return the correct number of resamples", async () => {
    const msg = await runWorker<number[]>({
      type: "bootstrap",
      data: [1, 2, 3, 4, 5],
      nResamples: 100,
      seed: 42,
      statisticSource: "(d) => d.reduce((a,b) => a+b, 0) / d.length",
    });

    expect(msg.success).toBe(true);
    expect(msg.result).toHaveLength(100);
  });

  it("should produce numeric finite values for each resample", async () => {
    const msg = await runWorker<number[]>({
      type: "bootstrap",
      data: [10, 20, 30, 40, 50],
      nResamples: 200,
      seed: 99,
      statisticSource: "(d) => d.reduce((a,b) => a+b, 0) / d.length",
    });

    expect(msg.success).toBe(true);
    for (const v of msg.result!) {
      expect(typeof v).toBe("number");
      expect(Number.isFinite(v)).toBe(true);
    }
  });

  it("should produce bootstrap mean near the population mean", async () => {
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const populationMean = 5.5;

    const msg = await runWorker<number[]>({
      type: "bootstrap",
      data,
      nResamples: 2000,
      seed: 7,
      statisticSource: "(d) => d.reduce((a,b) => a+b, 0) / d.length",
    });

    expect(msg.success).toBe(true);
    const bootstrapMean =
      msg.result!.reduce((a, b) => a + b, 0) / msg.result!.length;
    expect(Math.abs(bootstrapMean - populationMean)).toBeLessThan(0.5);
  });

  it("should produce deterministic results for the same seed", async () => {
    const taskData = {
      type: "bootstrap",
      data: [1, 2, 3, 4, 5],
      nResamples: 50,
      seed: 555,
      statisticSource: "(d) => d.reduce((a,b) => a+b, 0) / d.length",
    };

    const msg1 = await runWorker<number[]>(taskData);
    const msg2 = await runWorker<number[]>(taskData);

    expect(msg1.success).toBe(true);
    expect(msg2.success).toBe(true);
    expect(msg1.result).toEqual(msg2.result);
  });

  it("should handle single-element data", async () => {
    const msg = await runWorker<number[]>({
      type: "bootstrap",
      data: [42],
      nResamples: 50,
      seed: 1,
      statisticSource: "(d) => d.reduce((a,b) => a+b, 0) / d.length",
    });

    expect(msg.success).toBe(true);
    expect(msg.result).toHaveLength(50);
    // All resamples from a single element should produce the same value
    for (const v of msg.result!) {
      expect(v).toBe(42);
    }
  });

  it("should handle zero resamples", async () => {
    const msg = await runWorker<number[]>({
      type: "bootstrap",
      data: [1, 2, 3],
      nResamples: 0,
      seed: 1,
      statisticSource: "(d) => d.reduce((a,b) => a+b, 0) / d.length",
    });

    expect(msg.success).toBe(true);
    expect(msg.result).toHaveLength(0);
  });

  it("should propagate error for invalid statistic source", async () => {
    const msg = await runWorker({
      type: "bootstrap",
      data: [1, 2, 3],
      nResamples: 10,
      seed: 1,
      statisticSource: "not valid js {{{}}}",
    });

    expect(msg.success).toBe(false);
    expect(msg.error).toBeDefined();
  });

  it("should work with a different statistic (max)", async () => {
    const data = [1, 2, 3, 4, 5];
    const msg = await runWorker<number[]>({
      type: "bootstrap",
      data,
      nResamples: 100,
      seed: 42,
      statisticSource: "(d) => Math.max(...d)",
    });

    expect(msg.success).toBe(true);
    expect(msg.result).toHaveLength(100);
    for (const v of msg.result!) {
      // Max of any resample from [1..5] must be between 1 and 5
      expect(v).toBeGreaterThanOrEqual(1);
      expect(v).toBeLessThanOrEqual(5);
    }
  });
});

// ---------------------------------------------------------------------------
// Cross-validation task
// ---------------------------------------------------------------------------

describe("worker-thread-runner: Cross-validation task", () => {
  const meanPredictorFn = `(trX, trY, teX, teY) => {
    const m = trY.reduce((a,b) => a+b, 0) / trY.length;
    const errs = teY.map(y => (y - m) ** 2);
    return errs.reduce((a,b) => a+b, 0) / errs.length;
  }`;

  it("should run a single fold and return a numeric score", async () => {
    const msg = await runWorker<number>({
      type: "crossvalidation",
      trainData: [[1], [2], [3], [4]],
      trainLabels: [1, 2, 3, 4],
      testData: [[5], [6]],
      testLabels: [5, 6],
      modelFnSource: meanPredictorFn,
    });

    expect(msg.success).toBe(true);
    expect(typeof msg.result).toBe("number");
    expect(Number.isFinite(msg.result!)).toBe(true);
    expect(msg.result!).toBeGreaterThanOrEqual(0);
  });

  it("should return zero MSE when test labels match predicted mean", async () => {
    // Train mean = 3, test labels = [3, 3] -> MSE = 0
    const msg = await runWorker<number>({
      type: "crossvalidation",
      trainData: [[1], [2], [3], [4], [5]],
      trainLabels: [1, 2, 3, 4, 5],
      testData: [[3]],
      testLabels: [3],
      modelFnSource: meanPredictorFn,
    });

    expect(msg.success).toBe(true);
    expect(msg.result).toBeCloseTo(0, 10);
  });

  it("should propagate error for invalid model function source", async () => {
    const msg = await runWorker({
      type: "crossvalidation",
      trainData: [[1]],
      trainLabels: [1],
      testData: [[2]],
      testLabels: [2],
      modelFnSource: "invalid js |||",
    });

    expect(msg.success).toBe(false);
    expect(msg.error).toBeDefined();
  });

  it("should handle empty test set", async () => {
    // statistic returns NaN/0 for empty arrays depending on implementation
    const msg = await runWorker<number>({
      type: "crossvalidation",
      trainData: [[1], [2], [3]],
      trainLabels: [1, 2, 3],
      testData: [],
      testLabels: [],
      modelFnSource: `(trX, trY, teX, teY) => {
        if (teY.length === 0) return 0;
        const m = trY.reduce((a,b) => a+b, 0) / trY.length;
        const errs = teY.map(y => (y - m) ** 2);
        return errs.reduce((a,b) => a+b, 0) / errs.length;
      }`,
    });

    expect(msg.success).toBe(true);
    expect(msg.result).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// Generic task
// ---------------------------------------------------------------------------

describe("worker-thread-runner: Generic task", () => {
  const tmpDir = join(__dirname, "__tmp_worker_test__");
  const taskWithDefault = join(tmpDir, "task-default.js");
  const taskWithRun = join(tmpDir, "task-run.js");
  const taskWithNone = join(tmpDir, "task-none.js");
  const taskThrows = join(tmpDir, "task-throws.js");
  const taskAsync = join(tmpDir, "task-async.js");

  beforeAll(() => {
    if (!existsSync(tmpDir)) {
      mkdirSync(tmpDir, { recursive: true });
    }

    // Module with default export
    writeFileSync(
      taskWithDefault,
      `module.exports = function(input) { return input * 2; };
       module.exports.default = module.exports;`,
    );

    // Module with run export
    writeFileSync(
      taskWithRun,
      `module.exports.run = function(input) { return input + 10; };`,
    );

    // Module with neither default nor run
    writeFileSync(
      taskWithNone,
      `module.exports.something = function() { return 1; };`,
    );

    // Module that throws
    writeFileSync(
      taskThrows,
      `function throwFn(input) { throw new Error("task failed: " + input); }
       module.exports = throwFn;
       module.exports.default = throwFn;`,
    );

    // Async module
    writeFileSync(
      taskAsync,
      `module.exports = async function(input) { return { doubled: input * 2 }; };
       module.exports.default = module.exports;`,
    );
  });

  afterAll(() => {
    const files = [
      taskWithDefault,
      taskWithRun,
      taskWithNone,
      taskThrows,
      taskAsync,
    ];
    for (const f of files) {
      try {
        unlinkSync(f);
      } catch {
        // ignore
      }
    }
    try {
      const { rmdirSync } = require("fs");
      rmdirSync(tmpDir);
    } catch {
      // ignore
    }
  });

  it("should call default export and return result", async () => {
    const msg = await runWorker<number>({
      type: "generic",
      taskFile: taskWithDefault,
      input: 21,
    });

    expect(msg.success).toBe(true);
    expect(msg.result).toBe(42);
  });

  it("should call run export when no default", async () => {
    const msg = await runWorker<number>({
      type: "generic",
      taskFile: taskWithRun,
      input: 5,
    });

    expect(msg.success).toBe(true);
    expect(msg.result).toBe(15);
  });

  it("should report error when module exports neither default nor run", async () => {
    const msg = await runWorker({
      type: "generic",
      taskFile: taskWithNone,
      input: 1,
    });

    expect(msg.success).toBe(false);
    expect(msg.error).toContain("must export a default function");
  });

  it("should propagate errors thrown inside the task", async () => {
    const msg = await runWorker({
      type: "generic",
      taskFile: taskThrows,
      input: "oops",
    });

    expect(msg.success).toBe(false);
    expect(msg.error).toContain("task failed: oops");
  });

  it("should handle async tasks", async () => {
    const msg = await runWorker<{ doubled: number }>({
      type: "generic",
      taskFile: taskAsync,
      input: 7,
    });

    expect(msg.success).toBe(true);
    expect(msg.result).toEqual({ doubled: 14 });
  });

  it("should report error for nonexistent task file", async () => {
    const msg = await runWorker({
      type: "generic",
      taskFile: join(tmpDir, "nonexistent-module.js"),
      input: 1,
    });

    expect(msg.success).toBe(false);
    expect(msg.error).toBeDefined();
  });
});

// ---------------------------------------------------------------------------
// Unknown task type
// ---------------------------------------------------------------------------

describe("worker-thread-runner: Unknown task type", () => {
  it("should report error for unknown task type", async () => {
    const msg = await runWorker({
      type: "foobar",
    });

    expect(msg.success).toBe(false);
    expect(msg.error).toContain("Unknown task type");
  });
});

// ---------------------------------------------------------------------------
// Multiple concurrent tasks
// ---------------------------------------------------------------------------

describe("worker-thread-runner: Concurrent execution", () => {
  it("should handle multiple MCMC workers running concurrently", async () => {
    const seeds = [1, 2, 3, 4];
    const promises = seeds.map((seed) =>
      runWorker<number[]>({
        type: "mcmc",
        iterations: 1000,
        burnIn: 200,
        proposalStd: 1.0,
        seed,
        logDensitySource: "(x) => -0.5 * x * x",
      }),
    );

    const results = await Promise.all(promises);

    for (const msg of results) {
      expect(msg.success).toBe(true);
      expect(msg.result).toHaveLength(800);
    }

    // Different seeds should produce different chains
    expect(results[0].result).not.toEqual(results[1].result);
    expect(results[2].result).not.toEqual(results[3].result);
  });

  it("should handle mixed task types running concurrently", async () => {
    const mcmcPromise = runWorker<number[]>({
      type: "mcmc",
      iterations: 500,
      burnIn: 100,
      proposalStd: 1.0,
      seed: 42,
      logDensitySource: "(x) => -0.5 * x * x",
    });

    const bootstrapPromise = runWorker<number[]>({
      type: "bootstrap",
      data: [1, 2, 3, 4, 5],
      nResamples: 100,
      seed: 42,
      statisticSource: "(d) => d.reduce((a,b) => a+b, 0) / d.length",
    });

    const cvPromise = runWorker<number>({
      type: "crossvalidation",
      trainData: [[1], [2], [3], [4]],
      trainLabels: [1, 2, 3, 4],
      testData: [[5], [6]],
      testLabels: [5, 6],
      modelFnSource: `(trX, trY, teX, teY) => {
        const m = trY.reduce((a,b) => a+b, 0) / trY.length;
        const errs = teY.map(y => (y - m) ** 2);
        return errs.reduce((a,b) => a+b, 0) / errs.length;
      }`,
    });

    const [mcmc, bootstrap, cv] = await Promise.all([
      mcmcPromise,
      bootstrapPromise,
      cvPromise,
    ]);

    expect(mcmc.success).toBe(true);
    expect(mcmc.result).toHaveLength(400);

    expect(bootstrap.success).toBe(true);
    expect(bootstrap.result).toHaveLength(100);

    expect(cv.success).toBe(true);
    expect(typeof cv.result).toBe("number");
  });

  it("should isolate failures between concurrent workers", async () => {
    const goodPromise = runWorker<number[]>({
      type: "mcmc",
      iterations: 500,
      burnIn: 100,
      proposalStd: 1.0,
      seed: 42,
      logDensitySource: "(x) => -0.5 * x * x",
    });

    const badPromise = runWorker({
      type: "mcmc",
      iterations: 100,
      burnIn: 10,
      proposalStd: 1.0,
      seed: 1,
      logDensitySource: "invalid javascript !!!",
    });

    const [good, bad] = await Promise.all([goodPromise, badPromise]);

    expect(good.success).toBe(true);
    expect(good.result).toHaveLength(400);

    expect(bad.success).toBe(false);
    expect(bad.error).toBeDefined();
  });
});

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

describe("worker-thread-runner: Edge cases", () => {
  it("should handle large bootstrap data", async () => {
    const largeData = Array.from({ length: 1000 }, (_, i) => i);

    const msg = await runWorker<number[]>({
      type: "bootstrap",
      data: largeData,
      nResamples: 500,
      seed: 42,
      statisticSource: "(d) => d.reduce((a,b) => a+b, 0) / d.length",
    });

    expect(msg.success).toBe(true);
    expect(msg.result).toHaveLength(500);
    const expectedMean = 499.5; // mean of 0..999
    const bootstrapMean =
      msg.result!.reduce((a, b) => a + b, 0) / msg.result!.length;
    expect(Math.abs(bootstrapMean - expectedMean)).toBeLessThan(10);
  });

  it("should handle many MCMC iterations", async () => {
    const msg = await runWorker<number[]>({
      type: "mcmc",
      iterations: 20000,
      burnIn: 5000,
      proposalStd: 1.0,
      seed: 42,
      logDensitySource: "(x) => -0.5 * x * x",
    });

    expect(msg.success).toBe(true);
    expect(msg.result).toHaveLength(15000);
  });

  it("should handle empty bootstrap data", async () => {
    const msg = await runWorker<number[]>({
      type: "bootstrap",
      data: [],
      nResamples: 10,
      seed: 1,
      statisticSource: "(d) => d.length === 0 ? 0 : d.reduce((a,b) => a+b, 0) / d.length",
    });

    expect(msg.success).toBe(true);
    expect(msg.result).toHaveLength(10);
  });

  it("should handle MCMC with zero iterations", async () => {
    const msg = await runWorker<number[]>({
      type: "mcmc",
      iterations: 0,
      burnIn: 0,
      proposalStd: 1.0,
      seed: 1,
      logDensitySource: "(x) => -0.5 * x * x",
    });

    expect(msg.success).toBe(true);
    expect(msg.result).toHaveLength(0);
  });

  it("should handle MCMC with burnIn greater than iterations", async () => {
    const msg = await runWorker<number[]>({
      type: "mcmc",
      iterations: 100,
      burnIn: 200,
      proposalStd: 1.0,
      seed: 1,
      logDensitySource: "(x) => -0.5 * x * x",
    });

    expect(msg.success).toBe(true);
    // burnIn > iterations means no samples are kept
    expect(msg.result).toHaveLength(0);
  });

  it("should handle bootstrap with a variance statistic", async () => {
    const msg = await runWorker<number[]>({
      type: "bootstrap",
      data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
      nResamples: 300,
      seed: 42,
      statisticSource: `(d) => {
        const m = d.reduce((a,b) => a+b, 0) / d.length;
        return d.reduce((a, x) => a + (x - m) ** 2, 0) / d.length;
      }`,
    });

    expect(msg.success).toBe(true);
    expect(msg.result).toHaveLength(300);
    for (const v of msg.result!) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(Number.isFinite(v)).toBe(true);
    }
  });

  it("should handle cross-validation with single-feature data", async () => {
    const msg = await runWorker<number>({
      type: "crossvalidation",
      trainData: [[1], [2], [3]],
      trainLabels: [10, 20, 30],
      testData: [[4]],
      testLabels: [40],
      modelFnSource: `(trX, trY, teX, teY) => {
        const m = trY.reduce((a,b) => a+b, 0) / trY.length;
        return Math.abs(teY[0] - m);
      }`,
    });

    expect(msg.success).toBe(true);
    // Train mean = 20, |40 - 20| = 20
    expect(msg.result).toBeCloseTo(20, 5);
  });
});

// ---------------------------------------------------------------------------
// WorkerRng determinism (tested indirectly through MCMC seeds)
// ---------------------------------------------------------------------------

describe("worker-thread-runner: WorkerRng determinism", () => {
  it("should produce different results with different seeds", async () => {
    const msg1 = await runWorker<number[]>({
      type: "mcmc",
      iterations: 200,
      burnIn: 50,
      proposalStd: 1.0,
      seed: 1,
      logDensitySource: "(x) => -0.5 * x * x",
    });

    const msg2 = await runWorker<number[]>({
      type: "mcmc",
      iterations: 200,
      burnIn: 50,
      proposalStd: 1.0,
      seed: 999,
      logDensitySource: "(x) => -0.5 * x * x",
    });

    expect(msg1.success).toBe(true);
    expect(msg2.success).toBe(true);
    expect(msg1.result).not.toEqual(msg2.result);
  });

  it("should produce identical bootstrap results with the same seed", async () => {
    const taskData = {
      type: "bootstrap",
      data: [10, 20, 30, 40, 50],
      nResamples: 100,
      seed: 42,
      statisticSource: "(d) => Math.max(...d)",
    };

    const msg1 = await runWorker<number[]>(taskData);
    const msg2 = await runWorker<number[]>(taskData);

    expect(msg1.success).toBe(true);
    expect(msg2.success).toBe(true);
    expect(msg1.result).toEqual(msg2.result);
  });
});
