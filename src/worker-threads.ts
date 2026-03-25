/**
 * Worker thread utilities for parallelizing expensive statistical computations.
 *
 * Supports running MCMC chains, bootstrap resampling, and cross-validation
 * folds in parallel using Node.js worker_threads.
 *
 * Since functions cannot be serialized across thread boundaries, the API
 * accepts serialized function source strings (e.g. `"(x) => -0.5 * x * x"`)
 * which are reconstituted inside each worker via `new Function`.
 */

import { Worker } from "node:worker_threads";
import { cpus } from "node:os";
import { join } from "node:path";
import { mean, variance } from "./utils/descriptive";

// ---------------------------------------------------------------------------
// Resolve the worker script path
// ---------------------------------------------------------------------------

function resolveWorkerScript(): string {
  // In compiled output the runner sits next to this file.
  // During development (ts-node / jest) we need to point at the .ts source
  // and rely on ts-node to compile it.
  const base = typeof __dirname !== "undefined" ? __dirname : process.cwd();
  // Try .js first (compiled), then .ts (source)
  const jsPath = join(base, "worker-thread-runner.js");
  const tsPath = join(base, "worker-thread-runner.ts");

  try {
    require.resolve(jsPath);
    return jsPath;
  } catch {
    return tsPath;
  }
}

// ---------------------------------------------------------------------------
// WorkerPool
// ---------------------------------------------------------------------------

/** Options for creating a {@link WorkerPool}. */
export interface WorkerPoolOptions {
  /** Maximum number of worker threads. Defaults to `os.cpus().length - 1`. */
  maxWorkers?: number;
}

/**
 * A simple worker pool that distributes tasks across threads.
 *
 * Each call to {@link runTasks} spawns workers (up to `maxWorkers`),
 * distributes the inputs, collects results, and terminates the workers.
 *
 * @example
 * ```ts
 * const pool = new WorkerPool({ maxWorkers: 4 });
 * const results = await pool.runTasks<number>('./my-task.js', [1, 2, 3]);
 * await pool.terminate();
 * ```
 */
export class WorkerPool {
  private readonly maxWorkers: number;
  private workers: Worker[] = [];
  private terminated = false;

  constructor(options?: WorkerPoolOptions) {
    this.maxWorkers = options?.maxWorkers ?? Math.max(1, cpus().length - 1);
  }

  /**
   * Run multiple tasks in parallel, returning results in input order.
   *
   * @param taskFile - Absolute path to a module that exports a default or `run` function.
   * @param inputs - Array of inputs; each is passed to one worker invocation.
   * @returns Array of results in the same order as `inputs`.
   */
  async runTasks<T>(taskFile: string, inputs: unknown[]): Promise<T[]> {
    if (this.terminated) {
      throw new Error("WorkerPool has been terminated");
    }

    const workerScript = resolveWorkerScript();
    const results: T[] = new Array(inputs.length);
    let nextIndex = 0;

    const workerCount = Math.min(this.maxWorkers, inputs.length);

    await new Promise<void>((resolve, reject) => {
      let completed = 0;
      let hasErrored = false;

      const spawnWorker = () => {
        if (nextIndex >= inputs.length || hasErrored) return;

        const idx = nextIndex++;
        const worker = new Worker(workerScript, {
          workerData: {
            type: "generic" as const,
            taskFile,
            input: inputs[idx],
          },
        });
        this.workers.push(worker);

        worker.on("message", (msg: { success: boolean; result?: T; error?: string }) => {
          if (msg.success) {
            results[idx] = msg.result as T;
          } else {
            hasErrored = true;
            reject(new Error(msg.error ?? "Worker task failed"));
            return;
          }
          completed++;
          if (completed === inputs.length) {
            resolve();
          } else {
            // Reuse this slot for the next task
            worker.terminate();
            spawnWorker();
          }
        });

        worker.on("error", (err) => {
          if (!hasErrored) {
            hasErrored = true;
            reject(err);
          }
        });
      };

      for (let i = 0; i < workerCount; i++) {
        spawnWorker();
      }
    });

    return results;
  }

  /** Terminate all workers in the pool. */
  async terminate(): Promise<void> {
    this.terminated = true;
    await Promise.all(this.workers.map((w) => w.terminate()));
    this.workers = [];
  }
}

// ---------------------------------------------------------------------------
// parallelMCMC
// ---------------------------------------------------------------------------

/**
 * Run multiple MCMC chains in parallel.
 *
 * Each chain runs a Metropolis-Hastings sampler in its own worker thread.
 * The `logDensity` parameter must be a string containing valid JavaScript
 * source for a function, e.g. `"(x) => -0.5 * x * x"`.
 *
 * When at least two chains are run, the Gelman-Rubin convergence diagnostic
 * (R-hat) is computed automatically.
 *
 * @example
 * ```ts
 * const result = await parallelMCMC({
 *   logDensity: '(x) => -0.5 * x * x',
 *   numChains: 4,
 *   iterationsPerChain: 10000,
 *   burnIn: 2000,
 * });
 * console.log(result.gelmanRubin); // should be close to 1.0
 * ```
 */
export async function parallelMCMC(options: {
  /** Serialized log-density function source code. */
  logDensity: string;
  /** Number of parallel chains. */
  numChains: number;
  /** Number of iterations per chain (including burn-in). */
  iterationsPerChain: number;
  /** Burn-in period per chain. Defaults to 20% of iterationsPerChain. */
  burnIn?: number;
  /** Proposal standard deviation. Defaults to 1. */
  proposalStd?: number;
  /** Optional per-chain seeds for reproducibility. */
  seeds?: number[];
}): Promise<{ chains: number[][]; gelmanRubin?: number }> {
  const {
    logDensity,
    numChains,
    iterationsPerChain,
    burnIn = Math.floor(iterationsPerChain * 0.2),
    proposalStd = 1,
    seeds,
  } = options;

  if (numChains < 1) throw new Error("numChains must be at least 1");

  const workerScript = resolveWorkerScript();

  const chainPromises: Promise<number[]>[] = [];

  for (let i = 0; i < numChains; i++) {
    const seed = seeds?.[i] ?? Math.floor(Math.random() * 2147483647);
    chainPromises.push(
      new Promise<number[]>((resolve, reject) => {
        const worker = new Worker(workerScript, {
          workerData: {
            type: "mcmc" as const,
            iterations: iterationsPerChain,
            burnIn,
            proposalStd,
            seed,
            logDensitySource: logDensity,
          },
        });

        worker.on(
          "message",
          (msg: { success: boolean; result?: number[]; error?: string }) => {
            worker.terminate();
            if (msg.success) {
              resolve(msg.result!);
            } else {
              reject(new Error(msg.error ?? "MCMC worker failed"));
            }
          },
        );

        worker.on("error", (err) => {
          worker.terminate();
          reject(err);
        });
      }),
    );
  }

  const chains = await Promise.all(chainPromises);

  // Compute Gelman-Rubin if we have at least 2 chains
  let gr: number | undefined;
  if (chains.length >= 2) {
    gr = computeGelmanRubin(chains);
  }

  return { chains, gelmanRubin: gr };
}

// ---------------------------------------------------------------------------
// parallelBootstrap
// ---------------------------------------------------------------------------

/**
 * Run bootstrap resampling in parallel across worker threads.
 *
 * The `statistic` parameter must be a string containing valid JavaScript
 * source for a function that takes an array and returns a number,
 * e.g. `"(data) => data.reduce((a,b) => a+b, 0) / data.length"`.
 *
 * @example
 * ```ts
 * const stats = await parallelBootstrap({
 *   data: [1, 2, 3, 4, 5],
 *   statistic: '(d) => d.reduce((a,b) => a+b, 0) / d.length',
 *   nResamples: 10000,
 *   nWorkers: 4,
 * });
 * ```
 */
export async function parallelBootstrap(options: {
  /** Original dataset. */
  data: number[];
  /** Serialized statistic function source code. */
  statistic: string;
  /** Total number of bootstrap resamples. */
  nResamples: number;
  /** Number of worker threads to use. Defaults to `os.cpus().length - 1`. */
  nWorkers?: number;
}): Promise<number[]> {
  const { data, statistic, nResamples, nWorkers } = options;
  const numWorkers = Math.min(
    nWorkers ?? Math.max(1, cpus().length - 1),
    nResamples,
  );

  const workerScript = resolveWorkerScript();

  // Divide resamples among workers
  const basePerWorker = Math.floor(nResamples / numWorkers);
  const remainder = nResamples % numWorkers;

  const workerPromises: Promise<number[]>[] = [];

  for (let i = 0; i < numWorkers; i++) {
    const count = basePerWorker + (i < remainder ? 1 : 0);
    const seed = Math.floor(Math.random() * 2147483647);

    workerPromises.push(
      new Promise<number[]>((resolve, reject) => {
        const worker = new Worker(workerScript, {
          workerData: {
            type: "bootstrap" as const,
            data,
            nResamples: count,
            seed,
            statisticSource: statistic,
          },
        });

        worker.on(
          "message",
          (msg: { success: boolean; result?: number[]; error?: string }) => {
            worker.terminate();
            if (msg.success) {
              resolve(msg.result!);
            } else {
              reject(new Error(msg.error ?? "Bootstrap worker failed"));
            }
          },
        );

        worker.on("error", (err) => {
          worker.terminate();
          reject(err);
        });
      }),
    );
  }

  const chunks = await Promise.all(workerPromises);
  // Flatten all chunks into a single results array
  const results: number[] = [];
  for (const chunk of chunks) {
    for (const v of chunk) results.push(v);
  }
  return results;
}

// ---------------------------------------------------------------------------
// parallelCrossValidation
// ---------------------------------------------------------------------------

/**
 * Run cross-validation folds in parallel across worker threads.
 *
 * The `modelFn` parameter must be a string containing valid JavaScript
 * source for a function with signature
 * `(trainData, trainLabels, testData, testLabels) => number` that returns
 * the score for that fold.
 *
 * @example
 * ```ts
 * const cv = await parallelCrossValidation({
 *   data: [[1],[2],[3],[4],[5],[6]],
 *   labels: [1, 2, 3, 4, 5, 6],
 *   nFolds: 3,
 *   modelFn: `(trX, trY, teX, teY) => {
 *     const m = trY.reduce((a,b)=>a+b,0)/trY.length;
 *     const errs = teY.map(y => (y - m) ** 2);
 *     return errs.reduce((a,b)=>a+b,0)/errs.length;
 *   }`,
 * });
 * ```
 */
export async function parallelCrossValidation(options: {
  /** Feature matrix (n x p). */
  data: number[][];
  /** Labels / target values. */
  labels: number[];
  /** Number of folds. */
  nFolds: number;
  /** Serialized model function source code. */
  modelFn: string;
}): Promise<{ foldScores: number[]; mean: number; std: number }> {
  const { data, labels, nFolds, modelFn } = options;
  const n = data.length;

  if (n !== labels.length) {
    throw new Error("data and labels must have the same length");
  }
  if (nFolds < 2) throw new Error("nFolds must be at least 2");
  if (nFolds > n) throw new Error("nFolds cannot exceed the number of observations");

  // Create fold indices
  const indices = Array.from({ length: n }, (_, i) => i);
  // Simple sequential split (no shuffle for reproducibility)
  const foldSize = Math.floor(n / nFolds);

  const workerScript = resolveWorkerScript();
  const foldPromises: Promise<number>[] = [];

  for (let fold = 0; fold < nFolds; fold++) {
    const testStart = fold * foldSize;
    const testEnd = fold === nFolds - 1 ? n : testStart + foldSize;

    const testIndices = indices.slice(testStart, testEnd);
    const trainIndices = [
      ...indices.slice(0, testStart),
      ...indices.slice(testEnd),
    ];

    const trainData = trainIndices.map((i) => data[i]);
    const trainLabels = trainIndices.map((i) => labels[i]);
    const testData = testIndices.map((i) => data[i]);
    const testLabels = testIndices.map((i) => labels[i]);

    foldPromises.push(
      new Promise<number>((resolve, reject) => {
        const worker = new Worker(workerScript, {
          workerData: {
            type: "crossvalidation" as const,
            trainData,
            trainLabels,
            testData,
            testLabels,
            modelFnSource: modelFn,
          },
        });

        worker.on(
          "message",
          (msg: { success: boolean; result?: number; error?: string }) => {
            worker.terminate();
            if (msg.success) {
              resolve(msg.result!);
            } else {
              reject(new Error(msg.error ?? "CV worker failed"));
            }
          },
        );

        worker.on("error", (err) => {
          worker.terminate();
          reject(err);
        });
      }),
    );
  }

  const foldScores = await Promise.all(foldPromises);
  const m = mean(foldScores);
  const std = Math.sqrt(variance(foldScores));

  return { foldScores, mean: m, std };
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Compute Gelman-Rubin R-hat diagnostic for multiple chains.
 * Mirrors the implementation in mcmc.ts but inlined here to avoid
 * circular dependency issues in worker contexts.
 */
function computeGelmanRubin(chains: number[][]): number {
  const m = chains.length;
  const n = chains[0].length;

  const chainMeans = chains.map(mean);
  const overallMean = mean(chainMeans);

  // Between-chain variance
  let B = 0;
  for (const cm of chainMeans) B += (cm - overallMean) ** 2;
  B = (n * B) / (m - 1);

  // Within-chain variance
  let W = 0;
  for (const chain of chains) {
    W += variance(chain);
  }
  W /= m;

  const V = ((n - 1) / n) * W + (1 / n) * B;
  return Math.sqrt(V / W);
}
