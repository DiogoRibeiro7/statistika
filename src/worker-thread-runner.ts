/**
 * Worker thread runner script.
 *
 * This file is executed inside each worker thread. It receives task
 * descriptions via the parent port, performs the requested computation,
 * and posts the result back.
 *
 * Because functions cannot be serialized across threads, each built-in
 * task type imports the required modules directly and operates on
 * configuration objects rather than function references.
 */

import { parentPort, workerData } from "node:worker_threads";

// ---------------------------------------------------------------------------
// Minimal seeded PRNG (xoshiro128** — same quality as the main SeededRng)
// ---------------------------------------------------------------------------

class WorkerRng {
  private s: Uint32Array;

  constructor(seed: number) {
    this.s = new Uint32Array(4);
    // SplitMix32 to seed the state
    let z = (seed | 0) >>> 0;
    for (let i = 0; i < 4; i++) {
      z = (z + 0x9e3779b9) >>> 0;
      let t = z ^ (z >>> 16);
      t = Math.imul(t, 0x21f0aaad);
      t = t ^ (t >>> 15);
      t = Math.imul(t, 0x735a2d97);
      t = t ^ (t >>> 15);
      this.s[i] = t >>> 0;
    }
  }

  next(): number {
    const s = this.s;
    const result = Math.imul(rotl(Math.imul(s[1], 5), 7), 9) >>> 0;
    const t = (s[1] << 9) >>> 0;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 11) >>> 0;
    return result / 0x100000000;
  }

  nextNormal(mu: number, sigma: number): number {
    // Box-Muller transform
    let u: number, v: number, s: number;
    do {
      u = 2 * this.next() - 1;
      v = 2 * this.next() - 1;
      s = u * u + v * v;
    } while (s >= 1 || s === 0);
    const mul = Math.sqrt((-2 * Math.log(s)) / s);
    return mu + sigma * u * mul;
  }
}

function rotl(x: number, k: number): number {
  return ((x << k) | (x >>> (32 - k))) >>> 0;
}

// ---------------------------------------------------------------------------
// Task handlers
// ---------------------------------------------------------------------------

interface MCMCTaskData {
  type: "mcmc";
  iterations: number;
  burnIn: number;
  proposalStd: number;
  seed: number;
  logDensitySource: string;
}

interface BootstrapTaskData {
  type: "bootstrap";
  data: number[];
  nResamples: number;
  seed: number;
  statisticSource: string;
}

interface CrossValidationTaskData {
  type: "crossvalidation";
  trainData: number[][];
  trainLabels: number[];
  testData: number[][];
  testLabels: number[];
  modelFnSource: string;
}

interface GenericTaskData {
  type: "generic";
  taskFile: string;
  input: unknown;
}

type TaskData =
  | MCMCTaskData
  | BootstrapTaskData
  | CrossValidationTaskData
  | GenericTaskData;

function runMCMC(task: MCMCTaskData): number[] {
  const rng = new WorkerRng(task.seed);
  // eslint-disable-next-line @typescript-eslint/no-implied-eval
  const logDensity = new Function("x", `return (${task.logDensitySource})(x)`) as (
    x: number,
  ) => number;

  let current = 0;
  let currentLD = logDensity(current);
  if (Number.isNaN(currentLD)) currentLD = -Infinity;

  const chain: number[] = [];

  for (let i = 0; i < task.iterations; i++) {
    const proposal = current + rng.nextNormal(0, task.proposalStd);
    let proposalLD = logDensity(proposal);
    if (Number.isNaN(proposalLD)) proposalLD = -Infinity;

    const logAlpha = proposalLD - currentLD;
    if (Math.log(rng.next()) < logAlpha) {
      current = proposal;
      currentLD = proposalLD;
    }

    if (i >= task.burnIn) {
      chain.push(current);
    }
  }

  return chain;
}

function runBootstrap(task: BootstrapTaskData): number[] {
  const rng = new WorkerRng(task.seed);
  // eslint-disable-next-line @typescript-eslint/no-implied-eval
  const statistic = new Function("data", `return (${task.statisticSource})(data)`) as (
    data: number[],
  ) => number;

  const n = task.data.length;
  const results: number[] = [];

  for (let i = 0; i < task.nResamples; i++) {
    const sample: number[] = new Array(n);
    for (let j = 0; j < n; j++) {
      sample[j] = task.data[Math.floor(rng.next() * n)];
    }
    results.push(statistic(sample));
  }

  return results;
}

function runCrossValidation(task: CrossValidationTaskData): number {
  // eslint-disable-next-line @typescript-eslint/no-implied-eval
  const modelFn = new Function(
    "trainData",
    "trainLabels",
    "testData",
    "testLabels",
    `return (${task.modelFnSource})(trainData, trainLabels, testData, testLabels)`,
  ) as (
    trainData: number[][],
    trainLabels: number[],
    testData: number[][],
    testLabels: number[],
  ) => number;

  return modelFn(task.trainData, task.trainLabels, task.testData, task.testLabels);
}

async function runGenericTask(task: GenericTaskData): Promise<unknown> {
  // Dynamic import of the task file
  const mod = await import(task.taskFile);
  if (typeof mod.default === "function") {
    return mod.default(task.input);
  }
  if (typeof mod.run === "function") {
    return mod.run(task.input);
  }
  throw new Error(
    `Task module must export a default function or a run() function: received ${task.taskFile}`,
  );
}

// ---------------------------------------------------------------------------
// Message handling
// ---------------------------------------------------------------------------

if (parentPort) {
  // If workerData is set, process it directly (pool mode)
  if (workerData) {
    handleTask(workerData as TaskData)
      .then((result) => parentPort!.postMessage({ success: true, result }))
      .catch((err) =>
        parentPort!.postMessage({
          success: false,
          error: err instanceof Error ? err.message : String(err),
        }),
      );
  } else {
    // Listen for messages (reusable worker mode)
    parentPort.on("message", (task: TaskData) => {
      handleTask(task)
        .then((result) => parentPort!.postMessage({ success: true, result }))
        .catch((err) =>
          parentPort!.postMessage({
            success: false,
            error: err instanceof Error ? err.message : String(err),
          }),
        );
    });
  }
}

async function handleTask(task: TaskData): Promise<unknown> {
  switch (task.type) {
    case "mcmc":
      return runMCMC(task);
    case "bootstrap":
      return runBootstrap(task);
    case "crossvalidation":
      return runCrossValidation(task);
    case "generic":
      return runGenericTask(task);
    default:
      throw new Error(`Unknown task type: ${(task as TaskData).type}`);
  }
}
