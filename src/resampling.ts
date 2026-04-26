import { Dataset } from "./types";
import { mean } from "./utils/descriptive";
import { createRng } from "./utils/linalg";

/**
 * Result of a cross-validation procedure (k-fold or LOOCV).
 */
export interface CrossValidationResult {
  /** Error/score for each fold. */
  foldScores: number[];
  /** Mean score across folds. */
  meanScore: number;
  /** Standard deviation of fold scores. */
  stdScore: number;
  /** Number of folds. */
  nFolds: number;
}

/**
 * K-fold cross-validation for regression models.
 *
 * Splits data into k folds, trains on k-1 folds, and evaluates on the
 * held-out fold. Reports mean squared error by default.
 *
 * @param X - Feature matrix (n x p). All rows must have the same length.
 * @param y - Response variable
 * @param fitPredict - Function that takes (trainX, trainY, testX) and returns predictions
 * @param options - Configuration
 * @returns Cross-validation results including per-fold scores, mean, and standard deviation
 * @throws {Error} If X and y have different lengths
 * @throws {Error} If k is less than 2 or exceeds the number of observations
 * @throws {Error} If rows of X have inconsistent lengths
 *
 * @example
 * ```ts
 * const X = [[1], [2], [3], [4], [5], [6]];
 * const y = [2, 4, 6, 8, 10, 12];
 * const result = kFoldCV(X, y, (trX, trY, teX) => {
 *   // simple mean predictor
 *   const m = trY.reduce((a, b) => a + b, 0) / trY.length;
 *   return teX.map(() => m);
 * }, { k: 3 });
 * // result.meanScore — average MSE across 3 folds
 * ```
 */
export function kFoldCV(
  X: number[][],
  y: Dataset,
  fitPredict: (trainX: number[][], trainY: Dataset, testX: number[][]) => number[],
  options: {
    k?: number;
    scorer?: (actual: Dataset, predicted: Dataset) => number;
    seed?: number;
  } = {},
): CrossValidationResult {
  const n = X.length;
  if (n !== y.length) throw new Error(`Invalid parameters 'X', 'y': expected same length, received X.length=${n}, y.length=${y.length}`);

  // Validate consistent row lengths
  if (n > 0) {
    const p = X[0].length;
    for (let i = 1; i < n; i++) {
      if (X[i].length !== p) {
        throw new Error(
          `Invalid parameter 'X': Inconsistent row lengths, expected ${p} columns at row ${i}, received ${X[i].length}`,
        );
      }
    }
  }

  const k = options.k ?? 5;
  if (k < 2) throw new Error(`Invalid parameter 'k': expected at least 2, received ${k}`);
  if (k > n) throw new Error(`Invalid parameter 'k': cannot exceed ${n} observations, received ${k}`);

  const scorer = options.scorer ?? mse;
  const rng = options.seed != null ? createRng(options.seed) : Math.random;

  // Shuffle indices
  const indices = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }

  // Create folds
  const foldSize = Math.floor(n / k);
  const foldScores: number[] = [];

  for (let fold = 0; fold < k; fold++) {
    const testStart = fold * foldSize;
    const testEnd = fold === k - 1 ? n : testStart + foldSize;
    const testIndices = indices.slice(testStart, testEnd);
    const trainIndices = [...indices.slice(0, testStart), ...indices.slice(testEnd)];

    const trainX = trainIndices.map((i) => X[i]);
    const trainY = trainIndices.map((i) => y[i]);
    const testX = testIndices.map((i) => X[i]);
    const testY = testIndices.map((i) => y[i]);

    const predicted = fitPredict(trainX, trainY, testX);
    foldScores.push(scorer(testY, predicted));
  }

  const meanScore = mean(foldScores);
  const stdScore = Math.sqrt(
    foldScores.reduce((s, v) => s + (v - meanScore) ** 2, 0) / (k - 1),
  );

  return { foldScores, meanScore, stdScore, nFolds: k };
}

/**
 * Leave-one-out cross-validation (LOOCV).
 *
 * Special case of k-fold CV where k = n. Each observation is used as
 * a test set exactly once.
 *
 * @param X - Feature matrix (n x p)
 * @param y - Response variable
 * @param fitPredict - Function that takes (trainX, trainY, testX) and returns predictions
 * @param scorer - Scoring function (default: MSE)
 * @returns Cross-validation results with n folds
 * @throws {Error} If X and y have different lengths
 */
export function loocv(
  X: number[][],
  y: Dataset,
  fitPredict: (trainX: number[][], trainY: Dataset, testX: number[][]) => number[],
  scorer: (actual: Dataset, predicted: Dataset) => number = mse,
): CrossValidationResult {
  return kFoldCV(X, y, fitPredict, { k: X.length, scorer });
}

/**
 * Jackknife estimation of bias and standard error.
 *
 * Resamples by systematically leaving out one observation at a time.
 * Bias is estimated as (n-1) * (jackknife_mean - full_estimate).
 * Standard error uses the leave-one-out variance formula:
 * SE = sqrt(((n-1)/n) * sum((theta_i - theta_bar)^2)).
 * Pseudo-values are computed as: pv_i = n * theta_all - (n-1) * theta_{-i}.
 *
 * @param data - Input dataset
 * @param statistic - Function computing the statistic of interest
 * @returns Object containing the full-sample estimate, bias, standard error, and pseudo-values
 * @throws {Error} If data has fewer than 2 observations
 *
 * @example
 * ```ts
 * const data = [1, 2, 3, 4, 5];
 * const result = jackknife(data, (s) => s.reduce((a, b) => a + b) / s.length);
 * // result.estimate — the mean of the full sample
 * // result.bias — estimated bias of the statistic
 * // result.standardError — jackknife standard error
 * ```
 */
export function jackknife(
  data: Dataset,
  statistic: (sample: Dataset) => number,
): { estimate: number; bias: number; standardError: number; pseudoValues: number[] } {
  if (data.length < 2) throw new Error(`Invalid parameter 'data': expected at least 2 observations, received ${data.length}`);

  const n = data.length;
  const fullEstimate = statistic(data);

  const jackValues = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    const sample = [...data.slice(0, i), ...data.slice(i + 1)];
    jackValues[i] = statistic(sample);
  }

  const jackMean = mean(jackValues);

  // Pseudo-values
  const pseudoValues = jackValues.map((jv) => n * fullEstimate - (n - 1) * jv);

  // Bias: (n-1) * (jackknife_mean - full_estimate)
  const bias = (n - 1) * (jackMean - fullEstimate);

  // Standard error
  const se = Math.sqrt(
    ((n - 1) / n) * jackValues.reduce((s, v) => s + (v - jackMean) ** 2, 0),
  );

  return {
    estimate: fullEstimate,
    bias,
    standardError: se,
    pseudoValues,
  };
}

/**
 * Stratified random sampling.
 *
 * Samples from data while maintaining the proportion of each stratum.
 * If rounding causes the total allocated samples to exceed `sampleSize`,
 * excess samples are trimmed from the largest stratum.
 *
 * @param data - Dataset values
 * @param strata - Stratum label for each observation (integer-coded)
 * @param sampleSize - Total number of samples to draw
 * @param seed - Optional random seed for reproducibility
 * @returns Object containing the sampled values and their original indices
 * @throws {Error} If data and strata have different lengths
 * @throws {Error} If sampleSize is not between 1 and the dataset size
 *
 * @example
 * ```ts
 * const data = [10, 20, 30, 40, 50, 60];
 * const strata = [0, 0, 0, 1, 1, 1];
 * const result = stratifiedSample(data, strata, 4, 42);
 * // result.sample — 4 values proportionally drawn from each stratum
 * // result.indices — original indices of the drawn values
 * ```
 */
export function stratifiedSample(
  data: Dataset,
  strata: number[],
  sampleSize: number,
  seed?: number,
): { sample: number[]; indices: number[] } {
  if (data.length !== strata.length) {
    throw new Error(`Invalid parameters 'data', 'strata': expected same length, received data.length=${data.length}, strata.length=${strata.length}`);
  }
  if (sampleSize <= 0 || sampleSize > data.length) {
    throw new Error(`Invalid parameter 'sampleSize': expected a value between 1 and ${data.length}, received ${sampleSize}`);
  }

  const rng = seed != null ? createRng(seed) : Math.random;

  // Group by stratum
  const groups = new Map<number, number[]>();
  for (let i = 0; i < data.length; i++) {
    const s = strata[i];
    if (!groups.has(s)) groups.set(s, []);
    groups.get(s)!.push(i);
  }

  const n = data.length;

  // Compute per-group sample sizes via rounding, then fix any overshoot
  const groupEntries = [...groups.entries()];
  const groupSizes = groupEntries.map(([, indices]) =>
    Math.round((indices.length / n) * sampleSize),
  );

  let totalAllocated = groupSizes.reduce((a, b) => a + b, 0);
  // Trim excess from the largest groups (by allocated size) until we match sampleSize
  while (totalAllocated > sampleSize) {
    let maxIdx = 0;
    for (let i = 1; i < groupSizes.length; i++) {
      if (groupSizes[i] > groupSizes[maxIdx]) maxIdx = i;
    }
    groupSizes[maxIdx]--;
    totalAllocated--;
  }

  const sample: number[] = [];
  const resultIndices: number[] = [];

  for (let g = 0; g < groupEntries.length; g++) {
    const [, groupIndices] = groupEntries[g];
    const groupSampleSize = groupSizes[g];
    // Shuffle group indices
    const shuffled = [...groupIndices];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    for (let i = 0; i < Math.min(groupSampleSize, shuffled.length); i++) {
      resultIndices.push(shuffled[i]);
      sample.push(data[shuffled[i]]);
    }
  }

  return { sample, indices: resultIndices };
}

// ── Scoring Functions ───────────────────────────────────────────────────

/**
 * Mean Squared Error.
 *
 * @param actual - Actual values
 * @param predicted - Predicted values
 * @returns The mean squared error between actual and predicted
 * @throws {Error} If arrays have different lengths
 */
export function mse(actual: Dataset, predicted: Dataset): number {
  if (actual.length !== predicted.length) throw new Error(`Invalid parameters 'actual', 'predicted': expected same length, received actual.length=${actual.length}, predicted.length=${predicted.length}`);
  let sum = 0;
  for (let i = 0; i < actual.length; i++) sum += (actual[i] - predicted[i]) ** 2;
  return sum / actual.length;
}

/**
 * Root Mean Squared Error.
 *
 * @param actual - Actual values
 * @param predicted - Predicted values
 * @returns The root mean squared error between actual and predicted
 * @throws {Error} If arrays have different lengths
 */
export function rmse(actual: Dataset, predicted: Dataset): number {
  return Math.sqrt(mse(actual, predicted));
}

/**
 * Mean Absolute Error.
 *
 * @param actual - Actual values
 * @param predicted - Predicted values
 * @returns The mean absolute error between actual and predicted
 * @throws {Error} If arrays have different lengths
 */
export function mae(actual: Dataset, predicted: Dataset): number {
  if (actual.length !== predicted.length) throw new Error(`Invalid parameters 'actual', 'predicted': expected same length, received actual.length=${actual.length}, predicted.length=${predicted.length}`);
  let sum = 0;
  for (let i = 0; i < actual.length; i++) sum += Math.abs(actual[i] - predicted[i]);
  return sum / actual.length;
}

/**
 * R² (coefficient of determination).
 *
 * @param actual - Actual values
 * @param predicted - Predicted values
 * @returns The R² score (1.0 for perfect prediction, 0.0 for mean-level prediction)
 * @throws {Error} If arrays have different lengths
 */
export function r2Score(actual: Dataset, predicted: Dataset): number {
  if (actual.length !== predicted.length) throw new Error(`Invalid parameters 'actual', 'predicted': expected same length, received actual.length=${actual.length}, predicted.length=${predicted.length}`);
  const yMean = mean(actual);
  let ssRes = 0;
  let ssTot = 0;
  for (let i = 0; i < actual.length; i++) {
    ssRes += (actual[i] - predicted[i]) ** 2;
    ssTot += (actual[i] - yMean) ** 2;
  }
  return 1 - ssRes / ssTot;
}

// ── Advanced Resampling Methods ─────────────────────────────────────────

/**
 * Result of a bootstrap procedure.
 */
export interface BootstrapResult {
  /** Point estimate from the original data. */
  estimate: number;
  /** Standard error of the bootstrap distribution. */
  standardError: number;
  /** Confidence interval computed from bootstrap percentiles. */
  confidenceInterval: { lower: number; upper: number; level: number };
  /** Full bootstrap distribution of the statistic. */
  bootstrapDistribution: number[];
  /** Number of bootstrap replicates. */
  nBootstrap: number;
}

/**
 * Result of time-series cross-validation.
 */
export interface TimeSeriesCVResult {
  /** Score for each evaluation window. */
  scores: number[];
  /** Mean score across all windows. */
  meanScore: number;
  /** Standard deviation of scores. */
  stdScore: number;
  /** Train/test window boundaries for each split. */
  windows: { trainStart: number; trainEnd: number; testStart: number; testEnd: number }[];
}

/**
 * Result of nested cross-validation.
 */
export interface NestedCVResult {
  /** Outer fold scores. */
  outerScores: number[];
  /** Mean of outer scores. */
  meanScore: number;
  /** Standard deviation of outer scores. */
  stdScore: number;
  /** Best hyperparameters selected in each outer fold's inner CV. */
  bestParams: unknown[];
}

// ── Bootstrap helpers ───────────────────────────────────────────────────

/**
 * Build a BootstrapResult from an array of bootstrap replicate values and the
 * original-sample estimate.
 */
function buildBootstrapResult(
  estimate: number,
  distribution: number[],
  level = 0.95,
): BootstrapResult {
  const nBootstrap = distribution.length;
  const sorted = [...distribution].sort((a, b) => a - b);
  const alpha = 1 - level;
  const lowerIdx = Math.max(0, Math.floor((alpha / 2) * nBootstrap) - 1);
  const upperIdx = Math.min(nBootstrap - 1, Math.ceil((1 - alpha / 2) * nBootstrap) - 1);
  const bsMean = mean(distribution);
  const standardError = Math.sqrt(
    distribution.reduce((s, v) => s + (v - bsMean) ** 2, 0) / (nBootstrap - 1),
  );
  return {
    estimate,
    standardError,
    confidenceInterval: { lower: sorted[lowerIdx], upper: sorted[upperIdx], level },
    bootstrapDistribution: distribution,
    nBootstrap,
  };
}

// ── Block Bootstrap ─────────────────────────────────────────────────────

/**
 * Moving block bootstrap for time-series data.
 *
 * Resamples by drawing contiguous blocks of observations, preserving the
 * local dependence structure of the series.  When `circular` is true, the
 * series is treated as if it wraps around, so every index can start a full
 * block.
 *
 * @param data - Time-series observations (order matters)
 * @param statistic - Function computing the statistic of interest
 * @param options - Configuration
 * @returns Bootstrap result with estimate, SE, CI, and distribution
 * @throws {Error} If blockSize is less than 1 or exceeds data length
 * @throws {Error} If data is empty
 */
export function blockBootstrap(
  data: Dataset,
  statistic: (sample: Dataset) => number,
  options: {
    blockSize?: number;
    nBootstrap?: number;
    seed?: number;
    circular?: boolean;
  } = {},
): BootstrapResult {
  const n = data.length;
  if (n === 0) throw new Error(`Invalid parameter 'data': expected a non-empty array, received length 0`);

  const blockSize = options.blockSize ?? Math.max(1, Math.floor(Math.sqrt(n)));
  if (blockSize < 1 || blockSize > n) throw new Error(`Invalid parameter 'blockSize': expected a value between 1 and ${n}, received ${blockSize}`);

  const nBootstrap = options.nBootstrap ?? 1000;
  const circular = options.circular ?? false;
  const rng = options.seed != null ? createRng(options.seed) : Math.random;

  const estimate = statistic(data);
  const nBlocks = Math.ceil(n / blockSize);
  const maxStart = circular ? n : n - blockSize + 1;

  const distribution: number[] = [];
  for (let b = 0; b < nBootstrap; b++) {
    const sample: number[] = [];
    for (let blk = 0; blk < nBlocks && sample.length < n; blk++) {
      const start = Math.floor(rng() * maxStart);
      for (let j = 0; j < blockSize && sample.length < n; j++) {
        sample.push(data[(start + j) % n]);
      }
    }
    distribution.push(statistic(sample));
  }

  return buildBootstrapResult(estimate, distribution);
}

// ── Stationary Bootstrap ────────────────────────────────────────────────

/**
 * Stationary bootstrap for time-series data.
 *
 * Similar to block bootstrap but uses random block lengths drawn from a
 * geometric distribution with the given expected block size.  This yields a
 * strictly stationary resampling scheme.
 *
 * @param data - Time-series observations
 * @param statistic - Function computing the statistic of interest
 * @param options - Configuration
 * @returns Bootstrap result
 * @throws {Error} If data is empty
 * @throws {Error} If expectedBlockSize is less than 1
 */
export function stationaryBootstrap(
  data: Dataset,
  statistic: (sample: Dataset) => number,
  options: {
    expectedBlockSize?: number;
    nBootstrap?: number;
    seed?: number;
  } = {},
): BootstrapResult {
  const n = data.length;
  if (n === 0) throw new Error(`Invalid parameter 'data': expected a non-empty array, received length 0`);

  const expectedBlockSize = options.expectedBlockSize ?? Math.max(1, Math.floor(Math.sqrt(n)));
  if (expectedBlockSize < 1) throw new Error(`Invalid parameter 'expectedBlockSize': expected at least 1, received ${expectedBlockSize}`);

  const nBootstrap = options.nBootstrap ?? 1000;
  const rng = options.seed != null ? createRng(options.seed) : Math.random;
  const p = 1 / expectedBlockSize; // probability of starting a new block

  const estimate = statistic(data);
  const distribution: number[] = [];

  for (let b = 0; b < nBootstrap; b++) {
    const sample: number[] = [];
    let idx = Math.floor(rng() * n);
    for (let i = 0; i < n; i++) {
      sample.push(data[idx]);
      if (rng() < p) {
        // Start a new block at a random position
        idx = Math.floor(rng() * n);
      } else {
        idx = (idx + 1) % n;
      }
    }
    distribution.push(statistic(sample));
  }

  return buildBootstrapResult(estimate, distribution);
}

// ── Wild Bootstrap ──────────────────────────────────────────────────────

/**
 * Wild bootstrap for heteroscedastic regression residuals.
 *
 * Multiplies each residual by a random weight drawn from either the
 * Rademacher distribution (+1 / -1 with equal probability) or the Mammen
 * two-point distribution, then adds the result back to the fitted values.
 *
 * @param residuals - Regression residuals
 * @param fitted - Fitted (predicted) values
 * @param statistic - Function computing the statistic from a bootstrap sample of y*
 * @param options - Configuration
 * @returns Bootstrap result
 * @throws {Error} If residuals and fitted have different lengths
 * @throws {Error} If data is empty
 */
export function wildBootstrap(
  residuals: Dataset,
  fitted: Dataset,
  statistic: (sample: Dataset) => number,
  options: {
    nBootstrap?: number;
    seed?: number;
    distribution?: "rademacher" | "mammen";
  } = {},
): BootstrapResult {
  const n = residuals.length;
  if (n !== fitted.length) throw new Error(`Invalid parameters 'residuals', 'fitted': expected same length, received residuals.length=${n}, fitted.length=${fitted.length}`);
  if (n === 0) throw new Error(`Invalid parameter 'residuals': expected a non-empty array, received length 0`);

  const nBootstrap = options.nBootstrap ?? 1000;
  const dist = options.distribution ?? "rademacher";
  const rng = options.seed != null ? createRng(options.seed) : Math.random;

  // Original y = fitted + residuals
  const original: number[] = new Array(n);
  for (let i = 0; i < n; i++) original[i] = fitted[i] + residuals[i];
  const estimate = statistic(original);

  // Mammen two-point distribution constants
  const sqrt5 = Math.sqrt(5);
  const mammenP = (sqrt5 + 1) / (2 * sqrt5); // P(w = -(sqrt(5)-1)/2)
  const mammenPos = (sqrt5 + 1) / 2;
  const mammenNeg = -(sqrt5 - 1) / 2;

  const distribution: number[] = [];
  for (let b = 0; b < nBootstrap; b++) {
    const yStar: number[] = new Array(n);
    for (let i = 0; i < n; i++) {
      let w: number;
      if (dist === "rademacher") {
        w = rng() < 0.5 ? -1 : 1;
      } else {
        // Mammen distribution
        w = rng() < mammenP ? mammenNeg : mammenPos;
      }
      yStar[i] = fitted[i] + residuals[i] * w;
    }
    distribution.push(statistic(yStar));
  }

  return buildBootstrapResult(estimate, distribution);
}

// ── Bayesian Bootstrap ──────────────────────────────────────────────────

/**
 * Bayesian bootstrap via Dirichlet-weighted resampling.
 *
 * Instead of resampling with equal integer weights, draws continuous
 * Dirichlet(1,…,1) weights for the observations.  The statistic function
 * receives the original data, so statistics that depend on weights should
 * use the weighted variant.  Here, a weighted mean is implicitly created by
 * building resampled datasets where each data point is replicated
 * proportionally to its Dirichlet weight (multinomial resampling from the
 * Dirichlet probabilities).
 *
 * @param data - Input dataset
 * @param statistic - Function computing the statistic of interest
 * @param options - Configuration
 * @returns Bootstrap result
 * @throws {Error} If data is empty
 */
export function bayesianBootstrap(
  data: Dataset,
  statistic: (sample: Dataset) => number,
  options: {
    nBootstrap?: number;
    seed?: number;
  } = {},
): BootstrapResult {
  const n = data.length;
  if (n === 0) throw new Error(`Invalid parameter 'data': expected a non-empty array, received length 0`);

  const nBootstrap = options.nBootstrap ?? 1000;
  const rng = options.seed != null ? createRng(options.seed) : Math.random;

  const estimate = statistic(data);
  const distribution: number[] = [];

  for (let b = 0; b < nBootstrap; b++) {
    // Sample Dirichlet(1,...,1) by drawing Exponential(1) values and normalising
    const weights = new Array<number>(n);
    let wSum = 0;
    for (let i = 0; i < n; i++) {
      weights[i] = -Math.log(rng());
      wSum += weights[i];
    }
    for (let i = 0; i < n; i++) weights[i] /= wSum;

    // Multinomial resampling according to Dirichlet weights
    const sample: number[] = new Array(n);
    for (let i = 0; i < n; i++) {
      let u = rng();
      let cumulative = 0;
      let chosen = n - 1;
      for (let j = 0; j < n; j++) {
        cumulative += weights[j];
        if (u <= cumulative) {
          chosen = j;
          break;
        }
      }
      sample[i] = data[chosen];
    }

    distribution.push(statistic(sample));
  }

  return buildBootstrapResult(estimate, distribution);
}

// ── Time-Series Cross-Validation ────────────────────────────────────────

/**
 * Time-series cross-validation with expanding or rolling windows.
 *
 * Respects temporal ordering: the training set always precedes the test set.
 * With `maxTrainSize` set to `null` (the default) the training window expands
 * over time; with a finite value the window rolls forward.
 *
 * @param data - Time-series observations
 * @param fitPredict - Function (trainData, testData) => predictions for the test window
 * @param options - Configuration
 * @returns Time-series CV results including per-window scores and window boundaries
 * @throws {Error} If minTrainSize + horizon exceeds data length
 */
export function timeSeriesCV(
  data: Dataset,
  fitPredict: (train: Dataset, test: Dataset) => number[],
  options: {
    minTrainSize?: number;
    step?: number;
    maxTrainSize?: number | null;
    horizon?: number;
    scorer?: (actual: Dataset, predicted: Dataset) => number;
  } = {},
): TimeSeriesCVResult {
  const n = data.length;
  const minTrainSize = options.minTrainSize ?? Math.max(1, Math.floor(n / 3));
  const step = options.step ?? 1;
  const maxTrainSize = options.maxTrainSize === undefined ? null : options.maxTrainSize;
  const horizon = options.horizon ?? 1;
  const scorer = options.scorer ?? mse;

  if (minTrainSize + horizon > n) {
    throw new Error(`Invalid parameters 'minTrainSize', 'horizon': expected minTrainSize + horizon <= ${n}, received ${minTrainSize} + ${horizon} = ${minTrainSize + horizon}`);
  }

  const scores: number[] = [];
  const windows: { trainStart: number; trainEnd: number; testStart: number; testEnd: number }[] = [];

  for (let trainEnd = minTrainSize; trainEnd + horizon <= n; trainEnd += step) {
    const trainStart = maxTrainSize != null ? Math.max(0, trainEnd - maxTrainSize) : 0;
    const testStart = trainEnd;
    const testEnd = trainEnd + horizon;

    const trainData = data.slice(trainStart, trainEnd);
    const testData = data.slice(testStart, testEnd);

    const predicted = fitPredict(trainData, testData);
    scores.push(scorer(testData, predicted));
    windows.push({ trainStart, trainEnd, testStart, testEnd });
  }

  const meanScore = scores.length > 0 ? mean(scores) : 0;
  const stdScore =
    scores.length > 1
      ? Math.sqrt(scores.reduce((s, v) => s + (v - meanScore) ** 2, 0) / (scores.length - 1))
      : 0;

  return { scores, meanScore, stdScore, windows };
}

// ── Nested Cross-Validation ─────────────────────────────────────────────

/**
 * Nested cross-validation for simultaneous model evaluation and
 * hyperparameter selection.
 *
 * The outer loop evaluates generalisation performance while the inner loop
 * selects the best hyperparameters for each outer training fold.
 *
 * @param X - Feature matrix (n x p)
 * @param y - Response variable
 * @param fitPredictFactory - Function (params) => fitPredict, where fitPredict
 *   takes (trainX, trainY, testX) and returns predictions.  The factory is
 *   called once per candidate hyperparameter set.
 * @param options - Configuration including the parameter grid
 * @returns Nested CV results with outer scores and best parameters per fold
 * @throws {Error} If X and y have different lengths
 * @throws {Error} If paramGrid is empty
 */
export function nestedCV(
  X: number[][],
  y: Dataset,
  fitPredictFactory: (
    params: unknown,
  ) => (trainX: number[][], trainY: Dataset, testX: number[][]) => number[],
  options: {
    outerK?: number;
    innerK?: number;
    seed?: number;
    paramGrid: unknown[];
    scorer?: (actual: Dataset, predicted: Dataset) => number;
  },
): NestedCVResult {
  const n = X.length;
  if (n !== y.length) throw new Error(`Invalid parameters 'X', 'y': expected same length, received X.length=${n}, y.length=${y.length}`);
  if (!options.paramGrid || options.paramGrid.length === 0) {
    throw new Error(`Invalid parameter 'paramGrid': expected a non-empty array, received ${options.paramGrid ? 'length 0' : 'undefined'}`);
  }

  const outerK = options.outerK ?? 5;
  const innerK = options.innerK ?? 3;
  const scorer = options.scorer ?? mse;
  const rng = options.seed != null ? createRng(options.seed) : Math.random;

  // Shuffle indices once
  const indices = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }

  const outerFoldSize = Math.floor(n / outerK);
  const outerScores: number[] = [];
  const bestParams: unknown[] = [];

  for (let outerFold = 0; outerFold < outerK; outerFold++) {
    const outerTestStart = outerFold * outerFoldSize;
    const outerTestEnd = outerFold === outerK - 1 ? n : outerTestStart + outerFoldSize;
    const outerTestIdx = indices.slice(outerTestStart, outerTestEnd);
    const outerTrainIdx = [
      ...indices.slice(0, outerTestStart),
      ...indices.slice(outerTestEnd),
    ];

    const outerTrainX = outerTrainIdx.map((i) => X[i]);
    const outerTrainY = outerTrainIdx.map((i) => y[i]);
    const outerTestX = outerTestIdx.map((i) => X[i]);
    const outerTestY = outerTestIdx.map((i) => y[i]);

    // Inner CV: select best params
    let bestInnerScore = Infinity;
    let bestParam: unknown = options.paramGrid[0];

    for (const params of options.paramGrid) {
      const innerResult = kFoldCV(outerTrainX, outerTrainY, fitPredictFactory(params), {
        k: Math.min(innerK, outerTrainIdx.length),
        scorer,
      });
      if (innerResult.meanScore < bestInnerScore) {
        bestInnerScore = innerResult.meanScore;
        bestParam = params;
      }
    }

    bestParams.push(bestParam);

    // Evaluate on outer test fold with best params
    const fitPredict = fitPredictFactory(bestParam);
    const predicted = fitPredict(outerTrainX, outerTrainY, outerTestX);
    outerScores.push(scorer(outerTestY, predicted));
  }

  const meanScore = mean(outerScores);
  const stdScore = Math.sqrt(
    outerScores.reduce((s, v) => s + (v - meanScore) ** 2, 0) / (outerK - 1),
  );

  return { outerScores, meanScore, stdScore, bestParams };
}

// ── Monte Carlo Cross-Validation ────────────────────────────────────────

/**
 * Monte Carlo cross-validation (repeated random sub-sampling).
 *
 * Repeatedly splits data into random train/test sets and evaluates the model.
 * Unlike k-fold CV, splits are independent and observations may appear in
 * multiple test sets.
 *
 * @param X - Feature matrix (n x p)
 * @param y - Response variable
 * @param fitPredict - Function (trainX, trainY, testX) => predictions
 * @param options - Configuration
 * @returns Cross-validation results
 * @throws {Error} If X and y have different lengths
 * @throws {Error} If testFraction is not in (0, 1)
 */
export function monteCarloCV(
  X: number[][],
  y: Dataset,
  fitPredict: (trainX: number[][], trainY: Dataset, testX: number[][]) => number[],
  options: {
    nSplits?: number;
    testFraction?: number;
    seed?: number;
    scorer?: (actual: Dataset, predicted: Dataset) => number;
  } = {},
): CrossValidationResult {
  const n = X.length;
  if (n !== y.length) throw new Error(`Invalid parameters 'X', 'y': expected same length, received X.length=${n}, y.length=${y.length}`);

  const nSplits = options.nSplits ?? 100;
  const testFraction = options.testFraction ?? 0.2;
  if (testFraction <= 0 || testFraction >= 1) {
    throw new Error(`Invalid parameter 'testFraction': expected a value in (0, 1), received ${testFraction}`);
  }

  const scorer = options.scorer ?? mse;
  const rng = options.seed != null ? createRng(options.seed) : Math.random;
  const testSize = Math.max(1, Math.round(n * testFraction));

  const foldScores: number[] = [];

  for (let s = 0; s < nSplits; s++) {
    // Shuffle indices
    const indices = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }

    const testIndices = indices.slice(0, testSize);
    const trainIndices = indices.slice(testSize);

    const trainX = trainIndices.map((i) => X[i]);
    const trainY = trainIndices.map((i) => y[i]);
    const testX = testIndices.map((i) => X[i]);
    const testY = testIndices.map((i) => y[i]);

    const predicted = fitPredict(trainX, trainY, testX);
    foldScores.push(scorer(testY, predicted));
  }

  const meanScore = mean(foldScores);
  const stdScore = Math.sqrt(
    foldScores.reduce((s, v) => s + (v - meanScore) ** 2, 0) / (nSplits - 1),
  );

  return { foldScores, meanScore, stdScore, nFolds: nSplits };
}
