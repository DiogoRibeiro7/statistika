/**
 * Changepoint detection algorithms: CUSUM, PELT, Binary Segmentation, and
 * Bayesian Online Changepoint Detection (BOCPD).
 */

import { mean, variance } from "./utils/descriptive";

// ---- Interfaces ----

/**
 * Result of a CUSUM (cumulative sum) test for a change in mean.
 */
export interface CUSUMResult {
  /** Test statistic (maximum absolute CUSUM value) */
  statistic: number;
  /** Estimated changepoint location (index) */
  changepoint: number;
  /** Approximate p-value */
  pValue: number;
  /** Whether the null hypothesis of no changepoint is rejected */
  reject: boolean;
  /** CUSUM path values for each index */
  cusumValues: number[];
}

/**
 * Result of a changepoint detection algorithm (PELT or Binary Segmentation).
 */
export interface ChangepointResult {
  /** Indices of detected changepoints */
  changepoints: number[];
  /** Segments defined by the changepoints, each with summary statistics */
  segments: { start: number; end: number; mean: number; variance: number }[];
  /** Total cost across all segments */
  cost: number;
  /** Description of the penalty used */
  penalty: string;
}

/**
 * Result of Bayesian Online Changepoint Detection.
 */
export interface BOCPDResult {
  /** Indices where changepoints were detected (run length dropped to 0) */
  changepoints: number[];
  /** Run length probability distribution at each time step; runLengthProbabilities[t][r] */
  runLengthProbabilities: number[][];
  /** Most probable run length at each time step */
  maxRunLengthProb: number[];
}

/**
 * Cost function type for changepoint detection.
 * - "mean": detects changes in mean (normal log-likelihood cost).
 * - "variance": detects changes in variance.
 * - "meanvar": detects changes in both mean and variance.
 */
export type CostFunction = "mean" | "variance" | "meanvar";

/**
 * Options for the PELT algorithm.
 */
export interface PELTOptions {
  /** Cost function to use (default: "meanvar") */
  costFunction?: CostFunction;
  /** Penalty type: "BIC", "mBIC", or a numeric value for manual penalty (default: "BIC") */
  penalty?: "BIC" | "mBIC" | number;
  /** Minimum segment length (default: 2) */
  minSegmentLength?: number;
}

/**
 * Options for Binary Segmentation.
 */
export interface BinarySegmentationOptions {
  /** Cost function to use (default: "meanvar") */
  costFunction?: CostFunction;
  /** Penalty type: "BIC", "mBIC", or a numeric value for manual penalty (default: "BIC") */
  penalty?: "BIC" | "mBIC" | number;
  /** Minimum segment length (default: 2) */
  minSegmentLength?: number;
  /** Maximum number of changepoints to detect (default: Infinity) */
  maxChangepoints?: number;
}

/**
 * Options for Bayesian Online Changepoint Detection.
 */
export interface BOCPDOptions {
  /** Hazard rate lambda: expected run length is 1/lambda (default: 1/250) */
  hazardLambda?: number;
  /** Prior mean for the Normal-Gamma conjugate prior (default: 0) */
  priorMu?: number;
  /** Prior precision scaling factor kappa (default: 1) */
  priorKappa?: number;
  /** Prior shape parameter alpha (default: 1) */
  priorAlpha?: number;
  /** Prior rate parameter beta (default: 1) */
  priorBeta?: number;
  /** Threshold on run length 0 probability to declare a changepoint (default: 0.5) */
  threshold?: number;
}

// ---- Cost Functions ----

/**
 * Computes the segment cost for a change-in-mean model.
 *
 * Uses the normal log-likelihood cost: n * log(variance) where variance is
 * the maximum likelihood (population) variance of the segment. For segments
 * with zero variance, returns 0.
 *
 * @param data - The full data array.
 * @param start - Start index (inclusive).
 * @param end - End index (exclusive).
 * @returns The cost for the segment.
 * @throws {Error} If the segment has fewer than 1 element.
 *
 * @example
 * ```ts
 * costMean([1, 1, 5, 5], 0, 4); // cost for the whole segment
 * costMean([1, 1, 5, 5], 0, 2); // cost for [1, 1]
 * ```
 */
export function costMean(data: number[], start: number, end: number): number {
  const n = end - start;
  if (n < 1) throw new Error("Segment must have at least 1 element");
  if (n === 1) return 0;

  const segment = data.slice(start, end);
  const m = mean(segment);
  let sumSq = 0;
  for (let i = 0; i < n; i++) {
    sumSq += (segment[i] - m) ** 2;
  }
  const v = sumSq / n;
  if (v <= 0) return 0;
  return n * Math.log(v);
}

/**
 * Computes the segment cost for a change-in-variance model.
 *
 * Assumes a known overall mean. The cost is n * log(segment variance)
 * where variance is computed around the overall data mean.
 *
 * @param data - The full data array.
 * @param start - Start index (inclusive).
 * @param end - End index (exclusive).
 * @param overallMean - The overall mean of the full dataset.
 * @returns The cost for the segment.
 * @throws {Error} If the segment has fewer than 1 element.
 *
 * @example
 * ```ts
 * costVariance([1, 1, 5, 5], 0, 2, 3); // cost around overall mean 3
 * ```
 */
export function costVariance(
  data: number[],
  start: number,
  end: number,
  overallMean?: number,
): number {
  const n = end - start;
  if (n < 1) throw new Error("Segment must have at least 1 element");
  if (n === 1) return 0;

  const mu = overallMean ?? mean(data);
  let sumSq = 0;
  for (let i = start; i < end; i++) {
    sumSq += (data[i] - mu) ** 2;
  }
  const v = sumSq / n;
  if (v <= 0) return 0;
  return n * Math.log(v);
}

/**
 * Computes the segment cost for a change in both mean and variance.
 *
 * Uses the full normal log-likelihood cost: n * log(2 * pi * variance) + n,
 * simplified to n * (log(variance) + 1 + log(2*pi)).
 *
 * @param data - The full data array.
 * @param start - Start index (inclusive).
 * @param end - End index (exclusive).
 * @returns The cost for the segment.
 * @throws {Error} If the segment has fewer than 1 element.
 *
 * @example
 * ```ts
 * costMeanVar([1, 1, 5, 5], 0, 4);
 * ```
 */
export function costMeanVar(data: number[], start: number, end: number): number {
  const n = end - start;
  if (n < 1) throw new Error("Segment must have at least 1 element");
  if (n === 1) return 0;

  const segment = data.slice(start, end);
  const m = mean(segment);
  let sumSq = 0;
  for (let i = 0; i < n; i++) {
    sumSq += (segment[i] - m) ** 2;
  }
  const v = sumSq / n;
  if (v <= 0) return 0;
  return n * (Math.log(v) + 1 + Math.log(2 * Math.PI));
}

// ---- Helper: select cost function ----

function getCostFn(
  type: CostFunction,
  data: number[],
): (start: number, end: number) => number {
  const overallMu = mean(data);
  switch (type) {
    case "mean":
      return (s, e) => costMean(data, s, e);
    case "variance":
      return (s, e) => costVariance(data, s, e, overallMu);
    case "meanvar":
      return (s, e) => costMeanVar(data, s, e);
    default:
      throw new Error(`Unknown cost function: ${type}`);
  }
}

// ---- Helper: compute penalty value ----

function computePenalty(
  penalty: "BIC" | "mBIC" | number,
  n: number,
  costFnType: CostFunction,
): number {
  const numParams = costFnType === "meanvar" ? 2 : 1;
  if (typeof penalty === "number") return penalty;
  if (penalty === "BIC") return numParams * Math.log(n);
  // mBIC: modified BIC with an additional log(n) term
  if (penalty === "mBIC") return numParams * Math.log(n) + 2 * Math.log(Math.log(n));
  throw new Error(`Unknown penalty type: ${penalty}`);
}

// ---- Helper: build segments from changepoints ----

function buildSegments(
  data: number[],
  changepoints: number[],
): { start: number; end: number; mean: number; variance: number }[] {
  const boundaries = [0, ...changepoints, data.length];
  const segments: { start: number; end: number; mean: number; variance: number }[] = [];
  for (let i = 0; i < boundaries.length - 1; i++) {
    const s = boundaries[i];
    const e = boundaries[i + 1];
    const segment = data.slice(s, e);
    const m = mean(segment);
    const v = segment.length >= 2 ? variance(segment, false) : 0;
    segments.push({ start: s, end: e, mean: m, variance: v });
  }
  return segments;
}

// ---- CUSUM Test ----

/**
 * Performs a CUSUM (Cumulative Sum) test for detecting a single change in mean.
 *
 * Computes the CUSUM statistic as the maximum absolute deviation of the
 * cumulative sum of centered residuals. The p-value is approximated using
 * the asymptotic distribution of the Kolmogorov-Smirnov statistic.
 *
 * @param data - Array of numeric values.
 * @param alpha - Significance level for the test (default: 0.05).
 * @returns A {@link CUSUMResult} with the test statistic, changepoint location,
 *   p-value, rejection decision, and the full CUSUM path.
 * @throws {Error} If the data array has fewer than 2 elements.
 * @throws {Error} If alpha is not in (0, 1).
 *
 * @example
 * ```ts
 * const data = [1, 1, 1, 1, 5, 5, 5, 5];
 * const result = cusumTest(data);
 * console.log(result.changepoint); // 4
 * console.log(result.reject);      // true
 * ```
 */
export function cusumTest(data: number[], alpha = 0.05): CUSUMResult {
  if (data.length < 2) {
    throw new Error("Data must have at least 2 elements");
  }
  if (alpha <= 0 || alpha >= 1) {
    throw new Error("Alpha must be between 0 and 1 (exclusive)");
  }

  const n = data.length;
  const mu = mean(data);

  // Compute CUSUM path: S_k = sum_{i=0}^{k-1} (x_i - mu)
  const cusumValues: number[] = new Array(n + 1);
  cusumValues[0] = 0;
  for (let i = 0; i < n; i++) {
    cusumValues[i + 1] = cusumValues[i] + (data[i] - mu);
  }

  // Normalize by sqrt(n) * sigma
  let sumSq = 0;
  for (let i = 0; i < n; i++) {
    sumSq += (data[i] - mu) ** 2;
  }
  const sigma = Math.sqrt(sumSq / n);

  let maxStat = 0;
  let changepoint = 0;
  const normalizedCusum: number[] = new Array(n + 1);

  for (let k = 0; k <= n; k++) {
    const normalized = sigma > 0 ? Math.abs(cusumValues[k]) / (sigma * Math.sqrt(n)) : 0;
    normalizedCusum[k] = normalized;
    if (normalized > maxStat) {
      maxStat = normalized;
      changepoint = k;
    }
  }

  // Approximate p-value using Kolmogorov-Smirnov asymptotic distribution
  // P(D > x) ~ 2 * sum_{k=1}^{inf} (-1)^{k+1} * exp(-2*k^2*x^2)
  const statistic = maxStat;
  let pValue = 0;
  for (let k = 1; k <= 100; k++) {
    pValue += (k % 2 === 1 ? 1 : -1) * Math.exp(-2 * k * k * statistic * statistic);
  }
  pValue = 2 * pValue;
  pValue = Math.max(0, Math.min(1, 1 - pValue));

  // Adjust changepoint: it's the index in the original data
  // The maximum CUSUM is at position k meaning the change happened before index k
  const cp = Math.max(0, Math.min(changepoint, n - 1));

  return {
    statistic,
    changepoint: cp,
    pValue,
    reject: pValue < alpha,
    cusumValues: normalizedCusum,
  };
}

// ---- PELT Algorithm ----

/**
 * Detects multiple changepoints using the PELT (Pruned Exact Linear Time) algorithm.
 *
 * PELT finds the optimal segmentation of the data by minimizing the total cost
 * plus a penalty for each changepoint. It achieves linear expected time complexity
 * by pruning candidate changepoints that cannot improve the optimal segmentation.
 *
 * @param data - Array of numeric values.
 * @param options - Configuration options for the algorithm.
 * @returns A {@link ChangepointResult} with detected changepoints, segments, cost,
 *   and penalty description.
 * @throws {Error} If the data array has fewer than 2 elements.
 * @throws {Error} If minSegmentLength is less than 2.
 *
 * @example
 * ```ts
 * const data = [1, 1, 1, 5, 5, 5, 1, 1, 1];
 * const result = pelt(data, { costFunction: "meanvar", penalty: "BIC" });
 * console.log(result.changepoints); // [3, 6]
 * ```
 */
export function pelt(data: number[], options: PELTOptions = {}): ChangepointResult {
  const {
    costFunction = "meanvar",
    penalty = "BIC",
    minSegmentLength = 2,
  } = options;

  const n = data.length;
  if (n < 2) {
    throw new Error("Data must have at least 2 elements");
  }
  if (minSegmentLength < 2) {
    throw new Error("Minimum segment length must be at least 2");
  }

  const cost = getCostFn(costFunction, data);
  const pen = computePenalty(penalty, n, costFunction);
  const penaltyLabel = typeof penalty === "number" ? `manual(${penalty})` : penalty;

  // F[t] = optimal cost for data[0..t)
  // lastChange[t] = last changepoint index for optimal segmentation of data[0..t)
  const F: number[] = new Array(n + 1).fill(Infinity);
  const lastChange: number[] = new Array(n + 1).fill(0);
  F[0] = -pen; // so that F[0] + cost(0, t) + pen = cost(0, t)

  // Set of candidate changepoints
  let candidates: number[] = [0];

  for (let t = minSegmentLength; t <= n; t++) {
    let bestCost = Infinity;
    let bestTau = 0;

    for (const tau of candidates) {
      if (t - tau < minSegmentLength) continue;
      const c = F[tau] + cost(tau, t) + pen;
      if (c < bestCost) {
        bestCost = c;
        bestTau = tau;
      }
    }

    F[t] = bestCost;
    lastChange[t] = bestTau;

    // Pruning: keep only candidates whose F[tau] + cost(tau, t) <= F[t]
    const prunedCandidates: number[] = [];
    for (const tau of candidates) {
      if (t - tau < minSegmentLength) {
        prunedCandidates.push(tau);
        continue;
      }
      if (F[tau] + cost(tau, t) <= F[t]) {
        prunedCandidates.push(tau);
      }
    }
    prunedCandidates.push(t);
    candidates = prunedCandidates;
  }

  // Backtrack to find changepoints
  const changepoints: number[] = [];
  let idx = n;
  while (idx > 0) {
    const cp = lastChange[idx];
    if (cp > 0) {
      changepoints.push(cp);
    }
    idx = cp;
  }
  changepoints.sort((a, b) => a - b);

  const segments = buildSegments(data, changepoints);
  let totalCost = 0;
  const boundaries = [0, ...changepoints, n];
  for (let i = 0; i < boundaries.length - 1; i++) {
    totalCost += cost(boundaries[i], boundaries[i + 1]);
  }

  return {
    changepoints,
    segments,
    cost: totalCost,
    penalty: penaltyLabel,
  };
}

// ---- Binary Segmentation ----

/**
 * Detects multiple changepoints using the Binary Segmentation algorithm.
 *
 * Recursively splits the data at the point that maximizes the reduction in
 * cost. Splitting stops when no further split reduces the cost by more than
 * the penalty, or when the maximum number of changepoints is reached.
 *
 * @param data - Array of numeric values.
 * @param options - Configuration options for the algorithm.
 * @returns A {@link ChangepointResult} with detected changepoints, segments, cost,
 *   and penalty description.
 * @throws {Error} If the data array has fewer than 2 elements.
 * @throws {Error} If minSegmentLength is less than 2.
 *
 * @example
 * ```ts
 * const data = [1, 1, 1, 5, 5, 5, 1, 1, 1];
 * const result = binarySegmentation(data, { costFunction: "meanvar" });
 * console.log(result.changepoints); // [3, 6]
 * ```
 */
export function binarySegmentation(
  data: number[],
  options: BinarySegmentationOptions = {},
): ChangepointResult {
  const {
    costFunction = "meanvar",
    penalty = "BIC",
    minSegmentLength = 2,
    maxChangepoints = Infinity,
  } = options;

  const n = data.length;
  if (n < 2) {
    throw new Error("Data must have at least 2 elements");
  }
  if (minSegmentLength < 2) {
    throw new Error("Minimum segment length must be at least 2");
  }

  const cost = getCostFn(costFunction, data);
  const pen = computePenalty(penalty, n, costFunction);
  const penaltyLabel = typeof penalty === "number" ? `manual(${penalty})` : penalty;

  const changepoints: number[] = [];

  /**
   * Recursively find the best split point within [start, end).
   */
  function findSplit(start: number, end: number): void {
    if (changepoints.length >= maxChangepoints) return;
    if (end - start < 2 * minSegmentLength) return;

    const baseCost = cost(start, end);
    let bestGain = -Infinity;
    let bestSplit = -1;

    for (let t = start + minSegmentLength; t <= end - minSegmentLength; t++) {
      const splitCost = cost(start, t) + cost(t, end);
      const gain = baseCost - splitCost;
      if (gain > bestGain) {
        bestGain = gain;
        bestSplit = t;
      }
    }

    if (bestSplit >= 0 && bestGain > pen) {
      changepoints.push(bestSplit);
      findSplit(start, bestSplit);
      findSplit(bestSplit, end);
    }
  }

  findSplit(0, n);
  changepoints.sort((a, b) => a - b);

  const segments = buildSegments(data, changepoints);
  let totalCost = 0;
  const boundaries = [0, ...changepoints, n];
  for (let i = 0; i < boundaries.length - 1; i++) {
    totalCost += cost(boundaries[i], boundaries[i + 1]);
  }

  return {
    changepoints,
    segments,
    cost: totalCost,
    penalty: penaltyLabel,
  };
}

// ---- Bayesian Online Changepoint Detection (BOCPD) ----

/**
 * Performs Bayesian Online Changepoint Detection (BOCPD) on a sequence of
 * observations assuming Gaussian data with a conjugate Normal-Gamma prior.
 *
 * At each time step, the algorithm maintains a probability distribution over
 * run lengths (the number of observations since the last changepoint). A
 * changepoint is detected when the probability of run length 0 exceeds the
 * given threshold.
 *
 * @param data - Array of numeric values (observations arriving sequentially).
 * @param options - Configuration options for the algorithm.
 * @returns A {@link BOCPDResult} with detected changepoints, the full run length
 *   probability matrix, and the most probable run length at each time step.
 * @throws {Error} If the data array has fewer than 1 element.
 * @throws {Error} If hazardLambda is not positive.
 * @throws {Error} If threshold is not in (0, 1).
 *
 * @example
 * ```ts
 * const data = [1, 1, 1, 1, 5, 5, 5, 5];
 * const result = bocpd(data, { hazardLambda: 1/4 });
 * console.log(result.changepoints); // [4] or similar
 * ```
 */
export function bocpd(data: number[], options: BOCPDOptions = {}): BOCPDResult {
  const {
    hazardLambda = 1 / 250,
    priorMu = 0,
    priorKappa = 1,
    priorAlpha = 1,
    priorBeta = 1,
    threshold = 0.5,
  } = options;

  if (data.length < 1) {
    throw new Error("Data must have at least 1 element");
  }
  if (hazardLambda <= 0) {
    throw new Error("Hazard lambda must be positive");
  }
  if (threshold <= 0 || threshold >= 1) {
    throw new Error("Threshold must be between 0 and 1 (exclusive)");
  }

  const n = data.length;
  const H = hazardLambda; // constant hazard: P(changepoint) = H

  // Run length probabilities: R[t] is the distribution over run lengths at time t
  // R[t][r] = P(run_length = r | x_{1:t})
  const runLengthProbs: number[][] = [];
  const maxRunLengthProb: number[] = [];
  const changepoints: number[] = [];

  // Sufficient statistics for each run length hypothesis
  // For Normal-Gamma conjugate: track mu, kappa, alpha, beta per run length
  let muArr: number[] = [priorMu];
  let kappaArr: number[] = [priorKappa];
  let alphaArr: number[] = [priorAlpha];
  let betaArr: number[] = [priorBeta];

  // Initial run length distribution: P(r_0 = 0) = 1
  let R: number[] = [1.0];

  for (let t = 0; t < n; t++) {
    const x = data[t];
    const rLen = R.length;

    // 1. Compute predictive probabilities under each run length hypothesis
    // Student-t predictive: p(x | mu, kappa, alpha, beta)
    const predProbs: number[] = new Array(rLen);
    for (let r = 0; r < rLen; r++) {
      predProbs[r] = studentTPredictive(
        x,
        muArr[r],
        kappaArr[r],
        alphaArr[r],
        betaArr[r],
      );
    }

    // 2. Growth probabilities: P(r_t = r+1, x_{1:t}) = P(r_{t-1}=r) * predProb * (1-H)
    const growthProbs: number[] = new Array(rLen);
    for (let r = 0; r < rLen; r++) {
      growthProbs[r] = R[r] * predProbs[r] * (1 - H);
    }

    // 3. Changepoint probability: P(r_t = 0, x_{1:t}) = sum over r of P(r_{t-1}=r)*predProb*H
    let cpProb = 0;
    for (let r = 0; r < rLen; r++) {
      cpProb += R[r] * predProbs[r] * H;
    }

    // 4. Assemble new run length distribution
    const newR: number[] = new Array(rLen + 1);
    newR[0] = cpProb;
    for (let r = 0; r < rLen; r++) {
      newR[r + 1] = growthProbs[r];
    }

    // 5. Normalize
    let total = 0;
    for (let r = 0; r <= rLen; r++) {
      total += newR[r];
    }
    if (total > 0) {
      for (let r = 0; r <= rLen; r++) {
        newR[r] /= total;
      }
    }

    // 6. Update sufficient statistics
    // For run length 0 (new segment), reset to prior
    const newMu: number[] = new Array(rLen + 1);
    const newKappa: number[] = new Array(rLen + 1);
    const newAlpha: number[] = new Array(rLen + 1);
    const newBeta: number[] = new Array(rLen + 1);

    newMu[0] = priorMu;
    newKappa[0] = priorKappa;
    newAlpha[0] = priorAlpha;
    newBeta[0] = priorBeta;

    for (let r = 0; r < rLen; r++) {
      const k = kappaArr[r];
      const m = muArr[r];
      const a = alphaArr[r];
      const b = betaArr[r];

      newMu[r + 1] = (k * m + x) / (k + 1);
      newKappa[r + 1] = k + 1;
      newAlpha[r + 1] = a + 0.5;
      newBeta[r + 1] = b + (k * (x - m) ** 2) / (2 * (k + 1));
    }

    R = newR;
    muArr = newMu;
    kappaArr = newKappa;
    alphaArr = newAlpha;
    betaArr = newBeta;

    // Record run length probabilities
    runLengthProbs.push([...R]);

    // Find most probable run length
    let maxProb = -1;
    let maxR = 0;
    for (let r = 0; r < R.length; r++) {
      if (R[r] > maxProb) {
        maxProb = R[r];
        maxR = r;
      }
    }
    maxRunLengthProb.push(maxR);

    // Detect changepoint: run length 0 has high probability
    // Skip the very first time step (t=0) since it's trivially run length 0
    if (t > 0 && R[0] > threshold) {
      changepoints.push(t);
    }
  }

  return {
    changepoints,
    runLengthProbabilities: runLengthProbs,
    maxRunLengthProb,
  };
}

// ---- Helper: Student-t predictive probability ----

/**
 * Computes the predictive probability of observation x under a Normal-Gamma
 * posterior, which is a Student-t distribution.
 *
 * @param x - The observation.
 * @param mu - Posterior mean.
 * @param kappa - Posterior precision scaling.
 * @param alpha - Posterior shape.
 * @param beta - Posterior rate.
 * @returns The predictive probability density at x.
 */
function studentTPredictive(
  x: number,
  mu: number,
  kappa: number,
  alpha: number,
  beta: number,
): number {
  // Student-t with 2*alpha degrees of freedom
  // location = mu, scale^2 = beta*(kappa+1)/(alpha*kappa)
  const df = 2 * alpha;
  const scaleSq = (beta * (kappa + 1)) / (alpha * kappa);
  const scale = Math.sqrt(scaleSq);

  const z = (x - mu) / scale;
  // Student-t PDF: Gamma((df+1)/2) / (Gamma(df/2) * sqrt(df*pi)) * (1 + z^2/df)^(-(df+1)/2)
  // We use log-gamma for numerical stability
  const logDensity =
    gammaLn((df + 1) / 2) -
    gammaLn(df / 2) -
    0.5 * Math.log(df * Math.PI) -
    Math.log(scale) -
    ((df + 1) / 2) * Math.log(1 + (z * z) / df);

  return Math.exp(logDensity);
}

/**
 * Log-gamma function using the Lanczos approximation.
 *
 * @param x - Input value (must be positive).
 * @returns ln(Gamma(x)).
 */
function gammaLn(x: number): number {
  if (x <= 0) return Infinity;

  const g = 7;
  const coefficients = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];

  if (x < 0.5) {
    // Reflection formula
    return (
      Math.log(Math.PI / Math.sin(Math.PI * x)) - gammaLn(1 - x)
    );
  }

  x -= 1;
  let a = coefficients[0];
  const t = x + g + 0.5;
  for (let i = 1; i < coefficients.length; i++) {
    a += coefficients[i] / (x + i);
  }

  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}
