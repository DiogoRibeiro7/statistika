import { SeededRng } from "./random";

// ── Interfaces ──────────────────────────────────────────────────────────

/**
 * Result of Sobol sensitivity analysis.
 */
export interface SobolResult {
  /** First-order sensitivity indices S_i for each dimension */
  firstOrder: number[];
  /** Total-order sensitivity indices S_Ti for each dimension */
  totalOrder: number[];
  /** Second-order interaction indices S_ij (optional) */
  secondOrder?: number[][];
  /** Confidence intervals for indices */
  confidence?: {
    firstOrderCI: [number, number][];
    totalOrderCI: [number, number][];
  };
  /** Number of model evaluations used */
  nSamples: number;
}

/**
 * Result of Morris method (elementary effects) screening.
 */
export interface MorrisResult {
  /** Absolute mean of elementary effects per dimension */
  muStar: number[];
  /** Standard deviation of elementary effects per dimension */
  sigma: number[];
  /** Signed mean of elementary effects per dimension */
  mu: number[];
  /** Variable importance ranking (1 = most important) */
  rankings: number[];
}

/**
 * Result of Fourier Amplitude Sensitivity Testing.
 */
export interface FASTResult {
  /** First-order sensitivity indices for each dimension */
  firstOrder: number[];
  /** Number of model evaluations used */
  nSamples: number;
}

/**
 * Result of correlation-based screening.
 */
export interface CorrelationScreeningResult {
  /** Pearson correlation between each input and the output */
  pearson: number[];
  /** Spearman rank correlation between each input and the output */
  spearman: number[];
  /** Partial correlation coefficients */
  partialCorrelation: number[];
  /** Variable importance ranking (1 = most important, based on absolute partial correlation) */
  rankings: number[];
}

/**
 * Structured data for scatter plots of each input versus the output.
 */
export interface ScatterPlotDataResult {
  /** Array of per-variable scatter plot data */
  plots: { variable: number; name: string; x: number[]; y: number[] }[];
}

// ── Options types ───────────────────────────────────────────────────────

/** Bounds for a single dimension: [min, max]. */
export type DimensionBounds = [number, number];

/** Common options for sensitivity analysis methods. */
export interface SensitivityOptions {
  /** Random seed for reproducibility */
  seed?: number;
  /** Bounds per dimension, each [min, max]. Defaults to [0, 1] for each. */
  bounds?: DimensionBounds[];
}

/** Options specific to the Sobol method. */
export interface SobolOptions extends SensitivityOptions {
  /** Number of base samples (total evaluations = N * (d + 2)). Default: 1000 */
  nSamples?: number;
}

/** Options specific to the Morris method. */
export interface MorrisOptions extends SensitivityOptions {
  /** Number of trajectories. Default: 10 */
  nTrajectories?: number;
  /** Number of grid levels. Default: 4 */
  levels?: number;
}

/** Options specific to the FAST method. */
export interface FASTOptions extends SensitivityOptions {
  /** Number of samples along the search curve. Default: 1000 */
  nSamples?: number;
}

// ── Helpers ─────────────────────────────────────────────────────────────

/**
 * Scale a [0,1] value to within [lo, hi].
 */
function scaleValue(u: number, lo: number, hi: number): number {
  return lo + u * (hi - lo);
}

/**
 * Resolve bounds, defaulting each dimension to [0, 1].
 */
function resolveBounds(
  dimensions: number,
  bounds?: DimensionBounds[],
): DimensionBounds[] {
  if (bounds) {
    if (bounds.length !== dimensions) {
      throw new Error(
        `Invalid parameter 'bounds': expected length ${dimensions} to match dimensions, received ${bounds.length}`,
      );
    }
    for (let i = 0; i < bounds.length; i++) {
      if (bounds[i][0] >= bounds[i][1]) {
        throw new Error(
          `Invalid parameter 'bounds': expected min < max at index ${i}, received [${bounds[i][0]}, ${bounds[i][1]}]`,
        );
      }
    }
    return bounds;
  }
  return Array.from({ length: dimensions }, () => [0, 1] as DimensionBounds);
}

/**
 * Compute mean of an array.
 */
function mean(arr: number[]): number {
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += arr[i];
  return s / arr.length;
}

/**
 * Compute variance of an array (population variance).
 */
function variance(arr: number[]): number {
  const m = mean(arr);
  let s = 0;
  for (let i = 0; i < arr.length; i++) {
    const d = arr[i] - m;
    s += d * d;
  }
  return s / arr.length;
}

/**
 * Rank an array of values (1-based, average ranks for ties).
 */
function rankArray(arr: number[]): number[] {
  const n = arr.length;
  const indexed = arr.map((v, i) => ({ v, i }));
  indexed.sort((a, b) => a.v - b.v);
  const ranks = new Array<number>(n);
  let i = 0;
  while (i < n) {
    let j = i;
    while (j < n - 1 && indexed[j + 1].v === indexed[j].v) j++;
    const avgRank = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) {
      ranks[indexed[k].i] = avgRank;
    }
    i = j + 1;
  }
  return ranks;
}

/**
 * Compute importance rankings from absolute values (1 = largest absolute value).
 */
function computeRankings(values: number[]): number[] {
  const n = values.length;
  const absVals = values.map((v) => Math.abs(v));
  const indexed = absVals.map((v, i) => ({ v, i }));
  indexed.sort((a, b) => b.v - a.v); // descending
  const rankings = new Array<number>(n);
  for (let k = 0; k < n; k++) {
    rankings[indexed[k].i] = k + 1;
  }
  return rankings;
}

// ── Sobol Indices ───────────────────────────────────────────────────────

/**
 * Compute Sobol sensitivity indices using the Saltelli sampling scheme.
 *
 * Estimates first-order (S_i) and total-order (S_Ti) variance-based
 * sensitivity indices for each input dimension of a model.
 *
 * The Saltelli scheme generates two independent sample matrices A and B,
 * then for each dimension i creates A_B^(i) (A with column i from B).
 * Total model evaluations: N * (d + 2).
 *
 * @param model - Function mapping a d-dimensional input vector to a scalar output
 * @param dimensions - Number of input dimensions (d)
 * @param options - Sampling options (nSamples, seed, bounds)
 * @returns A {@link SobolResult} with first-order and total-order indices
 * @throws {Error} If dimensions is less than 1
 * @throws {Error} If nSamples is less than 2
 * @throws {Error} If bounds length does not match dimensions
 *
 * @example
 * ```ts
 * // Ishigami function: f(x) = sin(x1) + 7*sin(x2)^2 + 0.1*x3^4*sin(x1)
 * const ishigami = (x: number[]) =>
 *   Math.sin(x[0]) + 7 * Math.sin(x[1]) ** 2 + 0.1 * x[2] ** 4 * Math.sin(x[0]);
 *
 * const result = sobolIndices(ishigami, 3, {
 *   nSamples: 4096,
 *   seed: 42,
 *   bounds: [[-Math.PI, Math.PI], [-Math.PI, Math.PI], [-Math.PI, Math.PI]],
 * });
 * console.log(result.firstOrder);  // S_i for each dimension
 * console.log(result.totalOrder);  // S_Ti for each dimension
 * ```
 */
export function sobolIndices(
  model: (x: number[]) => number,
  dimensions: number,
  options: SobolOptions = {},
): SobolResult {
  if (dimensions < 1) {
    throw new Error(`Invalid parameter 'dimensions': expected at least 1, received ${dimensions}`);
  }
  const N = options.nSamples ?? 1000;
  if (N < 2) {
    throw new Error(`Invalid parameter 'nSamples': expected at least 2, received ${N}`);
  }
  const bounds = resolveBounds(dimensions, options.bounds);
  const rng = new SeededRng(options.seed ?? 42);

  const d = dimensions;

  // Generate base matrices A and B in [0,1]^d, then scale to bounds
  const A: number[][] = new Array(N);
  const B: number[][] = new Array(N);
  for (let i = 0; i < N; i++) {
    A[i] = new Array(d);
    B[i] = new Array(d);
    for (let j = 0; j < d; j++) {
      A[i][j] = scaleValue(rng.next(), bounds[j][0], bounds[j][1]);
    }
    for (let j = 0; j < d; j++) {
      B[i][j] = scaleValue(rng.next(), bounds[j][0], bounds[j][1]);
    }
  }

  // Evaluate model on A and B
  const fA = new Array<number>(N);
  const fB = new Array<number>(N);
  for (let i = 0; i < N; i++) {
    fA[i] = model(A[i]);
    fB[i] = model(B[i]);
  }

  // For each dimension, create AB_i matrix and evaluate
  const fAB: number[][] = new Array(d);
  for (let j = 0; j < d; j++) {
    fAB[j] = new Array(N);
    for (let i = 0; i < N; i++) {
      // A_B^(j): copy of A with column j replaced by B's column j
      const point = A[i].slice();
      point[j] = B[i][j];
      fAB[j][i] = model(point);
    }
  }

  // Compute total variance
  const allOutputs = fA.concat(fB);
  const grandMean = mean(allOutputs);
  const totalVar = variance(allOutputs);

  if (totalVar === 0) {
    // Model is constant — all indices are 0
    return {
      firstOrder: new Array(d).fill(0),
      totalOrder: new Array(d).fill(0),
      nSamples: N * (d + 2),
    };
  }

  // Compute first-order and total-order indices (Jansen estimators)
  const firstOrder = new Array<number>(d);
  const totalOrder = new Array<number>(d);

  for (let j = 0; j < d; j++) {
    // First order: S_i = (1/N) * sum(fB * (fAB_i - fA)) / Var(Y)
    let numFirst = 0;
    let numTotal = 0;
    for (let i = 0; i < N; i++) {
      numFirst += fB[i] * (fAB[j][i] - fA[i]);
      numTotal += (fA[i] - fAB[j][i]) * (fA[i] - fAB[j][i]);
    }
    firstOrder[j] = (numFirst / N) / totalVar;
    totalOrder[j] = (numTotal / (2 * N)) / totalVar;
  }

  return {
    firstOrder,
    totalOrder,
    nSamples: N * (d + 2),
  };
}

// ── Morris Method ───────────────────────────────────────────────────────

/**
 * Perform Morris method (Elementary Effects) screening.
 *
 * Generates r trajectories through a p-level grid in the input space.
 * Each trajectory has (d+1) points, varying one factor at a time.
 * Computes mu* (absolute mean), mu (signed mean), and sigma (std) of
 * elementary effects for each input.
 *
 * @param model - Function mapping a d-dimensional input vector to a scalar output
 * @param dimensions - Number of input dimensions (d)
 * @param options - Screening options (nTrajectories, levels, seed, bounds)
 * @returns A {@link MorrisResult} with mu*, sigma, mu, and rankings
 * @throws {Error} If dimensions is less than 1
 * @throws {Error} If nTrajectories is less than 1
 * @throws {Error} If levels is less than 2
 * @throws {Error} If bounds length does not match dimensions
 *
 * @example
 * ```ts
 * const model = (x: number[]) => x[0] + 2 * x[1] + 0.5 * x[0] * x[1];
 * const result = morrisMethod(model, 2, {
 *   nTrajectories: 20,
 *   levels: 4,
 *   seed: 123,
 *   bounds: [[0, 1], [0, 1]],
 * });
 * console.log(result.muStar);   // absolute mean of elementary effects
 * console.log(result.rankings); // [2, 1] => x[1] is most important
 * ```
 */
export function morrisMethod(
  model: (x: number[]) => number,
  dimensions: number,
  options: MorrisOptions = {},
): MorrisResult {
  if (dimensions < 1) {
    throw new Error(`Invalid parameter 'dimensions': expected at least 1, received ${dimensions}`);
  }
  const r = options.nTrajectories ?? 10;
  if (r < 1) {
    throw new Error(`Invalid parameter 'nTrajectories': expected at least 1, received ${r}`);
  }
  const p = options.levels ?? 4;
  if (p < 2) {
    throw new Error(`Invalid parameter 'levels': expected at least 2, received ${p}`);
  }
  const bounds = resolveBounds(dimensions, options.bounds);
  const rng = new SeededRng(options.seed ?? 42);

  const d = dimensions;
  const delta = p / (2 * (p - 1));

  // Collect elementary effects for each dimension
  const effects: number[][] = Array.from({ length: d }, () => []);

  for (let t = 0; t < r; t++) {
    // Generate a random base point on the p-level grid
    const base = new Array<number>(d);
    for (let j = 0; j < d; j++) {
      const level = rng.nextInt(0, Math.floor(p / 2) - 1);
      base[j] = level / (p - 1);
    }

    // Random order of dimensions to vary
    const order = Array.from({ length: d }, (_, i) => i);
    rng.shuffle(order);

    // Random direction (+delta or -delta)
    const directions = new Array<number>(d);
    for (let j = 0; j < d; j++) {
      directions[j] = rng.next() < 0.5 ? -1 : 1;
    }

    // Build trajectory: start from base, vary one dimension at a time
    let current = base.slice();
    let fCurrent = model(
      current.map((v, j) => scaleValue(v, bounds[j][0], bounds[j][1])),
    );

    for (let step = 0; step < d; step++) {
      const dim = order[step];
      const next = current.slice();
      next[dim] = current[dim] + directions[dim] * delta;

      // Clamp to [0, 1]
      if (next[dim] > 1) next[dim] = current[dim] - delta;
      if (next[dim] < 0) next[dim] = current[dim] + delta;
      // Still clamp just in case
      next[dim] = Math.max(0, Math.min(1, next[dim]));

      const fNext = model(
        next.map((v, j) => scaleValue(v, bounds[j][0], bounds[j][1])),
      );

      // Elementary effect
      const actualDelta =
        (scaleValue(next[dim], bounds[dim][0], bounds[dim][1]) -
          scaleValue(current[dim], bounds[dim][0], bounds[dim][1]));

      if (actualDelta !== 0) {
        const ee = (fNext - fCurrent) / actualDelta;
        effects[dim].push(ee);
      }

      current = next;
      fCurrent = fNext;
    }
  }

  // Compute statistics
  const muStar = new Array<number>(d);
  const sigma = new Array<number>(d);
  const mu = new Array<number>(d);

  for (let j = 0; j < d; j++) {
    const eff = effects[j];
    if (eff.length === 0) {
      muStar[j] = 0;
      sigma[j] = 0;
      mu[j] = 0;
      continue;
    }
    const n = eff.length;
    let sumAbs = 0;
    let sum = 0;
    for (let i = 0; i < n; i++) {
      sumAbs += Math.abs(eff[i]);
      sum += eff[i];
    }
    muStar[j] = sumAbs / n;
    mu[j] = sum / n;

    let sumSq = 0;
    for (let i = 0; i < n; i++) {
      const diff = eff[i] - mu[j];
      sumSq += diff * diff;
    }
    sigma[j] = Math.sqrt(sumSq / n);
  }

  const rankings = computeRankings(muStar);

  return { muStar, sigma, mu, rankings };
}

// ── FAST Method ─────────────────────────────────────────────────────────

/**
 * Perform Fourier Amplitude Sensitivity Testing (FAST).
 *
 * Estimates first-order sensitivity indices by assigning each input
 * dimension a unique integer frequency and sampling the model along
 * a parametric curve. Spectral analysis of the output identifies the
 * variance contribution of each input.
 *
 * @param model - Function mapping a d-dimensional input vector to a scalar output
 * @param dimensions - Number of input dimensions (d)
 * @param options - Options (nSamples, seed, bounds)
 * @returns A {@link FASTResult} with first-order indices
 * @throws {Error} If dimensions is less than 1
 * @throws {Error} If nSamples is less than 4 * maxFrequency + 1
 * @throws {Error} If bounds length does not match dimensions
 *
 * @example
 * ```ts
 * const model = (x: number[]) => x[0] + 2 * x[1] * x[1];
 * const result = fastMethod(model, 2, { nSamples: 1000, seed: 42 });
 * console.log(result.firstOrder); // first-order indices
 * ```
 */
export function fastMethod(
  model: (x: number[]) => number,
  dimensions: number,
  options: FASTOptions = {},
): FASTResult {
  if (dimensions < 1) {
    throw new Error(`Invalid parameter 'dimensions': expected at least 1, received ${dimensions}`);
  }
  const bounds = resolveBounds(dimensions, options.bounds);
  const d = dimensions;

  // Assign integer frequencies to each dimension (must be distinct and incommensurate)
  // Using a standard set of FAST frequencies
  const omega = assignFASTFrequencies(d);
  const maxOmega = Math.max(...omega);
  const minN = 4 * maxOmega + 1;

  const N = options.nSamples ?? Math.max(1000, minN);
  if (N < minN) {
    throw new Error(
      `Invalid parameter 'nSamples': expected at least ${minN} for ${d} dimensions, received ${N}`,
    );
  }

  const rng = new SeededRng(options.seed ?? 42);

  // Random phase shifts
  const phi = new Array<number>(d);
  for (let j = 0; j < d; j++) {
    phi[j] = rng.next() * 2 * Math.PI;
  }

  // Sample the search curve
  const outputs = new Array<number>(N);
  for (let i = 0; i < N; i++) {
    const s = -Math.PI + (2 * Math.PI * i) / N;
    const x = new Array<number>(d);
    for (let j = 0; j < d; j++) {
      // Transform from [-pi, pi] to [0, 1] using the sin transformation
      const u = 0.5 + (1 / Math.PI) * Math.asin(Math.sin(omega[j] * s + phi[j]));
      x[j] = scaleValue(u, bounds[j][0], bounds[j][1]);
    }
    outputs[i] = model(x);
  }

  // Compute total variance of outputs
  const totalVar = variance(outputs);

  if (totalVar === 0) {
    return {
      firstOrder: new Array(d).fill(0),
      nSamples: N,
    };
  }

  // Compute Fourier coefficients and first-order indices
  const firstOrder = new Array<number>(d);

  for (let j = 0; j < d; j++) {
    // Compute variance at frequency omega[j] and its harmonics (up to M harmonics)
    const M = Math.min(Math.floor((N - 1) / (2 * maxOmega)), 4);
    let varJ = 0;

    for (let m = 1; m <= M; m++) {
      const freq = m * omega[j];
      let cosSum = 0;
      let sinSum = 0;
      for (let i = 0; i < N; i++) {
        const s = -Math.PI + (2 * Math.PI * i) / N;
        cosSum += outputs[i] * Math.cos(freq * s);
        sinSum += outputs[i] * Math.sin(freq * s);
      }
      const a = (2 / N) * cosSum;
      const b = (2 / N) * sinSum;
      varJ += (a * a + b * b) / 2;
    }

    firstOrder[j] = varJ / totalVar;
  }

  return {
    firstOrder,
    nSamples: N,
  };
}

/**
 * Assign incommensurate frequencies for FAST.
 * Uses the standard scheme: omega_1 is the base frequency, others are chosen
 * to avoid interference.
 */
function assignFASTFrequencies(d: number): number[] {
  if (d === 1) return [1];
  // Standard FAST frequency assignment
  // omega_i for the dimension of interest is large, others are small
  // We use a simple scheme with distinct frequencies
  const omega = new Array<number>(d);
  const baseFreq = Math.max(8, d * 4);
  omega[0] = baseFreq;
  for (let j = 1; j < d; j++) {
    omega[j] = j;
  }
  return omega;
}

// ── Correlation-Based Screening ─────────────────────────────────────────

/**
 * Perform correlation-based sensitivity screening.
 *
 * Computes Pearson correlation, Spearman rank correlation, and partial
 * correlation coefficients between each input variable and the output.
 * Returns a ranking of variable importance based on absolute partial correlations.
 *
 * @param X - Input matrix: X[i][j] is the j-th input of the i-th sample
 * @param y - Output vector of length equal to X.length
 * @returns A {@link CorrelationScreeningResult} with correlations and rankings
 * @throws {Error} If X is empty or has no columns
 * @throws {Error} If X and y have different lengths
 * @throws {Error} If X has fewer than 3 rows
 *
 * @example
 * ```ts
 * const X = [[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]];
 * const y = [2.1, 4.2, 5.9, 8.1, 9.8];
 * const result = correlationScreening(X, y);
 * console.log(result.pearson);            // Pearson r for each input
 * console.log(result.spearman);           // Spearman rho for each input
 * console.log(result.partialCorrelation); // partial r for each input
 * console.log(result.rankings);           // importance ranking
 * ```
 */
export function correlationScreening(
  X: number[][],
  y: number[],
): CorrelationScreeningResult {
  if (X.length === 0) {
    throw new Error("Invalid parameter 'X': expected non-empty array, received length 0");
  }
  if (X.length !== y.length) {
    throw new Error(`Invalid parameter 'y': expected length ${X.length} to match X rows, received length ${y.length}`);
  }
  if (X.length < 3) {
    throw new Error(`Invalid parameter 'X': expected at least 3 rows, received ${X.length}`);
  }
  const n = X.length;
  const d = X[0].length;
  if (d === 0) {
    throw new Error("Invalid parameter 'X': expected at least one column, received 0");
  }

  // Extract columns
  const cols: number[][] = new Array(d);
  for (let j = 0; j < d; j++) {
    cols[j] = new Array(n);
    for (let i = 0; i < n; i++) {
      cols[j][i] = X[i][j];
    }
  }

  // Pearson correlations
  const pearson = new Array<number>(d);
  for (let j = 0; j < d; j++) {
    pearson[j] = pearsonCorrelation(cols[j], y);
  }

  // Spearman correlations
  const spearman = new Array<number>(d);
  const yRanks = rankArray(y);
  for (let j = 0; j < d; j++) {
    const xRanks = rankArray(cols[j]);
    spearman[j] = pearsonCorrelation(xRanks, yRanks);
  }

  // Partial correlations
  const partialCorrelation = new Array<number>(d);
  for (let j = 0; j < d; j++) {
    partialCorrelation[j] = computePartialCorrelation(cols, y, j);
  }

  const rankings = computeRankings(partialCorrelation);

  return { pearson, spearman, partialCorrelation, rankings };
}

/**
 * Compute Pearson correlation between two arrays.
 */
function pearsonCorrelation(a: number[], b: number[]): number {
  const n = a.length;
  const ma = mean(a);
  const mb = mean(b);
  let num = 0;
  let da = 0;
  let db = 0;
  for (let i = 0; i < n; i++) {
    const ai = a[i] - ma;
    const bi = b[i] - mb;
    num += ai * bi;
    da += ai * ai;
    db += bi * bi;
  }
  const denom = Math.sqrt(da * db);
  return denom === 0 ? 0 : num / denom;
}

/**
 * Compute partial correlation between X[j] and y, controlling for all other X columns.
 * Uses recursive formula for removing a single control variable at a time,
 * or the residual-based approach for efficiency.
 */
function computePartialCorrelation(
  cols: number[][],
  y: number[],
  targetIdx: number,
): number {
  const d = cols.length;
  const n = cols[0].length;

  if (d === 1) {
    // No other variables to control for
    return pearsonCorrelation(cols[0], y);
  }

  // Compute residuals of X[targetIdx] and y after regressing on all other X columns
  // Collect the "other" columns as the control set
  const controlCols: number[][] = [];
  for (let j = 0; j < d; j++) {
    if (j !== targetIdx) controlCols.push(cols[j]);
  }

  const residX = residualsFromOLS(cols[targetIdx], controlCols, n);
  const residY = residualsFromOLS(y, controlCols, n);

  return pearsonCorrelation(residX, residY);
}

/**
 * Compute residuals of target after OLS regression on control columns.
 * Uses normal equations with a simple approach.
 */
function residualsFromOLS(
  target: number[],
  controls: number[][],
  n: number,
): number[] {
  const k = controls.length;

  if (k === 0) return target.slice();

  // Build X^T X and X^T y for the controls
  // Add intercept
  const p = k + 1; // +1 for intercept
  const XtX: number[][] = Array.from({ length: p }, () =>
    new Array<number>(p).fill(0),
  );
  const Xty: number[] = new Array<number>(p).fill(0);

  for (let i = 0; i < n; i++) {
    // row = [1, controls[0][i], controls[1][i], ...]
    const row = new Array<number>(p);
    row[0] = 1;
    for (let j = 0; j < k; j++) row[j + 1] = controls[j][i];

    for (let a = 0; a < p; a++) {
      for (let b = 0; b < p; b++) {
        XtX[a][b] += row[a] * row[b];
      }
      Xty[a] += row[a] * target[i];
    }
  }

  // Solve XtX * beta = Xty via Gaussian elimination with partial pivoting
  const beta = solveLinearSystem(XtX, Xty);

  if (!beta) {
    // Singular system — return original values
    return target.slice();
  }

  // Compute residuals
  const residuals = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    let predicted = beta[0];
    for (let j = 0; j < k; j++) {
      predicted += beta[j + 1] * controls[j][i];
    }
    residuals[i] = target[i] - predicted;
  }
  return residuals;
}

/**
 * Solve a linear system Ax = b via Gaussian elimination with partial pivoting.
 * Returns null if the system is singular.
 */
function solveLinearSystem(
  A: number[][],
  b: number[],
): number[] | null {
  const n = A.length;
  // Augmented matrix
  const aug: number[][] = A.map((row, i) => [...row, b[i]]);

  for (let col = 0; col < n; col++) {
    // Partial pivoting
    let maxVal = Math.abs(aug[col][col]);
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(aug[row][col]) > maxVal) {
        maxVal = Math.abs(aug[row][col]);
        maxRow = row;
      }
    }
    if (maxVal < 1e-12) return null;

    // Swap rows
    if (maxRow !== col) {
      [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];
    }

    // Eliminate below
    for (let row = col + 1; row < n; row++) {
      const factor = aug[row][col] / aug[col][col];
      for (let j = col; j <= n; j++) {
        aug[row][j] -= factor * aug[col][j];
      }
    }
  }

  // Back substitution
  const x = new Array<number>(n);
  for (let i = n - 1; i >= 0; i--) {
    if (Math.abs(aug[i][i]) < 1e-12) return null;
    let sum = aug[i][n];
    for (let j = i + 1; j < n; j++) {
      sum -= aug[i][j] * x[j];
    }
    x[i] = sum / aug[i][i];
  }

  return x;
}

// ── Scatter Plot Data ───────────────────────────────────────────────────

/**
 * Generate structured scatter plot data for each input variable versus the output.
 *
 * Produces an array of plot descriptors, one per input dimension, containing
 * the input values (x) and corresponding output values (y) suitable for
 * visualization.
 *
 * @param X - Input matrix: X[i][j] is the j-th input of the i-th sample
 * @param y - Output vector of length equal to X.length
 * @param variableNames - Optional names for each input variable. Defaults to "x0", "x1", etc.
 * @returns A {@link ScatterPlotDataResult} with one plot entry per input variable
 * @throws {Error} If X is empty or has no columns
 * @throws {Error} If X and y have different lengths
 * @throws {Error} If variableNames length does not match the number of columns in X
 *
 * @example
 * ```ts
 * const X = [[1, 10], [2, 20], [3, 30]];
 * const y = [5, 10, 15];
 * const result = scatterPlotData(X, y, ["temperature", "pressure"]);
 * console.log(result.plots[0].name); // "temperature"
 * console.log(result.plots[0].x);    // [1, 2, 3]
 * console.log(result.plots[0].y);    // [5, 10, 15]
 * ```
 */
export function scatterPlotData(
  X: number[][],
  y: number[],
  variableNames?: string[],
): ScatterPlotDataResult {
  if (X.length === 0) {
    throw new Error("Invalid parameter 'X': expected non-empty array, received length 0");
  }
  if (X.length !== y.length) {
    throw new Error(`Invalid parameter 'y': expected length ${X.length} to match X rows, received length ${y.length}`);
  }
  const n = X.length;
  const d = X[0].length;
  if (d === 0) {
    throw new Error("Invalid parameter 'X': expected at least one column, received 0");
  }
  if (variableNames !== undefined && variableNames.length !== d) {
    throw new Error(
      `Invalid parameter 'variableNames': expected length ${d} to match columns, received ${variableNames.length}`,
    );
  }

  const names = variableNames ?? Array.from({ length: d }, (_, j) => `x${j}`);

  const plots = new Array(d);
  for (let j = 0; j < d; j++) {
    const xVals = new Array<number>(n);
    for (let i = 0; i < n; i++) {
      xVals[i] = X[i][j];
    }
    plots[j] = {
      variable: j,
      name: names[j],
      x: xVals,
      y: y.slice(),
    };
  }

  return { plots };
}
