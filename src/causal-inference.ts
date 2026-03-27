/**
 * Causal inference methods.
 *
 * - **Propensity score** estimation (logistic), matching (nearest-neighbour),
 *   IPW (inverse probability weighting), and ATT/ATE estimation.
 * - **Difference-in-Differences (DiD)** with 2×2 and panel data.
 * - **Instrumental Variables / 2SLS** (two-stage least squares).
 * - **Regression Discontinuity Design (RDD)** — sharp and fuzzy.
 */

import { mean } from "./utils/descriptive";
import { solveLinearSystem } from "./utils/linalg";

// ── Propensity Scores ─────────────────────────────────────────────────────

/**
 * Result of propensity score estimation via logistic regression.
 *
 * The propensity score e(X) = P(T=1 | X) is the probability of receiving
 * treatment given the observed covariates. Used for matching, stratification,
 * and inverse probability weighting in causal inference.
 */
export interface PropensityScoreResult {
  /** Estimated propensity scores P(T=1|X) for each observation. */
  scores: number[];
  /** Logistic regression coefficients (including intercept at [0]). */
  coefficients: number[];
  /** Number of IRLS iterations. */
  iterations: number;
}

/**
 * Estimate propensity scores via logistic regression (IRLS).
 *
 * Fits a logistic regression model P(T=1 | X) = sigmoid(X * beta) using
 * iteratively reweighted least squares (Newton-Raphson). The propensity
 * score is the fitted probability of receiving treatment.
 *
 * @param X - Covariate matrix (n observations x p features)
 * @param treatment - Treatment indicator array (0 or 1, length n)
 * @param options - Optional configuration
 * @param options.maxIterations - Maximum IRLS iterations (default 100)
 * @param options.tolerance - Convergence tolerance for coefficient updates (default 1e-8)
 * @returns A {@link PropensityScoreResult} with estimated scores, logistic regression
 *   coefficients, and iteration count
 * @throws {Error} If X and treatment have different lengths
 * @throws {Error} If fewer than 2 observations
 *
 * @example
 * ```ts
 * const X = [[1.2, 0.5], [0.8, 1.1], [1.5, 0.3], [0.9, 1.4]];
 * const treatment = [1, 0, 1, 0];
 * const result = propensityScore(X, treatment);
 * console.log(result.scores); // [0.65, 0.35, 0.72, 0.28]
 * ```
 */
export function propensityScore(
  X: number[][],
  treatment: number[],
  options: { maxIterations?: number; tolerance?: number } = {},
): PropensityScoreResult {
  const { maxIterations = 100, tolerance = 1e-8 } = options;
  const n = X.length;
  if (n !== treatment.length) {
    throw new Error(`Invalid parameters 'X', 'treatment': expected same length, received X.length=${n}, treatment.length=${treatment.length}`);
  }
  if (n < 2) throw new Error(`Invalid parameter 'X': expected at least 2 observations, received ${n}`);

  const p = X[0].length;
  const cols = p + 1;
  const beta = new Array(cols).fill(0);

  let iterations = 0;
  for (let iter = 0; iter < maxIterations; iter++) {
    iterations = iter + 1;

    const mu = new Array(n);
    const w = new Array(n);
    for (let i = 0; i < n; i++) {
      let z = beta[0];
      for (let j = 0; j < p; j++) z += beta[j + 1] * X[i][j];
      mu[i] = sigmoid(z);
      w[i] = Math.max(mu[i] * (1 - mu[i]), 1e-12);
    }

    const gradient = new Array(cols).fill(0);
    for (let i = 0; i < n; i++) {
      const r = treatment[i] - mu[i];
      gradient[0] += r;
      for (let j = 0; j < p; j++) gradient[j + 1] += X[i][j] * r;
    }

    const H: number[][] = Array.from({ length: cols }, () =>
      new Array(cols).fill(0),
    );
    for (let i = 0; i < n; i++) {
      const row = [1, ...X[i]];
      for (let j = 0; j < cols; j++) {
        for (let k = j; k < cols; k++) {
          H[j][k] += row[j] * row[k] * w[i];
          if (k !== j) H[k][j] = H[j][k];
        }
      }
    }

    const delta = solveLinearSystem(H, gradient);
    let maxDelta = 0;
    for (let j = 0; j < cols; j++) {
      beta[j] += delta[j];
      maxDelta = Math.max(maxDelta, Math.abs(delta[j]));
    }
    if (maxDelta < tolerance) break;
  }

  const scores = new Array(n);
  for (let i = 0; i < n; i++) {
    let z = beta[0];
    for (let j = 0; j < p; j++) z += beta[j + 1] * X[i][j];
    scores[i] = sigmoid(z);
  }

  return { scores, coefficients: beta, iterations };
}

// ── IPW (Inverse Probability Weighting) ───────────────────────────────────

/**
 * Result of inverse probability weighting estimation.
 *
 * Contains the Average Treatment Effect (ATE) and Average Treatment
 * Effect on the Treated (ATT), along with the (clipped) propensity scores used.
 */
export interface IPWResult {
  /** Average Treatment Effect (ATE). */
  ate: number;
  /** Average Treatment Effect on the Treated (ATT). */
  att: number;
  /** Propensity scores used. */
  scores: number[];
}

/**
 * Estimate ATE and ATT using inverse probability weighting (Horvitz-Thompson).
 *
 * The ATE uses the Horvitz-Thompson estimator:
 *   ATE = (1/n) * sum[ T_i * Y_i / e_i  -  (1-T_i) * Y_i / (1-e_i) ]
 *
 * The ATT compares treated outcomes against reweighted control outcomes:
 *   ATT = mean(Y | T=1) - sum[ (1-T_i) * w_i * Y_i ] / sum[ (1-T_i) * w_i ]
 * where w_i = e_i / (1 - e_i).
 *
 * Propensity scores are clipped to [0.01, 0.99] to avoid extreme weights.
 *
 * @param y - Outcome variable (length n)
 * @param treatment - Treatment indicator (0 or 1, length n)
 * @param scores - Propensity scores P(T=1|X) for each observation (length n)
 * @returns An {@link IPWResult} with ATE, ATT, and clipped propensity scores
 * @throws {Error} If y, treatment, and scores have different lengths
 *
 * @example
 * ```ts
 * const result = ipw(outcomes, treatment, propensityScores);
 * console.log(result.ate); // Average Treatment Effect
 * console.log(result.att); // Average Treatment Effect on the Treated
 * ```
 */
export function ipw(
  y: number[],
  treatment: number[],
  scores: number[],
): IPWResult {
  const n = y.length;
  if (n !== treatment.length || n !== scores.length) {
    throw new Error(`Invalid parameters 'y', 'treatment', 'scores': expected same length, received y.length=${n}, treatment.length=${treatment.length}, scores.length=${scores.length}`);
  }

  // Clip scores to avoid division by zero
  const e = scores.map((s) => Math.max(0.01, Math.min(0.99, s)));

  // ATE via Horvitz-Thompson
  let sumATE = 0;
  for (let i = 0; i < n; i++) {
    sumATE += (treatment[i] * y[i]) / e[i] - ((1 - treatment[i]) * y[i]) / (1 - e[i]);
  }
  const ate = sumATE / n;

  // ATT
  let sumTreated = 0;
  let nTreated = 0;
  let sumControlWeighted = 0;
  let sumControlWeights = 0;
  for (let i = 0; i < n; i++) {
    if (treatment[i] === 1) {
      sumTreated += y[i];
      nTreated++;
    } else {
      const w = e[i] / (1 - e[i]);
      sumControlWeighted += w * y[i];
      sumControlWeights += w;
    }
  }
  const att =
    nTreated > 0 && sumControlWeights > 0
      ? sumTreated / nTreated - sumControlWeighted / sumControlWeights
      : 0;

  return { ate, att, scores: e };
}

// ── Propensity Score Matching ─────────────────────────────────────────────

/**
 * Result of propensity score matching.
 *
 * Contains the ATT estimate, matched pairs (treated-control index pairs),
 * and the mean outcomes for matched treated and control groups.
 */
export interface MatchingResult {
  /** Average Treatment Effect on the Treated (ATT). */
  att: number;
  /** Matched pairs: [treated index, control index]. */
  matches: [number, number][];
  /** Mean outcome for matched treated. */
  treatedMean: number;
  /** Mean outcome for matched controls. */
  controlMean: number;
}

/**
 * Nearest-neighbour propensity score matching (with replacement).
 *
 * Each treated unit is matched to the control unit with the closest
 * propensity score (greedy, with replacement). The ATT is estimated as:
 *   ATT = mean(Y_treated) - mean(Y_matched_control)
 *
 * @param y - Outcome variable (length n)
 * @param treatment - Treatment indicator (0 or 1, length n)
 * @param scores - Propensity scores P(T=1|X) for each observation (length n)
 * @returns A {@link MatchingResult} with ATT, matched pairs, and group means
 * @throws {Error} If y, treatment, and scores have different lengths
 * @throws {Error} If there are no treated or no control observations
 *
 * @example
 * ```ts
 * const result = propensityMatching(outcomes, treatment, scores);
 * console.log(result.att);     // ATT estimate
 * console.log(result.matches); // [[treated_idx, control_idx], ...]
 * ```
 */
export function propensityMatching(
  y: number[],
  treatment: number[],
  scores: number[],
): MatchingResult {
  const n = y.length;
  if (n !== treatment.length || n !== scores.length) {
    throw new Error(`Invalid parameters 'y', 'treatment', 'scores': expected same length, received y.length=${n}, treatment.length=${treatment.length}, scores.length=${scores.length}`);
  }

  const treatedIdx: number[] = [];
  const controlIdx: number[] = [];
  for (let i = 0; i < n; i++) {
    if (treatment[i] === 1) treatedIdx.push(i);
    else controlIdx.push(i);
  }

  if (treatedIdx.length === 0 || controlIdx.length === 0) {
    throw new Error(`Invalid parameter 'treatment': expected both treated and control observations, received ${treatedIdx.length} treated and ${controlIdx.length} control`);
  }

  const matches: [number, number][] = [];
  let sumTreated = 0;
  let sumControl = 0;

  for (const ti of treatedIdx) {
    let bestDist = Infinity;
    let bestIdx = controlIdx[0];
    for (const ci of controlIdx) {
      const dist = Math.abs(scores[ti] - scores[ci]);
      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = ci;
      }
    }
    matches.push([ti, bestIdx]);
    sumTreated += y[ti];
    sumControl += y[bestIdx];
  }

  const treatedMean = sumTreated / treatedIdx.length;
  const controlMean = sumControl / treatedIdx.length;

  return {
    att: treatedMean - controlMean,
    matches,
    treatedMean,
    controlMean,
  };
}

// ── Difference-in-Differences ─────────────────────────────────────────────

/**
 * Result of a Difference-in-Differences estimation.
 *
 * The DiD estimate is:
 *   delta = (Y_treat_post - Y_treat_pre) - (Y_control_post - Y_control_pre)
 *
 * Under the parallel trends assumption, this identifies the causal effect
 * of the treatment.
 */
export interface DiDResult {
  /** The DiD estimate: (Y_treat_post − Y_treat_pre) − (Y_ctrl_post − Y_ctrl_pre). */
  estimate: number;
  /** Mean outcome for treatment group, pre-period. */
  treatPre: number;
  /** Mean outcome for treatment group, post-period. */
  treatPost: number;
  /** Mean outcome for control group, pre-period. */
  controlPre: number;
  /** Mean outcome for control group, post-period. */
  controlPost: number;
  /** Standard error of the DiD estimate. */
  standardError: number;
  /** t-statistic. */
  tStatistic: number;
}

/**
 * Difference-in-Differences (DiD) estimator.
 *
 * Computes the DiD estimate by comparing the change in outcomes between
 * treatment and control groups across pre and post periods. The standard
 * error is computed via pooled group-level variances.
 *
 * @param y - Outcome variable (length n)
 * @param treatment - Treatment group indicator (0 = control, 1 = treatment, length n)
 * @param post - Post-period indicator (0 = pre, 1 = post, length n)
 * @returns A {@link DiDResult} with the DiD estimate, group means, standard error,
 *   and t-statistic
 * @throws {Error} If y, treatment, and post have different lengths
 * @throws {Error} If any of the four groups (treated+post, treated+pre,
 *   control+post, control+pre) has no observations
 *
 * @example
 * ```ts
 * const result = differenceInDifferences(y, treatment, post);
 * console.log(result.estimate);     // DiD estimate
 * console.log(result.tStatistic);   // t-statistic for significance
 * ```
 */
export function differenceInDifferences(
  y: number[],
  treatment: number[],
  post: number[],
): DiDResult {
  const n = y.length;
  if (n !== treatment.length || n !== post.length) {
    throw new Error(`Invalid parameters 'y', 'treatment', 'post': expected same length, received y.length=${n}, treatment.length=${treatment.length}, post.length=${post.length}`);
  }

  const groups = { tp: [] as number[], tc: [] as number[], cp: [] as number[], cc: [] as number[] };
  for (let i = 0; i < n; i++) {
    if (treatment[i] === 1 && post[i] === 1) groups.tp.push(y[i]);
    else if (treatment[i] === 1 && post[i] === 0) groups.tc.push(y[i]);
    else if (treatment[i] === 0 && post[i] === 1) groups.cp.push(y[i]);
    else groups.cc.push(y[i]);
  }

  for (const [key, arr] of Object.entries(groups)) {
    if (arr.length === 0) {
      throw new Error(`Invalid parameter 'y'/'treatment'/'post': expected at least 1 observation in group ${key === "tp" ? "treated+post" : key === "tc" ? "treated+pre" : key === "cp" ? "control+post" : "control+pre"}, received 0`);
    }
  }

  const treatPost = mean(groups.tp);
  const treatPre = mean(groups.tc);
  const controlPost = mean(groups.cp);
  const controlPre = mean(groups.cc);

  const estimate = (treatPost - treatPre) - (controlPost - controlPre);

  // Standard error via pooled variance
  const varTP = sampleVar(groups.tp);
  const varTC = sampleVar(groups.tc);
  const varCP = sampleVar(groups.cp);
  const varCC = sampleVar(groups.cc);
  const se = Math.sqrt(
    varTP / groups.tp.length +
    varTC / groups.tc.length +
    varCP / groups.cp.length +
    varCC / groups.cc.length,
  );
  const tStatistic = se > 0 ? estimate / se : 0;

  return { estimate, treatPre, treatPost, controlPre, controlPost, standardError: se, tStatistic };
}

// ── Instrumental Variables / 2SLS ─────────────────────────────────────────

/**
 * Result of Two-Stage Least Squares (2SLS) instrumental variable estimation.
 *
 * Contains the regression coefficients, intercept, slopes for endogenous
 * variables, the first-stage F-statistic (testing instrument relevance),
 * and a prediction function.
 */
export interface TwoSLSResult {
  /** 2SLS coefficients (including intercept at [0]). */
  coefficients: number[];
  /** Intercept. */
  intercept: number;
  /** Endogenous variable coefficient(s). */
  slopes: number[];
  /** First-stage F-statistic (instrument relevance). */
  firstStageF: number;
  /** Predict y for given endogenous variable value(s). */
  predict: (x: number | number[]) => number;
}

/**
 * Two-Stage Least Squares (2SLS) for instrumental variable estimation.
 *
 * Stage 1: Regress each endogenous variable X_j on instruments Z (with intercept)
 *   to obtain fitted values X_hat.
 * Stage 2: Regress outcome Y on X_hat (with intercept) to obtain consistent
 *   estimates of the causal effect of X on Y.
 *
 * A first-stage F-statistic > 10 is generally considered evidence of
 * instrument relevance (Staiger & Stock rule of thumb).
 *
 * @param y - Outcome variable (length n)
 * @param X - Endogenous regressor(s): a flat array of length n for a single
 *   regressor, or an n x p matrix for multiple endogenous variables
 * @param Z - Instrument matrix (n x q, where q >= p). May include exogenous controls.
 * @returns A {@link TwoSLSResult} with coefficients, first-stage F-statistic,
 *   and a prediction function
 * @throws {Error} If y, X, and Z have different numbers of observations
 * @throws {Error} If fewer instruments than endogenous variables (q < p)
 *
 * @example
 * ```ts
 * const result = twoSLS(wages, education, [parentEducation, proximity]);
 * console.log(result.slopes[0]);    // causal effect of education on wages
 * console.log(result.firstStageF); // instrument strength (want > 10)
 * ```
 */
export function twoSLS(
  y: number[],
  X: number[] | number[][],
  Z: number[][],
): TwoSLSResult {
  const n = y.length;
  const isSimple = typeof X[0] === "number";
  const xMat: number[][] = isSimple
    ? (X as number[]).map((x) => [x])
    : (X as number[][]);

  if (xMat.length !== n || Z.length !== n) {
    throw new Error(`Invalid parameters 'y', 'X', 'Z': expected same number of observations, received y.length=${n}, X.length=${xMat.length}, Z.length=${Z.length}`);
  }

  const p = xMat[0].length;
  const q = Z[0].length;
  if (q < p) throw new Error(`Invalid parameter 'Z': expected at least as many instruments as endogenous variables (${p}), received ${q}`);

  // Stage 1: Regress each column of X on Z (with intercept)
  const zCols = q + 1;
  const xHat: number[][] = Array.from({ length: n }, () => new Array(p));

  let firstStageF = 0;
  for (let col = 0; col < p; col++) {
    const xCol = xMat.map((row) => row[col]);
    const { beta, rss } = olsWithIntercept(Z, xCol);

    // Compute fitted values
    for (let i = 0; i < n; i++) {
      let val = beta[0];
      for (let j = 0; j < q; j++) val += beta[j + 1] * Z[i][j];
      xHat[i][col] = val;
    }

    // First-stage F-statistic
    const xMean = mean(xCol);
    let ssTot = 0;
    for (let i = 0; i < n; i++) ssTot += (xCol[i] - xMean) ** 2;
    const ssReg = ssTot - rss;
    const f = (ssReg / q) / (rss / (n - zCols));
    firstStageF += f;
  }
  firstStageF /= p;

  // Stage 2: Regress y on X̂ (with intercept)
  const { beta: beta2 } = olsWithIntercept(xHat, y);

  const intercept = beta2[0];
  const slopes = beta2.slice(1);

  return {
    coefficients: beta2,
    intercept,
    slopes,
    firstStageF,
    predict: (x: number | number[]) => {
      if (typeof x === "number") return intercept + slopes[0] * x;
      let result = intercept;
      for (let j = 0; j < slopes.length; j++) result += slopes[j] * x[j];
      return result;
    },
  };
}

// ── Regression Discontinuity Design ───────────────────────────────────────

/**
 * Result of a Regression Discontinuity Design (RDD) estimation.
 *
 * Contains the estimated treatment effect at the cutoff, the linear
 * regression fits on each side of the cutoff, sample sizes, and
 * the standard error of the treatment effect estimate.
 */
export interface RDDResult {
  /** Estimated treatment effect at the cutoff. */
  estimate: number;
  /** Intercept for observations below the cutoff. */
  interceptBelow: number;
  /** Slope for observations below the cutoff. */
  slopeBelow: number;
  /** Intercept for observations above the cutoff. */
  interceptAbove: number;
  /** Slope for observations above the cutoff. */
  slopeAbove: number;
  /** Number of observations below / above the cutoff. */
  nBelow: number;
  nAbove: number;
  /** Standard error of the estimate (from pooled regression). */
  standardError: number;
}

/**
 * Sharp Regression Discontinuity Design.
 *
 * Fits separate linear regressions on each side of the cutoff (centred at 0)
 * and estimates the treatment effect as the difference in intercepts:
 *   tau = intercept_above - intercept_below
 *
 * An optional bandwidth restricts the analysis to observations close to the cutoff.
 *
 * @param y - Outcome variable (length n)
 * @param running - Running/forcing variable (length n)
 * @param cutoff - The discontinuity threshold value
 * @param options - Optional configuration
 * @param options.bandwidth - If specified, only observations within this distance
 *   of the cutoff are used
 * @returns An {@link RDDResult} with the treatment effect estimate, regression fits,
 *   sample sizes, and standard error
 * @throws {Error} If y and running have different lengths
 * @throws {Error} If fewer than 2 observations on either side of the cutoff
 *
 * @example
 * ```ts
 * const result = rdd(testScores, runningVariable, 50);
 * console.log(result.estimate);      // treatment effect at cutoff
 * console.log(result.standardError); // SE of the estimate
 * ```
 */
export function rdd(
  y: number[],
  running: number[],
  cutoff: number,
  options: { bandwidth?: number } = {},
): RDDResult {
  const n = y.length;
  if (n !== running.length) {
    throw new Error(`Invalid parameters 'y', 'running': expected same length, received y.length=${n}, running.length=${running.length}`);
  }

  const { bandwidth } = options;

  const below: { x: number; y: number }[] = [];
  const above: { x: number; y: number }[] = [];

  for (let i = 0; i < n; i++) {
    const centred = running[i] - cutoff;
    if (bandwidth !== undefined && Math.abs(centred) > bandwidth) continue;
    if (running[i] < cutoff) {
      below.push({ x: centred, y: y[i] });
    } else {
      above.push({ x: centred, y: y[i] });
    }
  }

  if (below.length < 2 || above.length < 2) {
    throw new Error(`Invalid parameter 'running': expected at least 2 observations on each side of the cutoff, received ${below.length} below and ${above.length} above`);
  }

  // Simple linear regression on each side
  const fitBelow = simpleOLS(
    below.map((d) => d.x),
    below.map((d) => d.y),
  );
  const fitAbove = simpleOLS(
    above.map((d) => d.x),
    above.map((d) => d.y),
  );

  // Treatment effect = intercept_above - intercept_below (at x = 0, i.e. the cutoff)
  const estimate = fitAbove.intercept - fitBelow.intercept;

  // Pooled SE
  const seBelow = fitBelow.residualSE;
  const seAbove = fitAbove.residualSE;
  const se = Math.sqrt(
    (seBelow ** 2) / below.length + (seAbove ** 2) / above.length,
  );

  return {
    estimate,
    interceptBelow: fitBelow.intercept,
    slopeBelow: fitBelow.slope,
    interceptAbove: fitAbove.intercept,
    slopeAbove: fitAbove.slope,
    nBelow: below.length,
    nAbove: above.length,
    standardError: se,
  };
}

/**
 * Fuzzy Regression Discontinuity Design.
 *
 * Unlike sharp RDD where treatment assignment is deterministic at the cutoff,
 * fuzzy RDD handles cases where the probability of treatment changes
 * discontinuously but not from 0 to 1.
 *
 * Uses 2SLS where the instrument is the indicator 1(running >= cutoff)
 * and the centred running variable, and the endogenous variable is the
 * actual treatment take-up.
 *
 * @param y - Outcome variable (length n)
 * @param treatment - Actual treatment take-up (continuous or binary, length n)
 * @param running - Running/forcing variable (length n)
 * @param cutoff - The discontinuity threshold
 * @param options - Optional configuration
 * @param options.bandwidth - If specified, only observations within this distance
 *   of the cutoff are used
 * @returns Object with the treatment effect estimate, first-stage F-statistic,
 *   and number of observations used
 * @throws {Error} If y, treatment, and running have different lengths
 * @throws {Error} If fewer than 4 observations within bandwidth
 *
 * @example
 * ```ts
 * const result = fuzzyRDD(outcomes, actualTreatment, runningVar, 50);
 * console.log(result.estimate);    // LATE at the cutoff
 * console.log(result.firstStageF); // instrument strength
 * ```
 */
export function fuzzyRDD(
  y: number[],
  treatment: number[],
  running: number[],
  cutoff: number,
  options: { bandwidth?: number } = {},
): { estimate: number; firstStageF: number; nUsed: number } {
  const n = y.length;
  if (n !== treatment.length || n !== running.length) {
    throw new Error(`Invalid parameters 'y', 'treatment', 'running': expected same length, received y.length=${n}, treatment.length=${treatment.length}, running.length=${running.length}`);
  }

  const { bandwidth } = options;

  // Filter by bandwidth
  const yF: number[] = [];
  const tF: number[] = [];
  const zF: number[][] = [];

  for (let i = 0; i < n; i++) {
    const centred = running[i] - cutoff;
    if (bandwidth !== undefined && Math.abs(centred) > bandwidth) continue;
    yF.push(y[i]);
    tF.push(treatment[i]);
    // Instrument: above cutoff indicator + centred running variable
    zF.push([running[i] >= cutoff ? 1 : 0, centred]);
  }

  if (yF.length < 4) {
    throw new Error(`Invalid parameter 'bandwidth': expected at least 4 observations within bandwidth, received ${yF.length}`);
  }

  const result = twoSLS(yF, tF, zF);
  return {
    estimate: result.slopes[0],
    firstStageF: result.firstStageF,
    nUsed: yF.length,
  };
}

// ── Helpers ───────────────────────────────────────────────────────────────

function sigmoid(z: number): number {
  if (z > 500) return 1;
  if (z < -500) return 0;
  return 1 / (1 + Math.exp(-z));
}

function sampleVar(arr: number[]): number {
  if (arr.length < 2) return 0;
  const m = mean(arr);
  let s = 0;
  for (const x of arr) s += (x - m) ** 2;
  return s / (arr.length - 1);
}

function simpleOLS(
  x: number[],
  y: number[],
): { intercept: number; slope: number; residualSE: number } {
  const n = x.length;
  const xBar = mean(x);
  const yBar = mean(y);
  let ssXX = 0;
  let ssXY = 0;
  for (let i = 0; i < n; i++) {
    ssXX += (x[i] - xBar) ** 2;
    ssXY += (x[i] - xBar) * (y[i] - yBar);
  }
  const slope = ssXX > 0 ? ssXY / ssXX : 0;
  const intercept = yBar - slope * xBar;
  let rss = 0;
  for (let i = 0; i < n; i++) {
    rss += (y[i] - intercept - slope * x[i]) ** 2;
  }
  const residualSE = n > 2 ? Math.sqrt(rss / (n - 2)) : 0;
  return { intercept, slope, residualSE };
}

function olsWithIntercept(
  X: number[][],
  y: number[],
): { beta: number[]; rss: number } {
  const n = X.length;
  const p = X[0].length;
  const cols = p + 1;
  const XtX: number[][] = Array.from({ length: cols }, () =>
    new Array(cols).fill(0),
  );
  const Xty = new Array(cols).fill(0);

  for (let i = 0; i < n; i++) {
    Xty[0] += y[i];
    XtX[0][0] += 1;
    for (let j = 0; j < p; j++) {
      Xty[j + 1] += X[i][j] * y[i];
      XtX[j + 1][0] += X[i][j];
      XtX[0][j + 1] += X[i][j];
      for (let k = 0; k < p; k++) {
        XtX[j + 1][k + 1] += X[i][j] * X[i][k];
      }
    }
  }

  const beta = solveLinearSystem(XtX, Xty);
  let rss = 0;
  for (let i = 0; i < n; i++) {
    let yHat = beta[0];
    for (let j = 0; j < p; j++) yHat += beta[j + 1] * X[i][j];
    rss += (y[i] - yHat) ** 2;
  }
  return { beta, rss };
}
