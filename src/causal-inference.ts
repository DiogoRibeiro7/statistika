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

export interface PropensityScoreResult {
  /** Estimated propensity scores P(T=1|X) for each observation. */
  scores: number[];
  /** Logistic regression coefficients (including intercept at [0]). */
  coefficients: number[];
  /** Number of IRLS iterations. */
  iterations: number;
}

/**
 * Estimate propensity scores via logistic regression.
 *
 * @param X  Covariates (n × p).
 * @param treatment  Treatment indicator (0 or 1, length n).
 * @param options.maxIterations  Maximum IRLS iterations (default 100).
 * @param options.tolerance  Convergence tolerance (default 1e-8).
 */
export function propensityScore(
  X: number[][],
  treatment: number[],
  options: { maxIterations?: number; tolerance?: number } = {},
): PropensityScoreResult {
  const { maxIterations = 100, tolerance = 1e-8 } = options;
  const n = X.length;
  if (n !== treatment.length) {
    throw new Error("X and treatment must have the same length");
  }
  if (n < 2) throw new Error("Need at least 2 observations");

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

export interface IPWResult {
  /** Average Treatment Effect (ATE). */
  ate: number;
  /** Average Treatment Effect on the Treated (ATT). */
  att: number;
  /** Propensity scores used. */
  scores: number[];
}

/**
 * Estimate ATE and ATT using inverse probability weighting.
 *
 * ATE = (1/n) Σ [Tᵢ Yᵢ / eᵢ − (1−Tᵢ) Yᵢ / (1−eᵢ)]
 * ATT = (1/n₁) Σ Tᵢ Yᵢ − Σ [(1−Tᵢ) eᵢ Yᵢ / (1−eᵢ)] / Σ [(1−Tᵢ) eᵢ / (1−eᵢ)]
 *
 * @param y  Outcome variable (length n).
 * @param treatment  Treatment indicator (0 or 1, length n).
 * @param scores  Propensity scores (length n).
 */
export function ipw(
  y: number[],
  treatment: number[],
  scores: number[],
): IPWResult {
  const n = y.length;
  if (n !== treatment.length || n !== scores.length) {
    throw new Error("y, treatment, and scores must have the same length");
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
 * Nearest-neighbour propensity score matching.
 *
 * Each treated unit is matched to the control unit with the closest
 * propensity score. Returns ATT = mean(Y_treated) − mean(Y_matched_control).
 *
 * @param y  Outcome variable (length n).
 * @param treatment  Treatment indicator (0 or 1, length n).
 * @param scores  Propensity scores (length n).
 */
export function propensityMatching(
  y: number[],
  treatment: number[],
  scores: number[],
): MatchingResult {
  const n = y.length;
  if (n !== treatment.length || n !== scores.length) {
    throw new Error("y, treatment, and scores must have the same length");
  }

  const treatedIdx: number[] = [];
  const controlIdx: number[] = [];
  for (let i = 0; i < n; i++) {
    if (treatment[i] === 1) treatedIdx.push(i);
    else controlIdx.push(i);
  }

  if (treatedIdx.length === 0 || controlIdx.length === 0) {
    throw new Error("Need both treated and control observations");
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
 * Difference-in-Differences estimator.
 *
 * @param y  Outcome variable (length n).
 * @param treatment  Treatment group indicator (0 or 1, length n).
 * @param post  Post-period indicator (0 or 1, length n).
 */
export function differenceInDifferences(
  y: number[],
  treatment: number[],
  post: number[],
): DiDResult {
  const n = y.length;
  if (n !== treatment.length || n !== post.length) {
    throw new Error("y, treatment, and post must have the same length");
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
      throw new Error(`No observations in group: ${key === "tp" ? "treated+post" : key === "tc" ? "treated+pre" : key === "cp" ? "control+post" : "control+pre"}`);
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
 * Stage 1: Regress endogenous variable(s) X on instruments Z.
 * Stage 2: Regress outcome Y on fitted values X̂.
 *
 * @param y  Outcome variable (length n).
 * @param X  Endogenous regressor(s) (n × p or length n for simple case).
 * @param Z  Instruments (n × q, q ≥ p). May include exogenous controls.
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
    throw new Error("y, X, and Z must have the same number of observations");
  }

  const p = xMat[0].length;
  const q = Z[0].length;
  if (q < p) throw new Error("Need at least as many instruments as endogenous variables");

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
 * Fits separate linear regressions on each side of the cutoff and
 * estimates the treatment effect as the jump at the discontinuity.
 *
 * @param y  Outcome variable (length n).
 * @param running  Running/forcing variable (length n).
 * @param cutoff  The discontinuity threshold.
 * @param options.bandwidth  Only use observations within this distance of the cutoff.
 */
export function rdd(
  y: number[],
  running: number[],
  cutoff: number,
  options: { bandwidth?: number } = {},
): RDDResult {
  const n = y.length;
  if (n !== running.length) {
    throw new Error("y and running must have the same length");
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
    throw new Error("Need at least 2 observations on each side of the cutoff");
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
 * Uses 2SLS where the instrument is the indicator 1(running ≥ cutoff)
 * and the endogenous variable is the actual treatment take-up.
 *
 * @param y  Outcome variable (length n).
 * @param treatment  Actual treatment take-up (continuous or binary, length n).
 * @param running  Running/forcing variable (length n).
 * @param cutoff  The discontinuity threshold.
 * @param options.bandwidth  Only use observations within this distance of the cutoff.
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
    throw new Error("y, treatment, and running must have the same length");
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
    throw new Error("Need at least 4 observations within bandwidth");
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
