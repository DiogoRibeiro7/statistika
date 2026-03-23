/**
 * Linear Mixed Effects Models (LMM).
 *
 * - **Random intercept model** — varying intercepts by group.
 * - **Random intercept + slope model** — varying intercepts and slopes.
 * - **REML estimation** — restricted maximum likelihood via EM algorithm.
 * - **ICC** — intraclass correlation coefficient.
 * - **BLUPs** — best linear unbiased predictors for random effects.
 * - **Model comparison** — likelihood ratio test, AIC, BIC.
 */

import { solveLinearSystem, invertMatrix } from "./utils/linalg";
import { mean } from "./utils/descriptive";

// ── Types ─────────────────────────────────────────────────────────────────

export interface LMMResult {
  /** Fixed effect coefficients (including intercept at [0]). */
  fixedEffects: number[];
  /** Random effects variance components. */
  randomEffects: {
    /** Variance of random intercepts. */
    interceptVariance: number;
    /** Variance of random slopes (if model includes random slopes). */
    slopeVariance?: number;
    /** Covariance between random intercept and slope. */
    interceptSlopeCovariance?: number;
  };
  /** Residual variance σ². */
  residualVariance: number;
  /** Group-level BLUPs (best linear unbiased predictors). */
  blups: Map<number | string, number[]>;
  /** Log-likelihood (REML). */
  logLikelihood: number;
  /** AIC = −2ℓ + 2k. */
  aic: number;
  /** BIC = −2ℓ + k ln(n). */
  bic: number;
  /** Number of variance parameters estimated. */
  nVarParams: number;
  /** Number of EM iterations. */
  iterations: number;
  /** Predict for new data. */
  predict: (x: number[], group?: number | string) => number;
}

export interface ICCResult {
  /** Intraclass correlation coefficient: σ²_group / (σ²_group + σ²_resid). */
  icc: number;
  /** Between-group variance. */
  betweenVariance: number;
  /** Within-group (residual) variance. */
  withinVariance: number;
}

// ── Random Intercept Model ────────────────────────────────────────────────

/**
 * Fit a linear mixed model with random intercepts.
 *
 * y = Xβ + Zb + ε
 * where b ~ N(0, σ²_b I), ε ~ N(0, σ² I)
 *
 * Estimated via EM algorithm (REML-like).
 *
 * @param y  Response variable (length n).
 * @param X  Fixed effects design matrix (n × p). If null, intercept-only.
 * @param groups  Group assignments (length n, integer or string-coded).
 * @param options.maxIterations  Max EM iterations (default 200).
 * @param options.tolerance  Convergence tolerance (default 1e-8).
 */
export function lmmRandomIntercept(
  y: number[],
  X: number[][] | null,
  groups: (number | string)[],
  options: { maxIterations?: number; tolerance?: number } = {},
): LMMResult {
  const { maxIterations = 200, tolerance = 1e-8 } = options;
  const n = y.length;
  if (groups.length !== n) throw new Error("y and groups must have same length");

  // Build design matrix with intercept
  const xMat = buildDesignMatrix(X, n);
  const p = xMat[0].length;

  // Map groups
  const { groupLabels, groupIndices } = mapGroups(groups);
  const nGroups = groupLabels.length;

  // Initial estimates
  let sigmaB2 = variance(y) * 0.5;
  let sigma2 = variance(y) * 0.5;
  let beta = new Array(p).fill(0);
  beta[0] = mean(y);

  let iterations = 0;
  for (let iter = 0; iter < maxIterations; iter++) {
    iterations = iter + 1;
    const oldSigmaB2 = sigmaB2;
    const oldSigma2 = sigma2;

    // E-step: compute BLUPs for random effects
    const blupVals = new Array(nGroups).fill(0);
    const blupVar = new Array(nGroups).fill(0);

    for (let g = 0; g < nGroups; g++) {
      const idx = groupIndices[g];
      const ni = idx.length;
      let sumResid = 0;
      for (const i of idx) {
        let xb = 0;
        for (let j = 0; j < p; j++) xb += xMat[i][j] * beta[j];
        sumResid += y[i] - xb;
      }
      const denom = ni + sigma2 / sigmaB2;
      blupVals[g] = sumResid / denom;
      blupVar[g] = sigma2 / denom;
    }

    // M-step: update fixed effects
    // Solve (X'X) β = X'(y - Zb)
    const yAdj = new Array(n);
    for (let i = 0; i < n; i++) {
      const gIdx = findGroupIndex(groups[i], groupLabels);
      yAdj[i] = y[i] - blupVals[gIdx];
    }
    beta = solveOLS(xMat, yAdj);

    // Update variance components
    let ssResid = 0;
    for (let i = 0; i < n; i++) {
      let xb = 0;
      for (let j = 0; j < p; j++) xb += xMat[i][j] * beta[j];
      const gIdx = findGroupIndex(groups[i], groupLabels);
      const resid = y[i] - xb - blupVals[gIdx];
      ssResid += resid * resid;
    }
    sigma2 = ssResid / n;

    let ssBLUP = 0;
    for (let g = 0; g < nGroups; g++) {
      ssBLUP += blupVals[g] * blupVals[g] + blupVar[g];
    }
    sigmaB2 = Math.max(1e-10, ssBLUP / nGroups);

    if (
      Math.abs(sigmaB2 - oldSigmaB2) + Math.abs(sigma2 - oldSigma2) < tolerance
    ) {
      break;
    }
  }

  // Final BLUPs
  const blups = new Map<number | string, number[]>();
  for (let g = 0; g < nGroups; g++) {
    const idx = groupIndices[g];
    const ni = idx.length;
    let sumResid = 0;
    for (const i of idx) {
      let xb = 0;
      for (let j = 0; j < p; j++) xb += xMat[i][j] * beta[j];
      sumResid += y[i] - xb;
    }
    const blup = sumResid / (ni + sigma2 / sigmaB2);
    blups.set(groupLabels[g], [blup]);
  }

  // Log-likelihood (REML)
  let ll = -0.5 * n * Math.log(2 * Math.PI);
  for (let i = 0; i < n; i++) {
    let xb = 0;
    for (let j = 0; j < p; j++) xb += xMat[i][j] * beta[j];
    const gIdx = findGroupIndex(groups[i], groupLabels);
    const blup = blups.get(groupLabels[gIdx])![0];
    const resid = y[i] - xb - blup;
    ll -= 0.5 * Math.log(sigma2) + 0.5 * resid * resid / sigma2;
  }

  const nVarParams = 2; // sigma2, sigmaB2
  const totalParams = p + nVarParams;
  const aic = -2 * ll + 2 * totalParams;
  const bic = -2 * ll + totalParams * Math.log(n);

  return {
    fixedEffects: beta,
    randomEffects: { interceptVariance: sigmaB2 },
    residualVariance: sigma2,
    blups,
    logLikelihood: ll,
    aic,
    bic,
    nVarParams,
    iterations,
    predict: (x: number[], group?: number | string) => {
      const row = [1, ...x];
      let val = 0;
      for (let j = 0; j < beta.length; j++) val += beta[j] * (row[j] ?? 0);
      if (group !== undefined && blups.has(group)) {
        val += blups.get(group)![0];
      }
      return val;
    },
  };
}

// ── Random Intercept + Slope Model ────────────────────────────────────────

/**
 * Fit a linear mixed model with random intercepts and random slopes.
 *
 * y = Xβ + Z[b₀ b₁]' + ε
 * where [b₀, b₁] ~ N(0, G), ε ~ N(0, σ² I)
 *
 * The random slope is on the first column of X (excluding intercept).
 *
 * @param y  Response variable (length n).
 * @param X  Fixed effects covariates (n × p, without intercept column).
 * @param groups  Group assignments (length n).
 * @param options.maxIterations  Max EM iterations (default 200).
 * @param options.tolerance  Convergence tolerance (default 1e-8).
 */
export function lmmRandomSlope(
  y: number[],
  X: number[][],
  groups: (number | string)[],
  options: { maxIterations?: number; tolerance?: number } = {},
): LMMResult {
  const { maxIterations = 200, tolerance = 1e-8 } = options;
  const n = y.length;
  if (X.length !== n || groups.length !== n) {
    throw new Error("y, X, and groups must have same length");
  }
  if (X[0].length < 1) throw new Error("X must have at least 1 column for random slope");

  // Build design matrix with intercept
  const xMat = X.map((row) => [1, ...row]);
  const p = xMat[0].length;

  const { groupLabels, groupIndices } = mapGroups(groups);
  const nGroups = groupLabels.length;

  // Initial estimates
  let sigma2 = variance(y) * 0.3;
  let g00 = variance(y) * 0.3; // intercept variance
  let g11 = variance(y) * 0.1; // slope variance
  let g01 = 0; // covariance
  let beta = new Array(p).fill(0);
  beta[0] = mean(y);

  let iterations = 0;
  for (let iter = 0; iter < maxIterations; iter++) {
    iterations = iter + 1;
    const oldG00 = g00;
    const oldG11 = g11;
    const oldSigma2 = sigma2;

    // E-step: compute BLUPs for each group
    const blupIntercepts = new Array(nGroups).fill(0);
    const blupSlopes = new Array(nGroups).fill(0);

    for (let g = 0; g < nGroups; g++) {
      const idx = groupIndices[g];
      const ni = idx.length;

      // Z'Z for this group: Z is ni × 2, columns are [1, x_slope]
      let z11 = 0, z12 = 0, z22 = 0;
      let zr1 = 0, zr2 = 0;
      for (const i of idx) {
        const xi = X[i][0]; // slope variable
        z11 += 1;
        z12 += xi;
        z22 += xi * xi;
        let xb = 0;
        for (let j = 0; j < p; j++) xb += xMat[i][j] * beta[j];
        const resid = y[i] - xb;
        zr1 += resid;
        zr2 += xi * resid;
      }

      // V_g^{-1} = (Z'Z/σ² + G^{-1})
      const detG = g00 * g11 - g01 * g01;
      if (Math.abs(detG) < 1e-20) {
        // Nearly singular G, just use simple estimate
        blupIntercepts[g] = zr1 / (ni + sigma2 / Math.max(g00, 1e-10));
        blupSlopes[g] = 0;
        continue;
      }
      const gInv00 = g11 / detG;
      const gInv11 = g00 / detG;
      const gInv01 = -g01 / detG;

      const v00 = z11 / sigma2 + gInv00;
      const v01 = z12 / sigma2 + gInv01;
      const v11 = z22 / sigma2 + gInv11;

      const rhs0 = zr1 / sigma2;
      const rhs1 = zr2 / sigma2;

      const detV = v00 * v11 - v01 * v01;
      if (Math.abs(detV) < 1e-20) continue;
      blupIntercepts[g] = (v11 * rhs0 - v01 * rhs1) / detV;
      blupSlopes[g] = (-v01 * rhs0 + v00 * rhs1) / detV;
    }

    // M-step: update fixed effects
    const yAdj = new Array(n);
    for (let i = 0; i < n; i++) {
      const gIdx = findGroupIndex(groups[i], groupLabels);
      yAdj[i] = y[i] - blupIntercepts[gIdx] - blupSlopes[gIdx] * X[i][0];
    }
    beta = solveOLS(xMat, yAdj);

    // Update residual variance
    let ssResid = 0;
    for (let i = 0; i < n; i++) {
      let xb = 0;
      for (let j = 0; j < p; j++) xb += xMat[i][j] * beta[j];
      const gIdx = findGroupIndex(groups[i], groupLabels);
      const resid = y[i] - xb - blupIntercepts[gIdx] - blupSlopes[gIdx] * X[i][0];
      ssResid += resid * resid;
    }
    sigma2 = Math.max(1e-10, ssResid / n);

    // Update G matrix
    let sumB00 = 0, sumB11 = 0, sumB01 = 0;
    for (let g = 0; g < nGroups; g++) {
      sumB00 += blupIntercepts[g] * blupIntercepts[g];
      sumB11 += blupSlopes[g] * blupSlopes[g];
      sumB01 += blupIntercepts[g] * blupSlopes[g];
    }
    g00 = Math.max(1e-10, sumB00 / nGroups);
    g11 = Math.max(1e-10, sumB11 / nGroups);
    g01 = sumB01 / nGroups;

    if (
      Math.abs(g00 - oldG00) + Math.abs(g11 - oldG11) + Math.abs(sigma2 - oldSigma2) < tolerance
    ) {
      break;
    }
  }

  // Final BLUPs
  const blups = new Map<number | string, number[]>();
  for (let g = 0; g < nGroups; g++) {
    const idx = groupIndices[g];
    let zr1 = 0, zr2 = 0;
    for (const i of idx) {
      let xb = 0;
      for (let j = 0; j < p; j++) xb += xMat[i][j] * beta[j];
      const resid = y[i] - xb;
      zr1 += resid;
      zr2 += X[i][0] * resid;
    }
    const ni = idx.length;
    const detG = g00 * g11 - g01 * g01;
    let b0 = zr1 / (ni + sigma2 / Math.max(g00, 1e-10));
    let b1 = 0;
    if (Math.abs(detG) > 1e-20) {
      const gInv00 = g11 / detG, gInv11 = g00 / detG, gInv01 = -g01 / detG;
      let z11 = 0, z12 = 0, z22 = 0;
      for (const i of idx) { z11 += 1; z12 += X[i][0]; z22 += X[i][0] ** 2; }
      const v00 = z11 / sigma2 + gInv00, v01 = z12 / sigma2 + gInv01, v11 = z22 / sigma2 + gInv11;
      const detV = v00 * v11 - v01 * v01;
      if (Math.abs(detV) > 1e-20) {
        b0 = (v11 * zr1 / sigma2 - v01 * zr2 / sigma2) / detV;
        b1 = (-v01 * zr1 / sigma2 + v00 * zr2 / sigma2) / detV;
      }
    }
    blups.set(groupLabels[g], [b0, b1]);
  }

  // Log-likelihood
  let ll = -0.5 * n * Math.log(2 * Math.PI);
  for (let i = 0; i < n; i++) {
    let xb = 0;
    for (let j = 0; j < p; j++) xb += xMat[i][j] * beta[j];
    const gIdx = findGroupIndex(groups[i], groupLabels);
    const b = blups.get(groupLabels[gIdx])!;
    const resid = y[i] - xb - b[0] - b[1] * X[i][0];
    ll -= 0.5 * Math.log(sigma2) + 0.5 * resid * resid / sigma2;
  }

  const nVarParams = 4; // sigma2, g00, g11, g01
  const totalParams = p + nVarParams;
  const aic = -2 * ll + 2 * totalParams;
  const bic = -2 * ll + totalParams * Math.log(n);

  return {
    fixedEffects: beta,
    randomEffects: {
      interceptVariance: g00,
      slopeVariance: g11,
      interceptSlopeCovariance: g01,
    },
    residualVariance: sigma2,
    blups,
    logLikelihood: ll,
    aic,
    bic,
    nVarParams,
    iterations,
    predict: (x: number[], group?: number | string) => {
      const row = [1, ...x];
      let val = 0;
      for (let j = 0; j < beta.length; j++) val += beta[j] * (row[j] ?? 0);
      if (group !== undefined && blups.has(group)) {
        const b = blups.get(group)!;
        val += b[0] + b[1] * x[0];
      }
      return val;
    },
  };
}

// ── Intraclass Correlation Coefficient ────────────────────────────────────

/**
 * Compute the Intraclass Correlation Coefficient (ICC).
 *
 * ICC = σ²_between / (σ²_between + σ²_within)
 *
 * Uses one-way random effects ANOVA decomposition.
 *
 * @param values  Observations (length n).
 * @param groups  Group assignments (length n).
 */
export function icc(values: number[], groups: (number | string)[]): ICCResult {
  const n = values.length;
  if (n !== groups.length) throw new Error("values and groups must have same length");
  if (n < 2) throw new Error("Need at least 2 observations");

  const { groupLabels, groupIndices } = mapGroups(groups);
  const nGroups = groupLabels.length;
  if (nGroups < 2) throw new Error("Need at least 2 groups");

  const grandMean = mean(values);

  // ANOVA decomposition
  let ssBetween = 0;
  let ssWithin = 0;
  const groupSizes: number[] = [];

  for (let g = 0; g < nGroups; g++) {
    const idx = groupIndices[g];
    const ni = idx.length;
    groupSizes.push(ni);
    let groupSum = 0;
    for (const i of idx) groupSum += values[i];
    const groupMean = groupSum / ni;
    ssBetween += ni * (groupMean - grandMean) ** 2;
    for (const i of idx) ssWithin += (values[i] - groupMean) ** 2;
  }

  const msBetween = ssBetween / (nGroups - 1);
  const msWithin = ssWithin / (n - nGroups);

  // Average group size (harmonic or simple for balanced)
  const n0Num = n - groupSizes.reduce((a, b) => a + b * b, 0) / n;
  const n0 = n0Num / (nGroups - 1);

  const betweenVariance = Math.max(0, (msBetween - msWithin) / n0);
  const withinVariance = msWithin;
  const iccVal = betweenVariance / (betweenVariance + withinVariance);

  return { icc: iccVal, betweenVariance, withinVariance };
}

// ── Likelihood Ratio Test ─────────────────────────────────────────────────

export interface LRTResult {
  /** Likelihood ratio statistic: −2(ℓ₀ − ℓ₁). */
  statistic: number;
  /** Degrees of freedom (difference in number of parameters). */
  df: number;
  /** p-value from chi-squared distribution. */
  pValue: number;
}

/**
 * Likelihood ratio test comparing two nested mixed models.
 *
 * @param restricted  Log-likelihood of the restricted (null) model.
 * @param full  Log-likelihood of the full (alternative) model.
 * @param dfDiff  Difference in number of parameters.
 */
export function lrtTest(
  restricted: number,
  full: number,
  dfDiff: number,
): LRTResult {
  if (dfDiff < 1) throw new Error("dfDiff must be positive");
  const statistic = -2 * (restricted - full);
  // Chi-squared p-value approximation
  const pValue = 1 - chiSquaredCdf(Math.max(0, statistic), dfDiff);
  return { statistic: Math.max(0, statistic), df: dfDiff, pValue };
}

// ── Helpers ───────────────────────────────────────────────────────────────

function buildDesignMatrix(X: number[][] | null, n: number): number[][] {
  if (X === null) return Array.from({ length: n }, () => [1]);
  return X.map((row) => [1, ...row]);
}

function mapGroups(groups: (number | string)[]): {
  groupLabels: (number | string)[];
  groupIndices: number[][];
} {
  const labelMap = new Map<number | string, number>();
  const groupLabels: (number | string)[] = [];
  const groupIndices: number[][] = [];

  for (let i = 0; i < groups.length; i++) {
    let gIdx = labelMap.get(groups[i]);
    if (gIdx === undefined) {
      gIdx = groupLabels.length;
      labelMap.set(groups[i], gIdx);
      groupLabels.push(groups[i]);
      groupIndices.push([]);
    }
    groupIndices[gIdx].push(i);
  }

  return { groupLabels, groupIndices };
}

function findGroupIndex(
  group: number | string,
  labels: (number | string)[],
): number {
  for (let i = 0; i < labels.length; i++) {
    if (labels[i] === group) return i;
  }
  return 0;
}

function solveOLS(X: number[][], y: number[]): number[] {
  const n = X.length;
  const p = X[0].length;
  const XtX: number[][] = Array.from({ length: p }, () => new Array(p).fill(0));
  const Xty = new Array(p).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < p; j++) {
      Xty[j] += X[i][j] * y[i];
      for (let k = 0; k < p; k++) {
        XtX[j][k] += X[i][j] * X[i][k];
      }
    }
  }
  return solveLinearSystem(XtX, Xty);
}

function variance(data: number[]): number {
  const n = data.length;
  if (n < 2) return 0;
  const m = mean(data);
  let s = 0;
  for (const x of data) s += (x - m) ** 2;
  return s / (n - 1);
}

/** Chi-squared CDF approximation using regularized gamma. */
function chiSquaredCdf(x: number, k: number): number {
  if (x <= 0) return 0;
  return regularizedGammaLower(k / 2, x / 2);
}

/** Regularized lower incomplete gamma P(a, x) via series expansion. */
function regularizedGammaLower(a: number, x: number): number {
  if (x < 0) return 0;
  if (x === 0) return 0;
  let sum = 0;
  let term = 1 / a;
  for (let n = 0; n < 200; n++) {
    sum += term;
    term *= x / (a + n + 1);
    if (Math.abs(term) < 1e-14) break;
  }
  return sum * Math.exp(-x + a * Math.log(x) - gammaLn(a));
}

function gammaLn(x: number): number {
  const c = [
    76.18009172947146, -86.50532032941677, 24.01409824083091,
    -1.231739572450155, 0.001208650973866179, -0.000005395239384953,
  ];
  let y = x;
  let tmp = x + 5.5;
  tmp -= (x + 0.5) * Math.log(tmp);
  let ser = 1.000000000190015;
  for (let j = 0; j < 6; j++) ser += c[j] / ++y;
  return -tmp + Math.log((2.5066282746310005 * ser) / x);
}
