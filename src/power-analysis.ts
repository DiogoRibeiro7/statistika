import { Normal } from "./distributions/continuous/normal";
import { ChiSquared } from "./distributions/continuous/chi-squared";
import { FDistribution } from "./distributions/continuous/f-distribution";
import { regularizedBeta } from "./utils/math";

// ---- Result types ----

/**
 * Result of a statistical power analysis.
 */
export interface PowerResult {
  /** Computed statistical power (probability of rejecting a false H0). */
  power: number;
  /** Significance level used. */
  alpha: number;
  /** Effect size used. */
  effectSize: number;
  /** Sample size used. */
  sampleSize: number;
  /** Name of the statistical test. */
  test: string;
}

/**
 * Result of a sample size determination.
 */
export interface SampleSizeResult {
  /** Minimum sample size required. */
  sampleSize: number;
  /** Actual power achieved at the computed sample size. */
  achievedPower: number;
  /** Significance level used. */
  alpha: number;
  /** Effect size used. */
  effectSize: number;
  /** Name of the statistical test. */
  test: string;
}

// ---- Non-central distribution helpers ----

const stdNorm = new Normal();

/**
 * CDF of the non-central t-distribution via normal approximation
 * (Abramowitz & Stegun 26.7.10 / Laubscher's approximation).
 */
function noncentralTCdf(t: number, df: number, ncp: number): number {
  // For large df, use normal approximation directly
  if (df > 1000) {
    return stdNorm.cdf(t - ncp);
  }

  // Laubscher's normal approximation to the non-central t
  const a = 1 - 1 / (4 * df);
  const b = 1 + t * t / (2 * df);
  const z = (t * a - ncp) / Math.sqrt(b);
  return stdNorm.cdf(z);
}

/**
 * CDF of the non-central chi-squared distribution.
 * Uses the series expansion with Poisson weights.
 */
function noncentralChiSqCdf(
  x: number,
  df: number,
  lambda: number,
): number {
  if (x <= 0) return 0;
  if (lambda === 0) {
    // Central chi-squared
    const cs = new ChiSquared(df);
    return cs.cdf(x);
  }

  // Sum Poisson-weighted central chi-squared CDFs
  const halfLambda = lambda / 2;
  let totalProb = 0;
  let poissonWeight = Math.exp(-halfLambda); // P(J=0)

  for (let j = 0; j < 200; j++) {
    if (j > 0) {
      poissonWeight *= halfLambda / j;
    }

    // Central chi-squared CDF with df + 2j degrees of freedom
    // Using regularized gamma: P(k/2, x/2) where k = df + 2j
    const k = df + 2 * j;
    const centralCdf = regularizedGammaPLocal(k / 2, x / 2);
    totalProb += poissonWeight * centralCdf;

    // Early termination when remaining tail is negligible
    if (j > 5 && poissonWeight < 1e-15) break;
  }

  return Math.min(1, Math.max(0, totalProb));
}

/**
 * CDF of the non-central F-distribution.
 * F_nc(x; d1, d2, lambda) where lambda is the non-centrality parameter.
 */
function noncentralFCdf(
  x: number,
  d1: number,
  d2: number,
  lambda: number,
): number {
  if (x <= 0) return 0;
  if (lambda === 0) {
    const f = new FDistribution(d1, d2);
    return f.cdf(x);
  }

  // Convert to non-central beta: Y = (d1*x)/(d1*x + d2), Y ~ Beta_nc(d1/2, d2/2, lambda)
  const y = (d1 * x) / (d1 * x + d2);
  return noncentralBetaCdf(y, d1 / 2, d2 / 2, lambda);
}

/**
 * CDF of the non-central beta distribution via series expansion.
 */
function noncentralBetaCdf(
  x: number,
  a: number,
  b: number,
  lambda: number,
): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;

  const halfLambda = lambda / 2;
  let totalProb = 0;
  let poissonWeight = Math.exp(-halfLambda);

  for (let j = 0; j < 200; j++) {
    if (j > 0) {
      poissonWeight *= halfLambda / j;
    }

    const centralBetaCdf = regularizedBeta(x, a + j, b);
    totalProb += poissonWeight * centralBetaCdf;

    if (j > 5 && poissonWeight < 1e-15) break;
  }

  return Math.min(1, Math.max(0, totalProb));
}

/**
 * Local regularized gamma P implementation to handle non-integer df.
 * Uses the series expansion: P(a, x) = (e^-x * x^a / Gamma(a)) * sum(x^n / Gamma(a+n+1))
 */
function regularizedGammaPLocal(a: number, x: number): number {
  if (x <= 0) return 0;
  if (x > a + 50) return 1; // effectively 1 for large x

  // Series expansion
  let sum = 1 / a;
  let term = 1 / a;
  for (let n = 1; n < 300; n++) {
    term *= x / (a + n);
    sum += term;
    if (Math.abs(term) < 1e-15 * Math.abs(sum)) break;
  }

  const logPrefix = a * Math.log(x) - x - gammaLnLocal(a);
  return Math.min(1, sum * Math.exp(logPrefix));
}

/**
 * Log-gamma via Stirling's approximation with Lanczos coefficients.
 */
function gammaLnLocal(x: number): number {
  const g = 7;
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];

  if (x < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - gammaLnLocal(1 - x);
  }

  x -= 1;
  let a = c[0];
  const t = x + g + 0.5;
  for (let i = 1; i < c.length; i++) {
    a += c[i] / (x + i);
  }

  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

// ---- Bisection solver for sample size ----

function solveSampleSize(
  powerFn: (n: number) => number,
  targetPower: number,
  maxN: number = 1_000_000,
): number {
  // Find upper bound
  let lo = 2;
  let hi = 10;
  while (hi < maxN && powerFn(hi) < targetPower) {
    lo = hi;
    hi = hi * 2;
  }
  if (hi > maxN) hi = maxN;

  // Bisect to find minimum integer n achieving target power
  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (powerFn(mid) >= targetPower) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }

  return lo;
}

// ====================================================================
//  T-TEST POWER
// ====================================================================

/**
 * Power of a two-sample t-test.
 *
 * Computes the probability of rejecting H0 (no difference) when the true
 * effect is Cohen's d. Uses the non-central t-distribution with
 * ncp = d * sqrt(n/2) and df = 2n - 2.
 *
 * @param effectSize - Cohen's d (standardized mean difference)
 * @param n - Sample size per group
 * @param alpha - Significance level (default 0.05)
 * @param tails - 1 or 2 (default 2)
 * @returns PowerResult containing the computed power and input parameters
 * @throws Error if effect size is not finite
 * @throws Error if n is not an integer >= 2
 * @throws Error if alpha is not in (0, 1)
 *
 * @example
 * ```ts
 * const result = tTestPower(0.5, 64);
 * // result.power — probability of detecting a medium effect with n=64 per group
 * ```
 */
export function tTestPower(
  effectSize: number,
  n: number,
  alpha = 0.05,
  tails: 1 | 2 = 2,
): PowerResult {
  validateInputs(effectSize, n, alpha);
  const df = 2 * n - 2;
  const ncp = effectSize * Math.sqrt(n / 2);

  let power: number;
  if (tails === 2) {
    const tCrit = criticalT(alpha / 2, df);
    power =
      1 -
      noncentralTCdf(tCrit, df, ncp) +
      noncentralTCdf(-tCrit, df, ncp);
  } else {
    const tCrit = criticalT(alpha, df);
    power = 1 - noncentralTCdf(tCrit, df, ncp);
  }

  return { power, alpha, effectSize, sampleSize: n, test: "Two-sample t-test" };
}

/**
 * Required sample size per group for a two-sample t-test.
 *
 * Finds the smallest integer n such that the power of the two-sample
 * t-test reaches the desired level, using bisection search.
 *
 * @param effectSize - Cohen's d (standardized mean difference)
 * @param power - Desired power (default 0.80)
 * @param alpha - Significance level (default 0.05)
 * @param tails - 1 or 2 (default 2)
 * @returns SampleSizeResult with the minimum sample size and achieved power
 * @throws Error if effect size is zero
 * @throws Error if power is not in (0, 1)
 * @throws Error if alpha is not in (0, 1)
 *
 * @example
 * ```ts
 * const result = tTestSampleSize(0.5);
 * // result.sampleSize — minimum n per group for 80% power at d=0.5
 * ```
 */
export function tTestSampleSize(
  effectSize: number,
  power = 0.80,
  alpha = 0.05,
  tails: 1 | 2 = 2,
): SampleSizeResult {
  if (effectSize === 0) throw new Error(`Invalid parameter 'effectSize': expected a non-zero number, received ${effectSize}`);
  if (power <= 0 || power >= 1) throw new Error(`Invalid parameter 'power': expected a value in (0, 1), received ${power}`);
  validateAlpha(alpha);

  const n = solveSampleSize(
    (nn) => tTestPower(effectSize, nn, alpha, tails).power,
    power,
  );

  return {
    sampleSize: n,
    achievedPower: tTestPower(effectSize, n, alpha, tails).power,
    alpha,
    effectSize,
    test: "Two-sample t-test",
  };
}

/**
 * Power of a one-sample t-test.
 *
 * Uses the non-central t-distribution with ncp = d * sqrt(n) and df = n - 1.
 *
 * @param effectSize - Cohen's d (mean / sd)
 * @param n - Sample size
 * @param alpha - Significance level (default 0.05)
 * @param tails - 1 or 2 (default 2)
 * @returns PowerResult containing the computed power
 * @throws Error if effect size is not finite
 * @throws Error if n is not an integer >= 2
 * @throws Error if alpha is not in (0, 1)
 *
 * @example
 * ```ts
 * const result = oneSampleTTestPower(0.5, 30);
 * console.log(result.power); // power for detecting d=0.5 with n=30
 * ```
 */
export function oneSampleTTestPower(
  effectSize: number,
  n: number,
  alpha = 0.05,
  tails: 1 | 2 = 2,
): PowerResult {
  validateInputs(effectSize, n, alpha);
  const df = n - 1;
  const ncp = effectSize * Math.sqrt(n);

  let power: number;
  if (tails === 2) {
    const tCrit = criticalT(alpha / 2, df);
    power =
      1 -
      noncentralTCdf(tCrit, df, ncp) +
      noncentralTCdf(-tCrit, df, ncp);
  } else {
    const tCrit = criticalT(alpha, df);
    power = 1 - noncentralTCdf(tCrit, df, ncp);
  }

  return { power, alpha, effectSize, sampleSize: n, test: "One-sample t-test" };
}

/**
 * Required sample size for a one-sample t-test.
 *
 * @param effectSize - Cohen's d (mean / sd)
 * @param power - Desired power (default 0.80)
 * @param alpha - Significance level (default 0.05)
 * @param tails - 1 or 2 (default 2)
 * @returns SampleSizeResult with the minimum sample size and achieved power
 * @throws Error if effect size is zero
 * @throws Error if power is not in (0, 1)
 * @throws Error if alpha is not in (0, 1)
 *
 * @example
 * ```ts
 * const result = oneSampleTTestSampleSize(0.5);
 * console.log(result.sampleSize); // minimum n for 80% power at d=0.5
 * ```
 */
export function oneSampleTTestSampleSize(
  effectSize: number,
  power = 0.80,
  alpha = 0.05,
  tails: 1 | 2 = 2,
): SampleSizeResult {
  if (effectSize === 0) throw new Error(`Invalid parameter 'effectSize': expected a non-zero number, received ${effectSize}`);
  if (power <= 0 || power >= 1) throw new Error(`Invalid parameter 'power': expected a value in (0, 1), received ${power}`);
  validateAlpha(alpha);

  const n = solveSampleSize(
    (nn) => oneSampleTTestPower(effectSize, nn, alpha, tails).power,
    power,
  );

  return {
    sampleSize: n,
    achievedPower: oneSampleTTestPower(effectSize, n, alpha, tails).power,
    alpha,
    effectSize,
    test: "One-sample t-test",
  };
}

/**
 * Power of a paired t-test.
 *
 * Equivalent to a one-sample t-test on the paired differences.
 *
 * @param effectSize - Cohen's d for paired differences
 * @param n - Number of pairs
 * @param alpha - Significance level (default 0.05)
 * @param tails - 1 or 2 (default 2)
 * @returns PowerResult containing the computed power
 * @throws Error if effect size is not finite
 * @throws Error if n is not an integer >= 2
 * @throws Error if alpha is not in (0, 1)
 *
 * @example
 * ```ts
 * const result = pairedTTestPower(0.5, 30);
 * console.log(result.power); // power for paired t-test with 30 pairs
 * ```
 */
export function pairedTTestPower(
  effectSize: number,
  n: number,
  alpha = 0.05,
  tails: 1 | 2 = 2,
): PowerResult {
  const result = oneSampleTTestPower(effectSize, n, alpha, tails);
  return { ...result, test: "Paired t-test" };
}

/**
 * Required sample size (number of pairs) for a paired t-test.
 *
 * @param effectSize - Cohen's d for paired differences
 * @param power - Desired power (default 0.80)
 * @param alpha - Significance level (default 0.05)
 * @param tails - 1 or 2 (default 2)
 * @returns SampleSizeResult with the minimum number of pairs and achieved power
 * @throws Error if effect size is zero
 * @throws Error if power is not in (0, 1)
 * @throws Error if alpha is not in (0, 1)
 *
 * @example
 * ```ts
 * const result = pairedTTestSampleSize(0.5);
 * console.log(result.sampleSize); // minimum pairs for 80% power
 * ```
 */
export function pairedTTestSampleSize(
  effectSize: number,
  power = 0.80,
  alpha = 0.05,
  tails: 1 | 2 = 2,
): SampleSizeResult {
  const result = oneSampleTTestSampleSize(effectSize, power, alpha, tails);
  return { ...result, test: "Paired t-test" };
}

// ====================================================================
//  ONE-WAY ANOVA POWER
// ====================================================================

/**
 * Power of a one-way ANOVA (F-test).
 *
 * Uses the non-central F-distribution with df1 = k - 1, df2 = k(n - 1),
 * and non-centrality parameter lambda = k * n * f^2 where f is Cohen's f.
 *
 * @param effectSize - Cohen's f (sqrt of variance of group means / within-group SD)
 * @param k - Number of groups
 * @param n - Sample size per group
 * @param alpha - Significance level (default 0.05)
 * @returns PowerResult containing the computed power
 * @throws Error if fewer than 2 groups
 * @throws Error if effect size is not finite
 * @throws Error if n is not an integer >= 2
 * @throws Error if alpha is not in (0, 1)
 *
 * @example
 * ```ts
 * const result = anovaPower(0.25, 3, 30);
 * // result.power — power for detecting a small-medium effect across 3 groups
 * ```
 */
export function anovaPower(
  effectSize: number,
  k: number,
  n: number,
  alpha = 0.05,
): PowerResult {
  if (k < 2) throw new Error(`Invalid parameter 'k': expected at least 2 groups, received ${k}`);
  validateInputs(effectSize, n, alpha);

  const df1 = k - 1;
  const df2 = k * (n - 1);
  const lambda = k * n * effectSize * effectSize; // non-centrality parameter

  // Critical F value
  const fDist = new FDistribution(df1, Math.max(3, df2));
  const fCrit = fDist.quantile(1 - alpha);

  // Power = P(F_nc > fCrit)
  const power = 1 - noncentralFCdf(fCrit, df1, df2, lambda);

  return {
    power,
    alpha,
    effectSize,
    sampleSize: n,
    test: `One-way ANOVA (k=${k})`,
  };
}

/**
 * Required sample size per group for a one-way ANOVA.
 *
 * @param effectSize - Cohen's f
 * @param k - Number of groups
 * @param power - Desired power (default 0.80)
 * @param alpha - Significance level (default 0.05)
 * @returns SampleSizeResult with the minimum sample size per group and achieved power
 * @throws Error if effect size is zero
 * @throws Error if fewer than 2 groups
 * @throws Error if power is not in (0, 1)
 * @throws Error if alpha is not in (0, 1)
 *
 * @example
 * ```ts
 * const result = anovaSampleSize(0.25, 3);
 * console.log(result.sampleSize); // n per group for 80% power
 * ```
 */
export function anovaSampleSize(
  effectSize: number,
  k: number,
  power = 0.80,
  alpha = 0.05,
): SampleSizeResult {
  if (effectSize === 0) throw new Error(`Invalid parameter 'effectSize': expected a non-zero number, received ${effectSize}`);
  if (k < 2) throw new Error(`Invalid parameter 'k': expected at least 2 groups, received ${k}`);
  if (power <= 0 || power >= 1) throw new Error(`Invalid parameter 'power': expected a value in (0, 1), received ${power}`);
  validateAlpha(alpha);

  const n = solveSampleSize(
    (nn) => anovaPower(effectSize, k, nn, alpha).power,
    power,
  );

  return {
    sampleSize: n,
    achievedPower: anovaPower(effectSize, k, n, alpha).power,
    alpha,
    effectSize,
    test: `One-way ANOVA (k=${k})`,
  };
}

// ====================================================================
//  CHI-SQUARED TEST POWER
// ====================================================================

/**
 * Power of a chi-squared test of independence.
 *
 * Uses the non-central chi-squared distribution with non-centrality
 * parameter lambda = n * w^2 where w is Cohen's w.
 *
 * @param effectSize - Cohen's w (= Cramer's V for 2x2 tables)
 * @param df - Degrees of freedom = (rows-1)*(cols-1)
 * @param n - Total sample size
 * @param alpha - Significance level (default 0.05)
 * @returns PowerResult containing the computed power
 * @throws Error if df < 1
 * @throws Error if effect size is not finite
 * @throws Error if n is not an integer >= 2
 * @throws Error if alpha is not in (0, 1)
 *
 * @example
 * ```ts
 * const result = chiSquaredPower(0.3, 1, 100);
 * console.log(result.power); // power for a 2x2 table with n=100
 * ```
 */
export function chiSquaredPower(
  effectSize: number,
  df: number,
  n: number,
  alpha = 0.05,
): PowerResult {
  if (df < 1) throw new Error(`Invalid parameter 'df': expected at least 1, received ${df}`);
  validateInputs(effectSize, n, alpha);

  const lambda = n * effectSize * effectSize; // non-centrality
  const chiSq = new ChiSquared(df);
  const chiCrit = chiSq.quantile(1 - alpha);

  const power = 1 - noncentralChiSqCdf(chiCrit, df, lambda);

  return {
    power,
    alpha,
    effectSize,
    sampleSize: n,
    test: `Chi-squared test (df=${df})`,
  };
}

/**
 * Required total sample size for a chi-squared test.
 *
 * @param effectSize - Cohen's w
 * @param df - Degrees of freedom
 * @param power - Desired power (default 0.80)
 * @param alpha - Significance level (default 0.05)
 * @returns SampleSizeResult with the minimum total sample size and achieved power
 * @throws Error if effect size is zero
 * @throws Error if df < 1
 * @throws Error if power is not in (0, 1)
 * @throws Error if alpha is not in (0, 1)
 *
 * @example
 * ```ts
 * const result = chiSquaredSampleSize(0.3, 1);
 * console.log(result.sampleSize); // minimum n for 80% power
 * ```
 */
export function chiSquaredSampleSize(
  effectSize: number,
  df: number,
  power = 0.80,
  alpha = 0.05,
): SampleSizeResult {
  if (effectSize === 0) throw new Error(`Invalid parameter 'effectSize': expected a non-zero number, received ${effectSize}`);
  if (df < 1) throw new Error(`Invalid parameter 'df': expected at least 1, received ${df}`);
  if (power <= 0 || power >= 1) throw new Error(`Invalid parameter 'power': expected a value in (0, 1), received ${power}`);
  validateAlpha(alpha);

  const n = solveSampleSize(
    (nn) => chiSquaredPower(effectSize, df, nn, alpha).power,
    power,
  );

  return {
    sampleSize: n,
    achievedPower: chiSquaredPower(effectSize, df, n, alpha).power,
    alpha,
    effectSize,
    test: `Chi-squared test (df=${df})`,
  };
}

// ====================================================================
//  PROPORTION TEST POWER (Z-test for two proportions)
// ====================================================================

/**
 * Power of a two-proportion z-test.
 *
 * Uses the normal approximation with pooled standard error under H0
 * and unpooled standard error under H1.
 *
 * @param p1 - Proportion in group 1
 * @param p2 - Proportion in group 2
 * @param n - Sample size per group
 * @param alpha - Significance level (default 0.05)
 * @param tails - 1 or 2 (default 2)
 * @returns PowerResult containing the computed power
 * @throws Error if proportions are not in [0, 1]
 * @throws Error if n < 2
 * @throws Error if alpha is not in (0, 1)
 *
 * @example
 * ```ts
 * const result = proportionTestPower(0.5, 0.3, 100);
 * console.log(result.power); // power to detect 0.5 vs 0.3 with n=100 per group
 * ```
 */
export function proportionTestPower(
  p1: number,
  p2: number,
  n: number,
  alpha = 0.05,
  tails: 1 | 2 = 2,
): PowerResult {
  if (p1 < 0 || p1 > 1 || p2 < 0 || p2 > 1) {
    throw new Error(`Invalid parameters 'p1', 'p2': expected values in [0, 1], received p1=${p1}, p2=${p2}`);
  }
  if (n < 2) throw new Error(`Invalid parameter 'n': expected at least 2, received ${n}`);
  validateAlpha(alpha);

  const pBar = (p1 + p2) / 2;
  const se0 = Math.sqrt(2 * pBar * (1 - pBar) / n); // under H0
  const se1 = Math.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n); // under H1

  const effectSize = Math.abs(p1 - p2);

  let power: number;
  if (tails === 2) {
    const zCrit = stdNorm.quantile(1 - alpha / 2);
    power =
      stdNorm.cdf((Math.abs(p1 - p2) - zCrit * se0) / se1) +
      stdNorm.cdf((-Math.abs(p1 - p2) - zCrit * se0) / se1);
  } else {
    const zCrit = stdNorm.quantile(1 - alpha);
    power = stdNorm.cdf((Math.abs(p1 - p2) - zCrit * se0) / se1);
  }

  return {
    power: Math.max(0, Math.min(1, power)),
    alpha,
    effectSize,
    sampleSize: n,
    test: "Two-proportion z-test",
  };
}

/**
 * Required sample size per group for a two-proportion z-test.
 *
 * @param p1 - Proportion in group 1
 * @param p2 - Proportion in group 2
 * @param power - Desired power (default 0.80)
 * @param alpha - Significance level (default 0.05)
 * @param tails - 1 or 2 (default 2)
 * @returns SampleSizeResult with the minimum sample size per group and achieved power
 * @throws Error if proportions are equal
 * @throws Error if power is not in (0, 1)
 *
 * @example
 * ```ts
 * const result = proportionTestSampleSize(0.5, 0.3);
 * console.log(result.sampleSize); // n per group for 80% power
 * ```
 */
export function proportionTestSampleSize(
  p1: number,
  p2: number,
  power = 0.80,
  alpha = 0.05,
  tails: 1 | 2 = 2,
): SampleSizeResult {
  if (p1 === p2) throw new Error(`Invalid parameters 'p1', 'p2': expected different values, received p1=${p1}, p2=${p2}`);
  if (power <= 0 || power >= 1) throw new Error(`Invalid parameter 'power': expected a value in (0, 1), received ${power}`);

  const n = solveSampleSize(
    (nn) => proportionTestPower(p1, p2, nn, alpha, tails).power,
    power,
  );

  return {
    sampleSize: n,
    achievedPower: proportionTestPower(p1, p2, n, alpha, tails).power,
    alpha,
    effectSize: Math.abs(p1 - p2),
    test: "Two-proportion z-test",
  };
}

// ====================================================================
//  CORRELATION POWER
// ====================================================================

/**
 * Power to detect a Pearson correlation.
 *
 * Uses Fisher's z-transform: z_r = 0.5 * ln((1+r)/(1-r)) with
 * standard error SE = 1 / sqrt(n - 3).
 *
 * @param r - Expected correlation coefficient (must be in (-1, 1))
 * @param n - Sample size (must be >= 4)
 * @param alpha - Significance level (default 0.05)
 * @param tails - 1 or 2 (default 2)
 * @returns PowerResult containing the computed power
 * @throws Error if |r| >= 1
 * @throws Error if n < 4
 * @throws Error if alpha is not in (0, 1)
 *
 * @example
 * ```ts
 * const result = correlationPower(0.3, 50);
 * console.log(result.power); // power to detect r=0.3 with n=50
 * ```
 */
export function correlationPower(
  r: number,
  n: number,
  alpha = 0.05,
  tails: 1 | 2 = 2,
): PowerResult {
  if (Math.abs(r) >= 1) throw new Error(`Invalid parameter 'r': expected a value in (-1, 1), received ${r}`);
  if (n < 4) throw new Error(`Invalid parameter 'n': expected at least 4, received ${n}`);
  validateAlpha(alpha);

  // Fisher z-transform
  const zr = 0.5 * Math.log((1 + r) / (1 - r));
  const se = 1 / Math.sqrt(n - 3);

  let power: number;
  if (tails === 2) {
    const zCrit = stdNorm.quantile(1 - alpha / 2);
    power =
      1 -
      stdNorm.cdf(zCrit - zr / se) +
      stdNorm.cdf(-zCrit - zr / se);
  } else {
    const zCrit = stdNorm.quantile(1 - alpha);
    power = 1 - stdNorm.cdf(zCrit - zr / se);
  }

  return {
    power: Math.max(0, Math.min(1, power)),
    alpha,
    effectSize: Math.abs(r),
    sampleSize: n,
    test: "Pearson correlation",
  };
}

/**
 * Required sample size to detect a Pearson correlation.
 *
 * @param r - Expected correlation coefficient (must be non-zero)
 * @param power - Desired power (default 0.80)
 * @param alpha - Significance level (default 0.05)
 * @param tails - 1 or 2 (default 2)
 * @returns SampleSizeResult with the minimum sample size and achieved power
 * @throws Error if r is zero
 * @throws Error if power is not in (0, 1)
 *
 * @example
 * ```ts
 * const result = correlationSampleSize(0.3);
 * console.log(result.sampleSize); // minimum n for 80% power at r=0.3
 * ```
 */
export function correlationSampleSize(
  r: number,
  power = 0.80,
  alpha = 0.05,
  tails: 1 | 2 = 2,
): SampleSizeResult {
  if (r === 0) throw new Error(`Invalid parameter 'r': expected a non-zero number, received ${r}`);
  if (power <= 0 || power >= 1) throw new Error(`Invalid parameter 'power': expected a value in (0, 1), received ${power}`);

  const n = solveSampleSize(
    (nn) => correlationPower(r, nn, alpha, tails).power,
    power,
    1_000_000,
  );

  return {
    sampleSize: n,
    achievedPower: correlationPower(r, n, alpha, tails).power,
    alpha,
    effectSize: Math.abs(r),
    test: "Pearson correlation",
  };
}

// ---- Utility helpers ----

function criticalT(alphaOneSide: number, df: number): number {
  // Find t such that P(T > t) = alphaOneSide using the central t CDF
  // P(T <= t) = 1 - alphaOneSide
  // Use bisection
  const target = 1 - alphaOneSide;
  let lo = 0;
  let hi = 50;
  for (let i = 0; i < 100; i++) {
    const mid = (lo + hi) / 2;
    const t2 = mid * mid;
    const xt = df / (df + t2);
    const cdf = 1 - 0.5 * regularizedBeta(xt, df / 2, 0.5);
    if (cdf < target) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return (lo + hi) / 2;
}

function validateInputs(effectSize: number, n: number, alpha: number): void {
  if (!isFinite(effectSize)) throw new Error(`Invalid parameter 'effectSize': expected a finite number, received ${effectSize}`);
  if (n < 2 || !Number.isInteger(n)) throw new Error(`Invalid parameter 'n': expected an integer >= 2, received ${n}`);
  validateAlpha(alpha);
}

function validateAlpha(alpha: number): void {
  if (alpha <= 0 || alpha >= 1) throw new Error(`Invalid parameter 'alpha': expected a value in (0, 1), received ${alpha}`);
}
