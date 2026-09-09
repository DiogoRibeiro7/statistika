import { Dataset, HypothesisTestResult } from "../types";
import { mean, variance } from "../utils/descriptive";
import { regularizedBeta } from "../utils/math";

/**
 * Compute the two-tailed p-value from a t-statistic and degrees of freedom
 * using the regularized incomplete beta function.
 */
function tPValue(t: number, df: number): number {
  const x = df / (df + t * t);
  return regularizedBeta(x, df / 2, 0.5);
}

/**
 * Performs a one-sample t-test.
 *
 * Tests the null hypothesis that the population mean of `data` equals `mu0`.
 * Uses the t-distribution to compute a two-tailed p-value.
 *
 * @param data - Array of numeric observations (at least 2 required)
 * @param mu0 - Hypothesized population mean (default 0)
 * @param alpha - Significance level for the test (default 0.05)
 * @returns A {@link HypothesisTestResult} containing the t-statistic, two-tailed p-value,
 *   degrees of freedom (n - 1), and whether the null hypothesis is rejected
 * @throws {Error} If `data` contains fewer than 2 observations
 *
 * @example
 * ```ts
 * const result = oneSampleTTest([2.3, 1.9, 2.5, 2.1, 2.8], 2.0);
 * console.log(result.pValue, result.rejected);
 * ```
 */
export function oneSampleTTest(
  data: Dataset,
  mu0: number = 0,
  alpha: number = 0.05,
): HypothesisTestResult {
  if (data.length < 2) throw new Error(`Invalid parameter 'data': expected at least 2 observations, received ${data.length}`);
  const n = data.length;
  const m = mean(data);
  const s = Math.sqrt(variance(data, true));
  const t = (m - mu0) / (s / Math.sqrt(n));
  const df = n - 1;
  const pValue = tPValue(t, df);
  return { statistic: t, pValue, degreesOfFreedom: df, rejected: pValue < alpha };
}

/**
 * Performs a two-sample (independent) t-test assuming equal variances.
 *
 * Tests the null hypothesis that the population means of two independent
 * groups are equal. Uses a pooled variance estimate and the t-distribution
 * to compute a two-tailed p-value.
 *
 * @param data1 - First sample of numeric observations (at least 2 required)
 * @param data2 - Second sample of numeric observations (at least 2 required)
 * @param alpha - Significance level for the test (default 0.05)
 * @returns A {@link HypothesisTestResult} containing the t-statistic, two-tailed p-value,
 *   degrees of freedom (n1 + n2 - 2), and whether the null hypothesis is rejected
 * @throws {Error} If either sample contains fewer than 2 observations
 *
 * @example
 * ```ts
 * const result = twoSampleTTest([5.1, 4.9, 5.3], [4.2, 4.0, 3.8]);
 * console.log(result.statistic, result.pValue);
 * ```
 */
export function twoSampleTTest(
  data1: Dataset,
  data2: Dataset,
  alpha: number = 0.05,
): HypothesisTestResult {
  if (data1.length < 2 || data2.length < 2) throw new Error(`Invalid parameter 'data1'/'data2': expected at least 2 observations per group, received data1.length=${data1.length}, data2.length=${data2.length}`);
  const n1 = data1.length;
  const n2 = data2.length;
  const m1 = mean(data1);
  const m2 = mean(data2);
  const v1 = variance(data1, true);
  const v2 = variance(data2, true);

  // Pooled variance
  const sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2);
  const t = (m1 - m2) / Math.sqrt(sp2 * (1 / n1 + 1 / n2));
  const df = n1 + n2 - 2;
  const pValue = tPValue(t, df);
  return { statistic: t, pValue, degreesOfFreedom: df, rejected: pValue < alpha };
}

/**
 * Performs Welch's t-test for two samples with potentially unequal variances.
 *
 * Tests the null hypothesis that the population means of two independent
 * groups are equal without assuming equal variances. Uses the
 * Welch-Satterthwaite approximation for degrees of freedom.
 *
 * @param data1 - First sample of numeric observations (at least 2 required)
 * @param data2 - Second sample of numeric observations (at least 2 required)
 * @param alpha - Significance level for the test (default 0.05)
 * @returns A {@link HypothesisTestResult} containing the t-statistic, two-tailed p-value,
 *   Welch-Satterthwaite degrees of freedom, and whether the null hypothesis is rejected
 * @throws {Error} If either sample contains fewer than 2 observations
 *
 * @example
 * ```ts
 * const result = welchTTest([10, 12, 14, 16], [8, 9, 10]);
 * console.log(result.pValue, result.degreesOfFreedom);
 * ```
 */
export function welchTTest(
  data1: Dataset,
  data2: Dataset,
  alpha: number = 0.05,
): HypothesisTestResult {
  if (data1.length < 2 || data2.length < 2) throw new Error(`Invalid parameter 'data1'/'data2': expected at least 2 observations per group, received data1.length=${data1.length}, data2.length=${data2.length}`);
  const n1 = data1.length;
  const n2 = data2.length;
  const m1 = mean(data1);
  const m2 = mean(data2);
  const v1 = variance(data1, true);
  const v2 = variance(data2, true);

  const se = Math.sqrt(v1 / n1 + v2 / n2);
  const t = (m1 - m2) / se;

  // Welch-Satterthwaite degrees of freedom
  const num = (v1 / n1 + v2 / n2) ** 2;
  const den = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1);
  const df = num / den;

  const pValue = tPValue(t, df);
  return { statistic: t, pValue, degreesOfFreedom: df, rejected: pValue < alpha };
}

/**
 * Performs a paired (dependent) t-test.
 *
 * Tests the null hypothesis that the mean difference between paired
 * observations is zero. Computes pairwise differences and delegates
 * to {@link oneSampleTTest} with mu0 = 0.
 *
 * @param data1 - First set of paired observations
 * @param data2 - Second set of paired observations (must be same length as data1)
 * @param alpha - Significance level for the test (default 0.05)
 * @returns A {@link HypothesisTestResult} containing the t-statistic, two-tailed p-value,
 *   degrees of freedom (n - 1), and whether the null hypothesis is rejected
 * @throws {Error} If the two samples have different lengths
 * @throws {Error} If the samples contain fewer than 2 observations
 *
 * @example
 * ```ts
 * const before = [200, 190, 210, 205];
 * const after = [180, 175, 195, 190];
 * const result = pairedTTest(before, after);
 * console.log(result.rejected); // true if significant difference
 * ```
 */
export function pairedTTest(
  data1: Dataset,
  data2: Dataset,
  alpha: number = 0.05,
): HypothesisTestResult {
  if (data1.length !== data2.length) throw new Error(`Invalid parameter 'data1'/'data2': expected equal length for paired samples, received data1.length=${data1.length}, data2.length=${data2.length}`);
  const diffs = data1.map((v, i) => v - data2[i]);
  return oneSampleTTest(diffs, 0, alpha);
}
