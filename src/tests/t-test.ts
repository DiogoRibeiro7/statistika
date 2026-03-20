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

/** One-sample t-test: tests whether the population mean equals mu0. */
export function oneSampleTTest(
  data: Dataset,
  mu0: number = 0,
  alpha: number = 0.05,
): HypothesisTestResult {
  if (data.length < 2) throw new Error("Need at least 2 observations");
  const n = data.length;
  const m = mean(data);
  const s = Math.sqrt(variance(data, true));
  const t = (m - mu0) / (s / Math.sqrt(n));
  const df = n - 1;
  const pValue = tPValue(t, df);
  return { statistic: t, pValue, degreesOfFreedom: df, rejected: pValue < alpha };
}

/** Two-sample t-test (equal variances assumed). */
export function twoSampleTTest(
  data1: Dataset,
  data2: Dataset,
  alpha: number = 0.05,
): HypothesisTestResult {
  if (data1.length < 2 || data2.length < 2) throw new Error("Need at least 2 observations per group");
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

/** Welch's t-test (unequal variances). */
export function welchTTest(
  data1: Dataset,
  data2: Dataset,
  alpha: number = 0.05,
): HypothesisTestResult {
  if (data1.length < 2 || data2.length < 2) throw new Error("Need at least 2 observations per group");
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

/** Paired t-test: tests whether the mean difference is zero. */
export function pairedTTest(
  data1: Dataset,
  data2: Dataset,
  alpha: number = 0.05,
): HypothesisTestResult {
  if (data1.length !== data2.length) throw new Error("Paired samples must have equal length");
  const diffs = data1.map((v, i) => v - data2[i]);
  return oneSampleTTest(diffs, 0, alpha);
}
