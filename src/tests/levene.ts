import { Dataset } from "../types";
import { mean } from "../utils/descriptive";
import { regularizedBeta } from "../utils/math";

/**
 * Result of Levene's test for equality of variances.
 */
export interface LeveneTestResult {
  /** The F-like test statistic. */
  statistic: number;
  /** The p-value from the F-distribution. */
  pValue: number;
  /** Numerator degrees of freedom (k - 1). */
  df1: number;
  /** Denominator degrees of freedom (N - k). */
  df2: number;
  /** Whether to reject the null hypothesis of equal variances. */
  rejectNull: boolean;
}

/**
 * Upper-tail p-value for the F-distribution: P(F >= f | d1, d2).
 */
function fPValue(f: number, d1: number, d2: number): number {
  if (f <= 0) return 1;
  const x = (d1 * f) / (d1 * f + d2);
  return 1 - regularizedBeta(x, d1 / 2, d2 / 2);
}

/**
 * Computes the median of a numeric array.
 */
function computeMedian(data: number[]): number {
  const sorted = [...data].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0
    ? sorted[mid]
    : (sorted[mid - 1] + sorted[mid]) / 2;
}

/**
 * Performs Levene's test for equality of variances across multiple groups.
 *
 * Tests the null hypothesis that all groups have equal variance. By default
 * uses the Brown-Forsythe variant (median-based), which is more robust
 * to departures from normality than the original mean-based Levene's test.
 *
 * The test statistic is computed as an ANOVA F-statistic on the absolute
 * deviations of each observation from its group center (mean or median).
 *
 * @param groups - Array of datasets, one per group (at least 2 groups required;
 *   each group must have at least 2 observations)
 * @param options - Configuration options
 * @param options.center - Method for computing group center: "median" (default,
 *   Brown-Forsythe variant) or "mean" (original Levene's test)
 * @param options.alpha - Significance level for the rejection decision (default 0.05)
 * @returns A {@link LeveneTestResult} containing the test statistic, p-value,
 *   degrees of freedom, and rejection decision
 * @throws {Error} If fewer than 2 groups are provided
 * @throws {Error} If any group has fewer than 2 observations
 *
 * @example
 * ```ts
 * const result = leveneTest(
 *   [[5, 6, 7, 8], [10, 12, 14, 16], [1, 2, 3, 4]],
 *   { center: "median" }
 * );
 * console.log(result.statistic, result.pValue);
 * ```
 */
export function leveneTest(
  groups: Dataset[],
  options: {
    center?: "mean" | "median";
    alpha?: number;
  } = {},
): LeveneTestResult {
  const k = groups.length;
  if (k < 2) {
    throw new Error("leveneTest requires at least 2 groups (got " + k + ")");
  }

  for (let i = 0; i < k; i++) {
    if (groups[i].length < 2) {
      throw new Error(
        "leveneTest requires at least 2 observations per group (group " + i + " has " + groups[i].length + ")",
      );
    }
  }

  const centerMethod = options.center ?? "median";
  const alpha = options.alpha ?? 0.05;

  // Compute group centers
  const centers = groups.map((g) =>
    centerMethod === "median" ? computeMedian(g) : mean(g),
  );

  // Compute absolute deviations from group centers
  const deviations: number[][] = groups.map((g, gi) =>
    g.map((x) => Math.abs(x - centers[gi])),
  );

  // Compute ANOVA F-statistic on the deviations
  const totalN = groups.reduce((sum, g) => sum + g.length, 0);
  const groupSizes = groups.map((g) => g.length);

  // Group means of deviations
  const devMeans = deviations.map((d) => mean(d));

  // Grand mean of deviations
  let grandSum = 0;
  for (const d of deviations) {
    for (const v of d) {
      grandSum += v;
    }
  }
  const grandMean = grandSum / totalN;

  // Between-group sum of squares
  let ssBetween = 0;
  for (let i = 0; i < k; i++) {
    ssBetween += groupSizes[i] * (devMeans[i] - grandMean) ** 2;
  }

  // Within-group sum of squares
  let ssWithin = 0;
  for (let i = 0; i < k; i++) {
    for (const v of deviations[i]) {
      ssWithin += (v - devMeans[i]) ** 2;
    }
  }

  const df1 = k - 1;
  const df2 = totalN - k;
  const msBetween = ssBetween / df1;
  const msWithin = ssWithin / df2;
  const fStat = msWithin > 0 ? msBetween / msWithin : 0;

  const pValue = fPValue(fStat, df1, df2);

  return {
    statistic: fStat,
    pValue,
    df1,
    df2,
    rejectNull: pValue < alpha,
  };
}
