import { Dataset, AnovaResult } from "../types";
import { mean } from "../utils/descriptive";
import { regularizedBeta } from "../utils/math";

/**
 * Upper-tail p-value for the F-distribution: P(F >= f | d1, d2).
 */
function fPValue(f: number, d1: number, d2: number): number {
  if (f <= 0) return 1;
  const x = (d1 * f) / (d1 * f + d2);
  return 1 - regularizedBeta(x, d1 / 2, d2 / 2);
}

/**
 * Performs a one-way analysis of variance (ANOVA).
 *
 * Tests the null hypothesis that the population means of all k groups
 * are equal. Partitions total variation into between-group and within-group
 * components, and uses the F-distribution to compute a p-value.
 *
 * @param groups - Array of datasets, one per group (at least 2 groups required;
 *   each group must have at least 1 observation; total observations must exceed
 *   the number of groups)
 * @param alpha - Significance level for the test (default 0.05)
 * @returns An {@link AnovaResult} containing the F-statistic, p-value,
 *   between/within degrees of freedom, sum of squares, mean squares,
 *   and whether the null hypothesis is rejected
 * @throws {Error} If fewer than 2 groups are provided
 * @throws {Error} If any group has zero observations
 * @throws {Error} If total observations do not exceed the number of groups
 *
 * @example
 * ```ts
 * const result = oneWayAnova([[5, 6, 7], [8, 9, 10], [3, 4, 5]]);
 * console.log(result.fStatistic, result.pValue, result.rejected);
 * ```
 */
export function oneWayAnova(
  groups: Dataset[],
  alpha: number = 0.05,
): AnovaResult {
  const k = groups.length;
  if (k < 2) throw new Error(`Invalid parameter 'groups': expected at least 2 groups, received ${k}`);
  for (let gi = 0; gi < groups.length; gi++) {
    if (groups[gi].length < 1) throw new Error(`Invalid parameter 'groups': expected at least 1 observation in group ${gi}, received ${groups[gi].length}`);
  }

  const totalN = groups.reduce((sum, g) => sum + g.length, 0);
  if (totalN <= k) throw new Error(`Invalid parameter 'groups': expected total observations > number of groups, received totalN=${totalN}, k=${k}`);

  // Grand mean
  const grandMean = groups.reduce((sum, g) => sum + g.reduce((a, b) => a + b, 0), 0) / totalN;

  // Between-group sum of squares
  let ssBetween = 0;
  for (const g of groups) {
    const gMean = mean(g);
    ssBetween += g.length * (gMean - grandMean) ** 2;
  }

  // Within-group sum of squares
  let ssWithin = 0;
  for (const g of groups) {
    const gMean = mean(g);
    for (const v of g) {
      ssWithin += (v - gMean) ** 2;
    }
  }

  const dfBetween = k - 1;
  const dfWithin = totalN - k;
  const msBetween = ssBetween / dfBetween;
  const msWithin = ssWithin / dfWithin;
  const fStatistic = msBetween / msWithin;
  const pValue = fPValue(fStatistic, dfBetween, dfWithin);

  return {
    fStatistic,
    pValue,
    dfBetween,
    dfWithin,
    ssBetween,
    ssWithin,
    msBetween,
    msWithin,
    rejected: pValue < alpha,
  };
}
