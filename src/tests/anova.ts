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
 * One-way ANOVA: tests whether the means of k groups are all equal.
 * Input: array of datasets (one per group).
 */
export function oneWayAnova(
  groups: Dataset[],
  alpha: number = 0.05,
): AnovaResult {
  const k = groups.length;
  if (k < 2) throw new Error("Need at least 2 groups");
  for (const g of groups) {
    if (g.length < 1) throw new Error("Each group must have at least 1 observation");
  }

  const totalN = groups.reduce((sum, g) => sum + g.length, 0);
  if (totalN <= k) throw new Error("Total observations must exceed number of groups");

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
