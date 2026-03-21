import { Dataset, HypothesisTestResult } from "../types";
import { normalCdf } from "../utils/linalg";

/**
 * Two-tailed p-value from a z-score.
 */
function zPValue(z: number): number {
  return 2 * (1 - normalCdf(Math.abs(z)));
}

/**
 * Mann-Whitney U test (two-tailed).
 *
 * Non-parametric test to assess whether two independent samples come from
 * the same distribution. Uses the normal approximation with continuity
 * correction and tie correction.
 *
 * @param data1 - First sample
 * @param data2 - Second sample
 * @param alpha - Significance level (default 0.05)
 * @returns HypothesisTestResult where statistic is U, degreesOfFreedom is 0
 */
export function mannWhitneyU(
  data1: Dataset,
  data2: Dataset,
  alpha: number = 0.05,
): HypothesisTestResult {
  if (data1.length < 1 || data2.length < 1) {
    throw new Error("Both samples must have at least 1 observation");
  }

  const n1 = data1.length;
  const n2 = data2.length;

  // Combine and rank
  const combined: { value: number; group: number }[] = [
    ...data1.map((v) => ({ value: v, group: 0 })),
    ...data2.map((v) => ({ value: v, group: 1 })),
  ];
  combined.sort((a, b) => a.value - b.value);

  const ranks = assignRanks(combined.map((c) => c.value));

  // Sum of ranks for group 1
  let R1 = 0;
  for (let i = 0; i < combined.length; i++) {
    if (combined[i].group === 0) {
      R1 += ranks[i];
    }
  }

  const U1 = R1 - (n1 * (n1 + 1)) / 2;
  const U2 = n1 * n2 - U1;
  const U = Math.min(U1, U2);

  // Normal approximation with tie correction
  const meanU = (n1 * n2) / 2;
  const N = n1 + n2;

  // Tie correction factor
  const tieGroups = countTieGroups(combined.map((c) => c.value));
  let tieCorrection = 0;
  for (const t of tieGroups) {
    tieCorrection += t * t * t - t;
  }

  const varU =
    (n1 * n2 * (N + 1)) / 12 -
    (n1 * n2 * tieCorrection) / (12 * N * (N - 1));

  // Continuity correction
  const z = (Math.abs(U - meanU) - 0.5) / Math.sqrt(varU);
  const pValue = zPValue(z);

  return {
    statistic: U,
    pValue,
    degreesOfFreedom: 0,
    rejected: pValue < alpha,
  };
}

/**
 * Wilcoxon signed-rank test (two-tailed).
 *
 * Non-parametric test for paired samples. Tests whether the median
 * difference is zero. Uses normal approximation with tie correction.
 *
 * @param data1 - First paired sample
 * @param data2 - Second paired sample
 * @param alpha - Significance level (default 0.05)
 * @returns HypothesisTestResult where statistic is W (smaller of W+, W-), degreesOfFreedom is 0
 */
export function wilcoxonSignedRank(
  data1: Dataset,
  data2: Dataset,
  alpha: number = 0.05,
): HypothesisTestResult {
  if (data1.length !== data2.length) {
    throw new Error("Paired samples must have equal length");
  }
  if (data1.length < 2) {
    throw new Error("Need at least 2 paired observations");
  }

  // Compute differences, excluding zeros
  const diffs: { absDiff: number; sign: number }[] = [];
  for (let i = 0; i < data1.length; i++) {
    const diff = data1[i] - data2[i];
    if (diff !== 0) {
      diffs.push({ absDiff: Math.abs(diff), sign: Math.sign(diff) });
    }
  }

  if (diffs.length === 0) {
    return { statistic: 0, pValue: 1, degreesOfFreedom: 0, rejected: false };
  }

  // Rank absolute differences
  diffs.sort((a, b) => a.absDiff - b.absDiff);
  const absValues = diffs.map((d) => d.absDiff);
  const ranks = assignRanks(absValues);

  // Compute W+ and W-
  let wPlus = 0;
  let wMinus = 0;
  for (let i = 0; i < diffs.length; i++) {
    if (diffs[i].sign > 0) {
      wPlus += ranks[i];
    } else {
      wMinus += ranks[i];
    }
  }

  const W = Math.min(wPlus, wMinus);
  const n = diffs.length;

  // Normal approximation with tie correction
  const meanW = (n * (n + 1)) / 4;

  const tieGroups = countTieGroups(absValues);
  let tieCorrection = 0;
  for (const t of tieGroups) {
    tieCorrection += t * t * t - t;
  }

  const varW = (n * (n + 1) * (2 * n + 1)) / 24 - tieCorrection / 48;

  const z = (Math.abs(W - meanW) - 0.5) / Math.sqrt(varW);
  const pValue = zPValue(z);

  return {
    statistic: W,
    pValue,
    degreesOfFreedom: 0,
    rejected: pValue < alpha,
  };
}

/** Assign ranks with tie averaging (input must be sorted). */
function assignRanks(sorted: number[]): number[] {
  const n = sorted.length;
  const ranks = new Array(n);
  let i = 0;
  while (i < n) {
    let j = i;
    while (j < n && sorted[j] === sorted[i]) {
      j++;
    }
    const avgRank = (i + j + 1) / 2; // 1-based average
    for (let k = i; k < j; k++) {
      ranks[k] = avgRank;
    }
    i = j;
  }
  return ranks;
}

/** Count tie group sizes for an already-sorted array. */
function countTieGroups(sorted: number[]): number[] {
  const groups: number[] = [];
  let i = 0;
  while (i < sorted.length) {
    let j = i;
    while (j < sorted.length && sorted[j] === sorted[i]) {
      j++;
    }
    if (j - i > 1) {
      groups.push(j - i);
    }
    i = j;
  }
  return groups;
}
