import { Dataset, HypothesisTestResult } from "../types";
import { normalCdf } from "../utils/linalg";
import { regularizedGammaP } from "../utils/math";

/**
 * Computes a two-tailed p-value from a z-score using the standard normal CDF.
 *
 * p = 2 * (1 - Phi(|z|))
 *
 * @param z - The z-score (test statistic).
 * @returns The two-tailed p-value in [0, 1].
 */
function zPValue(z: number): number {
  return 2 * (1 - normalCdf(Math.abs(z)));
}

/**
 * Performs the Mann-Whitney U test (Wilcoxon rank-sum test), two-tailed.
 *
 * A non-parametric test for assessing whether two independent samples come
 * from the same distribution (i.e., whether one sample tends to have larger
 * values than the other).
 *
 * The test statistic U is the smaller of U1 and U2, where:
 *   U1 = R1 - n1*(n1+1)/2
 *   U2 = n1*n2 - U1
 *
 * and R1 is the sum of ranks assigned to the first sample in the combined
 * ranking. Uses the normal approximation with continuity correction:
 *   z = (|U - mean(U)| - 0.5) / sqrt(Var(U))
 *
 * The variance includes a tie correction factor:
 *   Var(U) = n1*n2*(N+1)/12 - n1*n2 * sum(t^3 - t) / (12*N*(N-1))
 *
 * @param data1 - First sample (at least 1 observation).
 * @param data2 - Second sample (at least 1 observation).
 * @param alpha - Significance level for the hypothesis test (default 0.05).
 * @returns A {@link HypothesisTestResult} where:
 *   - `statistic` is the U statistic (min of U1, U2)
 *   - `pValue` is the two-tailed p-value from the normal approximation
 *   - `degreesOfFreedom` is 0 (non-parametric test)
 *   - `rejected` is true if pValue < alpha
 * @throws {Error} If either sample is empty.
 *
 * @example
 * ```ts
 * const result = mannWhitneyU([1, 2, 3, 4], [5, 6, 7, 8]);
 * result.pValue; // small p-value indicating the samples differ
 * ```
 */
export function mannWhitneyU(
  data1: Dataset,
  data2: Dataset,
  alpha: number = 0.05,
): HypothesisTestResult {
  if (data1.length < 1 || data2.length < 1) {
    throw new Error(`Invalid parameter 'data1'/'data2': expected at least 1 observation per sample, received data1.length=${data1.length}, data2.length=${data2.length}`);
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
 * Performs the Wilcoxon signed-rank test (two-tailed) for paired samples.
 *
 * A non-parametric test for whether the median of paired differences is zero.
 * Pairs with zero difference are excluded. The remaining absolute differences
 * are ranked, and the test statistic W is the smaller of:
 *   - W+: sum of ranks where difference > 0
 *   - W-: sum of ranks where difference < 0
 *
 * Uses the normal approximation with continuity correction:
 *   z = (|W - mean(W)| - 0.5) / sqrt(Var(W))
 *
 * where mean(W) = n*(n+1)/4 and the variance includes a tie correction:
 *   Var(W) = n*(n+1)*(2n+1)/24 - sum(t^3 - t)/48
 *
 * @param data1 - First paired sample.
 * @param data2 - Second paired sample (must have the same length as data1).
 * @param alpha - Significance level for the hypothesis test (default 0.05).
 * @returns A {@link HypothesisTestResult} where:
 *   - `statistic` is W (min of W+, W-)
 *   - `pValue` is the two-tailed p-value from the normal approximation
 *   - `degreesOfFreedom` is 0 (non-parametric test)
 *   - `rejected` is true if pValue < alpha
 * @throws {Error} If paired samples have different lengths.
 * @throws {Error} If fewer than 2 paired observations are provided.
 *
 * @example
 * ```ts
 * const before = [125, 115, 130, 140, 140, 115, 140, 125, 140, 135];
 * const after  = [110, 122, 125, 120, 140, 124, 123, 137, 135, 145];
 * const result = wilcoxonSignedRank(before, after);
 * result.pValue; // two-tailed p-value for H0: median difference = 0
 * ```
 */
export function wilcoxonSignedRank(
  data1: Dataset,
  data2: Dataset,
  alpha: number = 0.05,
): HypothesisTestResult {
  if (data1.length !== data2.length) {
    throw new Error(`Invalid parameter 'data1'/'data2': expected equal length for paired samples, received data1.length=${data1.length}, data2.length=${data2.length}`);
  }
  if (data1.length < 2) {
    throw new Error(`Invalid parameter 'data1'/'data2': expected at least 2 paired observations, received ${data1.length}`);
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

/**
 * Assigns 1-based ranks with tie averaging to a pre-sorted array.
 *
 * When multiple values are equal (ties), each tied element receives the
 * average of the ranks they would have occupied.
 *
 * @param sorted - A numeric array that must already be sorted in ascending order.
 * @returns An array of ranks with the same length as the input.
 */
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

/**
 * Counts the sizes of tie groups in a pre-sorted array.
 *
 * Only groups of size > 1 (i.e., actual ties) are included in the result.
 * Used for computing tie correction factors in rank-based tests.
 *
 * @param sorted - A numeric array that must already be sorted in ascending order.
 * @returns An array of tie group sizes (each > 1).
 */
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

/**
 * Result of a Kruskal-Wallis test.
 */
export interface KruskalWallisResult {
  /** The H test statistic (chi-squared approximation). */
  statistic: number;
  /** Approximate p-value from the chi-squared distribution. */
  pValue: number;
  /** Degrees of freedom (k - 1, where k is the number of groups). */
  df: number;
  /** Whether to reject the null hypothesis at the given alpha. */
  rejected: boolean;
}

/**
 * Chi-squared survival function: P(chi-squared > x) with given degrees of freedom.
 */
function chiSquaredSf(x: number, df: number): number {
  if (x <= 0) return 1;
  return 1 - regularizedGammaP(df / 2, x / 2);
}

/**
 * Performs the Kruskal-Wallis H test for comparing multiple independent samples.
 *
 * A non-parametric alternative to one-way ANOVA that tests whether two or
 * more independent samples come from the same distribution. The test ranks
 * all observations together and compares the mean ranks across groups.
 *
 * The H statistic follows a chi-squared distribution with (k - 1) degrees
 * of freedom under the null hypothesis, where k is the number of groups.
 * A tie correction factor is applied when ties are present.
 *
 * @param groups - Array of datasets, one per group (at least 2 groups required;
 *   each group must have at least 1 observation)
 * @param alpha - Significance level for the test (default 0.05)
 * @returns A {@link KruskalWallisResult} containing the H statistic, p-value,
 *   degrees of freedom, and whether the null hypothesis is rejected
 * @throws {Error} If fewer than 2 groups are provided
 * @throws {Error} If any group is empty
 * @throws {Error} If total observations are fewer than 3
 *
 * @example
 * ```ts
 * const result = kruskalWallisTest([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
 * console.log(result.statistic, result.pValue, result.df);
 * ```
 */
export function kruskalWallisTest(
  groups: Dataset[],
  alpha: number = 0.05,
): KruskalWallisResult {
  const k = groups.length;
  if (k < 2) {
    throw new Error(`Invalid parameter 'groups': expected at least 2 groups, received ${k}`);
  }
  for (let i = 0; i < k; i++) {
    if (groups[i].length < 1) {
      throw new Error(
        `Invalid parameter 'groups': expected at least 1 observation in group ${i}, received ${groups[i].length}`,
      );
    }
  }

  const N = groups.reduce((sum, g) => sum + g.length, 0);
  if (N < 3) {
    throw new Error(`Invalid parameter 'groups': expected at least 3 total observations, received ${N}`);
  }

  // Combine all observations with group labels
  const combined: { value: number; group: number }[] = [];
  for (let gi = 0; gi < k; gi++) {
    for (const v of groups[gi]) {
      combined.push({ value: v, group: gi });
    }
  }
  combined.sort((a, b) => a.value - b.value);

  // Assign ranks with tie averaging
  const values = combined.map((c) => c.value);
  const ranks = assignRanks(values);

  // Sum of ranks per group
  const rankSums = new Array<number>(k).fill(0);
  for (let i = 0; i < combined.length; i++) {
    rankSums[combined[i].group] += ranks[i];
  }

  // Compute H statistic
  // H = (12 / (N*(N+1))) * sum(R_i^2 / n_i) - 3*(N+1)
  let sumTerm = 0;
  for (let i = 0; i < k; i++) {
    const ni = groups[i].length;
    sumTerm += (rankSums[i] * rankSums[i]) / ni;
  }
  let H = (12 / (N * (N + 1))) * sumTerm - 3 * (N + 1);

  // Tie correction
  const tieGroups = countTieGroups(values);
  if (tieGroups.length > 0) {
    let tieCorrection = 0;
    for (const t of tieGroups) {
      tieCorrection += t * t * t - t;
    }
    const correctionFactor = 1 - tieCorrection / (N * N * N - N);
    if (correctionFactor > 0) {
      H = H / correctionFactor;
    }
  }

  const df = k - 1;
  const pValue = chiSquaredSf(H, df);

  return {
    statistic: H,
    pValue,
    df,
    rejected: pValue < alpha,
  };
}
