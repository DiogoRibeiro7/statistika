import { Dataset, CorrelationResult } from "../types";
import { mean } from "../utils/descriptive";
import { regularizedBeta } from "../utils/math";
import { normalCdf } from "../utils/linalg";

/**
 * Computes the Pearson product-moment correlation coefficient between two datasets
 * with a two-tailed p-value.
 *
 * The coefficient measures the linear relationship between `x` and `y`:
 *
 *   r = sum((xi - x_mean)(yi - y_mean)) / sqrt(sum((xi - x_mean)^2) * sum((yi - y_mean)^2))
 *
 * The p-value is derived from the t-distribution approximation:
 *   t = r * sqrt((n - 2) / (1 - r^2)), with df = n - 2.
 *
 * @param x - First dataset (numeric array of length n).
 * @param y - Second dataset (numeric array of length n).
 * @returns A {@link CorrelationResult} containing the correlation coefficient in [-1, 1]
 *   and the two-tailed p-value. Returns `{ coefficient: NaN, pValue: NaN }` if either
 *   dataset has zero variance.
 * @throws {Error} If `x` and `y` have different lengths.
 * @throws {Error} If the datasets have fewer than 3 elements.
 *
 * @example
 * ```ts
 * const result = pearsonCorrelation([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]);
 * // result.coefficient === 1.0 (perfect positive linear relationship)
 * ```
 */
export function pearsonCorrelation(x: Dataset, y: Dataset): CorrelationResult {
  validateInputs(x, y);
  const n = x.length;
  const xMean = mean(x);
  const yMean = mean(y);

  let ssXY = 0;
  let ssXX = 0;
  let ssYY = 0;

  for (let i = 0; i < n; i++) {
    const dx = x[i] - xMean;
    const dy = y[i] - yMean;
    ssXY += dx * dy;
    ssXX += dx * dx;
    ssYY += dy * dy;
  }

  if (ssXX === 0 || ssYY === 0) {
    return { coefficient: NaN, pValue: NaN };
  }

  const r = ssXY / Math.sqrt(ssXX * ssYY);
  const pValue = correlationPValue(r, n);

  return { coefficient: r, pValue };
}

/**
 * Computes the Spearman rank correlation coefficient between two datasets
 * with a two-tailed p-value.
 *
 * Transforms both datasets to their ranks (with tie averaging) and then
 * computes the Pearson correlation on the ranks. This measures the monotonic
 * (not necessarily linear) relationship between the variables.
 *
 * @param x - First dataset (numeric array of length n).
 * @param y - Second dataset (numeric array of length n).
 * @returns A {@link CorrelationResult} containing the rank correlation coefficient
 *   in [-1, 1] and the two-tailed p-value.
 * @throws {Error} If `x` and `y` have different lengths.
 * @throws {Error} If the datasets have fewer than 3 elements.
 *
 * @example
 * ```ts
 * const result = spearmanCorrelation([1, 2, 3, 4, 5], [5, 6, 7, 8, 7]);
 * // result.coefficient is the rank-based correlation
 * ```
 */
export function spearmanCorrelation(
  x: Dataset,
  y: Dataset,
): CorrelationResult {
  validateInputs(x, y);

  const rankX = computeRanks(x);
  const rankY = computeRanks(y);

  return pearsonCorrelation(rankX, rankY);
}

/**
 * Computes Kendall's tau-b rank correlation coefficient between two datasets
 * with a two-tailed p-value.
 *
 * Tau-b adjusts for ties using the formula:
 *
 *   tau_b = (concordant - discordant) / sqrt((n0 - n1) * (n0 - n2))
 *
 * where n0 = n(n-1)/2 (total pairs), n1 = tied pairs in x, n2 = tied pairs in y.
 *
 * The p-value is computed via a normal approximation (valid for n >= 10):
 *   z = tau / sqrt(2(2n + 5) / (9n(n - 1))).
 *
 * @param x - First dataset (numeric array of length n).
 * @param y - Second dataset (numeric array of length n).
 * @returns A {@link CorrelationResult} containing the tau-b coefficient in [-1, 1]
 *   and the two-tailed p-value. Returns `{ coefficient: NaN, pValue: NaN }` if
 *   all pairs are tied in one or both variables.
 * @throws {Error} If `x` and `y` have different lengths.
 * @throws {Error} If the datasets have fewer than 3 elements.
 *
 * @example
 * ```ts
 * const result = kendallCorrelation([1, 2, 3, 4], [1, 2, 3, 4]);
 * // result.coefficient === 1.0 (perfect concordance)
 * ```
 */
export function kendallCorrelation(
  x: Dataset,
  y: Dataset,
): CorrelationResult {
  validateInputs(x, y);
  const n = x.length;

  let concordant = 0;
  let discordant = 0;
  let tiedX = 0;
  let tiedY = 0;

  for (let i = 0; i < n - 1; i++) {
    for (let j = i + 1; j < n; j++) {
      const dx = Math.sign(x[j] - x[i]);
      const dy = Math.sign(y[j] - y[i]);

      if (dx === 0 && dy === 0) {
        tiedX++;
        tiedY++;
      } else if (dx === 0) {
        tiedX++;
      } else if (dy === 0) {
        tiedY++;
      } else if (dx === dy) {
        concordant++;
      } else {
        discordant++;
      }
    }
  }

  const totalPairs = (n * (n - 1)) / 2;
  const denom = Math.sqrt(
    (totalPairs - tiedX) * (totalPairs - tiedY),
  );

  if (denom === 0) {
    return { coefficient: NaN, pValue: NaN };
  }

  const tau = (concordant - discordant) / denom;

  // Normal approximation for p-value (valid for n >= 10)
  const variance = (2 * (2 * n + 5)) / (9 * n * (n - 1));
  const z = tau / Math.sqrt(variance);
  const pValue = 2 * (1 - normalCdf(Math.abs(z)));

  return { coefficient: tau, pValue };
}

/**
 * Validates that two datasets have the same length and at least 3 elements.
 *
 * @param x - First dataset.
 * @param y - Second dataset.
 * @throws {Error} If the datasets differ in length or have fewer than 3 elements.
 */
function validateInputs(x: Dataset, y: Dataset): void {
  if (x.length !== y.length) {
    throw new Error(`Invalid parameters 'x', 'y': expected same length, received x.length=${x.length}, y.length=${y.length}`);
  }
  if (x.length < 3) {
    throw new Error(`Invalid parameter 'x': expected at least 3 elements, received ${x.length}`);
  }
}

/**
 * Computes 1-based ranks for the given data with tie averaging.
 *
 * When multiple values are equal (ties), each tied element receives the
 * average of the ranks they would have occupied.
 *
 * @param data - Numeric array to rank.
 * @returns An array of ranks in the same order as the input data.
 */
function computeRanks(data: Dataset): number[] {
  const indexed = data.map((value, index) => ({ value, index }));
  indexed.sort((a, b) => a.value - b.value);

  const ranks = new Array(data.length);
  let i = 0;
  while (i < indexed.length) {
    let j = i;
    while (j < indexed.length && indexed[j].value === indexed[i].value) {
      j++;
    }
    // Average rank for tied values
    const avgRank = (i + j + 1) / 2; // 1-based average
    for (let k = i; k < j; k++) {
      ranks[indexed[k].index] = avgRank;
    }
    i = j;
  }

  return ranks;
}

/**
 * Computes a two-tailed p-value from a Pearson correlation coefficient
 * using the t-distribution via the regularized incomplete beta function.
 *
 * Transforms r to t^2 = r^2 * df / (1 - r^2) where df = n - 2, then
 * computes P(T^2 > t^2) = I_{df/(df+t^2)}(df/2, 1/2).
 *
 * @param r - Pearson correlation coefficient.
 * @param n - Sample size.
 * @returns The two-tailed p-value. Returns 0 if |r| >= 1.
 */
function correlationPValue(r: number, n: number): number {
  if (Math.abs(r) >= 1) return 0;
  const df = n - 2;
  const t2 = (r * r * df) / (1 - r * r);
  // P(T² > t²) = 1 - I_{df/(df+t²)}(df/2, 1/2) using regularized beta
  const x = df / (df + t2);
  return regularizedBeta(x, df / 2, 0.5);
}

