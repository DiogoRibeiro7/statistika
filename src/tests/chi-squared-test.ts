import { HypothesisTestResult } from "../types";
import { regularizedGammaP } from "../utils/math";

/**
 * Upper-tail p-value for chi-squared: P(X >= x) = 1 - P(s, x/2)
 * where s = df/2.
 */
function chiSqPValue(x: number, df: number): number {
  return 1 - regularizedGammaP(df / 2, x / 2);
}

/**
 * Performs a chi-squared goodness-of-fit test.
 *
 * Tests the null hypothesis that the observed frequency distribution
 * matches the expected frequency distribution. The test statistic is
 * sum((O_i - E_i)^2 / E_i), which follows a chi-squared distribution
 * with (k - 1) degrees of freedom under the null.
 *
 * @param observed - Array of observed frequencies (at least 2 categories)
 * @param expected - Array of expected frequencies (must be positive; same length as `observed`)
 * @param alpha - Significance level for the test (default 0.05)
 * @returns A {@link HypothesisTestResult} containing the chi-squared statistic,
 *   upper-tail p-value, degrees of freedom (k - 1), and whether the null is rejected
 * @throws {Error} If `observed` and `expected` have different lengths
 * @throws {Error} If fewer than 2 categories are provided
 * @throws {Error} If any expected frequency is non-positive
 *
 * @example
 * ```ts
 * const result = chiSquaredGoodnessOfFit([50, 30, 20], [40, 40, 20]);
 * console.log(result.pValue, result.rejected);
 * ```
 */
export function chiSquaredGoodnessOfFit(
  observed: number[],
  expected: number[],
  alpha: number = 0.05,
): HypothesisTestResult {
  if (observed.length !== expected.length) {
    throw new Error(`Invalid parameter 'expected': expected length ${observed.length} to match observed, received length ${expected.length}`);
  }
  if (observed.length < 2) throw new Error(`Invalid parameter 'observed': expected at least 2 categories, received ${observed.length}`);

  let stat = 0;
  for (let i = 0; i < observed.length; i++) {
    if (expected[i] <= 0) throw new Error(`Invalid parameter 'expected': expected positive frequency at index ${i}, received ${expected[i]}`);
    stat += (observed[i] - expected[i]) ** 2 / expected[i];
  }

  const df = observed.length - 1;
  const pValue = chiSqPValue(stat, df);
  return { statistic: stat, pValue, degreesOfFreedom: df, rejected: pValue < alpha };
}

/**
 * Performs a chi-squared test of independence on a contingency table.
 *
 * Tests the null hypothesis that the row and column variables are
 * independent. Expected frequencies are computed from marginal totals
 * as E_ij = (row_i_total * col_j_total) / grand_total.
 *
 * @param table - A 2D contingency table where `table[i][j]` is the count
 *   for row i, column j (at least 2 rows and 2 columns required)
 * @param alpha - Significance level for the test (default 0.05)
 * @returns A {@link HypothesisTestResult} containing the chi-squared statistic,
 *   upper-tail p-value, degrees of freedom ((rows - 1) * (cols - 1)),
 *   and whether the null hypothesis is rejected
 * @throws {Error} If the table has fewer than 2 rows or 2 columns
 * @throws {Error} If the grand total is zero
 *
 * @example
 * ```ts
 * const table = [[10, 20, 30], [6, 9, 17]];
 * const result = chiSquaredIndependence(table);
 * console.log(result.statistic, result.pValue);
 * ```
 */
export function chiSquaredIndependence(
  table: number[][],
  alpha: number = 0.05,
): HypothesisTestResult {
  const nRows = table.length;
  if (nRows < 2) throw new Error(`Invalid parameter 'table': expected at least 2 rows, received ${nRows}`);
  const nCols = table[0].length;
  if (nCols < 2) throw new Error(`Invalid parameter 'table': expected at least 2 columns, received ${nCols}`);

  // Row totals, column totals, grand total
  const rowTotals = table.map((row) => row.reduce((a, b) => a + b, 0));
  const colTotals = new Array(nCols).fill(0);
  for (let j = 0; j < nCols; j++) {
    for (let i = 0; i < nRows; i++) {
      colTotals[j] += table[i][j];
    }
  }
  const grandTotal = rowTotals.reduce((a, b) => a + b, 0);
  if (grandTotal === 0) throw new Error(`Invalid parameter 'table': expected positive total, received ${grandTotal}`);

  let stat = 0;
  for (let i = 0; i < nRows; i++) {
    for (let j = 0; j < nCols; j++) {
      const expected = (rowTotals[i] * colTotals[j]) / grandTotal;
      if (expected > 0) {
        stat += (table[i][j] - expected) ** 2 / expected;
      }
    }
  }

  const df = (nRows - 1) * (nCols - 1);
  const pValue = chiSqPValue(stat, df);
  return { statistic: stat, pValue, degreesOfFreedom: df, rejected: pValue < alpha };
}
