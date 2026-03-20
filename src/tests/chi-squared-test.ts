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
 * Chi-squared goodness-of-fit test.
 * Tests whether observed frequencies match expected frequencies.
 */
export function chiSquaredGoodnessOfFit(
  observed: number[],
  expected: number[],
  alpha: number = 0.05,
): HypothesisTestResult {
  if (observed.length !== expected.length) {
    throw new Error("observed and expected must have the same length");
  }
  if (observed.length < 2) throw new Error("Need at least 2 categories");

  let stat = 0;
  for (let i = 0; i < observed.length; i++) {
    if (expected[i] <= 0) throw new Error("Expected frequencies must be positive");
    stat += (observed[i] - expected[i]) ** 2 / expected[i];
  }

  const df = observed.length - 1;
  const pValue = chiSqPValue(stat, df);
  return { statistic: stat, pValue, degreesOfFreedom: df, rejected: pValue < alpha };
}

/**
 * Chi-squared test of independence on a contingency table.
 * Input: 2D array where table[i][j] is the count for row i, column j.
 */
export function chiSquaredIndependence(
  table: number[][],
  alpha: number = 0.05,
): HypothesisTestResult {
  const nRows = table.length;
  if (nRows < 2) throw new Error("Table must have at least 2 rows");
  const nCols = table[0].length;
  if (nCols < 2) throw new Error("Table must have at least 2 columns");

  // Row totals, column totals, grand total
  const rowTotals = table.map((row) => row.reduce((a, b) => a + b, 0));
  const colTotals = new Array(nCols).fill(0);
  for (let j = 0; j < nCols; j++) {
    for (let i = 0; i < nRows; i++) {
      colTotals[j] += table[i][j];
    }
  }
  const grandTotal = rowTotals.reduce((a, b) => a + b, 0);
  if (grandTotal === 0) throw new Error("Table total must be positive");

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
