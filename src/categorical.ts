import { Normal } from "./distributions/continuous/normal";

/**
 * Contingency table utilities and categorical data analysis.
 */

export interface ContingencyTableSummary {
  /** The table of observed counts. */
  observed: number[][];
  /** Expected counts under independence. */
  expected: number[][];
  /** Row marginal totals. */
  rowTotals: number[];
  /** Column marginal totals. */
  colTotals: number[];
  /** Grand total. */
  grandTotal: number;
  /** Number of rows. */
  nRows: number;
  /** Number of columns. */
  nCols: number;
}

/**
 * Analyze a contingency table: compute margins and expected counts.
 */
export function contingencyTable(observed: number[][]): ContingencyTableSummary {
  const nRows = observed.length;
  if (nRows === 0) throw new Error("Table must not be empty");
  const nCols = observed[0].length;
  if (nCols === 0) throw new Error("Table must have at least 1 column");

  const rowTotals = observed.map((row) => row.reduce((a, b) => a + b, 0));
  const colTotals = new Array<number>(nCols).fill(0);
  for (let j = 0; j < nCols; j++) {
    for (let i = 0; i < nRows; i++) {
      colTotals[j] += observed[i][j];
    }
  }
  const grandTotal = rowTotals.reduce((a, b) => a + b, 0);

  const expected = Array.from({ length: nRows }, (_, i) =>
    Array.from({ length: nCols }, (_, j) => (rowTotals[i] * colTotals[j]) / grandTotal),
  );

  return { observed, expected, rowTotals, colTotals, grandTotal, nRows, nCols };
}

/**
 * McNemar's test for paired nominal data (2x2 table).
 *
 * Tests the null hypothesis that the marginal proportions are equal.
 * Used for before/after studies or matched-pair designs.
 *
 * @param table - 2x2 contingency table [[a,b],[c,d]]
 * @param alpha - Significance level (default: 0.05)
 */
export function mcnemarsTest(
  table: number[][],
  alpha = 0.05,
): { statistic: number; pValue: number; rejected: boolean } {
  if (table.length !== 2 || table[0].length !== 2 || table[1].length !== 2) {
    throw new Error("McNemar's test requires a 2×2 table");
  }

  const b = table[0][1]; // discordant pair 1
  const c = table[1][0]; // discordant pair 2

  if (b + c === 0) {
    return { statistic: 0, pValue: 1, rejected: false };
  }

  // Chi-squared version (with continuity correction)
  const statistic = (Math.abs(b - c) - 1) ** 2 / (b + c);
  // p-value from chi-squared(1) approximation
  const normal = new Normal();
  const pValue = 2 * (1 - normal.cdf(Math.sqrt(statistic)));

  return { statistic, pValue, rejected: pValue < alpha };
}

/**
 * Cochran-Mantel-Haenszel test for stratified 2x2 tables.
 *
 * Tests for a common odds ratio across multiple strata.
 *
 * @param tables - Array of 2×2 contingency tables, one per stratum
 * @param alpha - Significance level (default: 0.05)
 */
export function cochranMantelHaenszel(
  tables: number[][][],
  alpha = 0.05,
): { statistic: number; pValue: number; commonOddsRatio: number; rejected: boolean } {
  if (tables.length === 0) throw new Error("Must provide at least one table");

  for (const table of tables) {
    if (table.length !== 2 || table[0].length !== 2 || table[1].length !== 2) {
      throw new Error("All tables must be 2×2");
    }
  }

  let numerator = 0;
  let denominator = 0;
  let orNum = 0;
  let orDen = 0;

  for (const t of tables) {
    const a = t[0][0], b = t[0][1], c = t[1][0], d = t[1][1];
    const n = a + b + c + d;

    numerator += a - ((a + b) * (a + c)) / n;
    denominator +=
      ((a + b) * (c + d) * (a + c) * (b + d)) / (n * n * (n - 1));

    // Mantel-Haenszel odds ratio components
    orNum += (a * d) / n;
    orDen += (b * c) / n;
  }

  const statistic = numerator ** 2 / denominator;
  const normal = new Normal();
  const pValue = 2 * (1 - normal.cdf(Math.sqrt(statistic)));
  const commonOddsRatio = orDen > 0 ? orNum / orDen : Infinity;

  return { statistic, pValue, commonOddsRatio, rejected: pValue < alpha };
}

/**
 * G-test (log-likelihood ratio test) for contingency tables.
 *
 * An alternative to the chi-squared test that uses the log-likelihood
 * ratio statistic: G = 2 * Σ O * ln(O/E).
 *
 * @param observed - Contingency table of observed counts
 * @param alpha - Significance level (default: 0.05)
 */
export function gTest(
  observed: number[][],
  alpha = 0.05,
): { statistic: number; pValue: number; degreesOfFreedom: number; rejected: boolean } {
  const ct = contingencyTable(observed);
  const df = (ct.nRows - 1) * (ct.nCols - 1);

  let g = 0;
  for (let i = 0; i < ct.nRows; i++) {
    for (let j = 0; j < ct.nCols; j++) {
      if (observed[i][j] > 0 && ct.expected[i][j] > 0) {
        g += 2 * observed[i][j] * Math.log(observed[i][j] / ct.expected[i][j]);
      }
    }
  }

  // Approximate p-value using normal for df=1, chi-squared approximation for larger
  const pValue = chiSquaredSurvival(g, df);

  return { statistic: g, pValue, degreesOfFreedom: df, rejected: pValue < alpha };
}

/**
 * Standardized residuals for a contingency table.
 *
 * Identifies which cells deviate most from expected. Values > 2 or < -2
 * indicate significant deviation.
 */
export function standardizedResiduals(observed: number[][]): number[][] {
  const ct = contingencyTable(observed);
  return ct.observed.map((row, i) =>
    row.map((o, j) => {
      const e = ct.expected[i][j];
      if (e === 0) return 0;
      return (o - e) / Math.sqrt(e);
    }),
  );
}

/**
 * Adjusted standardized residuals (Haberman residuals).
 *
 * Accounts for the margins, making them approximately standard normal
 * under independence.
 */
export function adjustedResiduals(observed: number[][]): number[][] {
  const ct = contingencyTable(observed);
  return ct.observed.map((row, i) =>
    row.map((o, j) => {
      const e = ct.expected[i][j];
      if (e === 0) return 0;
      const v =
        e *
        (1 - ct.rowTotals[i] / ct.grandTotal) *
        (1 - ct.colTotals[j] / ct.grandTotal);
      return (o - e) / Math.sqrt(v);
    }),
  );
}

// ── Helpers ─────────────────────────────────────────────────────────────

function chiSquaredSurvival(x: number, df: number): number {
  // Use regularized incomplete gamma function approximation
  // For df=1, use normal approximation
  if (df === 1) {
    const normal = new Normal();
    return 2 * (1 - normal.cdf(Math.sqrt(x)));
  }
  // Wilson-Hilferty approximation for chi-squared CDF
  const z = Math.pow(x / df, 1 / 3) - (1 - 2 / (9 * df));
  const se = Math.sqrt(2 / (9 * df));
  const normal = new Normal();
  return 1 - normal.cdf(z / se);
}
