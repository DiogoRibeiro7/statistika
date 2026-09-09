import { normalCdf } from "./utils/linalg";
import { regularizedGammaP } from "./utils/math";

/**
 * Contingency table utilities and categorical data analysis.
 */

/**
 * Summary of a contingency table including observed counts, expected counts
 * under independence, marginal totals, and table dimensions.
 *
 * Expected counts are computed as E_ij = (row_i_total * col_j_total) / grand_total,
 * representing the counts expected if rows and columns were independent.
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
 *
 * @param observed - 2D array of observed cell counts
 * @returns ContingencyTableSummary with observed, expected, marginals, and dimensions
 * @throws Error if the table is empty or has no columns
 * @throws Error if rows have inconsistent column counts
 * @throws Error if any cell count is negative or NaN
 *
 * @example
 * ```ts
 * const ct = contingencyTable([[10, 20], [30, 40]]);
 * // ct.expected[0][0] === (30 * 40) / 100 === 12
 * // ct.grandTotal === 100
 * ```
 */
export function contingencyTable(observed: number[][]): ContingencyTableSummary {
  const nRows = observed.length;
  if (nRows === 0) throw new Error(`Invalid parameter 'observed': expected a non-empty table, received 0 rows`);
  const nCols = observed[0].length;
  if (nCols === 0) throw new Error(`Invalid parameter 'observed': expected at least 1 column, received 0 columns`);

  for (let i = 0; i < nRows; i++) {
    if (observed[i].length !== nCols) {
      throw new Error(
        `Invalid parameter 'observed': expected ${nCols} columns at row ${i}, received ${observed[i].length}`,
      );
    }
    for (let j = 0; j < nCols; j++) {
      const v = observed[i][j];
      if (Number.isNaN(v)) {
        throw new Error(`Invalid parameter 'observed': expected finite number at [${i}][${j}], received NaN`);
      }
      if (v < 0) {
        throw new Error(`Invalid parameter 'observed': expected non-negative count at [${i}][${j}], received ${v}`);
      }
    }
  }

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
 * @returns Object with test statistic, p-value, and rejection decision
 * @throws Error if the table is not exactly 2x2
 *
 * @example
 * ```ts
 * const result = mcnemarsTest([[20, 5], [10, 15]]);
 * console.log(result.pValue); // p-value for marginal homogeneity
 * ```
 */
export function mcnemarsTest(
  table: number[][],
  alpha = 0.05,
): { statistic: number; pValue: number; rejected: boolean } {
  if (table.length !== 2 || table[0].length !== 2 || table[1].length !== 2) {
    throw new Error(`Invalid parameter 'table': expected a 2x2 table, received ${table.length}x${table[0]?.length ?? 0}`);
  }

  const b = table[0][1]; // discordant pair 1
  const c = table[1][0]; // discordant pair 2

  if (b + c === 0) {
    return { statistic: 0, pValue: 1, rejected: false };
  }

  // Chi-squared version (with continuity correction)
  const statistic = (Math.abs(b - c) - 1) ** 2 / (b + c);
  // p-value from chi-squared(1) survival function
  const pValue = chiSquaredSurvival(statistic, 1);

  return { statistic, pValue, rejected: pValue < alpha };
}

/**
 * Cochran-Mantel-Haenszel test for stratified 2x2 tables.
 *
 * Tests for a common odds ratio across multiple strata.
 *
 * @param tables - Array of 2x2 contingency tables, one per stratum
 * @param alpha - Significance level (default: 0.05)
 * @returns Object with test statistic, p-value, common odds ratio, and rejection decision
 * @throws Error if no tables are provided
 * @throws Error if any table is not exactly 2x2
 *
 * @example
 * ```ts
 * const result = cochranMantelHaenszel([[[10, 5], [3, 12]], [[8, 6], [4, 10]]]);
 * console.log(result.commonOddsRatio); // pooled odds ratio across strata
 * ```
 */
export function cochranMantelHaenszel(
  tables: number[][][],
  alpha = 0.05,
): { statistic: number; pValue: number; commonOddsRatio: number; rejected: boolean } {
  if (tables.length === 0) throw new Error(`Invalid parameter 'tables': expected at least one table, received length 0`);

  for (const table of tables) {
    if (table.length !== 2 || table[0].length !== 2 || table[1].length !== 2) {
      throw new Error(`Invalid parameter 'tables': expected all tables to be 2x2, received ${table.length}×${table[0]?.length} table`);
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
  const pValue = chiSquaredSurvival(statistic, 1);
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
 * @returns Object with G statistic, p-value, degrees of freedom, and rejection decision
 * @throws Error if the table is empty, has inconsistent row lengths, or contains negative counts
 *
 * @example
 * ```ts
 * const result = gTest([[10, 20], [30, 40]]);
 * console.log(result.statistic); // G statistic
 * console.log(result.pValue);    // p-value
 * ```
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

  const pValue = chiSquaredSurvival(g, df);

  return { statistic: g, pValue, degreesOfFreedom: df, rejected: pValue < alpha };
}

/**
 * Standardized residuals for a contingency table.
 *
 * Identifies which cells deviate most from expected. Values > 2 or < -2
 * indicate significant deviation.
 *
 * @param observed - Contingency table of observed counts
 * @returns 2D array of standardized residuals (O - E) / sqrt(E)
 * @throws Error if the table is empty, has inconsistent row lengths, or contains negative counts
 *
 * @example
 * ```ts
 * const resid = standardizedResiduals([[10, 20], [30, 40]]);
 * // resid[i][j] > 2 or < -2 indicates significant deviation
 * ```
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
 *
 * @param observed - Contingency table of observed counts
 * @returns 2D array of adjusted standardized residuals
 * @throws Error if the table is empty, has inconsistent row lengths, or contains negative counts
 *
 * @example
 * ```ts
 * const resid = adjustedResiduals([[10, 20], [30, 40]]);
 * // Approximately standard normal under independence
 * ```
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

/**
 * Chi-squared survival function: P(X > x) for X ~ chi-squared(df).
 * Uses the regularized lower incomplete gamma function for accuracy.
 */
function chiSquaredSurvival(x: number, df: number): number {
  if (x <= 0) return 1;
  // P(X <= x) = regularizedGammaP(df/2, x/2), so survival = 1 - P
  return 1 - regularizedGammaP(df / 2, x / 2);
}
