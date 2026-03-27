import { HypothesisTestResult } from "../types";
import { logFactorial } from "../utils/math";

/**
 * Performs Fisher's exact test for independence on a 2x2 contingency table (two-tailed).
 *
 * Computes the exact p-value by summing hypergeometric probabilities of all
 * table configurations that are as extreme or more extreme than the observed table.
 * "Extreme" is defined as having probability <= the observed table's probability.
 *
 * The probability of a specific table configuration with cell (1,1) = a is:
 *
 *   P(a) = C(r1, a) * C(r2, c) / C(n, c1)
 *
 * where r1, r2 are row totals, c1, c2 are column totals, and n is the grand total.
 * Computations are done in log-space using log-factorials for numerical stability.
 *
 * @param table - A 2x2 contingency table `[[a, b], [c, d]]` where all entries
 *   must be non-negative integers.
 * @param alpha - Significance level for the hypothesis test (default 0.05).
 * @returns A {@link HypothesisTestResult} where:
 *   - `statistic` is the odds ratio (a*d) / (b*c), or Infinity if b*c = 0
 *   - `pValue` is the exact two-tailed p-value
 *   - `degreesOfFreedom` is 0 (exact test, no chi-squared approximation)
 *   - `rejected` is true if pValue < alpha
 * @throws {Error} If any table entry is negative or not an integer.
 * @throws {Error} If the table has no observations (all entries are 0).
 *
 * @example
 * ```ts
 * // Test whether treatment and outcome are independent
 * const result = fisherExactTest([[1, 9], [11, 3]]);
 * result.pValue;    // exact two-tailed p-value
 * result.statistic; // odds ratio
 * result.rejected;  // true if significant at alpha=0.05
 * ```
 */
export function fisherExactTest(
  table: [[number, number], [number, number]],
  alpha: number = 0.05,
): HypothesisTestResult {
  const [[a, b], [c, d]] = table;

  if ([a, b, c, d].some((v) => v < 0 || !Number.isInteger(v))) {
    throw new Error(`Invalid parameter 'table': expected non-negative integers, received [${a}, ${b}, ${c}, ${d}]`);
  }

  const n = a + b + c + d;
  if (n === 0) {
    throw new Error(`Invalid parameter 'table': expected at least one observation, received total 0`);
  }

  const r1 = a + b; // row 1 total
  const r2 = c + d; // row 2 total
  const c1 = a + c; // col 1 total
  const c2 = b + d; // col 2 total

  // Odds ratio as the test statistic
  const oddsRatio = b * c === 0 ? Infinity : (a * d) / (b * c);

  // Probability of a specific table configuration (hypergeometric)
  // P = C(r1, a) * C(r2, c) / C(n, c1)
  // Using log-space for numerical stability
  const logDenom =
    logFactorial(n) -
    logFactorial(r1) -
    logFactorial(r2) -
    logFactorial(c1) -
    logFactorial(c2);

  function logTableProb(ai: number): number {
    const bi = r1 - ai;
    const ci = c1 - ai;
    const di = r2 - ci;
    if (bi < 0 || ci < 0 || di < 0) return -Infinity;
    return (
      logFactorial(r1) +
      logFactorial(r2) +
      logFactorial(c1) +
      logFactorial(c2) -
      logFactorial(ai) -
      logFactorial(bi) -
      logFactorial(ci) -
      logFactorial(di) -
      logFactorial(n)
    );
  }

  // Two-tailed: sum probabilities of all tables with P <= P_observed
  const observedLogProb = logTableProb(a);
  const observedProb = Math.exp(observedLogProb);

  const minA = Math.max(0, c1 - r2);
  const maxA = Math.min(r1, c1);

  let pValue = 0;
  for (let ai = minA; ai <= maxA; ai++) {
    const prob = Math.exp(logTableProb(ai));
    if (prob <= observedProb + 1e-12) {
      // small tolerance for floating point
      pValue += prob;
    }
  }

  // Clamp to [0, 1]
  pValue = Math.min(1, Math.max(0, pValue));

  return {
    statistic: oddsRatio,
    pValue,
    degreesOfFreedom: 0,
    rejected: pValue < alpha,
  };
}
