import { Dataset } from "../types";
import { mean, variance } from "../utils/descriptive";
import { normalCdf } from "../utils/linalg";

/**
 * Result of an Anderson-Darling goodness-of-fit test.
 */
export interface AndersonDarlingResult {
  /** The A-squared statistic. */
  statistic: number;
  /** Approximate p-value for the test. */
  pValue: number;
  /** Whether to reject the null hypothesis at the given alpha. */
  rejectNull: boolean;
}

/**
 * Computes the approximate p-value for the Anderson-Darling test
 * against a normal distribution using the modified statistic A*².
 *
 * Uses the approximation from D'Agostino and Stephens (1986),
 * "Goodness-of-Fit Techniques", Marcel Dekker.
 *
 * @param aSqStar - The modified A-squared statistic A*(1 + 0.75/n + 2.25/n²).
 * @returns Approximate p-value.
 */
function andersonDarlingPValue(aSqStar: number): number {
  // Approximation from Marsaglia & Marsaglia (2004) and
  // D'Agostino & Stephens (1986) for the normal case.
  if (aSqStar <= 0.2) {
    return 1 - Math.exp(-13.436 + 101.14 * aSqStar - 223.73 * aSqStar * aSqStar);
  }
  if (aSqStar <= 0.34) {
    return 1 - Math.exp(-8.318 + 42.796 * aSqStar - 59.938 * aSqStar * aSqStar);
  }
  if (aSqStar <= 0.6) {
    return Math.exp(0.9177 - 4.279 * aSqStar - 1.38 * aSqStar * aSqStar);
  }
  if (aSqStar <= 13) {
    return Math.exp(1.2937 - 5.709 * aSqStar + 0.0186 * aSqStar * aSqStar);
  }
  return 0;
}

/**
 * Performs the Anderson-Darling test for normality (goodness-of-fit).
 *
 * Tests the null hypothesis that the data comes from a normal distribution.
 * The data is standardized using its sample mean and standard deviation
 * before computing the test statistic. The A-squared statistic measures the
 * weighted squared distance between the empirical CDF and the normal CDF,
 * giving more weight to the tails of the distribution.
 *
 * The p-value is computed using the modified statistic A*² with a
 * sample-size correction factor, following D'Agostino and Stephens (1986).
 *
 * @param data - Array of numeric observations (at least 7 recommended for reliable p-values)
 * @param alpha - Significance level for the rejection decision (default 0.05)
 * @returns An {@link AndersonDarlingResult} containing the A² statistic,
 *   approximate p-value, and whether the null hypothesis is rejected
 * @throws {Error} If `data` has fewer than 2 observations
 * @throws {Error} If `data` has zero variance (all values identical)
 *
 * @example
 * ```ts
 * const result = andersonDarlingTest([0.1, -0.3, 0.5, 0.2, -0.1, 0.4, -0.2]);
 * console.log(result.statistic); // A² value
 * console.log(result.rejectNull); // false if data looks normal
 * ```
 */
export function andersonDarlingTest(
  data: Dataset,
  alpha: number = 0.05,
): AndersonDarlingResult {
  if (data.length < 2) {
    throw new Error("andersonDarlingTest requires at least 2 observations (got " + data.length + ")");
  }

  const n = data.length;
  const m = mean(data);
  const s = Math.sqrt(variance(data, true));

  if (s === 0) {
    throw new Error("andersonDarlingTest requires non-constant data (variance is zero)");
  }

  // Standardize and sort
  const z = data.map((x) => (x - m) / s);
  z.sort((a, b) => a - b);

  // Compute A² statistic
  // A² = -n - (1/n) * sum_{i=1}^{n} (2i - 1) * [ln(F(z_i)) + ln(1 - F(z_{n+1-i}))]
  let sumTerms = 0;
  for (let i = 0; i < n; i++) {
    const fi = normalCdf(z[i]);
    const fni = normalCdf(z[n - 1 - i]);

    // Clamp to avoid log(0)
    const logFi = Math.log(Math.max(fi, 1e-15));
    const logOneMFni = Math.log(Math.max(1 - fni, 1e-15));

    sumTerms += (2 * (i + 1) - 1) * (logFi + logOneMFni);
  }

  const aSq = -n - sumTerms / n;

  // Apply sample size correction: A*² = A² * (1 + 0.75/n + 2.25/n²)
  const aSqStar = aSq * (1 + 0.75 / n + 2.25 / (n * n));

  const pValue = andersonDarlingPValue(aSqStar);
  const pClamped = Math.min(1, Math.max(0, pValue));

  return {
    statistic: aSq,
    pValue: pClamped,
    rejectNull: pClamped < alpha,
  };
}
