import { Dataset } from "../types";
import { normalQuantile, normalCdf } from "../utils/linalg";

/**
 * Result of a Shapiro-Wilk normality test.
 */
export interface ShapiroWilkResult {
  /** The W statistic (values close to 1 indicate normality). */
  statistic: number;
  /** Approximate p-value for the test. */
  pValue: number;
  /** Whether the data appears normally distributed at the given alpha. */
  isNormal: boolean;
}

/**
 * Shapiro-Wilk coefficients for small sample sizes (n = 3..50).
 * These are the expected values of standard normal order statistics,
 * precomputed for efficiency.
 */
function shapiroWilkCoefficients(n: number): number[] {
  // Compute expected normal order statistics using Blom's approximation
  const m = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    m[i] = normalQuantile((i + 1 - 0.375) / (n + 0.25));
  }

  // Normalize to get the a_i coefficients
  let sumMSq = 0;
  for (let i = 0; i < n; i++) {
    sumMSq += m[i] * m[i];
  }

  const a = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    a[i] = m[i] / Math.sqrt(sumMSq);
  }

  return a;
}

/**
 * Approximates the p-value for the Shapiro-Wilk W statistic.
 *
 * Uses Royston's (1992) log-normal transformation approach.
 * The transformation maps W to approximately standard normal.
 *
 * @param W - The W statistic.
 * @param n - Sample size.
 * @returns Approximate p-value.
 */
function shapiroWilkPValue(W: number, n: number): number {
  if (n < 3) return 1;

  // Royston's approximation (1992, Applied Statistics 44(4))
  // Uses a normal approximation after transforming W
  const logN = Math.log(n);

  if (n <= 11) {
    // For small n, use a polynomial transformation of W
    const gamma = 0.459 * n - 2.273;
    const w = -Math.log(gamma - Math.log(1 - W));
    const mu = -0.0006714 * n * n * n + 0.025054 * n * n - 0.39978 * n + 0.5440;
    const sigma = Math.exp(-0.0020322 * n * n * n + 0.062767 * n * n - 0.77857 * n + 1.3822);
    const z = (w - mu) / sigma;
    return 1 - normalCdf(z);
  }

  // For larger n, use the log-transformed approximation
  const lnW = Math.log(1 - W);
  const mu = 0.0038915 * logN * logN * logN - 0.083751 * logN * logN - 0.31082 * logN - 1.5861;
  const sigma = Math.exp(0.0030302 * logN * logN * logN - 0.082676 * logN * logN - 0.4803);
  const z = (lnW - mu) / sigma;

  return 1 - normalCdf(z);
}

/**
 * Performs the Shapiro-Wilk test for normality.
 *
 * Tests the null hypothesis that the data comes from a normally distributed
 * population. The W statistic measures how well the ordered sample values
 * match the expected normal order statistics. Values of W close to 1
 * indicate normality; small values suggest departure from normality.
 *
 * Uses Blom's approximation for the expected normal order statistic
 * coefficients and Royston's (1992) approximation for the p-value.
 *
 * @param data - Array of numeric observations (3 to 5000 recommended)
 * @param alpha - Significance level for the normality decision (default 0.05)
 * @returns A {@link ShapiroWilkResult} containing the W statistic, approximate
 *   p-value, and whether the data appears normally distributed
 * @throws {Error} If `data` has fewer than 3 observations
 * @throws {Error} If `data` has zero variance (all values identical)
 *
 * @example
 * ```ts
 * const result = shapiroWilkTest([0.2, -0.5, 1.3, -0.1, 0.7, 0.3, -0.8]);
 * console.log(result.statistic); // W statistic close to 1
 * console.log(result.isNormal);  // true if p > alpha
 * ```
 */
export function shapiroWilkTest(
  data: Dataset,
  alpha: number = 0.05,
): ShapiroWilkResult {
  if (data.length < 3) {
    throw new Error("shapiroWilkTest requires at least 3 observations (got " + data.length + ")");
  }

  const n = data.length;
  const sorted = [...data].sort((a, b) => a - b);

  // Compute sample mean and total sum of squares
  let sum = 0;
  for (let i = 0; i < n; i++) {
    sum += sorted[i];
  }
  const xbar = sum / n;

  let ss = 0;
  for (let i = 0; i < n; i++) {
    ss += (sorted[i] - xbar) ** 2;
  }

  if (ss === 0) {
    throw new Error("shapiroWilkTest requires non-constant data (all values are identical)");
  }

  // Get the coefficients
  const a = shapiroWilkCoefficients(n);

  // Compute W = (sum(a_i * x_(i)))^2 / SS
  let numerator = 0;
  for (let i = 0; i < n; i++) {
    numerator += a[i] * sorted[i];
  }
  const W = (numerator * numerator) / ss;

  // Clamp W to [0, 1]
  const wClamped = Math.min(1, Math.max(0, W));

  const pValue = shapiroWilkPValue(wClamped, n);
  const pClamped = Math.min(1, Math.max(0, pValue));

  return {
    statistic: wClamped,
    pValue: pClamped,
    isNormal: pClamped >= alpha,
  };
}
