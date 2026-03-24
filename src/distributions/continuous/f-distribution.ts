import { BaseContinuous } from "../base";
import { betaFn, regularizedBeta, quantileBisect } from "../../utils/math";
import { RandomFn } from "../../types";

/**
 * F-distribution (Fisher-Snedecor distribution).
 *
 * The F-distribution arises as the ratio of two scaled chi-squared random
 * variables, each divided by its degrees of freedom. It is widely used in
 * analysis of variance (ANOVA) and F-tests for comparing model fits.
 *
 * If X1 ~ Chi-squared(d1) and X2 ~ Chi-squared(d2) are independent, then
 * (X1/d1) / (X2/d2) ~ F(d1, d2).
 *
 * PDF: f(x; d1, d2) = (d1/d2)^(d1/2) * x^(d1/2 - 1) / ((1 + d1*x/d2)^((d1+d2)/2) * B(d1/2, d2/2))
 *
 * @example
 * ```ts
 * const dist = new FDistribution(5, 10);
 * dist.mean();       // 10 / (10 - 2) = 1.25
 * dist.pdf(1);       // density at x = 1
 * dist.cdf(2);       // P(X <= 2)
 * dist.quantile(0.95); // critical value at 95%
 * ```
 */
export class FDistribution extends BaseContinuous {
  readonly name: string;

  /**
   * Creates a new F-distribution.
   *
   * @param d1 - Numerator degrees of freedom (must be a positive integer).
   * @param d2 - Denominator degrees of freedom (must be a positive integer).
   * @param rng - Optional random number generator returning values in [0, 1).
   * @throws {Error} If d1 is not a positive integer.
   * @throws {Error} If d2 is not a positive integer.
   *
   * @example
   * ```ts
   * const f = new FDistribution(3, 20);
   * ```
   */
  constructor(
    public readonly d1: number,
    public readonly d2: number,
    rng?: RandomFn,
  ) {
    super(rng);
    if (d1 <= 0 || !Number.isInteger(d1)) throw new Error(`Invalid parameter 'd1': expected a positive integer, received ${d1}`);
    if (d2 <= 0 || !Number.isInteger(d2)) throw new Error(`Invalid parameter 'd2': expected a positive integer, received ${d2}`);
    this.name = `F(${d1}, ${d2})`;
  }

  /**
   * Returns the mean of the F-distribution.
   *
   * Formula: E[X] = d2 / (d2 - 2), defined only for d2 > 2.
   *
   * @returns The expected value, or NaN if d2 <= 2.
   */
  mean(): number {
    if (this.d2 <= 2) return NaN;
    return this.d2 / (this.d2 - 2);
  }

  /**
   * Returns the variance of the F-distribution.
   *
   * Formula: Var(X) = 2 * d2^2 * (d1 + d2 - 2) / (d1 * (d2 - 2)^2 * (d2 - 4)),
   * defined only for d2 > 4.
   *
   * @returns The variance, or NaN if d2 <= 4.
   */
  variance(): number {
    if (this.d2 <= 4) return NaN;
    const { d1, d2 } = this;
    return (2 * d2 * d2 * (d1 + d2 - 2)) / (d1 * (d2 - 2) ** 2 * (d2 - 4));
  }

  /**
   * Evaluates the probability density function (PDF) at x.
   *
   * PDF: f(x) = (d1/d2)^(d1/2) * x^(d1/2-1) / ((1 + d1*x/d2)^((d1+d2)/2) * B(d1/2, d2/2))
   *
   * @param x - The point at which to evaluate the density.
   * @returns The probability density f(x) >= 0. Returns 0 for x <= 0.
   *
   * @example
   * ```ts
   * const dist = new FDistribution(5, 10);
   * dist.pdf(1.0); // density at x = 1
   * ```
   */
  pdf(x: number): number {
    if (x <= 0) return 0;
    const { d1, d2 } = this;
    const half1 = d1 / 2;
    const half2 = d2 / 2;
    const num = Math.pow(d1 / d2, half1) * Math.pow(x, half1 - 1);
    const den = Math.pow(1 + (d1 / d2) * x, (d1 + d2) / 2) * betaFn(half1, half2);
    return num / den;
  }

  /**
   * Evaluates the cumulative distribution function (CDF) at x.
   *
   * Computed using the regularized incomplete Beta function:
   * F(x) = I_{d1*x/(d1*x+d2)}(d1/2, d2/2)
   *
   * @param x - The point at which to evaluate the CDF.
   * @returns P(X <= x), a probability in [0, 1].
   */
  cdf(x: number): number {
    if (x <= 0) return 0;
    const { d1, d2 } = this;
    const t = (d1 * x) / (d1 * x + d2);
    return regularizedBeta(t, d1 / 2, d2 / 2);
  }

  /**
   * Computes the quantile (inverse CDF) for a given probability.
   *
   * Uses bisection search on the CDF.
   *
   * @param p - A probability in [0, 1].
   * @returns The quantile value x >= 0.
   * @throws {Error} If p is not in [0, 1].
   *
   * @example
   * ```ts
   * const dist = new FDistribution(5, 10);
   * dist.quantile(0.95); // F critical value at 95%
   * ```
   */
  quantile(p: number): number {
    if (p < 0 || p > 1) throw new Error(`Invalid parameter 'p': expected a value in [0, 1], received ${p}`);
    if (p === 0) return 0;
    if (p === 1) return Infinity;
    return quantileBisect((x) => this.cdf(x), p, 0, 1000);
  }

  /**
   * Draws a single random sample from the F-distribution.
   *
   * Uses the ratio of two independent chi-squared samples, each divided
   * by its degrees of freedom: (X1/d1) / (X2/d2).
   *
   * @returns A non-negative random variate.
   */
  sample(): number {
    // Ratio of two chi-squared samples
    const x1 = sampleChiSq(this.d1, this.rng);
    const x2 = sampleChiSq(this.d2, this.rng);
    return (x1 / this.d1) / (x2 / this.d2);
  }
}

/**
 * Generates a chi-squared random variate with k degrees of freedom
 * using the sum of squared standard normal variates (Box-Muller method).
 *
 * @param k - Degrees of freedom (positive integer).
 * @param rng - Random number generator returning values in [0, 1).
 * @returns A chi-squared random variate.
 */
function sampleChiSq(k: number, rng: RandomFn): number {
  let sum = 0;
  for (let i = 0; i < k; i++) {
    const u1 = rng();
    const u2 = rng();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    sum += z * z;
  }
  return sum;
}
