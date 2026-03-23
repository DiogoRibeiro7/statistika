import { BaseContinuous } from "../base";
import { GammaDistribution } from "./gamma";
import { RandomFn } from "../../types";

/**
 * Chi-squared distribution with k degrees of freedom.
 *
 * The chi-squared distribution is the distribution of a sum of squares
 * of k independent standard normal random variables. It is a special case
 * of the Gamma distribution: Chi-squared(k) = Gamma(k/2, 1/2).
 *
 * PDF: f(x; k) = x^(k/2 - 1) * exp(-x/2) / (2^(k/2) * Gamma(k/2))
 *
 * Commonly used in hypothesis testing (chi-squared tests) and in
 * constructing confidence intervals for variance estimates.
 *
 * @example
 * ```ts
 * const dist = new ChiSquared(5);
 * dist.mean();       // 5
 * dist.variance();   // 10
 * dist.pdf(3);       // density at x = 3
 * dist.cdf(5);       // P(X <= 5)
 * dist.quantile(0.95); // critical value at 95%
 * ```
 */
export class ChiSquared extends BaseContinuous {
  readonly name: string;

  /** @internal Underlying Gamma(k/2, 1/2) distribution used for all computations. */
  private readonly gammaDistribution: GammaDistribution;

  /**
   * Creates a new Chi-squared distribution.
   *
   * @param k - Degrees of freedom (must be a positive integer). Defaults to 1.
   * @param rng - Optional random number generator returning values in [0, 1).
   * @throws {Error} If k is not a positive integer.
   *
   * @example
   * ```ts
   * const chi2 = new ChiSquared(10);
   * chi2.mean(); // 10
   * ```
   */
  constructor(public readonly k: number = 1, rng?: RandomFn) {
    super(rng);
    if (k <= 0 || !Number.isInteger(k)) {
      throw new Error("k (degrees of freedom) must be a positive integer");
    }
    this.name = `ChiSquared(${k})`;
    this.gammaDistribution = new GammaDistribution(k / 2, 0.5, rng);
  }

  /**
   * Returns the mean of the Chi-squared distribution.
   *
   * Formula: E[X] = k
   *
   * @returns The expected value, equal to the degrees of freedom k.
   */
  mean(): number {
    return this.k;
  }

  /**
   * Returns the variance of the Chi-squared distribution.
   *
   * Formula: Var(X) = 2k
   *
   * @returns The variance, equal to twice the degrees of freedom.
   */
  variance(): number {
    return 2 * this.k;
  }

  /**
   * Evaluates the probability density function (PDF) at x.
   *
   * Delegates to the underlying Gamma(k/2, 1/2) distribution.
   *
   * @param x - The point at which to evaluate the density.
   * @returns The probability density f(x) >= 0. Returns 0 for x <= 0.
   */
  pdf(x: number): number {
    return this.gammaDistribution.pdf(x);
  }

  /**
   * Evaluates the cumulative distribution function (CDF) at x.
   *
   * Delegates to the underlying Gamma(k/2, 1/2) distribution.
   *
   * @param x - The point at which to evaluate the CDF.
   * @returns P(X <= x), a probability in [0, 1].
   */
  cdf(x: number): number {
    return this.gammaDistribution.cdf(x);
  }

  /**
   * Computes the quantile (inverse CDF) for a given probability.
   *
   * Delegates to the underlying Gamma(k/2, 1/2) distribution.
   *
   * @param p - A probability in [0, 1].
   * @returns The quantile value x such that P(X <= x) = p.
   * @throws {Error} If p is not in [0, 1].
   */
  quantile(p: number): number {
    return this.gammaDistribution.quantile(p);
  }

  /**
   * Draws a single random sample from the Chi-squared distribution.
   *
   * Delegates to the underlying Gamma(k/2, 1/2) distribution.
   *
   * @returns A non-negative random variate.
   */
  sample(): number {
    return this.gammaDistribution.sample();
  }
}
