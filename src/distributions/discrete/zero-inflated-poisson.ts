/**
 * Zero-Inflated Poisson (ZIP) distribution.
 *
 * A mixture model:  with probability pi the outcome is always 0,
 * with probability (1 - pi) the outcome follows Poisson(lambda).
 *
 * P(X = 0)   = pi + (1 - pi) e^{-lambda}
 * P(X = k)   = (1 - pi) * Poisson(k; lambda)     for k >= 1
 *
 * This distribution is useful for modeling count data with excess zeros,
 * such as insurance claims or rare event counts.
 *
 * @example
 * ```ts
 * const dist = new ZeroInflatedPoisson(3, 0.2);
 * dist.mean();     // 2.4
 * dist.pmf(0);     // inflated zero probability
 * dist.sample();   // random non-negative integer
 * ```
 */

import { BaseDiscrete } from "../base";
import { logFactorial, regularizedGammaP } from "../../utils/math";
import { RandomFn } from "../../types";

/**
 * Zero-Inflated Poisson (ZIP) distribution class.
 *
 * Combines a point mass at zero (with probability pi) and a standard
 * Poisson distribution (with probability 1 - pi) to model count data
 * exhibiting excess zeros.
 */
export class ZeroInflatedPoisson extends BaseDiscrete {
  readonly name: string;

  /**
   * Creates a new Zero-Inflated Poisson distribution.
   *
   * @param lambda - Poisson rate parameter (must be positive).
   * @param pi - Zero-inflation probability, P(structural zero), must be in [0, 1).
   * @param rng - Optional custom random number generator; defaults to Math.random.
   * @throws {Error} If lambda is not positive.
   * @throws {Error} If pi is not in [0, 1).
   *
   * @example
   * ```ts
   * const dist = new ZeroInflatedPoisson(5, 0.3);
   * ```
   */
  constructor(
    public readonly lambda: number,
    public readonly pi: number,
    rng?: RandomFn,
  ) {
    super(rng);
    if (lambda <= 0) throw new Error("lambda must be positive");
    if (pi < 0 || pi >= 1) throw new Error("pi must be in [0, 1)");
    this.name = `ZIP(${lambda}, ${pi})`;
  }

  /**
   * Computes the mean (expected value) of the distribution.
   *
   * Formula: E[X] = (1 - pi) * lambda
   *
   * @returns The mean of the distribution.
   *
   * @example
   * ```ts
   * new ZeroInflatedPoisson(5, 0.2).mean(); // 4
   * ```
   */
  mean(): number {
    return (1 - this.pi) * this.lambda;
  }

  /**
   * Computes the variance of the distribution.
   *
   * Formula: Var(X) = (1 - pi) * lambda * (1 + pi * lambda)
   *
   * Note: The variance exceeds that of a standard Poisson (overdispersion)
   * whenever pi > 0.
   *
   * @returns The variance of the distribution.
   *
   * @example
   * ```ts
   * new ZeroInflatedPoisson(5, 0.2).variance(); // 8
   * ```
   */
  variance(): number {
    return (1 - this.pi) * this.lambda * (1 + this.pi * this.lambda);
  }

  /**
   * Computes the probability mass function P(X = k).
   *
   * For k = 0: P(X = 0) = pi + (1 - pi) * e^{-lambda}
   * For k >= 1: P(X = k) = (1 - pi) * lambda^k * e^{-lambda} / k!
   *
   * @param k - The value at which to evaluate the PMF (non-negative integer).
   * @returns The probability P(X = k). Returns 0 for non-integer or negative k.
   *
   * @example
   * ```ts
   * const dist = new ZeroInflatedPoisson(3, 0.2);
   * dist.pmf(0);  // inflated: 0.2 + 0.8 * e^{-3} ~ 0.2398
   * dist.pmf(1);  // 0.8 * 3 * e^{-3} / 1! ~ 0.1194
   * ```
   */
  pmf(k: number): number {
    if (!Number.isInteger(k) || k < 0) return 0;
    if (k === 0) {
      return this.pi + (1 - this.pi) * Math.exp(-this.lambda);
    }
    const logPoisson =
      k * Math.log(this.lambda) - this.lambda - logFactorial(k);
    return (1 - this.pi) * Math.exp(logPoisson);
  }

  /**
   * Computes the cumulative distribution function P(X <= k).
   *
   * Formula: F(k) = pi + (1 - pi) * PoissonCDF(k; lambda)
   *
   * where PoissonCDF is computed using the regularized upper incomplete gamma function.
   *
   * @param k - The value at which to evaluate the CDF.
   * @returns The cumulative probability P(X <= k).
   *
   * @example
   * ```ts
   * const dist = new ZeroInflatedPoisson(3, 0.2);
   * dist.cdf(0);   // P(X <= 0)
   * dist.cdf(5);   // P(X <= 5)
   * ```
   */
  cdf(k: number): number {
    if (k < 0) return 0;
    const kFloor = Math.floor(k);
    // CDF = pi + (1-pi) * Poisson_CDF(k; lambda)
    const poissonCdf = 1 - regularizedGammaP(kFloor + 1, this.lambda);
    return this.pi + (1 - this.pi) * poissonCdf;
  }

  /**
   * Computes the quantile (inverse CDF) function.
   *
   * Returns the smallest integer k such that P(X <= k) >= prob.
   * Uses iterative PMF summation with a safety limit based on lambda.
   *
   * @param prob - The probability, must be in [0, 1].
   * @returns The quantile value.
   * @throws {Error} If prob is not in [0, 1].
   *
   * @example
   * ```ts
   * const dist = new ZeroInflatedPoisson(3, 0.2);
   * dist.quantile(0);    // 0
   * dist.quantile(0.5);  // median
   * dist.quantile(1);    // Infinity
   * ```
   */
  quantile(prob: number): number {
    if (prob < 0 || prob > 1) throw new Error("p must be in [0, 1]");
    if (prob === 0) return 0;
    if (prob === 1) return Infinity;
    let cumulative = 0;
    let k = 0;
    const limit = this.lambda + 40 * Math.sqrt(this.lambda) + 100;
    while (cumulative < prob) {
      cumulative += this.pmf(k);
      if (cumulative >= prob) return k;
      k++;
      if (k > limit) return k;
    }
    return k;
  }

  /**
   * Draws a single random sample from the distribution.
   *
   * With probability pi, returns 0 (structural zero). Otherwise, samples
   * from a Poisson(lambda) distribution using Knuth's algorithm for small
   * lambda or the quantile method for large lambda.
   *
   * @returns A random non-negative integer.
   *
   * @example
   * ```ts
   * const dist = new ZeroInflatedPoisson(3, 0.2);
   * const count = dist.sample();
   * ```
   */
  sample(): number {
    // With probability pi, return 0 (structural zero)
    if (this.rng() < this.pi) return 0;
    // Otherwise sample from Poisson(lambda)
    if (this.lambda < 30) {
      const L = Math.exp(-this.lambda);
      let count = 0;
      let p = 1;
      do {
        count++;
        p *= this.rng();
      } while (p > L);
      return count - 1;
    }
    return this.quantile(this.rng());
  }
}
