import { BaseDiscrete } from "../base";
import { RandomFn } from "../../types";

/**
 * Geometric distribution modeling the number of failures before the first success.
 *
 * Each trial independently succeeds with probability p. The random variable X
 * counts the number of failures before the first success, so X takes values 0, 1, 2, ...
 *
 * PMF: P(X = k) = p * (1 - p)^k for k = 0, 1, 2, ...
 *
 * @example
 * ```ts
 * const dist = new Geometric(0.3);
 * dist.mean();     // ~2.333
 * dist.pmf(0);     // 0.3
 * dist.sample();   // random non-negative integer
 * ```
 */
export class Geometric extends BaseDiscrete {
  readonly name: string;

  /**
   * Creates a new Geometric distribution.
   *
   * @param p - Probability of success on each trial, must be in (0, 1].
   * @param rng - Optional custom random number generator; defaults to Math.random.
   * @throws {Error} If p is not in (0, 1].
   *
   * @example
   * ```ts
   * const dist = new Geometric(0.5);
   * ```
   */
  constructor(public readonly p: number = 0.5, rng?: RandomFn) {
    super(rng);
    if (p <= 0 || p > 1) throw new Error("p must be in (0, 1]");
    this.name = `Geometric(${p})`;
  }

  /**
   * Computes the mean (expected value) of the distribution.
   *
   * Formula: E[X] = (1 - p) / p
   *
   * @returns The mean number of failures before the first success.
   *
   * @example
   * ```ts
   * new Geometric(0.5).mean(); // 1
   * new Geometric(0.25).mean(); // 3
   * ```
   */
  mean(): number {
    return (1 - this.p) / this.p;
  }

  /**
   * Computes the variance of the distribution.
   *
   * Formula: Var(X) = (1 - p) / p^2
   *
   * @returns The variance of the distribution.
   *
   * @example
   * ```ts
   * new Geometric(0.5).variance(); // 2
   * ```
   */
  variance(): number {
    return (1 - this.p) / (this.p * this.p);
  }

  /**
   * Computes the probability mass function P(X = k).
   *
   * P(X = k) = p * (1 - p)^k, where k is the number of failures before the first success.
   *
   * @param k - The number of failures (non-negative integer).
   * @returns The probability P(X = k). Returns 0 for non-integer or negative k.
   *
   * @example
   * ```ts
   * const dist = new Geometric(0.5);
   * dist.pmf(0);  // 0.5
   * dist.pmf(1);  // 0.25
   * dist.pmf(2);  // 0.125
   * ```
   */
  pmf(k: number): number {
    if (!Number.isInteger(k) || k < 0) return 0;
    return this.p * Math.pow(1 - this.p, k);
  }

  /**
   * Computes the cumulative distribution function P(X <= k).
   *
   * Formula: F(k) = 1 - (1 - p)^(floor(k) + 1)
   *
   * @param k - The value at which to evaluate the CDF.
   * @returns The cumulative probability P(X <= k).
   *
   * @example
   * ```ts
   * const dist = new Geometric(0.5);
   * dist.cdf(0);  // 0.5
   * dist.cdf(2);  // 0.875
   * ```
   */
  cdf(k: number): number {
    if (k < 0) return 0;
    return 1 - Math.pow(1 - this.p, Math.floor(k) + 1);
  }

  /**
   * Computes the quantile (inverse CDF) function.
   *
   * Returns the smallest integer k such that P(X <= k) >= prob.
   *
   * Formula: k = floor(log(1 - prob) / log(1 - p))
   *
   * @param prob - The probability, must be in [0, 1].
   * @returns The quantile value.
   * @throws {Error} If prob is not in [0, 1].
   *
   * @example
   * ```ts
   * const dist = new Geometric(0.5);
   * dist.quantile(0.5);   // 0
   * dist.quantile(0.75);  // 1
   * dist.quantile(1);     // Infinity
   * ```
   */
  quantile(prob: number): number {
    if (prob < 0 || prob > 1) throw new Error("p must be in [0, 1]");
    if (prob === 0) return 0;
    if (prob === 1) return Infinity;
    return Math.floor(Math.log(1 - prob) / Math.log(1 - this.p));
  }

  /**
   * Draws a single random sample from the distribution.
   *
   * Uses inverse transform sampling via the logarithmic method.
   *
   * @returns A random non-negative integer representing the number of failures before the first success.
   *
   * @example
   * ```ts
   * const dist = new Geometric(0.3);
   * const failures = dist.sample();
   * ```
   */
  sample(): number {
    return Math.floor(Math.log(1 - this.rng()) / Math.log(1 - this.p));
  }
}
