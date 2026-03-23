import { BaseDiscrete } from "../base";
import { RandomFn } from "../../types";

/**
 * Bernoulli distribution.
 *
 * Models a single trial with two outcomes (success/failure).
 *
 * P(X = 1) = p, P(X = 0) = 1 - p
 *
 * @example
 * ```ts
 * const coin = new Bernoulli(0.5);
 * coin.mean();     // 0.5
 * coin.pmf(1);     // 0.5
 * coin.sample();   // 0 or 1
 * ```
 */
export class Bernoulli extends BaseDiscrete {
  readonly name: string;

  /**
   * @param p - Probability of success, must be in [0, 1]. Defaults to 0.5.
   * @param rng - Optional random number generator.
   */
  constructor(public readonly p: number = 0.5, rng?: RandomFn) {
    super(rng);
    if (p < 0 || p > 1) throw new Error("p must be in [0, 1]");
    this.name = `Bernoulli(${p})`;
  }

  /** Returns the mean: E[X] = p. */
  mean(): number {
    return this.p;
  }

  /** Returns the variance: Var(X) = p(1 - p). */
  variance(): number {
    return this.p * (1 - this.p);
  }

  /**
   * Probability mass function.
   *
   * P(X = k) = p if k = 1, (1 - p) if k = 0, and 0 otherwise.
   *
   * @param k - The value at which to evaluate the PMF (0 or 1).
   * @returns The probability P(X = k).
   */
  pmf(k: number): number {
    if (k === 0) return 1 - this.p;
    if (k === 1) return this.p;
    return 0;
  }

  /**
   * Cumulative distribution function.
   *
   * F(k) = 0 for k < 0, (1 - p) for 0 <= k < 1, and 1 for k >= 1.
   *
   * @param k - The value at which to evaluate the CDF.
   * @returns The cumulative probability P(X <= k).
   */
  cdf(k: number): number {
    if (k < 0) return 0;
    if (k < 1) return 1 - this.p;
    return 1;
  }

  /**
   * Quantile function (inverse CDF).
   *
   * @param prob - The probability, must be in [0, 1].
   * @returns The smallest integer k such that F(k) >= prob.
   */
  quantile(prob: number): number {
    if (prob < 0 || prob > 1) throw new Error("p must be in [0, 1]");
    return prob <= 1 - this.p ? 0 : 1;
  }

  /**
   * Draws a single random sample from the Bernoulli distribution.
   *
   * @returns 1 with probability p, 0 with probability (1 - p).
   */
  sample(): number {
    return this.rng() < this.p ? 1 : 0;
  }
}
