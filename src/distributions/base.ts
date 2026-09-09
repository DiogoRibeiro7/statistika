import {
  ContinuousDistribution,
  DiscreteDistribution,
  RandomFn,
} from "../types";
import { resolveRng } from "../random";

/**
 * Abstract base class for all continuous probability distributions.
 *
 * Provides default implementations for {@link stdDev}, {@link sf}, and
 * {@link sampleN}. Subclasses must implement the core statistical methods:
 * {@link mean}, {@link variance}, {@link pdf}, {@link cdf}, {@link quantile},
 * and {@link sample}.
 *
 * @example
 * ```ts
 * class MyDist extends BaseContinuous {
 *   // implement abstract members ...
 * }
 * const d = new MyDist();
 * d.stdDev(); // sqrt(d.variance())
 * ```
 */
export abstract class BaseContinuous implements ContinuousDistribution {
  abstract readonly name: string;

  /** Random number generator used by `sample()`. Defaults to `Math.random`. */
  protected rng: RandomFn;

  /**
   * @param rng - Optional random number generator returning values in [0, 1).
   *              Falls back to the global seeded RNG (if set) or `Math.random`.
   */
  constructor(rng?: RandomFn) {
    this.rng = resolveRng(rng);
  }

  /** Returns the theoretical mean (expected value) of the distribution. */
  abstract mean(): number;

  /** Returns the theoretical variance of the distribution. */
  abstract variance(): number;

  /**
   * Evaluates the probability density function (PDF) at the given point.
   * @param x - The point at which to evaluate the density.
   * @returns The probability density f(x) >= 0.
   */
  abstract pdf(x: number): number;

  /**
   * Evaluates the cumulative distribution function (CDF), P(X <= x).
   * @param x - The point at which to evaluate the CDF.
   * @returns A probability in [0, 1].
   */
  abstract cdf(x: number): number;

  /**
   * Computes the quantile (inverse CDF) for a given probability.
   * @param p - A probability in [0, 1].
   * @returns The smallest x such that CDF(x) >= p.
   */
  abstract quantile(p: number): number;

  /**
   * Draws a single random sample from the distribution.
   * @returns A random variate.
   */
  abstract sample(): number;

  /**
   * Returns the standard deviation, defined as sqrt(variance()).
   * @returns The standard deviation of the distribution.
   */
  stdDev(): number {
    return Math.sqrt(this.variance());
  }

  /**
   * Computes the survival function (SF), also known as the complementary CDF.
   *
   * SF(x) = 1 - CDF(x) = P(X > x)
   *
   * @param x - The point at which to evaluate the survival function.
   * @returns A probability in [0, 1].
   */
  sf(x: number): number {
    return 1 - this.cdf(x);
  }

  /**
   * Draws `n` independent random samples from the distribution.
   * @param n - The number of samples to draw.
   * @returns An array of `n` random variates.
   */
  sampleN(n: number): number[] {
    const result: number[] = new Array(n);
    for (let i = 0; i < n; i++) result[i] = this.sample();
    return result;
  }
}

/**
 * Abstract base class for all discrete probability distributions.
 *
 * Provides default implementations for {@link stdDev}, {@link sf}, and
 * {@link sampleN}. Subclasses must implement the core statistical methods:
 * {@link mean}, {@link variance}, {@link pmf}, {@link cdf}, {@link quantile},
 * and {@link sample}.
 */
export abstract class BaseDiscrete implements DiscreteDistribution {
  abstract readonly name: string;

  /** Random number generator used by `sample()`. Defaults to `Math.random`. */
  protected rng: RandomFn;

  /**
   * @param rng - Optional random number generator returning values in [0, 1).
   *              Falls back to the global seeded RNG (if set) or `Math.random`.
   */
  constructor(rng?: RandomFn) {
    this.rng = resolveRng(rng);
  }

  /** Returns the theoretical mean (expected value) of the distribution. */
  abstract mean(): number;

  /** Returns the theoretical variance of the distribution. */
  abstract variance(): number;

  /**
   * Evaluates the probability mass function (PMF) at the given integer point.
   * @param k - The integer value at which to evaluate the PMF.
   * @returns The probability P(X = k) in [0, 1].
   */
  abstract pmf(k: number): number;

  /**
   * Evaluates the cumulative distribution function (CDF), P(X <= k).
   * @param k - The point at which to evaluate the CDF.
   * @returns A probability in [0, 1].
   */
  abstract cdf(k: number): number;

  /**
   * Computes the quantile (inverse CDF) for a given probability.
   * @param p - A probability in [0, 1].
   * @returns The smallest integer k such that CDF(k) >= p.
   */
  abstract quantile(p: number): number;

  /**
   * Draws a single random sample from the distribution.
   * @returns A random integer variate.
   */
  abstract sample(): number;

  /**
   * Returns the standard deviation, defined as sqrt(variance()).
   * @returns The standard deviation of the distribution.
   */
  stdDev(): number {
    return Math.sqrt(this.variance());
  }

  /**
   * Computes the survival function (SF), also known as the complementary CDF.
   *
   * SF(k) = 1 - CDF(k) = P(X > k)
   *
   * @param k - The point at which to evaluate the survival function.
   * @returns A probability in [0, 1].
   */
  sf(k: number): number {
    return 1 - this.cdf(k);
  }

  /**
   * Draws `n` independent random samples from the distribution.
   * @param n - The number of samples to draw.
   * @returns An array of `n` random integer variates.
   */
  sampleN(n: number): number[] {
    const result: number[] = new Array(n);
    for (let i = 0; i < n; i++) result[i] = this.sample();
    return result;
  }
}
