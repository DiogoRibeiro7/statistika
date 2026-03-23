import { BaseDiscrete } from "../base";
import { RandomFn } from "../../types";

/**
 * Discrete Uniform distribution over consecutive integers [a, b].
 *
 * Each integer in the range has equal probability 1 / (b - a + 1).
 *
 * @example
 * ```ts
 * const dist = new DiscreteUniform(1, 6); // fair die
 * dist.mean();     // 3.5
 * dist.pmf(3);     // ~0.1667
 * dist.sample();   // random integer in [1, 6]
 * ```
 */
export class DiscreteUniform extends BaseDiscrete {
  readonly name: string;
  private readonly range: number;

  /**
   * Creates a new Discrete Uniform distribution over integers [a, b].
   *
   * @param a - Lower bound (inclusive). Must be an integer strictly less than b.
   * @param b - Upper bound (inclusive). Must be an integer strictly greater than a.
   * @param rng - Optional custom random number generator; defaults to Math.random.
   * @throws {Error} If a or b are not integers.
   * @throws {Error} If a >= b.
   *
   * @example
   * ```ts
   * const die = new DiscreteUniform(1, 6);
   * const coin = new DiscreteUniform(0, 1);
   * ```
   */
  constructor(
    public readonly a: number = 0,
    public readonly b: number = 1,
    rng?: RandomFn,
  ) {
    super(rng);
    if (!Number.isInteger(a) || !Number.isInteger(b)) {
      throw new Error("a and b must be integers");
    }
    if (a >= b) throw new Error("a must be less than b");
    this.name = `DiscreteUniform(${a}, ${b})`;
    this.range = b - a + 1;
  }

  /**
   * Computes the mean (expected value) of the distribution.
   *
   * Formula: E[X] = (a + b) / 2
   *
   * @returns The mean of the distribution.
   *
   * @example
   * ```ts
   * new DiscreteUniform(1, 6).mean(); // 3.5
   * ```
   */
  mean(): number {
    return (this.a + this.b) / 2;
  }

  /**
   * Computes the variance of the distribution.
   *
   * Formula: Var(X) = ((b - a + 1)^2 - 1) / 12
   *
   * @returns The variance of the distribution.
   *
   * @example
   * ```ts
   * new DiscreteUniform(1, 6).variance(); // ~2.9167
   * ```
   */
  variance(): number {
    return (this.range * this.range - 1) / 12;
  }

  /**
   * Computes the probability mass function P(X = k).
   *
   * Returns 1 / (b - a + 1) if k is an integer in [a, b], otherwise 0.
   *
   * @param k - The value at which to evaluate the PMF.
   * @returns The probability P(X = k).
   *
   * @example
   * ```ts
   * const dist = new DiscreteUniform(1, 6);
   * dist.pmf(3);   // ~0.1667
   * dist.pmf(7);   // 0
   * dist.pmf(2.5); // 0
   * ```
   */
  pmf(k: number): number {
    if (!Number.isInteger(k) || k < this.a || k > this.b) return 0;
    return 1 / this.range;
  }

  /**
   * Computes the cumulative distribution function P(X <= k).
   *
   * Formula: F(k) = (floor(k) - a + 1) / (b - a + 1) for a <= k < b.
   *
   * @param k - The value at which to evaluate the CDF.
   * @returns The cumulative probability P(X <= k).
   *
   * @example
   * ```ts
   * const dist = new DiscreteUniform(1, 6);
   * dist.cdf(3);   // 0.5
   * dist.cdf(0);   // 0
   * dist.cdf(6);   // 1
   * ```
   */
  cdf(k: number): number {
    if (k < this.a) return 0;
    if (k >= this.b) return 1;
    return (Math.floor(k) - this.a + 1) / this.range;
  }

  /**
   * Computes the quantile (inverse CDF) function.
   *
   * Returns the smallest integer k such that P(X <= k) >= p.
   *
   * @param p - The probability, must be in [0, 1].
   * @returns The quantile value.
   * @throws {Error} If p is not in [0, 1].
   *
   * @example
   * ```ts
   * const dist = new DiscreteUniform(1, 6);
   * dist.quantile(0);   // 1
   * dist.quantile(0.5); // 3
   * dist.quantile(1);   // 6
   * ```
   */
  quantile(p: number): number {
    if (p < 0 || p > 1) throw new Error("p must be in [0, 1]");
    if (p === 0) return this.a;
    if (p === 1) return this.b;
    return Math.min(this.a + Math.floor(p * this.range), this.b);
  }

  /**
   * Draws a single random sample from the distribution.
   *
   * @returns A random integer uniformly distributed in [a, b].
   *
   * @example
   * ```ts
   * const dist = new DiscreteUniform(1, 6);
   * const roll = dist.sample(); // random integer 1..6
   * ```
   */
  sample(): number {
    return this.a + Math.floor(this.rng() * this.range);
  }
}
