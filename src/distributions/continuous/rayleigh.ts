import { BaseContinuous } from "../base";
import { RandomFn } from "../../types";

/**
 * Rayleigh distribution.
 *
 * Arises as the magnitude of a 2D vector whose components are independent,
 * identically distributed Gaussian random variables with zero mean and
 * equal variance. Commonly used in communications theory, wind speed
 * modeling, and physical sciences.
 *
 * PDF: f(x) = (x / sigma^2) * exp(-x^2 / (2 * sigma^2))
 *
 * CDF: F(x) = 1 - exp(-x^2 / (2 * sigma^2))
 *
 * Support: x >= 0
 *
 * @example
 * ```ts
 * const dist = new Rayleigh(1);
 * dist.mean();       // sigma * sqrt(pi/2) ≈ 1.2533
 * dist.pdf(1);       // density at x = 1
 * dist.cdf(2);       // P(X <= 2)
 * dist.quantile(0.5); // median
 * dist.sample();     // random non-negative variate
 * ```
 */
export class Rayleigh extends BaseContinuous {
  readonly name: string;

  /**
   * Creates a new Rayleigh distribution.
   *
   * @param sigma - Scale parameter (must be > 0). Defaults to 1.
   * @param rng - Optional random number generator returning values in [0, 1).
   * @throws {Error} If sigma is not positive.
   */
  constructor(
    public readonly sigma: number = 1,
    rng?: RandomFn,
  ) {
    super(rng);
    if (sigma <= 0) throw new Error("sigma must be positive");
    this.name = `Rayleigh(${sigma})`;
  }

  /**
   * Returns the mean of the Rayleigh distribution.
   *
   * Formula: E[X] = sigma * sqrt(pi / 2)
   *
   * @returns The expected value.
   */
  mean(): number {
    return this.sigma * Math.sqrt(Math.PI / 2);
  }

  /**
   * Returns the variance of the Rayleigh distribution.
   *
   * Formula: Var(X) = (4 - pi) / 2 * sigma^2
   *
   * @returns The variance.
   */
  variance(): number {
    return ((4 - Math.PI) / 2) * this.sigma ** 2;
  }

  /**
   * Evaluates the probability density function at `x`.
   *
   * PDF: f(x) = (x / sigma^2) * exp(-x^2 / (2 * sigma^2))
   *
   * @param x - The point at which to evaluate the density.
   * @returns The probability density f(x) >= 0. Returns 0 for x < 0.
   */
  pdf(x: number): number {
    if (x < 0) return 0;
    const s2 = this.sigma ** 2;
    return (x / s2) * Math.exp(-x * x / (2 * s2));
  }

  /**
   * Evaluates the cumulative distribution function at `x`.
   *
   * CDF: F(x) = 1 - exp(-x^2 / (2 * sigma^2))
   *
   * @param x - The point at which to evaluate the CDF.
   * @returns P(X <= x) in [0, 1]. Returns 0 for x < 0.
   */
  cdf(x: number): number {
    if (x < 0) return 0;
    return 1 - Math.exp(-x * x / (2 * this.sigma ** 2));
  }

  /**
   * Computes the quantile (inverse CDF) for a given probability.
   *
   * Formula: Q(p) = sigma * sqrt(-2 * ln(1 - p))
   *
   * @param p - A probability in [0, 1].
   * @returns The quantile value x >= 0.
   * @throws {Error} If p is not in [0, 1].
   */
  quantile(p: number): number {
    if (p < 0 || p > 1) throw new Error("p must be in [0, 1]");
    if (p === 0) return 0;
    if (p === 1) return Infinity;
    return this.sigma * Math.sqrt(-2 * Math.log(1 - p));
  }

  /**
   * Draws a single random sample using the inverse CDF method.
   *
   * @returns A non-negative random variate.
   */
  sample(): number {
    return this.quantile(this.rng());
  }
}
