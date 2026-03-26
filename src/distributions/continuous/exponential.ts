import { BaseContinuous } from "../base";
import { RandomFn } from "../../types";

/**
 * Exponential distribution with rate parameter `lambda`.
 *
 * The PDF is:
 *
 *   f(x) = lambda * exp(-lambda * x)   for x >= 0
 *
 * Support: [0, +Infinity)
 *
 * @example
 * ```ts
 * const dist = new Exponential(2);
 * dist.mean();       // 0.5
 * dist.pdf(1);       // 2 * exp(-2) ~ 0.2707
 * dist.cdf(0.5);     // 1 - exp(-1) ~ 0.6321
 * ```
 */
export class Exponential extends BaseContinuous {
  readonly name: string;

  /**
   * Creates an Exponential distribution.
   * @param lambda - Rate parameter (must be > 0). Defaults to 1.
   * @param rng - Optional random number generator.
   * @throws If `lambda` is not positive.
   */
  constructor(public readonly lambda: number = 1, rng?: RandomFn) {
    super(rng);
    if (lambda <= 0) throw new Error(`Invalid parameter 'lambda': expected a positive number, received ${lambda}`);
    this.name = `Exponential(${lambda})`;
  }

  /**
   * Returns the mean: `1 / lambda`.
   */
  mean(): number {
    return 1 / this.lambda;
  }

  /**
   * Returns the variance: `1 / lambda^2`.
   */
  variance(): number {
    return 1 / this.lambda ** 2;
  }

  /**
   * Evaluates the PDF at `x`.
   *
   * f(x) = lambda * exp(-lambda * x) for x >= 0, 0 otherwise.
   *
   * @param x - The point at which to evaluate the density.
   * @returns The density f(x).
   */
  pdf(x: number): number {
    return x < 0 ? 0 : this.lambda * Math.exp(-this.lambda * x);
  }

  /**
   * Computes the log of the probability density function at `x`.
   *
   * log f(x) = log(lambda) - lambda * x for x >= 0.
   *
   * @param x - The point at which to evaluate the log-density.
   * @returns The log-density. Returns -Infinity for x < 0.
   */
  logPdf(x: number): number {
    if (x < 0) return -Infinity;
    return Math.log(this.lambda) - this.lambda * x;
  }

  /**
   * Returns the skewness of the Exponential distribution.
   *
   * The skewness is always 2.
   */
  get skewness(): number {
    return 2;
  }

  /**
   * Returns the excess kurtosis of the Exponential distribution.
   *
   * The excess kurtosis is always 6.
   */
  get kurtosis(): number {
    return 6;
  }

  /**
   * Returns the mode of the Exponential distribution.
   *
   * The mode is always 0.
   */
  get mode(): number {
    return 0;
  }

  /**
   * Evaluates the CDF at `x`.
   *
   * F(x) = 1 - exp(-lambda * x) for x >= 0.
   *
   * @param x - The point at which to evaluate the CDF.
   * @returns P(X <= x) in [0, 1].
   */
  cdf(x: number): number {
    return x < 0 ? 0 : 1 - Math.exp(-this.lambda * x);
  }

  /**
   * Computes the quantile (inverse CDF): `-ln(1 - p) / lambda`.
   * @param p - A probability in [0, 1].
   * @returns The value x such that P(X <= x) = p.
   * @throws If `p` is outside [0, 1].
   */
  quantile(p: number): number {
    if (p < 0 || p > 1) throw new Error(`Invalid parameter 'p': expected a value in [0, 1], received ${p}`);
    if (p === 1) return Infinity;
    return -Math.log(1 - p) / this.lambda;
  }

  /**
   * Draws a random sample via the inverse CDF method.
   * @returns A random variate from this Exponential distribution.
   */
  sample(): number {
    return -Math.log(1 - this.rng()) / this.lambda;
  }
}
