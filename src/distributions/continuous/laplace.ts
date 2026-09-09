import { BaseContinuous } from "../base";
import { RandomFn } from "../../types";

/**
 * Laplace (double exponential) distribution.
 *
 * Characterized by a location parameter `mu` and a scale parameter `b`.
 * The Laplace distribution has heavier tails than the normal distribution
 * and is commonly used in robust statistics and signal processing.
 *
 * PDF: f(x) = (1 / (2b)) * exp(-|x - mu| / b)
 *
 * CDF: F(x) = 0.5 + 0.5 * sign(x - mu) * (1 - exp(-|x - mu| / b))
 *
 * Support: (-Infinity, +Infinity)
 *
 * @example
 * ```ts
 * const dist = new Laplace(0, 1);
 * dist.mean();       // 0
 * dist.variance();   // 2
 * dist.pdf(0);       // 0.5
 * dist.cdf(0);       // 0.5
 * dist.sample();     // random variate
 * ```
 */
export class Laplace extends BaseContinuous {
  readonly name: string;

  /**
   * Creates a new Laplace distribution.
   *
   * @param mu - Location parameter. Defaults to 0.
   * @param b - Scale parameter (must be > 0). Defaults to 1.
   * @param rng - Optional random number generator returning values in [0, 1).
   * @throws {Error} If b is not positive.
   */
  constructor(
    public readonly mu: number = 0,
    public readonly b: number = 1,
    rng?: RandomFn,
  ) {
    super(rng);
    if (b <= 0) throw new Error(`Invalid parameter 'b': expected a positive number, received ${b}`);
    this.name = `Laplace(${mu}, ${b})`;
  }

  /**
   * Returns the mean of the Laplace distribution.
   *
   * Formula: E[X] = mu
   *
   * @returns The expected value.
   */
  mean(): number {
    return this.mu;
  }

  /**
   * Returns the variance of the Laplace distribution.
   *
   * Formula: Var(X) = 2 * b^2
   *
   * @returns The variance.
   */
  variance(): number {
    return 2 * this.b ** 2;
  }

  /**
   * Evaluates the probability density function at `x`.
   *
   * PDF: f(x) = (1 / (2b)) * exp(-|x - mu| / b)
   *
   * @param x - The point at which to evaluate the density.
   * @returns The probability density f(x) >= 0.
   */
  pdf(x: number): number {
    return Math.exp(-Math.abs(x - this.mu) / this.b) / (2 * this.b);
  }

  /**
   * Computes the log of the probability density function at `x`.
   *
   * log f(x) = -|x - mu| / b - log(2 * b)
   *
   * @param x - The point at which to evaluate the log-density.
   * @returns The log-density.
   */
  logPdf(x: number): number {
    return -Math.abs(x - this.mu) / this.b - Math.log(2 * this.b);
  }

  /**
   * Returns the skewness of the Laplace distribution.
   *
   * The skewness is always 0 (symmetric distribution).
   */
  get skewness(): number {
    return 0;
  }

  /**
   * Returns the excess kurtosis of the Laplace distribution.
   *
   * The excess kurtosis is always 3.
   */
  get kurtosis(): number {
    return 3;
  }

  /**
   * Returns the mode of the Laplace distribution.
   *
   * The mode equals the location parameter `mu`.
   */
  get mode(): number {
    return this.mu;
  }

  /**
   * Evaluates the cumulative distribution function at `x`.
   *
   * CDF: F(x) = 0.5 + 0.5 * sign(x - mu) * (1 - exp(-|x - mu| / b))
   *
   * @param x - The point at which to evaluate the CDF.
   * @returns P(X <= x) in [0, 1].
   */
  cdf(x: number): number {
    const diff = x - this.mu;
    if (diff < 0) {
      return 0.5 * Math.exp(diff / this.b);
    }
    return 1 - 0.5 * Math.exp(-diff / this.b);
  }

  /**
   * Computes the quantile (inverse CDF) for a given probability.
   *
   * Formula: Q(p) = mu - b * sign(p - 0.5) * ln(1 - 2|p - 0.5|)
   *
   * @param p - A probability in [0, 1].
   * @returns The quantile value.
   * @throws {Error} If p is not in [0, 1].
   */
  quantile(p: number): number {
    if (p < 0 || p > 1) throw new Error(`Invalid parameter 'p': expected a value in [0, 1], received ${p}`);
    if (p === 0) return -Infinity;
    if (p === 1) return Infinity;
    if (p <= 0.5) {
      return this.mu + this.b * Math.log(2 * p);
    }
    return this.mu - this.b * Math.log(2 * (1 - p));
  }

  /**
   * Draws a single random sample using the inverse CDF method.
   *
   * @returns A random variate.
   */
  sample(): number {
    return this.quantile(this.rng());
  }
}
