import { BaseContinuous } from "../base";
import { RandomFn } from "../../types";

/**
 * Gumbel distribution (Type I extreme value distribution).
 *
 * Also known as the log-Weibull distribution. Models the distribution
 * of the maximum (or minimum) of a number of samples of various distributions.
 * It is the special case of the GEV distribution with shape parameter xi = 0.
 *
 * PDF: f(x; mu, beta) = (1/beta) * exp(-(z + exp(-z))), where z = (x - mu) / beta
 *
 * CDF: F(x; mu, beta) = exp(-exp(-z))
 *
 * The Gumbel distribution has support on the entire real line and features
 * an exponentially decaying right tail.
 *
 * @example
 * ```ts
 * const dist = new Gumbel(0, 1);
 * dist.mean();       // ~0.5772 (Euler-Mascheroni constant)
 * dist.pdf(0);       // density at x = 0
 * dist.cdf(1);       // P(X <= 1)
 * dist.quantile(0.9); // 90th percentile
 * dist.sample();     // random variate
 * ```
 */
export class Gumbel extends BaseContinuous {
  readonly name: string;

  /**
   * Creates a new Gumbel distribution.
   *
   * @param mu - Location parameter (the mode of the distribution). Defaults to 0.
   * @param beta - Scale parameter (must be > 0). Defaults to 1.
   * @param rng - Optional random number generator returning values in [0, 1).
   * @throws {Error} If beta is not positive.
   *
   * @example
   * ```ts
   * const standard = new Gumbel();       // mu=0, beta=1
   * const shifted = new Gumbel(10, 3);   // mu=10, beta=3
   * ```
   */
  constructor(
    public readonly mu: number = 0,
    public readonly beta: number = 1,
    rng?: RandomFn,
  ) {
    super(rng);
    if (beta <= 0) throw new Error("beta must be positive");
    this.name = `Gumbel(${mu}, ${beta})`;
  }

  /**
   * Returns the mean of the Gumbel distribution.
   *
   * Formula: E[X] = mu + beta * gamma_EM, where gamma_EM is the
   * Euler-Mascheroni constant (~0.5772).
   *
   * @returns The expected value.
   */
  mean(): number {
    // Euler-Mascheroni constant
    return this.mu + this.beta * 0.5772156649015329;
  }

  /**
   * Returns the variance of the Gumbel distribution.
   *
   * Formula: Var(X) = pi^2 * beta^2 / 6
   *
   * @returns The variance.
   */
  variance(): number {
    return (Math.PI ** 2 * this.beta ** 2) / 6;
  }

  /**
   * Evaluates the probability density function (PDF) at x.
   *
   * PDF: f(x) = (1/beta) * exp(-(z + exp(-z))), where z = (x - mu) / beta
   *
   * @param x - The point at which to evaluate the density.
   * @returns The probability density f(x) >= 0.
   *
   * @example
   * ```ts
   * const dist = new Gumbel(0, 1);
   * dist.pdf(0); // ~0.3679
   * ```
   */
  pdf(x: number): number {
    const z = (x - this.mu) / this.beta;
    return Math.exp(-(z + Math.exp(-z))) / this.beta;
  }

  /**
   * Evaluates the cumulative distribution function (CDF) at x.
   *
   * CDF: F(x) = exp(-exp(-(x - mu) / beta))
   *
   * @param x - The point at which to evaluate the CDF.
   * @returns P(X <= x), a probability in [0, 1].
   *
   * @example
   * ```ts
   * const dist = new Gumbel(0, 1);
   * dist.cdf(0); // exp(-1) ≈ 0.3679
   * ```
   */
  cdf(x: number): number {
    const z = (x - this.mu) / this.beta;
    return Math.exp(-Math.exp(-z));
  }

  /**
   * Computes the quantile (inverse CDF) for a given probability.
   *
   * Formula: Q(p) = mu - beta * ln(-ln(p))
   *
   * @param p - A probability in [0, 1].
   * @returns The quantile value.
   * @throws {Error} If p is not in [0, 1].
   *
   * @example
   * ```ts
   * const dist = new Gumbel(0, 1);
   * dist.quantile(0.5);  // median ≈ 0.3665
   * dist.quantile(0.99); // 99th percentile
   * ```
   */
  quantile(p: number): number {
    if (p < 0 || p > 1) throw new Error("p must be in [0, 1]");
    if (p === 0) return -Infinity;
    if (p === 1) return Infinity;
    return this.mu - this.beta * Math.log(-Math.log(p));
  }

  /**
   * Draws a single random sample from the Gumbel distribution.
   *
   * Uses the inverse CDF method.
   *
   * @returns A random variate from the Gumbel distribution.
   */
  sample(): number {
    return this.quantile(this.rng());
  }
}
