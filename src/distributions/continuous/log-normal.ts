import { BaseContinuous } from "../base";
import { Normal } from "./normal";
import { RandomFn } from "../../types";

/**
 * Log-normal distribution.
 *
 * A continuous probability distribution of a random variable whose
 * logarithm is normally distributed. If X ~ Normal(mu, sigma), then
 * exp(X) ~ LogNormal(mu, sigma).
 *
 * PDF: f(x; mu, sigma) = 1 / (x * sigma * sqrt(2*pi)) * exp(-(ln(x) - mu)^2 / (2*sigma^2))
 *
 * Support: x > 0
 *
 * The log-normal distribution is widely used to model quantities that are
 * inherently positive and right-skewed, such as income, stock prices,
 * and biological measurements.
 *
 * @example
 * ```ts
 * const dist = new LogNormal(0, 1);
 * dist.mean();       // exp(mu + sigma^2/2) = exp(0.5) ≈ 1.6487
 * dist.pdf(1);       // density at x = 1
 * dist.cdf(1);       // P(X <= 1) = 0.5
 * dist.quantile(0.5); // median = exp(mu) = 1
 * dist.sample();     // random positive variate
 * ```
 */
export class LogNormal extends BaseContinuous {
  readonly name: string;

  /** @internal Underlying Normal distribution used for CDF, quantile, and sampling. */
  private readonly normalDist: Normal;

  /**
   * Creates a new Log-normal distribution.
   *
   * @param mu - Mean of the underlying normal distribution (log-scale location). Defaults to 0.
   * @param sigma - Standard deviation of the underlying normal distribution (log-scale spread, must be > 0). Defaults to 1.
   * @param rng - Optional random number generator returning values in [0, 1).
   * @throws {Error} If sigma is not positive.
   *
   * @example
   * ```ts
   * const dist = new LogNormal(0, 1);    // standard log-normal
   * const dist2 = new LogNormal(2, 0.5); // shifted, narrower
   * ```
   */
  constructor(
    public readonly mu: number = 0,
    public readonly sigma: number = 1,
    rng?: RandomFn,
  ) {
    super(rng);
    if (sigma <= 0) throw new Error(`Invalid parameter 'sigma': expected a positive number, received ${sigma}`);
    this.name = `LogNormal(${mu}, ${sigma})`;
    this.normalDist = new Normal(mu, sigma, rng);
  }

  /**
   * Returns the mean of the Log-normal distribution.
   *
   * Formula: E[X] = exp(mu + sigma^2 / 2)
   *
   * @returns The expected value.
   */
  mean(): number {
    return Math.exp(this.mu + this.sigma ** 2 / 2);
  }

  /**
   * Returns the variance of the Log-normal distribution.
   *
   * Formula: Var(X) = (exp(sigma^2) - 1) * exp(2*mu + sigma^2)
   *
   * @returns The variance.
   */
  variance(): number {
    const s2 = this.sigma ** 2;
    return (Math.exp(s2) - 1) * Math.exp(2 * this.mu + s2);
  }

  /**
   * Evaluates the probability density function (PDF) at x.
   *
   * PDF: f(x) = 1 / (x * sigma * sqrt(2*pi)) * exp(-(ln(x) - mu)^2 / (2*sigma^2))
   *
   * @param x - The point at which to evaluate the density.
   * @returns The probability density f(x) >= 0. Returns 0 for x <= 0.
   *
   * @example
   * ```ts
   * const dist = new LogNormal(0, 1);
   * dist.pdf(1);  // 1 / sqrt(2*pi) ≈ 0.3989
   * dist.pdf(-1); // 0
   * ```
   */
  pdf(x: number): number {
    if (x <= 0) return 0;
    const logX = Math.log(x);
    const z = (logX - this.mu) / this.sigma;
    return Math.exp(-0.5 * z * z) / (x * this.sigma * Math.sqrt(2 * Math.PI));
  }

  /**
   * Computes the log of the probability density function at `x`.
   *
   * log f(x) = -0.5 * ((ln(x) - mu) / sigma)^2 - log(x) - log(sigma) - 0.5 * log(2*pi)
   *
   * @param x - The point at which to evaluate the log-density.
   * @returns The log-density. Returns -Infinity for x <= 0.
   */
  logPdf(x: number): number {
    if (x <= 0) return -Infinity;
    const logX = Math.log(x);
    const z = (logX - this.mu) / this.sigma;
    return -0.5 * z * z - logX - Math.log(this.sigma) - 0.5 * Math.log(2 * Math.PI);
  }

  /**
   * Returns the skewness of the Log-normal distribution.
   *
   * Formula: (exp(sigma^2) + 2) * sqrt(exp(sigma^2) - 1)
   */
  get skewness(): number {
    const es2 = Math.exp(this.sigma ** 2);
    return (es2 + 2) * Math.sqrt(es2 - 1);
  }

  /**
   * Returns the excess kurtosis of the Log-normal distribution.
   *
   * Formula: exp(4*sigma^2) + 2*exp(3*sigma^2) + 3*exp(2*sigma^2) - 6
   */
  get kurtosis(): number {
    const s2 = this.sigma ** 2;
    return Math.exp(4 * s2) + 2 * Math.exp(3 * s2) + 3 * Math.exp(2 * s2) - 6;
  }

  /**
   * Returns the mode of the Log-normal distribution.
   *
   * Formula: exp(mu - sigma^2)
   */
  get mode(): number {
    return Math.exp(this.mu - this.sigma ** 2);
  }

  /**
   * Evaluates the cumulative distribution function (CDF) at x.
   *
   * Computed as the CDF of the underlying Normal distribution evaluated at ln(x):
   * F(x) = Phi((ln(x) - mu) / sigma)
   *
   * @param x - The point at which to evaluate the CDF.
   * @returns P(X <= x), a probability in [0, 1]. Returns 0 for x <= 0.
   *
   * @example
   * ```ts
   * const dist = new LogNormal(0, 1);
   * dist.cdf(1); // 0.5 (since ln(1) = 0 = mu)
   * ```
   */
  cdf(x: number): number {
    if (x <= 0) return 0;
    return this.normalDist.cdf(Math.log(x));
  }

  /**
   * Computes the quantile (inverse CDF) for a given probability.
   *
   * Formula: Q(p) = exp(Normal.quantile(p)), where Normal has parameters mu and sigma.
   *
   * @param p - A probability in [0, 1].
   * @returns The quantile value x > 0.
   * @throws {Error} If p is not in [0, 1].
   *
   * @example
   * ```ts
   * const dist = new LogNormal(0, 1);
   * dist.quantile(0.5);  // median = exp(mu) = 1
   * dist.quantile(0.95); // 95th percentile
   * ```
   */
  quantile(p: number): number {
    if (p < 0 || p > 1) throw new Error(`Invalid parameter 'p': expected a value in [0, 1], received ${p}`);
    if (p === 0) return 0;
    if (p === 1) return Infinity;
    return Math.exp(this.normalDist.quantile(p));
  }

  /**
   * Draws a single random sample from the Log-normal distribution.
   *
   * Generates a Normal(mu, sigma) variate and exponentiates it.
   *
   * @returns A positive random variate.
   */
  sample(): number {
    return Math.exp(this.normalDist.sample());
  }
}
