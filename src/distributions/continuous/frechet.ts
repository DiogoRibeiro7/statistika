import { BaseContinuous } from "../base";
import { gamma } from "../../utils/math";
import { RandomFn } from "../../types";

/**
 * Frechet distribution (Type II extreme value distribution).
 *
 * Models the distribution of the maximum of samples from heavy-tailed
 * distributions (e.g., Pareto, Cauchy). The Frechet distribution is used
 * in extreme value theory to model maxima with polynomial tail decay.
 *
 * PDF: f(x; alpha, s, m) = (alpha/s) * ((x-m)/s)^(-1-alpha) * exp(-((x-m)/s)^(-alpha))
 *
 * Support: x > m
 *
 * Parameters:
 *   alpha -- shape (> 0), also called the tail index
 *   s     -- scale (> 0)
 *   m     -- location
 *
 * @example
 * ```ts
 * const dist = new Frechet(2, 1, 0);
 * dist.mean();       // m + s * Gamma(1 - 1/alpha)
 * dist.pdf(1.5);     // density at x = 1.5
 * dist.cdf(2);       // P(X <= 2)
 * dist.quantile(0.9); // 90th percentile
 * dist.sample();     // random variate > m
 * ```
 */
export class Frechet extends BaseContinuous {
  readonly name: string;

  /**
   * Creates a new Frechet distribution.
   *
   * @param alpha - Shape parameter / tail index (must be > 0). Defaults to 1.
   * @param s - Scale parameter (must be > 0). Defaults to 1.
   * @param m - Location parameter. Defaults to 0.
   * @param rng - Optional random number generator returning values in [0, 1).
   * @throws {Error} If alpha is not positive.
   * @throws {Error} If s is not positive.
   *
   * @example
   * ```ts
   * const dist = new Frechet(3, 2, 0);
   * ```
   */
  constructor(
    public readonly alpha: number = 1,
    public readonly s: number = 1,
    public readonly m: number = 0,
    rng?: RandomFn,
  ) {
    super(rng);
    if (alpha <= 0) throw new Error(`Invalid parameter 'alpha': expected a positive number, received ${alpha}`);
    if (s <= 0) throw new Error(`Invalid parameter 's': expected a positive number, received ${s}`);
    this.name = `Frechet(${alpha}, ${s}, ${m})`;
  }

  /**
   * Returns the mean of the Frechet distribution.
   *
   * Formula: E[X] = m + s * Gamma(1 - 1/alpha), defined only for alpha > 1.
   *
   * @returns The expected value, or Infinity if alpha <= 1.
   */
  mean(): number {
    if (this.alpha <= 1) return Infinity;
    return this.m + this.s * gamma(1 - 1 / this.alpha);
  }

  /**
   * Returns the variance of the Frechet distribution.
   *
   * Formula: Var(X) = s^2 * (Gamma(1 - 2/alpha) - Gamma(1 - 1/alpha)^2),
   * defined only for alpha > 2.
   *
   * @returns The variance, or Infinity if alpha <= 2.
   */
  variance(): number {
    if (this.alpha <= 2) return Infinity;
    const g1 = gamma(1 - 1 / this.alpha);
    const g2 = gamma(1 - 2 / this.alpha);
    return this.s ** 2 * (g2 - g1 ** 2);
  }

  /**
   * Evaluates the probability density function (PDF) at x.
   *
   * PDF: f(x) = (alpha/s) * ((x-m)/s)^(-1-alpha) * exp(-((x-m)/s)^(-alpha))
   *
   * @param x - The point at which to evaluate the density.
   * @returns The probability density f(x) >= 0. Returns 0 for x <= m.
   *
   * @example
   * ```ts
   * const dist = new Frechet(2, 1, 0);
   * dist.pdf(1); // density at x = 1
   * dist.pdf(0); // 0 (below support)
   * ```
   */
  pdf(x: number): number {
    if (x <= this.m) return 0;
    const z = (x - this.m) / this.s;
    return (
      (this.alpha / this.s) *
      Math.pow(z, -1 - this.alpha) *
      Math.exp(-Math.pow(z, -this.alpha))
    );
  }

  /**
   * Computes the log of the probability density function at `x`.
   *
   * log f(x) = log(alpha/s) + (-1 - alpha)*log((x-m)/s) - ((x-m)/s)^(-alpha)
   *
   * @param x - The point at which to evaluate the log-density.
   * @returns The log-density. Returns -Infinity for x <= m.
   */
  logPdf(x: number): number {
    if (x <= this.m) return -Infinity;
    const z = (x - this.m) / this.s;
    return (
      Math.log(this.alpha / this.s) +
      (-1 - this.alpha) * Math.log(z) -
      Math.pow(z, -this.alpha)
    );
  }

  /**
   * Returns the skewness of the Frechet distribution.
   *
   * Defined only for alpha > 3. Uses gamma function formula.
   */
  get skewness(): number {
    if (this.alpha <= 3) return NaN;
    const g1 = gamma(1 - 1 / this.alpha);
    const g2 = gamma(1 - 2 / this.alpha);
    const g3 = gamma(1 - 3 / this.alpha);
    const sigma2 = g2 - g1 * g1;
    const sigma = Math.sqrt(sigma2);
    return (g3 - 3 * g1 * sigma2 - g1 * g1 * g1) / (sigma * sigma * sigma);
  }

  /**
   * Returns the excess kurtosis of the Frechet distribution.
   *
   * Defined only for alpha > 4. Uses gamma function formula.
   */
  get kurtosis(): number {
    if (this.alpha <= 4) return NaN;
    const g1 = gamma(1 - 1 / this.alpha);
    const g2 = gamma(1 - 2 / this.alpha);
    const g3 = gamma(1 - 3 / this.alpha);
    const g4 = gamma(1 - 4 / this.alpha);
    const mu = g1;
    const sigma2 = g2 - g1 * g1;
    const sigma4 = sigma2 * sigma2;
    return (g4 - 4 * g3 * mu + 6 * g2 * mu * mu - 3 * mu * mu * mu * mu) / sigma4 - 3;
  }

  /**
   * Returns the mode of the Frechet distribution.
   *
   * Formula: m + s * (alpha / (1 + alpha))^(1/alpha)
   */
  get mode(): number {
    return this.m + this.s * Math.pow(this.alpha / (1 + this.alpha), 1 / this.alpha);
  }

  /**
   * Evaluates the cumulative distribution function (CDF) at x.
   *
   * CDF: F(x) = exp(-((x-m)/s)^(-alpha))
   *
   * @param x - The point at which to evaluate the CDF.
   * @returns P(X <= x), a probability in [0, 1]. Returns 0 for x <= m.
   */
  cdf(x: number): number {
    if (x <= this.m) return 0;
    const z = (x - this.m) / this.s;
    return Math.exp(-Math.pow(z, -this.alpha));
  }

  /**
   * Computes the quantile (inverse CDF) for a given probability.
   *
   * Formula: Q(p) = m + s * (-ln(p))^(-1/alpha)
   *
   * @param p - A probability in [0, 1].
   * @returns The quantile value x > m.
   * @throws {Error} If p is not in [0, 1].
   *
   * @example
   * ```ts
   * const dist = new Frechet(2, 1, 0);
   * dist.quantile(0.5);  // median
   * dist.quantile(0.99); // 99th percentile
   * ```
   */
  quantile(p: number): number {
    if (p < 0 || p > 1) throw new Error(`Invalid parameter 'p': expected a value in [0, 1], received ${p}`);
    if (p === 0) return this.m;
    if (p === 1) return Infinity;
    return this.m + this.s * Math.pow(-Math.log(p), -1 / this.alpha);
  }

  /**
   * Draws a single random sample from the Frechet distribution.
   *
   * Uses the inverse CDF method.
   *
   * @returns A random variate x > m.
   */
  sample(): number {
    return this.quantile(this.rng());
  }
}
