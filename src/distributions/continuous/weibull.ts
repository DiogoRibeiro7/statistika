import { BaseContinuous } from "../base";
import { gamma } from "../../utils/math";
import { RandomFn } from "../../types";

/**
 * Weibull distribution.
 *
 * Widely used in reliability engineering, survival analysis, and failure
 * time modeling. The Weibull distribution is flexible enough to model
 * increasing (k > 1), decreasing (k < 1), or constant (k = 1) hazard rates.
 * When k = 1, it reduces to the exponential distribution.
 *
 * The Weibull minimum is the Type III extreme value distribution for minima.
 *
 * PDF: f(x; k, lambda) = (k/lambda) * (x/lambda)^(k-1) * exp(-(x/lambda)^k)
 *
 * CDF: F(x; k, lambda) = 1 - exp(-(x/lambda)^k)
 *
 * Support: x >= 0
 *
 * @example
 * ```ts
 * const dist = new Weibull(2, 1);
 * dist.mean();       // lambda * Gamma(1 + 1/k)
 * dist.pdf(0.5);     // density at x = 0.5
 * dist.cdf(1);       // P(X <= 1)
 * dist.quantile(0.63); // ~lambda (characteristic life)
 * dist.sample();     // random non-negative variate
 * ```
 */
export class Weibull extends BaseContinuous {
  readonly name: string;

  /**
   * Creates a new Weibull distribution.
   *
   * @param k - Shape parameter (must be > 0). Controls the failure rate behavior. Defaults to 1.
   *   - k < 1: decreasing hazard rate (infant mortality)
   *   - k = 1: constant hazard rate (exponential distribution)
   *   - k > 1: increasing hazard rate (aging/wear-out)
   * @param lambda - Scale parameter (must be > 0). Also called the characteristic life. Defaults to 1.
   * @param rng - Optional random number generator returning values in [0, 1).
   * @throws {Error} If k is not positive.
   * @throws {Error} If lambda is not positive.
   *
   * @example
   * ```ts
   * const exponential = new Weibull(1, 2); // k=1 gives exponential(rate=1/2)
   * const rayleigh = new Weibull(2, 1);    // k=2 gives Rayleigh-like shape
   * ```
   */
  constructor(
    public readonly k: number = 1,
    public readonly lambda: number = 1,
    rng?: RandomFn,
  ) {
    super(rng);
    if (k <= 0) throw new Error(`Invalid parameter 'k': expected a positive number, received ${k}`);
    if (lambda <= 0) throw new Error(`Invalid parameter 'lambda': expected a positive number, received ${lambda}`);
    this.name = `Weibull(${k}, ${lambda})`;
  }

  /**
   * Returns the mean of the Weibull distribution.
   *
   * Formula: E[X] = lambda * Gamma(1 + 1/k)
   *
   * @returns The expected value.
   */
  mean(): number {
    return this.lambda * gamma(1 + 1 / this.k);
  }

  /**
   * Returns the variance of the Weibull distribution.
   *
   * Formula: Var(X) = lambda^2 * (Gamma(1 + 2/k) - Gamma(1 + 1/k)^2)
   *
   * @returns The variance.
   */
  variance(): number {
    const g1 = gamma(1 + 1 / this.k);
    const g2 = gamma(1 + 2 / this.k);
    return this.lambda ** 2 * (g2 - g1 ** 2);
  }

  /**
   * Evaluates the probability density function (PDF) at x.
   *
   * PDF: f(x) = (k/lambda) * (x/lambda)^(k-1) * exp(-(x/lambda)^k)
   *
   * @param x - The point at which to evaluate the density.
   * @returns The probability density f(x) >= 0. Returns 0 for x < 0.
   *
   * @example
   * ```ts
   * const dist = new Weibull(2, 1);
   * dist.pdf(0.5); // density at x = 0.5
   * dist.pdf(-1);  // 0
   * ```
   */
  pdf(x: number): number {
    if (x < 0) return 0;
    if (x === 0) {
      if (this.k === 1) return 1 / this.lambda;
      if (this.k < 1) return Infinity;
      return 0;
    }
    const z = x / this.lambda;
    return (
      (this.k / this.lambda) *
      Math.pow(z, this.k - 1) *
      Math.exp(-Math.pow(z, this.k))
    );
  }

  /**
   * Computes the log of the probability density function at `x`.
   *
   * log f(x) = log(k/lambda) + (k-1)*log(x/lambda) - (x/lambda)^k
   *
   * @param x - The point at which to evaluate the log-density.
   * @returns The log-density. Returns -Infinity for x < 0.
   */
  logPdf(x: number): number {
    if (x < 0) return -Infinity;
    if (x === 0) {
      if (this.k === 1) return -Math.log(this.lambda);
      if (this.k < 1) return Infinity;
      return -Infinity;
    }
    const z = x / this.lambda;
    return (
      Math.log(this.k / this.lambda) +
      (this.k - 1) * Math.log(z) -
      Math.pow(z, this.k)
    );
  }

  /**
   * Returns the skewness of the Weibull distribution.
   *
   * Uses the formula involving gamma functions.
   */
  get skewness(): number {
    const g1 = gamma(1 + 1 / this.k);
    const g2 = gamma(1 + 2 / this.k);
    const g3 = gamma(1 + 3 / this.k);
    const mu = g1;
    const sigma2 = g2 - g1 * g1;
    const sigma = Math.sqrt(sigma2);
    return (g3 - 3 * mu * sigma2 - mu * mu * mu) / (sigma * sigma * sigma);
  }

  /**
   * Returns the excess kurtosis of the Weibull distribution.
   *
   * Uses the formula involving gamma functions.
   */
  get kurtosis(): number {
    const g1 = gamma(1 + 1 / this.k);
    const g2 = gamma(1 + 2 / this.k);
    const g3 = gamma(1 + 3 / this.k);
    const g4 = gamma(1 + 4 / this.k);
    const mu = g1;
    const sigma2 = g2 - g1 * g1;
    const sigma4 = sigma2 * sigma2;
    return (g4 - 4 * g3 * mu + 6 * g2 * mu * mu - 3 * mu * mu * mu * mu) / sigma4 - 3;
  }

  /**
   * Returns the mode of the Weibull distribution.
   *
   * Formula: lambda * ((k - 1) / k)^(1/k) for k > 1, 0 for k <= 1.
   */
  get mode(): number {
    if (this.k <= 1) return 0;
    return this.lambda * Math.pow((this.k - 1) / this.k, 1 / this.k);
  }

  /**
   * Evaluates the cumulative distribution function (CDF) at x.
   *
   * CDF: F(x) = 1 - exp(-(x/lambda)^k)
   *
   * @param x - The point at which to evaluate the CDF.
   * @returns P(X <= x), a probability in [0, 1]. Returns 0 for x < 0.
   *
   * @example
   * ```ts
   * const dist = new Weibull(2, 1);
   * dist.cdf(1); // 1 - exp(-1) ≈ 0.6321
   * ```
   */
  cdf(x: number): number {
    if (x < 0) return 0;
    return 1 - Math.exp(-Math.pow(x / this.lambda, this.k));
  }

  /**
   * Computes the quantile (inverse CDF) for a given probability.
   *
   * Formula: Q(p) = lambda * (-ln(1 - p))^(1/k)
   *
   * @param p - A probability in [0, 1].
   * @returns The quantile value x >= 0.
   * @throws {Error} If p is not in [0, 1].
   *
   * @example
   * ```ts
   * const dist = new Weibull(2, 1);
   * dist.quantile(0.5);  // median
   * dist.quantile(0.95); // 95th percentile
   * ```
   */
  quantile(p: number): number {
    if (p < 0 || p > 1) throw new Error(`Invalid parameter 'p': expected a value in [0, 1], received ${p}`);
    if (p === 0) return 0;
    if (p === 1) return Infinity;
    return this.lambda * Math.pow(-Math.log(1 - p), 1 / this.k);
  }

  /**
   * Draws a single random sample from the Weibull distribution.
   *
   * Uses the inverse CDF method.
   *
   * @returns A non-negative random variate.
   */
  sample(): number {
    return this.quantile(this.rng());
  }
}
