import { BaseContinuous } from "../base";
import { RandomFn } from "../../types";

/**
 * Log-Logistic distribution.
 *
 * Also known as the Fisk distribution in economics. If X ~ LogLogistic(alpha, beta),
 * then log(X) follows a logistic distribution. It is used in survival analysis,
 * hydrology, and income distribution modeling.
 *
 * PDF: f(x) = (beta/alpha) * (x/alpha)^(beta-1) / (1 + (x/alpha)^beta)^2
 *
 * CDF: F(x) = 1 / (1 + (x/alpha)^(-beta))
 *
 * Quantile: Q(p) = alpha * (p / (1 - p))^(1/beta)
 *
 * Support: x >= 0
 *
 * @example
 * ```ts
 * const dist = new LogLogistic(1, 2);
 * dist.mean();       // alpha * pi/beta / sin(pi/beta) (for beta > 1)
 * dist.pdf(1);       // density at x = 1
 * dist.cdf(1);       // 0.5 (median = alpha)
 * dist.quantile(0.5); // alpha = 1
 * dist.sample();     // random non-negative variate
 * ```
 */
export class LogLogistic extends BaseContinuous {
  readonly name: string;

  /**
   * Creates a new Log-Logistic distribution.
   *
   * @param alpha - Scale parameter (must be > 0). Defaults to 1.
   * @param beta - Shape parameter (must be > 0). Defaults to 1.
   * @param rng - Optional random number generator returning values in [0, 1).
   * @throws {Error} If alpha is not positive.
   * @throws {Error} If beta is not positive.
   */
  constructor(
    public readonly alpha: number = 1,
    public readonly beta: number = 1,
    rng?: RandomFn,
  ) {
    super(rng);
    if (alpha <= 0) throw new Error(`Invalid parameter 'alpha': expected a positive number, received ${alpha}`);
    if (beta <= 0) throw new Error(`Invalid parameter 'beta': expected a positive number, received ${beta}`);
    this.name = `LogLogistic(${alpha}, ${beta})`;
  }

  /**
   * Returns the mean of the Log-Logistic distribution.
   *
   * Formula: E[X] = alpha * (pi/beta) / sin(pi/beta) for beta > 1.
   *
   * @returns The expected value, or Infinity if beta <= 1.
   */
  mean(): number {
    if (this.beta <= 1) return Infinity;
    const piOverBeta = Math.PI / this.beta;
    return this.alpha * piOverBeta / Math.sin(piOverBeta);
  }

  /**
   * Returns the variance of the Log-Logistic distribution.
   *
   * Defined only for beta > 2.
   *
   * @returns The variance, or Infinity if beta <= 2.
   */
  variance(): number {
    if (this.beta <= 2) return Infinity;
    const b = this.beta;
    const piOverB = Math.PI / b;
    const m1 = piOverB / Math.sin(piOverB);
    const m2 = 2 * piOverB / Math.sin(2 * piOverB);
    return this.alpha ** 2 * (m2 - m1 * m1);
  }

  /**
   * Evaluates the probability density function at `x`.
   *
   * PDF: f(x) = (beta/alpha) * (x/alpha)^(beta-1) / (1 + (x/alpha)^beta)^2
   *
   * @param x - The point at which to evaluate the density.
   * @returns The probability density f(x) >= 0. Returns 0 for x < 0.
   */
  pdf(x: number): number {
    if (x < 0) return 0;
    if (x === 0) {
      if (this.beta === 1) return 1 / this.alpha;
      if (this.beta < 1) return Infinity;
      return 0;
    }
    const z = x / this.alpha;
    const zb = Math.pow(z, this.beta);
    const denom = (1 + zb) * (1 + zb);
    return (this.beta / this.alpha) * Math.pow(z, this.beta - 1) / denom;
  }

  /**
   * Computes the log of the probability density function at `x`.
   *
   * log f(x) = log(beta/alpha) + (beta-1)*log(x/alpha) - 2*log(1 + (x/alpha)^beta)
   *
   * @param x - The point at which to evaluate the log-density.
   * @returns The log-density. Returns -Infinity for x < 0.
   */
  logPdf(x: number): number {
    if (x < 0) return -Infinity;
    if (x === 0) {
      if (this.beta === 1) return -Math.log(this.alpha);
      if (this.beta < 1) return Infinity;
      return -Infinity;
    }
    const z = x / this.alpha;
    const zb = Math.pow(z, this.beta);
    return (
      Math.log(this.beta / this.alpha) +
      (this.beta - 1) * Math.log(z) -
      2 * Math.log(1 + zb)
    );
  }

  /**
   * Returns the skewness of the Log-Logistic distribution.
   *
   * Defined only for beta > 3.
   */
  get skewness(): number {
    if (this.beta <= 3) return NaN;
    const b = this.beta;
    const p1 = Math.PI / b;
    const m1 = p1 / Math.sin(p1);
    const m2 = 2 * p1 / Math.sin(2 * p1);
    const m3 = 3 * p1 / Math.sin(3 * p1);
    const sigma2 = m2 - m1 * m1;
    const sigma = Math.sqrt(sigma2);
    return (m3 - 3 * m1 * sigma2 - m1 * m1 * m1) / (sigma * sigma * sigma);
  }

  /**
   * Returns the excess kurtosis of the Log-Logistic distribution.
   *
   * Defined only for beta > 4.
   */
  get kurtosis(): number {
    if (this.beta <= 4) return NaN;
    const b = this.beta;
    const p1 = Math.PI / b;
    const m1 = p1 / Math.sin(p1);
    const m2 = 2 * p1 / Math.sin(2 * p1);
    const m3 = 3 * p1 / Math.sin(3 * p1);
    const m4 = 4 * p1 / Math.sin(4 * p1);
    const mu = m1;
    const sigma2 = m2 - m1 * m1;
    const sigma4 = sigma2 * sigma2;
    return (m4 - 4 * m3 * mu + 6 * m2 * mu * mu - 3 * mu * mu * mu * mu) / sigma4 - 3;
  }

  /**
   * Returns the mode of the Log-Logistic distribution.
   *
   * Formula: alpha * ((beta - 1) / (beta + 1))^(1/beta) for beta > 1, 0 for beta <= 1.
   */
  get mode(): number {
    if (this.beta <= 1) return 0;
    return this.alpha * Math.pow((this.beta - 1) / (this.beta + 1), 1 / this.beta);
  }

  /**
   * Evaluates the cumulative distribution function at `x`.
   *
   * CDF: F(x) = 1 / (1 + (x/alpha)^(-beta))
   *
   * @param x - The point at which to evaluate the CDF.
   * @returns P(X <= x) in [0, 1]. Returns 0 for x < 0.
   */
  cdf(x: number): number {
    if (x <= 0) return 0;
    const zb = Math.pow(x / this.alpha, this.beta);
    return zb / (1 + zb);
  }

  /**
   * Computes the quantile (inverse CDF) for a given probability.
   *
   * Formula: Q(p) = alpha * (p / (1 - p))^(1/beta)
   *
   * @param p - A probability in [0, 1].
   * @returns The quantile value x >= 0.
   * @throws {Error} If p is not in [0, 1].
   */
  quantile(p: number): number {
    if (p < 0 || p > 1) throw new Error(`Invalid parameter 'p': expected a value in [0, 1], received ${p}`);
    if (p === 0) return 0;
    if (p === 1) return Infinity;
    return this.alpha * Math.pow(p / (1 - p), 1 / this.beta);
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
