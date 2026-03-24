import { BaseContinuous } from "../base";
import { gammaLn, regularizedBeta, quantileBisect } from "../../utils/math";
import { GammaDistribution } from "./gamma";
import { RandomFn } from "../../types";

/**
 * Beta distribution defined on the interval [0, 1].
 *
 * The Beta distribution is parameterized by two positive shape parameters,
 * alpha and beta, and is commonly used to model proportions, probabilities,
 * and random variables bounded between 0 and 1.
 *
 * PDF: f(x; alpha, beta) = x^(alpha-1) * (1-x)^(beta-1) / B(alpha, beta)
 *
 * where B(alpha, beta) is the Beta function.
 *
 * @example
 * ```ts
 * const dist = new BetaDistribution(2, 5);
 * dist.mean();       // 0.2857...
 * dist.pdf(0.3);     // density at x = 0.3
 * dist.cdf(0.5);     // P(X <= 0.5)
 * dist.quantile(0.9); // 90th percentile
 * dist.sample();     // random variate in [0, 1]
 * ```
 */
export class BetaDistribution extends BaseContinuous {
  readonly name: string;

  /**
   * Creates a new Beta distribution.
   *
   * @param alpha - First shape parameter (must be > 0). Defaults to 1.
   * @param beta - Second shape parameter (must be > 0). Defaults to 1.
   * @param rng - Optional random number generator returning values in [0, 1).
   * @throws {Error} If alpha is not positive.
   * @throws {Error} If beta is not positive.
   *
   * @example
   * ```ts
   * const uniform = new BetaDistribution(1, 1); // uniform on [0,1]
   * const skewed = new BetaDistribution(2, 5);
   * ```
   */
  constructor(
    public readonly alpha: number = 1,
    public readonly beta: number = 1,
    rng?: RandomFn,
  ) {
    super(rng);
    if (alpha <= 0) throw new Error(`Invalid parameter 'alpha': expected a positive number, received ${alpha}`);
    if (beta <= 0) throw new Error(`Invalid parameter 'beta': expected a positive number, received ${beta}`);
    this.name = `Beta(${alpha}, ${beta})`;
  }

  /**
   * Returns the mean of the Beta distribution.
   *
   * Formula: E[X] = alpha / (alpha + beta)
   *
   * @returns The expected value.
   */
  mean(): number {
    return this.alpha / (this.alpha + this.beta);
  }

  /**
   * Returns the variance of the Beta distribution.
   *
   * Formula: Var(X) = (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1))
   *
   * @returns The variance.
   */
  variance(): number {
    const ab = this.alpha + this.beta;
    return (this.alpha * this.beta) / (ab * ab * (ab + 1));
  }

  /**
   * Evaluates the probability density function (PDF) at x.
   *
   * PDF: f(x) = x^(alpha-1) * (1-x)^(beta-1) / B(alpha, beta)
   *
   * Computed in log-space for numerical stability.
   *
   * @param x - The point at which to evaluate the density.
   * @returns The probability density f(x). Returns 0 for x outside [0, 1].
   *
   * @example
   * ```ts
   * const dist = new BetaDistribution(2, 5);
   * dist.pdf(0.3); // ~2.2689...
   * dist.pdf(-1);  // 0
   * ```
   */
  pdf(x: number): number {
    if (x < 0 || x > 1) return 0;
    if (x === 0) {
      if (this.alpha === 1) return this.beta;
      if (this.alpha < 1) return Infinity;
      return 0;
    }
    if (x === 1) {
      if (this.beta === 1) return this.alpha;
      if (this.beta < 1) return Infinity;
      return 0;
    }
    const logPdf =
      (this.alpha - 1) * Math.log(x) +
      (this.beta - 1) * Math.log(1 - x) -
      (gammaLn(this.alpha) + gammaLn(this.beta) - gammaLn(this.alpha + this.beta));
    return Math.exp(logPdf);
  }

  /**
   * Evaluates the cumulative distribution function (CDF) at x.
   *
   * Computed using the regularized incomplete Beta function I_x(alpha, beta).
   *
   * @param x - The point at which to evaluate the CDF.
   * @returns P(X <= x), a probability in [0, 1].
   *
   * @example
   * ```ts
   * const dist = new BetaDistribution(2, 5);
   * dist.cdf(0.5); // P(X <= 0.5)
   * ```
   */
  cdf(x: number): number {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    return regularizedBeta(x, this.alpha, this.beta);
  }

  /**
   * Computes the quantile (inverse CDF) for a given probability.
   *
   * Uses bisection search on the CDF to find the value x such that CDF(x) = p.
   *
   * @param p - A probability in [0, 1].
   * @returns The quantile value x in [0, 1].
   * @throws {Error} If p is not in [0, 1].
   *
   * @example
   * ```ts
   * const dist = new BetaDistribution(2, 5);
   * dist.quantile(0.5);  // median
   * dist.quantile(0.95); // 95th percentile
   * ```
   */
  quantile(p: number): number {
    if (p < 0 || p > 1) throw new Error(`Invalid parameter 'p': expected a value in [0, 1], received ${p}`);
    if (p === 0) return 0;
    if (p === 1) return 1;
    return quantileBisect((x) => this.cdf(x), p, 0, 1);
  }

  /**
   * Draws a single random sample from the Beta distribution.
   *
   * Uses the ratio of two independent Gamma variates:
   * if X ~ Gamma(alpha, 1) and Y ~ Gamma(beta, 1), then X/(X+Y) ~ Beta(alpha, beta).
   *
   * @returns A random variate in [0, 1].
   */
  sample(): number {
    const x = new GammaDistribution(this.alpha, 1, this.rng).sample();
    const y = new GammaDistribution(this.beta, 1, this.rng).sample();
    return x / (x + y);
  }
}
