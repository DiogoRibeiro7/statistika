import { BaseContinuous } from "../base";
import { RandomFn } from "../../types";

/**
 * Pareto distribution (Type I).
 *
 * A heavy-tailed power-law distribution originally used to describe the
 * distribution of wealth, and now widely applied to model phenomena where
 * large values are rare but significant (e.g., city sizes, insurance claims,
 * file sizes).
 *
 * PDF: f(x; alpha, xm) = alpha * xm^alpha / x^(alpha + 1), for x >= xm
 *
 * CDF: F(x; alpha, xm) = 1 - (xm / x)^alpha
 *
 * Support: x >= xm
 *
 * @example
 * ```ts
 * const dist = new Pareto(3, 1);
 * dist.mean();       // alpha * xm / (alpha - 1) = 1.5
 * dist.pdf(2);       // density at x = 2
 * dist.cdf(2);       // P(X <= 2)
 * dist.quantile(0.8); // 80th percentile
 * dist.sample();     // random variate >= xm
 * ```
 */
export class Pareto extends BaseContinuous {
  readonly name: string;

  /**
   * Creates a new Pareto distribution.
   *
   * @param alpha - Shape parameter / tail exponent (must be > 0).
   * @param xm - Scale parameter / minimum value (must be > 0). Defaults to 1.
   * @param rng - Optional random number generator returning values in [0, 1).
   * @throws {Error} If alpha is not positive.
   * @throws {Error} If xm is not positive.
   *
   * @example
   * ```ts
   * const dist = new Pareto(2.5, 1);  // shape=2.5, minimum=1
   * ```
   */
  constructor(
    public readonly alpha: number,
    public readonly xm: number = 1,
    rng?: RandomFn,
  ) {
    super(rng);
    if (alpha <= 0) throw new Error(`Invalid parameter 'alpha': expected a positive number, received ${alpha}`);
    if (xm <= 0) throw new Error(`Invalid parameter 'xm': expected a positive number, received ${xm}`);
    this.name = `Pareto(${alpha}, ${xm})`;
  }

  /**
   * Returns the mean of the Pareto distribution.
   *
   * Formula: E[X] = alpha * xm / (alpha - 1), defined only for alpha > 1.
   *
   * @returns The expected value, or Infinity if alpha <= 1.
   */
  mean(): number {
    if (this.alpha <= 1) return Infinity;
    return (this.alpha * this.xm) / (this.alpha - 1);
  }

  /**
   * Returns the variance of the Pareto distribution.
   *
   * Formula: Var(X) = xm^2 * alpha / ((alpha - 1)^2 * (alpha - 2)),
   * defined only for alpha > 2.
   *
   * @returns The variance, or Infinity if alpha <= 2.
   */
  variance(): number {
    if (this.alpha <= 2) return Infinity;
    const { alpha, xm } = this;
    return (xm * xm * alpha) / ((alpha - 1) ** 2 * (alpha - 2));
  }

  /**
   * Evaluates the probability density function (PDF) at x.
   *
   * PDF: f(x) = alpha * xm^alpha / x^(alpha + 1)
   *
   * @param x - The point at which to evaluate the density.
   * @returns The probability density f(x) >= 0. Returns 0 for x < xm.
   *
   * @example
   * ```ts
   * const dist = new Pareto(3, 1);
   * dist.pdf(1);  // alpha = 3 (density at xm)
   * dist.pdf(0.5); // 0 (below support)
   * ```
   */
  pdf(x: number): number {
    if (x < this.xm) return 0;
    const { alpha, xm } = this;
    return (alpha * Math.pow(xm, alpha)) / Math.pow(x, alpha + 1);
  }

  /**
   * Evaluates the cumulative distribution function (CDF) at x.
   *
   * CDF: F(x) = 1 - (xm / x)^alpha
   *
   * @param x - The point at which to evaluate the CDF.
   * @returns P(X <= x), a probability in [0, 1]. Returns 0 for x < xm.
   */
  cdf(x: number): number {
    if (x < this.xm) return 0;
    return 1 - Math.pow(this.xm / x, this.alpha);
  }

  /**
   * Computes the quantile (inverse CDF) for a given probability.
   *
   * Formula: Q(p) = xm / (1 - p)^(1/alpha)
   *
   * @param p - A probability in [0, 1].
   * @returns The quantile value x >= xm.
   * @throws {Error} If p is not in [0, 1].
   *
   * @example
   * ```ts
   * const dist = new Pareto(3, 1);
   * dist.quantile(0);   // xm = 1
   * dist.quantile(0.5); // median
   * ```
   */
  quantile(p: number): number {
    if (p < 0 || p > 1) throw new Error(`Invalid parameter 'p': expected a value in [0, 1], received ${p}`);
    if (p === 0) return this.xm;
    if (p === 1) return Infinity;
    return this.xm / Math.pow(1 - p, 1 / this.alpha);
  }

  /**
   * Draws a single random sample from the Pareto distribution.
   *
   * Uses the inverse CDF method.
   *
   * @returns A random variate x >= xm.
   */
  sample(): number {
    const u = this.rng();
    return this.quantile(u);
  }
}
