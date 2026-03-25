import { BaseContinuous } from "../base";
import { gamma, quantileBisect } from "../../utils/math";
import { RandomFn } from "../../types";

/**
 * Generalized Extreme Value (GEV) distribution.
 *
 * Unifies the three extreme value types into a single family controlled
 * by the shape parameter xi:
 *   - xi = 0: Gumbel (Type I) -- light-tailed (exponential decay)
 *   - xi > 0: Frechet (Type II) -- heavy-tailed (polynomial decay)
 *   - xi < 0: Reversed Weibull (Type III) -- bounded upper tail
 *
 * The GEV distribution is fundamental to extreme value theory and is used
 * to model block maxima (e.g., annual maximum river levels, wind speeds).
 *
 * CDF: F(x) = exp(-t(x)), where t(x) = (1 + xi*(x-mu)/sigma)^(-1/xi) for xi != 0,
 * and t(x) = exp(-(x-mu)/sigma) for xi = 0.
 *
 * @example
 * ```ts
 * const gumbel = new GEV(0, 1, 0);    // Gumbel type
 * const frechet = new GEV(0, 1, 0.5); // Frechet type (heavy tail)
 * const weibull = new GEV(0, 1, -0.5); // Reversed Weibull type (bounded)
 * gumbel.cdf(2);       // P(X <= 2)
 * gumbel.quantile(0.99); // 99th percentile
 * ```
 */
export class GEV extends BaseContinuous {
  readonly name: string;

  /**
   * Creates a new Generalized Extreme Value distribution.
   *
   * @param mu - Location parameter. Defaults to 0.
   * @param sigma - Scale parameter (must be > 0). Defaults to 1.
   * @param xi - Shape parameter controlling the tail type. Defaults to 0 (Gumbel).
   * @param rng - Optional random number generator returning values in [0, 1).
   * @throws {Error} If sigma is not positive.
   *
   * @example
   * ```ts
   * const dist = new GEV(100, 10, 0.1);
   * ```
   */
  constructor(
    public readonly mu: number = 0,
    public readonly sigma: number = 1,
    public readonly xi: number = 0,
    rng?: RandomFn,
  ) {
    super(rng);
    if (sigma <= 0) throw new Error(`Invalid parameter 'sigma': expected a positive number, received ${sigma}`);
    this.name = `GEV(${mu}, ${sigma}, ${xi})`;
  }

  /**
   * Returns the mean of the GEV distribution.
   *
   * - xi = 0: E[X] = mu + sigma * gamma_EM (Euler-Mascheroni constant ~0.5772)
   * - xi != 0, xi < 1: E[X] = mu + sigma * (Gamma(1 - xi) - 1) / xi
   * - xi >= 1: E[X] = Infinity
   *
   * @returns The expected value, or Infinity if xi >= 1.
   */
  mean(): number {
    if (this.xi >= 1) return Infinity;
    if (this.xi === 0) {
      // Euler-Mascheroni constant
      return this.mu + this.sigma * 0.5772156649015329;
    }
    // mu + sigma * (gamma(1 - xi) - 1) / xi
    return this.mu + (this.sigma * (gamma(1 - this.xi) - 1)) / this.xi;
  }

  /**
   * Returns the variance of the GEV distribution.
   *
   * - xi = 0: Var(X) = sigma^2 * pi^2 / 6
   * - xi != 0, xi < 0.5: Var(X) = sigma^2 * (Gamma(1-2*xi) - Gamma(1-xi)^2) / xi^2
   * - xi >= 0.5: Var(X) = Infinity
   *
   * @returns The variance, or Infinity if xi >= 0.5.
   */
  variance(): number {
    if (this.xi >= 0.5) return Infinity;
    if (this.xi === 0) {
      return (this.sigma ** 2 * Math.PI ** 2) / 6;
    }
    const g1 = gamma(1 - this.xi);
    const g2 = gamma(1 - 2 * this.xi);
    return (this.sigma ** 2 * (g2 - g1 ** 2)) / this.xi ** 2;
  }

  /**
   * Computes the auxiliary function t(x) used in the PDF and CDF.
   *
   * t(x) = (1 + xi*(x-mu)/sigma)^(-1/xi) for xi != 0,
   * t(x) = exp(-(x-mu)/sigma) for xi = 0.
   *
   * @param x - The point at which to evaluate.
   * @returns The t(x) value.
   */
  private t(x: number): number {
    const z = (x - this.mu) / this.sigma;
    if (this.xi === 0) return Math.exp(-z);
    const v = 1 + this.xi * z;
    if (v <= 0) return this.xi > 0 ? Infinity : 0;
    return Math.pow(v, -1 / this.xi);
  }

  /**
   * Checks whether x is within the support of the distribution.
   *
   * - xi = 0: all real numbers
   * - xi > 0: x > mu - sigma/xi
   * - xi < 0: x < mu - sigma/xi
   *
   * @param x - The point to check.
   * @returns True if x is in the support.
   */
  private isInSupport(x: number): boolean {
    if (this.xi === 0) return true;
    const z = (x - this.mu) / this.sigma;
    return 1 + this.xi * z > 0;
  }

  /**
   * Evaluates the probability density function (PDF) at x.
   *
   * PDF: f(x) = t(x)^(xi+1) * exp(-t(x)) / sigma
   *
   * @param x - The point at which to evaluate the density.
   * @returns The probability density f(x) >= 0. Returns 0 outside the support.
   *
   * @example
   * ```ts
   * const dist = new GEV(0, 1, 0);
   * dist.pdf(0); // Gumbel density at 0
   * ```
   */
  pdf(x: number): number {
    if (!this.isInSupport(x)) return 0;
    const tx = this.t(x);
    if (!isFinite(tx)) return 0;
    return (tx ** (this.xi + 1) * Math.exp(-tx)) / this.sigma;
  }

  /**
   * Evaluates the cumulative distribution function (CDF) at x.
   *
   * CDF: F(x) = exp(-t(x))
   *
   * @param x - The point at which to evaluate the CDF.
   * @returns P(X <= x), a probability in [0, 1].
   *
   * @example
   * ```ts
   * const dist = new GEV(0, 1, 0);
   * dist.cdf(0); // P(X <= 0) for Gumbel
   * ```
   */
  cdf(x: number): number {
    if (this.xi > 0 && x <= this.mu - this.sigma / this.xi) return 0;
    if (this.xi < 0 && x >= this.mu - this.sigma / this.xi) return 1;
    return Math.exp(-this.t(x));
  }

  /**
   * Computes the quantile (inverse CDF) for a given probability.
   *
   * - xi = 0: Q(p) = mu - sigma * ln(-ln(p))
   * - xi != 0: Q(p) = mu + sigma * ((-ln(p))^(-xi) - 1) / xi
   *
   * @param p - A probability in [0, 1].
   * @returns The quantile value.
   * @throws {Error} If p is not in [0, 1].
   *
   * @example
   * ```ts
   * const dist = new GEV(0, 1, 0);
   * dist.quantile(0.5);  // median
   * dist.quantile(0.99); // 99th percentile (return level)
   * ```
   */
  quantile(p: number): number {
    if (p < 0 || p > 1) throw new Error(`Invalid parameter 'p': expected a value in [0, 1], received ${p}`);
    if (p === 0) {
      return this.xi > 0 ? this.mu - this.sigma / this.xi : -Infinity;
    }
    if (p === 1) {
      return this.xi < 0 ? this.mu - this.sigma / this.xi : Infinity;
    }
    const lnp = -Math.log(p);
    if (this.xi === 0) {
      return this.mu - this.sigma * Math.log(lnp);
    }
    return this.mu + (this.sigma * (Math.pow(lnp, -this.xi) - 1)) / this.xi;
  }

  /**
   * Draws a single random sample from the GEV distribution.
   *
   * Uses the inverse CDF method.
   *
   * @returns A random variate from the GEV distribution.
   */
  sample(): number {
    return this.quantile(this.rng());
  }
}
