import { BaseContinuous } from "../base";
import { RandomFn } from "../../types";

/**
 * Generalized Pareto Distribution (GPD).
 *
 * Used in extreme value theory (EVT) for modelling exceedances over a
 * threshold (peaks-over-threshold method). The GPD is the limiting
 * distribution of scaled excesses over a high threshold for a wide class
 * of underlying distributions.
 *
 * CDF: F(x) = 1 - (1 + xi*(x-mu)/sigma)^(-1/xi) for xi != 0,
 *      F(x) = 1 - exp(-(x-mu)/sigma) for xi = 0.
 *
 * Support:
 *   - xi >= 0: x >= mu (unbounded above)
 *   - xi < 0:  mu <= x <= mu - sigma/xi (bounded above)
 *
 * @example
 * ```ts
 * const dist = new GPD(0, 1, 0.2);
 * dist.mean();       // mu + sigma / (1 - xi)
 * dist.pdf(1);       // density at x = 1
 * dist.cdf(2);       // P(X <= 2)
 * dist.quantile(0.99); // 99th percentile
 * dist.sample();     // random variate
 * ```
 */
export class GPD extends BaseContinuous {
  readonly name: string;

  /**
   * Creates a new Generalized Pareto Distribution.
   *
   * @param mu - Location / threshold parameter. Defaults to 0.
   * @param sigma - Scale parameter (must be > 0). Defaults to 1.
   * @param xi - Shape / tail index parameter. Defaults to 0 (exponential tail).
   * @param rng - Optional random number generator returning values in [0, 1).
   * @throws {Error} If sigma is not positive.
   *
   * @example
   * ```ts
   * const exponential = new GPD(0, 1, 0);   // exponential distribution
   * const heavyTail = new GPD(0, 1, 0.5);   // heavy-tailed
   * const bounded = new GPD(0, 1, -0.5);    // bounded above at mu - sigma/xi = 2
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
    this.name = `GPD(${mu}, ${sigma}, ${xi})`;
  }

  /**
   * Returns the mean of the GPD.
   *
   * Formula: E[X] = mu + sigma / (1 - xi), defined only for xi < 1.
   *
   * @returns The expected value, or Infinity if xi >= 1.
   */
  mean(): number {
    if (this.xi >= 1) return Infinity;
    return this.mu + this.sigma / (1 - this.xi);
  }

  /**
   * Returns the variance of the GPD.
   *
   * Formula: Var(X) = sigma^2 / ((1 - xi)^2 * (1 - 2*xi)), defined only for xi < 0.5.
   *
   * @returns The variance, or Infinity if xi >= 0.5.
   */
  variance(): number {
    if (this.xi >= 0.5) return Infinity;
    return this.sigma ** 2 / ((1 - this.xi) ** 2 * (1 - 2 * this.xi));
  }

  /**
   * Checks whether x is within the support of the distribution.
   *
   * @param x - The point to check.
   * @returns True if x is in the support.
   */
  private isInSupport(x: number): boolean {
    if (x < this.mu) return false;
    if (this.xi < 0) return x <= this.mu - this.sigma / this.xi;
    return true;
  }

  /**
   * Evaluates the probability density function (PDF) at x.
   *
   * - xi = 0: f(x) = exp(-(x-mu)/sigma) / sigma
   * - xi != 0: f(x) = (1 + xi*(x-mu)/sigma)^(-1/xi - 1) / sigma
   *
   * @param x - The point at which to evaluate the density.
   * @returns The probability density f(x) >= 0. Returns 0 outside the support.
   *
   * @example
   * ```ts
   * const dist = new GPD(0, 1, 0.2);
   * dist.pdf(1); // density at x = 1
   * ```
   */
  pdf(x: number): number {
    if (!this.isInSupport(x)) return 0;
    const z = (x - this.mu) / this.sigma;
    if (this.xi === 0) {
      return Math.exp(-z) / this.sigma;
    }
    const v = 1 + this.xi * z;
    if (v <= 0) return 0;
    return Math.pow(v, -(1 / this.xi + 1)) / this.sigma;
  }

  /**
   * Computes the log of the probability density function at `x`.
   *
   * @param x - The point at which to evaluate the log-density.
   * @returns The log-density. Returns -Infinity outside the support.
   */
  logPdf(x: number): number {
    if (!this.isInSupport(x)) return -Infinity;
    const z = (x - this.mu) / this.sigma;
    if (this.xi === 0) {
      return -z - Math.log(this.sigma);
    }
    const v = 1 + this.xi * z;
    if (v <= 0) return -Infinity;
    return -(1 / this.xi + 1) * Math.log(v) - Math.log(this.sigma);
  }

  /**
   * Returns the mode of the GPD.
   *
   * The mode is always at the lower bound `mu`.
   */
  get mode(): number {
    return this.mu;
  }

  /**
   * Evaluates the cumulative distribution function (CDF) at x.
   *
   * - xi = 0: F(x) = 1 - exp(-(x-mu)/sigma)
   * - xi != 0: F(x) = 1 - (1 + xi*(x-mu)/sigma)^(-1/xi)
   *
   * @param x - The point at which to evaluate the CDF.
   * @returns P(X <= x), a probability in [0, 1].
   */
  cdf(x: number): number {
    if (x < this.mu) return 0;
    if (this.xi < 0 && x >= this.mu - this.sigma / this.xi) return 1;
    const z = (x - this.mu) / this.sigma;
    if (this.xi === 0) {
      return 1 - Math.exp(-z);
    }
    return 1 - Math.pow(1 + this.xi * z, -1 / this.xi);
  }

  /**
   * Computes the quantile (inverse CDF) for a given probability.
   *
   * - xi = 0: Q(p) = mu - sigma * ln(1 - p)
   * - xi != 0: Q(p) = mu + sigma * ((1 - p)^(-xi) - 1) / xi
   *
   * @param p - A probability in [0, 1].
   * @returns The quantile value.
   * @throws {Error} If p is not in [0, 1].
   *
   * @example
   * ```ts
   * const dist = new GPD(0, 1, 0.2);
   * dist.quantile(0.5);  // median
   * dist.quantile(0.99); // 99th percentile
   * ```
   */
  quantile(p: number): number {
    if (p < 0 || p > 1) throw new Error(`Invalid parameter 'p': expected a value in [0, 1], received ${p}`);
    if (p === 0) return this.mu;
    if (p === 1) {
      return this.xi < 0 ? this.mu - this.sigma / this.xi : Infinity;
    }
    if (this.xi === 0) {
      return this.mu - this.sigma * Math.log(1 - p);
    }
    return this.mu + (this.sigma * (Math.pow(1 - p, -this.xi) - 1)) / this.xi;
  }

  /**
   * Draws a single random sample from the GPD.
   *
   * Uses the inverse CDF method.
   *
   * @returns A random variate from the GPD.
   */
  sample(): number {
    return this.quantile(this.rng());
  }
}
