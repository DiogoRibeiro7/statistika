import { BaseContinuous } from "../base";
import { gammaLn, regularizedBeta, quantileBisect } from "../../utils/math";
import { GammaDistribution } from "./gamma";
import { RandomFn } from "../../types";

/**
 * Student's t-distribution.
 *
 * A symmetric, bell-shaped distribution that arises when estimating the mean
 * of a normally distributed population with unknown variance using a small
 * sample size. It has heavier tails than the normal distribution, with the
 * tail weight controlled by the degrees of freedom parameter nu.
 *
 * PDF: f(x; nu) = Gamma((nu+1)/2) / (sqrt(nu*pi) * Gamma(nu/2)) * (1 + x^2/nu)^(-(nu+1)/2)
 *
 * As nu approaches infinity, the t-distribution converges to the standard normal.
 * With nu = 1, it is equivalent to the Cauchy distribution.
 *
 * @example
 * ```ts
 * const dist = new StudentT(10);
 * dist.mean();       // 0
 * dist.variance();   // nu / (nu - 2) = 1.25
 * dist.pdf(0);       // peak density
 * dist.cdf(1.96);    // P(X <= 1.96)
 * dist.quantile(0.975); // critical value for two-tailed 95% test
 * ```
 */
export class StudentT extends BaseContinuous {
  readonly name: string;

  /**
   * Creates a new Student's t-distribution.
   *
   * @param nu - Degrees of freedom (must be > 0). Defaults to 1 (Cauchy distribution).
   * @param rng - Optional random number generator returning values in [0, 1).
   * @throws {Error} If nu is not positive.
   *
   * @example
   * ```ts
   * const t10 = new StudentT(10);
   * const cauchy = new StudentT(1); // equivalent to Cauchy(0, 1)
   * ```
   */
  constructor(public readonly nu: number = 1, rng?: RandomFn) {
    super(rng);
    if (nu <= 0) throw new Error(`Invalid parameter 'nu': expected a positive number, received ${nu}`);
    this.name = `StudentT(${nu})`;
  }

  /**
   * Returns the mean of the Student's t-distribution.
   *
   * The mean is 0 for nu > 1, and undefined (NaN) for nu <= 1.
   *
   * @returns 0 if nu > 1, NaN otherwise.
   */
  mean(): number {
    if (this.nu <= 1) return NaN;
    return 0;
  }

  /**
   * Returns the variance of the Student's t-distribution.
   *
   * - nu > 2: Var(X) = nu / (nu - 2)
   * - 1 < nu <= 2: Var(X) = Infinity
   * - nu <= 1: Var(X) = NaN (undefined)
   *
   * @returns The variance, Infinity, or NaN depending on nu.
   */
  variance(): number {
    if (this.nu <= 2) return this.nu > 1 ? Infinity : NaN;
    return this.nu / (this.nu - 2);
  }

  /**
   * Evaluates the probability density function (PDF) at x.
   *
   * PDF: f(x) = Gamma((nu+1)/2) / (sqrt(nu*pi) * Gamma(nu/2)) * (1 + x^2/nu)^(-(nu+1)/2)
   *
   * Computed in log-space for numerical stability.
   *
   * @param x - The point at which to evaluate the density.
   * @returns The probability density f(x) >= 0.
   *
   * @example
   * ```ts
   * const dist = new StudentT(5);
   * dist.pdf(0);  // peak density at the center
   * dist.pdf(3);  // density in the tails
   * ```
   */
  pdf(x: number): number {
    const logPdf =
      gammaLn((this.nu + 1) / 2) -
      gammaLn(this.nu / 2) -
      0.5 * Math.log(this.nu * Math.PI) -
      ((this.nu + 1) / 2) * Math.log(1 + (x * x) / this.nu);
    return Math.exp(logPdf);
  }

  /**
   * Computes the log of the probability density function at `x`.
   *
   * log f(x) = gammaLn((nu+1)/2) - gammaLn(nu/2) - 0.5*log(nu*pi)
   *            - ((nu+1)/2) * log(1 + x^2/nu)
   *
   * @param x - The point at which to evaluate the log-density.
   * @returns The log-density.
   */
  logPdf(x: number): number {
    return (
      gammaLn((this.nu + 1) / 2) -
      gammaLn(this.nu / 2) -
      0.5 * Math.log(this.nu * Math.PI) -
      ((this.nu + 1) / 2) * Math.log(1 + (x * x) / this.nu)
    );
  }

  /**
   * Returns the skewness of the Student's t-distribution.
   *
   * The skewness is 0 for nu > 3, and undefined for nu <= 3.
   */
  get skewness(): number {
    if (this.nu <= 3) return NaN;
    return 0;
  }

  /**
   * Returns the excess kurtosis of the Student's t-distribution.
   *
   * Formula: 6 / (nu - 4) for nu > 4.
   * Returns Infinity for 2 < nu <= 4, NaN for nu <= 2.
   */
  get kurtosis(): number {
    if (this.nu <= 2) return NaN;
    if (this.nu <= 4) return Infinity;
    return 6 / (this.nu - 4);
  }

  /**
   * Returns the mode of the Student's t-distribution.
   *
   * The mode is always 0.
   */
  get mode(): number {
    return 0;
  }

  /**
   * Evaluates the cumulative distribution function (CDF) at x.
   *
   * Computed using the regularized incomplete Beta function:
   * F(x) = 1 - 0.5 * I_{nu/(nu+x^2)}(nu/2, 1/2) for x >= 0,
   * and F(x) = 0.5 * I_{nu/(nu+x^2)}(nu/2, 1/2) for x < 0.
   *
   * @param x - The point at which to evaluate the CDF.
   * @returns P(X <= x), a probability in [0, 1].
   *
   * @example
   * ```ts
   * const dist = new StudentT(10);
   * dist.cdf(0);    // 0.5 (symmetric about 0)
   * dist.cdf(2.23); // ~0.975 for a 95% confidence interval
   * ```
   */
  cdf(x: number): number {
    const t2 = x * x;
    const xt = this.nu / (this.nu + t2);
    const ib = 0.5 * regularizedBeta(xt, this.nu / 2, 0.5);
    return x >= 0 ? 1 - ib : ib;
  }

  /**
   * Computes the quantile (inverse CDF) for a given probability.
   *
   * Uses the symmetry property (Q(p) = -Q(1-p) for p < 0.5) and
   * bisection search on the CDF.
   *
   * @param p - A probability in [0, 1].
   * @returns The quantile value.
   * @throws {Error} If p is not in [0, 1].
   *
   * @example
   * ```ts
   * const dist = new StudentT(10);
   * dist.quantile(0.5);   // 0 (median)
   * dist.quantile(0.975); // t critical value for 95% two-sided test
   * ```
   */
  quantile(p: number): number {
    if (p < 0 || p > 1) throw new Error(`Invalid parameter 'p': expected a value in [0, 1], received ${p}`);
    if (p === 0) return -Infinity;
    if (p === 1) return Infinity;
    // Use symmetry
    if (p < 0.5) return -this.quantile(1 - p);
    const upper = 10 * Math.sqrt(this.variance() || 100);
    return quantileBisect((x) => this.cdf(x), p, 0, upper);
  }

  /**
   * Draws a single random sample from the Student's t-distribution.
   *
   * Uses the representation T = Z / sqrt(V/nu), where Z ~ Normal(0,1)
   * generated via the Box-Muller transform, and V ~ Chi-squared(nu)
   * generated via a Gamma distribution.
   *
   * @returns A random variate from the t-distribution.
   */
  sample(): number {
    // Ratio of standard normal to sqrt(chi-squared / nu)
    const u1 = this.rng();
    const u2 = this.rng();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    // Chi-squared with nu degrees of freedom via sum of squared normals
    // For efficiency, use gamma sampling
    const chi2 = new GammaDistribution(this.nu / 2, 0.5, this.rng).sample();
    return z / Math.sqrt(chi2 / this.nu);
  }
}
