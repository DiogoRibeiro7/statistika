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

  /** Evaluates the probability density function (PDF) at x. */
  pdf(x: number): number {
    const logPdf =
      gammaLn((this.nu + 1) / 2) -
      gammaLn(this.nu / 2) -
      0.5 * Math.log(this.nu * Math.PI) -
      ((this.nu + 1) / 2) * Math.log(1 + (x * x) / this.nu);
    return Math.exp(logPdf);
  }

  /** Computes the log of the probability density function at `x`. */
  logPdf(x: number): number {
    return (
      gammaLn((this.nu + 1) / 2) -
      gammaLn(this.nu / 2) -
      0.5 * Math.log(this.nu * Math.PI) -
      ((this.nu + 1) / 2) * Math.log(1 + (x * x) / this.nu)
    );
  }

  /** Returns the skewness of the Student's t-distribution. */
  get skewness(): number {
    if (this.nu <= 3) return NaN;
    return 0;
  }

  /** Returns the excess kurtosis of the Student's t-distribution. */
  get kurtosis(): number {
    if (this.nu <= 2) return NaN;
    if (this.nu <= 4) return Infinity;
    return 6 / (this.nu - 4);
  }

  /** Returns the mode of the Student's t-distribution. */
  get mode(): number {
    return 0;
  }

  /** Evaluates the cumulative distribution function (CDF) at x. */
  cdf(x: number): number {
    const t2 = x * x;
    const xt = this.nu / (this.nu + t2);
    const ib = 0.5 * regularizedBeta(xt, this.nu / 2, 0.5);
    return x >= 0 ? 1 - ib : ib;
  }

  /**
   * Computes the quantile (inverse CDF) for a given probability.
   *
   * The positive half is bracketed adaptively before bisection. This avoids
   * using the theoretical variance as a search bound, because the variance is
   * infinite for 1 < nu <= 2 and undefined for nu <= 1 even though every
   * interior Student-t quantile is finite.
   */
  quantile(p: number): number {
    if (p < 0 || p > 1) throw new Error(`Invalid parameter 'p': expected a value in [0, 1], received ${p}`);
    if (p === 0) return -Infinity;
    if (p === 1) return Infinity;
    if (p === 0.5) return 0;

    // Use symmetry so the numerical search only needs the positive half-line.
    if (p < 0.5) return -this.quantile(1 - p);

    let upper = 1;
    while (this.cdf(upper) < p && upper < Number.MAX_VALUE / 2) {
      upper *= 2;
    }

    return quantileBisect((x) => this.cdf(x), p, 0, upper);
  }

  /** Draws a single random sample from the Student's t-distribution. */
  sample(): number {
    const u1 = this.rng();
    const u2 = this.rng();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    const chi2 = new GammaDistribution(this.nu / 2, 0.5, this.rng).sample();
    return z / Math.sqrt(chi2 / this.nu);
  }
}
