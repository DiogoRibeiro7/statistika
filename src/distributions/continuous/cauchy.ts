import { BaseContinuous } from "../base";
import { RandomFn } from "../../types";

/**
 * Cauchy distribution (also known as the Lorentz distribution).
 *
 * A heavy-tailed distribution that arises as the ratio of two independent
 * standard normal random variables. The Cauchy distribution has no defined
 * mean or variance because its tails are so heavy that the integrals diverge.
 *
 * PDF: f(x; x0, gamma) = 1 / (pi * gamma * (1 + ((x - x0) / gamma)^2))
 *
 * It is also a special case of the Student's t-distribution with 1 degree of freedom.
 *
 * @example
 * ```ts
 * const dist = new Cauchy(0, 1); // standard Cauchy
 * dist.pdf(0);        // peak density
 * dist.cdf(0);        // 0.5
 * dist.quantile(0.75); // third quartile
 * dist.sample();      // random variate
 * ```
 */
export class Cauchy extends BaseContinuous {
  readonly name: string;

  /**
   * Creates a new Cauchy distribution.
   *
   * @param x0 - Location parameter (median). Defaults to 0.
   * @param gammaParam - Scale parameter (half-width at half-maximum, must be > 0). Defaults to 1.
   * @param rng - Optional random number generator returning values in [0, 1).
   * @throws {Error} If gammaParam is not positive.
   *
   * @example
   * ```ts
   * const standard = new Cauchy();        // x0=0, gamma=1
   * const shifted = new Cauchy(5, 2);     // x0=5, gamma=2
   * ```
   */
  constructor(
    public readonly x0: number = 0,
    public readonly gammaParam: number = 1,
    rng?: RandomFn,
  ) {
    super(rng);
    if (gammaParam <= 0) throw new Error("gamma must be positive");
    this.name = `Cauchy(${x0}, ${gammaParam})`;
  }

  /**
   * Returns the mean of the Cauchy distribution.
   *
   * The mean is undefined (NaN) for the Cauchy distribution because
   * the integral E[X] = integral of x * f(x) does not converge.
   *
   * @returns NaN (undefined).
   */
  mean(): number {
    return NaN; // undefined for Cauchy
  }

  /**
   * Returns the variance of the Cauchy distribution.
   *
   * The variance is undefined (NaN) for the Cauchy distribution because
   * the second moment does not exist.
   *
   * @returns NaN (undefined).
   */
  variance(): number {
    return NaN; // undefined for Cauchy
  }

  /**
   * Evaluates the probability density function (PDF) at x.
   *
   * PDF: f(x) = 1 / (pi * gamma * (1 + ((x - x0) / gamma)^2))
   *
   * @param x - The point at which to evaluate the density.
   * @returns The probability density f(x) >= 0.
   *
   * @example
   * ```ts
   * const dist = new Cauchy(0, 1);
   * dist.pdf(0);  // 1 / pi ≈ 0.3183
   * ```
   */
  pdf(x: number): number {
    const z = (x - this.x0) / this.gammaParam;
    return 1 / (Math.PI * this.gammaParam * (1 + z * z));
  }

  /**
   * Evaluates the cumulative distribution function (CDF) at x.
   *
   * CDF: F(x) = 0.5 + arctan((x - x0) / gamma) / pi
   *
   * @param x - The point at which to evaluate the CDF.
   * @returns P(X <= x), a probability in [0, 1].
   *
   * @example
   * ```ts
   * const dist = new Cauchy(0, 1);
   * dist.cdf(0);  // 0.5
   * ```
   */
  cdf(x: number): number {
    return 0.5 + Math.atan((x - this.x0) / this.gammaParam) / Math.PI;
  }

  /**
   * Computes the quantile (inverse CDF) for a given probability.
   *
   * Formula: Q(p) = x0 + gamma * tan(pi * (p - 0.5))
   *
   * @param p - A probability in [0, 1].
   * @returns The quantile value.
   * @throws {Error} If p is not in [0, 1].
   *
   * @example
   * ```ts
   * const dist = new Cauchy(0, 1);
   * dist.quantile(0.5);  // 0 (median)
   * dist.quantile(0.75); // 1 (third quartile)
   * ```
   */
  quantile(p: number): number {
    if (p < 0 || p > 1) throw new Error("p must be in [0, 1]");
    if (p === 0) return -Infinity;
    if (p === 1) return Infinity;
    return this.x0 + this.gammaParam * Math.tan(Math.PI * (p - 0.5));
  }

  /**
   * Draws a single random sample from the Cauchy distribution.
   *
   * Uses the inverse CDF method: generates a uniform variate and
   * applies the quantile function.
   *
   * @returns A random variate from the Cauchy distribution.
   */
  sample(): number {
    // Inverse CDF method
    const u = this.rng();
    return this.quantile(u);
  }
}
