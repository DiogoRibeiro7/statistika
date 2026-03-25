import { BaseContinuous } from "../base";
import { erf, quantileBisect } from "../../utils/math";
import { RandomFn } from "../../types";

/**
 * Normal (Gaussian) distribution with mean `mu` and standard deviation `sigma`.
 *
 * The PDF is defined as:
 *
 *   f(x) = (1 / (sigma * sqrt(2*pi))) * exp(-0.5 * ((x - mu) / sigma)^2)
 *
 * Support: (-Infinity, +Infinity)
 *
 * @example
 * ```ts
 * const dist = new Normal(0, 1);
 * dist.pdf(0);      // ~0.3989
 * dist.cdf(1.96);   // ~0.975
 * dist.sample();
 * ```
 */
export class Normal extends BaseContinuous {
  readonly name: string;

  /**
   * Creates a Normal distribution.
   * @param mu - Mean (location parameter). Defaults to 0.
   * @param sigma - Standard deviation (scale parameter, must be > 0). Defaults to 1.
   * @param rng - Optional random number generator.
   * @throws If `sigma` is not positive.
   */
  constructor(
    public readonly mu: number = 0,
    public readonly sigma: number = 1,
    rng?: RandomFn,
  ) {
    super(rng);
    if (sigma <= 0) throw new Error(`Invalid parameter 'sigma': expected a positive number, received ${sigma}`);
    this.name = `Normal(${mu}, ${sigma})`;
  }

  /**
   * Returns the mean: `mu`.
   */
  mean(): number {
    return this.mu;
  }

  /**
   * Returns the variance: `sigma^2`.
   */
  variance(): number {
    return this.sigma ** 2;
  }

  /**
   * Evaluates the probability density function at `x`.
   *
   * f(x) = (1 / (sigma * sqrt(2*pi))) * exp(-0.5 * ((x - mu) / sigma)^2)
   *
   * @param x - The point at which to evaluate the density.
   * @returns The density f(x).
   */
  pdf(x: number): number {
    const z = (x - this.mu) / this.sigma;
    return Math.exp(-0.5 * z * z) / (this.sigma * Math.sqrt(2 * Math.PI));
  }

  /**
   * Evaluates the cumulative distribution function at `x` using the error function.
   *
   * F(x) = 0.5 * (1 + erf((x - mu) / (sigma * sqrt(2))))
   *
   * @param x - The point at which to evaluate the CDF.
   * @returns P(X <= x) in [0, 1].
   */
  cdf(x: number): number {
    return 0.5 * (1 + erf((x - this.mu) / (this.sigma * Math.SQRT2)));
  }

  /**
   * Computes the quantile (inverse CDF) using a rational approximation
   * (Beasley-Springer-Moro) for the standard normal, then applies the
   * location-scale transform.
   *
   * @param p - A probability in [0, 1].
   * @returns The value x such that P(X <= x) = p.
   * @throws If `p` is outside [0, 1].
   */
  quantile(p: number): number {
    if (p < 0 || p > 1) throw new Error(`Invalid parameter 'p': expected a value in [0, 1], received ${p}`);
    if (p === 0) return -Infinity;
    if (p === 1) return Infinity;
    // Rational approximation for the standard normal quantile
    return this.mu + this.sigma * standardNormalQuantile(p);
  }

  /**
   * Draws a random sample using the Box-Muller transform.
   * @returns A random variate from this Normal distribution.
   */
  sample(): number {
    // Box-Muller transform
    const u1 = this.rng();
    const u2 = this.rng();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return this.mu + this.sigma * z;
  }
}

/** Rational approximation for the standard normal quantile (Beasley-Springer-Moro). */
function standardNormalQuantile(p: number): number {
  if (p < 0.5) return -standardNormalQuantile(1 - p);

  const t = Math.sqrt(-2 * Math.log(1 - p));
  const c0 = 2.515517;
  const c1 = 0.802853;
  const c2 = 0.010328;
  const d1 = 1.432788;
  const d2 = 0.189269;
  const d3 = 0.001308;

  return t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t);
}
