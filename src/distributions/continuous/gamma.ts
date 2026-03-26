import { BaseContinuous } from "../base";
import { gammaLn, regularizedGammaP, quantileBisect } from "../../utils/math";
import { RandomFn } from "../../types";

/**
 * Gamma distribution parameterized by `shape` (alpha) and `rate` (beta).
 *
 * The PDF is:
 *
 *   f(x) = (rate^shape / Gamma(shape)) * x^(shape-1) * exp(-rate * x)
 *
 * for x > 0. The scale parameterization uses scale = 1/rate.
 *
 * Support: [0, +Infinity) (or (0, +Infinity) when shape < 1)
 *
 * @example
 * ```ts
 * const dist = new GammaDistribution(2, 1);
 * dist.mean();    // 2
 * dist.pdf(1);    // exp(-1) ~ 0.3679
 * ```
 */
export class GammaDistribution extends BaseContinuous {
  readonly name: string;

  /**
   * Creates a Gamma distribution.
   * @param shape - Shape parameter alpha (must be > 0). Defaults to 1.
   * @param rate - Rate parameter beta (must be > 0). Defaults to 1.
   * @param rng - Optional random number generator.
   * @throws If `shape` or `rate` is not positive.
   */
  constructor(
    public readonly shape: number = 1,
    public readonly rate: number = 1,
    rng?: RandomFn,
  ) {
    super(rng);
    if (shape <= 0) throw new Error(`Invalid parameter 'shape': expected a positive number, received ${shape}`);
    if (rate <= 0) throw new Error(`Invalid parameter 'rate': expected a positive number, received ${rate}`);
    this.name = `Gamma(${shape}, ${rate})`;
  }

  /**
   * Returns the mean: `shape / rate`.
   */
  mean(): number {
    return this.shape / this.rate;
  }

  /**
   * Returns the variance: `shape / rate^2`.
   */
  variance(): number {
    return this.shape / this.rate ** 2;
  }

  /**
   * Evaluates the PDF at `x`.
   *
   * f(x) = (rate^shape / Gamma(shape)) * x^(shape-1) * exp(-rate * x)
   *
   * Computed in log-space for numerical stability.
   *
   * @param x - The point at which to evaluate the density.
   * @returns The density f(x).
   */
  pdf(x: number): number {
    if (x < 0) return 0;
    if (x === 0) {
      if (this.shape === 1) return this.rate;
      if (this.shape < 1) return Infinity;
      return 0;
    }
    const logPdf =
      this.shape * Math.log(this.rate) +
      (this.shape - 1) * Math.log(x) -
      this.rate * x -
      gammaLn(this.shape);
    return Math.exp(logPdf);
  }

  /**
   * Computes the log of the probability density function at `x`.
   *
   * log f(x) = shape * log(rate) + (shape - 1) * log(x) - rate * x - gammaLn(shape)
   *
   * @param x - The point at which to evaluate the log-density.
   * @returns The log-density. Returns -Infinity for x < 0.
   */
  logPdf(x: number): number {
    if (x < 0) return -Infinity;
    if (x === 0) {
      if (this.shape === 1) return Math.log(this.rate);
      if (this.shape < 1) return Infinity;
      return -Infinity;
    }
    return (
      this.shape * Math.log(this.rate) +
      (this.shape - 1) * Math.log(x) -
      this.rate * x -
      gammaLn(this.shape)
    );
  }

  /**
   * Returns the skewness of the Gamma distribution.
   *
   * Formula: 2 / sqrt(shape)
   */
  get skewness(): number {
    return 2 / Math.sqrt(this.shape);
  }

  /**
   * Returns the excess kurtosis of the Gamma distribution.
   *
   * Formula: 6 / shape
   */
  get kurtosis(): number {
    return 6 / this.shape;
  }

  /**
   * Returns the mode of the Gamma distribution.
   *
   * Formula: (shape - 1) / rate for shape >= 1, 0 for shape < 1.
   */
  get mode(): number {
    if (this.shape < 1) return 0;
    return (this.shape - 1) / this.rate;
  }

  /**
   * Evaluates the CDF at `x` using the regularized lower incomplete gamma function.
   *
   * F(x) = P(shape, rate * x) = gammaP(shape, rate * x)
   *
   * @param x - The point at which to evaluate the CDF.
   * @returns P(X <= x) in [0, 1].
   */
  cdf(x: number): number {
    if (x <= 0) return 0;
    return regularizedGammaP(this.shape, this.rate * x);
  }

  /**
   * Computes the quantile (inverse CDF) via bisection search.
   * @param p - A probability in [0, 1].
   * @returns The value x such that P(X <= x) = p.
   * @throws If `p` is outside [0, 1].
   */
  quantile(p: number): number {
    if (p < 0 || p > 1) throw new Error(`Invalid parameter 'p': expected a value in [0, 1], received ${p}`);
    if (p === 0) return 0;
    if (p === 1) return Infinity;
    // Initial bracket
    const m = this.mean();
    const sd = this.stdDev();
    const upper = m + 10 * sd;
    return quantileBisect((x) => this.cdf(x), p, 0, upper);
  }

  /**
   * Draws a random sample using the Marsaglia-Tsang method (shape >= 1)
   * with a shape-shifting technique for shape < 1.
   * @returns A random variate from this Gamma distribution.
   */
  sample(): number {
    // Marsaglia-Tsang method for shape >= 1, shift for shape < 1
    if (this.shape < 1) {
      const g = new GammaDistribution(this.shape + 1, 1, this.rng).sample();
      return (g * Math.pow(this.rng(), 1 / this.shape)) / this.rate;
    }
    const d = this.shape - 1 / 3;
    const c = 1 / Math.sqrt(9 * d);
    while (true) {
      let x: number, v: number;
      do {
        x = standardNormal(this.rng);
        v = 1 + c * x;
      } while (v <= 0);
      v = v * v * v;
      const u = this.rng();
      if (
        u < 1 - 0.0331 * (x * x) * (x * x) ||
        Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))
      ) {
        return (d * v) / this.rate;
      }
    }
  }
}

function standardNormal(rng: RandomFn): number {
  const u1 = rng();
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}
