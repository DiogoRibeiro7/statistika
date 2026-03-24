import { BaseContinuous } from "../base";
import { RandomFn } from "../../types";

/**
 * Continuous Uniform distribution on the interval [a, b].
 *
 * The PDF is constant on the support:
 *
 *   f(x) = 1 / (b - a)   for a <= x <= b
 *
 * Support: [a, b]
 *
 * @example
 * ```ts
 * const dist = new Uniform(0, 1);
 * dist.pdf(0.5);    // 1.0
 * dist.cdf(0.25);   // 0.25
 * dist.sample();
 * ```
 */
export class Uniform extends BaseContinuous {
  readonly name: string;

  /**
   * Creates a Uniform distribution.
   * @param a - Lower bound of the support. Defaults to 0.
   * @param b - Upper bound of the support (must be > a). Defaults to 1.
   * @param rng - Optional random number generator.
   * @throws If `a >= b`.
   */
  constructor(
    public readonly a: number = 0,
    public readonly b: number = 1,
    rng?: RandomFn,
  ) {
    super(rng);
    if (a >= b) throw new Error(`Invalid parameters 'a', 'b': expected a < b, received a=${a}, b=${b}`);
    this.name = `Uniform(${a}, ${b})`;
  }

  /**
   * Returns the mean: `(a + b) / 2`.
   */
  mean(): number {
    return (this.a + this.b) / 2;
  }

  /**
   * Returns the variance: `(b - a)^2 / 12`.
   */
  variance(): number {
    return (this.b - this.a) ** 2 / 12;
  }

  /**
   * Evaluates the PDF at `x`.
   *
   * f(x) = 1 / (b - a) for a <= x <= b, 0 otherwise.
   *
   * @param x - The point at which to evaluate the density.
   * @returns The density f(x).
   */
  pdf(x: number): number {
    return x >= this.a && x <= this.b ? 1 / (this.b - this.a) : 0;
  }

  /**
   * Evaluates the CDF at `x`.
   *
   * F(x) = (x - a) / (b - a) for a <= x <= b.
   *
   * @param x - The point at which to evaluate the CDF.
   * @returns P(X <= x) in [0, 1].
   */
  cdf(x: number): number {
    if (x < this.a) return 0;
    if (x > this.b) return 1;
    return (x - this.a) / (this.b - this.a);
  }

  /**
   * Computes the quantile (inverse CDF): `a + p * (b - a)`.
   * @param p - A probability in [0, 1].
   * @returns The value x such that P(X <= x) = p.
   * @throws If `p` is outside [0, 1].
   */
  quantile(p: number): number {
    if (p < 0 || p > 1) throw new Error(`Invalid parameter 'p': expected a value in [0, 1], received ${p}`);
    return this.a + p * (this.b - this.a);
  }

  /**
   * Draws a random sample via linear scaling of a uniform [0,1) variate.
   * @returns A random variate in [a, b).
   */
  sample(): number {
    return this.a + this.rng() * (this.b - this.a);
  }
}
