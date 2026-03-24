import { BaseContinuous } from "../base";
import { gammaLn, regularizedGammaP, quantileBisect } from "../../utils/math";
import { RandomFn } from "../../types";

/**
 * Inverse Gamma distribution.
 *
 * If X ~ Gamma(alpha, beta), then 1/X ~ InverseGamma(alpha, beta).
 * Commonly used as a conjugate prior for the variance of a normal distribution.
 *
 * PDF: f(x) = (beta^alpha / Gamma(alpha)) * x^(-alpha-1) * exp(-beta/x)
 *
 * CDF: F(x) = 1 - P(alpha, beta/x) where P is the lower regularized incomplete gamma function.
 *
 * Support: x > 0
 *
 * @example
 * ```ts
 * const dist = new InverseGamma(3, 2);
 * dist.mean();       // beta / (alpha - 1) = 1
 * dist.pdf(1);       // density at x = 1
 * dist.cdf(1);       // P(X <= 1)
 * dist.sample();     // random positive variate
 * ```
 */
export class InverseGamma extends BaseContinuous {
  readonly name: string;
  private readonly logNormConst: number;

  /**
   * Creates a new Inverse Gamma distribution.
   *
   * @param alpha - Shape parameter (must be > 0). Defaults to 1.
   * @param beta - Scale parameter (must be > 0). Defaults to 1.
   * @param rng - Optional random number generator returning values in [0, 1).
   * @throws {Error} If alpha is not positive.
   * @throws {Error} If beta is not positive.
   */
  constructor(
    public readonly alpha: number = 1,
    public readonly beta: number = 1,
    rng?: RandomFn,
  ) {
    super(rng);
    if (alpha <= 0) throw new Error("alpha (shape) must be positive");
    if (beta <= 0) throw new Error("beta (scale) must be positive");
    this.name = `InverseGamma(${alpha}, ${beta})`;
    this.logNormConst = alpha * Math.log(beta) - gammaLn(alpha);
  }

  /**
   * Returns the mean of the Inverse Gamma distribution.
   *
   * Formula: E[X] = beta / (alpha - 1) for alpha > 1.
   *
   * @returns The expected value, or Infinity if alpha <= 1.
   */
  mean(): number {
    if (this.alpha <= 1) return Infinity;
    return this.beta / (this.alpha - 1);
  }

  /**
   * Returns the variance of the Inverse Gamma distribution.
   *
   * Formula: Var(X) = beta^2 / ((alpha - 1)^2 * (alpha - 2)) for alpha > 2.
   *
   * @returns The variance, or Infinity if alpha <= 2.
   */
  variance(): number {
    if (this.alpha <= 2) return Infinity;
    return this.beta ** 2 / ((this.alpha - 1) ** 2 * (this.alpha - 2));
  }

  /**
   * Evaluates the probability density function at `x`.
   *
   * PDF: f(x) = (beta^alpha / Gamma(alpha)) * x^(-alpha-1) * exp(-beta/x)
   *
   * @param x - The point at which to evaluate the density.
   * @returns The probability density f(x) >= 0. Returns 0 for x <= 0.
   */
  pdf(x: number): number {
    if (x <= 0) return 0;
    const logPdf = this.logNormConst + (-this.alpha - 1) * Math.log(x) - this.beta / x;
    return Math.exp(logPdf);
  }

  /**
   * Evaluates the cumulative distribution function at `x`.
   *
   * Uses the identity: if X ~ InvGamma(alpha, beta), then
   * F(x) = 1 - P(alpha, beta/x), where P is the lower regularized gamma.
   *
   * @param x - The point at which to evaluate the CDF.
   * @returns P(X <= x) in [0, 1]. Returns 0 for x <= 0.
   */
  cdf(x: number): number {
    if (x <= 0) return 0;
    return 1 - regularizedGammaP(this.alpha, this.beta / x);
  }

  /**
   * Computes the quantile (inverse CDF) using bisection search.
   *
   * @param p - A probability in [0, 1].
   * @returns The quantile value x > 0.
   * @throws {Error} If p is not in [0, 1].
   */
  quantile(p: number): number {
    if (p < 0 || p > 1) throw new Error("p must be in [0, 1]");
    if (p === 0) return 0;
    if (p === 1) return Infinity;
    return quantileBisect((x) => this.cdf(x), p, 1e-15, 1e6);
  }

  /**
   * Draws a single random sample via 1 / Gamma(alpha, 1/beta).
   *
   * Uses the Marsaglia-Tsang method for generating gamma variates.
   *
   * @returns A random positive variate.
   */
  sample(): number {
    // Sample from Gamma(alpha, 1) then transform: X = beta / G
    const g = sampleGamma(this.alpha, this.rng);
    return this.beta / g;
  }
}

/**
 * Samples from the Gamma(shape, 1) distribution using the Marsaglia-Tsang method.
 *
 * For shape < 1, uses the identity: Gamma(shape) = Gamma(shape+1) * U^{1/shape}.
 *
 * @param shape - The shape parameter (must be positive).
 * @param rng - Random number generator.
 * @returns A random variate from Gamma(shape, 1).
 */
function sampleGamma(shape: number, rng: RandomFn): number {
  if (shape < 1) {
    return sampleGamma(shape + 1, rng) * Math.pow(rng(), 1 / shape);
  }
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  while (true) {
    let x: number, v: number;
    do {
      const u1 = rng();
      const u2 = rng();
      x = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = rng();
    if (
      u < 1 - 0.0331 * (x * x) * (x * x) ||
      Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))
    ) {
      return d * v;
    }
  }
}
