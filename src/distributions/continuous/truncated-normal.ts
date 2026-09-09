import { BaseContinuous } from "../base";
import { erf } from "../../utils/math";
import { RandomFn } from "../../types";

/**
 * Standard normal CDF: Phi(x) = 0.5 * (1 + erf(x / sqrt(2))).
 */
function normalCdf(x: number): number {
  return 0.5 * (1 + erf(x / Math.SQRT2));
}

/**
 * Standard normal PDF: phi(x) = exp(-0.5*x^2) / sqrt(2*pi).
 */
function normalPdf(x: number): number {
  return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
}

/**
 * Standard normal quantile via rational approximation (Beasley-Springer-Moro).
 */
function normalQuantile(p: number): number {
  if (p < 0.5) return -normalQuantile(1 - p);
  const t = Math.sqrt(-2 * Math.log(1 - p));
  const c0 = 2.515517;
  const c1 = 0.802853;
  const c2 = 0.010328;
  const d1 = 1.432788;
  const d2 = 0.189269;
  const d3 = 0.001308;
  return t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t);
}

/**
 * Truncated Normal distribution.
 *
 * A normal distribution with mean `mu` and standard deviation `sigma`,
 * restricted to the interval [a, b]. The PDF is proportional to the
 * normal PDF within [a, b] and zero outside.
 *
 * PDF: f(x) = phi((x-mu)/sigma) / (sigma * (Phi((b-mu)/sigma) - Phi((a-mu)/sigma)))
 *
 * CDF: F(x) = (Phi((x-mu)/sigma) - Phi((a-mu)/sigma)) / (Phi((b-mu)/sigma) - Phi((a-mu)/sigma))
 *
 * Support: [a, b]
 *
 * @example
 * ```ts
 * const dist = new TruncatedNormal(0, 1, -1, 1);
 * dist.mean();       // 0 (by symmetry)
 * dist.pdf(0.5);     // density at x = 0.5
 * dist.cdf(0);       // 0.5 (by symmetry)
 * dist.sample();     // random variate in [-1, 1]
 * ```
 */
export class TruncatedNormal extends BaseContinuous {
  readonly name: string;
  private readonly alphaStd: number; // (a - mu) / sigma
  private readonly betaStd: number;  // (b - mu) / sigma
  private readonly phiAlpha: number; // Phi(alphaStd)
  private readonly phiBeta: number;  // Phi(betaStd)
  private readonly Z: number;        // phiBeta - phiAlpha (normalization)

  /**
   * Creates a new Truncated Normal distribution.
   *
   * @param mu - Mean of the underlying normal distribution. Defaults to 0.
   * @param sigma - Standard deviation of the underlying normal (must be > 0). Defaults to 1.
   * @param a - Lower truncation bound. Defaults to -Infinity.
   * @param b - Upper truncation bound. Defaults to +Infinity.
   * @param rng - Optional random number generator returning values in [0, 1).
   * @throws {Error} If sigma is not positive.
   * @throws {Error} If a >= b.
   */
  constructor(
    public readonly mu: number = 0,
    public readonly sigma: number = 1,
    public readonly a: number = -Infinity,
    public readonly b: number = Infinity,
    rng?: RandomFn,
  ) {
    super(rng);
    if (sigma <= 0) throw new Error(`Invalid parameter 'sigma': expected a positive number, received ${sigma}`);
    if (a >= b) throw new Error(`Invalid parameters 'a', 'b': expected a < b, received a=${a}, b=${b}`);
    this.name = `TruncatedNormal(${mu}, ${sigma}, ${a}, ${b})`;
    this.alphaStd = (a - mu) / sigma;
    this.betaStd = (b - mu) / sigma;
    this.phiAlpha = normalCdf(this.alphaStd);
    this.phiBeta = normalCdf(this.betaStd);
    this.Z = this.phiBeta - this.phiAlpha;
  }

  /**
   * Returns the mean of the truncated normal distribution.
   *
   * Formula: E[X] = mu + sigma * (phi(alpha) - phi(beta)) / Z
   *
   * where alpha = (a - mu)/sigma, beta = (b - mu)/sigma, and Z = Phi(beta) - Phi(alpha).
   *
   * @returns The expected value.
   */
  mean(): number {
    const phiA = normalPdf(this.alphaStd);
    const phiB = normalPdf(this.betaStd);
    return this.mu + this.sigma * (phiA - phiB) / this.Z;
  }

  /**
   * Returns the variance of the truncated normal distribution.
   *
   * Formula: Var(X) = sigma^2 * (1 + (alpha*phi(alpha) - beta*phi(beta))/Z - ((phi(alpha) - phi(beta))/Z)^2)
   *
   * @returns The variance.
   */
  variance(): number {
    const phiA = normalPdf(this.alphaStd);
    const phiB = normalPdf(this.betaStd);
    const aPhiA = isFinite(this.alphaStd) ? this.alphaStd * phiA : 0;
    const bPhiB = isFinite(this.betaStd) ? this.betaStd * phiB : 0;
    const ratio = (phiA - phiB) / this.Z;
    return this.sigma ** 2 * (1 + (aPhiA - bPhiB) / this.Z - ratio * ratio);
  }

  /**
   * Evaluates the probability density function at `x`.
   *
   * @param x - The point at which to evaluate the density.
   * @returns The probability density f(x). Returns 0 for x outside [a, b].
   */
  pdf(x: number): number {
    if (x < this.a || x > this.b) return 0;
    const z = (x - this.mu) / this.sigma;
    return normalPdf(z) / (this.sigma * this.Z);
  }

  /**
   * Computes the log of the probability density function at `x`.
   *
   * log f(x) = -0.5 * z^2 - 0.5 * log(2*pi) - log(sigma * Z)
   *
   * where z = (x - mu) / sigma and Z = Phi(betaStd) - Phi(alphaStd).
   *
   * @param x - The point at which to evaluate the log-density.
   * @returns The log-density. Returns -Infinity for x outside [a, b].
   */
  logPdf(x: number): number {
    if (x < this.a || x > this.b) return -Infinity;
    const z = (x - this.mu) / this.sigma;
    return -0.5 * z * z - 0.5 * Math.log(2 * Math.PI) - Math.log(this.sigma * this.Z);
  }

  /**
   * Returns the mode of the Truncated Normal distribution.
   *
   * The mode is mu clamped to [a, b].
   */
  get mode(): number {
    if (this.mu < this.a) return this.a;
    if (this.mu > this.b) return this.b;
    return this.mu;
  }

  /**
   * Evaluates the cumulative distribution function at `x`.
   *
   * @param x - The point at which to evaluate the CDF.
   * @returns P(X <= x) in [0, 1].
   */
  cdf(x: number): number {
    if (x <= this.a) return 0;
    if (x >= this.b) return 1;
    const z = (x - this.mu) / this.sigma;
    return (normalCdf(z) - this.phiAlpha) / this.Z;
  }

  /**
   * Computes the quantile (inverse CDF) for a given probability.
   *
   * Formula: Q(p) = mu + sigma * Phi^{-1}(Phi(alpha) + p * Z)
   *
   * @param p - A probability in [0, 1].
   * @returns The quantile value x in [a, b].
   * @throws {Error} If p is not in [0, 1].
   */
  quantile(p: number): number {
    if (p < 0 || p > 1) throw new Error(`Invalid parameter 'p': expected a value in [0, 1], received ${p}`);
    if (p === 0) return this.a;
    if (p === 1) return this.b;
    return this.mu + this.sigma * normalQuantile(this.phiAlpha + p * this.Z);
  }

  /**
   * Draws a single random sample using rejection sampling.
   *
   * Generates samples from the underlying normal distribution and rejects
   * those that fall outside [a, b].
   *
   * @returns A random variate in [a, b].
   */
  sample(): number {
    // Rejection sampling: draw from normal, reject if outside [a, b]
    while (true) {
      const u1 = this.rng();
      const u2 = this.rng();
      const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      const x = this.mu + this.sigma * z;
      if (x >= this.a && x <= this.b) return x;
    }
  }
}
