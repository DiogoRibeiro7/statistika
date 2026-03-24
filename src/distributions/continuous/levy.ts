import { BaseContinuous } from "../base";
import { RandomFn } from "../../types";
import { erfc } from "../../utils/math";

/**
 * Lévy distribution — a stable distribution with parameters α = 0.5, β = 1.
 *
 * Used in finance for modeling heavy-tailed returns and in physics for
 * anomalous diffusion processes. The support is (μ, ∞).
 *
 * PDF: f(x; μ, c) = sqrt(c / (2π)) * exp(-c / (2(x - μ))) / (x - μ)^(3/2)
 * CDF: F(x; μ, c) = erfc(sqrt(c / (2(x - μ))))
 *
 * Both mean and variance are infinite.
 *
 * @example
 * ```ts
 * const dist = new Levy(0, 1); // location=0, scale=1
 * dist.pdf(2);      // probability density at x=2
 * dist.cdf(2);      // cumulative probability at x=2
 * dist.quantile(0.5); // median
 * dist.sample();     // random variate
 * ```
 */
export class Levy extends BaseContinuous {
  readonly name: string;

  /**
   * Creates a new Lévy distribution.
   *
   * @param mu - Location parameter. Defaults to 0.
   * @param c - Scale parameter (must be > 0). Defaults to 1.
   * @param rng - Optional random number generator returning values in [0, 1).
   * @throws {Error} If c is not positive.
   *
   * @example
   * ```ts
   * const standard = new Levy();       // mu=0, c=1
   * const shifted = new Levy(2, 3);    // mu=2, c=3
   * ```
   */
  constructor(
    public readonly mu: number = 0,
    public readonly c: number = 1,
    rng?: RandomFn,
  ) {
    super(rng);
    if (c <= 0) throw new Error("c must be positive");
    this.name = `Levy(${mu}, ${c})`;
  }

  /**
   * Returns the mean of the Lévy distribution.
   *
   * The mean is infinite for all Lévy distributions.
   *
   * @returns Infinity.
   */
  mean(): number {
    return Infinity;
  }

  /**
   * Returns the variance of the Lévy distribution.
   *
   * The variance is infinite for all Lévy distributions.
   *
   * @returns Infinity.
   */
  variance(): number {
    return Infinity;
  }

  /**
   * Evaluates the probability density function (PDF) at x.
   *
   * PDF: f(x) = sqrt(c / (2π)) * exp(-c / (2(x - μ))) / (x - μ)^(3/2)
   *
   * Returns 0 for x <= μ (outside the support).
   *
   * @param x - The point at which to evaluate the density.
   * @returns The probability density f(x) >= 0.
   *
   * @example
   * ```ts
   * const dist = new Levy(0, 1);
   * dist.pdf(1); // ≈ 0.2420
   * ```
   */
  pdf(x: number): number {
    if (x <= this.mu) return 0;
    const diff = x - this.mu;
    return (
      Math.sqrt(this.c / (2 * Math.PI)) *
      Math.exp(-this.c / (2 * diff)) /
      Math.pow(diff, 1.5)
    );
  }

  /**
   * Evaluates the cumulative distribution function (CDF) at x.
   *
   * CDF: F(x) = erfc(sqrt(c / (2(x - μ))))
   *
   * Returns 0 for x <= μ (outside the support).
   *
   * @param x - The point at which to evaluate the CDF.
   * @returns P(X <= x), a probability in [0, 1].
   *
   * @example
   * ```ts
   * const dist = new Levy(0, 1);
   * dist.cdf(2); // ≈ 0.4795
   * ```
   */
  cdf(x: number): number {
    if (x <= this.mu) return 0;
    return erfc(Math.sqrt(this.c / (2 * (x - this.mu))));
  }

  /**
   * Computes the quantile (inverse CDF) for a given probability.
   *
   * Uses the inverse of the CDF: Q(p) = μ + c / (2 * (erfcinv(p))^2)
   * where erfcinv is computed via Newton's method.
   *
   * @param p - A probability in [0, 1].
   * @returns The quantile value.
   * @throws {Error} If p is not in [0, 1].
   *
   * @example
   * ```ts
   * const dist = new Levy(0, 1);
   * dist.quantile(0.5); // median
   * ```
   */
  quantile(p: number): number {
    if (p < 0 || p > 1) throw new Error("p must be in [0, 1]");
    if (p === 0) return this.mu;
    if (p === 1) return Infinity;
    // CDF(x) = erfc(sqrt(c / (2(x - mu)))) = p
    // erfc(z) = p => z = erfcinv(p)
    // sqrt(c / (2(x - mu))) = erfcinv(p)
    // x = mu + c / (2 * erfcinv(p)^2)
    const z = erfcinv(p);
    return this.mu + this.c / (2 * z * z);
  }

  /**
   * Draws a single random sample from the Lévy distribution.
   *
   * Uses the inverse CDF method.
   *
   * @returns A random variate from the Lévy distribution.
   */
  sample(): number {
    const u = this.rng();
    return this.quantile(u);
  }
}

/**
 * Inverse complementary error function.
 * Computes x such that erfc(x) = p, for p in (0, 2).
 *
 * Uses the ndtri (inverse normal CDF) approach:
 * erfcinv(p) = -ndtri(p/2) / sqrt(2)
 *
 * The initial approximation uses the rational minimax approximation
 * from Peter Acklam, refined with Halley's method.
 */
function erfcinv(p: number): number {
  if (p <= 0) return Infinity;
  if (p >= 2) return -Infinity;
  if (p === 1) return 0;

  // erfcinv(p) = -ndtri(p/2) / sqrt(2)
  // where ndtri is the inverse of the normal CDF Phi
  return -ndtri(p / 2) / Math.SQRT2;
}

/**
 * Inverse of the standard normal CDF (probit function).
 * Uses Acklam's rational approximation with Halley refinement.
 */
function ndtri(p: number): number {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  if (p === 0.5) return 0;

  // Coefficients for rational approximation
  const a = [
    -3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
    1.383577518672690e2, -3.066479806614716e1, 2.506628277459239e0,
  ];
  const b = [
    -5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
    6.680131188771972e1, -1.328068155288572e1,
  ];
  const c = [
    -7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838e0,
    -2.549732539343734e0, 4.374664141464968e0, 2.938163982698783e0,
  ];
  const d = [
    7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996e0,
    3.754408661907416e0,
  ];

  const pLow = 0.02425;
  const pHigh = 1 - pLow;
  let q: number, r: number, x: number;

  if (p < pLow) {
    // Rational approximation for lower region
    q = Math.sqrt(-2 * Math.log(p));
    x =
      (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  } else if (p <= pHigh) {
    // Rational approximation for central region
    q = p - 0.5;
    r = q * q;
    x =
      ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) /
      (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  } else {
    // Rational approximation for upper region
    q = Math.sqrt(-2 * Math.log(1 - p));
    x =
      -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }

  // Halley refinement
  const SQRT_2PI = Math.sqrt(2 * Math.PI);
  const e = 0.5 * erfc(-x / Math.SQRT2) - p;
  const u = e * SQRT_2PI * Math.exp((x * x) / 2);
  x = x - u / (1 + (x * u) / 2);

  return x;
}
