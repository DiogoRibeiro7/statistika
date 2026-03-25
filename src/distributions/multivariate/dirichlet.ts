/**
 * Dirichlet distribution.
 *
 * A multivariate generalization of the Beta distribution, parameterized
 * by a concentration vector alpha. The Dirichlet distribution is the conjugate
 * prior for the categorical and multinomial distributions.
 *
 * The PDF is defined on the (k-1)-simplex (vectors summing to 1):
 * f(x; alpha) = (1 / B(alpha)) * prod_i x_i^{alpha_i - 1}
 *
 * where B(alpha) = prod Gamma(alpha_i) / Gamma(sum alpha_i) is the multivariate
 * Beta function.
 *
 * @example
 * ```ts
 * const dist = new Dirichlet([2, 3, 5]);
 * dist.mean();               // [0.2, 0.3, 0.5]
 * dist.pdf([0.2, 0.3, 0.5]); // density at the mean
 * dist.sample();              // random point on the 2-simplex
 * ```
 */

import { gammaLn } from "../../utils/math";
import { RandomFn } from "../../types";

/**
 * Represents a Dirichlet distribution over the (k-1)-dimensional simplex.
 *
 * The Dirichlet distribution is parameterized by a vector of positive
 * concentration parameters alpha = (alpha_1, ..., alpha_k). It produces
 * random vectors whose components are non-negative and sum to 1.
 */
export class Dirichlet {
  readonly name: string;
  readonly dim: number;
  private readonly alphaSum: number;
  private readonly lnBeta: number;
  private rng: RandomFn;

  /**
   * Creates a new Dirichlet distribution.
   *
   * @param alpha - Concentration parameters (array of positive numbers, length >= 2).
   * @param rng - Optional random number generator; defaults to Math.random.
   * @throws {Error} If alpha has fewer than 2 elements.
   * @throws {Error} If any alpha value is not positive.
   *
   * @example
   * ```ts
   * const uniform = new Dirichlet([1, 1, 1]); // uniform on 2-simplex
   * const concentrated = new Dirichlet([10, 10, 10]); // concentrated near center
   * ```
   */
  constructor(public readonly alpha: number[], rng?: RandomFn) {
    this.rng = rng ?? Math.random;
    const k = alpha.length;
    if (k < 2) throw new Error(`Invalid parameter 'alpha': expected at least 2 dimensions, received ${k}`);
    for (let i = 0; i < k; i++) {
      if (alpha[i] <= 0) throw new Error(`Invalid parameter 'alpha[${i}]': expected a positive number, received ${alpha[i]}`);
    }
    this.dim = k;
    this.alphaSum = alpha.reduce((a, b) => a + b, 0);
    this.name = `Dirichlet(dim=${k})`;

    // Log of the multivariate Beta function: B(alpha) = prod Gamma(alpha_i) / Gamma(sum alpha_i)
    let lnNum = 0;
    for (let i = 0; i < k; i++) lnNum += gammaLn(alpha[i]);
    this.lnBeta = lnNum - gammaLn(this.alphaSum);
  }

  /**
   * Computes the mean vector of the distribution.
   *
   * Formula: E[X_i] = alpha_i / sum(alpha)
   *
   * @returns An array of length k with the expected value of each component.
   *
   * @example
   * ```ts
   * new Dirichlet([2, 3, 5]).mean(); // [0.2, 0.3, 0.5]
   * ```
   */
  mean(): number[] {
    return this.alpha.map((a) => a / this.alphaSum);
  }

  /**
   * Computes the variance vector of the distribution.
   *
   * Formula: Var(X_i) = alpha_i * (alpha_0 - alpha_i) / (alpha_0^2 * (alpha_0 + 1))
   *
   * where alpha_0 = sum(alpha).
   *
   * @returns An array of length k with the variance of each component.
   *
   * @example
   * ```ts
   * new Dirichlet([2, 3, 5]).variance(); // variances for each component
   * ```
   */
  variance(): number[] {
    const a0 = this.alphaSum;
    return this.alpha.map((ai) => (ai * (a0 - ai)) / (a0 * a0 * (a0 + 1)));
  }

  /**
   * Computes the log probability density function at a point x.
   *
   * Formula: log f(x; alpha) = sum_i (alpha_i - 1) * log(x_i) - log B(alpha)
   *
   * @param x - A point on the (k-1)-simplex (non-negative values summing to 1).
   * @returns The log-density at x. Returns -Infinity if x is outside the simplex.
   * @throws {Error} If x has incorrect length.
   *
   * @example
   * ```ts
   * const dist = new Dirichlet([2, 3, 5]);
   * dist.logPdf([0.2, 0.3, 0.5]); // log-density at this point
   * dist.logPdf([0.5, 0.5, 0.1]); // -Infinity (does not sum to 1)
   * ```
   */
  logPdf(x: number[]): number {
    if (x.length !== this.dim) {
      throw new Error(`x must have length ${this.dim}`);
    }

    let sum = 0;
    let xSum = 0;
    for (let i = 0; i < this.dim; i++) {
      if (x[i] < 0 || x[i] > 1) return -Infinity;
      xSum += x[i];
      sum += (this.alpha[i] - 1) * Math.log(x[i]);
    }

    // Check that x sums to approximately 1
    if (Math.abs(xSum - 1) > 1e-6) return -Infinity;

    return sum - this.lnBeta;
  }

  /**
   * Computes the probability density function at a point x.
   *
   * Formula: f(x; alpha) = (1 / B(alpha)) * prod_i x_i^{alpha_i - 1}
   *
   * @param x - A point on the (k-1)-simplex (non-negative values summing to 1).
   * @returns The density at x. Returns 0 if x is outside the simplex.
   * @throws {Error} If x has incorrect length.
   *
   * @example
   * ```ts
   * const dist = new Dirichlet([2, 3, 5]);
   * dist.pdf([0.2, 0.3, 0.5]); // density at this point
   * ```
   */
  pdf(x: number[]): number {
    return Math.exp(this.logPdf(x));
  }

  /**
   * Draws a single random sample from the distribution.
   *
   * Uses the gamma variate method: if X_i ~ Gamma(alpha_i, 1) independently,
   * then (X_1/S, ..., X_k/S) ~ Dirichlet(alpha) where S = sum X_i.
   *
   * @returns A random vector on the (k-1)-simplex (non-negative, sums to 1).
   *
   * @example
   * ```ts
   * const dist = new Dirichlet([2, 3, 5]);
   * const proportions = dist.sample(); // e.g., [0.18, 0.32, 0.50]
   * ```
   */
  sample(): number[] {
    const k = this.dim;
    const y = new Array(k);
    let sum = 0;

    for (let i = 0; i < k; i++) {
      y[i] = sampleGamma(this.alpha[i], this.rng);
      sum += y[i];
    }

    for (let i = 0; i < k; i++) y[i] /= sum;
    return y;
  }

  /**
   * Draws n independent samples from the distribution.
   *
   * @param n - Number of samples to draw.
   * @returns An array of n random vectors, each on the (k-1)-simplex.
   *
   * @example
   * ```ts
   * const dist = new Dirichlet([2, 3, 5]);
   * const samples = dist.sampleN(100); // 100 random proportion vectors
   * ```
   */
  sampleN(n: number): number[][] {
    const samples: number[][] = new Array(n);
    for (let i = 0; i < n; i++) samples[i] = this.sample();
    return samples;
  }

  /**
   * Computes the log-likelihood of a set of observations.
   *
   * Formula: LL = sum_j log f(x_j; alpha)
   *
   * @param data - An array of observed points, each on the (k-1)-simplex.
   * @returns The total log-likelihood.
   *
   * @example
   * ```ts
   * const dist = new Dirichlet([2, 3, 5]);
   * const obs = dist.sampleN(50);
   * dist.logLikelihood(obs); // log-likelihood of the observations
   * ```
   */
  logLikelihood(data: number[][]): number {
    let ll = 0;
    for (const x of data) ll += this.logPdf(x);
    return ll;
  }
}

/**
 * Samples from the Gamma(shape, 1) distribution using the Marsaglia-Tsang method.
 *
 * For shape < 1, uses the identity: Gamma(shape) = Gamma(shape+1) * U^{1/shape}
 * where U ~ Uniform(0,1).
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
