/**
 * Dirichlet distribution.
 *
 * A multivariate generalization of the Beta distribution, parameterized
 * by a concentration vector α. The Dirichlet distribution is the conjugate
 * prior for the categorical and multinomial distributions.
 */

import { gammaLn } from "../../utils/math";
import { RandomFn } from "../../types";

export class Dirichlet {
  readonly name: string;
  readonly dim: number;
  private readonly alphaSum: number;
  private readonly lnBeta: number;
  private rng: RandomFn;

  /**
   * @param alpha - Concentration parameters (all must be positive)
   * @param rng - Optional random number generator (defaults to Math.random).
   */
  constructor(public readonly alpha: number[], rng?: RandomFn) {
    this.rng = rng ?? Math.random;
    const k = alpha.length;
    if (k < 2) throw new Error("Dirichlet requires at least 2 dimensions");
    for (let i = 0; i < k; i++) {
      if (alpha[i] <= 0) throw new Error(`alpha[${i}] must be positive`);
    }
    this.dim = k;
    this.alphaSum = alpha.reduce((a, b) => a + b, 0);
    this.name = `Dirichlet(dim=${k})`;

    // Log of the multivariate Beta function: B(α) = Π Γ(αi) / Γ(Σ αi)
    let lnNum = 0;
    for (let i = 0; i < k; i++) lnNum += gammaLn(alpha[i]);
    this.lnBeta = lnNum - gammaLn(this.alphaSum);
  }

  /**
   * Mean vector: E[X_i] = α_i / Σα
   */
  mean(): number[] {
    return this.alpha.map((a) => a / this.alphaSum);
  }

  /**
   * Variance of component i: Var[X_i] = α_i(α_0 - α_i) / (α_0^2 (α_0 + 1))
   */
  variance(): number[] {
    const a0 = this.alphaSum;
    return this.alpha.map((ai) => (ai * (a0 - ai)) / (a0 * a0 * (a0 + 1)));
  }

  /**
   * Log probability density function.
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
   * Probability density function.
   */
  pdf(x: number[]): number {
    return Math.exp(this.logPdf(x));
  }

  /**
   * Draw a sample using Gamma variates.
   * If X_i ~ Gamma(α_i, 1), then (X_1/S, ..., X_k/S) ~ Dirichlet(α)
   * where S = Σ X_i.
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
   * Draw n samples.
   */
  sampleN(n: number): number[][] {
    const samples: number[][] = new Array(n);
    for (let i = 0; i < n; i++) samples[i] = this.sample();
    return samples;
  }

  /**
   * Log-likelihood of a set of observations.
   */
  logLikelihood(data: number[][]): number {
    let ll = 0;
    for (const x of data) ll += this.logPdf(x);
    return ll;
  }
}

/** Sample from Gamma(shape, 1) using Marsaglia-Tsang. */
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
