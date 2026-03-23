/**
 * Multinomial distribution.
 *
 * Models the outcome of n independent trials, each falling into one
 * of k categories with fixed probabilities p₁, …, pₖ.
 */

import { gammaLn, logFactorial } from "../../utils/math";
import { RandomFn } from "../../types";

export class Multinomial {
  readonly name: string;
  readonly k: number;
  private rng: RandomFn;

  /**
   * @param n - Number of trials (positive integer).
   * @param probs - Probability vector (must sum to 1, all non-negative).
   * @param rng - Optional random number generator (defaults to Math.random).
   */
  constructor(
    public readonly n: number,
    public readonly probs: number[],
    rng?: RandomFn,
  ) {
    this.rng = rng ?? Math.random;
    if (!Number.isInteger(n) || n < 1) {
      throw new Error("n must be a positive integer");
    }
    const k = probs.length;
    if (k < 2) throw new Error("Must have at least 2 categories");

    let sum = 0;
    for (let i = 0; i < k; i++) {
      if (probs[i] < 0) throw new Error(`probs[${i}] must be non-negative`);
      sum += probs[i];
    }
    if (Math.abs(sum - 1) > 1e-8) {
      throw new Error("probs must sum to 1");
    }

    this.k = k;
    this.name = `Multinomial(${n}, k=${k})`;
  }

  /** Mean vector: E[Xᵢ] = n pᵢ. */
  mean(): number[] {
    return this.probs.map((p) => this.n * p);
  }

  /** Variance vector: Var[Xᵢ] = n pᵢ (1 − pᵢ). */
  variance(): number[] {
    return this.probs.map((p) => this.n * p * (1 - p));
  }

  /** Covariance between components i and j: Cov[Xᵢ, Xⱼ] = −n pᵢ pⱼ. */
  covariance(i: number, j: number): number {
    if (i === j) return this.variance()[i];
    return -this.n * this.probs[i] * this.probs[j];
  }

  /**
   * Log of the probability mass function.
   * log P(X = x) = log(n!) − Σ log(xᵢ!) + Σ xᵢ log(pᵢ)
   */
  logPmf(x: number[]): number {
    if (x.length !== this.k) {
      throw new Error(`x must have length ${this.k}`);
    }

    let sum = 0;
    let logP = logFactorial(this.n);
    for (let i = 0; i < this.k; i++) {
      if (!Number.isInteger(x[i]) || x[i] < 0) return -Infinity;
      sum += x[i];
      logP -= logFactorial(x[i]);
      if (this.probs[i] > 0) {
        logP += x[i] * Math.log(this.probs[i]);
      } else if (x[i] > 0) {
        return -Infinity; // positive count for zero-probability category
      }
    }

    if (sum !== this.n) return -Infinity;
    return logP;
  }

  /** Probability mass function. */
  pmf(x: number[]): number {
    return Math.exp(this.logPmf(x));
  }

  /**
   * Draw a sample using sequential conditional binomials.
   * Returns an array of counts summing to n.
   */
  sample(): number[] {
    const result = new Array(this.k).fill(0);
    let remaining = this.n;
    let probRemaining = 1;

    for (let i = 0; i < this.k - 1; i++) {
      if (remaining === 0) break;
      const p = probRemaining > 0 ? this.probs[i] / probRemaining : 0;
      result[i] = sampleBinomial(remaining, Math.min(1, p), this.rng);
      remaining -= result[i];
      probRemaining -= this.probs[i];
    }
    result[this.k - 1] = remaining;
    return result;
  }

  /** Draw m samples. */
  sampleN(m: number): number[][] {
    const samples: number[][] = new Array(m);
    for (let i = 0; i < m; i++) samples[i] = this.sample();
    return samples;
  }

  /** Log-likelihood of a set of observations. */
  logLikelihood(data: number[][]): number {
    let ll = 0;
    for (const x of data) ll += this.logPmf(x);
    return ll;
  }
}

/** Sample from Binomial(n, p) via simple trial simulation. */
function sampleBinomial(n: number, p: number, rng: RandomFn): number {
  let count = 0;
  for (let i = 0; i < n; i++) {
    if (rng() < p) count++;
  }
  return count;
}
