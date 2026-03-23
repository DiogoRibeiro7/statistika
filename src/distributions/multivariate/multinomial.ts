/**
 * Multinomial distribution.
 *
 * Models the outcome of n independent trials, each falling into one
 * of k categories with fixed probabilities p_1, ..., p_k.
 *
 * PMF: P(X = x) = n! / (x_1! * ... * x_k!) * prod_i p_i^{x_i}
 *
 * where x_i >= 0 are integers summing to n.
 *
 * @example
 * ```ts
 * const dist = new Multinomial(10, [0.2, 0.3, 0.5]);
 * dist.mean();          // [2, 3, 5]
 * dist.pmf([2, 3, 5]);  // probability of this specific outcome
 * dist.sample();        // random count vector summing to 10
 * ```
 */

import { gammaLn, logFactorial } from "../../utils/math";
import { RandomFn } from "../../types";

/**
 * Represents a Multinomial distribution over count vectors.
 *
 * The multinomial distribution generalizes the binomial distribution to
 * multiple categories. It describes the probability of observing particular
 * counts across k categories in n independent trials.
 */
export class Multinomial {
  readonly name: string;
  readonly k: number;
  private rng: RandomFn;

  /**
   * Creates a new Multinomial distribution.
   *
   * @param n - Number of trials (positive integer).
   * @param probs - Probability vector (must sum to 1, all non-negative, length >= 2).
   * @param rng - Optional random number generator; defaults to Math.random.
   * @throws {Error} If n is not a positive integer.
   * @throws {Error} If probs has fewer than 2 elements.
   * @throws {Error} If any probability is negative.
   * @throws {Error} If probabilities do not sum to 1.
   *
   * @example
   * ```ts
   * const dist = new Multinomial(10, [0.2, 0.3, 0.5]);
   * ```
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

  /**
   * Computes the mean vector of the distribution.
   *
   * Formula: E[X_i] = n * p_i
   *
   * @returns An array of length k with the expected count for each category.
   *
   * @example
   * ```ts
   * new Multinomial(10, [0.2, 0.3, 0.5]).mean(); // [2, 3, 5]
   * ```
   */
  mean(): number[] {
    return this.probs.map((p) => this.n * p);
  }

  /**
   * Computes the variance vector of the distribution.
   *
   * Formula: Var(X_i) = n * p_i * (1 - p_i)
   *
   * @returns An array of length k with the variance for each category.
   *
   * @example
   * ```ts
   * new Multinomial(10, [0.2, 0.3, 0.5]).variance(); // [1.6, 2.1, 2.5]
   * ```
   */
  variance(): number[] {
    return this.probs.map((p) => this.n * p * (1 - p));
  }

  /**
   * Computes the covariance between components i and j.
   *
   * Formula: Cov(X_i, X_j) = -n * p_i * p_j for i != j
   *          Cov(X_i, X_i) = Var(X_i) = n * p_i * (1 - p_i)
   *
   * @param i - First component index.
   * @param j - Second component index.
   * @returns The covariance between X_i and X_j.
   *
   * @example
   * ```ts
   * const dist = new Multinomial(10, [0.2, 0.3, 0.5]);
   * dist.covariance(0, 1); // -10 * 0.2 * 0.3 = -0.6
   * dist.covariance(0, 0); // Var(X_0) = 1.6
   * ```
   */
  covariance(i: number, j: number): number {
    if (i === j) return this.variance()[i];
    return -this.n * this.probs[i] * this.probs[j];
  }

  /**
   * Computes the log of the probability mass function.
   *
   * Formula: log P(X = x) = log(n!) - sum_i log(x_i!) + sum_i x_i * log(p_i)
   *
   * @param x - A count vector (non-negative integers summing to n).
   * @returns The log-probability. Returns -Infinity if x is invalid.
   * @throws {Error} If x has incorrect length.
   *
   * @example
   * ```ts
   * const dist = new Multinomial(10, [0.2, 0.3, 0.5]);
   * dist.logPmf([2, 3, 5]); // log-probability of this outcome
   * ```
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

  /**
   * Computes the probability mass function P(X = x).
   *
   * Formula: P(X = x) = n! / (x_1! * ... * x_k!) * prod_i p_i^{x_i}
   *
   * @param x - A count vector (non-negative integers summing to n).
   * @returns The probability. Returns 0 if x is invalid.
   * @throws {Error} If x has incorrect length.
   *
   * @example
   * ```ts
   * const dist = new Multinomial(10, [0.2, 0.3, 0.5]);
   * dist.pmf([2, 3, 5]); // probability of this outcome
   * ```
   */
  pmf(x: number[]): number {
    return Math.exp(this.logPmf(x));
  }

  /**
   * Draws a single random sample from the distribution.
   *
   * Uses sequential conditional binomials: for each category i, sample the
   * count from Binomial(remaining, p_i / p_remaining), then assign the rest
   * to the last category.
   *
   * @returns An array of k non-negative integer counts summing to n.
   *
   * @example
   * ```ts
   * const dist = new Multinomial(10, [0.2, 0.3, 0.5]);
   * const counts = dist.sample(); // e.g., [1, 4, 5]
   * ```
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

  /**
   * Draws m independent samples from the distribution.
   *
   * @param m - Number of samples to draw.
   * @returns An array of m count vectors, each summing to n.
   *
   * @example
   * ```ts
   * const dist = new Multinomial(10, [0.2, 0.3, 0.5]);
   * const samples = dist.sampleN(100); // 100 random count vectors
   * ```
   */
  sampleN(m: number): number[][] {
    const samples: number[][] = new Array(m);
    for (let i = 0; i < m; i++) samples[i] = this.sample();
    return samples;
  }

  /**
   * Computes the log-likelihood of a set of observations.
   *
   * Formula: LL = sum_j log P(X = x_j)
   *
   * @param data - An array of observed count vectors, each summing to n.
   * @returns The total log-likelihood.
   *
   * @example
   * ```ts
   * const dist = new Multinomial(10, [0.2, 0.3, 0.5]);
   * const obs = dist.sampleN(50);
   * dist.logLikelihood(obs); // log-likelihood of the observations
   * ```
   */
  logLikelihood(data: number[][]): number {
    let ll = 0;
    for (const x of data) ll += this.logPmf(x);
    return ll;
  }
}

/**
 * Samples from a Binomial(n, p) distribution via simple trial simulation.
 *
 * @param n - Number of trials.
 * @param p - Probability of success per trial.
 * @param rng - Random number generator.
 * @returns A random count of successes in n trials.
 */
function sampleBinomial(n: number, p: number, rng: RandomFn): number {
  let count = 0;
  for (let i = 0; i < n; i++) {
    if (rng() < p) count++;
  }
  return count;
}
