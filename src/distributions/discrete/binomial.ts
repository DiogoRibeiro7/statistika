import { BaseDiscrete } from "../base";
import { logFactorial, regularizedBeta } from "../../utils/math";
import { RandomFn } from "../../types";

/**
 * Binomial distribution.
 *
 * Models the number of successes in n independent Bernoulli trials,
 * each with success probability p.
 *
 * P(X = k) = C(n, k) * p^k * (1 - p)^(n - k), for k = 0, 1, ..., n
 *
 * @example
 * ```ts
 * const dist = new Binomial(10, 0.3);
 * dist.mean();     // 3
 * dist.pmf(3);     // P(X = 3)
 * dist.cdf(5);     // P(X <= 5)
 * ```
 */
export class Binomial extends BaseDiscrete {
  readonly name: string;

  /**
   * @param n - Number of trials (positive integer). Defaults to 1.
   * @param p - Probability of success per trial, must be in [0, 1]. Defaults to 0.5.
   * @param rng - Optional random number generator.
   */
  constructor(
    public readonly n: number = 1,
    public readonly p: number = 0.5,
    rng?: RandomFn,
  ) {
    super(rng);
    if (n < 1 || !Number.isInteger(n)) throw new Error(`Invalid parameter 'n': expected a positive integer, received ${n}`);
    if (p < 0 || p > 1) throw new Error(`Invalid parameter 'p': expected a value in [0, 1], received ${p}`);
    this.name = `Binomial(${n}, ${p})`;
  }

  /** Returns the mean: E[X] = np. */
  mean(): number {
    return this.n * this.p;
  }

  /** Returns the variance: Var(X) = np(1 - p). */
  variance(): number {
    return this.n * this.p * (1 - this.p);
  }

  /**
   * Probability mass function.
   *
   * P(X = k) = C(n, k) * p^k * (1 - p)^(n - k)
   *
   * Computed in log-space for numerical stability.
   *
   * @param k - The number of successes (integer in [0, n]).
   * @returns The probability P(X = k).
   */
  pmf(k: number): number {
    if (!Number.isInteger(k) || k < 0 || k > this.n) return 0;
    if (this.p === 0) return k === 0 ? 1 : 0;
    if (this.p === 1) return k === this.n ? 1 : 0;
    const logPmf =
      logFactorial(this.n) -
      logFactorial(k) -
      logFactorial(this.n - k) +
      k * Math.log(this.p) +
      (this.n - k) * Math.log(1 - this.p);
    return Math.exp(logPmf);
  }

  /**
   * Computes the log of the probability mass function at `k`.
   *
   * log P(X = k) = log(C(n,k)) + k*log(p) + (n-k)*log(1-p)
   *
   * @param k - The number of successes (integer in [0, n]).
   * @returns The log-probability. Returns -Infinity for values outside [0, n].
   */
  logPmf(k: number): number {
    if (!Number.isInteger(k) || k < 0 || k > this.n) return -Infinity;
    if (this.p === 0) return k === 0 ? 0 : -Infinity;
    if (this.p === 1) return k === this.n ? 0 : -Infinity;
    return (
      logFactorial(this.n) -
      logFactorial(k) -
      logFactorial(this.n - k) +
      k * Math.log(this.p) +
      (this.n - k) * Math.log(1 - this.p)
    );
  }

  /**
   * Returns the skewness of the Binomial distribution.
   *
   * Formula: (1 - 2*p) / sqrt(n * p * (1 - p))
   */
  get skewness(): number {
    return (1 - 2 * this.p) / Math.sqrt(this.n * this.p * (1 - this.p));
  }

  /**
   * Returns the excess kurtosis of the Binomial distribution.
   *
   * Formula: (1 - 6*p*(1-p)) / (n * p * (1 - p))
   */
  get kurtosis(): number {
    return (1 - 6 * this.p * (1 - this.p)) / (this.n * this.p * (1 - this.p));
  }

  /**
   * Returns the mode of the Binomial distribution.
   *
   * Formula: floor((n + 1) * p)
   */
  get mode(): number {
    return Math.floor((this.n + 1) * this.p);
  }

  /**
   * Cumulative distribution function.
   *
   * P(X <= k) = I_{1-p}(n - k, k + 1), where I is the regularized incomplete beta function.
   *
   * @param k - The value at which to evaluate the CDF.
   * @returns The cumulative probability P(X <= k).
   */
  cdf(k: number): number {
    if (k < 0) return 0;
    if (k >= this.n) return 1;
    const kFloor = Math.floor(k);
    // CDF via regularized incomplete beta: P(X <= k) = I_{1-p}(n-k, k+1)
    return regularizedBeta(1 - this.p, this.n - kFloor, kFloor + 1);
  }

  /**
   * Quantile function (inverse CDF).
   *
   * Returns the smallest integer k such that P(X <= k) >= prob.
   *
   * @param prob - The probability, must be in [0, 1].
   * @returns The quantile value.
   */
  quantile(prob: number): number {
    if (prob < 0 || prob > 1) throw new Error(`Invalid parameter 'prob': expected a value in [0, 1], received ${prob}`);
    if (prob === 0) return 0;
    if (prob === 1) return this.n;
    // Linear search (fine for moderate n)
    let cumulative = 0;
    for (let k = 0; k <= this.n; k++) {
      cumulative += this.pmf(k);
      if (cumulative >= prob) return k;
    }
    return this.n;
  }

  /**
   * Draws a single random sample by simulating n independent Bernoulli trials.
   *
   * @returns The number of successes in n trials.
   */
  sample(): number {
    let successes = 0;
    for (let i = 0; i < this.n; i++) {
      if (this.rng() < this.p) successes++;
    }
    return successes;
  }
}
