import { BaseDiscrete } from "../base";
import { logFactorial, regularizedBeta } from "../../utils/math";
import { RandomFn } from "../../types";

/**
 * Negative Binomial distribution modeling the number of failures
 * before achieving a specified number of successes.
 *
 * Given independent Bernoulli trials each with success probability p,
 * the random variable X counts the number of failures before the r-th success.
 *
 * PMF: P(X = k) = C(k + r - 1, k) * p^r * (1 - p)^k for k = 0, 1, 2, ...
 *
 * @example
 * ```ts
 * const dist = new NegativeBinomial(5, 0.4);
 * dist.mean();     // 7.5
 * dist.pmf(3);     // probability of exactly 3 failures before 5 successes
 * dist.sample();   // random non-negative integer
 * ```
 */
export class NegativeBinomial extends BaseDiscrete {
  readonly name: string;

  /**
   * Creates a new Negative Binomial distribution.
   *
   * @param r - Number of successes required (positive integer).
   * @param p - Probability of success on each trial, must be in (0, 1].
   * @param rng - Optional custom random number generator; defaults to Math.random.
   * @throws {Error} If r is not a positive integer.
   * @throws {Error} If p is not in (0, 1].
   *
   * @example
   * ```ts
   * const dist = new NegativeBinomial(3, 0.5);
   * ```
   */
  constructor(
    public readonly r: number,
    public readonly p: number,
    rng?: RandomFn,
  ) {
    super(rng);
    if (r <= 0 || !Number.isInteger(r)) throw new Error(`Invalid parameter 'r': expected a positive integer, received ${r}`);
    if (p <= 0 || p > 1) throw new Error(`Invalid parameter 'p': expected a value in (0, 1], received ${p}`);
    this.name = `NegBin(${r}, ${p})`;
  }

  /**
   * Computes the mean (expected value) of the distribution.
   *
   * Formula: E[X] = r * (1 - p) / p
   *
   * @returns The expected number of failures before the r-th success.
   *
   * @example
   * ```ts
   * new NegativeBinomial(5, 0.5).mean(); // 5
   * ```
   */
  mean(): number {
    return this.r * (1 - this.p) / this.p;
  }

  /**
   * Computes the variance of the distribution.
   *
   * Formula: Var(X) = r * (1 - p) / p^2
   *
   * @returns The variance of the distribution.
   *
   * @example
   * ```ts
   * new NegativeBinomial(5, 0.5).variance(); // 10
   * ```
   */
  variance(): number {
    return this.r * (1 - this.p) / (this.p * this.p);
  }

  /**
   * Computes the probability mass function P(X = k).
   *
   * Uses log-factorials for numerical stability:
   * P(X = k) = C(k + r - 1, k) * p^r * (1 - p)^k
   *
   * @param k - The number of failures (non-negative integer).
   * @returns The probability P(X = k). Returns 0 for non-integer or negative k.
   *
   * @example
   * ```ts
   * const dist = new NegativeBinomial(3, 0.5);
   * dist.pmf(0);  // p^r = 0.125
   * dist.pmf(2);  // probability of 2 failures before 3 successes
   * ```
   */
  pmf(k: number): number {
    if (!Number.isInteger(k) || k < 0) return 0;
    const { r, p } = this;
    // C(k+r-1, k) * p^r * (1-p)^k
    const logPmf =
      logFactorial(k + r - 1) -
      logFactorial(k) -
      logFactorial(r - 1) +
      r * Math.log(p) +
      k * Math.log(1 - p);
    return Math.exp(logPmf);
  }

  /**
   * Computes the log of the probability mass function at `k`.
   *
   * log P(X = k) = log(C(k+r-1, k)) + r*log(p) + k*log(1-p)
   *
   * @param k - The number of failures (non-negative integer).
   * @returns The log-probability. Returns -Infinity for non-integer or negative k.
   */
  logPmf(k: number): number {
    if (!Number.isInteger(k) || k < 0) return -Infinity;
    const { r, p } = this;
    return (
      logFactorial(k + r - 1) -
      logFactorial(k) -
      logFactorial(r - 1) +
      r * Math.log(p) +
      k * Math.log(1 - p)
    );
  }

  /**
   * Returns the skewness of the Negative Binomial distribution.
   *
   * Formula: (2 - p) / sqrt(r * (1 - p))
   */
  get skewness(): number {
    return (2 - this.p) / Math.sqrt(this.r * (1 - this.p));
  }

  /**
   * Returns the excess kurtosis of the Negative Binomial distribution.
   *
   * Formula: 6 / r + p^2 / (r * (1 - p))
   */
  get kurtosis(): number {
    return 6 / this.r + (this.p * this.p) / (this.r * (1 - this.p));
  }

  /**
   * Returns the mode of the Negative Binomial distribution.
   *
   * Formula: floor((r - 1) * (1 - p) / p) for r > 1, 0 for r <= 1.
   */
  get mode(): number {
    if (this.r <= 1) return 0;
    return Math.floor((this.r - 1) * (1 - this.p) / this.p);
  }

  /**
   * Computes the cumulative distribution function P(X <= k).
   *
   * Uses the regularized incomplete beta function:
   * F(k) = I_p(r, floor(k) + 1)
   *
   * @param k - The value at which to evaluate the CDF.
   * @returns The cumulative probability P(X <= k).
   *
   * @example
   * ```ts
   * const dist = new NegativeBinomial(3, 0.5);
   * dist.cdf(3);  // P(X <= 3)
   * dist.cdf(-1); // 0
   * ```
   */
  cdf(k: number): number {
    if (k < 0) return 0;
    const kFloor = Math.floor(k);
    // CDF = I_p(r, k+1)
    return regularizedBeta(this.p, this.r, kFloor + 1);
  }

  /**
   * Computes the quantile (inverse CDF) function.
   *
   * Returns the smallest integer k such that P(X <= k) >= prob.
   * Uses iterative PMF summation with a safety limit of 10000.
   *
   * @param prob - The probability, must be in [0, 1].
   * @returns The quantile value.
   * @throws {Error} If prob is not in [0, 1].
   *
   * @example
   * ```ts
   * const dist = new NegativeBinomial(3, 0.5);
   * dist.quantile(0);    // 0
   * dist.quantile(0.5);  // median
   * dist.quantile(1);    // Infinity
   * ```
   */
  quantile(prob: number): number {
    if (prob < 0 || prob > 1) throw new Error(`Invalid parameter 'prob': expected a value in [0, 1], received ${prob}`);
    if (prob === 0) return 0;
    if (prob === 1) return Infinity;
    let cumulative = 0;
    for (let k = 0; ; k++) {
      cumulative += this.pmf(k);
      if (cumulative >= prob) return k;
      if (k > 10000) return k; // safety limit
    }
  }

  /**
   * Draws a single random sample from the distribution.
   *
   * Uses the sum-of-geometrics method: the negative binomial with parameters
   * (r, p) is the sum of r independent geometric(p) random variables.
   *
   * @returns A random non-negative integer representing the number of failures before r successes.
   *
   * @example
   * ```ts
   * const dist = new NegativeBinomial(3, 0.5);
   * const failures = dist.sample();
   * ```
   */
  sample(): number {
    // Sum of r geometric samples
    let total = 0;
    for (let i = 0; i < this.r; i++) {
      let k = 0;
      while (this.rng() >= this.p) k++;
      total += k;
    }
    return total;
  }
}
