import { BaseDiscrete } from "../base";
import { logFactorial } from "../../utils/math";
import { RandomFn } from "../../types";

/**
 * Hypergeometric distribution modeling the number of successes in draws
 * without replacement from a finite population.
 *
 * Given a population of N items containing K successes, and drawing n items
 * without replacement, the random variable X counts the number of successes
 * in the draw.
 *
 * PMF: P(X = k) = C(K, k) * C(N - K, n - k) / C(N, n)
 *
 * where C(a, b) denotes the binomial coefficient "a choose b".
 *
 * @example
 * ```ts
 * // Urn with 50 balls: 10 red, 40 blue. Draw 5 without replacement.
 * const dist = new Hypergeometric(50, 10, 5);
 * dist.mean();     // 1
 * dist.pmf(2);     // probability of drawing exactly 2 red balls
 * dist.sample();   // random number of red balls drawn
 * ```
 */
export class Hypergeometric extends BaseDiscrete {
  readonly name: string;

  /**
   * Creates a new Hypergeometric distribution.
   *
   * @param N - Population size (non-negative integer).
   * @param K - Number of success states in the population, must be in [0, N].
   * @param n - Number of draws (sample size), must be in [0, N].
   * @param rng - Optional custom random number generator; defaults to Math.random.
   * @throws {Error} If N is not a non-negative integer.
   * @throws {Error} If K is not an integer in [0, N].
   * @throws {Error} If n is not an integer in [0, N].
   *
   * @example
   * ```ts
   * const dist = new Hypergeometric(100, 30, 10);
   * ```
   */
  constructor(
    public readonly N: number,
    public readonly K: number,
    public readonly n: number,
    rng?: RandomFn,
  ) {
    super(rng);
    if (N < 0 || !Number.isInteger(N)) throw new Error(`Invalid parameter 'N': expected a non-negative integer, received ${N}`);
    if (K < 0 || K > N || !Number.isInteger(K)) throw new Error(`Invalid parameter 'K': expected an integer in [0, N=${N}], received ${K}`);
    if (n < 0 || n > N || !Number.isInteger(n)) throw new Error(`Invalid parameter 'n': expected an integer in [0, N=${N}], received ${n}`);
    this.name = `Hypergeometric(${N}, ${K}, ${n})`;
  }

  /**
   * Computes the mean (expected value) of the distribution.
   *
   * Formula: E[X] = n * K / N
   *
   * @returns The expected number of successes in the draw.
   *
   * @example
   * ```ts
   * new Hypergeometric(50, 10, 5).mean(); // 1
   * ```
   */
  mean(): number {
    return this.n * this.K / this.N;
  }

  /**
   * Computes the variance of the distribution.
   *
   * Formula: Var(X) = n * K * (N - K) * (N - n) / (N^2 * (N - 1))
   *
   * @returns The variance of the distribution.
   *
   * @example
   * ```ts
   * new Hypergeometric(50, 10, 5).variance(); // ~0.6531
   * ```
   */
  variance(): number {
    const { N, K, n } = this;
    return (n * K * (N - K) * (N - n)) / (N * N * (N - 1));
  }

  /**
   * Computes the probability mass function P(X = k).
   *
   * Uses log-factorials for numerical stability:
   * P(X = k) = C(K, k) * C(N - K, n - k) / C(N, n)
   *
   * @param k - The number of observed successes. Must be an integer in [max(0, n + K - N), min(n, K)].
   * @returns The probability P(X = k). Returns 0 for values outside the support.
   *
   * @example
   * ```ts
   * const dist = new Hypergeometric(50, 10, 5);
   * dist.pmf(1);   // probability of exactly 1 success
   * dist.pmf(0);   // probability of no successes
   * ```
   */
  pmf(k: number): number {
    if (!Number.isInteger(k)) return 0;
    const { N, K, n } = this;
    const lo = Math.max(0, n + K - N);
    const hi = Math.min(n, K);
    if (k < lo || k > hi) return 0;
    // C(K,k) * C(N-K, n-k) / C(N, n)
    const logPmf =
      logFactorial(K) - logFactorial(k) - logFactorial(K - k) +
      logFactorial(N - K) - logFactorial(n - k) - logFactorial(N - K - n + k) -
      logFactorial(N) + logFactorial(n) + logFactorial(N - n);
    return Math.exp(logPmf);
  }

  /**
   * Computes the log of the probability mass function at `k`.
   *
   * log P(X = k) = log(C(K,k)) + log(C(N-K, n-k)) - log(C(N, n))
   *
   * @param k - The number of observed successes.
   * @returns The log-probability. Returns -Infinity outside the support.
   */
  logPmf(k: number): number {
    if (!Number.isInteger(k)) return -Infinity;
    const { N, K, n } = this;
    const lo = Math.max(0, n + K - N);
    const hi = Math.min(n, K);
    if (k < lo || k > hi) return -Infinity;
    return (
      logFactorial(K) - logFactorial(k) - logFactorial(K - k) +
      logFactorial(N - K) - logFactorial(n - k) - logFactorial(N - K - n + k) -
      logFactorial(N) + logFactorial(n) + logFactorial(N - n)
    );
  }

  /**
   * Returns the skewness of the Hypergeometric distribution.
   *
   * Formula: ((N - 2*K) * sqrt(N - 1) * (N - 2*n)) /
   *          (sqrt(n * K * (N - K) * (N - n)) * (N - 2))
   */
  get skewness(): number {
    const { N, K, n } = this;
    if (N <= 2) return NaN;
    return (
      ((N - 2 * K) * Math.sqrt(N - 1) * (N - 2 * n)) /
      (Math.sqrt(n * K * (N - K) * (N - n)) * (N - 2))
    );
  }

  /**
   * Returns the excess kurtosis of the Hypergeometric distribution.
   *
   * Formula: ((N-1) * N^2 * [N*(N+1) - 6*K*(N-K) - 6*n*(N-n)]) /
   *          (n * K * (N-K) * (N-n) * (N-2) * (N-3))
   */
  get kurtosis(): number {
    const { N, K, n } = this;
    if (N <= 3) return NaN;
    const denom = n * K * (N - K) * (N - n) * (N - 2) * (N - 3);
    if (denom === 0) return NaN;
    const numer =
      (N - 1) * N * N *
      (N * (N + 1) - 6 * K * (N - K) - 6 * n * (N - n));
    return numer / denom;
  }

  /**
   * Returns the mode of the Hypergeometric distribution.
   *
   * Formula: floor((n + 1) * (K + 1) / (N + 2))
   */
  get mode(): number {
    return Math.floor(((this.n + 1) * (this.K + 1)) / (this.N + 2));
  }

  /**
   * Computes the cumulative distribution function P(X <= k).
   *
   * Computed by summing the PMF from the lower bound of the support up to floor(k).
   *
   * @param k - The value at which to evaluate the CDF.
   * @returns The cumulative probability P(X <= k).
   *
   * @example
   * ```ts
   * const dist = new Hypergeometric(50, 10, 5);
   * dist.cdf(1);   // P(X <= 1)
   * dist.cdf(5);   // 1 (if 5 >= min(n, K))
   * ```
   */
  cdf(k: number): number {
    const { N, K, n } = this;
    const lo = Math.max(0, n + K - N);
    if (k < lo) return 0;
    const hi = Math.min(n, K);
    if (k >= hi) return 1;
    const kFloor = Math.floor(k);
    let sum = 0;
    for (let i = lo; i <= kFloor; i++) {
      sum += this.pmf(i);
    }
    return sum;
  }

  /**
   * Computes the quantile (inverse CDF) function.
   *
   * Returns the smallest integer k such that P(X <= k) >= prob.
   *
   * @param prob - The probability, must be in [0, 1].
   * @returns The quantile value.
   * @throws {Error} If prob is not in [0, 1].
   *
   * @example
   * ```ts
   * const dist = new Hypergeometric(50, 10, 5);
   * dist.quantile(0.5);  // median number of successes
   * ```
   */
  quantile(prob: number): number {
    if (prob < 0 || prob > 1) throw new Error(`Invalid parameter 'prob': expected a value in [0, 1], received ${prob}`);
    const { N, K, n } = this;
    const lo = Math.max(0, n + K - N);
    const hi = Math.min(n, K);
    if (prob === 0) return lo;
    if (prob === 1) return hi;
    let cumulative = 0;
    for (let k = lo; k <= hi; k++) {
      cumulative += this.pmf(k);
      if (cumulative >= prob) return k;
    }
    return hi;
  }

  /**
   * Draws a single random sample using direct simulation.
   *
   * Simulates drawing n balls one at a time (without replacement) from an urn
   * containing K success balls and N - K failure balls.
   *
   * @returns A random integer representing the number of successes in the draw.
   *
   * @example
   * ```ts
   * const dist = new Hypergeometric(50, 10, 5);
   * const successes = dist.sample();
   * ```
   */
  sample(): number {
    // Direct simulation: draw n balls from urn of N (K success, N-K failure)
    const { N, K, n } = this;
    let successes = 0;
    let remaining = N;
    let kRemaining = K;
    for (let i = 0; i < n; i++) {
      if (this.rng() < kRemaining / remaining) {
        successes++;
        kRemaining--;
      }
      remaining--;
    }
    return successes;
  }
}
