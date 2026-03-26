import { BaseDiscrete } from "../base";
import { logFactorial, regularizedGammaP } from "../../utils/math";
import { RandomFn } from "../../types";

/**
 * Poisson distribution.
 *
 * Models the number of events occurring in a fixed interval, given a
 * constant average rate lambda.
 *
 * P(X = k) = (lambda^k * e^{-lambda}) / k!, for k = 0, 1, 2, ...
 *
 * @example
 * ```ts
 * const dist = new Poisson(4);
 * dist.mean();     // 4
 * dist.variance(); // 4
 * dist.pmf(3);     // P(X = 3)
 * ```
 */
export class Poisson extends BaseDiscrete {
  readonly name: string;

  /**
   * @param lambda - Rate parameter (must be positive). Defaults to 1.
   * @param rng - Optional random number generator.
   */
  constructor(public readonly lambda: number = 1, rng?: RandomFn) {
    super(rng);
    if (lambda <= 0) throw new Error(`Invalid parameter 'lambda': expected a positive number, received ${lambda}`);
    this.name = `Poisson(${lambda})`;
  }

  /** Returns the mean: E[X] = lambda. */
  mean(): number {
    return this.lambda;
  }

  /** Returns the variance: Var(X) = lambda. */
  variance(): number {
    return this.lambda;
  }

  /**
   * Probability mass function.
   *
   * P(X = k) = (lambda^k * e^{-lambda}) / k!
   *
   * Computed in log-space for numerical stability.
   *
   * @param k - Non-negative integer value.
   * @returns The probability P(X = k).
   */
  pmf(k: number): number {
    if (!Number.isInteger(k) || k < 0) return 0;
    const logPmf = k * Math.log(this.lambda) - this.lambda - logFactorial(k);
    return Math.exp(logPmf);
  }

  /**
   * Computes the log of the probability mass function at `k`.
   *
   * log P(X = k) = k * log(lambda) - lambda - log(k!)
   *
   * @param k - Non-negative integer value.
   * @returns The log-probability. Returns -Infinity for non-integer or negative k.
   */
  logPmf(k: number): number {
    if (!Number.isInteger(k) || k < 0) return -Infinity;
    return k * Math.log(this.lambda) - this.lambda - logFactorial(k);
  }

  /**
   * Returns the skewness of the Poisson distribution.
   *
   * Formula: 1 / sqrt(lambda)
   */
  get skewness(): number {
    return 1 / Math.sqrt(this.lambda);
  }

  /**
   * Returns the excess kurtosis of the Poisson distribution.
   *
   * Formula: 1 / lambda
   */
  get kurtosis(): number {
    return 1 / this.lambda;
  }

  /**
   * Returns the mode of the Poisson distribution.
   *
   * Formula: floor(lambda)
   */
  get mode(): number {
    return Math.floor(this.lambda);
  }

  /**
   * Cumulative distribution function.
   *
   * P(X <= k) = 1 - P(k + 1, lambda), where P is the regularized lower
   * incomplete gamma function.
   *
   * @param k - The value at which to evaluate the CDF.
   * @returns The cumulative probability P(X <= k).
   */
  cdf(k: number): number {
    if (k < 0) return 0;
    const kFloor = Math.floor(k);
    // CDF via regularized upper incomplete gamma: P(X <= k) = Q(k+1, lambda) = 1 - P(k+1, lambda)
    return 1 - regularizedGammaP(kFloor + 1, this.lambda);
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
    if (prob === 1) return Infinity;
    let cumulative = 0;
    let k = 0;
    while (cumulative < prob) {
      cumulative += this.pmf(k);
      if (cumulative >= prob) return k;
      k++;
      if (k > this.lambda + 40 * Math.sqrt(this.lambda)) return k;
    }
    return k;
  }

  /**
   * Draws a single random sample.
   *
   * Uses Knuth's algorithm for small lambda (< 30) and inverse transform
   * sampling for larger values.
   *
   * @returns A non-negative integer drawn from the Poisson distribution.
   */
  sample(): number {
    // Knuth's algorithm for small lambda, otherwise use transformed rejection
    if (this.lambda < 30) {
      const L = Math.exp(-this.lambda);
      let k = 0;
      let p = 1;
      do {
        k++;
        p *= this.rng();
      } while (p > L);
      return k - 1;
    }
    // For large lambda, use inverse transform
    return this.quantile(this.rng());
  }
}
