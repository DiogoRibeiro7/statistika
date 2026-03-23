/**
 * Zero-Inflated Poisson (ZIP) distribution.
 *
 * A mixture model:  with probability π the outcome is always 0,
 * with probability (1 − π) the outcome follows Poisson(λ).
 *
 * P(X = 0)   = π + (1 − π) e^{−λ}
 * P(X = k)   = (1 − π) × Poisson(k; λ)     for k ≥ 1
 */

import { BaseDiscrete } from "../base";
import { logFactorial, regularizedGammaP } from "../../utils/math";
import { RandomFn } from "../../types";

export class ZeroInflatedPoisson extends BaseDiscrete {
  readonly name: string;

  /**
   * @param lambda - Poisson rate parameter (positive).
   * @param pi - Zero-inflation probability, P(structural zero), in [0, 1).
   */
  constructor(
    public readonly lambda: number,
    public readonly pi: number,
    rng?: RandomFn,
  ) {
    super(rng);
    if (lambda <= 0) throw new Error("lambda must be positive");
    if (pi < 0 || pi >= 1) throw new Error("pi must be in [0, 1)");
    this.name = `ZIP(${lambda}, ${pi})`;
  }

  /** E[X] = (1 − π) λ. */
  mean(): number {
    return (1 - this.pi) * this.lambda;
  }

  /** Var[X] = (1 − π) λ (1 + π λ). */
  variance(): number {
    return (1 - this.pi) * this.lambda * (1 + this.pi * this.lambda);
  }

  pmf(k: number): number {
    if (!Number.isInteger(k) || k < 0) return 0;
    if (k === 0) {
      return this.pi + (1 - this.pi) * Math.exp(-this.lambda);
    }
    const logPoisson =
      k * Math.log(this.lambda) - this.lambda - logFactorial(k);
    return (1 - this.pi) * Math.exp(logPoisson);
  }

  cdf(k: number): number {
    if (k < 0) return 0;
    const kFloor = Math.floor(k);
    // CDF = pi + (1-pi) * Poisson_CDF(k; lambda)
    const poissonCdf = 1 - regularizedGammaP(kFloor + 1, this.lambda);
    return this.pi + (1 - this.pi) * poissonCdf;
  }

  quantile(prob: number): number {
    if (prob < 0 || prob > 1) throw new Error("p must be in [0, 1]");
    if (prob === 0) return 0;
    if (prob === 1) return Infinity;
    let cumulative = 0;
    let k = 0;
    const limit = this.lambda + 40 * Math.sqrt(this.lambda) + 100;
    while (cumulative < prob) {
      cumulative += this.pmf(k);
      if (cumulative >= prob) return k;
      k++;
      if (k > limit) return k;
    }
    return k;
  }

  sample(): number {
    // With probability pi, return 0 (structural zero)
    if (this.rng() < this.pi) return 0;
    // Otherwise sample from Poisson(lambda)
    if (this.lambda < 30) {
      const L = Math.exp(-this.lambda);
      let count = 0;
      let p = 1;
      do {
        count++;
        p *= this.rng();
      } while (p > L);
      return count - 1;
    }
    return this.quantile(this.rng());
  }
}
