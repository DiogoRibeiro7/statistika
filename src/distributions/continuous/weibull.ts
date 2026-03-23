import { BaseContinuous } from "../base";
import { gamma } from "../../utils/math";
import { RandomFn } from "../../types";

/**
 * Weibull distribution.
 *
 * Widely used in reliability engineering and survival analysis.
 * The Weibull minimum is the Type III extreme value distribution
 * for minima.
 *
 * Parameters:
 *   k      — shape (> 0)
 *   lambda — scale (> 0)
 */
export class Weibull extends BaseContinuous {
  readonly name: string;

  constructor(
    public readonly k: number = 1,
    public readonly lambda: number = 1,
    rng?: RandomFn,
  ) {
    super(rng);
    if (k <= 0) throw new Error("k (shape) must be positive");
    if (lambda <= 0) throw new Error("lambda (scale) must be positive");
    this.name = `Weibull(${k}, ${lambda})`;
  }

  mean(): number {
    return this.lambda * gamma(1 + 1 / this.k);
  }

  variance(): number {
    const g1 = gamma(1 + 1 / this.k);
    const g2 = gamma(1 + 2 / this.k);
    return this.lambda ** 2 * (g2 - g1 ** 2);
  }

  pdf(x: number): number {
    if (x < 0) return 0;
    if (x === 0) {
      if (this.k === 1) return 1 / this.lambda;
      if (this.k < 1) return Infinity;
      return 0;
    }
    const z = x / this.lambda;
    return (
      (this.k / this.lambda) *
      Math.pow(z, this.k - 1) *
      Math.exp(-Math.pow(z, this.k))
    );
  }

  cdf(x: number): number {
    if (x < 0) return 0;
    return 1 - Math.exp(-Math.pow(x / this.lambda, this.k));
  }

  quantile(p: number): number {
    if (p < 0 || p > 1) throw new Error("p must be in [0, 1]");
    if (p === 0) return 0;
    if (p === 1) return Infinity;
    return this.lambda * Math.pow(-Math.log(1 - p), 1 / this.k);
  }

  sample(): number {
    return this.quantile(this.rng());
  }
}
