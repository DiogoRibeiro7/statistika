import { BaseContinuous } from "../base";
import { gamma } from "../../utils/math";
import { RandomFn } from "../../types";

/**
 * Fréchet distribution (Type II extreme value distribution).
 *
 * Models the distribution of the maximum of samples from heavy-tailed
 * distributions (e.g., Pareto, Cauchy).
 *
 * Parameters:
 *   alpha — shape (> 0), also called the tail index
 *   s     — scale (> 0)
 *   m     — location
 */
export class Frechet extends BaseContinuous {
  readonly name: string;

  constructor(
    public readonly alpha: number = 1,
    public readonly s: number = 1,
    public readonly m: number = 0,
    rng?: RandomFn,
  ) {
    super(rng);
    if (alpha <= 0) throw new Error("alpha must be positive");
    if (s <= 0) throw new Error("s must be positive");
    this.name = `Frechet(${alpha}, ${s}, ${m})`;
  }

  mean(): number {
    if (this.alpha <= 1) return Infinity;
    return this.m + this.s * gamma(1 - 1 / this.alpha);
  }

  variance(): number {
    if (this.alpha <= 2) return Infinity;
    const g1 = gamma(1 - 1 / this.alpha);
    const g2 = gamma(1 - 2 / this.alpha);
    return this.s ** 2 * (g2 - g1 ** 2);
  }

  pdf(x: number): number {
    if (x <= this.m) return 0;
    const z = (x - this.m) / this.s;
    return (
      (this.alpha / this.s) *
      Math.pow(z, -1 - this.alpha) *
      Math.exp(-Math.pow(z, -this.alpha))
    );
  }

  cdf(x: number): number {
    if (x <= this.m) return 0;
    const z = (x - this.m) / this.s;
    return Math.exp(-Math.pow(z, -this.alpha));
  }

  quantile(p: number): number {
    if (p < 0 || p > 1) throw new Error("p must be in [0, 1]");
    if (p === 0) return this.m;
    if (p === 1) return Infinity;
    return this.m + this.s * Math.pow(-Math.log(p), -1 / this.alpha);
  }

  sample(): number {
    return this.quantile(this.rng());
  }
}
