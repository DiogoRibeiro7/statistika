import { BaseContinuous } from "../base";
import { RandomFn } from "../../types";

/**
 * Gumbel distribution (Type I extreme value distribution).
 *
 * Also known as the log-Weibull distribution. Models the distribution
 * of the maximum (or minimum) of a number of samples of various distributions.
 *
 * Parameters:
 *   mu    — location
 *   beta  — scale (> 0)
 */
export class Gumbel extends BaseContinuous {
  readonly name: string;

  constructor(
    public readonly mu: number = 0,
    public readonly beta: number = 1,
    rng?: RandomFn,
  ) {
    super(rng);
    if (beta <= 0) throw new Error("beta must be positive");
    this.name = `Gumbel(${mu}, ${beta})`;
  }

  mean(): number {
    // Euler-Mascheroni constant
    return this.mu + this.beta * 0.5772156649015329;
  }

  variance(): number {
    return (Math.PI ** 2 * this.beta ** 2) / 6;
  }

  pdf(x: number): number {
    const z = (x - this.mu) / this.beta;
    return Math.exp(-(z + Math.exp(-z))) / this.beta;
  }

  cdf(x: number): number {
    const z = (x - this.mu) / this.beta;
    return Math.exp(-Math.exp(-z));
  }

  quantile(p: number): number {
    if (p < 0 || p > 1) throw new Error("p must be in [0, 1]");
    if (p === 0) return -Infinity;
    if (p === 1) return Infinity;
    return this.mu - this.beta * Math.log(-Math.log(p));
  }

  sample(): number {
    return this.quantile(this.rng());
  }
}
