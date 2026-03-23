import { BaseDiscrete } from "../base";
import { RandomFn } from "../../types";

export class Geometric extends BaseDiscrete {
  readonly name: string;

  constructor(public readonly p: number = 0.5, rng?: RandomFn) {
    super(rng);
    if (p <= 0 || p > 1) throw new Error("p must be in (0, 1]");
    this.name = `Geometric(${p})`;
  }

  mean(): number {
    return (1 - this.p) / this.p;
  }

  variance(): number {
    return (1 - this.p) / (this.p * this.p);
  }

  /** P(X = k) where k = 0, 1, 2, ... (number of failures before first success). */
  pmf(k: number): number {
    if (!Number.isInteger(k) || k < 0) return 0;
    return this.p * Math.pow(1 - this.p, k);
  }

  cdf(k: number): number {
    if (k < 0) return 0;
    return 1 - Math.pow(1 - this.p, Math.floor(k) + 1);
  }

  quantile(prob: number): number {
    if (prob < 0 || prob > 1) throw new Error("p must be in [0, 1]");
    if (prob === 0) return 0;
    if (prob === 1) return Infinity;
    return Math.floor(Math.log(1 - prob) / Math.log(1 - this.p));
  }

  sample(): number {
    return Math.floor(Math.log(1 - this.rng()) / Math.log(1 - this.p));
  }
}
