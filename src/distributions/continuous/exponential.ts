import { BaseContinuous } from "../base";
import { RandomFn } from "../../types";

export class Exponential extends BaseContinuous {
  readonly name: string;

  constructor(public readonly lambda: number = 1, rng?: RandomFn) {
    super(rng);
    if (lambda <= 0) throw new Error("lambda must be positive");
    this.name = `Exponential(${lambda})`;
  }

  mean(): number {
    return 1 / this.lambda;
  }

  variance(): number {
    return 1 / this.lambda ** 2;
  }

  pdf(x: number): number {
    return x < 0 ? 0 : this.lambda * Math.exp(-this.lambda * x);
  }

  cdf(x: number): number {
    return x < 0 ? 0 : 1 - Math.exp(-this.lambda * x);
  }

  quantile(p: number): number {
    if (p < 0 || p > 1) throw new Error("p must be in [0, 1]");
    if (p === 1) return Infinity;
    return -Math.log(1 - p) / this.lambda;
  }

  sample(): number {
    return -Math.log(1 - this.rng()) / this.lambda;
  }
}
