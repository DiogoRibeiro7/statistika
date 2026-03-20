import { BaseDiscrete } from "../base";

export class Bernoulli extends BaseDiscrete {
  readonly name: string;

  constructor(public readonly p: number = 0.5) {
    super();
    if (p < 0 || p > 1) throw new Error("p must be in [0, 1]");
    this.name = `Bernoulli(${p})`;
  }

  mean(): number {
    return this.p;
  }

  variance(): number {
    return this.p * (1 - this.p);
  }

  pmf(k: number): number {
    if (k === 0) return 1 - this.p;
    if (k === 1) return this.p;
    return 0;
  }

  cdf(k: number): number {
    if (k < 0) return 0;
    if (k < 1) return 1 - this.p;
    return 1;
  }

  quantile(prob: number): number {
    if (prob < 0 || prob > 1) throw new Error("p must be in [0, 1]");
    return prob <= 1 - this.p ? 0 : 1;
  }

  sample(): number {
    return Math.random() < this.p ? 1 : 0;
  }
}
