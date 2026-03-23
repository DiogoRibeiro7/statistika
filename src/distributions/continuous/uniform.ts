import { BaseContinuous } from "../base";
import { RandomFn } from "../../types";

export class Uniform extends BaseContinuous {
  readonly name: string;

  constructor(
    public readonly a: number = 0,
    public readonly b: number = 1,
    rng?: RandomFn,
  ) {
    super(rng);
    if (a >= b) throw new Error("a must be less than b");
    this.name = `Uniform(${a}, ${b})`;
  }

  mean(): number {
    return (this.a + this.b) / 2;
  }

  variance(): number {
    return (this.b - this.a) ** 2 / 12;
  }

  pdf(x: number): number {
    return x >= this.a && x <= this.b ? 1 / (this.b - this.a) : 0;
  }

  cdf(x: number): number {
    if (x < this.a) return 0;
    if (x > this.b) return 1;
    return (x - this.a) / (this.b - this.a);
  }

  quantile(p: number): number {
    if (p < 0 || p > 1) throw new Error("p must be in [0, 1]");
    return this.a + p * (this.b - this.a);
  }

  sample(): number {
    return this.a + this.rng() * (this.b - this.a);
  }
}
