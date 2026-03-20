import { BaseDiscrete } from "../base";

export class DiscreteUniform extends BaseDiscrete {
  readonly name: string;
  private readonly range: number;

  constructor(
    public readonly a: number = 0,
    public readonly b: number = 1,
  ) {
    super();
    if (!Number.isInteger(a) || !Number.isInteger(b)) {
      throw new Error("a and b must be integers");
    }
    if (a >= b) throw new Error("a must be less than b");
    this.name = `DiscreteUniform(${a}, ${b})`;
    this.range = b - a + 1;
  }

  mean(): number {
    return (this.a + this.b) / 2;
  }

  variance(): number {
    return (this.range * this.range - 1) / 12;
  }

  pmf(k: number): number {
    if (!Number.isInteger(k) || k < this.a || k > this.b) return 0;
    return 1 / this.range;
  }

  cdf(k: number): number {
    if (k < this.a) return 0;
    if (k >= this.b) return 1;
    return (Math.floor(k) - this.a + 1) / this.range;
  }

  quantile(p: number): number {
    if (p < 0 || p > 1) throw new Error("p must be in [0, 1]");
    if (p === 0) return this.a;
    if (p === 1) return this.b;
    return Math.min(this.a + Math.floor(p * this.range), this.b);
  }

  sample(): number {
    return this.a + Math.floor(Math.random() * this.range);
  }
}
