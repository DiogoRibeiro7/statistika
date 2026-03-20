import { BaseContinuous } from "../base";

export class Pareto extends BaseContinuous {
  readonly name: string;

  constructor(
    public readonly alpha: number,
    public readonly xm: number = 1,
  ) {
    super();
    if (alpha <= 0) throw new Error("alpha must be positive");
    if (xm <= 0) throw new Error("xm must be positive");
    this.name = `Pareto(${alpha}, ${xm})`;
  }

  mean(): number {
    if (this.alpha <= 1) return Infinity;
    return (this.alpha * this.xm) / (this.alpha - 1);
  }

  variance(): number {
    if (this.alpha <= 2) return Infinity;
    const { alpha, xm } = this;
    return (xm * xm * alpha) / ((alpha - 1) ** 2 * (alpha - 2));
  }

  pdf(x: number): number {
    if (x < this.xm) return 0;
    const { alpha, xm } = this;
    return (alpha * Math.pow(xm, alpha)) / Math.pow(x, alpha + 1);
  }

  cdf(x: number): number {
    if (x < this.xm) return 0;
    return 1 - Math.pow(this.xm / x, this.alpha);
  }

  quantile(p: number): number {
    if (p < 0 || p > 1) throw new Error("p must be in [0, 1]");
    if (p === 0) return this.xm;
    if (p === 1) return Infinity;
    return this.xm / Math.pow(1 - p, 1 / this.alpha);
  }

  sample(): number {
    const u = Math.random();
    return this.quantile(u);
  }
}
