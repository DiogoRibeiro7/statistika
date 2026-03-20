import { BaseContinuous } from "../base";
import { Normal } from "./normal";

export class LogNormal extends BaseContinuous {
  readonly name: string;
  private readonly normalDist: Normal;

  constructor(
    public readonly mu: number = 0,
    public readonly sigma: number = 1,
  ) {
    super();
    if (sigma <= 0) throw new Error("sigma must be positive");
    this.name = `LogNormal(${mu}, ${sigma})`;
    this.normalDist = new Normal(mu, sigma);
  }

  mean(): number {
    return Math.exp(this.mu + this.sigma ** 2 / 2);
  }

  variance(): number {
    const s2 = this.sigma ** 2;
    return (Math.exp(s2) - 1) * Math.exp(2 * this.mu + s2);
  }

  pdf(x: number): number {
    if (x <= 0) return 0;
    const logX = Math.log(x);
    const z = (logX - this.mu) / this.sigma;
    return Math.exp(-0.5 * z * z) / (x * this.sigma * Math.sqrt(2 * Math.PI));
  }

  cdf(x: number): number {
    if (x <= 0) return 0;
    return this.normalDist.cdf(Math.log(x));
  }

  quantile(p: number): number {
    if (p < 0 || p > 1) throw new Error("p must be in [0, 1]");
    if (p === 0) return 0;
    if (p === 1) return Infinity;
    return Math.exp(this.normalDist.quantile(p));
  }

  sample(): number {
    return Math.exp(this.normalDist.sample());
  }
}
