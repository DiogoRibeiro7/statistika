import { BaseContinuous } from "../base";
import { RandomFn } from "../../types";

export class Cauchy extends BaseContinuous {
  readonly name: string;

  constructor(
    public readonly x0: number = 0,
    public readonly gammaParam: number = 1,
    rng?: RandomFn,
  ) {
    super(rng);
    if (gammaParam <= 0) throw new Error("gamma must be positive");
    this.name = `Cauchy(${x0}, ${gammaParam})`;
  }

  mean(): number {
    return NaN; // undefined for Cauchy
  }

  variance(): number {
    return NaN; // undefined for Cauchy
  }

  pdf(x: number): number {
    const z = (x - this.x0) / this.gammaParam;
    return 1 / (Math.PI * this.gammaParam * (1 + z * z));
  }

  cdf(x: number): number {
    return 0.5 + Math.atan((x - this.x0) / this.gammaParam) / Math.PI;
  }

  quantile(p: number): number {
    if (p < 0 || p > 1) throw new Error("p must be in [0, 1]");
    if (p === 0) return -Infinity;
    if (p === 1) return Infinity;
    return this.x0 + this.gammaParam * Math.tan(Math.PI * (p - 0.5));
  }

  sample(): number {
    // Inverse CDF method
    const u = this.rng();
    return this.quantile(u);
  }
}
