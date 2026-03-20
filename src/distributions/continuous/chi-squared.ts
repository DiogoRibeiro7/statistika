import { BaseContinuous } from "../base";
import { GammaDistribution } from "./gamma";

export class ChiSquared extends BaseContinuous {
  readonly name: string;
  private readonly gammaDistribution: GammaDistribution;

  constructor(public readonly k: number = 1) {
    super();
    if (k <= 0 || !Number.isInteger(k)) {
      throw new Error("k (degrees of freedom) must be a positive integer");
    }
    this.name = `ChiSquared(${k})`;
    this.gammaDistribution = new GammaDistribution(k / 2, 0.5);
  }

  mean(): number {
    return this.k;
  }

  variance(): number {
    return 2 * this.k;
  }

  pdf(x: number): number {
    return this.gammaDistribution.pdf(x);
  }

  cdf(x: number): number {
    return this.gammaDistribution.cdf(x);
  }

  quantile(p: number): number {
    return this.gammaDistribution.quantile(p);
  }

  sample(): number {
    return this.gammaDistribution.sample();
  }
}
