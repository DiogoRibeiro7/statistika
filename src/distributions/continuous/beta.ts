import { BaseContinuous } from "../base";
import { gammaLn, regularizedBeta, quantileBisect } from "../../utils/math";
import { GammaDistribution } from "./gamma";

export class BetaDistribution extends BaseContinuous {
  readonly name: string;

  constructor(
    public readonly alpha: number = 1,
    public readonly beta: number = 1,
  ) {
    super();
    if (alpha <= 0) throw new Error("alpha must be positive");
    if (beta <= 0) throw new Error("beta must be positive");
    this.name = `Beta(${alpha}, ${beta})`;
  }

  mean(): number {
    return this.alpha / (this.alpha + this.beta);
  }

  variance(): number {
    const ab = this.alpha + this.beta;
    return (this.alpha * this.beta) / (ab * ab * (ab + 1));
  }

  pdf(x: number): number {
    if (x < 0 || x > 1) return 0;
    if (x === 0) {
      if (this.alpha === 1) return this.beta;
      if (this.alpha < 1) return Infinity;
      return 0;
    }
    if (x === 1) {
      if (this.beta === 1) return this.alpha;
      if (this.beta < 1) return Infinity;
      return 0;
    }
    const logPdf =
      (this.alpha - 1) * Math.log(x) +
      (this.beta - 1) * Math.log(1 - x) -
      (gammaLn(this.alpha) + gammaLn(this.beta) - gammaLn(this.alpha + this.beta));
    return Math.exp(logPdf);
  }

  cdf(x: number): number {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    return regularizedBeta(x, this.alpha, this.beta);
  }

  quantile(p: number): number {
    if (p < 0 || p > 1) throw new Error("p must be in [0, 1]");
    if (p === 0) return 0;
    if (p === 1) return 1;
    return quantileBisect((x) => this.cdf(x), p, 0, 1);
  }

  sample(): number {
    const x = new GammaDistribution(this.alpha, 1).sample();
    const y = new GammaDistribution(this.beta, 1).sample();
    return x / (x + y);
  }
}
