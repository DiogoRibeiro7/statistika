import { BaseContinuous } from "../base";
import { gammaLn, regularizedBeta, quantileBisect } from "../../utils/math";
import { GammaDistribution } from "./gamma";
import { RandomFn } from "../../types";

export class StudentT extends BaseContinuous {
  readonly name: string;

  constructor(public readonly nu: number = 1, rng?: RandomFn) {
    super(rng);
    if (nu <= 0) throw new Error("nu (degrees of freedom) must be positive");
    this.name = `StudentT(${nu})`;
  }

  mean(): number {
    if (this.nu <= 1) return NaN;
    return 0;
  }

  variance(): number {
    if (this.nu <= 2) return this.nu > 1 ? Infinity : NaN;
    return this.nu / (this.nu - 2);
  }

  pdf(x: number): number {
    const logPdf =
      gammaLn((this.nu + 1) / 2) -
      gammaLn(this.nu / 2) -
      0.5 * Math.log(this.nu * Math.PI) -
      ((this.nu + 1) / 2) * Math.log(1 + (x * x) / this.nu);
    return Math.exp(logPdf);
  }

  cdf(x: number): number {
    const t2 = x * x;
    const xt = this.nu / (this.nu + t2);
    const ib = 0.5 * regularizedBeta(xt, this.nu / 2, 0.5);
    return x >= 0 ? 1 - ib : ib;
  }

  quantile(p: number): number {
    if (p < 0 || p > 1) throw new Error("p must be in [0, 1]");
    if (p === 0) return -Infinity;
    if (p === 1) return Infinity;
    // Use symmetry
    if (p < 0.5) return -this.quantile(1 - p);
    const upper = 10 * Math.sqrt(this.variance() || 100);
    return quantileBisect((x) => this.cdf(x), p, 0, upper);
  }

  sample(): number {
    // Ratio of standard normal to sqrt(chi-squared / nu)
    const u1 = this.rng();
    const u2 = this.rng();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    // Chi-squared with nu degrees of freedom via sum of squared normals
    // For efficiency, use gamma sampling
    const chi2 = new GammaDistribution(this.nu / 2, 0.5, this.rng).sample();
    return z / Math.sqrt(chi2 / this.nu);
  }
}
