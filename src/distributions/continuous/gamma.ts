import { BaseContinuous } from "../base";
import { gammaLn, regularizedGammaP, quantileBisect } from "../../utils/math";
import { RandomFn } from "../../types";

export class GammaDistribution extends BaseContinuous {
  readonly name: string;

  constructor(
    public readonly shape: number = 1,
    public readonly rate: number = 1,
    rng?: RandomFn,
  ) {
    super(rng);
    if (shape <= 0) throw new Error("shape must be positive");
    if (rate <= 0) throw new Error("rate must be positive");
    this.name = `Gamma(${shape}, ${rate})`;
  }

  mean(): number {
    return this.shape / this.rate;
  }

  variance(): number {
    return this.shape / this.rate ** 2;
  }

  pdf(x: number): number {
    if (x < 0) return 0;
    if (x === 0) {
      if (this.shape === 1) return this.rate;
      if (this.shape < 1) return Infinity;
      return 0;
    }
    const logPdf =
      this.shape * Math.log(this.rate) +
      (this.shape - 1) * Math.log(x) -
      this.rate * x -
      gammaLn(this.shape);
    return Math.exp(logPdf);
  }

  cdf(x: number): number {
    if (x <= 0) return 0;
    return regularizedGammaP(this.shape, this.rate * x);
  }

  quantile(p: number): number {
    if (p < 0 || p > 1) throw new Error("p must be in [0, 1]");
    if (p === 0) return 0;
    if (p === 1) return Infinity;
    // Initial bracket
    const m = this.mean();
    const sd = this.stdDev();
    const upper = m + 10 * sd;
    return quantileBisect((x) => this.cdf(x), p, 0, upper);
  }

  sample(): number {
    // Marsaglia-Tsang method for shape >= 1, shift for shape < 1
    if (this.shape < 1) {
      const g = new GammaDistribution(this.shape + 1, 1, this.rng).sample();
      return (g * Math.pow(this.rng(), 1 / this.shape)) / this.rate;
    }
    const d = this.shape - 1 / 3;
    const c = 1 / Math.sqrt(9 * d);
    while (true) {
      let x: number, v: number;
      do {
        x = standardNormal(this.rng);
        v = 1 + c * x;
      } while (v <= 0);
      v = v * v * v;
      const u = this.rng();
      if (
        u < 1 - 0.0331 * (x * x) * (x * x) ||
        Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))
      ) {
        return (d * v) / this.rate;
      }
    }
  }
}

function standardNormal(rng: RandomFn): number {
  const u1 = rng();
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}
