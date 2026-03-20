import { BaseContinuous } from "../base";
import { betaFn, regularizedBeta, quantileBisect } from "../../utils/math";

export class FDistribution extends BaseContinuous {
  readonly name: string;

  constructor(
    public readonly d1: number,
    public readonly d2: number,
  ) {
    super();
    if (d1 <= 0 || !Number.isInteger(d1)) throw new Error("d1 must be a positive integer");
    if (d2 <= 0 || !Number.isInteger(d2)) throw new Error("d2 must be a positive integer");
    this.name = `F(${d1}, ${d2})`;
  }

  mean(): number {
    if (this.d2 <= 2) return NaN;
    return this.d2 / (this.d2 - 2);
  }

  variance(): number {
    if (this.d2 <= 4) return NaN;
    const { d1, d2 } = this;
    return (2 * d2 * d2 * (d1 + d2 - 2)) / (d1 * (d2 - 2) ** 2 * (d2 - 4));
  }

  pdf(x: number): number {
    if (x <= 0) return 0;
    const { d1, d2 } = this;
    const half1 = d1 / 2;
    const half2 = d2 / 2;
    const num = Math.pow(d1 / d2, half1) * Math.pow(x, half1 - 1);
    const den = Math.pow(1 + (d1 / d2) * x, (d1 + d2) / 2) * betaFn(half1, half2);
    return num / den;
  }

  cdf(x: number): number {
    if (x <= 0) return 0;
    const { d1, d2 } = this;
    const t = (d1 * x) / (d1 * x + d2);
    return regularizedBeta(t, d1 / 2, d2 / 2);
  }

  quantile(p: number): number {
    if (p < 0 || p > 1) throw new Error("p must be in [0, 1]");
    if (p === 0) return 0;
    if (p === 1) return Infinity;
    return quantileBisect((x) => this.cdf(x), p, 0, 1000);
  }

  sample(): number {
    // Ratio of two chi-squared samples
    const x1 = sampleChiSq(this.d1);
    const x2 = sampleChiSq(this.d2);
    return (x1 / this.d1) / (x2 / this.d2);
  }
}

function sampleChiSq(k: number): number {
  let sum = 0;
  for (let i = 0; i < k; i++) {
    const u1 = Math.random();
    const u2 = Math.random();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    sum += z * z;
  }
  return sum;
}
