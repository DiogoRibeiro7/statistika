import { BaseDiscrete } from "../base";
import { logFactorial, regularizedBeta } from "../../utils/math";

export class NegativeBinomial extends BaseDiscrete {
  readonly name: string;

  constructor(
    public readonly r: number,
    public readonly p: number,
  ) {
    super();
    if (r <= 0 || !Number.isInteger(r)) throw new Error("r must be a positive integer");
    if (p <= 0 || p > 1) throw new Error("p must be in (0, 1]");
    this.name = `NegBin(${r}, ${p})`;
  }

  mean(): number {
    return this.r * (1 - this.p) / this.p;
  }

  variance(): number {
    return this.r * (1 - this.p) / (this.p * this.p);
  }

  pmf(k: number): number {
    if (!Number.isInteger(k) || k < 0) return 0;
    const { r, p } = this;
    // C(k+r-1, k) * p^r * (1-p)^k
    const logPmf =
      logFactorial(k + r - 1) -
      logFactorial(k) -
      logFactorial(r - 1) +
      r * Math.log(p) +
      k * Math.log(1 - p);
    return Math.exp(logPmf);
  }

  cdf(k: number): number {
    if (k < 0) return 0;
    const kFloor = Math.floor(k);
    // CDF = I_p(r, k+1)
    return regularizedBeta(this.p, this.r, kFloor + 1);
  }

  quantile(prob: number): number {
    if (prob < 0 || prob > 1) throw new Error("p must be in [0, 1]");
    if (prob === 0) return 0;
    if (prob === 1) return Infinity;
    let cumulative = 0;
    for (let k = 0; ; k++) {
      cumulative += this.pmf(k);
      if (cumulative >= prob) return k;
      if (k > 10000) return k; // safety limit
    }
  }

  sample(): number {
    // Sum of r geometric samples
    let total = 0;
    for (let i = 0; i < this.r; i++) {
      let k = 0;
      while (Math.random() >= this.p) k++;
      total += k;
    }
    return total;
  }
}
