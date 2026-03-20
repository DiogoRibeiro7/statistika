import { BaseDiscrete } from "../base";
import { logFactorial, regularizedGammaP } from "../../utils/math";

export class Poisson extends BaseDiscrete {
  readonly name: string;

  constructor(public readonly lambda: number = 1) {
    super();
    if (lambda <= 0) throw new Error("lambda must be positive");
    this.name = `Poisson(${lambda})`;
  }

  mean(): number {
    return this.lambda;
  }

  variance(): number {
    return this.lambda;
  }

  pmf(k: number): number {
    if (!Number.isInteger(k) || k < 0) return 0;
    const logPmf = k * Math.log(this.lambda) - this.lambda - logFactorial(k);
    return Math.exp(logPmf);
  }

  cdf(k: number): number {
    if (k < 0) return 0;
    const kFloor = Math.floor(k);
    // CDF via regularized upper incomplete gamma: P(X <= k) = Q(k+1, lambda) = 1 - P(k+1, lambda)
    return 1 - regularizedGammaP(kFloor + 1, this.lambda);
  }

  quantile(prob: number): number {
    if (prob < 0 || prob > 1) throw new Error("p must be in [0, 1]");
    if (prob === 0) return 0;
    if (prob === 1) return Infinity;
    let cumulative = 0;
    let k = 0;
    while (cumulative < prob) {
      cumulative += this.pmf(k);
      if (cumulative >= prob) return k;
      k++;
      if (k > this.lambda + 40 * Math.sqrt(this.lambda)) return k;
    }
    return k;
  }

  sample(): number {
    // Knuth's algorithm for small lambda, otherwise use transformed rejection
    if (this.lambda < 30) {
      const L = Math.exp(-this.lambda);
      let k = 0;
      let p = 1;
      do {
        k++;
        p *= Math.random();
      } while (p > L);
      return k - 1;
    }
    // For large lambda, use inverse transform
    return this.quantile(Math.random());
  }
}
