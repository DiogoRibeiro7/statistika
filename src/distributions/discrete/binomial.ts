import { BaseDiscrete } from "../base";
import { logFactorial, regularizedBeta } from "../../utils/math";
import { RandomFn } from "../../types";

export class Binomial extends BaseDiscrete {
  readonly name: string;

  constructor(
    public readonly n: number = 1,
    public readonly p: number = 0.5,
    rng?: RandomFn,
  ) {
    super(rng);
    if (n < 1 || !Number.isInteger(n)) throw new Error("n must be a positive integer");
    if (p < 0 || p > 1) throw new Error("p must be in [0, 1]");
    this.name = `Binomial(${n}, ${p})`;
  }

  mean(): number {
    return this.n * this.p;
  }

  variance(): number {
    return this.n * this.p * (1 - this.p);
  }

  pmf(k: number): number {
    if (!Number.isInteger(k) || k < 0 || k > this.n) return 0;
    if (this.p === 0) return k === 0 ? 1 : 0;
    if (this.p === 1) return k === this.n ? 1 : 0;
    const logPmf =
      logFactorial(this.n) -
      logFactorial(k) -
      logFactorial(this.n - k) +
      k * Math.log(this.p) +
      (this.n - k) * Math.log(1 - this.p);
    return Math.exp(logPmf);
  }

  cdf(k: number): number {
    if (k < 0) return 0;
    if (k >= this.n) return 1;
    const kFloor = Math.floor(k);
    // CDF via regularized incomplete beta: P(X <= k) = I_{1-p}(n-k, k+1)
    return regularizedBeta(1 - this.p, this.n - kFloor, kFloor + 1);
  }

  quantile(prob: number): number {
    if (prob < 0 || prob > 1) throw new Error("p must be in [0, 1]");
    if (prob === 0) return 0;
    if (prob === 1) return this.n;
    // Linear search (fine for moderate n)
    let cumulative = 0;
    for (let k = 0; k <= this.n; k++) {
      cumulative += this.pmf(k);
      if (cumulative >= prob) return k;
    }
    return this.n;
  }

  sample(): number {
    let successes = 0;
    for (let i = 0; i < this.n; i++) {
      if (this.rng() < this.p) successes++;
    }
    return successes;
  }
}
