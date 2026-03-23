import { BaseDiscrete } from "../base";
import { logFactorial } from "../../utils/math";
import { RandomFn } from "../../types";

export class Hypergeometric extends BaseDiscrete {
  readonly name: string;

  constructor(
    public readonly N: number,
    public readonly K: number,
    public readonly n: number,
    rng?: RandomFn,
  ) {
    super(rng);
    if (N < 0 || !Number.isInteger(N)) throw new Error("N must be a non-negative integer");
    if (K < 0 || K > N || !Number.isInteger(K)) throw new Error("K must be an integer in [0, N]");
    if (n < 0 || n > N || !Number.isInteger(n)) throw new Error("n must be an integer in [0, N]");
    this.name = `Hypergeometric(${N}, ${K}, ${n})`;
  }

  mean(): number {
    return this.n * this.K / this.N;
  }

  variance(): number {
    const { N, K, n } = this;
    return (n * K * (N - K) * (N - n)) / (N * N * (N - 1));
  }

  pmf(k: number): number {
    if (!Number.isInteger(k)) return 0;
    const { N, K, n } = this;
    const lo = Math.max(0, n + K - N);
    const hi = Math.min(n, K);
    if (k < lo || k > hi) return 0;
    // C(K,k) * C(N-K, n-k) / C(N, n)
    const logPmf =
      logFactorial(K) - logFactorial(k) - logFactorial(K - k) +
      logFactorial(N - K) - logFactorial(n - k) - logFactorial(N - K - n + k) -
      logFactorial(N) + logFactorial(n) + logFactorial(N - n);
    return Math.exp(logPmf);
  }

  cdf(k: number): number {
    const { N, K, n } = this;
    const lo = Math.max(0, n + K - N);
    if (k < lo) return 0;
    const hi = Math.min(n, K);
    if (k >= hi) return 1;
    const kFloor = Math.floor(k);
    let sum = 0;
    for (let i = lo; i <= kFloor; i++) {
      sum += this.pmf(i);
    }
    return sum;
  }

  quantile(prob: number): number {
    if (prob < 0 || prob > 1) throw new Error("p must be in [0, 1]");
    const { N, K, n } = this;
    const lo = Math.max(0, n + K - N);
    const hi = Math.min(n, K);
    if (prob === 0) return lo;
    if (prob === 1) return hi;
    let cumulative = 0;
    for (let k = lo; k <= hi; k++) {
      cumulative += this.pmf(k);
      if (cumulative >= prob) return k;
    }
    return hi;
  }

  sample(): number {
    // Direct simulation: draw n balls from urn of N (K success, N-K failure)
    const { N, K, n } = this;
    let successes = 0;
    let remaining = N;
    let kRemaining = K;
    for (let i = 0; i < n; i++) {
      if (this.rng() < kRemaining / remaining) {
        successes++;
        kRemaining--;
      }
      remaining--;
    }
    return successes;
  }
}
