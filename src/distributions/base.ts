import { ContinuousDistribution, DiscreteDistribution } from "../types";

export abstract class BaseContinuous implements ContinuousDistribution {
  abstract readonly name: string;
  abstract mean(): number;
  abstract variance(): number;
  abstract pdf(x: number): number;
  abstract cdf(x: number): number;
  abstract quantile(p: number): number;
  abstract sample(): number;

  stdDev(): number {
    return Math.sqrt(this.variance());
  }

  sf(x: number): number {
    return 1 - this.cdf(x);
  }

  sampleN(n: number): number[] {
    const result: number[] = new Array(n);
    for (let i = 0; i < n; i++) result[i] = this.sample();
    return result;
  }
}

export abstract class BaseDiscrete implements DiscreteDistribution {
  abstract readonly name: string;
  abstract mean(): number;
  abstract variance(): number;
  abstract pmf(k: number): number;
  abstract cdf(k: number): number;
  abstract quantile(p: number): number;
  abstract sample(): number;

  stdDev(): number {
    return Math.sqrt(this.variance());
  }

  sf(k: number): number {
    return 1 - this.cdf(k);
  }

  sampleN(n: number): number[] {
    const result: number[] = new Array(n);
    for (let i = 0; i < n; i++) result[i] = this.sample();
    return result;
  }
}
