/** A numeric dataset represented as an array of numbers. */
export type Dataset = number[];

/** Descriptive statistics summary. */
export interface DescriptiveStats {
  count: number;
  mean: number;
  median: number;
  variance: number;
  stdDev: number;
  min: number;
  max: number;
}

/** Common interface for all probability distributions. */
export interface Distribution {
  readonly name: string;
  mean(): number;
  variance(): number;
  stdDev(): number;
  sample(): number;
  sampleN(n: number): number[];
}

/** A continuous probability distribution. */
export interface ContinuousDistribution extends Distribution {
  pdf(x: number): number;
  cdf(x: number): number;
  quantile(p: number): number;
  sf(x: number): number;
}

/** A discrete probability distribution. */
export interface DiscreteDistribution extends Distribution {
  pmf(k: number): number;
  cdf(k: number): number;
  quantile(p: number): number;
  sf(k: number): number;
}

/** Result of a linear regression fit. */
export interface LinearRegressionResult {
  slope: number;
  intercept: number;
  rSquared: number;
  predict: (x: number) => number;
}
