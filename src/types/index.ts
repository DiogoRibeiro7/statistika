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

/** Result of a linear regression fit. */
export interface LinearRegressionResult {
  slope: number;
  intercept: number;
  rSquared: number;
  predict: (x: number) => number;
}
