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

/** Result of a multiple linear regression fit. */
export interface MultipleRegressionResult {
  coefficients: number[];
  intercept: number;
  rSquared: number;
  predict: (x: number[]) => number;
}

/** Result of a polynomial regression fit. */
export interface PolynomialRegressionResult {
  coefficients: number[];
  degree: number;
  rSquared: number;
  predict: (x: number) => number;
}

/** Result of a logistic regression fit. */
export interface LogisticRegressionResult {
  coefficients: number[];
  intercept: number;
  iterations: number;
  predict: (x: number[]) => number;
}

/** Result of a correlation computation. */
export interface CorrelationResult {
  coefficient: number;
  pValue: number;
}

/** Result of a hypothesis test. */
export interface HypothesisTestResult {
  statistic: number;
  pValue: number;
  degreesOfFreedom: number;
  rejected: boolean;
}

/** Result of an ANOVA test. */
export interface AnovaResult {
  fStatistic: number;
  pValue: number;
  dfBetween: number;
  dfWithin: number;
  ssBetween: number;
  ssWithin: number;
  msBetween: number;
  msWithin: number;
  rejected: boolean;
}

/** Result of a Kolmogorov-Smirnov test. */
export interface KSTestResult {
  statistic: number;
  pValue: number;
  rejected: boolean;
}

/** Result of a confidence interval computation. */
export interface ConfidenceInterval {
  estimate: number;
  lower: number;
  upper: number;
  confidenceLevel: number;
  marginOfError: number;
}

/** Result of a regression coefficient confidence interval computation. */
export interface RegressionCoefficientCI {
  name: string;
  estimate: number;
  standardError: number;
  lower: number;
  upper: number;
  tStatistic: number;
  pValue: number;
}
