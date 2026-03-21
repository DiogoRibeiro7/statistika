export { linearRegression } from "./linear-regression";
export { multipleRegression } from "./multiple-regression";
export { polynomialRegression } from "./polynomial-regression";
export { logisticRegression } from "./logistic-regression";
export {
  pearsonCorrelation,
  spearmanCorrelation,
  kendallCorrelation,
} from "./correlation";
export {
  quantileRegression,
  multipleQuantileRegression,
} from "./quantile-regression";
export type {
  QuantileRegressionResult,
  SimpleQuantileRegressionResult,
} from "./quantile-regression";
export { huberRegression, ransacRegression } from "./robust-regression";
export type {
  HuberRegressionResult,
  RANSACResult,
} from "./robust-regression";
