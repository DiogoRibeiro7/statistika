export { mean, median, variance, stdDev, skewness, kurtosis, percentile, describe } from "./descriptive";
export {
  gammaLn,
  gamma,
  logFactorial,
  factorial,
  binomialCoeff,
  betaFn,
  erf,
  erfc,
  regularizedGammaP,
  regularizedBeta,
  quantileBisect,
} from "./math";
export {
  solveLinearSystem,
  invertMatrix,
  transpose,
  matMul,
  normalCdf,
  normalQuantile,
  createRng,
  randomSample,
} from "./linalg";
export type { Matrix } from "./linalg";
