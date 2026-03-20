import { Dataset, LinearRegressionResult } from "../types";
import { mean } from "../utils/descriptive";

export function linearRegression(
  x: Dataset,
  y: Dataset,
): LinearRegressionResult {
  if (x.length !== y.length) {
    throw new Error("x and y datasets must have the same length");
  }
  if (x.length < 2) {
    throw new Error("Datasets must have at least 2 data points");
  }

  const n = x.length;
  const xMean = mean(x);
  const yMean = mean(y);

  let ssXY = 0;
  let ssXX = 0;
  let ssTot = 0;

  for (let i = 0; i < n; i++) {
    ssXY += (x[i] - xMean) * (y[i] - yMean);
    ssXX += (x[i] - xMean) ** 2;
    ssTot += (y[i] - yMean) ** 2;
  }

  const slope = ssXY / ssXX;
  const intercept = yMean - slope * xMean;

  const ssRes = y.reduce((acc, yi, i) => {
    const predicted = slope * x[i] + intercept;
    return acc + (yi - predicted) ** 2;
  }, 0);

  const rSquared = 1 - ssRes / ssTot;

  return {
    slope,
    intercept,
    rSquared,
    predict: (xVal: number) => slope * xVal + intercept,
  };
}
