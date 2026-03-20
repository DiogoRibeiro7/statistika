import { Dataset, ConfidenceInterval, RegressionCoefficientCI } from "./types";
import { mean, variance } from "./utils/descriptive";
import { Normal } from "./distributions/continuous/normal";
import { StudentT } from "./distributions/continuous/student-t";

/**
 * Confidence interval for a population mean using the t-distribution.
 *
 * @param data - Sample data
 * @param confidence - Confidence level (default 0.95)
 */
export function meanCI(data: Dataset, confidence = 0.95): ConfidenceInterval {
  if (data.length < 2) {
    throw new Error("Dataset must have at least 2 elements");
  }
  validateConfidence(confidence);

  const n = data.length;
  const xBar = mean(data);
  const se = Math.sqrt(variance(data) / n);
  const t = new StudentT(n - 1);
  const alpha = 1 - confidence;
  const tCrit = t.quantile(1 - alpha / 2);
  const moe = tCrit * se;

  return {
    estimate: xBar,
    lower: xBar - moe,
    upper: xBar + moe,
    confidenceLevel: confidence,
    marginOfError: moe,
  };
}

/**
 * Confidence interval for the difference of two independent means
 * using Welch's approximation for unequal variances.
 *
 * @param data1 - First sample
 * @param data2 - Second sample
 * @param confidence - Confidence level (default 0.95)
 */
export function twoSampleMeanCI(
  data1: Dataset,
  data2: Dataset,
  confidence = 0.95,
): ConfidenceInterval {
  if (data1.length < 2 || data2.length < 2) {
    throw new Error("Both datasets must have at least 2 elements");
  }
  validateConfidence(confidence);

  const n1 = data1.length;
  const n2 = data2.length;
  const m1 = mean(data1);
  const m2 = mean(data2);
  const v1 = variance(data1);
  const v2 = variance(data2);

  const se = Math.sqrt(v1 / n1 + v2 / n2);

  // Welch-Satterthwaite degrees of freedom
  const num = (v1 / n1 + v2 / n2) ** 2;
  const denom =
    (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1);
  const df = num / denom;

  const alpha = 1 - confidence;
  const t = new StudentT(df);
  const tCrit = t.quantile(1 - alpha / 2);
  const diff = m1 - m2;
  const moe = tCrit * se;

  return {
    estimate: diff,
    lower: diff - moe,
    upper: diff + moe,
    confidenceLevel: confidence,
    marginOfError: moe,
  };
}

/**
 * Confidence interval for the difference of two paired means.
 *
 * @param data1 - First sample
 * @param data2 - Second sample (paired with data1)
 * @param confidence - Confidence level (default 0.95)
 */
export function pairedMeanCI(
  data1: Dataset,
  data2: Dataset,
  confidence = 0.95,
): ConfidenceInterval {
  if (data1.length !== data2.length) {
    throw new Error("Paired datasets must have the same length");
  }
  const diffs = data1.map((v, i) => v - data2[i]);
  return meanCI(diffs, confidence);
}

/**
 * Wilson score confidence interval for a single proportion.
 * More accurate than the Wald interval, especially for small samples
 * or proportions near 0 or 1.
 *
 * @param successes - Number of successes
 * @param n - Total number of trials
 * @param confidence - Confidence level (default 0.95)
 */
export function proportionCI(
  successes: number,
  n: number,
  confidence = 0.95,
): ConfidenceInterval {
  if (n < 1) {
    throw new Error("Number of trials must be at least 1");
  }
  if (successes < 0 || successes > n) {
    throw new Error("Successes must be between 0 and n");
  }
  if (!Number.isInteger(successes) || !Number.isInteger(n)) {
    throw new Error("Successes and n must be integers");
  }
  validateConfidence(confidence);

  const pHat = successes / n;
  const z = new Normal().quantile(1 - (1 - confidence) / 2);
  const z2 = z * z;

  // Wilson score interval
  const denom = 1 + z2 / n;
  const center = (pHat + z2 / (2 * n)) / denom;
  const halfWidth =
    (z / denom) * Math.sqrt(pHat * (1 - pHat) / n + z2 / (4 * n * n));

  return {
    estimate: pHat,
    lower: center - halfWidth,
    upper: center + halfWidth,
    confidenceLevel: confidence,
    marginOfError: halfWidth,
  };
}

/**
 * Confidence interval for the difference of two independent proportions
 * using the Wald method with continuity correction.
 *
 * @param successes1 - Successes in first sample
 * @param n1 - Total trials in first sample
 * @param successes2 - Successes in second sample
 * @param n2 - Total trials in second sample
 * @param confidence - Confidence level (default 0.95)
 */
export function twoProportionCI(
  successes1: number,
  n1: number,
  successes2: number,
  n2: number,
  confidence = 0.95,
): ConfidenceInterval {
  if (n1 < 1 || n2 < 1) {
    throw new Error("Number of trials must be at least 1");
  }
  if (successes1 < 0 || successes1 > n1 || successes2 < 0 || successes2 > n2) {
    throw new Error("Successes must be between 0 and n");
  }
  validateConfidence(confidence);

  const p1 = successes1 / n1;
  const p2 = successes2 / n2;
  const diff = p1 - p2;

  const se = Math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2);
  const z = new Normal().quantile(1 - (1 - confidence) / 2);
  const moe = z * se;

  return {
    estimate: diff,
    lower: diff - moe,
    upper: diff + moe,
    confidenceLevel: confidence,
    marginOfError: moe,
  };
}

/**
 * Confidence intervals for linear regression coefficients (slope and intercept).
 *
 * @param x - Predictor values
 * @param y - Response values
 * @param confidence - Confidence level (default 0.95)
 */
export function linearRegressionCI(
  x: Dataset,
  y: Dataset,
  confidence = 0.95,
): RegressionCoefficientCI[] {
  if (x.length !== y.length) {
    throw new Error("x and y datasets must have the same length");
  }
  if (x.length < 3) {
    throw new Error("Need at least 3 data points for regression CIs");
  }
  validateConfidence(confidence);

  const n = x.length;
  const xBar = mean(x);
  const yBar = mean(y);

  let ssXX = 0;
  let ssXY = 0;
  for (let i = 0; i < n; i++) {
    ssXX += (x[i] - xBar) ** 2;
    ssXY += (x[i] - xBar) * (y[i] - yBar);
  }

  const slope = ssXY / ssXX;
  const intercept = yBar - slope * xBar;

  // Residual standard error
  let ssRes = 0;
  for (let i = 0; i < n; i++) {
    const residual = y[i] - (slope * x[i] + intercept);
    ssRes += residual * residual;
  }
  const df = n - 2;
  const mse = ssRes / df;
  const rse = Math.sqrt(mse);

  const seSlope = rse / Math.sqrt(ssXX);
  const seIntercept = rse * Math.sqrt(1 / n + (xBar * xBar) / ssXX);

  const alpha = 1 - confidence;
  const t = new StudentT(df);
  const tCrit = t.quantile(1 - alpha / 2);

  // p-values for two-sided test (H0: coeff = 0)
  const tSlope = slope / seSlope;
  const tIntercept = intercept / seIntercept;
  const pSlope = 2 * (1 - t.cdf(Math.abs(tSlope)));
  const pIntercept = 2 * (1 - t.cdf(Math.abs(tIntercept)));

  return [
    {
      name: "intercept",
      estimate: intercept,
      standardError: seIntercept,
      lower: intercept - tCrit * seIntercept,
      upper: intercept + tCrit * seIntercept,
      tStatistic: tIntercept,
      pValue: pIntercept,
    },
    {
      name: "slope",
      estimate: slope,
      standardError: seSlope,
      lower: slope - tCrit * seSlope,
      upper: slope + tCrit * seSlope,
      tStatistic: tSlope,
      pValue: pSlope,
    },
  ];
}

/**
 * Confidence intervals for multiple regression coefficients
 * (intercept and all feature coefficients).
 *
 * @param X - Feature matrix (n observations x p features)
 * @param y - Response values (n observations)
 * @param confidence - Confidence level (default 0.95)
 * @param featureNames - Optional names for features
 */
export function multipleRegressionCI(
  X: number[][],
  y: Dataset,
  confidence = 0.95,
  featureNames?: string[],
): RegressionCoefficientCI[] {
  const n = X.length;
  if (n !== y.length) {
    throw new Error("X and y must have the same number of observations");
  }
  const p = X[0].length;
  if (n <= p + 1) {
    throw new Error(
      "Number of observations must exceed number of parameters (p + 1)",
    );
  }
  validateConfidence(confidence);

  const cols = p + 1; // including intercept

  // Build design matrix with intercept column
  const designRows: number[][] = X.map((row) => [1, ...row]);

  // Compute X^T X
  const XtX: number[][] = Array.from({ length: cols }, () =>
    new Array(cols).fill(0),
  );
  const Xty: number[] = new Array(cols).fill(0);

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < cols; j++) {
      Xty[j] += designRows[i][j] * y[i];
      for (let k = 0; k < cols; k++) {
        XtX[j][k] += designRows[i][j] * designRows[i][k];
      }
    }
  }

  // Solve for beta: (X^T X) beta = X^T y
  const beta = solveLinearSystem(
    XtX.map((row) => [...row]),
    [...Xty],
  );

  // Compute residual variance
  let ssRes = 0;
  for (let i = 0; i < n; i++) {
    let predicted = 0;
    for (let j = 0; j < cols; j++) {
      predicted += designRows[i][j] * beta[j];
    }
    ssRes += (y[i] - predicted) ** 2;
  }
  const df = n - cols;
  const mse = ssRes / df;

  // Invert X^T X using Gauss-Jordan elimination
  const inv = invertMatrix(XtX);

  const alpha = 1 - confidence;
  const t = new StudentT(df);
  const tCrit = t.quantile(1 - alpha / 2);

  const names = ["intercept"];
  for (let j = 0; j < p; j++) {
    names.push(featureNames?.[j] ?? `x${j + 1}`);
  }

  const results: RegressionCoefficientCI[] = [];
  for (let j = 0; j < cols; j++) {
    const se = Math.sqrt(mse * inv[j][j]);
    const tStat = beta[j] / se;
    const pVal = 2 * (1 - t.cdf(Math.abs(tStat)));

    results.push({
      name: names[j],
      estimate: beta[j],
      standardError: se,
      lower: beta[j] - tCrit * se,
      upper: beta[j] + tCrit * se,
      tStatistic: tStat,
      pValue: pVal,
    });
  }

  return results;
}

// ---- Internal helpers ----

function validateConfidence(confidence: number): void {
  if (confidence <= 0 || confidence >= 1) {
    throw new Error("Confidence level must be between 0 and 1 (exclusive)");
  }
}

/** Solve Ax = b via Gaussian elimination with partial pivoting. */
function solveLinearSystem(A: number[][], b: number[]): number[] {
  const n = A.length;
  const aug: number[][] = A.map((row, i) => [...row, b[i]]);

  for (let col = 0; col < n; col++) {
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(aug[row][col]) > Math.abs(aug[maxRow][col])) {
        maxRow = row;
      }
    }
    [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];

    if (Math.abs(aug[col][col]) < 1e-12) {
      throw new Error("Singular matrix: features may be linearly dependent");
    }

    for (let row = col + 1; row < n; row++) {
      const factor = aug[row][col] / aug[col][col];
      for (let j = col; j <= n; j++) {
        aug[row][j] -= factor * aug[col][j];
      }
    }
  }

  const x = new Array(n).fill(0);
  for (let row = n - 1; row >= 0; row--) {
    x[row] = aug[row][n];
    for (let col = row + 1; col < n; col++) {
      x[row] -= aug[row][col] * x[col];
    }
    x[row] /= aug[row][row];
  }

  return x;
}

/** Invert a square matrix using Gauss-Jordan elimination. */
function invertMatrix(M: number[][]): number[][] {
  const n = M.length;
  // Augment with identity
  const aug: number[][] = M.map((row, i) => {
    const id = new Array(n).fill(0);
    id[i] = 1;
    return [...row, ...id];
  });

  for (let col = 0; col < n; col++) {
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(aug[row][col]) > Math.abs(aug[maxRow][col])) {
        maxRow = row;
      }
    }
    [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];

    const pivot = aug[col][col];
    if (Math.abs(pivot) < 1e-12) {
      throw new Error("Singular matrix cannot be inverted");
    }

    for (let j = 0; j < 2 * n; j++) {
      aug[col][j] /= pivot;
    }

    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = aug[row][col];
      for (let j = 0; j < 2 * n; j++) {
        aug[row][j] -= factor * aug[col][j];
      }
    }
  }

  return aug.map((row) => row.slice(n));
}
