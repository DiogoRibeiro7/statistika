import { Dataset } from "./types";
import { mean, variance } from "./utils/descriptive";
import { solveLinearSystem, invertMatrix, normalCdf } from "./utils/linalg";

/**
 * Regression summary table with coefficient statistics.
 */
export interface RegressionSummary {
  coefficients: CoefficientRow[];
  rSquared: number;
  adjustedRSquared: number;
  fStatistic: number;
  fPValue: number;
  residualStdError: number;
  n: number;
  p: number;
}

export interface CoefficientRow {
  name: string;
  estimate: number;
  standardError: number;
  tStatistic: number;
  pValue: number;
}

/**
 * Compute a full regression summary with coefficient statistics.
 *
 * Similar to R's summary(lm(...)) -- provides coefficient table with
 * standard errors, t-statistics, and p-values plus model-level stats.
 *
 * @param X - Design matrix (n x p), without intercept column
 * @param y - Response variable
 * @param featureNames - Optional names for features
 * @returns A {@link RegressionSummary} containing coefficients, R-squared, F-statistic, and more
 * @throws If X has inconsistent row lengths, contains NaN/Infinity values, or n <= number of columns
 *
 * @example
 * ```ts
 * const X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]];
 * const y = [2.1, 4.0, 5.9, 8.1, 10.0];
 * const summary = regressionSummary(X, y, ["height", "weight"]);
 * console.log(summary.rSquared);       // close to 1
 * console.log(summary.coefficients);   // intercept + feature rows
 * ```
 */
export function regressionSummary(
  X: number[][],
  y: Dataset,
  featureNames?: string[],
): RegressionSummary {
  const n = X.length;
  if (n === 0) throw new Error("X must not be empty");
  const p = X[0].length;
  const cols = p + 1;

  // Validate consistent row lengths
  for (let i = 0; i < n; i++) {
    if (X[i].length !== p) {
      throw new Error(
        `Inconsistent row length at row ${i}: expected ${p} columns but got ${X[i].length}`,
      );
    }
  }

  if (n <= cols) {
    throw new Error(
      `Need more observations than parameters: n=${n} must be > cols=${cols}`,
    );
  }

  if (n !== y.length) {
    throw new Error("X and y must have the same number of rows");
  }

  // NaN / Infinity guards
  for (let i = 0; i < n; i++) {
    if (!Number.isFinite(y[i])) {
      throw new Error(`y[${i}] is not finite`);
    }
    for (let j = 0; j < p; j++) {
      if (!Number.isFinite(X[i][j])) {
        throw new Error(`X[${i}][${j}] is not finite`);
      }
    }
  }

  const names = featureNames ?? Array.from({ length: p }, (_, i) => `x${i + 1}`);

  // Build design matrix with intercept
  const designMatrix = X.map((row) => [1, ...row]);

  // Compute X^T X and X^T y
  const XtX = Array.from({ length: cols }, () => new Array<number>(cols).fill(0));
  const Xty = new Array<number>(cols).fill(0);

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < cols; j++) {
      Xty[j] += designMatrix[i][j] * y[i];
      for (let k = 0; k < cols; k++) {
        XtX[j][k] += designMatrix[i][j] * designMatrix[i][k];
      }
    }
  }

  const beta = solveLinearSystem(XtX, Xty);

  // Compute residuals
  const residuals = new Array<number>(n);
  const yMean = mean(y);
  let ssRes = 0;
  let ssTot = 0;
  for (let i = 0; i < n; i++) {
    let predicted = 0;
    for (let j = 0; j < cols; j++) {
      predicted += designMatrix[i][j] * beta[j];
    }
    residuals[i] = y[i] - predicted;
    ssRes += residuals[i] ** 2;
    ssTot += (y[i] - yMean) ** 2;
  }

  const rSquared = 1 - ssRes / ssTot;
  const adjustedRSquared = 1 - ((1 - rSquared) * (n - 1)) / (n - cols);
  const residualStdError = Math.sqrt(ssRes / (n - cols));

  // Invert X^T X for standard errors (uses LAPACK when available)
  const XtXInv = invertMatrix(XtX);
  if (XtXInv === null) {
    throw new Error("X'X matrix is singular — features may be linearly dependent");
  }

  const se = new Array<number>(cols);
  for (let j = 0; j < cols; j++) {
    se[j] = residualStdError * Math.sqrt(XtXInv[j][j]);
  }

  const coefficients: CoefficientRow[] = [];

  // Intercept
  const tInt = beta[0] / se[0];
  coefficients.push({
    name: "(Intercept)",
    estimate: beta[0],
    standardError: se[0],
    tStatistic: tInt,
    pValue: 2 * (1 - normalCdf(Math.abs(tInt))),
  });

  // Features
  for (let j = 0; j < p; j++) {
    const t = beta[j + 1] / se[j + 1];
    coefficients.push({
      name: names[j],
      estimate: beta[j + 1],
      standardError: se[j + 1],
      tStatistic: t,
      pValue: 2 * (1 - normalCdf(Math.abs(t))),
    });
  }

  // F-statistic
  const ssReg = ssTot - ssRes;
  const fStatistic = (ssReg / p) / (ssRes / (n - cols));
  // Approximate F p-value using normal for simplicity
  const fPValue = fStatistic > 0 ? Math.exp(-0.5 * fStatistic) : 1; // rough approximation

  return {
    coefficients,
    rSquared,
    adjustedRSquared,
    fStatistic,
    fPValue,
    residualStdError,
    n,
    p,
  };
}

/**
 * Residual diagnostics for a regression model.
 */
export interface ResidualDiagnostics {
  residuals: number[];
  standardizedResiduals: number[];
  durbinWatson: number;
  jarqueBera: { statistic: number; normalityLikely: boolean };
}

/**
 * Compute residual diagnostics.
 *
 * @param observed - Observed values
 * @param predicted - Predicted/fitted values
 * @returns A {@link ResidualDiagnostics} object with residuals, standardized residuals,
 *   Durbin-Watson statistic, and Jarque-Bera normality test
 * @throws If observed and predicted have different lengths or fewer than 3 observations
 */
export function residualDiagnostics(
  observed: Dataset,
  predicted: Dataset,
): ResidualDiagnostics {
  if (observed.length !== predicted.length) {
    throw new Error("Observed and predicted must have the same length");
  }
  const n = observed.length;
  if (n < 3) throw new Error("Need at least 3 observations");

  // NaN / Infinity guards
  for (let i = 0; i < n; i++) {
    if (!Number.isFinite(observed[i])) {
      throw new Error(`observed[${i}] is not finite`);
    }
    if (!Number.isFinite(predicted[i])) {
      throw new Error(`predicted[${i}] is not finite`);
    }
  }

  const residuals = observed.map((y, i) => y - predicted[i]);
  const resMean = mean(residuals);
  const resStd = Math.sqrt(variance(residuals));

  const standardizedResiduals = resStd > 0
    ? residuals.map((r) => (r - resMean) / resStd)
    : residuals.map(() => 0);

  // Durbin-Watson statistic
  let dwNum = 0;
  let dwDen = 0;
  for (let i = 0; i < n; i++) {
    dwDen += residuals[i] ** 2;
    if (i > 0) {
      dwNum += (residuals[i] - residuals[i - 1]) ** 2;
    }
  }
  const durbinWatson = dwDen > 0 ? dwNum / dwDen : 2;

  // Simple normality check via skewness/kurtosis of standardized residuals
  let m3 = 0;
  let m4 = 0;
  for (const r of standardizedResiduals) {
    m3 += r ** 3;
    m4 += r ** 4;
  }
  m3 /= n;
  m4 /= n;
  // Jarque-Bera statistic
  const jb = (n / 6) * (m3 ** 2 + (m4 - 3) ** 2 / 4);

  return {
    residuals,
    standardizedResiduals,
    durbinWatson,
    jarqueBera: {
      statistic: jb,
      normalityLikely: jb < 5.99, // chi-squared(2) at 0.05
    },
  };
}

/**
 * Variance Inflation Factor (VIF) for multicollinearity detection.
 *
 * VIF > 5 suggests moderate multicollinearity, VIF > 10 is severe.
 *
 * @param X - Design matrix (n x p), without intercept
 * @returns An array of VIF values, one per feature
 * @throws If fewer than 2 features or fewer observations than features
 */
export function vif(X: number[][]): number[] {
  const n = X.length;
  if (n === 0) throw new Error("X must not be empty");
  const p = X[0].length;
  if (p < 2) throw new Error("Need at least 2 features for VIF");
  if (n <= p) throw new Error("Need more observations than features");

  // NaN / Infinity guards
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < p; j++) {
      if (!Number.isFinite(X[i][j])) {
        throw new Error(`X[${i}][${j}] is not finite`);
      }
    }
  }

  const vifs = new Array<number>(p);

  for (let j = 0; j < p; j++) {
    // Regress x_j on all other features
    const y = X.map((row) => row[j]);
    const otherX = X.map((row) => row.filter((_, k) => k !== j));

    // Simple OLS: compute R-squared of x_j ~ other features
    const otherP = p - 1;
    const cols = otherP + 1;
    const XtX = Array.from({ length: cols }, () => new Array<number>(cols).fill(0));
    const Xty = new Array<number>(cols).fill(0);

    for (let i = 0; i < n; i++) {
      const row = [1, ...otherX[i]];
      for (let a = 0; a < cols; a++) {
        Xty[a] += row[a] * y[i];
        for (let b = 0; b < cols; b++) {
          XtX[a][b] += row[a] * row[b];
        }
      }
    }

    const beta = solveLinearSystem(XtX, Xty);
    const yMean = mean(y);
    let ssTot = 0;
    let ssRes = 0;
    for (let i = 0; i < n; i++) {
      let predicted = beta[0];
      for (let k = 0; k < otherP; k++) {
        predicted += beta[k + 1] * otherX[i][k];
      }
      ssRes += (y[i] - predicted) ** 2;
      ssTot += (y[i] - yMean) ** 2;
    }

    const r2 = 1 - ssRes / ssTot;
    vifs[j] = 1 / (1 - r2);
  }

  return vifs;
}
