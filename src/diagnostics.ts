import { Dataset } from "./types";
import { mean, variance } from "./utils/descriptive";
import { Normal } from "./distributions/continuous/normal";
import { solveLinearSystem } from "./utils/linalg";

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
 * Similar to R's summary(lm(...)) — provides coefficient table with
 * standard errors, t-statistics, and p-values plus model-level stats.
 *
 * @param X - Design matrix (n x p), without intercept column
 * @param y - Response variable
 * @param featureNames - Optional names for features
 */
export function regressionSummary(
  X: number[][],
  y: Dataset,
  featureNames?: string[],
): RegressionSummary {
  const n = X.length;
  const p = X[0].length;
  const cols = p + 1;

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

  // Invert X^T X for standard errors
  const XtXInv = invertMatrix(XtX);
  const se = new Array<number>(cols);
  for (let j = 0; j < cols; j++) {
    se[j] = residualStdError * Math.sqrt(XtXInv[j][j]);
  }

  const normal = new Normal();
  const coefficients: CoefficientRow[] = [];

  // Intercept
  const tInt = beta[0] / se[0];
  coefficients.push({
    name: "(Intercept)",
    estimate: beta[0],
    standardError: se[0],
    tStatistic: tInt,
    pValue: 2 * (1 - normal.cdf(Math.abs(tInt))), // approximate using normal for large n
  });

  // Features
  for (let j = 0; j < p; j++) {
    const t = beta[j + 1] / se[j + 1];
    coefficients.push({
      name: names[j],
      estimate: beta[j + 1],
      standardError: se[j + 1],
      tStatistic: t,
      pValue: 2 * (1 - normal.cdf(Math.abs(t))),
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
  shapiroWilkApprox: { statistic: number; normalityLikely: boolean };
}

/**
 * Compute residual diagnostics.
 *
 * @param observed - Observed values
 * @param predicted - Predicted/fitted values
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
  // Jarque-Bera-like statistic
  const jb = (n / 6) * (m3 ** 2 + (m4 - 3) ** 2 / 4);

  return {
    residuals,
    standardizedResiduals,
    durbinWatson,
    shapiroWilkApprox: {
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
 */
export function vif(X: number[][]): number[] {
  const n = X.length;
  const p = X[0].length;
  if (p < 2) throw new Error("Need at least 2 features for VIF");
  if (n <= p) throw new Error("Need more observations than features");

  const vifs = new Array<number>(p);

  for (let j = 0; j < p; j++) {
    // Regress x_j on all other features
    const y = X.map((row) => row[j]);
    const otherX = X.map((row) => row.filter((_, k) => k !== j));

    // Simple OLS: compute R² of x_j ~ other features
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

// ── Helper ──────────────────────────────────────────────────────────────

function invertMatrix(matrix: number[][]): number[][] {
  const n = matrix.length;
  const aug = matrix.map((row, i) => {
    const r = [...row];
    for (let j = 0; j < n; j++) r.push(i === j ? 1 : 0);
    return r;
  });

  for (let col = 0; col < n; col++) {
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(aug[row][col]) > Math.abs(aug[maxRow][col])) maxRow = row;
    }
    [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];

    const pivot = aug[col][col];
    if (Math.abs(pivot) < 1e-12) {
      throw new Error("Matrix is singular");
    }

    for (let j = 0; j < 2 * n; j++) aug[col][j] /= pivot;
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = aug[row][col];
      for (let j = 0; j < 2 * n; j++) aug[row][j] -= factor * aug[col][j];
    }
  }

  return aug.map((row) => row.slice(n));
}
