import { Dataset } from "./types";
import { mean, variance } from "./utils/descriptive";
import { solveLinearSystem, invertMatrix, normalCdf, transpose, matMul } from "./utils/linalg";
import { regularizedGammaP, regularizedBeta } from "./utils/math";

/**
 * Regression summary table with coefficient statistics.
 *
 * Similar to R's `summary(lm(...))` output, containing coefficient estimates
 * with standard errors, t-statistics, and p-values, plus model-level statistics
 * (R-squared, adjusted R-squared, F-statistic, residual standard error).
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

/**
 * A single row in the regression coefficient table.
 *
 * Contains the coefficient name, point estimate, standard error,
 * t-statistic, and two-sided p-value for testing H0: coefficient = 0.
 */
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
 * Residual diagnostics for evaluating regression model assumptions.
 *
 * Includes raw and standardized residuals, the Durbin-Watson statistic
 * for autocorrelation detection, and the Jarque-Bera test for normality.
 */
export interface ResidualDiagnostics {
  residuals: number[];
  standardizedResiduals: number[];
  durbinWatson: number;
  jarqueBera: { statistic: number; normalityLikely: boolean };
}

/**
 * Compute residual diagnostics for a regression model.
 *
 * Computes raw residuals (observed - predicted), standardized residuals
 * (z-scored), the Durbin-Watson statistic for detecting autocorrelation
 * (values near 2 suggest no autocorrelation), and the Jarque-Bera test
 * for residual normality.
 *
 * The Durbin-Watson statistic is: DW = sum((e_t - e_{t-1})^2) / sum(e_t^2).
 * The Jarque-Bera statistic is: JB = (n/6) * (S^2 + (K-3)^2/4) where
 * S is skewness and K is kurtosis of the standardized residuals.
 *
 * @param observed - Observed/actual values
 * @param predicted - Predicted/fitted values (same length as observed)
 * @returns A {@link ResidualDiagnostics} object with residuals, standardized residuals,
 *   Durbin-Watson statistic, and Jarque-Bera normality test
 * @throws {Error} If observed and predicted have different lengths
 * @throws {Error} If fewer than 3 observations
 * @throws {Error} If any value is not finite
 *
 * @example
 * ```ts
 * const diag = residualDiagnostics([1, 2, 3, 4], [1.1, 1.9, 3.2, 3.8]);
 * console.log(diag.durbinWatson);               // ~2.0 (no autocorrelation)
 * console.log(diag.jarqueBera.normalityLikely); // true
 * ```
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
 * For each feature j, VIF_j = 1 / (1 - R_j^2) where R_j^2 is the
 * R-squared from regressing feature j on all other features.
 *
 * Rules of thumb: VIF > 5 suggests moderate multicollinearity,
 * VIF > 10 indicates severe multicollinearity.
 *
 * @param X - Design matrix (n x p, without intercept column, p >= 2)
 * @returns An array of VIF values, one per feature (VIF >= 1 always)
 * @throws {Error} If X is empty
 * @throws {Error} If fewer than 2 features
 * @throws {Error} If n <= p (not enough observations)
 * @throws {Error} If any value is not finite
 *
 * @example
 * ```ts
 * const vifs = vif([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]);
 * console.log(vifs); // [VIF_x1, VIF_x2] -- high values suggest collinearity
 * ```
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

// ---------------------------------------------------------------------------
// New regression diagnostic interfaces
// ---------------------------------------------------------------------------

/**
 * Result of a heteroscedasticity test (Breusch-Pagan or White's).
 */
export interface HeteroscedasticityTestResult {
  statistic: number;
  pValue: number;
  df: number;
  reject: boolean;
}

/**
 * Result of the Ramsey RESET test for functional form misspecification.
 */
export interface RESETTestResult {
  fStatistic: number;
  pValue: number;
  df1: number;
  df2: number;
  reject: boolean;
}

/**
 * Influence diagnostics for each observation in a regression model.
 */
export interface InfluenceDiagnostics {
  cooksDistance: number[];
  leverage: number[];
  dffits: number[];
  dfbetas: number[][]; // [obs][coefficient]
  threshold: { cooks: number; leverage: number; dffits: number; dfbetas: number };
}

// ---------------------------------------------------------------------------
// Helper: chi-squared survival function P(chi² > x)
// ---------------------------------------------------------------------------
function chiSquaredSf(x: number, df: number): number {
  if (x <= 0) return 1;
  // P(chi² <= x) = regularizedGammaP(df/2, x/2)
  return 1 - regularizedGammaP(df / 2, x / 2);
}

// ---------------------------------------------------------------------------
// Helper: F-distribution survival function P(F > x)
// ---------------------------------------------------------------------------
function fDistSf(x: number, df1: number, df2: number): number {
  if (x <= 0) return 1;
  // P(F <= x) = I_{df1*x/(df1*x+df2)}(df1/2, df2/2)
  const t = (df1 * x) / (df1 * x + df2);
  return 1 - regularizedBeta(t, df1 / 2, df2 / 2);
}

// ---------------------------------------------------------------------------
// Helper: OLS fit returning beta, residuals, R² given a design matrix (with intercept already included)
// ---------------------------------------------------------------------------
function olsFit(
  designMatrix: number[][],
  y: number[],
): { beta: number[]; residuals: number[]; rSquared: number } {
  const n = designMatrix.length;
  const cols = designMatrix[0].length;

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
  const yMean = mean(y);
  let ssRes = 0;
  let ssTot = 0;
  const residuals = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    let predicted = 0;
    for (let j = 0; j < cols; j++) {
      predicted += designMatrix[i][j] * beta[j];
    }
    residuals[i] = y[i] - predicted;
    ssRes += residuals[i] ** 2;
    ssTot += (y[i] - yMean) ** 2;
  }

  const rSquared = ssTot > 0 ? 1 - ssRes / ssTot : 0;
  return { beta, residuals, rSquared };
}

// ---------------------------------------------------------------------------
// 1. Breusch-Pagan test
// ---------------------------------------------------------------------------

/**
 * Breusch-Pagan test for heteroscedasticity.
 *
 * Regresses squared residuals on the original regressors X. Under H0 of
 * homoscedasticity, LM = n * R² ~ chi²(p) where p is the number of columns
 * in X (excluding intercept).
 *
 * @param X - Design matrix (n x p, without intercept)
 * @param residuals - OLS residuals from the original regression
 * @returns A {@link HeteroscedasticityTestResult} with test statistic, p-value, df, and rejection at 5%
 * @throws {Error} If X and residuals have different lengths
 *
 * @example
 * ```ts
 * const result = breuschPaganTest(X, residuals);
 * console.log(result.reject); // true => evidence of heteroscedasticity
 * ```
 */
export function breuschPaganTest(
  X: number[][],
  residuals: number[],
): HeteroscedasticityTestResult {
  const n = X.length;
  if (n === 0) throw new Error("X must not be empty");
  if (n !== residuals.length) {
    throw new Error("X and residuals must have the same length");
  }
  const p = X[0].length;

  // Squared residuals as dependent variable
  const e2 = residuals.map((e) => e * e);

  // Build design matrix with intercept
  const designMatrix = X.map((row) => [1, ...row]);

  const { rSquared } = olsFit(designMatrix, e2);

  const statistic = n * rSquared;
  const df = p;
  const pValue = chiSquaredSf(statistic, df);

  return { statistic, pValue, df, reject: pValue < 0.05 };
}

// ---------------------------------------------------------------------------
// 2. White's test
// ---------------------------------------------------------------------------

/**
 * White's test for heteroscedasticity.
 *
 * Like the Breusch-Pagan test but augments X with squares of each feature
 * and all unique cross-products between features, making it a general test
 * that does not assume a specific form of heteroscedasticity.
 *
 * @param X - Design matrix (n x p, without intercept)
 * @param residuals - OLS residuals from the original regression
 * @returns A {@link HeteroscedasticityTestResult} with test statistic, p-value, df, and rejection at 5%
 * @throws {Error} If X and residuals have different lengths
 *
 * @example
 * ```ts
 * const result = whitesTest(X, residuals);
 * console.log(result.reject); // true => evidence of heteroscedasticity
 * ```
 */
export function whitesTest(
  X: number[][],
  residuals: number[],
): HeteroscedasticityTestResult {
  const n = X.length;
  if (n === 0) throw new Error("X must not be empty");
  if (n !== residuals.length) {
    throw new Error("X and residuals must have the same length");
  }
  const p = X[0].length;

  // Squared residuals
  const e2 = residuals.map((e) => e * e);

  // Augmented design: intercept, X, X², cross-products
  const augmented: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [1, ...X[i]];
    // Squares
    for (let j = 0; j < p; j++) {
      row.push(X[i][j] * X[i][j]);
    }
    // Cross-products
    for (let j = 0; j < p; j++) {
      for (let k = j + 1; k < p; k++) {
        row.push(X[i][j] * X[i][k]);
      }
    }
    augmented.push(row);
  }

  const df = augmented[0].length - 1; // number of regressors excluding intercept

  const { rSquared } = olsFit(augmented, e2);

  const statistic = n * rSquared;
  const pValue = chiSquaredSf(statistic, df);

  return { statistic, pValue, df, reject: pValue < 0.05 };
}

// ---------------------------------------------------------------------------
// 3. Ramsey RESET test
// ---------------------------------------------------------------------------

/**
 * Ramsey RESET test for functional form misspecification.
 *
 * Augments the original model by adding powers of the fitted values
 * (default: 2nd and 3rd powers) and tests whether these additions are
 * jointly significant via an F-test.
 *
 * @param X - Design matrix (n x p, without intercept)
 * @param y - Response variable
 * @param fitted - Fitted values from the original model
 * @param powers - Powers of fitted values to add (default [2, 3])
 * @returns A {@link RESETTestResult} with F-statistic, p-value, degrees of freedom, and rejection at 5%
 * @throws {Error} If X, y, and fitted have different lengths
 *
 * @example
 * ```ts
 * const result = ramseyReset(X, y, fitted);
 * console.log(result.reject); // true => evidence of misspecification
 * ```
 */
export function ramseyReset(
  X: number[][],
  y: Dataset,
  fitted: Dataset,
  powers: number[] = [2, 3],
): RESETTestResult {
  const n = X.length;
  if (n === 0) throw new Error("X must not be empty");
  if (n !== y.length || n !== fitted.length) {
    throw new Error("X, y, and fitted must have the same length");
  }
  const p = X[0].length;
  const cols = p + 1; // original model with intercept

  // Original model design matrix
  const origDesign = X.map((row) => [1, ...row]);

  // Fit original model to get SSR_restricted
  const origFit = olsFit(origDesign, y as number[]);
  const ssResRestricted = origFit.residuals.reduce((s, e) => s + e * e, 0);

  // Augmented design: original + powers of fitted
  const q = powers.length; // number of added terms
  const augDesign = origDesign.map((row, i) => {
    const extra = powers.map((pw) => Math.pow(fitted[i], pw));
    return [...row, ...extra];
  });

  const augFit = olsFit(augDesign, y as number[]);
  const ssResUnrestricted = augFit.residuals.reduce((s, e) => s + e * e, 0);

  // F-test: ((SSR_r - SSR_u) / q) / (SSR_u / (n - cols - q))
  const df1 = q;
  const df2 = n - cols - q;
  const fStatistic =
    df2 > 0
      ? ((ssResRestricted - ssResUnrestricted) / df1) / (ssResUnrestricted / df2)
      : 0;
  const pValue = df2 > 0 ? fDistSf(fStatistic, df1, df2) : 1;

  return { fStatistic, pValue, df1, df2, reject: pValue < 0.05 };
}

// ---------------------------------------------------------------------------
// 4. Leverage (hat) values
// ---------------------------------------------------------------------------

/**
 * Compute leverage (hat) values -- diagonal of the hat matrix H = X(X'X)^{-1}X'.
 *
 * Leverage measures how far an observation's predictor values are from
 * the mean of the predictor values. High leverage points have h_ii > 2p/n.
 *
 * @param X - Design matrix (n x p, without intercept). An intercept column is prepended internally.
 * @returns Array of leverage values h_ii for each observation
 * @throws {Error} If X is empty or (X'X) is singular
 *
 * @example
 * ```ts
 * const h = leverageValues([[1], [2], [3], [4], [5]]);
 * // h[i] are the hat matrix diagonal elements
 * ```
 */
export function leverageValues(X: number[][]): number[] {
  const n = X.length;
  if (n === 0) throw new Error("X must not be empty");
  const p = X[0].length;
  const cols = p + 1;

  // Design matrix with intercept
  const D = X.map((row) => [1, ...row]);

  // Compute (X'X)^{-1}
  const XtX = Array.from({ length: cols }, () => new Array<number>(cols).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < cols; j++) {
      for (let k = 0; k < cols; k++) {
        XtX[j][k] += D[i][j] * D[i][k];
      }
    }
  }

  const XtXInv = invertMatrix(XtX);
  if (XtXInv === null) {
    throw new Error("X'X matrix is singular");
  }

  // h_ii = D[i] * (X'X)^{-1} * D[i]'
  const h = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    let hii = 0;
    for (let j = 0; j < cols; j++) {
      for (let k = 0; k < cols; k++) {
        hii += D[i][j] * XtXInv[j][k] * D[i][k];
      }
    }
    h[i] = hii;
  }

  return h;
}

// ---------------------------------------------------------------------------
// 5. Cook's distance
// ---------------------------------------------------------------------------

/**
 * Compute Cook's distance for each observation.
 *
 * D_i = (e_i² / (p * MSE)) * (h_ii / (1 - h_ii)²)
 *
 * Cook's distance measures the influence of each observation on the
 * full set of fitted values. A common threshold is D_i > 4/n.
 *
 * @param X - Design matrix (n x p, without intercept)
 * @param y - Response variable
 * @returns Array of Cook's distance values
 * @throws {Error} If X and y have different lengths
 *
 * @example
 * ```ts
 * const d = cooksDistance([[1], [2], [3], [4], [5]], [2, 4, 5, 4, 5]);
 * // d[i] is Cook's distance for observation i
 * ```
 */
export function cooksDistance(X: number[][], y: Dataset): number[] {
  const n = X.length;
  if (n === 0) throw new Error("X must not be empty");
  if (n !== y.length) throw new Error("X and y must have the same length");
  const p = X[0].length;
  const cols = p + 1;

  const D = X.map((row) => [1, ...row]);
  const { residuals } = olsFit(D, y as number[]);
  const h = leverageValues(X);

  const mse = residuals.reduce((s, e) => s + e * e, 0) / (n - cols);

  const cd = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    const denom = cols * mse * (1 - h[i]) ** 2;
    cd[i] = denom > 0 ? (residuals[i] ** 2 * h[i]) / denom : 0;
  }

  return cd;
}

// ---------------------------------------------------------------------------
// 6. DFBETAS
// ---------------------------------------------------------------------------

/**
 * Compute DFBETAS for each observation and each coefficient.
 *
 * DFBETAS_{i,j} = (beta_j - beta_j(-i)) / (s(-i) * sqrt(c_jj))
 *
 * Measures the influence of observation i on coefficient j. A common
 * threshold is |DFBETAS| > 2/sqrt(n).
 *
 * @param X - Design matrix (n x p, without intercept)
 * @param y - Response variable
 * @returns A 2D array [n][cols] of DFBETAS values where cols = p + 1 (intercept + features)
 * @throws {Error} If X and y have different lengths
 *
 * @example
 * ```ts
 * const db = dfbetas([[1], [2], [3], [4], [5]], [2, 4, 5, 4, 5]);
 * // db[i][j] is the DFBETAS for observation i, coefficient j
 * ```
 */
export function dfbetas(X: number[][], y: Dataset): number[][] {
  const n = X.length;
  if (n === 0) throw new Error("X must not be empty");
  if (n !== y.length) throw new Error("X and y must have the same length");
  const p = X[0].length;
  const cols = p + 1;

  const D = X.map((row) => [1, ...row]);
  const fullFit = olsFit(D, y as number[]);
  const betaFull = fullFit.beta;

  // Compute (X'X)^{-1} for c_jj
  const XtX = Array.from({ length: cols }, () => new Array<number>(cols).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < cols; j++) {
      for (let k = 0; k < cols; k++) {
        XtX[j][k] += D[i][j] * D[i][k];
      }
    }
  }
  const XtXInv = invertMatrix(XtX);
  if (XtXInv === null) {
    throw new Error("X'X matrix is singular");
  }

  const result: number[][] = [];

  for (let i = 0; i < n; i++) {
    // Leave-one-out design and response
    const DMinusI = D.filter((_, idx) => idx !== i);
    const yMinusI = (y as number[]).filter((_, idx) => idx !== i);

    const leaveOneFit = olsFit(DMinusI, yMinusI);
    const betaMinusI = leaveOneFit.beta;

    // s(-i): residual standard error from leave-one-out
    const ssResMinusI = leaveOneFit.residuals.reduce((s, e) => s + e * e, 0);
    const sMinusI = Math.sqrt(ssResMinusI / (n - 1 - cols));

    const row = new Array<number>(cols);
    for (let j = 0; j < cols; j++) {
      const denom = sMinusI * Math.sqrt(XtXInv[j][j]);
      row[j] = denom > 0 ? (betaFull[j] - betaMinusI[j]) / denom : 0;
    }
    result.push(row);
  }

  return result;
}

// ---------------------------------------------------------------------------
// 7. DFFITS
// ---------------------------------------------------------------------------

/**
 * Compute DFFITS for each observation.
 *
 * DFFITS_i = e_i * sqrt(h_ii) / (s(-i) * (1 - h_ii))
 *
 * Measures the influence of observation i on its own fitted value.
 * A common threshold is |DFFITS| > 2 * sqrt(p/n).
 *
 * @param X - Design matrix (n x p, without intercept)
 * @param y - Response variable
 * @returns Array of DFFITS values
 * @throws {Error} If X and y have different lengths
 *
 * @example
 * ```ts
 * const df = dffits([[1], [2], [3], [4], [5]], [2, 4, 5, 4, 5]);
 * // df[i] is the DFFITS for observation i
 * ```
 */
export function dffits(X: number[][], y: Dataset): number[] {
  const n = X.length;
  if (n === 0) throw new Error("X must not be empty");
  if (n !== y.length) throw new Error("X and y must have the same length");
  const p = X[0].length;
  const cols = p + 1;

  const D = X.map((row) => [1, ...row]);
  const { residuals } = olsFit(D, y as number[]);
  const h = leverageValues(X);

  const result = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    // Leave-one-out residual standard error
    const DMinusI = D.filter((_, idx) => idx !== i);
    const yMinusI = (y as number[]).filter((_, idx) => idx !== i);
    const leaveOneFit = olsFit(DMinusI, yMinusI);
    const ssResMinusI = leaveOneFit.residuals.reduce((s, e) => s + e * e, 0);
    const sMinusI = Math.sqrt(ssResMinusI / (n - 1 - cols));

    const denom = sMinusI * (1 - h[i]);
    result[i] = denom > 0 ? (residuals[i] * Math.sqrt(h[i])) / denom : 0;
  }

  return result;
}

// ---------------------------------------------------------------------------
// 8. Condition number
// ---------------------------------------------------------------------------

/**
 * Compute the condition number of the design matrix X.
 *
 * The condition number is the ratio of the largest to smallest singular
 * value of X (with intercept prepended). A large condition number
 * (> 30) suggests numerical instability / multicollinearity.
 *
 * Singular values are obtained from the eigenvalues of X'X:
 * sigma_i = sqrt(lambda_i).
 *
 * @param X - Design matrix (n x p, without intercept)
 * @returns The condition number (>= 1)
 * @throws {Error} If X is empty or X'X is singular
 *
 * @example
 * ```ts
 * const cn = conditionNumber([[1, 2], [3, 4], [5, 6]]);
 * // cn > 30 suggests multicollinearity issues
 * ```
 */
export function conditionNumber(X: number[][]): number {
  const n = X.length;
  if (n === 0) throw new Error("X must not be empty");
  const p = X[0].length;
  const cols = p + 1;

  // Design matrix with intercept
  const D = X.map((row) => [1, ...row]);

  // Compute X'X
  const XtX = Array.from({ length: cols }, () => new Array<number>(cols).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < cols; j++) {
      for (let k = 0; k < cols; k++) {
        XtX[j][k] += D[i][j] * D[i][k];
      }
    }
  }

  // Power iteration to find largest eigenvalue, inverse iteration for smallest
  // Use a simple approach: compute eigenvalues via iterative QR-like method
  // For robustness, use the Jacobi eigenvalue algorithm for symmetric matrices

  const eigenvalues = jacobiEigenvalues(XtX);

  const maxEig = Math.max(...eigenvalues);
  const minEig = Math.min(...eigenvalues);

  if (minEig <= 0) {
    return Infinity;
  }

  return Math.sqrt(maxEig / minEig);
}

/**
 * Jacobi eigenvalue algorithm for a real symmetric matrix.
 * Returns eigenvalues only (not eigenvectors).
 */
function jacobiEigenvalues(A: number[][]): number[] {
  const n = A.length;
  // Work on a copy
  const M = A.map((row) => [...row]);

  const maxIter = 100 * n * n;
  for (let iter = 0; iter < maxIter; iter++) {
    // Find largest off-diagonal element
    let maxVal = 0;
    let pi = 0;
    let qi = 1;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (Math.abs(M[i][j]) > maxVal) {
          maxVal = Math.abs(M[i][j]);
          pi = i;
          qi = j;
        }
      }
    }

    if (maxVal < 1e-12) break;

    // Compute rotation
    const app = M[pi][pi];
    const aqq = M[qi][qi];
    const apq = M[pi][qi];

    let theta: number;
    if (Math.abs(app - aqq) < 1e-15) {
      theta = Math.PI / 4;
    } else {
      theta = 0.5 * Math.atan2(2 * apq, app - aqq);
    }

    const c = Math.cos(theta);
    const s = Math.sin(theta);

    // Apply rotation
    const newPP = c * c * app + 2 * s * c * apq + s * s * aqq;
    const newQQ = s * s * app - 2 * s * c * apq + c * c * aqq;
    M[pi][pi] = newPP;
    M[qi][qi] = newQQ;
    M[pi][qi] = 0;
    M[qi][pi] = 0;

    for (let i = 0; i < n; i++) {
      if (i === pi || i === qi) continue;
      const mip = M[i][pi];
      const miq = M[i][qi];
      M[i][pi] = c * mip + s * miq;
      M[pi][i] = M[i][pi];
      M[i][qi] = -s * mip + c * miq;
      M[qi][i] = M[i][qi];
    }
  }

  return Array.from({ length: n }, (_, i) => M[i][i]);
}
