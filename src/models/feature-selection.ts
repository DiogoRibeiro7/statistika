/**
 * Feature selection and regularised regression.
 *
 * - **Stepwise selection** (forward, backward, both) using AIC / BIC / adjusted-R².
 * - **Ridge regression** (L2 penalty, closed-form via augmented normal equations).
 * - **LASSO regression** (L1 penalty, coordinate descent).
 * - **Elastic Net** (combined L1 + L2, coordinate descent).
 *
 * All penalised methods standardise features internally and return
 * coefficients on the **original** scale.
 */

import { mean } from "../utils/descriptive";
import { solveLinearSystem } from "../utils/linalg";

// ── Result types ──────────────────────────────────────────────────────────

export interface StepwiseResult {
  /** Indices of selected features (0-based, referring to columns of X). */
  selectedFeatures: number[];
  /** Regression coefficients for selected features (same order). */
  coefficients: number[];
  /** Intercept. */
  intercept: number;
  /** R² on the full dataset. */
  rSquared: number;
  /** Final criterion value (AIC, BIC, or adjusted-R²). */
  criterion: number;
  /** Predict y for a full feature vector (all p features). */
  predict: (x: number[]) => number;
}

export interface RegularisedResult {
  /** Regression coefficients for all p features. */
  coefficients: number[];
  /** Intercept. */
  intercept: number;
  /** Regularisation parameter used. */
  lambda: number;
  /** Number of coordinate-descent iterations (LASSO / Elastic Net). */
  iterations: number;
  /** Predict y for a feature vector. */
  predict: (x: number[]) => number;
}

export type StepwiseCriterion = "aic" | "bic" | "adjr2";
export type StepwiseDirection = "forward" | "backward" | "both";

// ── Stepwise selection ────────────────────────────────────────────────────

/**
 * Stepwise feature selection.
 *
 * @param X  Feature matrix (n × p).
 * @param y  Response vector (length n).
 * @param options.direction  "forward" | "backward" | "both" (default "both").
 * @param options.criterion  "aic" | "bic" | "adjr2" (default "aic").
 * @param options.maxFeatures  Upper bound on selected features (default p).
 */
export function stepwiseSelection(
  X: number[][],
  y: number[],
  options: {
    direction?: StepwiseDirection;
    criterion?: StepwiseCriterion;
    maxFeatures?: number;
  } = {},
): StepwiseResult {
  const { direction = "both", criterion = "aic", maxFeatures } = options;
  const n = X.length;
  validateInputs(X, y);
  const p = X[0].length;
  const maxF = maxFeatures ?? p;

  const evaluate = (features: number[]): number => {
    if (features.length === 0) {
      // Intercept-only model
      const yMean = mean(y);
      let rss = 0;
      for (let i = 0; i < n; i++) rss += (y[i] - yMean) ** 2;
      return scoreCriterion(rss, n, 1, criterion);
    }
    const { rss } = fitOLS(X, y, features);
    return scoreCriterion(rss, n, features.length + 1, criterion);
  };

  // The lower the score, the better (for AIC/BIC); for adjR² we negate.
  const isBetter = (a: number, b: number): boolean => a < b;

  let current: number[] =
    direction === "backward"
      ? Array.from({ length: p }, (_, i) => i)
      : [];
  let bestScore = evaluate(current);

  let improved = true;
  while (improved) {
    improved = false;

    // Forward step: try adding each feature not yet in the model
    if (
      (direction === "forward" || direction === "both") &&
      current.length < maxF
    ) {
      let bestAdd = -1;
      let bestAddScore = bestScore;
      for (let f = 0; f < p; f++) {
        if (current.includes(f)) continue;
        const candidate = [...current, f];
        const score = evaluate(candidate);
        if (isBetter(score, bestAddScore)) {
          bestAddScore = score;
          bestAdd = f;
        }
      }
      if (bestAdd >= 0) {
        current = [...current, bestAdd];
        bestScore = bestAddScore;
        improved = true;
      }
    }

    // Backward step: try removing each feature in the model
    if (
      (direction === "backward" || direction === "both") &&
      current.length > 0
    ) {
      let bestRemove = -1;
      let bestRemoveScore = bestScore;
      for (let idx = 0; idx < current.length; idx++) {
        const candidate = current.filter((_, i) => i !== idx);
        const score = evaluate(candidate);
        if (isBetter(score, bestRemoveScore)) {
          bestRemoveScore = score;
          bestRemove = idx;
        }
      }
      if (bestRemove >= 0) {
        current = current.filter((_, i) => i !== bestRemove);
        bestScore = bestRemoveScore;
        improved = true;
      }
    }
  }

  // Final fit
  if (current.length === 0) {
    const yMean = mean(y);
    return {
      selectedFeatures: [],
      coefficients: [],
      intercept: yMean,
      rSquared: 0,
      criterion: bestScore,
      predict: () => yMean,
    };
  }

  const { beta, rss } = fitOLS(X, y, current);
  const intercept = beta[0];
  const coefficients = beta.slice(1);

  let ssTot = 0;
  const yMean = mean(y);
  for (let i = 0; i < n; i++) ssTot += (y[i] - yMean) ** 2;
  const rSquared = 1 - rss / ssTot;

  const selected = [...current];
  return {
    selectedFeatures: selected,
    coefficients,
    intercept,
    rSquared,
    criterion: bestScore,
    predict: (x: number[]) => {
      let result = intercept;
      for (let j = 0; j < selected.length; j++) {
        result += coefficients[j] * x[selected[j]];
      }
      return result;
    },
  };
}

// ── Ridge regression ──────────────────────────────────────────────────────

/**
 * Ridge regression (L2 penalty).
 *
 * Solves  min  ‖y − Xβ‖² + λ ‖β‖²
 * via the augmented normal equations (X'X + λI)β = X'y.
 *
 * Features are standardised internally; returned coefficients are on the
 * original scale.
 *
 * @param X  Feature matrix (n × p).
 * @param y  Response vector (length n).
 * @param lambda  Regularisation strength (≥ 0).
 */
export function ridgeRegression(
  X: number[][],
  y: number[],
  lambda: number,
): RegularisedResult {
  validateInputs(X, y);
  if (lambda < 0) throw new Error("lambda must be non-negative");
  const n = X.length;
  const p = X[0].length;

  // Standardise
  const { Xs, xMeans, xStds, yMean } = standardise(X, y);
  const yc = new Array(n);
  for (let i = 0; i < n; i++) yc[i] = y[i] - yMean;

  // Build (Xs'Xs + λI) and Xs'yc
  const XtX: number[][] = Array.from({ length: p }, () =>
    new Array(p).fill(0),
  );
  const Xty = new Array(p).fill(0);

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < p; j++) {
      Xty[j] += Xs[i][j] * yc[i];
      for (let k = j; k < p; k++) {
        XtX[j][k] += Xs[i][j] * Xs[i][k];
      }
    }
  }
  // Symmetrise and add penalty
  for (let j = 0; j < p; j++) {
    for (let k = j + 1; k < p; k++) {
      XtX[k][j] = XtX[j][k];
    }
    XtX[j][j] += lambda;
  }

  const betaStd = solveLinearSystem(XtX, Xty);

  // Unstandardise
  const coefficients = new Array(p);
  let intercept = yMean;
  for (let j = 0; j < p; j++) {
    coefficients[j] = xStds[j] > 0 ? betaStd[j] / xStds[j] : 0;
    intercept -= coefficients[j] * xMeans[j];
  }

  return {
    coefficients,
    intercept,
    lambda,
    iterations: 0,
    predict: (x: number[]) => {
      let result = intercept;
      for (let j = 0; j < p; j++) result += coefficients[j] * x[j];
      return result;
    },
  };
}

// ── LASSO regression ──────────────────────────────────────────────────────

/**
 * LASSO regression (L1 penalty) via coordinate descent.
 *
 * Solves  min  (1/2n) ‖y − Xβ‖² + λ ‖β‖₁
 *
 * Features are standardised internally; returned coefficients are on the
 * original scale.
 *
 * @param X  Feature matrix (n × p).
 * @param y  Response vector (length n).
 * @param lambda  Regularisation strength (≥ 0).
 * @param options.maxIterations  Maximum iterations (default 1000).
 * @param options.tolerance  Convergence tolerance (default 1e-7).
 */
export function lassoRegression(
  X: number[][],
  y: number[],
  lambda: number,
  options: { maxIterations?: number; tolerance?: number } = {},
): RegularisedResult {
  return elasticNet(X, y, lambda, { ...options, alpha: 1.0 });
}

// ── Elastic Net ───────────────────────────────────────────────────────────

/**
 * Elastic Net regression via coordinate descent.
 *
 * Solves  min  (1/2n) ‖y − Xβ‖² + λ [ α ‖β‖₁ + (1−α)/2 ‖β‖² ]
 *
 * - α = 1 → LASSO
 * - α = 0 → Ridge
 *
 * Features are standardised internally; returned coefficients are on the
 * original scale.
 *
 * @param X  Feature matrix (n × p).
 * @param y  Response vector (length n).
 * @param lambda  Regularisation strength (≥ 0).
 * @param options.alpha  Mixing parameter in [0, 1] (default 0.5).
 * @param options.maxIterations  Maximum iterations (default 1000).
 * @param options.tolerance  Convergence tolerance (default 1e-7).
 */
export function elasticNet(
  X: number[][],
  y: number[],
  lambda: number,
  options: {
    alpha?: number;
    maxIterations?: number;
    tolerance?: number;
  } = {},
): RegularisedResult {
  const { alpha = 0.5, maxIterations = 1000, tolerance = 1e-7 } = options;
  validateInputs(X, y);
  if (lambda < 0) throw new Error("lambda must be non-negative");
  if (alpha < 0 || alpha > 1) throw new Error("alpha must be in [0, 1]");
  const n = X.length;
  const p = X[0].length;

  // Standardise
  const { Xs, xMeans, xStds, yMean } = standardise(X, y);
  const yc = new Array(n);
  for (let i = 0; i < n; i++) yc[i] = y[i] - yMean;

  // Precompute column norms (Xs'Xs diagonal, all = n after standardisation for unit-variance cols)
  const colNorms = new Array(p);
  for (let j = 0; j < p; j++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += Xs[i][j] * Xs[i][j];
    colNorms[j] = s;
  }

  // Coordinate descent
  const betaStd = new Array(p).fill(0);
  const residuals = new Float64Array(yc);
  let iterations = 0;

  for (let iter = 0; iter < maxIterations; iter++) {
    iterations = iter + 1;
    let maxChange = 0;

    for (let j = 0; j < p; j++) {
      const oldBeta = betaStd[j];

      // Partial residual dot product
      let rho = 0;
      for (let i = 0; i < n; i++) {
        rho += Xs[i][j] * (residuals[i] + Xs[i][j] * oldBeta);
      }

      // Soft-threshold
      const lambdaAlpha = lambda * alpha * n;
      const lambdaL2 = lambda * (1 - alpha) * n;
      const newBeta =
        softThreshold(rho, lambdaAlpha) / (colNorms[j] + lambdaL2);

      if (newBeta !== oldBeta) {
        const diff = newBeta - oldBeta;
        // Update residuals
        for (let i = 0; i < n; i++) {
          residuals[i] -= Xs[i][j] * diff;
        }
        betaStd[j] = newBeta;
        maxChange = Math.max(maxChange, Math.abs(diff));
      }
    }

    if (maxChange < tolerance) break;
  }

  // Unstandardise
  const coefficients = new Array(p);
  let intercept = yMean;
  for (let j = 0; j < p; j++) {
    coefficients[j] = xStds[j] > 0 ? betaStd[j] / xStds[j] : 0;
    intercept -= coefficients[j] * xMeans[j];
  }

  return {
    coefficients,
    intercept,
    lambda,
    iterations,
    predict: (x: number[]) => {
      let result = intercept;
      for (let j = 0; j < p; j++) result += coefficients[j] * x[j];
      return result;
    },
  };
}

// ── Helpers ───────────────────────────────────────────────────────────────

function validateInputs(X: number[][], y: number[]): void {
  const n = X.length;
  if (n !== y.length) {
    throw new Error("X and y must have the same number of observations");
  }
  if (n < 2) {
    throw new Error("Must have at least 2 observations");
  }
  const p = X[0].length;
  if (p === 0) {
    throw new Error("Feature vectors must not be empty");
  }
  for (let i = 1; i < n; i++) {
    if (X[i].length !== p) {
      throw new Error("All feature vectors must have the same length");
    }
  }
}

function standardise(
  X: number[][],
  y: number[],
): {
  Xs: number[][];
  xMeans: number[];
  xStds: number[];
  yMean: number;
} {
  const n = X.length;
  const p = X[0].length;
  const xMeans = new Array(p).fill(0);
  const xStds = new Array(p).fill(0);

  for (let j = 0; j < p; j++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += X[i][j];
    xMeans[j] = s / n;
  }
  for (let j = 0; j < p; j++) {
    let s = 0;
    for (let i = 0; i < n; i++) {
      const d = X[i][j] - xMeans[j];
      s += d * d;
    }
    xStds[j] = Math.sqrt(s / n);
  }

  const Xs: number[][] = new Array(n);
  for (let i = 0; i < n; i++) {
    Xs[i] = new Array(p);
    for (let j = 0; j < p; j++) {
      Xs[i][j] = xStds[j] > 0 ? (X[i][j] - xMeans[j]) / xStds[j] : 0;
    }
  }

  const yMean = mean(y);

  return { Xs, xMeans, xStds, yMean };
}

/** Soft-thresholding operator S(z, γ) = sign(z) * max(|z| − γ, 0). */
function softThreshold(z: number, gamma: number): number {
  if (z > gamma) return z - gamma;
  if (z < -gamma) return z + gamma;
  return 0;
}

/**
 * Score a model by AIC, BIC, or adjusted-R².
 * Returns a value where **lower is better** (adjusted-R² is negated).
 */
function scoreCriterion(
  rss: number,
  n: number,
  k: number,
  criterion: StepwiseCriterion,
): number {
  switch (criterion) {
    case "aic":
      return n * Math.log(rss / n) + 2 * k;
    case "bic":
      return n * Math.log(rss / n) + Math.log(n) * k;
    case "adjr2": {
      // We want lower = better, so negate adjusted-R²
      // adjR² = 1 − (RSS/(n−k)) / (TSS/(n−1))
      // But we don't have TSS here. We only have RSS.
      // Use the negative of explained proportion.
      // Actually, since TSS is constant across models, we can use RSS/(n−k):
      return rss / (n - k);
    }
  }
}

/**
 * Fit OLS on a subset of features. Returns beta (with intercept at [0])
 * and the residual sum of squares.
 */
function fitOLS(
  X: number[][],
  y: number[],
  features: number[],
): { beta: number[]; rss: number } {
  const n = X.length;
  const q = features.length;
  const cols = q + 1; // intercept + selected features

  const XtX: number[][] = Array.from({ length: cols }, () =>
    new Array(cols).fill(0),
  );
  const Xty = new Array(cols).fill(0);

  for (let i = 0; i < n; i++) {
    const row0 = 1; // intercept
    Xty[0] += row0 * y[i];
    XtX[0][0] += row0 * row0;
    for (let j = 0; j < q; j++) {
      const xj = X[i][features[j]];
      Xty[j + 1] += xj * y[i];
      XtX[j + 1][0] += xj * row0;
      XtX[0][j + 1] += row0 * xj;
      for (let k = 0; k < q; k++) {
        XtX[j + 1][k + 1] += xj * X[i][features[k]];
      }
    }
  }

  const beta = solveLinearSystem(XtX, Xty);

  let rss = 0;
  for (let i = 0; i < n; i++) {
    let yHat = beta[0];
    for (let j = 0; j < q; j++) {
      yHat += beta[j + 1] * X[i][features[j]];
    }
    rss += (y[i] - yHat) ** 2;
  }

  return { beta, rss };
}
