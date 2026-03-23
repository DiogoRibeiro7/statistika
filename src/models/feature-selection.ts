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
import { nativeAddon } from "../utils/native-addon";

// ── Native interface ──────────────────────────────────────────────────────

/**
 * Interface for optional native (Fortran/C) accelerated implementations
 * of regularised regression solvers.
 */
interface NativeFeatureSelection {
  elasticNetCd(
    X: number[][],
    yc: number[],
    beta: number[],
    residuals: number[],
    colNorms: number[],
    lambda: number,
    alpha: number,
    maxIter: number,
    tol: number,
  ): { beta: number[]; residuals: number[]; iterations: number };
  ridgeSolve(
    X: number[][],
    yc: number[],
    lambda: number,
  ): { beta: number[]; info: number };
}

let native: NativeFeatureSelection | null = null;
try {
  if (nativeAddon && typeof nativeAddon.elasticNetCd === "function") {
    native = nativeAddon as unknown as NativeFeatureSelection;
  }
} catch {
  // Fallback to TypeScript
}

/** Whether native feature selection acceleration is available. */
export const hasNativeFeatureSelection = native !== null;

// ── Result types ──────────────────────────────────────────────────────────

/**
 * Result of stepwise feature selection, including the selected feature subset
 * and fitted regression model.
 */
export interface StepwiseResult {
  /** 0-based indices of selected features, referring to columns of the input matrix X. */
  selectedFeatures: number[];
  /** Regression coefficients for the selected features (same order as `selectedFeatures`). */
  coefficients: number[];
  /** Intercept (constant) term of the fitted model. */
  intercept: number;
  /** Coefficient of determination (R^2) on the full dataset. */
  rSquared: number;
  /** Final information criterion value (AIC, BIC, or negated adjusted-R^2). Lower is better. */
  criterion: number;
  /**
   * Predicts y for a full feature vector containing all p features.
   * Only the selected features are used internally.
   * @param x - Feature vector of length p (all original features).
   * @returns The predicted response value.
   */
  predict: (x: number[]) => number;
}

/**
 * Result of a regularised (penalised) regression fit (Ridge, LASSO, or Elastic Net).
 */
export interface RegularisedResult {
  /** Regression coefficients for all p features on the original (unstandardised) scale. */
  coefficients: number[];
  /** Intercept (constant) term on the original scale. */
  intercept: number;
  /** Regularisation strength parameter that was used. */
  lambda: number;
  /** Number of coordinate-descent iterations performed (0 for Ridge, which uses a closed-form solution). */
  iterations: number;
  /**
   * Predicts y for a feature vector.
   * @param x - Feature vector of length p.
   * @returns The predicted response value.
   */
  predict: (x: number[]) => number;
}

/** Model selection criterion for stepwise feature selection. */
export type StepwiseCriterion = "aic" | "bic" | "adjr2";

/** Search direction for stepwise feature selection. */
export type StepwiseDirection = "forward" | "backward" | "both";

// ── Stepwise selection ────────────────────────────────────────────────────

/**
 * Performs stepwise feature selection for linear regression models.
 *
 * Iteratively adds (forward) or removes (backward) features from the model,
 * evaluating each candidate model using the specified information criterion.
 * The "both" direction alternates between forward and backward steps.
 *
 * Criteria:
 * - **AIC**: n * ln(RSS/n) + 2k (Akaike Information Criterion)
 * - **BIC**: n * ln(RSS/n) + ln(n) * k (Bayesian Information Criterion)
 * - **adjR2**: RSS / (n - k) as a proxy (lower is better)
 *
 * @param X - Feature matrix of shape (n x p).
 * @param y - Response vector of length n.
 * @param options - Configuration options.
 * @param options.direction - Search direction: "forward", "backward", or "both" (default "both").
 * @param options.criterion - Model selection criterion: "aic", "bic", or "adjr2" (default "aic").
 * @param options.maxFeatures - Upper bound on the number of selected features (default p).
 * @returns A {@link StepwiseResult} containing the selected features, fitted model,
 *   and the final criterion value.
 * @throws {Error} If `X` and `y` have different numbers of observations.
 * @throws {Error} If fewer than 2 observations or empty feature vectors are provided.
 *
 * @example
 * ```ts
 * const X = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]];
 * const y = [1, 2, 3, 4];
 * const result = stepwiseSelection(X, y, { direction: "forward", criterion: "bic" });
 * result.selectedFeatures; // indices of the best feature subset
 * ```
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
 * Fits a Ridge regression model with L2 penalty.
 *
 * Solves the optimization problem:
 *
 *   min  ||y - X*beta||^2 + lambda * ||beta||^2
 *
 * via the augmented normal equations: (X^T X + lambda * I) beta = X^T y.
 *
 * Features are standardised internally (zero mean, unit variance); returned
 * coefficients are transformed back to the original scale.
 *
 * @param X - Feature matrix of shape (n x p).
 * @param y - Response vector of length n.
 * @param lambda - Regularisation strength (must be >= 0). Larger values
 *   shrink coefficients more aggressively toward zero.
 * @returns A {@link RegularisedResult} with coefficients on the original scale,
 *   intercept, and a `predict` function. The `iterations` field is always 0
 *   since Ridge uses a closed-form solution.
 * @throws {Error} If `X` and `y` have different numbers of observations.
 * @throws {Error} If fewer than 2 observations or empty feature vectors are provided.
 * @throws {Error} If `lambda` is negative.
 *
 * @example
 * ```ts
 * const X = [[1], [2], [3], [4]];
 * const y = [2.1, 3.9, 6.2, 7.8];
 * const result = ridgeRegression(X, y, 0.1);
 * result.predict([5]); // predicted value with slight L2 regularisation
 * ```
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

  let betaStd: number[];

  if (native) {
    const result = native.ridgeSolve(Xs, yc, lambda);
    if (result.info !== 0) {
      throw new Error("Ridge solve failed (singular system)");
    }
    betaStd = result.beta;
  } else {
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

    betaStd = solveLinearSystem(XtX, Xty);
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
 * Fits a LASSO regression model with L1 penalty via coordinate descent.
 *
 * Solves the optimization problem:
 *
 *   min  (1 / 2n) ||y - X*beta||^2 + lambda * ||beta||_1
 *
 * The L1 penalty promotes sparsity, driving some coefficients exactly to zero.
 * This is equivalent to calling {@link elasticNet} with alpha = 1.0.
 *
 * Features are standardised internally; returned coefficients are on the
 * original scale.
 *
 * @param X - Feature matrix of shape (n x p).
 * @param y - Response vector of length n.
 * @param lambda - Regularisation strength (must be >= 0). Larger values
 *   yield sparser models.
 * @param options - Optional configuration.
 * @param options.maxIterations - Maximum coordinate descent iterations (default 1000).
 * @param options.tolerance - Convergence tolerance on coefficient changes (default 1e-7).
 * @returns A {@link RegularisedResult} with coefficients (some may be exactly zero),
 *   intercept, iteration count, and a `predict` function.
 * @throws {Error} If `X` and `y` have different numbers of observations.
 * @throws {Error} If `lambda` is negative.
 *
 * @example
 * ```ts
 * const X = [[1, 0.5], [2, 1.1], [3, 1.4], [4, 2.0]];
 * const y = [2, 4, 6, 8];
 * const result = lassoRegression(X, y, 0.5);
 * // Some coefficients may be exactly 0 due to L1 sparsity
 * ```
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
 * Fits an Elastic Net regression model via coordinate descent.
 *
 * Solves the optimization problem:
 *
 *   min  (1 / 2n) ||y - X*beta||^2 + lambda * [ alpha * ||beta||_1 + (1 - alpha)/2 * ||beta||^2 ]
 *
 * The mixing parameter alpha interpolates between:
 * - alpha = 1: pure LASSO (L1 penalty, promotes sparsity)
 * - alpha = 0: pure Ridge (L2 penalty, shrinks coefficients)
 *
 * The coordinate descent update for each coefficient j uses soft-thresholding:
 *   beta_j = S(rho_j, lambda * alpha * n) / (||X_j||^2 + lambda * (1 - alpha) * n)
 *
 * Features are standardised internally (zero mean, unit variance); returned
 * coefficients are transformed back to the original scale.
 *
 * @param X - Feature matrix of shape (n x p).
 * @param y - Response vector of length n.
 * @param lambda - Regularisation strength (must be >= 0).
 * @param options - Optional configuration.
 * @param options.alpha - L1/L2 mixing parameter in [0, 1] (default 0.5).
 * @param options.maxIterations - Maximum coordinate descent iterations (default 1000).
 * @param options.tolerance - Convergence tolerance on the maximum absolute
 *   coefficient change per iteration (default 1e-7).
 * @returns A {@link RegularisedResult} with coefficients on the original scale,
 *   intercept, iteration count, and a `predict` function.
 * @throws {Error} If `X` and `y` have different numbers of observations.
 * @throws {Error} If fewer than 2 observations or empty feature vectors are provided.
 * @throws {Error} If `lambda` is negative.
 * @throws {Error} If `alpha` is not in [0, 1].
 *
 * @example
 * ```ts
 * const X = [[1, 2], [3, 4], [5, 6], [7, 8]];
 * const y = [1, 2, 3, 4];
 * const result = elasticNet(X, y, 0.1, { alpha: 0.5 });
 * result.coefficients; // regularised coefficients
 * result.predict([9, 10]); // predicted value
 * ```
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

  // Precompute column norms (Xs'Xs diagonal)
  const colNorms = new Array(p);
  for (let j = 0; j < p; j++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += Xs[i][j] * Xs[i][j];
    colNorms[j] = s;
  }

  let betaStd: number[];
  let iterations: number;

  if (native) {
    // Fortran-accelerated coordinate descent
    const initBeta = new Array(p).fill(0);
    const initResiduals = Array.from(yc);
    const result = native.elasticNetCd(
      Xs, yc, initBeta, initResiduals, colNorms,
      lambda, alpha, maxIterations, tolerance,
    );
    betaStd = result.beta;
    iterations = result.iterations;
  } else {
    // TypeScript coordinate descent
    betaStd = new Array(p).fill(0);
    const residuals = new Float64Array(yc);
    iterations = 0;

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

/**
 * Validates feature matrix and response vector dimensions.
 *
 * @param X - Feature matrix (n x p).
 * @param y - Response vector (length n).
 * @throws {Error} If dimensions are inconsistent, n < 2, or p = 0.
 */
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

/**
 * Standardises the feature matrix to zero mean and unit variance (population std dev).
 * Also computes the response mean for centering.
 *
 * @param X - Feature matrix (n x p).
 * @param y - Response vector (length n).
 * @returns Object with standardised matrix `Xs`, column means `xMeans`,
 *   column standard deviations `xStds`, and response mean `yMean`.
 */
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

/**
 * Soft-thresholding operator: S(z, gamma) = sign(z) * max(|z| - gamma, 0).
 *
 * This is the proximal operator of the L1 norm, used in coordinate descent
 * for LASSO and Elastic Net.
 *
 * @param z - Input value.
 * @param gamma - Threshold (non-negative).
 * @returns The soft-thresholded value.
 */
function softThreshold(z: number, gamma: number): number {
  if (z > gamma) return z - gamma;
  if (z < -gamma) return z + gamma;
  return 0;
}

/**
 * Scores a model by AIC, BIC, or adjusted-R^2.
 *
 * All values are returned on a "lower is better" scale:
 * - AIC: n * ln(RSS/n) + 2k
 * - BIC: n * ln(RSS/n) + ln(n) * k
 * - adjR2: RSS / (n - k) (a proxy; since TSS is constant across models, minimising this
 *   is equivalent to maximising adjusted-R^2)
 *
 * @param rss - Residual sum of squares.
 * @param n - Number of observations.
 * @param k - Number of estimated parameters (including intercept).
 * @param criterion - The criterion to use.
 * @returns The criterion score (lower is better).
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
 * Fits an OLS (ordinary least squares) model on a subset of features.
 *
 * Constructs the normal equations (X^T X) beta = X^T y for the selected
 * feature columns plus an intercept, and solves via LU decomposition.
 *
 * @param X - Full feature matrix (n x p).
 * @param y - Response vector (length n).
 * @param features - 0-based indices of the feature columns to include.
 * @returns An object with `beta` (intercept at index 0, then feature coefficients)
 *   and `rss` (residual sum of squares).
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
