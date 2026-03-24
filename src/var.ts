import { mean } from "./utils/descriptive";
import { solveLinearSystem, invertMatrix } from "./utils/linalg";

/**
 * Result of fitting a VAR(p) model.
 */
export interface VARResult {
  /** Coefficient matrices for each lag, array of p matrices each [k x k] */
  coefficients: number[][][];
  /** Intercept vector of length k */
  intercept: number[];
  /** Residual matrix [T-p x k] */
  residuals: number[][];
  /** Residual covariance matrix [k x k] */
  sigma: number[][];
  /** Number of lags */
  p: number;
  /** Number of variables */
  k: number;
  /** Number of observations used in estimation (T - p) */
  nobs: number;
  /** Original data */
  data: number[][];
  /** AIC value */
  aic: number;
  /** BIC value */
  bic: number;
}

/**
 * Result of VAR lag order selection.
 */
export interface VARLagSelection {
  /** AIC values for each lag from 1..maxLag */
  aic: number[];
  /** BIC values for each lag from 1..maxLag */
  bic: number[];
  /** Hannan-Quinn values for each lag from 1..maxLag */
  hq: number[];
  /** Optimal lag by AIC */
  aicLag: number;
  /** Optimal lag by BIC */
  bicLag: number;
  /** Optimal lag by HQ */
  hqLag: number;
}

/**
 * Result of a Granger causality test.
 */
export interface GrangerCausalityResult {
  /** F-statistic */
  fStatistic: number;
  /** p-value of the F-test */
  pValue: number;
  /** Degrees of freedom [numerator, denominator] */
  df: [number, number];
  /** Index of the causal variable */
  cause: number;
  /** Index of the effect variable */
  effect: number;
  /** Number of lags used */
  p: number;
}

/**
 * Result of impulse response function computation.
 */
export interface IRFResult {
  /** Orthogonalized impulse responses: irf[h][i][j] = response of variable i to shock in variable j at horizon h */
  irf: number[][][];
  /** Horizon length */
  horizon: number;
}

/**
 * Result of VAR forecasting.
 */
export interface VARForecastResult {
  /** Point forecasts [horizon x k] */
  forecast: number[][];
  /** Forecast horizon */
  horizon: number;
}

/**
 * Result of forecast error variance decomposition.
 */
export interface FEVDResult {
  /** Decomposition: fevd[h][i][j] = proportion of forecast error variance of variable i due to shock j at horizon h */
  fevd: number[][][];
  /** Horizon length */
  horizon: number;
}

/**
 * Result of the Johansen cointegration trace test (simplified).
 */
export interface JohansenResult {
  /** Trace statistics for each rank r = 0, 1, ..., k-1 */
  traceStats: number[];
  /** Critical values at 5% significance (approximate) */
  criticalValues: number[];
  /** Estimated cointegration rank */
  rank: number;
  /** Eigenvalues sorted descending */
  eigenvalues: number[];
}

/**
 * Cholesky decomposition of a symmetric positive-definite matrix.
 * Returns a lower-triangular matrix L such that A = L * L^T.
 *
 * @param A - Symmetric positive-definite matrix [n x n]
 * @returns Lower-triangular matrix L [n x n]
 */
function choleskyDecomposition(A: number[][]): number[][] {
  const n = A.length;
  const L: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));

  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = 0;
      for (let m = 0; m < j; m++) {
        sum += L[i][m] * L[j][m];
      }
      if (i === j) {
        const val = A[i][i] - sum;
        if (val <= 0) {
          throw new Error("Matrix is not positive definite");
        }
        L[i][j] = Math.sqrt(val);
      } else {
        L[i][j] = (A[i][j] - sum) / L[j][j];
      }
    }
  }

  return L;
}

/**
 * Multiply two matrices A [m x n] and B [n x p], returning C [m x p].
 */
function matMul(A: number[][], B: number[][]): number[][] {
  const m = A.length;
  const n = B.length;
  const p = B[0].length;
  const C: number[][] = Array.from({ length: m }, () => new Array(p).fill(0));
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < p; j++) {
      let s = 0;
      for (let l = 0; l < n; l++) {
        s += A[i][l] * B[l][j];
      }
      C[i][j] = s;
    }
  }
  return C;
}

/**
 * Transpose a matrix.
 */
function transpose(A: number[][]): number[][] {
  const m = A.length;
  const n = A[0].length;
  const T: number[][] = Array.from({ length: n }, () => new Array(m).fill(0));
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      T[j][i] = A[i][j];
    }
  }
  return T;
}

/**
 * Create an identity matrix of size n.
 */
function eye(n: number): number[][] {
  const I: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    I[i][i] = 1;
  }
  return I;
}

/**
 * Compute the log-determinant of a square matrix using LU-like decomposition.
 */
function logDet(A: number[][]): number {
  const n = A.length;
  const M = A.map((row) => [...row]);
  let logAbsDet = 0;

  for (let col = 0; col < n; col++) {
    // Partial pivoting
    let maxVal = Math.abs(M[col][col]);
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(M[row][col]) > maxVal) {
        maxVal = Math.abs(M[row][col]);
        maxRow = row;
      }
    }
    if (maxRow !== col) {
      [M[col], M[maxRow]] = [M[maxRow], M[col]];
    }

    if (Math.abs(M[col][col]) < 1e-15) {
      return -Infinity;
    }

    logAbsDet += Math.log(Math.abs(M[col][col]));

    for (let row = col + 1; row < n; row++) {
      const factor = M[row][col] / M[col][col];
      for (let j = col; j < n; j++) {
        M[row][j] -= factor * M[col][j];
      }
    }
  }

  return logAbsDet;
}

/**
 * Fit a Vector Autoregression model of order p using OLS equation-by-equation.
 *
 * @param data - Time series data as a 2D array [T x k] where T is the number of
 *               time observations and k is the number of variables.
 * @param p - Lag order of the VAR model.
 * @returns A VARResult containing estimated coefficients, residuals, and diagnostics.
 */
export function varFit(data: number[][], p: number): VARResult {
  const T = data.length;
  const k = data[0].length;
  const nobs = T - p;

  if (nobs <= 0) {
    throw new Error(`Not enough observations (T=${T}) for lag order p=${p}`);
  }

  // Build the regressor matrix Z [nobs x (k*p + 1)] with intercept as first column
  const nRegressors = k * p + 1;
  const Z: number[][] = [];
  const Y: number[][] = [];

  for (let t = p; t < T; t++) {
    const row: number[] = [1]; // intercept
    for (let lag = 1; lag <= p; lag++) {
      for (let j = 0; j < k; j++) {
        row.push(data[t - lag][j]);
      }
    }
    Z.push(row);
    Y.push(data[t]);
  }

  // OLS: B = (Z'Z)^{-1} Z'Y for each equation
  const Zt = transpose(Z);
  const ZtZ = matMul(Zt, Z);
  const ZtZinv = invertMatrix(ZtZ)!;
  const ZtY = matMul(Zt, Y);
  const B = matMul(ZtZinv, ZtY); // [nRegressors x k]

  // Extract intercept and coefficient matrices
  const intercept: number[] = [];
  for (let j = 0; j < k; j++) {
    intercept.push(B[0][j]);
  }

  const coefficients: number[][][] = [];
  for (let lag = 0; lag < p; lag++) {
    const A: number[][] = Array.from({ length: k }, () => new Array(k).fill(0));
    for (let i = 0; i < k; i++) {
      for (let j = 0; j < k; j++) {
        A[i][j] = B[1 + lag * k + i][j];
      }
    }
    // Transpose so that A[i][j] = effect of variable j (lagged) on variable i
    coefficients.push(transpose(A));
  }

  // Compute residuals
  const Yhat = matMul(Z, B);
  const residuals: number[][] = [];
  for (let t = 0; t < nobs; t++) {
    const row: number[] = [];
    for (let j = 0; j < k; j++) {
      row.push(Y[t][j] - Yhat[t][j]);
    }
    residuals.push(row);
  }

  // Residual covariance matrix
  const sigma: number[][] = Array.from({ length: k }, () => new Array(k).fill(0));
  for (let i = 0; i < k; i++) {
    for (let j = 0; j < k; j++) {
      let s = 0;
      for (let t = 0; t < nobs; t++) {
        s += residuals[t][i] * residuals[t][j];
      }
      sigma[i][j] = s / nobs;
    }
  }

  // Information criteria
  const logDetSigma = logDet(sigma);
  const aic = logDetSigma + (2 * k * k * p + 2 * k) / nobs;
  const bic = logDetSigma + (Math.log(nobs) * (k * k * p + k)) / nobs;

  return {
    coefficients,
    intercept,
    residuals,
    sigma,
    p,
    k,
    nobs,
    data,
    aic,
    bic,
  };
}

/**
 * Select the optimal lag order for a VAR model using information criteria (AIC, BIC, HQ).
 *
 * @param data - Time series data [T x k].
 * @param maxLag - Maximum lag order to evaluate.
 * @returns A VARLagSelection with criterion values and optimal lags.
 */
export function varLagSelect(data: number[][], maxLag: number): VARLagSelection {
  const T = data.length;
  const k = data[0].length;

  const aicValues: number[] = [];
  const bicValues: number[] = [];
  const hqValues: number[] = [];

  for (let lag = 1; lag <= maxLag; lag++) {
    const nobs = T - lag;
    if (nobs <= k * lag + 1) {
      aicValues.push(Infinity);
      bicValues.push(Infinity);
      hqValues.push(Infinity);
      continue;
    }

    const result = varFit(data, lag);
    const logDetSigma = logDet(result.sigma);
    const nParams = k * k * lag + k; // coefficients + intercepts

    const aic = logDetSigma + (2 * nParams) / nobs;
    const bic = logDetSigma + (Math.log(nobs) * nParams) / nobs;
    const hq = logDetSigma + (2 * Math.log(Math.log(nobs)) * nParams) / nobs;

    aicValues.push(aic);
    bicValues.push(bic);
    hqValues.push(hq);
  }

  const aicLag = aicValues.indexOf(Math.min(...aicValues)) + 1;
  const bicLag = bicValues.indexOf(Math.min(...bicValues)) + 1;
  const hqLag = hqValues.indexOf(Math.min(...hqValues)) + 1;

  return {
    aic: aicValues,
    bic: bicValues,
    hq: hqValues,
    aicLag,
    bicLag,
    hqLag,
  };
}

/**
 * Perform a Granger causality test to determine whether a causal variable
 * Granger-causes an effect variable within a VAR(p) framework using an F-test.
 *
 * @param data - Time series data [T x k].
 * @param p - Lag order.
 * @param cause - Index of the causal variable.
 * @param effect - Index of the effect variable.
 * @returns A GrangerCausalityResult with F-statistic and p-value.
 */
export function grangerCausality(
  data: number[][],
  p: number,
  cause: number,
  effect: number
): GrangerCausalityResult {
  const T = data.length;
  const k = data[0].length;
  const nobs = T - p;

  // Fit unrestricted model
  const unrestricted = varFit(data, p);
  const residUnrestricted = unrestricted.residuals.map((r) => r[effect]);
  const rssUnrestricted = residUnrestricted.reduce((s, r) => s + r * r, 0);

  // Fit restricted model: exclude the cause variable's lags from the effect equation
  // Build regressor matrix without cause variable lags
  const ZRestricted: number[][] = [];
  const YRestricted: number[] = [];

  for (let t = p; t < T; t++) {
    const row: number[] = [1]; // intercept
    for (let lag = 1; lag <= p; lag++) {
      for (let j = 0; j < k; j++) {
        if (j === cause) continue;
        row.push(data[t - lag][j]);
      }
    }
    ZRestricted.push(row);
    YRestricted.push(data[t][effect]);
  }

  // OLS for restricted model
  const ZRt = transpose(ZRestricted);
  const ZRtZR = matMul(ZRt, ZRestricted);
  const ZRtZRinv = invertMatrix(ZRtZR)!;
  const ZRtY: number[][] = ZRt.map((row) =>
    [row.reduce((s, val, i) => s + val * YRestricted[i], 0)]
  );
  const bRestricted = matMul(ZRtZRinv, ZRtY);

  // Compute restricted residuals
  let rssRestricted = 0;
  for (let t = 0; t < nobs; t++) {
    let yHat = 0;
    for (let j = 0; j < ZRestricted[t].length; j++) {
      yHat += ZRestricted[t][j] * bRestricted[j][0];
    }
    const resid = YRestricted[t] - yHat;
    rssRestricted += resid * resid;
  }

  // F-test
  const dfNum = p; // number of restrictions (p lags of the cause variable)
  const dfDen = nobs - k * p - 1; // denominator df
  const fStatistic = ((rssRestricted - rssUnrestricted) / dfNum) / (rssUnrestricted / dfDen);

  // Approximate p-value using the F-distribution via the regularized incomplete beta function
  const pValue = fDistPValue(fStatistic, dfNum, dfDen);

  return {
    fStatistic,
    pValue,
    df: [dfNum, dfDen],
    cause,
    effect,
    p,
  };
}

/**
 * Approximate the upper-tail p-value of the F-distribution using
 * the regularized incomplete beta function.
 */
function fDistPValue(f: number, d1: number, d2: number): number {
  if (f <= 0) return 1;
  const x = d2 / (d2 + d1 * f);
  return regularizedBeta(x, d2 / 2, d1 / 2);
}

/**
 * Regularized incomplete beta function I_x(a, b) using a continued fraction expansion.
 */
function regularizedBeta(x: number, a: number, b: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;

  const lnBeta = lnGamma(a) + lnGamma(b) - lnGamma(a + b);
  const front = Math.exp(Math.log(x) * a + Math.log(1 - x) * b - lnBeta);

  // Lentz's continued fraction
  if (x < (a + 1) / (a + b + 2)) {
    return (front * betaCF(x, a, b)) / a;
  } else {
    return 1 - (front * betaCF(1 - x, b, a)) / b;
  }
}

/**
 * Continued fraction for the incomplete beta function.
 */
function betaCF(x: number, a: number, b: number): number {
  const maxIter = 200;
  const eps = 1e-14;
  let qab = a + b;
  let qap = a + 1;
  let qam = a - 1;
  let c = 1;
  let d = 1 - (qab * x) / qap;
  if (Math.abs(d) < 1e-30) d = 1e-30;
  d = 1 / d;
  let h = d;

  for (let m = 1; m <= maxIter; m++) {
    const m2 = 2 * m;

    // Even step
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < 1e-30) d = 1e-30;
    c = 1 + aa / c;
    if (Math.abs(c) < 1e-30) c = 1e-30;
    d = 1 / d;
    h *= d * c;

    // Odd step
    aa = -((a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < 1e-30) d = 1e-30;
    c = 1 + aa / c;
    if (Math.abs(c) < 1e-30) c = 1e-30;
    d = 1 / d;
    const del = d * c;
    h *= del;

    if (Math.abs(del - 1) < eps) break;
  }

  return h;
}

/**
 * Log-gamma function using Stirling's approximation (Lanczos).
 */
function lnGamma(z: number): number {
  const g = 7;
  const coef = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];

  if (z < 0.5) {
    return (
      Math.log(Math.PI / Math.sin(Math.PI * z)) - lnGamma(1 - z)
    );
  }

  z -= 1;
  let x = coef[0];
  for (let i = 1; i < g + 2; i++) {
    x += coef[i] / (z + i);
  }
  const t = z + g + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}

/**
 * Compute orthogonalized impulse response functions via Cholesky decomposition
 * of the residual covariance matrix.
 *
 * @param result - A fitted VARResult.
 * @param horizon - Number of periods for the impulse response.
 * @returns An IRFResult containing the orthogonalized impulse response matrices.
 */
export function impulseResponse(result: VARResult, horizon: number): IRFResult {
  const { coefficients, sigma, k, p } = result;

  // Cholesky decomposition of sigma
  const P = choleskyDecomposition(sigma);

  // Compute MA coefficient matrices Phi[h] recursively:
  // Phi[0] = I_k
  // Phi[h] = sum_{j=1}^{min(h,p)} A_j * Phi[h-j]
  const Phi: number[][][] = [];
  Phi[0] = eye(k);

  for (let h = 1; h <= horizon; h++) {
    const PhiH: number[][] = Array.from({ length: k }, () => new Array(k).fill(0));
    for (let j = 1; j <= Math.min(h, p); j++) {
      const prod = matMul(coefficients[j - 1], Phi[h - j]);
      for (let r = 0; r < k; r++) {
        for (let c = 0; c < k; c++) {
          PhiH[r][c] += prod[r][c];
        }
      }
    }
    Phi[h] = PhiH;
  }

  // Orthogonalized IRF: Theta[h] = Phi[h] * P
  const irf: number[][][] = [];
  for (let h = 0; h <= horizon; h++) {
    irf.push(matMul(Phi[h], P));
  }

  return { irf, horizon };
}

/**
 * Compute forecast error variance decomposition (FEVD).
 *
 * @param result - A fitted VARResult.
 * @param horizon - Number of periods for the decomposition.
 * @returns A FEVDResult with the proportion of variance attributed to each shock.
 */
export function varianceDecomposition(result: VARResult, horizon: number): FEVDResult {
  const { k } = result;
  const { irf } = impulseResponse(result, horizon);

  const fevd: number[][][] = [];

  for (let h = 0; h <= horizon; h++) {
    // Cumulative squared impulse responses up to horizon h
    const mse: number[][] = Array.from({ length: k }, () => new Array(k).fill(0));

    for (let s = 0; s <= h; s++) {
      for (let i = 0; i < k; i++) {
        for (let j = 0; j < k; j++) {
          mse[i][j] += irf[s][i][j] * irf[s][i][j];
        }
      }
    }

    // Normalize: fevd[h][i][j] = mse[i][j] / sum_j(mse[i][j])
    const decomp: number[][] = Array.from({ length: k }, () => new Array(k).fill(0));
    for (let i = 0; i < k; i++) {
      let total = 0;
      for (let j = 0; j < k; j++) {
        total += mse[i][j];
      }
      for (let j = 0; j < k; j++) {
        decomp[i][j] = total > 0 ? mse[i][j] / total : 0;
      }
    }

    fevd.push(decomp);
  }

  return { fevd, horizon };
}

/**
 * Generate point forecasts from a fitted VAR model.
 *
 * @param result - A fitted VARResult.
 * @param horizon - Number of periods ahead to forecast.
 * @returns A VARForecastResult with point forecasts.
 */
export function varForecast(result: VARResult, horizon: number): VARForecastResult {
  const { coefficients, intercept, data, p, k } = result;
  const T = data.length;

  // Build a buffer of the last max(p, horizon+p) observations
  const buffer: number[][] = data.map((row) => [...row]);

  const forecast: number[][] = [];

  for (let h = 0; h < horizon; h++) {
    const yNew: number[] = [...intercept];
    const currentT = T + h;

    for (let lag = 1; lag <= p; lag++) {
      const pastIdx = currentT - lag;
      const pastObs = pastIdx < T ? data[pastIdx] : forecast[pastIdx - T];
      for (let i = 0; i < k; i++) {
        for (let j = 0; j < k; j++) {
          yNew[i] += coefficients[lag - 1][i][j] * pastObs[j];
        }
      }
    }

    forecast.push(yNew);
  }

  return { forecast, horizon };
}

/**
 * Perform a simplified Johansen trace test for cointegration.
 * Uses a reduced-rank regression approach to estimate eigenvalues and
 * compare trace statistics against approximate critical values.
 *
 * @param data - Time series data [T x k].
 * @param p - Lag order for the VECM representation.
 * @returns A JohansenResult with trace statistics, critical values, and estimated rank.
 */
export function johansenTest(data: number[][], p: number): JohansenResult {
  const T = data.length;
  const k = data[0].length;

  // Compute first differences
  const dY: number[][] = [];
  for (let t = 1; t < T; t++) {
    const row: number[] = [];
    for (let j = 0; j < k; j++) {
      row.push(data[t][j] - data[t - 1][j]);
    }
    dY.push(row);
  }

  const nobs = dY.length - p;
  if (nobs <= 0) {
    throw new Error("Not enough observations for Johansen test");
  }

  // Build regressors: lagged differences and lagged levels
  const Z0: number[][] = []; // dY_t
  const Z1: number[][] = []; // Y_{t-1} (levels)
  const Z2: number[][] = []; // lagged dY

  for (let t = p; t < dY.length; t++) {
    Z0.push(dY[t]);
    Z1.push(data[t]); // Y_{t} in levels (which is Y at index t in original data, i.e., lagged by 1 vs dY_t)

    const lagRow: number[] = [];
    for (let lag = 1; lag <= p - 1; lag++) {
      for (let j = 0; j < k; j++) {
        lagRow.push(dY[t - lag][j]);
      }
    }
    if (lagRow.length === 0) {
      lagRow.push(1); // intercept only if no lagged differences
    }
    Z2.push(lagRow);
  }

  // Concentrate out Z2 from Z0 and Z1 by OLS regression
  const R0 = concentrateOut(Z0, Z2);
  const R1 = concentrateOut(Z1, Z2);

  // Compute moment matrices
  const S00 = momentMatrix(R0, R0, nobs);
  const S01 = momentMatrix(R0, R1, nobs);
  const S10 = momentMatrix(R1, R0, nobs);
  const S11 = momentMatrix(R1, R1, nobs);

  // Solve the generalized eigenvalue problem:
  // |lambda * S11 - S10 * S00^{-1} * S01| = 0
  const S00inv = invertMatrix(S00)!;
  const M = matMul(matMul(S10, S00inv), S01);
  const S11inv = invertMatrix(S11)!;
  const eigenMatrix = matMul(S11inv, M);

  // Power iteration to find eigenvalues (simplified)
  const eigenvalues = approximateEigenvalues(eigenMatrix, k);
  eigenvalues.sort((a, b) => b - a);

  // Trace statistics
  const traceStats: number[] = [];
  for (let r = 0; r < k; r++) {
    let stat = 0;
    for (let i = r; i < k; i++) {
      const lambda = Math.max(eigenvalues[i], 1e-15);
      stat -= nobs * Math.log(1 - Math.min(lambda, 0.9999));
    }
    traceStats.push(stat);
  }

  // Approximate critical values at 5% for trace test (from MacKinnon et al.)
  // These are rough approximations for k-r unrestricted variables
  const criticalValues: number[] = [];
  for (let r = 0; r < k; r++) {
    const dim = k - r;
    // Approximate 5% critical values based on dimension
    const cv = approximateTraceCriticalValue(dim);
    criticalValues.push(cv);
  }

  // Determine rank: largest r such that trace stat > critical value for all r' < r
  let rank = 0;
  for (let r = 0; r < k; r++) {
    if (traceStats[r] > criticalValues[r]) {
      rank = r + 1;
    } else {
      break;
    }
  }

  return {
    traceStats,
    criticalValues,
    rank,
    eigenvalues,
  };
}

/**
 * Concentrate out the effect of Z2 from Z by OLS regression,
 * returning the residuals.
 */
function concentrateOut(Z: number[][], Z2: number[][]): number[][] {
  const n = Z.length;
  const kz = Z[0].length;

  const Z2t = transpose(Z2);
  const Z2tZ2 = matMul(Z2t, Z2);
  let Z2tZ2inv: number[][];
  try {
    Z2tZ2inv = invertMatrix(Z2tZ2)!;
  } catch {
    return Z; // If Z2 is singular, return original
  }

  const Z2tZ = matMul(Z2t, Z);
  const B = matMul(Z2tZ2inv, Z2tZ);
  const Zhat = matMul(Z2, B);

  const residuals: number[][] = [];
  for (let t = 0; t < n; t++) {
    const row: number[] = [];
    for (let j = 0; j < kz; j++) {
      row.push(Z[t][j] - Zhat[t][j]);
    }
    residuals.push(row);
  }

  return residuals;
}

/**
 * Compute the moment matrix (1/n) * A' * B.
 */
function momentMatrix(A: number[][], B: number[][], n: number): number[][] {
  const At = transpose(A);
  const M = matMul(At, B);
  const rows = M.length;
  const cols = M[0].length;
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      M[i][j] /= n;
    }
  }
  return M;
}

/**
 * Approximate eigenvalues of a square matrix using QR-like iteration (simplified).
 * Returns an array of approximate eigenvalues.
 */
function approximateEigenvalues(A: number[][], n: number): number[] {
  let M = A.map((row) => [...row]);
  const maxIter = 100;

  for (let iter = 0; iter < maxIter; iter++) {
    // QR decomposition via Gram-Schmidt
    const { Q, R } = qrDecomposition(M);
    M = matMul(R, Q);
  }

  // Eigenvalues are on the diagonal
  const eigenvalues: number[] = [];
  for (let i = 0; i < n; i++) {
    eigenvalues.push(M[i][i]);
  }
  return eigenvalues;
}

/**
 * QR decomposition via classical Gram-Schmidt.
 */
function qrDecomposition(A: number[][]): { Q: number[][]; R: number[][] } {
  const m = A.length;
  const n = A[0].length;
  const Q: number[][] = Array.from({ length: m }, () => new Array(n).fill(0));
  const R: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));

  for (let j = 0; j < n; j++) {
    // Copy column j of A
    const v: number[] = [];
    for (let i = 0; i < m; i++) {
      v.push(A[i][j]);
    }

    // Subtract projections
    for (let i = 0; i < j; i++) {
      let dot = 0;
      for (let r = 0; r < m; r++) {
        dot += Q[r][i] * v[r];
      }
      R[i][j] = dot;
      for (let r = 0; r < m; r++) {
        v[r] -= dot * Q[r][i];
      }
    }

    // Normalize
    let norm = 0;
    for (let i = 0; i < m; i++) {
      norm += v[i] * v[i];
    }
    norm = Math.sqrt(norm);
    R[j][j] = norm;

    if (norm > 1e-15) {
      for (let i = 0; i < m; i++) {
        Q[i][j] = v[i] / norm;
      }
    }
  }

  return { Q, R };
}

/**
 * Approximate 5% critical values for the Johansen trace test
 * based on the number of unrestricted dimensions.
 */
function approximateTraceCriticalValue(dim: number): number {
  // Approximate critical values from Osterwald-Lenum (1992) tables
  const table: Record<number, number> = {
    1: 3.84,
    2: 15.41,
    3: 29.68,
    4: 47.21,
    5: 68.52,
    6: 94.15,
    7: 124.24,
    8: 156.0,
    9: 192.89,
    10: 232.49,
  };

  if (dim in table) {
    return table[dim];
  }

  // Extrapolate for larger dimensions
  return 3.84 + (dim - 1) * 15.0;
}
