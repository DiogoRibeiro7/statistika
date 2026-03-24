/**
 * GARCH (Generalized Autoregressive Conditional Heteroscedasticity) module.
 *
 * Provides estimation for ARCH, GARCH, EGARCH, and GJR-GARCH models
 * using conditional maximum likelihood estimation with Nelder-Mead
 * simplex optimization.
 *
 * @module garch
 */

import { mean, variance } from "./utils/descriptive";
import {
  hasNativeGarch,
  garch11Loglik as nativeGarch11,
  garchPqLoglik as nativeGarchPq,
  gjrGarch11Loglik as nativeGjr,
  egarch11Loglik as nativeEgarch,
  garch11Forecast as nativeGarch11Forecast,
} from "./utils/native-garch";

/**
 * Result of a standard GARCH(p,q) model estimation.
 */
export interface GARCHResult {
  /** Intercept (omega) of the variance equation */
  omega: number;
  /** ARCH coefficients (alpha_1, ..., alpha_q) */
  alpha: number[];
  /** GARCH coefficients (beta_1, ..., beta_p) */
  beta: number[];
  /** Maximized log-likelihood value */
  logLikelihood: number;
  /** Conditional variance series */
  conditionalVariance: number[];
  /** Standardized residuals (eps_t / sqrt(sigma2_t)) */
  standardizedResiduals: number[];
  /** Akaike Information Criterion */
  aic: number;
  /** Bayesian Information Criterion */
  bic: number;
  /** Number of observations */
  nObs: number;
  /** Convergence flag */
  converged: boolean;
}

/**
 * Result of an EGARCH (Exponential GARCH) model estimation.
 */
export interface EGARCHResult {
  /** Intercept (omega) of the log-variance equation */
  omega: number;
  /** ARCH coefficients (alpha) */
  alpha: number[];
  /** GARCH coefficients (beta) for log-variance persistence */
  beta: number[];
  /** Leverage/asymmetry coefficients (gamma) */
  gamma: number[];
  /** Maximized log-likelihood value */
  logLikelihood: number;
  /** Conditional variance series */
  conditionalVariance: number[];
  /** Standardized residuals */
  standardizedResiduals: number[];
  /** Akaike Information Criterion */
  aic: number;
  /** Bayesian Information Criterion */
  bic: number;
  /** Number of observations */
  nObs: number;
  /** Convergence flag */
  converged: boolean;
}

/**
 * Result of a GJR-GARCH (Glosten-Jagannathan-Runkle) model estimation.
 */
export interface GJRGARCHResult {
  /** Intercept (omega) of the variance equation */
  omega: number;
  /** ARCH coefficients (alpha) */
  alpha: number[];
  /** GARCH coefficients (beta) */
  beta: number[];
  /** Leverage/asymmetry coefficients (gamma) for negative shocks */
  gamma: number[];
  /** Maximized log-likelihood value */
  logLikelihood: number;
  /** Conditional variance series */
  conditionalVariance: number[];
  /** Standardized residuals */
  standardizedResiduals: number[];
  /** Akaike Information Criterion */
  aic: number;
  /** Bayesian Information Criterion */
  bic: number;
  /** Number of observations */
  nObs: number;
  /** Convergence flag */
  converged: boolean;
}

/**
 * Forecast output from a GARCH model.
 */
export interface GARCHForecast {
  /** Forecasted conditional variances */
  variance: number[];
  /** Forecasted conditional standard deviations */
  standardDeviation: number[];
  /** Number of steps ahead */
  horizon: number;
}

/**
 * Result of the ARCH-LM (Lagrange Multiplier) test for conditional
 * heteroscedasticity.
 */
export interface ARCHLMTestResult {
  /** Test statistic (T * R^2) */
  statistic: number;
  /** p-value from chi-squared distribution */
  pValue: number;
  /** Number of lags used in the test */
  lags: number;
  /** R-squared from the auxiliary regression */
  rSquared: number;
}

/**
 * Diagnostic statistics for a fitted GARCH model.
 */
export interface GARCHDiagnostics {
  /** Ljung-Box Q statistic on standardized residuals */
  ljungBoxResiduals: number;
  /** Ljung-Box Q statistic on squared standardized residuals */
  ljungBoxSquaredResiduals: number;
  /** Mean of standardized residuals */
  meanStdResiduals: number;
  /** Variance of standardized residuals */
  varianceStdResiduals: number;
  /** Skewness of standardized residuals */
  skewness: number;
  /** Excess kurtosis of standardized residuals */
  kurtosis: number;
  /** ARCH-LM test on standardized residuals */
  archLMTest: ARCHLMTestResult;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

const LOG2PI = Math.log(2 * Math.PI);

/**
 * Compute the incomplete gamma function ratio P(a, x) using a series
 * expansion. Used for chi-squared p-value computation.
 */
function lowerIncompleteGamma(a: number, x: number): number {
  if (x < 0) return 0;
  if (x === 0) return 0;

  let sum = 0;
  let term = 1 / a;
  sum = term;
  for (let n = 1; n < 200; n++) {
    term *= x / (a + n);
    sum += term;
    if (Math.abs(term) < 1e-14 * Math.abs(sum)) break;
  }
  return sum * Math.exp(-x + a * Math.log(x) - lnGamma(a));
}

/**
 * Log-gamma function via Lanczos approximation.
 */
function lnGamma(z: number): number {
  const g = 7;
  const c = [
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
  let x = c[0];
  for (let i = 1; i < g + 2; i++) {
    x += c[i] / (z + i);
  }
  const t = z + g + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}

/**
 * Chi-squared survival function (1 - CDF) for computing p-values.
 */
function chiSquaredPValue(x: number, df: number): number {
  if (x <= 0) return 1;
  const p = lowerIncompleteGamma(df / 2, x / 2);
  return 1 - p;
}

/**
 * Nelder-Mead simplex optimization (minimization).
 *
 * @param fn - Objective function to minimize
 * @param x0 - Initial parameter vector
 * @param maxIter - Maximum iterations
 * @param tol - Function value tolerance for convergence
 * @returns Optimized parameter vector and convergence flag
 */
function nelderMead(
  fn: (x: number[]) => number,
  x0: number[],
  maxIter = 5000,
  tol = 1e-10,
): { params: number[]; converged: boolean } {
  const n = x0.length;
  const alpha = 1.0;
  const gammaCoeff = 2.0;
  const rho = 0.5;
  const sigma = 0.5;

  // Build initial simplex
  const simplex: { point: number[]; value: number }[] = [];
  simplex.push({ point: [...x0], value: fn(x0) });
  for (let i = 0; i < n; i++) {
    const p = [...x0];
    p[i] += Math.max(Math.abs(p[i]) * 0.05, 1e-4);
    simplex.push({ point: p, value: fn(p) });
  }

  let converged = false;
  for (let iter = 0; iter < maxIter; iter++) {
    // Sort by function value
    simplex.sort((a, b) => a.value - b.value);

    // Check convergence
    const range = Math.abs(simplex[n].value - simplex[0].value);
    if (range < tol) {
      converged = true;
      break;
    }

    // Centroid of the best n points
    const centroid = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        centroid[j] += simplex[i].point[j];
      }
    }
    for (let j = 0; j < n; j++) centroid[j] /= n;

    // Reflection
    const worst = simplex[n];
    const reflected = centroid.map((c, j) => c + alpha * (c - worst.point[j]));
    const fr = fn(reflected);

    if (fr < simplex[0].value) {
      // Expansion
      const expanded = centroid.map(
        (c, j) => c + gammaCoeff * (reflected[j] - c),
      );
      const fe = fn(expanded);
      if (fe < fr) {
        simplex[n] = { point: expanded, value: fe };
      } else {
        simplex[n] = { point: reflected, value: fr };
      }
    } else if (fr < simplex[n - 1].value) {
      simplex[n] = { point: reflected, value: fr };
    } else {
      // Contraction
      const contracted = centroid.map(
        (c, j) => c + rho * (worst.point[j] - c),
      );
      const fc = fn(contracted);
      if (fc < worst.value) {
        simplex[n] = { point: contracted, value: fc };
      } else {
        // Shrink
        for (let i = 1; i <= n; i++) {
          for (let j = 0; j < n; j++) {
            simplex[i].point[j] =
              simplex[0].point[j] +
              sigma * (simplex[i].point[j] - simplex[0].point[j]);
          }
          simplex[i].value = fn(simplex[i].point);
        }
      }
    }
  }

  simplex.sort((a, b) => a.value - b.value);
  return { params: simplex[0].point, converged };
}

/**
 * Compute GARCH(p,q) conditional variances given parameters and residuals.
 *
 * @param eps - Residual (demeaned) series
 * @param omega - Intercept
 * @param alpha - ARCH coefficients
 * @param beta - GARCH coefficients
 * @returns Conditional variance series
 */
function computeGarchVariances(
  eps: number[],
  omega: number,
  alpha: number[],
  beta: number[],
): number[] {
  const T = eps.length;
  const q = alpha.length;
  const p = beta.length;
  const sigma2 = new Array<number>(T);
  const uncondVar = variance(eps);

  // Initialize with unconditional variance
  for (let t = 0; t < Math.max(p, q); t++) {
    sigma2[t] = uncondVar;
  }

  for (let t = Math.max(p, q); t < T; t++) {
    let s = omega;
    for (let i = 0; i < q; i++) {
      s += alpha[i] * eps[t - 1 - i] * eps[t - 1 - i];
    }
    for (let j = 0; j < p; j++) {
      s += beta[j] * sigma2[t - 1 - j];
    }
    sigma2[t] = Math.max(s, 1e-12);
  }

  return sigma2;
}

/**
 * Gaussian log-likelihood for GARCH models.
 */
function garchLogLikelihood(eps: number[], sigma2: number[]): number {
  const T = eps.length;
  let ll = 0;
  for (let t = 0; t < T; t++) {
    ll += -0.5 * (LOG2PI + Math.log(sigma2[t]) + (eps[t] * eps[t]) / sigma2[t]);
  }
  return ll;
}

/**
 * Compute information criteria.
 */
function computeIC(
  logLik: number,
  nParams: number,
  nObs: number,
): { aic: number; bic: number } {
  return {
    aic: -2 * logLik + 2 * nParams,
    bic: -2 * logLik + nParams * Math.log(nObs),
  };
}

/**
 * Demean a series by removing the sample mean.
 */
function demean(x: number[]): number[] {
  const m = mean(x);
  return x.map((v) => v - m);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Fit an ARCH(q) model to a time series via conditional MLE.
 *
 * An ARCH(q) model is a special case of GARCH(0,q):
 *   sigma2_t = omega + alpha_1 * eps_{t-1}^2 + ... + alpha_q * eps_{t-q}^2
 *
 * @param data - Input time series (returns or residuals)
 * @param q - ARCH order (number of lagged squared residuals)
 * @returns Fitted ARCH model result (as a GARCHResult with empty beta)
 *
 * @example
 * ```ts
 * const result = archFit(returns, 1);
 * console.log(result.omega, result.alpha);
 * ```
 */
export function archFit(data: number[], q = 1): GARCHResult {
  return garchFit(data, 0, q);
}

/**
 * Fit a GARCH(p,q) model to a time series via conditional MLE.
 *
 * The conditional variance follows:
 *   sigma2_t = omega + sum_{i=1}^{q} alpha_i * eps_{t-i}^2
 *                     + sum_{j=1}^{p} beta_j * sigma2_{t-j}
 *
 * Parameters are estimated by maximizing the Gaussian log-likelihood:
 *   L = sum_t -0.5 * (log(2*pi) + log(sigma2_t) + eps_t^2 / sigma2_t)
 *
 * Optimization uses Nelder-Mead simplex method with positivity and
 * stationarity constraints enforced via penalty.
 *
 * @param data - Input time series (returns or residuals)
 * @param p - GARCH order (number of lagged conditional variances), default 1
 * @param q - ARCH order (number of lagged squared residuals), default 1
 * @returns Fitted GARCH model result
 *
 * @example
 * ```ts
 * const result = garchFit(returns, 1, 1);
 * console.log(result.omega, result.alpha, result.beta);
 * ```
 */
export function garchFit(data: number[], p = 1, q = 1): GARCHResult {
  const eps = demean(data);
  const T = eps.length;
  const uncondVar = variance(eps);

  // Initial parameter guess
  const alphaInit = new Array(q).fill(0.05);
  const betaInit = new Array(p).fill(0.9 / Math.max(p, 1));
  const omegaInit =
    uncondVar *
    (1 - alphaInit.reduce((s, a) => s + a, 0) - betaInit.reduce((s, b) => s + b, 0));

  const x0 = [Math.max(omegaInit, 1e-6), ...alphaInit, ...betaInit];

  const objective = (params: number[]): number => {
    const omega = params[0];
    const alpha = params.slice(1, 1 + q);
    const beta = params.slice(1 + q, 1 + q + p);

    // Enforce positivity
    if (omega <= 0) return 1e12;
    for (let i = 0; i < q; i++) if (alpha[i] < 0) return 1e12;
    for (let j = 0; j < p; j++) if (beta[j] < 0) return 1e12;

    // Stationarity constraint: sum(alpha) + sum(beta) < 1
    const persistence =
      alpha.reduce((s, a) => s + a, 0) + beta.reduce((s, b) => s + b, 0);
    if (persistence >= 1) return 1e12;

    // Use Fortran-accelerated variance recursion + log-likelihood when available
    if (hasNativeGarch) {
      if (p === 1 && q === 1) {
        const result = nativeGarch11(eps, omega, alpha[0], beta[0]);
        return -result.logLikelihood;
      }
      const result = nativeGarchPq(eps, omega, alpha, beta);
      return -result.logLikelihood;
    }

    const sigma2 = computeGarchVariances(eps, omega, alpha, beta);
    return -garchLogLikelihood(eps, sigma2);
  };

  const { params, converged } = nelderMead(objective, x0);

  const omega = params[0];
  const alpha = params.slice(1, 1 + q);
  const beta = params.slice(1 + q, 1 + q + p);

  let sigma2: number[];
  let ll: number;
  if (hasNativeGarch && p === 1 && q === 1) {
    const result = nativeGarch11(eps, omega, alpha[0], beta[0]);
    sigma2 = result.sigma2;
    ll = result.logLikelihood;
  } else if (hasNativeGarch) {
    const result = nativeGarchPq(eps, omega, alpha, beta);
    sigma2 = result.sigma2;
    ll = result.logLikelihood;
  } else {
    sigma2 = computeGarchVariances(eps, omega, alpha, beta);
    ll = garchLogLikelihood(eps, sigma2);
  }

  const nParams = 1 + q + p;
  const { aic, bic } = computeIC(ll, nParams, T);

  const stdResid = eps.map((e, t) => e / Math.sqrt(sigma2[t]));

  return {
    omega,
    alpha,
    beta,
    logLikelihood: ll,
    conditionalVariance: sigma2,
    standardizedResiduals: stdResid,
    aic,
    bic,
    nObs: T,
    converged,
  };
}

/**
 * Fit an EGARCH(p,q) model to a time series via conditional MLE.
 *
 * The EGARCH model of Nelson (1991) specifies the log of conditional variance:
 *   log(sigma2_t) = omega + sum_{i=1}^{q} [alpha_i * |z_{t-i}| + gamma_i * z_{t-i}]
 *                         + sum_{j=1}^{p} beta_j * log(sigma2_{t-j})
 *
 * where z_t = eps_t / sigma_t are standardized residuals.
 *
 * The exponential formulation ensures positivity of conditional variance
 * without parameter constraints, and the gamma terms capture asymmetric
 * (leverage) effects.
 *
 * @param data - Input time series
 * @param p - GARCH order, default 1
 * @param q - ARCH order, default 1
 * @returns Fitted EGARCH model result
 *
 * @example
 * ```ts
 * const result = egarchFit(returns, 1, 1);
 * console.log(result.gamma); // leverage coefficients
 * ```
 */
export function egarchFit(data: number[], p = 1, q = 1): EGARCHResult {
  const eps = demean(data);
  const T = eps.length;
  const uncondVar = variance(eps);
  const logUncondVar = Math.log(uncondVar);

  // Initial guesses: omega ~ log(uncondVar)*(1 - sum(beta)), alpha ~ 0.1, gamma ~ -0.05, beta ~ 0.9
  const betaInit = new Array(p).fill(0.85 / Math.max(p, 1));
  const omegaInit =
    logUncondVar * (1 - betaInit.reduce((s, b) => s + b, 0));
  const alphaInit = new Array(q).fill(0.1);
  const gammaInit = new Array(q).fill(-0.05);

  // Pack: [omega, alpha_1..q, gamma_1..q, beta_1..p]
  const x0 = [omegaInit, ...alphaInit, ...gammaInit, ...betaInit];

  const computeVariances = (params: number[]): number[] | null => {
    const omega = params[0];
    const alpha = params.slice(1, 1 + q);
    const gamma = params.slice(1 + q, 1 + 2 * q);
    const beta = params.slice(1 + 2 * q, 1 + 2 * q + p);

    const logSigma2 = new Array<number>(T);
    const sigma2 = new Array<number>(T);
    const startIdx = Math.max(p, q);

    for (let t = 0; t < startIdx; t++) {
      logSigma2[t] = logUncondVar;
      sigma2[t] = uncondVar;
    }

    for (let t = startIdx; t < T; t++) {
      let ls = omega;
      for (let i = 0; i < q; i++) {
        const z = eps[t - 1 - i] / Math.sqrt(sigma2[t - 1 - i]);
        ls += alpha[i] * Math.abs(z) + gamma[i] * z;
      }
      for (let j = 0; j < p; j++) {
        ls += beta[j] * logSigma2[t - 1 - j];
      }
      if (!isFinite(ls) || ls > 50 || ls < -50) return null;
      logSigma2[t] = ls;
      sigma2[t] = Math.exp(ls);
    }
    return sigma2;
  };

  const objective = (params: number[]): number => {
    const sigma2 = computeVariances(params);
    if (!sigma2) return 1e12;
    return -garchLogLikelihood(eps, sigma2);
  };

  const { params, converged } = nelderMead(objective, x0);

  const omega = params[0];
  const alpha = params.slice(1, 1 + q);
  const gamma = params.slice(1 + q, 1 + 2 * q);
  const beta = params.slice(1 + 2 * q, 1 + 2 * q + p);
  const sigma2 = computeVariances(params) ?? new Array(T).fill(uncondVar);
  const ll = garchLogLikelihood(eps, sigma2);
  const nParams = 1 + 2 * q + p;
  const { aic, bic } = computeIC(ll, nParams, T);

  const stdResid = eps.map((e, t) => e / Math.sqrt(sigma2[t]));

  return {
    omega,
    alpha,
    beta,
    gamma,
    logLikelihood: ll,
    conditionalVariance: sigma2,
    standardizedResiduals: stdResid,
    aic,
    bic,
    nObs: T,
    converged,
  };
}

/**
 * Fit a GJR-GARCH(p,q) model to a time series via conditional MLE.
 *
 * The GJR-GARCH model (Glosten, Jagannathan, Runkle 1993) extends GARCH
 * with an asymmetric leverage term:
 *   sigma2_t = omega + sum_{i=1}^{q} [alpha_i * eps_{t-i}^2 + gamma_i * I(eps_{t-i}<0) * eps_{t-i}^2]
 *                     + sum_{j=1}^{p} beta_j * sigma2_{t-j}
 *
 * where I(.) is the indicator function. Negative shocks (gamma_i > 0)
 * have a larger impact on future volatility than positive shocks.
 *
 * @param data - Input time series
 * @param p - GARCH order, default 1
 * @param q - ARCH order, default 1
 * @returns Fitted GJR-GARCH model result
 *
 * @example
 * ```ts
 * const result = gjrGarchFit(returns, 1, 1);
 * console.log(result.gamma); // leverage coefficients
 * ```
 */
export function gjrGarchFit(data: number[], p = 1, q = 1): GJRGARCHResult {
  const eps = demean(data);
  const T = eps.length;
  const uncondVar = variance(eps);

  const betaInit = new Array(p).fill(0.8 / Math.max(p, 1));
  const alphaInit = new Array(q).fill(0.05);
  const gammaInit = new Array(q).fill(0.05);
  const omegaInit =
    uncondVar *
    (1 -
      alphaInit.reduce((s, a) => s + a, 0) -
      0.5 * gammaInit.reduce((s, g) => s + g, 0) -
      betaInit.reduce((s, b) => s + b, 0));

  // Pack: [omega, alpha_1..q, gamma_1..q, beta_1..p]
  const x0 = [Math.max(omegaInit, 1e-6), ...alphaInit, ...gammaInit, ...betaInit];

  const computeVariances = (params: number[]): number[] | null => {
    const omega = params[0];
    const alpha = params.slice(1, 1 + q);
    const gamma = params.slice(1 + q, 1 + 2 * q);
    const beta = params.slice(1 + 2 * q, 1 + 2 * q + p);

    // Constraint checks
    if (omega <= 0) return null;
    for (let i = 0; i < q; i++) {
      if (alpha[i] < 0) return null;
      if (gamma[i] < 0) return null;
    }
    for (let j = 0; j < p; j++) if (beta[j] < 0) return null;

    // Stationarity: sum(alpha) + 0.5*sum(gamma) + sum(beta) < 1
    const persistence =
      alpha.reduce((s, a) => s + a, 0) +
      0.5 * gamma.reduce((s, g) => s + g, 0) +
      beta.reduce((s, b) => s + b, 0);
    if (persistence >= 1) return null;

    const sigma2 = new Array<number>(T);
    const startIdx = Math.max(p, q);

    for (let t = 0; t < startIdx; t++) {
      sigma2[t] = uncondVar;
    }

    for (let t = startIdx; t < T; t++) {
      let s = omega;
      for (let i = 0; i < q; i++) {
        const e2 = eps[t - 1 - i] * eps[t - 1 - i];
        s += alpha[i] * e2;
        if (eps[t - 1 - i] < 0) {
          s += gamma[i] * e2;
        }
      }
      for (let j = 0; j < p; j++) {
        s += beta[j] * sigma2[t - 1 - j];
      }
      sigma2[t] = Math.max(s, 1e-12);
    }
    return sigma2;
  };

  const objective = (params: number[]): number => {
    const sigma2 = computeVariances(params);
    if (!sigma2) return 1e12;
    return -garchLogLikelihood(eps, sigma2);
  };

  const { params, converged } = nelderMead(objective, x0);

  const omega = params[0];
  const alpha = params.slice(1, 1 + q);
  const gamma = params.slice(1 + q, 1 + 2 * q);
  const beta = params.slice(1 + 2 * q, 1 + 2 * q + p);
  const sigma2 = computeVariances(params) ?? new Array(T).fill(uncondVar);
  const ll = garchLogLikelihood(eps, sigma2);
  const nParams = 1 + 2 * q + p;
  const { aic, bic } = computeIC(ll, nParams, T);

  const stdResid = eps.map((e, t) => e / Math.sqrt(sigma2[t]));

  return {
    omega,
    alpha,
    beta,
    gamma,
    logLikelihood: ll,
    conditionalVariance: sigma2,
    standardizedResiduals: stdResid,
    aic,
    bic,
    nObs: T,
    converged,
  };
}

/**
 * Produce multi-step-ahead variance forecasts from a fitted GARCH(p,q) model.
 *
 * The forecast recursion is:
 *   sigma2_{T+h} = omega + sum_{i} alpha_i * E[eps_{T+h-i}^2]
 *                        + sum_{j} beta_j * sigma2_{T+h-j}
 *
 * For h > 1, E[eps_{T+h-i}^2] = sigma2_{T+h-i} (the forecasted variance).
 *
 * @param result - A fitted GARCHResult from garchFit or archFit
 * @param horizon - Number of steps ahead to forecast
 * @returns Forecasted variances and standard deviations
 *
 * @example
 * ```ts
 * const fit = garchFit(returns, 1, 1);
 * const forecast = garchForecast(fit, 10);
 * console.log(forecast.variance);
 * ```
 */
export function garchForecast(
  result: GARCHResult,
  horizon: number,
): GARCHForecast {
  const { omega, alpha, beta, conditionalVariance: sigma2 } = result;
  const eps2 = result.standardizedResiduals.map(
    (z, t) => z * z * sigma2[t],
  );

  const T = sigma2.length;
  const q = alpha.length;
  const p = beta.length;

  // Extended arrays: past data + forecast horizon
  const extSigma2 = [...sigma2, ...new Array(horizon).fill(0)];
  const extEps2 = [...eps2, ...new Array(horizon).fill(0)];

  for (let h = 0; h < horizon; h++) {
    const t = T + h;
    let s = omega;
    for (let i = 0; i < q; i++) {
      // For future periods, E[eps^2] = sigma2 (forecasted variance)
      const idx = t - 1 - i;
      s += alpha[i] * (idx >= T ? extSigma2[idx] : extEps2[idx]);
    }
    for (let j = 0; j < p; j++) {
      s += beta[j] * extSigma2[t - 1 - j];
    }
    extSigma2[t] = s;
    extEps2[t] = s; // E[eps^2_{T+h}] = sigma2_{T+h}
  }

  const forecastedVariance = extSigma2.slice(T);
  return {
    variance: forecastedVariance,
    standardDeviation: forecastedVariance.map(Math.sqrt),
    horizon,
  };
}

/**
 * Perform the ARCH-LM (Lagrange Multiplier) test for conditional
 * heteroscedasticity in a residual series.
 *
 * The test regresses squared residuals on their own lags:
 *   e_t^2 = c + a_1 * e_{t-1}^2 + ... + a_m * e_{t-m}^2 + u_t
 *
 * Under the null hypothesis of no ARCH effects, T * R^2 ~ chi^2(m).
 *
 * @param residuals - Residual or return series to test
 * @param lags - Number of lags to include in the auxiliary regression
 * @returns ARCH-LM test results including statistic and p-value
 *
 * @example
 * ```ts
 * const test = archLMTest(residuals, 5);
 * if (test.pValue < 0.05) {
 *   console.log("ARCH effects detected");
 * }
 * ```
 */
export function archLMTest(residuals: number[], lags = 5): ARCHLMTestResult {
  const T = residuals.length;
  const e2 = residuals.map((r) => r * r);
  const n = T - lags;

  // Build the dependent variable and regressor matrix
  const y = e2.slice(lags);
  const meanY = mean(y);

  // OLS regression of e2_t on lagged e2
  // Using normal equations: beta = (X'X)^{-1} X'y
  // For simplicity, compute R^2 directly

  // Compute fitted values via simple multiple regression
  // X is n x (lags+1) with intercept
  const X: number[][] = [];
  for (let t = lags; t < T; t++) {
    const row = [1];
    for (let l = 1; l <= lags; l++) {
      row.push(e2[t - l]);
    }
    X.push(row);
  }

  const k = lags + 1;
  // X'X
  const XtX: number[][] = Array.from({ length: k }, () => new Array(k).fill(0));
  const XtY: number[] = new Array(k).fill(0);

  for (let i = 0; i < n; i++) {
    for (let a = 0; a < k; a++) {
      XtY[a] += X[i][a] * y[i];
      for (let b = 0; b < k; b++) {
        XtX[a][b] += X[i][a] * X[i][b];
      }
    }
  }

  // Solve via Gauss elimination
  const aug: number[][] = XtX.map((row, i) => [...row, XtY[i]]);
  for (let col = 0; col < k; col++) {
    // Partial pivoting
    let maxRow = col;
    for (let row = col + 1; row < k; row++) {
      if (Math.abs(aug[row][col]) > Math.abs(aug[maxRow][col])) {
        maxRow = row;
      }
    }
    [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];

    if (Math.abs(aug[col][col]) < 1e-14) continue;

    for (let row = col + 1; row < k; row++) {
      const factor = aug[row][col] / aug[col][col];
      for (let j = col; j <= k; j++) {
        aug[row][j] -= factor * aug[col][j];
      }
    }
  }

  const betaCoeffs = new Array(k).fill(0);
  for (let i = k - 1; i >= 0; i--) {
    let s = aug[i][k];
    for (let j = i + 1; j < k; j++) {
      s -= aug[i][j] * betaCoeffs[j];
    }
    betaCoeffs[i] = Math.abs(aug[i][i]) > 1e-14 ? s / aug[i][i] : 0;
  }

  // Compute R^2
  let ssRes = 0;
  let ssTot = 0;
  for (let i = 0; i < n; i++) {
    let fitted = 0;
    for (let j = 0; j < k; j++) {
      fitted += X[i][j] * betaCoeffs[j];
    }
    ssRes += (y[i] - fitted) ** 2;
    ssTot += (y[i] - meanY) ** 2;
  }

  const rSquared = ssTot > 0 ? 1 - ssRes / ssTot : 0;
  const statistic = n * rSquared;
  const pValue = chiSquaredPValue(statistic, lags);

  return { statistic, pValue, lags, rSquared };
}

/**
 * Compute diagnostic statistics for a fitted GARCH model.
 *
 * Diagnostics include:
 * - Ljung-Box statistics on standardized and squared standardized residuals
 * - Moments (mean, variance, skewness, kurtosis) of standardized residuals
 * - ARCH-LM test on standardized residuals to check remaining heteroscedasticity
 *
 * @param standardizedResiduals - Standardized residuals from a fitted GARCH model
 * @param lags - Number of lags for Ljung-Box and ARCH-LM tests, default 10
 * @returns Diagnostic statistics
 *
 * @example
 * ```ts
 * const fit = garchFit(returns, 1, 1);
 * const diag = garchDiagnostics(fit.standardizedResiduals, 10);
 * console.log(diag.kurtosis); // should be close to 0 for normal
 * ```
 */
export function garchDiagnostics(
  standardizedResiduals: number[],
  lags = 10,
): GARCHDiagnostics {
  const z = standardizedResiduals;
  const T = z.length;

  const m = mean(z);
  const v = variance(z);
  const z2 = z.map((r) => r * r);

  // Skewness
  const m3 = mean(z.map((r) => (r - m) ** 3));
  const skewness = m3 / Math.pow(v, 1.5);

  // Excess kurtosis
  const m4 = mean(z.map((r) => (r - m) ** 4));
  const kurtosis = m4 / (v * v) - 3;

  // Ljung-Box statistic
  const ljungBox = (series: number[], maxLag: number): number => {
    const n = series.length;
    const mu = mean(series);
    const centered = series.map((s) => s - mu);
    let gamma0 = 0;
    for (let t = 0; t < n; t++) gamma0 += centered[t] * centered[t];
    gamma0 /= n;

    if (gamma0 < 1e-14) return 0;

    let Q = 0;
    for (let k = 1; k <= maxLag; k++) {
      let gammaK = 0;
      for (let t = k; t < n; t++) {
        gammaK += centered[t] * centered[t - k];
      }
      gammaK /= n;
      const rho = gammaK / gamma0;
      Q += (rho * rho) / (n - k);
    }
    Q *= n * (n + 2);
    return Q;
  };

  const ljungBoxResiduals = ljungBox(z, lags);
  const ljungBoxSquaredResiduals = ljungBox(z2, lags);

  const archTest = archLMTest(z, Math.min(lags, Math.floor(T / 3)));

  return {
    ljungBoxResiduals,
    ljungBoxSquaredResiduals,
    meanStdResiduals: m,
    varianceStdResiduals: v,
    skewness,
    kurtosis,
    archLMTest: archTest,
  };
}
