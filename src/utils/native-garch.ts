/**
 * Native GARCH model acceleration layer.
 *
 * Exposes Fortran-accelerated routines for GARCH variance recursion
 * and log-likelihood evaluation. Falls back to pure TypeScript when
 * the native addon is unavailable.
 */

import { nativeAddon } from "./native-addon";

// ── Native interface ────────────────────────────────────────────────────

interface NativeGarch {
  garch11Loglik(
    eps: number[],
    omega: number,
    alpha1: number,
    beta1: number,
  ): { sigma2: number[]; logLikelihood: number };
  garchPqLoglik(
    eps: number[],
    omega: number,
    alpha: number[],
    beta: number[],
  ): { sigma2: number[]; logLikelihood: number };
  gjrGarch11Loglik(
    eps: number[],
    omega: number,
    alpha1: number,
    beta1: number,
    gamma1: number,
  ): { sigma2: number[]; logLikelihood: number };
  egarch11Loglik(
    eps: number[],
    omega: number,
    alpha1: number,
    beta1: number,
    gamma1: number,
  ): { sigma2: number[]; logLikelihood: number };
  garch11Forecast(
    lastEps2: number,
    lastSigma2: number,
    omega: number,
    alpha1: number,
    beta1: number,
    h: number,
  ): number[];
}

// Probe for native GARCH availability
let native: NativeGarch | null = null;
try {
  if (nativeAddon && typeof nativeAddon.garch11Loglik === "function") {
    native = nativeAddon as unknown as NativeGarch;
  }
} catch {
  // Fallback to TypeScript
}

/** Whether native GARCH acceleration is available. */
export const hasNativeGarch = native !== null;

// ── GARCH(1,1) ──────────────────────────────────────────────────────────

/**
 * GARCH(1,1) conditional variance recursion and log-likelihood.
 * Uses Fortran when available for tight loop acceleration.
 *
 * @param eps - Residuals (mean-subtracted returns)
 * @param omega - Intercept (> 0)
 * @param alpha1 - ARCH coefficient (>= 0)
 * @param beta1 - GARCH coefficient (>= 0)
 * @returns Conditional variances and total log-likelihood
 */
export function garch11Loglik(
  eps: number[],
  omega: number,
  alpha1: number,
  beta1: number,
): { sigma2: number[]; logLikelihood: number } {
  if (native) {
    return native.garch11Loglik(eps, omega, alpha1, beta1);
  }
  return tsGarch11Loglik(eps, omega, alpha1, beta1);
}

function tsGarch11Loglik(
  eps: number[],
  omega: number,
  alpha1: number,
  beta1: number,
): { sigma2: number[]; logLikelihood: number } {
  const T = eps.length;
  const LOG2PI = Math.log(2 * Math.PI);
  const sigma2 = new Array(T);

  // Initialize with unconditional variance
  let s2: number;
  if (alpha1 + beta1 < 1) {
    s2 = omega / (1 - alpha1 - beta1);
  } else {
    s2 = eps.reduce((sum, e) => sum + e * e, 0) / T;
  }

  sigma2[0] = s2;
  let ll = -0.5 * (LOG2PI + Math.log(Math.max(s2, 1e-300)) + (eps[0] * eps[0]) / Math.max(s2, 1e-300));

  for (let t = 1; t < T; t++) {
    s2 = omega + alpha1 * eps[t - 1] * eps[t - 1] + beta1 * sigma2[t - 1];
    s2 = Math.max(s2, 1e-12);
    sigma2[t] = s2;
    ll -= 0.5 * (LOG2PI + Math.log(s2) + (eps[t] * eps[t]) / s2);
  }

  return { sigma2, logLikelihood: ll };
}

// ── General GARCH(p,q) ──────────────────────────────────────────────────

export function garchPqLoglik(
  eps: number[],
  omega: number,
  alpha: number[],
  beta: number[],
): { sigma2: number[]; logLikelihood: number } {
  if (native) {
    return native.garchPqLoglik(eps, omega, alpha, beta);
  }
  return tsGarchPqLoglik(eps, omega, alpha, beta);
}

function tsGarchPqLoglik(
  eps: number[],
  omega: number,
  alpha: number[],
  beta: number[],
): { sigma2: number[]; logLikelihood: number } {
  const T = eps.length;
  const LOG2PI = Math.log(2 * Math.PI);
  const q = alpha.length;
  const p = beta.length;
  const maxpq = Math.max(p, q);
  const sigma2 = new Array(T);

  const sumAB = alpha.reduce((s, a) => s + a, 0) + beta.reduce((s, b) => s + b, 0);
  const initVar = sumAB < 1 ? omega / (1 - sumAB) : eps.reduce((s, e) => s + e * e, 0) / T;

  for (let t = 0; t < Math.min(maxpq, T); t++) sigma2[t] = initVar;

  let ll = 0;
  for (let t = 0; t < Math.min(maxpq, T); t++) {
    ll -= 0.5 * (LOG2PI + Math.log(Math.max(sigma2[t], 1e-300)) + (eps[t] * eps[t]) / Math.max(sigma2[t], 1e-300));
  }

  for (let t = maxpq; t < T; t++) {
    let s2 = omega;
    for (let j = 0; j < q; j++) s2 += alpha[j] * eps[t - j - 1] * eps[t - j - 1];
    for (let j = 0; j < p; j++) s2 += beta[j] * sigma2[t - j - 1];
    s2 = Math.max(s2, 1e-12);
    sigma2[t] = s2;
    ll -= 0.5 * (LOG2PI + Math.log(s2) + (eps[t] * eps[t]) / s2);
  }

  return { sigma2, logLikelihood: ll };
}

// ── GJR-GARCH(1,1) ─────────────────────────────────────────────────────

export function gjrGarch11Loglik(
  eps: number[],
  omega: number,
  alpha1: number,
  beta1: number,
  gamma1: number,
): { sigma2: number[]; logLikelihood: number } {
  if (native) {
    return native.gjrGarch11Loglik(eps, omega, alpha1, beta1, gamma1);
  }
  return tsGjrGarch11Loglik(eps, omega, alpha1, beta1, gamma1);
}

function tsGjrGarch11Loglik(
  eps: number[],
  omega: number,
  alpha1: number,
  beta1: number,
  gamma1: number,
): { sigma2: number[]; logLikelihood: number } {
  const T = eps.length;
  const LOG2PI = Math.log(2 * Math.PI);
  const sigma2 = new Array(T);

  let s2: number;
  if (alpha1 + beta1 + 0.5 * gamma1 < 1) {
    s2 = omega / (1 - alpha1 - beta1 - 0.5 * gamma1);
  } else {
    s2 = eps.reduce((sum, e) => sum + e * e, 0) / T;
  }

  sigma2[0] = s2;
  let ll = -0.5 * (LOG2PI + Math.log(Math.max(s2, 1e-300)) + (eps[0] * eps[0]) / Math.max(s2, 1e-300));

  for (let t = 1; t < T; t++) {
    const indicator = eps[t - 1] < 0 ? 1 : 0;
    s2 = omega + alpha1 * eps[t - 1] * eps[t - 1] +
      gamma1 * indicator * eps[t - 1] * eps[t - 1] +
      beta1 * sigma2[t - 1];
    s2 = Math.max(s2, 1e-12);
    sigma2[t] = s2;
    ll -= 0.5 * (LOG2PI + Math.log(s2) + (eps[t] * eps[t]) / s2);
  }

  return { sigma2, logLikelihood: ll };
}

// ── EGARCH(1,1) ─────────────────────────────────────────────────────────

export function egarch11Loglik(
  eps: number[],
  omega: number,
  alpha1: number,
  beta1: number,
  gamma1: number,
): { sigma2: number[]; logLikelihood: number } {
  if (native) {
    return native.egarch11Loglik(eps, omega, alpha1, beta1, gamma1);
  }
  return tsEgarch11Loglik(eps, omega, alpha1, beta1, gamma1);
}

function tsEgarch11Loglik(
  eps: number[],
  omega: number,
  alpha1: number,
  beta1: number,
  gamma1: number,
): { sigma2: number[]; logLikelihood: number } {
  const T = eps.length;
  const LOG2PI = Math.log(2 * Math.PI);
  const SQRT_2_OVER_PI = Math.sqrt(2 / Math.PI);
  const sigma2 = new Array(T);

  let logS2: number;
  if (Math.abs(1 - beta1) > 1e-10) {
    logS2 = omega / (1 - beta1);
  } else {
    logS2 = Math.log(eps.reduce((s, e) => s + e * e, 0) / T);
  }

  let s2 = Math.exp(logS2);
  sigma2[0] = s2;
  let ll = -0.5 * (LOG2PI + logS2 + (eps[0] * eps[0]) / Math.max(s2, 1e-300));

  for (let t = 1; t < T; t++) {
    const z = eps[t - 1] / Math.sqrt(Math.max(sigma2[t - 1], 1e-300));
    logS2 = omega + beta1 * Math.log(Math.max(sigma2[t - 1], 1e-300)) +
      alpha1 * (Math.abs(z) - SQRT_2_OVER_PI) + gamma1 * z;
    s2 = Math.max(Math.exp(logS2), 1e-300);
    sigma2[t] = s2;
    ll -= 0.5 * (LOG2PI + logS2 + (eps[t] * eps[t]) / s2);
  }

  return { sigma2, logLikelihood: ll };
}

// ── GARCH(1,1) Forecast ─────────────────────────────────────────────────

export function garch11Forecast(
  lastEps2: number,
  lastSigma2: number,
  omega: number,
  alpha1: number,
  beta1: number,
  h: number,
): number[] {
  if (native) {
    return native.garch11Forecast(lastEps2, lastSigma2, omega, alpha1, beta1, h);
  }

  const forecast = new Array(h);
  forecast[0] = omega + alpha1 * lastEps2 + beta1 * lastSigma2;
  for (let t = 1; t < h; t++) {
    forecast[t] = omega + (alpha1 + beta1) * forecast[t - 1];
  }
  return forecast;
}
