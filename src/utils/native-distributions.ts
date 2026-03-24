/**
 * Native distribution function acceleration layer.
 *
 * Exposes Fortran-accelerated CDF/PDF computations for chi-squared,
 * Student's t, F, normal, gamma, and beta distributions.
 * Falls back to pure TypeScript when the native addon is unavailable.
 */

import { nativeAddon } from "./native-addon";

// ── Native interface ────────────────────────────────────────────────────

interface NativeDistributions {
  chi2Cdf(x: number, df: number): number;
  chi2Pdf(x: number, df: number): number;
  tCdf(x: number, df: number): number;
  tPdf(x: number, df: number): number;
  fCdf(x: number, d1: number, d2: number): number;
  fPdf(x: number, d1: number, d2: number): number;
  normalPdf(x: number): number;
  gammaCdf(x: number, shape: number, scale: number): number;
  betaCdf(x: number, a: number, b: number): number;
  chi2CdfBatch(x: number[], df: number): number[];
  tCdfBatch(x: number[], df: number): number[];
  normalCdfBatch(x: number[]): number[];
}

// Probe for native distributions availability
let native: NativeDistributions | null = null;
try {
  if (nativeAddon && typeof nativeAddon.chi2Cdf === "function") {
    native = nativeAddon as unknown as NativeDistributions;
  }
} catch {
  // Fallback to TypeScript
}

/** Whether native distribution acceleration is available. */
export const hasNativeDistributions = native !== null;

// ── Chi-squared ─────────────────────────────────────────────────────────

export function chi2Cdf(x: number, df: number): number {
  if (native) return native.chi2Cdf(x, df);
  return tsRegularizedGammaP(df / 2, x / 2);
}

export function chi2Pdf(x: number, df: number): number {
  if (native) return native.chi2Pdf(x, df);
  if (x <= 0) return 0;
  const halfDf = df / 2;
  return Math.exp(
    (halfDf - 1) * Math.log(x) - x / 2 - halfDf * Math.log(2) - tsGammaLn(halfDf),
  );
}

// ── Student's t ─────────────────────────────────────────────────────────

export function tCdf(x: number, df: number): number {
  if (native) return native.tCdf(x, df);
  const t2 = x * x;
  const u = df / (df + t2);
  if (x >= 0) {
    return 1 - 0.5 * tsRegularizedBeta(u, df / 2, 0.5);
  }
  return 0.5 * tsRegularizedBeta(u, df / 2, 0.5);
}

export function tPdf(x: number, df: number): number {
  if (native) return native.tPdf(x, df);
  const lnCoeff =
    tsGammaLn((df + 1) / 2) - tsGammaLn(df / 2) - 0.5 * Math.log(df * Math.PI);
  return Math.exp(lnCoeff - ((df + 1) / 2) * Math.log(1 + (x * x) / df));
}

// ── F-distribution ──────────────────────────────────────────────────────

export function fCdf(x: number, d1: number, d2: number): number {
  if (native) return native.fCdf(x, d1, d2);
  if (x <= 0) return 0;
  return tsRegularizedBeta((d1 * x) / (d1 * x + d2), d1 / 2, d2 / 2);
}

export function fPdf(x: number, d1: number, d2: number): number {
  if (native) return native.fPdf(x, d1, d2);
  if (x <= 0) return 0;
  const lnCoeff =
    (d1 / 2) * Math.log(d1 / d2) +
    (d1 / 2 - 1) * Math.log(x) -
    ((d1 + d2) / 2) * Math.log(1 + (d1 * x) / d2) -
    tsGammaLn(d1 / 2) -
    tsGammaLn(d2 / 2) +
    tsGammaLn((d1 + d2) / 2);
  return Math.exp(lnCoeff);
}

// ── Normal ──────────────────────────────────────────────────────────────

export function normalPdf(x: number): number {
  if (native) return native.normalPdf(x);
  return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
}

// ── Gamma distribution CDF ──────────────────────────────────────────────

export function gammaCdf(x: number, shape: number, scale: number): number {
  if (native) return native.gammaCdf(x, shape, scale);
  if (x <= 0) return 0;
  return tsRegularizedGammaP(shape, x / scale);
}

// ── Beta distribution CDF ───────────────────────────────────────────────

export function betaCdf(x: number, a: number, b: number): number {
  if (native) return native.betaCdf(x, a, b);
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  return tsRegularizedBeta(x, a, b);
}

// ── Batch operations ────────────────────────────────────────────────────

export function chi2CdfBatch(x: number[], df: number): number[] {
  if (native) return native.chi2CdfBatch(x, df);
  return x.map((v) => chi2Cdf(v, df));
}

export function tCdfBatch(x: number[], df: number): number[] {
  if (native) return native.tCdfBatch(x, df);
  return x.map((v) => tCdf(v, df));
}

export function normalCdfBatch(x: number[]): number[] {
  if (native) return native.normalCdfBatch(x);
  return x.map((v) => 0.5 * (1 + tsErf(v / Math.SQRT2)));
}

// ── TypeScript fallback implementations ─────────────────────────────────

const LANCZOS_G = 7;
const LANCZOS_C = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028,
  771.32342877765313, -176.61502916214059, 12.507343278686905,
  -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
];

function tsGammaLn(x: number): number {
  if (x < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - tsGammaLn(1 - x);
  }
  const xx = x - 1;
  let a = LANCZOS_C[0];
  for (let i = 1; i < LANCZOS_C.length; i++) {
    a += LANCZOS_C[i] / (xx + i);
  }
  const t = xx + LANCZOS_G + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (xx + 0.5) * Math.log(t) - t + Math.log(a);
}

function tsErf(x: number): number {
  const sgn = x >= 0 ? 1 : -1;
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const poly =
    t *
    (0.254829592 +
      t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  return sgn * (1 - poly * Math.exp(-ax * ax));
}

function tsRegularizedGammaP(s: number, x: number): number {
  if (x <= 0) return 0;
  if (x < s + 1) return gammaSeriesP(s, x);
  return 1 - gammaCfQ(s, x);
}

function gammaSeriesP(s: number, x: number): number {
  let term = 1 / s;
  let total = term;
  for (let n = 1; n <= 200; n++) {
    term *= x / (s + n);
    total += term;
    if (Math.abs(term) < Math.abs(total) * 1e-14) break;
  }
  return total * Math.exp(-x + s * Math.log(x) - tsGammaLn(s));
}

function gammaCfQ(s: number, x: number): number {
  const EPS = 1e-14;
  let f = x + 1 - s;
  if (Math.abs(f) < EPS) f = EPS;
  let c = f;
  let d = 0;
  for (let n = 1; n <= 200; n++) {
    const an = n * (s - n);
    const bn = x + 2 * n + 1 - s;
    d = bn + an * d;
    if (Math.abs(d) < EPS) d = EPS;
    c = bn + an / c;
    if (Math.abs(c) < EPS) c = EPS;
    d = 1 / d;
    const delta = c * d;
    f *= delta;
    if (Math.abs(delta - 1) < EPS) break;
  }
  return Math.exp(-x + s * Math.log(x) - tsGammaLn(s)) / f;
}

function tsRegularizedBeta(x: number, a: number, b: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  if (x > (a + 1) / (a + b + 2)) {
    return 1 - tsRegularizedBeta(1 - x, b, a);
  }
  const lnPre = tsGammaLn(a + b) - tsGammaLn(a) - tsGammaLn(b) + a * Math.log(x) + b * Math.log(1 - x);
  return (Math.exp(lnPre) * betaCf(x, a, b)) / a;
}

function betaCf(x: number, a: number, b: number): number {
  const EPS = 1e-14;
  let f = 1;
  let c = 1;
  let d = 1 - ((a + b) * x) / (a + 1);
  if (Math.abs(d) < EPS) d = EPS;
  d = 1 / d;
  f = d;

  for (let m = 1; m <= 200; m++) {
    // Even step: d_{2m} numerator
    let num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m));
    d = 1 + num * d;
    if (Math.abs(d) < EPS) d = EPS;
    c = 1 + num / c;
    if (Math.abs(c) < EPS) c = EPS;
    d = 1 / d;
    f *= c * d;

    // Odd step: d_{2m+1} numerator
    num = (-(a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1));
    d = 1 + num * d;
    if (Math.abs(d) < EPS) d = EPS;
    c = 1 + num / c;
    if (Math.abs(c) < EPS) c = EPS;
    d = 1 / d;
    const delta = c * d;
    f *= delta;
    if (Math.abs(delta - 1) < EPS) break;
  }
  return f;
}
