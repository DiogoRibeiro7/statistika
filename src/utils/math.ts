import { nativeAddon } from "./native-addon";

// Try to load the native Fortran addon; fall back to pure-TS implementations.
interface NativeSpecial {
  gammaLn(x: number): number;
  gamma(x: number): number;
  logFactorial(n: number): number;
  factorial(n: number): number;
  binomialCoeff(n: number, k: number): number;
  betaFn(a: number, b: number): number;
  erf(x: number): number;
  erfc(x: number): number;
  regularizedGammaP(s: number, x: number): number;
  regularizedBeta(x: number, a: number, b: number): number;
}

const native: NativeSpecial | null = nativeAddon as NativeSpecial | null;

// ==========================================================================
// Pure-TypeScript fallback implementations
// ==========================================================================

const EPSILON = 1e-14;
const MAX_ITERATIONS = 200;

const LANCZOS_G = 7;
const LANCZOS_COEFF = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028,
  771.32342877765313, -176.61502916214059, 12.507343278686905,
  -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
];

function tsGammaLn(x: number): number {
  if (x <= 0 && Number.isInteger(x)) {
    return NaN;
  }
  if (x < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - tsGammaLn(1 - x);
  }
  x -= 1;
  let a = LANCZOS_COEFF[0];
  for (let i = 1; i < LANCZOS_COEFF.length; i++) {
    a += LANCZOS_COEFF[i] / (x + i);
  }
  const t = x + LANCZOS_G + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

function tsGamma(x: number): number {
  return Math.exp(tsGammaLn(x));
}

function tsLogFactorial(n: number): number {
  if (n < 0 || !Number.isInteger(n)) {
    throw new Error("logFactorial requires a non-negative integer");
  }
  return tsGammaLn(n + 1);
}

function tsFactorial(n: number): number {
  if (n < 0 || !Number.isInteger(n)) {
    throw new Error("factorial requires a non-negative integer");
  }
  if (n > 170) return Infinity;
  return Math.exp(tsLogFactorial(n));
}

function tsBinomialCoeff(n: number, k: number): number {
  if (k < 0 || k > n) return 0;
  return Math.exp(tsLogFactorial(n) - tsLogFactorial(k) - tsLogFactorial(n - k));
}

function tsBetaFn(a: number, b: number): number {
  return Math.exp(tsGammaLn(a) + tsGammaLn(b) - tsGammaLn(a + b));
}

function tsErf(x: number): number {
  const sign = x >= 0 ? 1 : -1;
  x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const poly =
    t *
    (0.254829592 +
      t *
        (-0.284496736 +
          t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  return sign * (1 - poly * Math.exp(-x * x));
}

function tsErfc(x: number): number {
  return 1 - tsErf(x);
}

function gammaPSeries(s: number, x: number): number {
  let term = 1 / s;
  let sum = term;
  for (let n = 1; n < MAX_ITERATIONS; n++) {
    term *= x / (s + n);
    sum += term;
    if (Math.abs(term) < Math.abs(sum) * EPSILON) break;
  }
  return sum * Math.exp(-x + s * Math.log(x) - tsGammaLn(s));
}

function gammaPContinuedFraction(s: number, x: number): number {
  let f = x + 1 - s;
  if (Math.abs(f) < EPSILON) f = EPSILON;
  let c = f;
  let d = 0;
  for (let n = 1; n < MAX_ITERATIONS; n++) {
    const an = n * (s - n);
    const bn = x + 2 * n + 1 - s;
    d = bn + an * d;
    if (Math.abs(d) < EPSILON) d = EPSILON;
    c = bn + an / c;
    if (Math.abs(c) < EPSILON) c = EPSILON;
    d = 1 / d;
    const delta = c * d;
    f *= delta;
    if (Math.abs(delta - 1) < EPSILON) break;
  }
  return Math.exp(-x + s * Math.log(x) - tsGammaLn(s)) / f;
}

function tsRegularizedGammaP(s: number, x: number): number {
  if (x < 0) throw new Error("x must be non-negative");
  if (x === 0) return 0;
  if (x < s + 1) {
    return gammaPSeries(s, x);
  } else {
    return 1 - gammaPContinuedFraction(s, x);
  }
}

function betaCF(x: number, a: number, b: number): number {
  let c = 1;
  let d = 1 - (a + b) * x / (a + 1);
  if (Math.abs(d) < EPSILON) d = EPSILON;
  d = 1 / d;
  let f = d;
  for (let m = 1; m <= MAX_ITERATIONS; m++) {
    let numerator = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m));
    d = 1 + numerator * d;
    if (Math.abs(d) < EPSILON) d = EPSILON;
    c = 1 + numerator / c;
    if (Math.abs(c) < EPSILON) c = EPSILON;
    d = 1 / d;
    f *= c * d;
    numerator = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1));
    d = 1 + numerator * d;
    if (Math.abs(d) < EPSILON) d = EPSILON;
    c = 1 + numerator / c;
    if (Math.abs(c) < EPSILON) c = EPSILON;
    d = 1 / d;
    const delta = c * d;
    f *= delta;
    if (Math.abs(delta - 1) < EPSILON) break;
  }
  return f;
}

function tsRegularizedBeta(x: number, a: number, b: number): number {
  if (x < 0 || x > 1) throw new Error("x must be in [0, 1]");
  if (x === 0) return 0;
  if (x === 1) return 1;
  if (x > (a + 1) / (a + b + 2)) {
    return 1 - tsRegularizedBeta(1 - x, b, a);
  }
  const lnPrefactor =
    tsGammaLn(a + b) - tsGammaLn(a) - tsGammaLn(b) +
    a * Math.log(x) + b * Math.log(1 - x);
  const prefactor = Math.exp(lnPrefactor);
  return prefactor * betaCF(x, a, b) / a;
}

// ==========================================================================
// Public API — delegates to Fortran native addon when available,
// falls back to pure TypeScript.
// ==========================================================================

/** Natural log of the gamma function. */
export function gammaLn(x: number): number {
  if (native) return native.gammaLn(x);
  return tsGammaLn(x);
}

/** Gamma function. */
export function gamma(x: number): number {
  if (native) return native.gamma(x);
  return tsGamma(x);
}

/** Log of the factorial: ln(n!) */
export function logFactorial(n: number): number {
  if (native) return native.logFactorial(n);
  return tsLogFactorial(n);
}

/** Factorial n! (for n <= 170 to avoid Infinity). */
export function factorial(n: number): number {
  if (native) return native.factorial(n);
  return tsFactorial(n);
}

/** Binomial coefficient C(n, k). */
export function binomialCoeff(n: number, k: number): number {
  if (native) return native.binomialCoeff(n, k);
  return tsBinomialCoeff(n, k);
}

/** Beta function B(a, b). */
export function betaFn(a: number, b: number): number {
  if (native) return native.betaFn(a, b);
  return tsBetaFn(a, b);
}

/** Error function erf(x). */
export function erf(x: number): number {
  if (native) return native.erf(x);
  return tsErf(x);
}

/** Complementary error function erfc(x) = 1 - erf(x). */
export function erfc(x: number): number {
  if (native) return native.erfc(x);
  return tsErfc(x);
}

/** Lower regularized incomplete gamma function P(s, x). */
export function regularizedGammaP(s: number, x: number): number {
  if (native) return native.regularizedGammaP(s, x);
  return tsRegularizedGammaP(s, x);
}

/** Regularized incomplete beta function I_x(a, b). */
export function regularizedBeta(x: number, a: number, b: number): number {
  if (native) return native.regularizedBeta(x, a, b);
  return tsRegularizedBeta(x, a, b);
}

// ==========================================================================
// Quantile helper (pure TS — no Fortran needed)
// ==========================================================================

/** Generic quantile by bisection on the CDF. */
export function quantileBisect(
  cdf: (x: number) => number,
  p: number,
  lower: number,
  upper: number,
  tolerance = 1e-12,
): number {
  if (p < 0 || p > 1) throw new Error("p must be in [0, 1]");
  if (p === 0) return lower;
  if (p === 1) return upper;

  let lo = lower;
  let hi = upper;
  for (let i = 0; i < MAX_ITERATIONS; i++) {
    const mid = (lo + hi) / 2;
    if (cdf(mid) < p) {
      lo = mid;
    } else {
      hi = mid;
    }
    if (hi - lo < tolerance) break;
  }
  return (lo + hi) / 2;
}

/** Returns true if the native Fortran addon is loaded. */
export function isNativeAvailable(): boolean {
  return native !== null;
}
