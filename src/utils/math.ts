const EPSILON = 1e-14;
const MAX_ITERATIONS = 200;

// Lanczos approximation coefficients (g=7, n=9)
const LANCZOS_G = 7;
const LANCZOS_COEFF = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028,
  771.32342877765313, -176.61502916214059, 12.507343278686905,
  -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
];

/** Natural log of the gamma function. */
export function gammaLn(x: number): number {
  if (x <= 0 && Number.isInteger(x)) {
    throw new Error("gammaLn is not defined for non-positive integers");
  }
  if (x < 0.5) {
    // Reflection formula: Gamma(x) * Gamma(1-x) = pi / sin(pi*x)
    return (
      Math.log(Math.PI / Math.sin(Math.PI * x)) - gammaLn(1 - x)
    );
  }
  x -= 1;
  let a = LANCZOS_COEFF[0];
  for (let i = 1; i < LANCZOS_COEFF.length; i++) {
    a += LANCZOS_COEFF[i] / (x + i);
  }
  const t = x + LANCZOS_G + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

/** Gamma function. */
export function gamma(x: number): number {
  return Math.exp(gammaLn(x));
}

/** Log of the factorial: ln(n!) */
export function logFactorial(n: number): number {
  if (n < 0 || !Number.isInteger(n)) {
    throw new Error("logFactorial requires a non-negative integer");
  }
  return gammaLn(n + 1);
}

/** Factorial n! (for n <= 170 to avoid Infinity). */
export function factorial(n: number): number {
  if (n < 0 || !Number.isInteger(n)) {
    throw new Error("factorial requires a non-negative integer");
  }
  if (n > 170) return Infinity;
  return Math.exp(logFactorial(n));
}

/** Binomial coefficient C(n, k). */
export function binomialCoeff(n: number, k: number): number {
  if (k < 0 || k > n) return 0;
  return Math.exp(logFactorial(n) - logFactorial(k) - logFactorial(n - k));
}

/** Beta function B(a, b). */
export function betaFn(a: number, b: number): number {
  return Math.exp(gammaLn(a) + gammaLn(b) - gammaLn(a + b));
}

// --- Error function ---

/** Error function erf(x) using Horner approximation. */
export function erf(x: number): number {
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

/** Complementary error function erfc(x) = 1 - erf(x). */
export function erfc(x: number): number {
  return 1 - erf(x);
}

// --- Regularized incomplete gamma ---

/** Lower regularized incomplete gamma function P(s, x) = γ(s,x) / Γ(s). */
export function regularizedGammaP(s: number, x: number): number {
  if (x < 0) throw new Error("x must be non-negative");
  if (x === 0) return 0;

  if (x < s + 1) {
    // Series expansion
    return gammaPSeries(s, x);
  } else {
    // Continued fraction
    return 1 - gammaPContinuedFraction(s, x);
  }
}

function gammaPSeries(s: number, x: number): number {
  let term = 1 / s;
  let sum = term;
  for (let n = 1; n < MAX_ITERATIONS; n++) {
    term *= x / (s + n);
    sum += term;
    if (Math.abs(term) < Math.abs(sum) * EPSILON) break;
  }
  return sum * Math.exp(-x + s * Math.log(x) - gammaLn(s));
}

function gammaPContinuedFraction(s: number, x: number): number {
  // Lentz's algorithm for Q(s,x) = 1 - P(s,x)
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
  return Math.exp(-x + s * Math.log(x) - gammaLn(s)) / f;
}

// --- Regularized incomplete beta ---

/** Regularized incomplete beta function I_x(a, b). */
export function regularizedBeta(x: number, a: number, b: number): number {
  if (x < 0 || x > 1) throw new Error("x must be in [0, 1]");
  if (x === 0) return 0;
  if (x === 1) return 1;

  // Use symmetry for better convergence
  if (x > (a + 1) / (a + b + 2)) {
    return 1 - regularizedBeta(1 - x, b, a);
  }

  const lnPrefactor =
    gammaLn(a + b) - gammaLn(a) - gammaLn(b) +
    a * Math.log(x) + b * Math.log(1 - x);
  const prefactor = Math.exp(lnPrefactor);

  return prefactor * betaCF(x, a, b) / a;
}

/** Continued fraction for the incomplete beta (Lentz's algorithm). */
function betaCF(x: number, a: number, b: number): number {
  let f = 1;
  let c = 1;
  let d = 1 - (a + b) * x / (a + 1);
  if (Math.abs(d) < EPSILON) d = EPSILON;
  d = 1 / d;
  f = d;

  for (let m = 1; m <= MAX_ITERATIONS; m++) {
    // Even step
    let numerator = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m));
    d = 1 + numerator * d;
    if (Math.abs(d) < EPSILON) d = EPSILON;
    c = 1 + numerator / c;
    if (Math.abs(c) < EPSILON) c = EPSILON;
    d = 1 / d;
    f *= c * d;

    // Odd step
    numerator =
      -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1));
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

// --- Quantile helpers ---

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
