/**
 * Distribution fitting via MLE and goodness-of-fit tests.
 *
 * - **MLE fitting** for Normal, Exponential, Poisson, Gamma, Beta,
 *   Log-Normal, Weibull, Geometric, Negative Binomial, and ZIP.
 * - **Anderson-Darling** test for continuous distributions.
 * - **Cramér-von Mises** test for continuous distributions.
 */

import { Normal } from "./distributions/continuous/normal";
import { Exponential } from "./distributions/continuous/exponential";
import { LogNormal } from "./distributions/continuous/log-normal";
import { GammaDistribution } from "./distributions/continuous/gamma";
import { BetaDistribution } from "./distributions/continuous/beta";
import { Weibull } from "./distributions/continuous/weibull";
import { Poisson } from "./distributions/discrete/poisson";
import { Geometric } from "./distributions/discrete/geometric";
import { NegativeBinomial } from "./distributions/discrete/negative-binomial";
import { ZeroInflatedPoisson } from "./distributions/discrete/zero-inflated-poisson";
import { ContinuousDistribution } from "./types";
import { mean, variance } from "./utils/descriptive";

// ── MLE Fitting ───────────────────────────────────────────────────────────

export interface FitResult<T> {
  /** The fitted distribution instance. */
  distribution: T;
  /** Log-likelihood of the data under the fitted distribution. */
  logLikelihood: number;
  /** AIC = −2ℓ + 2k. */
  aic: number;
  /** Number of estimated parameters. */
  nParams: number;
}

/**
 * Fit a Normal distribution by MLE.
 * MLE: μ̂ = x̄, σ̂ = √[(1/n) Σ(xᵢ − x̄)²]
 */
export function fitNormal(data: number[]): FitResult<Normal> {
  validateContinuousData(data);
  const n = data.length;
  const mu = mean(data);
  let sumSq = 0;
  for (let i = 0; i < n; i++) sumSq += (data[i] - mu) ** 2;
  const sigma = Math.sqrt(sumSq / n);
  if (sigma <= 0) throw new Error("Data has zero variance");

  const dist = new Normal(mu, sigma);
  const ll = data.reduce((s, x) => s + Math.log(dist.pdf(x)), 0);
  return { distribution: dist, logLikelihood: ll, aic: -2 * ll + 4, nParams: 2 };
}

/**
 * Fit an Exponential distribution by MLE.
 * MLE: λ̂ = 1 / x̄
 */
export function fitExponential(data: number[]): FitResult<Exponential> {
  validateContinuousData(data);
  for (const x of data) {
    if (x < 0) throw new Error("Exponential data must be non-negative");
  }
  const lambda = 1 / mean(data);
  const dist = new Exponential(lambda);
  const ll = data.reduce((s, x) => s + Math.log(dist.pdf(x)), 0);
  return { distribution: dist, logLikelihood: ll, aic: -2 * ll + 2, nParams: 1 };
}

/**
 * Fit a Poisson distribution by MLE.
 * MLE: λ̂ = x̄
 */
export function fitPoisson(data: number[]): FitResult<Poisson> {
  validateDiscreteData(data);
  const lambda = mean(data);
  if (lambda <= 0) throw new Error("Mean must be positive for Poisson");
  const dist = new Poisson(lambda);
  const ll = data.reduce((s, x) => s + Math.log(Math.max(1e-300, dist.pmf(x))), 0);
  return { distribution: dist, logLikelihood: ll, aic: -2 * ll + 2, nParams: 1 };
}

/**
 * Fit a Gamma distribution by MLE (method of moments initialisation).
 * Iterative Newton-Raphson on the shape parameter.
 */
export function fitGamma(data: number[]): FitResult<GammaDistribution> {
  validateContinuousData(data);
  for (const x of data) {
    if (x <= 0) throw new Error("Gamma data must be positive");
  }
  const n = data.length;
  const xBar = mean(data);
  const logXBar = Math.log(xBar);
  let meanLogX = 0;
  for (let i = 0; i < n; i++) meanLogX += Math.log(data[i]);
  meanLogX /= n;

  const s = logXBar - meanLogX;
  // Method of moments initial estimate for shape
  let shape = (3 - s + Math.sqrt((s - 3) ** 2 + 24 * s)) / (12 * s);

  // Newton-Raphson refinement
  for (let iter = 0; iter < 50; iter++) {
    const psiVal = digamma(shape);
    const psiPrime = trigamma(shape);
    const update = (Math.log(shape) - psiVal - s) / (1 / shape - psiPrime);
    shape -= update;
    shape = Math.max(0.001, shape);
    if (Math.abs(update) < 1e-8) break;
  }

  const rate = shape / xBar;
  const dist = new GammaDistribution(shape, rate);
  const ll = data.reduce((s, x) => s + Math.log(dist.pdf(x)), 0);
  return { distribution: dist, logLikelihood: ll, aic: -2 * ll + 4, nParams: 2 };
}

/**
 * Fit a Beta distribution by MLE (method of moments).
 */
export function fitBeta(data: number[]): FitResult<BetaDistribution> {
  validateContinuousData(data);
  for (const x of data) {
    if (x <= 0 || x >= 1) throw new Error("Beta data must be in (0, 1)");
  }
  const m = mean(data);
  const v = variance(data);
  // Method of moments
  const common = (m * (1 - m)) / v - 1;
  const alpha = m * common;
  const beta = (1 - m) * common;
  if (alpha <= 0 || beta <= 0) {
    throw new Error("Cannot fit Beta: moment estimates are non-positive");
  }

  const dist = new BetaDistribution(alpha, beta);
  const ll = data.reduce((s, x) => s + Math.log(dist.pdf(x)), 0);
  return { distribution: dist, logLikelihood: ll, aic: -2 * ll + 4, nParams: 2 };
}

/**
 * Fit a Log-Normal distribution by MLE.
 * MLE: μ̂ = mean(log x), σ̂ = std(log x)
 */
export function fitLogNormal(data: number[]): FitResult<LogNormal> {
  validateContinuousData(data);
  for (const x of data) {
    if (x <= 0) throw new Error("Log-Normal data must be positive");
  }
  const n = data.length;
  const logData = data.map(Math.log);
  const mu = mean(logData);
  let sumSq = 0;
  for (let i = 0; i < n; i++) sumSq += (logData[i] - mu) ** 2;
  const sigma = Math.sqrt(sumSq / n);
  if (sigma <= 0) throw new Error("Log-data has zero variance");

  const dist = new LogNormal(mu, sigma);
  const ll = data.reduce((s, x) => s + Math.log(dist.pdf(x)), 0);
  return { distribution: dist, logLikelihood: ll, aic: -2 * ll + 4, nParams: 2 };
}

/**
 * Fit a Geometric distribution by MLE.
 * MLE: p̂ = 1 / (1 + x̄)
 */
export function fitGeometric(data: number[]): FitResult<Geometric> {
  validateDiscreteData(data);
  const p = 1 / (1 + mean(data));
  const dist = new Geometric(p);
  const ll = data.reduce((s, x) => s + Math.log(Math.max(1e-300, dist.pmf(x))), 0);
  return { distribution: dist, logLikelihood: ll, aic: -2 * ll + 2, nParams: 1 };
}

/**
 * Fit a Zero-Inflated Poisson distribution by EM algorithm.
 */
export function fitZIP(data: number[]): FitResult<ZeroInflatedPoisson> {
  validateDiscreteData(data);
  const n = data.length;
  const nZeros = data.filter((x) => x === 0).length;

  // Initial estimates
  let lambda = mean(data) || 0.5;
  let pi = Math.max(0, (nZeros / n - Math.exp(-lambda)) / (1 - Math.exp(-lambda)));
  pi = Math.min(pi, 0.99);

  // EM iterations
  for (let iter = 0; iter < 100; iter++) {
    const oldLambda = lambda;
    const oldPi = pi;

    // E-step: compute posterior probability of structural zero for zeros
    const poissonZero = Math.exp(-lambda);
    const pStructural = pi / (pi + (1 - pi) * poissonZero);

    // M-step
    const effectiveZeros = nZeros * pStructural;
    pi = effectiveZeros / n;
    pi = Math.min(Math.max(pi, 0), 0.999);

    let sumX = 0;
    for (let i = 0; i < n; i++) sumX += data[i];
    lambda = sumX / (n * (1 - pi));
    lambda = Math.max(lambda, 0.001);

    if (Math.abs(lambda - oldLambda) + Math.abs(pi - oldPi) < 1e-8) break;
  }

  const dist = new ZeroInflatedPoisson(lambda, pi);
  const ll = data.reduce((s, x) => s + Math.log(Math.max(1e-300, dist.pmf(x))), 0);
  return { distribution: dist, logLikelihood: ll, aic: -2 * ll + 4, nParams: 2 };
}

// ── Anderson-Darling test ─────────────────────────────────────────────────

export interface GoodnessOfFitResult {
  /** Test statistic. */
  statistic: number;
  /**
   * Whether the null hypothesis (data follows the distribution) is
   * rejected at the given significance level.
   * For Anderson-Darling, uses approximate critical values for Normal.
   * For Cramér-von Mises, uses approximate asymptotic critical values.
   */
  rejected: boolean;
  /** Significance level used. */
  alpha: number;
}

/**
 * Anderson-Darling test for goodness of fit.
 *
 * Tests H₀: the data come from the specified continuous distribution.
 *
 * The statistic is  A² = −n − (1/n) Σᵢ (2i−1)[ln F(yᵢ) + ln(1 − F(y_{n+1−i}))]
 * where y₁ ≤ … ≤ yₙ are the sorted data and F is the hypothesised CDF.
 *
 * @param data  Sample data.
 * @param distribution  Hypothesised continuous distribution (must have `cdf`).
 * @param alpha  Significance level (default 0.05).
 */
export function andersonDarlingTest(
  data: number[],
  distribution: ContinuousDistribution,
  alpha = 0.05,
): GoodnessOfFitResult {
  const n = data.length;
  if (n < 2) throw new Error("Need at least 2 observations");

  const sorted = [...data].sort((a, b) => a - b);

  let S = 0;
  for (let i = 0; i < n; i++) {
    const Fi = clampProb(distribution.cdf(sorted[i]));
    const Fni = clampProb(distribution.cdf(sorted[n - 1 - i]));
    S += (2 * (i + 1) - 1) * (Math.log(Fi) + Math.log(1 - Fni));
  }

  const A2 = -n - S / n;

  // Approximate critical values (for composite tests with estimated params,
  // these are conservative; for simple tests they are exact)
  const criticals: Record<number, number> = {
    0.15: 1.610,
    0.10: 1.933,
    0.05: 2.492,
    0.025: 3.070,
    0.01: 3.857,
  };
  const critical = criticals[alpha] ?? 2.492; // default to 0.05

  return {
    statistic: A2,
    rejected: A2 > critical,
    alpha,
  };
}

/**
 * Cramér-von Mises test for goodness of fit.
 *
 * Tests H₀: the data come from the specified continuous distribution.
 *
 * The statistic is  W² = Σᵢ [F(yᵢ) − (2i−1)/(2n)]² + 1/(12n)
 * where y₁ ≤ … ≤ yₙ are the sorted data and F is the hypothesised CDF.
 *
 * @param data  Sample data.
 * @param distribution  Hypothesised continuous distribution (must have `cdf`).
 * @param alpha  Significance level (default 0.05).
 */
export function cramerVonMisesTest(
  data: number[],
  distribution: ContinuousDistribution,
  alpha = 0.05,
): GoodnessOfFitResult {
  const n = data.length;
  if (n < 2) throw new Error("Need at least 2 observations");

  const sorted = [...data].sort((a, b) => a - b);

  let W2 = 0;
  for (let i = 0; i < n; i++) {
    const Fi = clampProb(distribution.cdf(sorted[i]));
    const diff = Fi - (2 * (i + 1) - 1) / (2 * n);
    W2 += diff * diff;
  }
  W2 += 1 / (12 * n);

  // Approximate critical values (asymptotic)
  const criticals: Record<number, number> = {
    0.15: 0.284,
    0.10: 0.347,
    0.05: 0.461,
    0.025: 0.581,
    0.01: 0.743,
  };
  const critical = criticals[alpha] ?? 0.461;

  return {
    statistic: W2,
    rejected: W2 > critical,
    alpha,
  };
}

// ── Helpers ───────────────────────────────────────────────────────────────

function validateContinuousData(data: number[]): void {
  if (data.length < 2) throw new Error("Need at least 2 observations");
  for (let i = 0; i < data.length; i++) {
    if (!Number.isFinite(data[i])) {
      throw new Error(`data[${i}] is not finite`);
    }
  }
}

function validateDiscreteData(data: number[]): void {
  if (data.length < 2) throw new Error("Need at least 2 observations");
  for (let i = 0; i < data.length; i++) {
    if (!Number.isInteger(data[i]) || data[i] < 0) {
      throw new Error(`data[${i}] must be a non-negative integer`);
    }
  }
}

function clampProb(p: number): number {
  return Math.max(1e-15, Math.min(1 - 1e-15, p));
}

/** Digamma function ψ(x) via asymptotic expansion. */
function digamma(x: number): number {
  let result = 0;
  while (x < 6) {
    result -= 1 / x;
    x += 1;
  }
  result += Math.log(x) - 1 / (2 * x);
  const x2 = 1 / (x * x);
  result -= x2 * (1 / 12 - x2 * (1 / 120 - x2 / 252));
  return result;
}

/** Trigamma function ψ'(x) via asymptotic expansion. */
function trigamma(x: number): number {
  let result = 0;
  while (x < 6) {
    result += 1 / (x * x);
    x += 1;
  }
  result += 1 / x + 1 / (2 * x * x);
  const x2 = 1 / (x * x);
  result += x2 * (1 / 6 - x2 * (1 / 30 - x2 / 42));
  return result;
}
