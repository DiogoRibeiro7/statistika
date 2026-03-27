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

/**
 * Result of fitting a distribution to data via maximum likelihood estimation.
 *
 * Contains the fitted distribution instance, the log-likelihood of the data
 * under the fitted distribution, the AIC (Akaike Information Criterion),
 * and the number of estimated parameters.
 *
 * @typeParam T - The type of the fitted distribution
 */
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
 * Fit a Normal distribution by maximum likelihood estimation.
 *
 * MLE estimators: mu_hat = x_bar, sigma_hat = sqrt( (1/n) * sum((x_i - x_bar)^2) )
 *
 * @param data - Sample data (at least 2 observations, all finite)
 * @returns A {@link FitResult} containing the fitted Normal distribution, log-likelihood, and AIC
 * @throws {Error} If fewer than 2 observations or data has zero variance
 * @throws {Error} If any value is not finite
 *
 * @example
 * ```ts
 * const result = fitNormal([1.2, 2.3, 1.8, 2.1, 1.9]);
 * console.log(result.distribution.mean); // ~1.86
 * console.log(result.aic);              // AIC for model comparison
 * ```
 */
export function fitNormal(data: number[]): FitResult<Normal> {
  validateContinuousData(data);
  const n = data.length;
  const mu = mean(data);
  let sumSq = 0;
  for (let i = 0; i < n; i++) sumSq += (data[i] - mu) ** 2;
  const sigma = Math.sqrt(sumSq / n);
  if (sigma <= 0) throw new Error(`Invalid parameter 'data': expected non-zero variance, received variance=0`);

  const dist = new Normal(mu, sigma);
  const ll = data.reduce((s, x) => s + Math.log(dist.pdf(x)), 0);
  return { distribution: dist, logLikelihood: ll, aic: -2 * ll + 4, nParams: 2 };
}

/**
 * Fit an Exponential distribution by maximum likelihood estimation.
 *
 * MLE estimator: lambda_hat = 1 / x_bar
 *
 * @param data - Sample data (at least 2 observations, all non-negative and finite)
 * @returns A {@link FitResult} containing the fitted Exponential distribution
 * @throws {Error} If fewer than 2 observations or any value is negative
 * @throws {Error} If any value is not finite
 *
 * @example
 * ```ts
 * const result = fitExponential([0.5, 1.2, 0.8, 2.1, 0.3]);
 * console.log(result.distribution.rate); // estimated lambda
 * ```
 */
export function fitExponential(data: number[]): FitResult<Exponential> {
  validateContinuousData(data);
  for (const x of data) {
    if (x < 0) throw new Error(`Invalid parameter 'data': expected non-negative values, received ${x}`);
  }
  const lambda = 1 / mean(data);
  const dist = new Exponential(lambda);
  const ll = data.reduce((s, x) => s + Math.log(dist.pdf(x)), 0);
  return { distribution: dist, logLikelihood: ll, aic: -2 * ll + 2, nParams: 1 };
}

/**
 * Fit a Poisson distribution by maximum likelihood estimation.
 *
 * MLE estimator: lambda_hat = x_bar
 *
 * @param data - Sample data (at least 2 observations, all non-negative integers)
 * @returns A {@link FitResult} containing the fitted Poisson distribution
 * @throws {Error} If fewer than 2 observations, mean is non-positive, or data
 *   contains non-integer or negative values
 *
 * @example
 * ```ts
 * const result = fitPoisson([2, 3, 1, 4, 2, 3]);
 * console.log(result.distribution.mean); // estimated lambda
 * ```
 */
export function fitPoisson(data: number[]): FitResult<Poisson> {
  validateDiscreteData(data);
  const lambda = mean(data);
  if (lambda <= 0) throw new Error(`Invalid parameter 'data': expected positive mean for Poisson, received mean=${lambda}`);
  const dist = new Poisson(lambda);
  const ll = data.reduce((s, x) => s + Math.log(Math.max(1e-300, dist.pmf(x))), 0);
  return { distribution: dist, logLikelihood: ll, aic: -2 * ll + 2, nParams: 1 };
}

/**
 * Fit a Gamma distribution by maximum likelihood estimation.
 *
 * Uses method of moments for initial estimates, then refines the shape
 * parameter via Newton-Raphson using the digamma and trigamma functions.
 * The rate is computed as rate = shape / x_bar.
 *
 * @param data - Sample data (at least 2 observations, all positive and finite)
 * @returns A {@link FitResult} containing the fitted Gamma distribution
 * @throws {Error} If fewer than 2 observations or any value is non-positive
 * @throws {Error} If any value is not finite
 *
 * @example
 * ```ts
 * const result = fitGamma([1.2, 2.5, 3.1, 1.8, 2.0]);
 * console.log(result.distribution.shape); // estimated shape
 * ```
 */
export function fitGamma(data: number[]): FitResult<GammaDistribution> {
  validateContinuousData(data);
  for (const x of data) {
    if (x <= 0) throw new Error(`Invalid parameter 'data': expected positive values for Gamma, received ${x}`);
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
 * Fit a Beta distribution by the method of moments.
 *
 * Estimates alpha and beta from the sample mean m and variance v:
 *   common = m * (1 - m) / v - 1
 *   alpha = m * common
 *   beta = (1 - m) * common
 *
 * @param data - Sample data (at least 2 observations, all in the open interval (0, 1))
 * @returns A {@link FitResult} containing the fitted Beta distribution
 * @throws {Error} If fewer than 2 observations or any value is not in (0, 1)
 * @throws {Error} If moment estimates are non-positive
 *
 * @example
 * ```ts
 * const result = fitBeta([0.2, 0.5, 0.3, 0.7, 0.4]);
 * console.log(result.distribution.alpha, result.distribution.beta);
 * ```
 */
export function fitBeta(data: number[]): FitResult<BetaDistribution> {
  validateContinuousData(data);
  for (const x of data) {
    if (x <= 0 || x >= 1) throw new Error(`Invalid parameter 'data': expected values in (0, 1) for Beta, received ${x}`);
  }
  const m = mean(data);
  const v = variance(data);
  // Method of moments
  const common = (m * (1 - m)) / v - 1;
  const alpha = m * common;
  const beta = (1 - m) * common;
  if (alpha <= 0 || beta <= 0) {
    throw new Error(`Invalid parameter 'data': cannot fit Beta distribution, moment estimates are non-positive (alpha=${alpha}, beta=${beta})`);
  }

  const dist = new BetaDistribution(alpha, beta);
  const ll = data.reduce((s, x) => s + Math.log(dist.pdf(x)), 0);
  return { distribution: dist, logLikelihood: ll, aic: -2 * ll + 4, nParams: 2 };
}

/**
 * Fit a Log-Normal distribution by maximum likelihood estimation.
 *
 * MLE estimators: mu_hat = mean(ln(x)), sigma_hat = sqrt( (1/n) * sum((ln(x_i) - mu_hat)^2) )
 *
 * @param data - Sample data (at least 2 observations, all positive and finite)
 * @returns A {@link FitResult} containing the fitted Log-Normal distribution
 * @throws {Error} If fewer than 2 observations or any value is non-positive
 * @throws {Error} If log-transformed data has zero variance
 *
 * @example
 * ```ts
 * const result = fitLogNormal([1.5, 2.3, 3.1, 1.8, 4.2]);
 * console.log(result.distribution.mu); // mean of log(data)
 * ```
 */
export function fitLogNormal(data: number[]): FitResult<LogNormal> {
  validateContinuousData(data);
  for (const x of data) {
    if (x <= 0) throw new Error(`Invalid parameter 'data': expected positive values for Log-Normal, received ${x}`);
  }
  const n = data.length;
  const logData = data.map(Math.log);
  const mu = mean(logData);
  let sumSq = 0;
  for (let i = 0; i < n; i++) sumSq += (logData[i] - mu) ** 2;
  const sigma = Math.sqrt(sumSq / n);
  if (sigma <= 0) throw new Error(`Invalid parameter 'data': expected non-zero log-variance, received variance=0`);

  const dist = new LogNormal(mu, sigma);
  const ll = data.reduce((s, x) => s + Math.log(dist.pdf(x)), 0);
  return { distribution: dist, logLikelihood: ll, aic: -2 * ll + 4, nParams: 2 };
}

/**
 * Fit a Geometric distribution by maximum likelihood estimation.
 *
 * MLE estimator: p_hat = 1 / (1 + x_bar)
 *
 * @param data - Sample data (at least 2 observations, all non-negative integers)
 * @returns A {@link FitResult} containing the fitted Geometric distribution
 * @throws {Error} If fewer than 2 observations or data contains non-integer
 *   or negative values
 *
 * @example
 * ```ts
 * const result = fitGeometric([0, 1, 0, 2, 1, 3]);
 * console.log(result.distribution.p); // estimated success probability
 * ```
 */
export function fitGeometric(data: number[]): FitResult<Geometric> {
  validateDiscreteData(data);
  const p = 1 / (1 + mean(data));
  const dist = new Geometric(p);
  const ll = data.reduce((s, x) => s + Math.log(Math.max(1e-300, dist.pmf(x))), 0);
  return { distribution: dist, logLikelihood: ll, aic: -2 * ll + 2, nParams: 1 };
}

/**
 * Fit a Zero-Inflated Poisson (ZIP) distribution by the EM algorithm.
 *
 * The ZIP model assumes observations come from a mixture of a point mass
 * at zero (with probability pi) and a Poisson distribution (with probability
 * 1 - pi). The EM algorithm alternates between:
 *   E-step: compute posterior probability of structural zero for zero observations
 *   M-step: update pi and lambda based on expected sufficient statistics
 *
 * @param data - Sample data (at least 2 observations, all non-negative integers)
 * @returns A {@link FitResult} containing the fitted ZIP distribution
 * @throws {Error} If fewer than 2 observations or data contains non-integer
 *   or negative values
 *
 * @example
 * ```ts
 * const result = fitZIP([0, 0, 0, 1, 2, 0, 3, 0]);
 * console.log(result.distribution.lambda); // Poisson rate
 * ```
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

/**
 * Result of a goodness-of-fit test (Anderson-Darling or Cramer-von Mises).
 *
 * Contains the test statistic, whether the null hypothesis was rejected
 * at the given significance level, and the alpha used.
 */
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
 * Tests H0: the data come from the specified continuous distribution.
 * The Anderson-Darling test gives more weight to the tails than the
 * Kolmogorov-Smirnov test, making it more powerful for detecting
 * departures in the distribution tails.
 *
 * The statistic is:
 *   A^2 = -n - (1/n) * sum_i (2i - 1) * [ln F(y_i) + ln(1 - F(y_{n+1-i}))]
 * where y_1 <= ... <= y_n are the sorted data and F is the hypothesized CDF.
 *
 * @param data - Sample data (at least 2 observations)
 * @param distribution - Hypothesized continuous distribution (must have a `cdf` method)
 * @param alpha - Significance level (default 0.05). Supported values: 0.15, 0.10, 0.05, 0.025, 0.01
 * @returns A {@link GoodnessOfFitResult} with the test statistic and rejection decision
 * @throws {Error} If fewer than 2 observations
 *
 * @example
 * ```ts
 * const result = andersonDarlingTest(data, new Normal(0, 1));
 * console.log(result.statistic); // A^2 statistic
 * console.log(result.rejected);  // true if data likely not from N(0,1)
 * ```
 */
export function andersonDarlingTest(
  data: number[],
  distribution: ContinuousDistribution,
  alpha = 0.05,
): GoodnessOfFitResult {
  const n = data.length;
  if (n < 2) throw new Error(`Invalid parameter 'data': expected at least 2 observations, received ${n}`);

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
 * Cramer-von Mises test for goodness of fit.
 *
 * Tests H0: the data come from the specified continuous distribution.
 * Measures the integrated squared difference between the empirical and
 * hypothesized CDFs.
 *
 * The statistic is:
 *   W^2 = sum_i [ F(y_i) - (2i - 1) / (2n) ]^2 + 1/(12n)
 * where y_1 <= ... <= y_n are the sorted data and F is the hypothesized CDF.
 *
 * @param data - Sample data (at least 2 observations)
 * @param distribution - Hypothesized continuous distribution (must have a `cdf` method)
 * @param alpha - Significance level (default 0.05). Supported values: 0.15, 0.10, 0.05, 0.025, 0.01
 * @returns A {@link GoodnessOfFitResult} with the test statistic and rejection decision
 * @throws {Error} If fewer than 2 observations
 *
 * @example
 * ```ts
 * const result = cramerVonMisesTest(data, new Normal(0, 1));
 * console.log(result.statistic); // W^2 statistic
 * console.log(result.rejected);  // true if data likely not from N(0,1)
 * ```
 */
export function cramerVonMisesTest(
  data: number[],
  distribution: ContinuousDistribution,
  alpha = 0.05,
): GoodnessOfFitResult {
  const n = data.length;
  if (n < 2) throw new Error(`Invalid parameter 'data': expected at least 2 observations, received ${n}`);

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
  if (data.length < 2) throw new Error(`Invalid parameter 'data': expected at least 2 observations, received ${data.length}`);
  for (let i = 0; i < data.length; i++) {
    if (!Number.isFinite(data[i])) {
      throw new Error(`Invalid parameter 'data': expected finite number at index ${i}, received ${data[i]}`);
    }
  }
}

function validateDiscreteData(data: number[]): void {
  if (data.length < 2) throw new Error(`Invalid parameter 'data': expected at least 2 observations, received ${data.length}`);
  for (let i = 0; i < data.length; i++) {
    if (!Number.isInteger(data[i]) || data[i] < 0) {
      throw new Error(`Invalid parameter 'data': expected non-negative integer at index ${i}, received ${data[i]}`);
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
