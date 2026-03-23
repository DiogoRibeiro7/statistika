/**
 * Bayesian methods: conjugate priors, posterior inference, and MCMC sampling.
 */

import { gammaLn } from "./utils/math";
import { normalQuantile } from "./utils/linalg";

// ---- Conjugate Prior Models ----

/**
 * Result of a Beta-Binomial conjugate update.
 *
 * Contains the prior and posterior parameters of the Beta distribution,
 * along with summary statistics (mean, variance, credible interval)
 * and the posterior predictive probability of success.
 */
export interface BetaBinomialResult {
  /** Prior alpha */
  priorAlpha: number;
  /** Prior beta */
  priorBeta: number;
  /** Posterior alpha */
  posteriorAlpha: number;
  /** Posterior beta */
  posteriorBeta: number;
  /** Posterior mean E[p] */
  posteriorMean: number;
  /** Posterior variance */
  posteriorVariance: number;
  /** 95% credible interval [lower, upper] */
  credibleInterval: [number, number];
  /** Posterior predictive probability of success */
  posteriorPredictive: number;
}

/**
 * Beta-Binomial conjugate model for binomial data.
 *
 * Prior: p ~ Beta(alpha, beta)
 * Likelihood: X | p ~ Binomial(n, p)
 * Posterior: p | X ~ Beta(alpha + successes, beta + failures)
 *
 * The posterior mean is E[p | data] = (alpha + successes) / (alpha + beta + trials),
 * and the posterior variance is Var[p | data] = (a' * b') / ((a' + b')^2 * (a' + b' + 1))
 * where a' = alpha + successes, b' = beta + failures.
 *
 * @param successes - Number of successes observed (non-negative integer <= trials)
 * @param trials - Number of trials (non-negative integer)
 * @param priorAlpha - Prior Beta alpha parameter (default 1, giving a uniform prior)
 * @param priorBeta - Prior Beta beta parameter (default 1, giving a uniform prior)
 * @returns A {@link BetaBinomialResult} with posterior parameters and summary statistics
 * @throws {Error} If successes < 0, trials < 0, or successes > trials
 * @throws {Error} If prior parameters are not positive
 *
 * @example
 * ```ts
 * // Observe 7 successes in 10 trials with a uniform prior
 * const result = betaBinomial(7, 10);
 * console.log(result.posteriorMean); // 0.667 (= 8/12)
 * console.log(result.credibleInterval); // approximately [0.39, 0.94]
 * ```
 */
export function betaBinomial(
  successes: number,
  trials: number,
  priorAlpha = 1,
  priorBeta = 1,
): BetaBinomialResult {
  if (successes < 0 || trials < 0 || successes > trials) {
    throw new Error("Invalid successes/trials");
  }
  if (priorAlpha <= 0 || priorBeta <= 0) {
    throw new Error("Prior parameters must be positive");
  }

  const failures = trials - successes;
  const postAlpha = priorAlpha + successes;
  const postBeta = priorBeta + failures;

  const posteriorMean = postAlpha / (postAlpha + postBeta);
  const posteriorVariance =
    (postAlpha * postBeta) /
    ((postAlpha + postBeta) ** 2 * (postAlpha + postBeta + 1));

  // Credible interval via Beta quantile approximation (normal approx for large params)
  const ci = betaCredibleInterval(postAlpha, postBeta, 0.95);

  return {
    priorAlpha,
    priorBeta,
    posteriorAlpha: postAlpha,
    posteriorBeta: postBeta,
    posteriorMean,
    posteriorVariance,
    credibleInterval: ci,
    posteriorPredictive: posteriorMean,
  };
}

/**
 * Result of a Normal-Normal conjugate update.
 *
 * Contains the prior and posterior parameters of the Normal distribution
 * for the unknown mean, along with a 95% credible interval.
 */
export interface NormalNormalResult {
  /** Prior mean */
  priorMean: number;
  /** Prior variance */
  priorVariance: number;
  /** Posterior mean */
  posteriorMean: number;
  /** Posterior variance */
  posteriorVariance: number;
  /** 95% credible interval */
  credibleInterval: [number, number];
}

/**
 * Normal-Normal conjugate model for a normal mean with known variance.
 *
 * Prior: mu ~ N(mu0, sigma0^2)
 * Likelihood: X_i | mu ~ N(mu, sigma^2)   (sigma known)
 * Posterior: mu | X ~ N(mu_n, sigma_n^2)
 *
 * The posterior precision is tau_n = 1/sigma0^2 + n/sigma^2,
 * and the posterior mean is mu_n = (mu0/sigma0^2 + n*x_bar/sigma^2) / tau_n.
 *
 * @param data - Observed data points (must be non-empty)
 * @param knownVariance - Known data variance sigma^2 (must be positive)
 * @param priorMean - Prior mean mu0 (default 0)
 * @param priorVariance - Prior variance sigma0^2 (default 1000, a diffuse prior)
 * @returns A {@link NormalNormalResult} with posterior mean, variance, and 95% credible interval
 * @throws {Error} If data is empty
 * @throws {Error} If knownVariance or priorVariance is not positive
 *
 * @example
 * ```ts
 * const result = normalNormal([5.1, 4.9, 5.0, 5.2], 0.25);
 * console.log(result.posteriorMean); // close to 5.05
 * ```
 */
export function normalNormal(
  data: number[],
  knownVariance: number,
  priorMean = 0,
  priorVariance = 1000,
): NormalNormalResult {
  if (data.length === 0) throw new Error("Data must not be empty");
  if (knownVariance <= 0) throw new Error("Known variance must be positive");
  if (priorVariance <= 0) throw new Error("Prior variance must be positive");

  const n = data.length;
  const dataMean = data.reduce((a, b) => a + b, 0) / n;

  const priorPrecision = 1 / priorVariance;
  const likePrecision = n / knownVariance;
  const postPrecision = priorPrecision + likePrecision;
  const postVariance = 1 / postPrecision;
  const postMean =
    (priorPrecision * priorMean + likePrecision * dataMean) / postPrecision;

  const z = 1.96;
  const postStd = Math.sqrt(postVariance);

  return {
    priorMean,
    priorVariance,
    posteriorMean: postMean,
    posteriorVariance: postVariance,
    credibleInterval: [postMean - z * postStd, postMean + z * postStd],
  };
}

/**
 * Result of a Gamma-Poisson conjugate update.
 *
 * Contains the prior and posterior parameters of the Gamma distribution
 * for the Poisson rate parameter lambda, along with summary statistics.
 */
export interface GammaPoissonResult {
  /** Prior shape (alpha) */
  priorAlpha: number;
  /** Prior rate (beta) */
  priorBeta: number;
  /** Posterior shape */
  posteriorAlpha: number;
  /** Posterior rate */
  posteriorBeta: number;
  /** Posterior mean E[lambda] */
  posteriorMean: number;
  /** Posterior variance */
  posteriorVariance: number;
  /** 95% credible interval */
  credibleInterval: [number, number];
}

/**
 * Gamma-Poisson conjugate model for Poisson count data.
 *
 * Prior: lambda ~ Gamma(alpha, beta)
 * Likelihood: X_i | lambda ~ Poisson(lambda)
 * Posterior: lambda | X ~ Gamma(alpha + sum(x), beta + n)
 *
 * The posterior mean is E[lambda | data] = (alpha + sum(x)) / (beta + n),
 * and the posterior variance is (alpha + sum(x)) / (beta + n)^2.
 *
 * @param data - Observed count data (non-negative integers, must be non-empty)
 * @param priorAlpha - Prior shape parameter (default 1, must be positive)
 * @param priorBeta - Prior rate parameter (default 1, must be positive)
 * @returns A {@link GammaPoissonResult} with posterior parameters and summary statistics
 * @throws {Error} If data is empty
 * @throws {Error} If prior parameters are not positive
 * @throws {Error} If any data value is not a non-negative integer
 *
 * @example
 * ```ts
 * const result = gammaPoisson([3, 5, 2, 4, 6]);
 * console.log(result.posteriorMean); // (1 + 20) / (1 + 5) = 3.5
 * ```
 */
export function gammaPoisson(
  data: number[],
  priorAlpha = 1,
  priorBeta = 1,
): GammaPoissonResult {
  if (data.length === 0) throw new Error("Data must not be empty");
  if (priorAlpha <= 0 || priorBeta <= 0) {
    throw new Error("Prior parameters must be positive");
  }
  for (const x of data) {
    if (x < 0 || !Number.isInteger(x)) {
      throw new Error("Data must be non-negative integers");
    }
  }

  const n = data.length;
  const total = data.reduce((a, b) => a + b, 0);
  const postAlpha = priorAlpha + total;
  const postBeta = priorBeta + n;

  const posteriorMean = postAlpha / postBeta;
  const posteriorVariance = postAlpha / (postBeta * postBeta);

  // Gamma credible interval via normal approximation
  const postStd = Math.sqrt(posteriorVariance);
  const z = 1.96;

  return {
    priorAlpha,
    priorBeta,
    posteriorAlpha: postAlpha,
    posteriorBeta: postBeta,
    posteriorMean,
    posteriorVariance,
    credibleInterval: [
      Math.max(0, posteriorMean - z * postStd),
      posteriorMean + z * postStd,
    ],
  };
}

// ---- MCMC: Metropolis-Hastings ----

/**
 * Options for the Metropolis-Hastings MCMC sampler.
 *
 * Controls the number of samples, burn-in period, thinning interval,
 * proposal distribution width, initial value, and random number generator.
 */
export interface MHOptions {
  /** Number of samples to draw */
  nSamples: number;
  /** Number of burn-in samples to discard */
  burnIn?: number;
  /** Thinning interval (keep every nth sample) */
  thin?: number;
  /** Proposal standard deviation */
  proposalStd?: number;
  /** Initial value */
  initial?: number;
  /** Random seed function (for reproducibility) */
  random?: () => number;
}

/**
 * Result of MCMC sampling.
 *
 * Contains the posterior samples, acceptance rate, and summary statistics
 * including the posterior mean, standard deviation, and 95% credible interval.
 */
export interface MCMCResult {
  /** Posterior samples */
  samples: number[];
  /** Acceptance rate */
  acceptanceRate: number;
  /** Posterior mean */
  mean: number;
  /** Posterior standard deviation */
  std: number;
  /** 95% credible interval (from quantiles) */
  credibleInterval: [number, number];
}

/**
 * Metropolis-Hastings MCMC sampler for a univariate distribution.
 *
 * Draws samples from a target distribution specified by its log-posterior
 * density (up to a normalizing constant). Uses a symmetric Normal random-walk
 * proposal: x* = x_t + N(0, proposalStd^2). The acceptance probability is
 * alpha = min(1, exp(logPosterior(x*) - logPosterior(x_t))).
 *
 * @param logPosterior - Log-posterior density function (up to an additive constant)
 * @param options - Sampler configuration options (see {@link MHOptions})
 * @returns A {@link MCMCResult} with posterior samples and summary statistics
 * @throws {Error} If nSamples < 1, thin < 1, or proposalStd <= 0
 *
 * @example
 * ```ts
 * // Sample from a standard normal distribution
 * const result = metropolisHastings(
 *   (x) => -0.5 * x * x,
 *   { nSamples: 10000, proposalStd: 1.0 }
 * );
 * console.log(result.mean);           // close to 0
 * console.log(result.acceptanceRate); // ideally around 0.23-0.44
 * ```
 */
export function metropolisHastings(
  logPosterior: (x: number) => number,
  options: MHOptions,
): MCMCResult {
  const {
    nSamples,
    burnIn = Math.floor(nSamples / 4),
    thin = 1,
    proposalStd = 1,
    initial = 0,
    random = Math.random,
  } = options;

  if (nSamples < 1) throw new Error("nSamples must be at least 1");
  if (thin < 1) throw new Error("thin must be at least 1");
  if (proposalStd <= 0) throw new Error("proposalStd must be positive");

  const totalIterations = burnIn + nSamples * thin;
  const samples: number[] = [];
  let current = initial;
  let currentLogP = logPosterior(current);
  let accepted = 0;
  let totalProposals = 0;

  for (let i = 0; i < totalIterations; i++) {
    // Normal proposal via Box-Muller
    const u1 = random();
    const u2 = random();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    const proposed = current + proposalStd * z;

    const proposedLogP = logPosterior(proposed);
    const logAlpha = proposedLogP - currentLogP;

    totalProposals++;
    if (Math.log(random()) < logAlpha) {
      current = proposed;
      currentLogP = proposedLogP;
      accepted++;
    }

    // After burn-in, collect samples with thinning
    if (i >= burnIn && (i - burnIn) % thin === 0) {
      samples.push(current);
    }
  }

  // Compute summary statistics
  const sampleMean = samples.reduce((a, b) => a + b, 0) / samples.length;
  let variance = 0;
  for (const s of samples) {
    variance += (s - sampleMean) ** 2;
  }
  variance /= samples.length - 1;

  // Credible interval from sorted samples
  const sorted = [...samples].sort((a, b) => a - b);
  const lo = sorted[Math.floor(0.025 * sorted.length)];
  const hi = sorted[Math.min(Math.floor(0.975 * sorted.length), sorted.length - 1)];

  return {
    samples,
    acceptanceRate: accepted / totalProposals,
    mean: sampleMean,
    std: Math.sqrt(variance),
    credibleInterval: [lo, hi],
  };
}

// ---- Bayes Factor ----

/**
 * Result of a Bayes factor computation.
 *
 * The Bayes factor B_10 quantifies evidence for model 1 over model 0.
 * Values > 1 favor model 1; values < 1 favor model 0.
 */
export interface BayesFactorResult {
  /** Bayes factor B_10 (model 1 vs model 0) */
  bayesFactor: number;
  /** Log Bayes factor */
  logBayesFactor: number;
  /** Interpretation */
  interpretation: string;
}

/**
 * Compute Bayes factor via marginal likelihood estimates.
 *
 * Uses the harmonic mean estimator from MCMC samples:
 *   p(D|M) approx 1 / mean(1/L_i)
 * where L_i are the likelihood values at each MCMC sample. The Bayes factor
 * is B_10 = p(D|M1) / p(D|M2).
 *
 * Note: the harmonic mean estimator can have high variance; use with caution.
 *
 * @param logLikelihood1 - Log-likelihood values from model 1 MCMC chain (must be non-empty)
 * @param logLikelihood2 - Log-likelihood values from model 2 MCMC chain (must be non-empty)
 * @returns A {@link BayesFactorResult} with the Bayes factor, log Bayes factor, and interpretation
 * @throws {Error} If either log-likelihood array is empty
 *
 * @example
 * ```ts
 * const bf = bayesFactor(model1LogLikelihoods, model2LogLikelihoods);
 * console.log(bf.bayesFactor);    // e.g., 15.3
 * console.log(bf.interpretation); // "Strong"
 * ```
 */
export function bayesFactor(
  logLikelihood1: number[],
  logLikelihood2: number[],
): BayesFactorResult {
  if (logLikelihood1.length === 0 || logLikelihood2.length === 0) {
    throw new Error("Both log-likelihood arrays must be non-empty");
  }

  // Harmonic mean estimator of marginal likelihood:
  // 1/p(D|M) ≈ (1/n) * sum(1/L_i)  =>  p(D|M) ≈ 1 / mean(1/L_i)
  // In log space: log(p(D|M)) ≈ -logSumExp(-logL) + log(n)
  const logMarginal1 = logMarginalLikelihood(logLikelihood1);
  const logMarginal2 = logMarginalLikelihood(logLikelihood2);

  const logBF = logMarginal1 - logMarginal2;
  const bf = Math.exp(logBF);

  let interpretation: string;
  const absBF = Math.abs(logBF);
  if (absBF < Math.log(1)) interpretation = "No evidence";
  else if (absBF < Math.log(3)) interpretation = "Anecdotal";
  else if (absBF < Math.log(10)) interpretation = "Moderate";
  else if (absBF < Math.log(30)) interpretation = "Strong";
  else if (absBF < Math.log(100)) interpretation = "Very strong";
  else interpretation = "Decisive";

  return { bayesFactor: bf, logBayesFactor: logBF, interpretation };
}

function logMarginalLikelihood(logLikelihoods: number[]): number {
  const n = logLikelihoods.length;
  // Harmonic mean in log space: -logSumExp(-logL) + log(n)
  const negLogL = logLikelihoods.map((l) => -l);
  const maxNegLogL = Math.max(...negLogL);
  let sumExp = 0;
  for (const v of negLogL) {
    sumExp += Math.exp(v - maxNegLogL);
  }
  const logSumExp = maxNegLogL + Math.log(sumExp);
  return Math.log(n) - logSumExp;
}

// ---- Helpers ----

/**
 * Beta credible interval via quantile approximation.
 * Uses the normal approximation to the Beta distribution.
 */
function betaCredibleInterval(
  alpha: number,
  beta: number,
  level: number,
): [number, number] {
  const mean = alpha / (alpha + beta);
  const variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1));
  const std = Math.sqrt(variance);

  // z-value for the given level
  const z = normalQuantile((1 + level) / 2);

  return [Math.max(0, mean - z * std), Math.min(1, mean + z * std)];
}

