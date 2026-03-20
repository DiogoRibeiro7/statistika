/**
 * Bayesian methods: conjugate priors, posterior inference, and MCMC sampling.
 */

import { gammaLn } from "./utils/math";
import { normalQuantile } from "./utils/linalg";

// ---- Conjugate Prior Models ----

/** Result of a Beta-Binomial conjugate update. */
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
 * @param successes - Number of successes observed
 * @param trials - Number of trials
 * @param priorAlpha - Prior Beta alpha (default 1 = uniform)
 * @param priorBeta - Prior Beta beta (default 1 = uniform)
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

/** Result of a Normal-Normal conjugate update. */
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
 * @param data - Observed data points
 * @param knownVariance - Known data variance (sigma^2)
 * @param priorMean - Prior mean (mu0)
 * @param priorVariance - Prior variance (sigma0^2)
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

/** Result of a Gamma-Poisson conjugate update. */
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
 * Gamma-Poisson conjugate model.
 *
 * Prior: lambda ~ Gamma(alpha, beta)
 * Likelihood: X_i | lambda ~ Poisson(lambda)
 * Posterior: lambda | X ~ Gamma(alpha + sum(x), beta + n)
 *
 * @param data - Observed count data
 * @param priorAlpha - Prior shape (default 1)
 * @param priorBeta - Prior rate (default 1)
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

/** Options for the Metropolis-Hastings sampler. */
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

/** Result of MCMC sampling. */
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
 * @param logPosterior - Log-posterior density function (up to a constant)
 * @param options - Sampler options
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

/** Result of a Bayes factor computation. */
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
 * Uses the harmonic mean estimator from MCMC samples.
 *
 * @param logLikelihood1 - Log-likelihood values from model 1 MCMC chain
 * @param logLikelihood2 - Log-likelihood values from model 2 MCMC chain
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

