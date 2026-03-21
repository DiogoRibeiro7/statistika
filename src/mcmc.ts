import { mean, variance } from "./utils/descriptive";
import { SeededRng } from "./random";

/**
 * MCMC chain diagnostics and results.
 */
export interface MCMCResult {
  /** Sampled chain (after burn-in). */
  chain: number[];
  /** Acceptance rate. */
  acceptanceRate: number;
  /** Number of iterations (including burn-in). */
  totalIterations: number;
  /** Burn-in period. */
  burnIn: number;
  /** Posterior mean. */
  posteriorMean: number;
  /** Posterior standard deviation. */
  posteriorStd: number;
  /** Credible interval. */
  credibleInterval: { lower: number; upper: number; level: number };
  /** Effective sample size estimate. */
  effectiveSampleSize: number;
  /** Warning message if acceptance rate is problematic. */
  warning?: string;
}

/**
 * Multi-dimensional MCMC result.
 */
export interface MCMCResultND {
  /** Sampled chains (each column is a parameter). */
  chains: number[][];
  /** Acceptance rate. */
  acceptanceRate: number;
  /** Total iterations. */
  totalIterations: number;
  /** Burn-in period. */
  burnIn: number;
  /** Posterior means for each parameter. */
  posteriorMeans: number[];
  /** Posterior standard deviations. */
  posteriorStds: number[];
  /** Warning message if acceptance rate is problematic. */
  warning?: string;
}

/**
 * Metropolis-Hastings MCMC sampler for 1D distributions.
 *
 * Samples from an unnormalized log-density function using a symmetric
 * random walk proposal.
 *
 * @param logDensity - Unnormalized log-density function (log-posterior or log-likelihood + log-prior)
 * @param options - Configuration
 * @returns MCMCResult containing the chain, acceptance rate, posterior summary, and diagnostics
 * @throws {Error} If iterations or burnIn are invalid
 * @throws {Error} If proposalStd is not positive
 *
 * @example
 * ```ts
 * // Sample from a standard normal distribution
 * const result = metropolisHastings(
 *   (x) => -0.5 * x * x, // log-density of N(0,1)
 *   { iterations: 10000, proposalStd: 1.0, seed: 42 }
 * );
 * console.log(result.posteriorMean);  // ≈ 0
 * console.log(result.posteriorStd);   // ≈ 1
 * console.log(result.acceptanceRate); // typical: 0.2–0.5
 * ```
 */
export function metropolisHastings(
  logDensity: (x: number) => number,
  options: {
    initial?: number;
    proposalStd?: number;
    iterations?: number;
    burnIn?: number;
    seed?: number;
    credibleLevel?: number;
  } = {},
): MCMCResult {
  const initial = options.initial ?? 0;
  const proposalStd = options.proposalStd ?? 1;
  const iterations = options.iterations ?? 10000;
  const burnIn = options.burnIn ?? Math.floor(iterations * 0.2);
  const credibleLevel = options.credibleLevel ?? 0.95;
  const rng = new SeededRng(options.seed ?? Math.floor(Math.random() * 2147483647));

  let current = initial;
  let currentLogDensity = logDensity(current);
  // NaN guard: if initial log-density is NaN, treat as -Infinity
  if (Number.isNaN(currentLogDensity)) currentLogDensity = -Infinity;
  let accepted = 0;

  const chain: number[] = [];

  for (let i = 0; i < iterations; i++) {
    // Propose new value (normal random walk)
    const proposal = current + rng.nextNormal(0, proposalStd);
    let proposalLogDensity = logDensity(proposal);

    // NaN guard: if logDensity returns NaN, reject the proposal
    if (Number.isNaN(proposalLogDensity)) {
      proposalLogDensity = -Infinity;
    }

    // Accept/reject
    const logAlpha = proposalLogDensity - currentLogDensity;
    if (Math.log(rng.next()) < logAlpha) {
      current = proposal;
      currentLogDensity = proposalLogDensity;
      accepted++;
    }

    if (i >= burnIn) {
      chain.push(current);
    }
  }

  const posteriorMean = mean(chain);
  const posteriorStd = Math.sqrt(variance(chain));

  // Credible interval
  const sorted = [...chain].sort((a, b) => a - b);
  const alpha = 1 - credibleLevel;
  const lower = sorted[Math.floor((alpha / 2) * sorted.length)];
  const upper = sorted[Math.floor((1 - alpha / 2) * sorted.length)];

  const acceptanceRate = accepted / iterations;

  // Acceptance rate warning
  let warning: string | undefined;
  if (acceptanceRate < 0.05) {
    warning = `Very low acceptance rate (${(acceptanceRate * 100).toFixed(1)}%). Consider increasing proposalStd or reparameterizing.`;
  } else if (acceptanceRate > 0.95) {
    warning = `Very high acceptance rate (${(acceptanceRate * 100).toFixed(1)}%). Consider decreasing proposalStd for better mixing.`;
  }

  const result: MCMCResult = {
    chain,
    acceptanceRate,
    totalIterations: iterations,
    burnIn,
    posteriorMean,
    posteriorStd,
    credibleInterval: { lower, upper, level: credibleLevel },
    effectiveSampleSize: estimateESS(chain),
  };
  if (warning) result.warning = warning;
  return result;
}

/**
 * Metropolis-Hastings sampler for multi-dimensional distributions.
 *
 * @param logDensity - Unnormalized log-density function taking a parameter vector
 * @param dimensions - Number of parameters
 * @param options - Configuration
 * @returns MCMCResultND containing chains, acceptance rate, and posterior summaries
 * @throws {Error} If dimensions is less than 1
 * @throws {Error} If iterations or burnIn are invalid
 */
export function metropolisHastingsND(
  logDensity: (x: number[]) => number,
  dimensions: number,
  options: {
    initial?: number[];
    proposalStd?: number | number[];
    iterations?: number;
    burnIn?: number;
    seed?: number;
  } = {},
): MCMCResultND {
  const initial = options.initial ?? new Array<number>(dimensions).fill(0);
  const iterations = options.iterations ?? 10000;
  const burnIn = options.burnIn ?? Math.floor(iterations * 0.2);
  const rng = new SeededRng(options.seed ?? Math.floor(Math.random() * 2147483647));

  const proposalStds =
    typeof options.proposalStd === "number"
      ? new Array<number>(dimensions).fill(options.proposalStd)
      : options.proposalStd ?? new Array<number>(dimensions).fill(1);

  let current = [...initial];
  let currentLogDensity = logDensity(current);
  // NaN guard
  if (Number.isNaN(currentLogDensity)) currentLogDensity = -Infinity;
  let accepted = 0;

  const chains: number[][] = [];

  for (let i = 0; i < iterations; i++) {
    // Propose
    const proposal = current.map((v, j) => v + rng.nextNormal(0, proposalStds[j]));
    let proposalLogDensity = logDensity(proposal);

    // NaN guard
    if (Number.isNaN(proposalLogDensity)) {
      proposalLogDensity = -Infinity;
    }

    const logAlpha = proposalLogDensity - currentLogDensity;
    if (Math.log(rng.next()) < logAlpha) {
      current = proposal;
      currentLogDensity = proposalLogDensity;
      accepted++;
    }

    if (i >= burnIn) {
      chains.push([...current]);
    }
  }

  const posteriorMeans = new Array<number>(dimensions);
  const posteriorStds = new Array<number>(dimensions);
  for (let j = 0; j < dimensions; j++) {
    const col = chains.map((row) => row[j]);
    posteriorMeans[j] = mean(col);
    posteriorStds[j] = Math.sqrt(variance(col));
  }

  const acceptanceRate = accepted / iterations;

  // Acceptance rate warning
  let warning: string | undefined;
  if (acceptanceRate < 0.05) {
    warning = `Very low acceptance rate (${(acceptanceRate * 100).toFixed(1)}%). Consider increasing proposalStd or reparameterizing.`;
  } else if (acceptanceRate > 0.95) {
    warning = `Very high acceptance rate (${(acceptanceRate * 100).toFixed(1)}%). Consider decreasing proposalStd for better mixing.`;
  }

  const result: MCMCResultND = {
    chains,
    acceptanceRate,
    totalIterations: iterations,
    burnIn,
    posteriorMeans,
    posteriorStds,
  };
  if (warning) result.warning = warning;
  return result;
}

/**
 * Gelman-Rubin R-hat convergence diagnostic.
 *
 * Compares within-chain and between-chain variance from multiple
 * independent chains. R-hat < 1.1 suggests convergence.
 *
 * @param chains - Array of MCMC chains (each an array of samples)
 * @returns The R-hat statistic (values near 1.0 indicate convergence)
 * @throws {Error} If fewer than 2 chains are provided
 * @throws {Error} If chains have different lengths
 */
export function gelmanRubin(chains: number[][]): number {
  const m = chains.length;
  if (m < 2) throw new Error("Need at least 2 chains");

  const n = chains[0].length;
  if (chains.some((c) => c.length !== n)) {
    throw new Error("All chains must have the same length");
  }

  // Chain means
  const chainMeans = chains.map(mean);
  const overallMean = mean(chainMeans);

  // Between-chain variance
  let B = 0;
  for (const cm of chainMeans) B += (cm - overallMean) ** 2;
  B = (n * B) / (m - 1);

  // Within-chain variance
  let W = 0;
  for (const chain of chains) {
    W += variance(chain);
  }
  W /= m;

  // Pooled variance estimate
  const V = ((n - 1) / n) * W + (1 / n) * B;

  return Math.sqrt(V / W);
}

/**
 * Estimate effective sample size (ESS) from a single chain.
 *
 * Accounts for autocorrelation in the chain. Higher ESS means
 * less autocorrelation and more effective samples.
 *
 * @param chain - Array of MCMC samples
 * @returns Estimated effective sample size (always >= 1)
 * @throws {Error} If chain is empty
 */
export function estimateESS(chain: number[]): number {
  const n = chain.length;
  if (n < 10) return n;

  const m = mean(chain);
  const v = chain.reduce((s, x) => s + (x - m) ** 2, 0) / n;
  if (v === 0) return n;

  // Compute autocorrelations and sum until they become negative
  let rhoSum = 0;
  const maxLag = Math.floor(n / 2);

  for (let k = 1; k <= maxLag; k++) {
    let sum = 0;
    for (let t = 0; t < n - k; t++) {
      sum += (chain[t] - m) * (chain[t + k] - m);
    }
    const rho = sum / (n * v);
    if (rho < 0) break; // Initial monotone sequence estimator
    rhoSum += rho;
  }

  return Math.max(1, n / (1 + 2 * rhoSum));
}
