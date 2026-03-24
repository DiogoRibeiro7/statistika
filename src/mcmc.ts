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
  if (m < 2) throw new Error(`Invalid parameter 'chains': expected at least 2 chains, received ${m}`);

  const n = chains[0].length;
  if (chains.some((c) => c.length !== n)) {
    throw new Error(`Invalid parameter 'chains': expected all chains to have the same length (${n}), received chains with different lengths`);
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

// ── Gibbs Sampling ──────────────────────────────────────────────────────

/**
 * Result of a Gibbs sampling run.
 */
export interface GibbsSamplerResult {
  /** Sampled chains: chains[i] is the parameter vector at sample i. */
  chains: number[][];
  /** Posterior means for each parameter. */
  posteriorMeans: number[];
  /** Posterior standard deviations for each parameter. */
  posteriorStds: number[];
  /** Acceptance rates per parameter (if applicable). */
  acceptanceRates?: number[];
  /** Effective sample sizes per parameter. */
  effectiveSampleSizes: number[];
  /** Number of burn-in iterations discarded. */
  burnIn: number;
  /** Total number of iterations (including burn-in). */
  totalIterations: number;
}

/**
 * Component-wise Gibbs sampler.
 *
 * Iterates over each parameter in turn, sampling from its full conditional
 * distribution given all other parameters at their current values.
 *
 * @param conditionals - Array of conditional sampling functions. Each function
 *   receives the current state vector and returns a new value for that parameter.
 * @param options - Configuration options
 * @returns GibbsSamplerResult containing chains and summary statistics
 * @throws {Error} If conditionals array is empty
 *
 * @example
 * ```ts
 * // Sample from a bivariate normal with correlation
 * const rng = new SeededRng(42);
 * const result = gibbsSampler(
 *   [
 *     (state) => rng.nextNormal(0.5 * state[1], Math.sqrt(0.75)),
 *     (state) => rng.nextNormal(0.5 * state[0], Math.sqrt(0.75)),
 *   ],
 *   { iterations: 5000, burnIn: 1000, seed: 42 }
 * );
 * ```
 */
export function gibbsSampler(
  conditionals: Array<(currentState: number[]) => number>,
  options: {
    iterations?: number;
    burnIn?: number;
    seed?: number;
    thinning?: number;
    initial?: number[];
  } = {},
): GibbsSamplerResult {
  const numParams = conditionals.length;
  if (numParams === 0) throw new Error(`Invalid parameter 'conditionals': expected a non-empty array, received length 0`);

  const iterations = options.iterations ?? 10000;
  const burnIn = options.burnIn ?? Math.floor(iterations * 0.2);
  const thinning = options.thinning ?? 1;
  const initial = options.initial ?? new Array<number>(numParams).fill(0);

  const state = [...initial];
  const chains: number[][] = [];

  for (let i = 0; i < iterations; i++) {
    // Sweep through all parameters
    for (let j = 0; j < numParams; j++) {
      state[j] = conditionals[j](state);
    }

    if (i >= burnIn && (i - burnIn) % thinning === 0) {
      chains.push([...state]);
    }
  }

  // Compute summaries
  const posteriorMeans = new Array<number>(numParams);
  const posteriorStds = new Array<number>(numParams);
  const effectiveSampleSizes = new Array<number>(numParams);

  for (let j = 0; j < numParams; j++) {
    const col = chains.map((row) => row[j]);
    posteriorMeans[j] = mean(col);
    posteriorStds[j] = Math.sqrt(variance(col));
    effectiveSampleSizes[j] = estimateESS(col);
  }

  return {
    chains,
    posteriorMeans,
    posteriorStds,
    effectiveSampleSizes,
    burnIn,
    totalIterations: iterations,
  };
}

/**
 * Block Gibbs sampler.
 *
 * Instead of updating one parameter at a time, updates blocks of parameters
 * jointly. Useful when parameters within a block are highly correlated.
 *
 * @param blocks - Array of block sampling functions. Each function receives the
 *   current full state vector and returns the updated values for that block.
 * @param blockIndices - Array of index arrays specifying which parameters each
 *   block updates. blockIndices[i] lists the parameter indices updated by blocks[i].
 * @param numParams - Total number of parameters in the state vector
 * @param options - Configuration options
 * @returns GibbsSamplerResult containing chains and summary statistics
 * @throws {Error} If blocks and blockIndices have different lengths
 * @throws {Error} If numParams is less than 1
 *
 * @example
 * ```ts
 * const result = blockGibbsSampler(
 *   [
 *     (state) => [rng.nextNormal(0, 1), rng.nextNormal(0, 1)],
 *     (state) => [rng.nextNormal(state[0], 1)],
 *   ],
 *   [[0, 1], [2]],
 *   3,
 *   { iterations: 5000, seed: 42 }
 * );
 * ```
 */
export function blockGibbsSampler(
  blocks: Array<(currentState: number[]) => number[]>,
  blockIndices: number[][],
  numParams: number,
  options: {
    iterations?: number;
    burnIn?: number;
    seed?: number;
    thinning?: number;
    initial?: number[];
  } = {},
): GibbsSamplerResult {
  if (blocks.length !== blockIndices.length) {
    throw new Error(`Invalid parameters 'blocks', 'blockIndices': expected same length, received blocks.length=${blocks.length}, blockIndices.length=${blockIndices.length}`);
  }
  if (numParams < 1) throw new Error(`Invalid parameter 'numParams': expected at least 1, received ${numParams}`);

  const iterations = options.iterations ?? 10000;
  const burnIn = options.burnIn ?? Math.floor(iterations * 0.2);
  const thinning = options.thinning ?? 1;
  const initial = options.initial ?? new Array<number>(numParams).fill(0);

  const state = [...initial];
  const chains: number[][] = [];

  for (let i = 0; i < iterations; i++) {
    // Sweep through all blocks
    for (let b = 0; b < blocks.length; b++) {
      const blockValues = blocks[b](state);
      const indices = blockIndices[b];
      for (let k = 0; k < indices.length; k++) {
        state[indices[k]] = blockValues[k];
      }
    }

    if (i >= burnIn && (i - burnIn) % thinning === 0) {
      chains.push([...state]);
    }
  }

  // Compute summaries
  const posteriorMeans = new Array<number>(numParams);
  const posteriorStds = new Array<number>(numParams);
  const effectiveSampleSizes = new Array<number>(numParams);

  for (let j = 0; j < numParams; j++) {
    const col = chains.map((row) => row[j]);
    posteriorMeans[j] = mean(col);
    posteriorStds[j] = Math.sqrt(variance(col));
    effectiveSampleSizes[j] = estimateESS(col);
  }

  return {
    chains,
    posteriorMeans,
    posteriorStds,
    effectiveSampleSizes,
    burnIn,
    totalIterations: iterations,
  };
}

/**
 * Hierarchical Normal model via Gibbs sampling.
 *
 * Fits a Normal-Inverse-Gamma hierarchical model:
 *   y_ij ~ Normal(theta_j, sigma2)
 *   theta_j ~ Normal(mu, tau2)
 *   mu ~ Normal(mu0, sigma2_0)   [prior on overall mean]
 *   sigma2 ~ InverseGamma(a0, b0) [prior on observation variance]
 *
 * The sampler draws from conjugate full conditionals for:
 *   - Each group mean theta_j
 *   - The overall mean mu
 *   - The observation variance sigma2
 *
 * @param data - Array of groups, where each group is an array of observations
 * @param options - Configuration and prior hyperparameters
 * @returns GibbsSamplerResult where parameters are [theta_1, ..., theta_J, mu, sigma2]
 * @throws {Error} If data is empty or any group is empty
 *
 * @example
 * ```ts
 * const data = [
 *   [1.2, 1.5, 1.1],
 *   [2.3, 2.1, 2.5],
 *   [1.8, 1.9, 2.0],
 * ];
 * const result = hierarchicalNormalGibbs(data, { iterations: 5000, seed: 42 });
 * // result.posteriorMeans contains [theta1, theta2, theta3, mu, sigma2]
 * ```
 */
export function hierarchicalNormalGibbs(
  data: number[][],
  options: {
    iterations?: number;
    burnIn?: number;
    seed?: number;
    thinning?: number;
    /** Prior mean for overall mean mu. Default 0. */
    mu0?: number;
    /** Prior variance for overall mean mu. Default 1e6 (vague). */
    sigma2_0?: number;
    /** Prior shape for sigma2 (InverseGamma). Default 0.01. */
    a0?: number;
    /** Prior rate for sigma2 (InverseGamma). Default 0.01. */
    b0?: number;
    /** Prior variance for group means around mu. Default 1e6 (vague). */
    tau2?: number;
  } = {},
): GibbsSamplerResult {
  const J = data.length;
  if (J === 0) throw new Error("data must contain at least one group");
  for (let j = 0; j < J; j++) {
    if (data[j].length === 0) throw new Error(`Group ${j} is empty`);
  }

  const iterations = options.iterations ?? 10000;
  const burnIn = options.burnIn ?? Math.floor(iterations * 0.2);
  const thinning = options.thinning ?? 1;
  const rng = new SeededRng(options.seed ?? Math.floor(Math.random() * 2147483647));

  // Prior hyperparameters
  const mu0 = options.mu0 ?? 0;
  const sigma2_0 = options.sigma2_0 ?? 1e6;
  const a0 = options.a0 ?? 0.01;
  const b0 = options.b0 ?? 0.01;
  const tau2 = options.tau2 ?? 1e6;

  // Precompute group statistics
  const groupSums = data.map((g) => g.reduce((s, x) => s + x, 0));
  const groupSizes = data.map((g) => g.length);
  const totalN = groupSizes.reduce((s, n) => s + n, 0);

  // Parameters: [theta_0, ..., theta_{J-1}, mu, sigma2]
  const numParams = J + 2;

  // Initialize
  const theta = new Array<number>(J);
  for (let j = 0; j < J; j++) {
    theta[j] = groupSums[j] / groupSizes[j]; // group sample means
  }
  let mu = mean(theta);
  let sigma2 = 1.0;

  const chains: number[][] = [];

  for (let i = 0; i < iterations; i++) {
    // 1. Sample each theta_j | mu, sigma2, data
    for (let j = 0; j < J; j++) {
      const nj = groupSizes[j];
      const precLikelihood = nj / sigma2;
      const precPrior = 1 / tau2;
      const precPost = precLikelihood + precPrior;
      const meanPost = (precLikelihood * (groupSums[j] / nj) + precPrior * mu) / precPost;
      const sdPost = Math.sqrt(1 / precPost);
      theta[j] = rng.nextNormal(meanPost, sdPost);
    }

    // 2. Sample mu | theta, sigma2
    {
      const precPrior = 1 / sigma2_0;
      const precLikelihood = J / tau2;
      const precPost = precPrior + precLikelihood;
      const thetaSum = theta.reduce((s, t) => s + t, 0);
      const meanPost = (precPrior * mu0 + precLikelihood * (thetaSum / J)) / precPost;
      const sdPost = Math.sqrt(1 / precPost);
      mu = rng.nextNormal(meanPost, sdPost);
    }

    // 3. Sample sigma2 | theta, data  (Inverse-Gamma conjugate update)
    {
      const aPost = a0 + totalN / 2;
      let sse = 0;
      for (let j = 0; j < J; j++) {
        for (const y of data[j]) {
          sse += (y - theta[j]) ** 2;
        }
      }
      const bPost = b0 + sse / 2;
      // Sample from InverseGamma(aPost, bPost) via Gamma:
      // If X ~ Gamma(aPost, 1/bPost) then 1/X ~ InverseGamma(aPost, bPost)
      sigma2 = 1 / sampleGamma(rng, aPost, 1 / bPost);
      if (!Number.isFinite(sigma2) || sigma2 <= 0) {
        sigma2 = 1e-6; // safeguard
      }
    }

    if (i >= burnIn && (i - burnIn) % thinning === 0) {
      const sample = new Array<number>(numParams);
      for (let j = 0; j < J; j++) sample[j] = theta[j];
      sample[J] = mu;
      sample[J + 1] = sigma2;
      chains.push(sample);
    }
  }

  // Compute summaries
  const posteriorMeans = new Array<number>(numParams);
  const posteriorStds = new Array<number>(numParams);
  const effectiveSampleSizes = new Array<number>(numParams);

  for (let j = 0; j < numParams; j++) {
    const col = chains.map((row) => row[j]);
    posteriorMeans[j] = mean(col);
    posteriorStds[j] = Math.sqrt(variance(col));
    effectiveSampleSizes[j] = estimateESS(col);
  }

  return {
    chains,
    posteriorMeans,
    posteriorStds,
    effectiveSampleSizes,
    burnIn,
    totalIterations: iterations,
  };
}

/**
 * Result of chain diagnostic analysis.
 */
export interface ChainDiagnosticsResult {
  /** Autocorrelations for each parameter at lags 0, 1, 2, ... */
  autocorrelations: number[][];
  /** Effective sample size for each parameter. */
  effectiveSampleSizes: number[];
  /** Trace plot data: the raw chain values for each parameter. */
  tracePlotData: number[][];
}

/**
 * Compute convergence diagnostics for Gibbs sampler chains.
 *
 * Computes autocorrelation, effective sample size, and extracts trace
 * plot data for each parameter in the chain output.
 *
 * @param chains - chains from GibbsSamplerResult (chains[i] is sample i, chains[i][j] is parameter j)
 * @returns ChainDiagnosticsResult with autocorrelations, ESS, and trace data
 * @throws {Error} If chains array is empty
 *
 * @example
 * ```ts
 * const result = gibbsSampler(conditionals, { iterations: 5000 });
 * const diag = chainDiagnostics(result.chains);
 * console.log(diag.effectiveSampleSizes); // ESS per parameter
 * console.log(diag.autocorrelations[0]);  // autocorrelation of parameter 0
 * ```
 */
export function chainDiagnostics(chains: number[][]): ChainDiagnosticsResult {
  if (chains.length === 0) throw new Error("chains must not be empty");

  const numSamples = chains.length;
  const numParams = chains[0].length;

  const autocorrelations: number[][] = [];
  const effectiveSampleSizes: number[] = [];
  const tracePlotData: number[][] = [];

  const maxLag = Math.min(Math.floor(numSamples / 2), 100);

  for (let j = 0; j < numParams; j++) {
    // Extract column j
    const col = chains.map((row) => row[j]);
    tracePlotData.push(col);

    // ESS
    effectiveSampleSizes.push(estimateESS(col));

    // Autocorrelation at various lags
    const m = mean(col);
    const v = col.reduce((s, x) => s + (x - m) ** 2, 0) / numSamples;
    const acf: number[] = [];

    if (v === 0) {
      for (let k = 0; k <= maxLag; k++) acf.push(k === 0 ? 1 : 0);
    } else {
      for (let k = 0; k <= maxLag; k++) {
        let sum = 0;
        for (let t = 0; t < numSamples - k; t++) {
          sum += (col[t] - m) * (col[t + k] - m);
        }
        acf.push(sum / (numSamples * v));
      }
    }

    autocorrelations.push(acf);
  }

  return {
    autocorrelations,
    effectiveSampleSizes,
    tracePlotData,
  };
}

// ── Gibbs Sampling Helpers ──────────────────────────────────────────────

/**
 * Sample from a Gamma(shape, rate) distribution using Marsaglia and Tsang's method.
 * For shape >= 1, uses the direct method. For shape < 1, uses the
 * transformation X = Gamma(shape+1, rate) * U^(1/shape).
 */
function sampleGamma(rng: SeededRng, shape: number, rate: number): number {
  if (shape < 1) {
    // Boost: Gamma(a) = Gamma(a+1) * U^(1/a)
    const g = sampleGamma(rng, shape + 1, 1);
    const u = rng.next();
    return (g * Math.pow(u, 1 / shape)) * rate;
  }

  // Marsaglia and Tsang's method for shape >= 1
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);

  for (;;) {
    let x: number;
    let v: number;
    do {
      x = rng.nextNormal(0, 1);
      v = 1 + c * x;
    } while (v <= 0);

    v = v * v * v;
    const u = rng.next();
    const x2 = x * x;

    if (u < 1 - 0.0331 * x2 * x2) {
      return d * v * rate;
    }
    if (Math.log(u) < 0.5 * x2 + d * (1 - v + Math.log(v))) {
      return d * v * rate;
    }
  }
}

// ── Hamiltonian Monte Carlo & NUTS ──────────────────────────────────────

/**
 * Result of a Hamiltonian Monte Carlo or NUTS sampling run.
 */
export interface HMCResult {
  /** Sampled chains: chains[i] is the parameter vector at sample i. */
  chains: number[][];
  /** Acceptance rate. */
  acceptanceRate: number;
  /** Posterior means for each parameter. */
  posteriorMeans: number[];
  /** Posterior standard deviations for each parameter. */
  posteriorStds: number[];
  /** Total number of iterations (including burn-in). */
  totalIterations: number;
  /** Number of burn-in iterations discarded. */
  burnIn: number;
  /** Effective sample sizes per parameter. */
  effectiveSampleSizes: number[];
  /** Number of divergent transitions detected. */
  divergences: number;
}

/**
 * Compute the numerical gradient of a log-density function using central differences.
 *
 * @param logDensity - The log-density function
 * @param x - Point at which to evaluate the gradient
 * @param epsilon - Step size for finite differences (default 1e-6)
 * @returns Approximate gradient vector
 */
export function numericalGradient(
  logDensity: (x: number[]) => number,
  x: number[],
  epsilon: number = 1e-6,
): number[] {
  const d = x.length;
  const grad = new Array<number>(d);
  for (let i = 0; i < d; i++) {
    const xPlus = [...x];
    const xMinus = [...x];
    xPlus[i] += epsilon;
    xMinus[i] -= epsilon;
    grad[i] = (logDensity(xPlus) - logDensity(xMinus)) / (2 * epsilon);
  }
  return grad;
}

/**
 * Leapfrog integrator for Hamiltonian dynamics.
 *
 * Performs symplectic integration of Hamilton's equations using the
 * Stormer-Verlet (leapfrog) method.
 *
 * @param q - Current position
 * @param p - Current momentum
 * @param gradLogDensity - Gradient of the log-density (provides the force)
 * @param stepSize - Integration step size (epsilon)
 * @param nSteps - Number of leapfrog steps
 * @param massMatrixInv - Inverse diagonal mass matrix (default: identity)
 * @returns Tuple of [newPosition, newMomentum]
 */
function leapfrog(
  q: number[],
  p: number[],
  gradLogDensity: (q: number[]) => number[],
  stepSize: number,
  nSteps: number,
  massMatrixInv?: number[],
): [number[], number[]] {
  const d = q.length;
  const newQ = [...q];
  const newP = [...p];
  const mInv = massMatrixInv ?? new Array<number>(d).fill(1);

  // Half step for momentum
  const grad0 = gradLogDensity(newQ);
  for (let i = 0; i < d; i++) {
    newP[i] += 0.5 * stepSize * grad0[i];
  }

  // Full steps for position and momentum
  for (let s = 0; s < nSteps - 1; s++) {
    for (let i = 0; i < d; i++) {
      newQ[i] += stepSize * mInv[i] * newP[i];
    }
    const grad = gradLogDensity(newQ);
    for (let i = 0; i < d; i++) {
      newP[i] += stepSize * grad[i];
    }
  }

  // Final full step for position
  for (let i = 0; i < d; i++) {
    newQ[i] += stepSize * mInv[i] * newP[i];
  }

  // Half step for momentum
  const gradFinal = gradLogDensity(newQ);
  for (let i = 0; i < d; i++) {
    newP[i] += 0.5 * stepSize * gradFinal[i];
  }

  // Negate momentum for reversibility
  for (let i = 0; i < d; i++) {
    newP[i] = -newP[i];
  }

  return [newQ, newP];
}

/**
 * Compute kinetic energy for given momentum and mass matrix.
 */
function kineticEnergy(p: number[], massMatrixInv?: number[]): number {
  let ke = 0;
  if (massMatrixInv) {
    for (let i = 0; i < p.length; i++) {
      ke += massMatrixInv[i] * p[i] * p[i];
    }
  } else {
    for (let i = 0; i < p.length; i++) {
      ke += p[i] * p[i];
    }
  }
  return 0.5 * ke;
}

/**
 * Dual averaging for step size adaptation.
 *
 * Implements the dual averaging scheme of Nesterov (2009) as used in
 * the NUTS paper for adapting the leapfrog step size during warmup.
 *
 * @param targetRate - Target acceptance rate (default 0.8)
 * @param options - Tuning parameters
 * @returns Object with update and finalStepSize methods
 */
function dualAveraging(
  targetRate: number = 0.8,
  options: {
    gamma?: number;
    t0?: number;
    kappa?: number;
    initialStepSize?: number;
  } = {},
): {
  update: (acceptProb: number) => number;
  finalStepSize: () => number;
} {
  const gamma = options.gamma ?? 0.05;
  const t0 = options.t0 ?? 10;
  const kappa = options.kappa ?? 0.75;
  const logEps0 = Math.log(options.initialStepSize ?? 1);

  let logEpsBar = 0;
  let hBar = 0;
  let mu = Math.log(10 * (options.initialStepSize ?? 1));
  let m = 0;

  return {
    update(acceptProb: number): number {
      m++;
      const w = 1.0 / (m + t0);
      hBar = (1 - w) * hBar + w * (targetRate - acceptProb);
      const logEps = mu - (Math.sqrt(m) / gamma) * hBar;
      const mKappa = Math.pow(m, -kappa);
      logEpsBar = mKappa * logEps + (1 - mKappa) * logEpsBar;
      return Math.exp(logEps);
    },
    finalStepSize(): number {
      return Math.exp(logEpsBar);
    },
  };
}

/**
 * Adapt the diagonal mass matrix from warmup samples.
 *
 * Estimates the variance of each parameter from collected samples
 * and returns the inverse diagonal mass matrix.
 *
 * @param samples - Warmup samples (each element is a parameter vector)
 * @param minVariance - Minimum variance to prevent numerical issues (default 1e-8)
 * @returns Inverse diagonal mass matrix
 */
function adaptMassMatrix(
  samples: number[][],
  minVariance: number = 1e-8,
): number[] {
  if (samples.length < 2) {
    return new Array<number>(samples[0].length).fill(1);
  }
  const d = samples[0].length;
  const massMatrixInv = new Array<number>(d);
  for (let j = 0; j < d; j++) {
    const col = samples.map((s) => s[j]);
    const v = variance(col);
    massMatrixInv[j] = 1 / Math.max(v, minVariance);
  }
  return massMatrixInv;
}

/** Default divergence threshold for Hamiltonian error. */
const DIVERGENCE_THRESHOLD = 1000;

/**
 * Hamiltonian Monte Carlo (HMC) sampler.
 *
 * Uses Hamiltonian dynamics simulated via the leapfrog integrator
 * to propose distant states with high acceptance probability.
 *
 * @param logDensity - Unnormalized log-density function
 * @param gradLogDensity - Gradient of the log-density function
 * @param dimensions - Number of parameters
 * @param options - Configuration options
 * @returns HMCResult containing chains and diagnostics
 *
 * @example
 * ```ts
 * // Sample from a 2D standard normal
 * const logDensity = (q: number[]) => -0.5 * (q[0] * q[0] + q[1] * q[1]);
 * const gradLogDensity = (q: number[]) => [-q[0], -q[1]];
 * const result = hamiltonianMC(logDensity, gradLogDensity, 2, {
 *   iterations: 5000,
 *   stepSize: 0.1,
 *   nLeapfrogSteps: 20,
 *   seed: 42,
 * });
 * ```
 */
export function hamiltonianMC(
  logDensity: (q: number[]) => number,
  gradLogDensity: (q: number[]) => number[],
  dimensions: number,
  options: {
    iterations?: number;
    burnIn?: number;
    stepSize?: number;
    nLeapfrogSteps?: number;
    seed?: number;
    massMatrix?: number[];
    initial?: number[];
    adaptStepSize?: boolean;
    adaptMass?: boolean;
    divergenceThreshold?: number;
  } = {},
): HMCResult {
  const iterations = options.iterations ?? 10000;
  const burnIn = options.burnIn ?? Math.floor(iterations * 0.2);
  const nLeapfrogSteps = options.nLeapfrogSteps ?? 20;
  const rng = new SeededRng(options.seed ?? Math.floor(Math.random() * 2147483647));
  const adaptStep = options.adaptStepSize ?? true;
  const adaptMass = options.adaptMass ?? true;
  const divThreshold = options.divergenceThreshold ?? DIVERGENCE_THRESHOLD;

  let stepSize = options.stepSize ?? 0.1;
  let massMatrixInv: number[] | undefined = options.massMatrix
    ? options.massMatrix.map((m) => 1 / m)
    : undefined;
  const massDiag = options.massMatrix ? [...options.massMatrix] : new Array<number>(dimensions).fill(1);

  let q = options.initial ? [...options.initial] : new Array<number>(dimensions).fill(0);
  let currentLogDensity = logDensity(q);
  if (Number.isNaN(currentLogDensity)) currentLogDensity = -Infinity;

  let accepted = 0;
  let divergences = 0;
  const chains: number[][] = [];
  const warmupSamples: number[][] = [];

  // Set up dual averaging for step size adaptation
  const da = adaptStep
    ? dualAveraging(0.65, { initialStepSize: stepSize })
    : null;

  for (let i = 0; i < iterations; i++) {
    // Sample momentum from N(0, M)
    const p = new Array<number>(dimensions);
    for (let j = 0; j < dimensions; j++) {
      const m = massDiag[j];
      p[j] = rng.nextNormal(0, Math.sqrt(m));
    }

    // Current Hamiltonian
    const currentKE = kineticEnergy(p, massMatrixInv);
    const currentH = -currentLogDensity + currentKE;

    // Leapfrog integration
    const [newQ, newP] = leapfrog(q, p, gradLogDensity, stepSize, nLeapfrogSteps, massMatrixInv);

    // Proposed Hamiltonian
    let proposedLogDensity = logDensity(newQ);
    if (Number.isNaN(proposedLogDensity)) proposedLogDensity = -Infinity;
    const proposedKE = kineticEnergy(newP, massMatrixInv);
    const proposedH = -proposedLogDensity + proposedKE;

    // Divergence check
    const deltaH = proposedH - currentH;
    if (Math.abs(deltaH) > divThreshold) {
      divergences++;
    }

    // Metropolis acceptance criterion
    const logAcceptProb = -deltaH;
    const acceptProb = Math.min(1, Math.exp(logAcceptProb));

    if (Math.log(rng.next()) < logAcceptProb && Number.isFinite(proposedH)) {
      q = newQ;
      currentLogDensity = proposedLogDensity;
      accepted++;
    }

    // Step size adaptation during warmup
    if (i < burnIn && da) {
      stepSize = da.update(Number.isNaN(acceptProb) ? 0 : acceptProb);
    } else if (i === burnIn && da) {
      stepSize = da.finalStepSize();
    }

    // Mass matrix adaptation during warmup
    if (i < burnIn && adaptMass) {
      warmupSamples.push([...q]);
      // Update mass matrix at midpoint and end of warmup
      if (i === Math.floor(burnIn / 2) || i === burnIn - 1) {
        massMatrixInv = adaptMassMatrix(warmupSamples);
        for (let j = 0; j < dimensions; j++) {
          massDiag[j] = 1 / massMatrixInv[j];
        }
      }
    }

    if (i >= burnIn) {
      chains.push([...q]);
    }
  }

  // Compute summaries
  const posteriorMeans = new Array<number>(dimensions);
  const posteriorStds = new Array<number>(dimensions);
  const effectiveSampleSizes = new Array<number>(dimensions);

  for (let j = 0; j < dimensions; j++) {
    const col = chains.map((row) => row[j]);
    posteriorMeans[j] = mean(col);
    posteriorStds[j] = Math.sqrt(variance(col));
    effectiveSampleSizes[j] = estimateESS(col);
  }

  return {
    chains,
    acceptanceRate: accepted / iterations,
    posteriorMeans,
    posteriorStds,
    totalIterations: iterations,
    burnIn,
    effectiveSampleSizes,
    divergences,
  };
}

/**
 * No-U-Turn Sampler (NUTS).
 *
 * Automatically tunes the path length of HMC by building a binary tree
 * of leapfrog steps and stopping when a U-turn is detected. This
 * eliminates the need to manually set the number of leapfrog steps.
 *
 * Implements the original NUTS algorithm (Algorithm 3 from Hoffman & Gelman, 2014).
 *
 * @param logDensity - Unnormalized log-density function
 * @param gradLogDensity - Gradient of the log-density function
 * @param dimensions - Number of parameters
 * @param options - Configuration options
 * @returns HMCResult containing chains and diagnostics
 *
 * @example
 * ```ts
 * // Sample from a 2D correlated normal
 * const logDensity = (q: number[]) => -0.5 * (q[0]*q[0] - q[0]*q[1] + q[1]*q[1]);
 * const gradLogDensity = (q: number[]) => [-(q[0] - 0.5*q[1]), -(q[1] - 0.5*q[0])];
 * const result = nutsSampler(logDensity, gradLogDensity, 2, {
 *   iterations: 2000,
 *   seed: 42,
 * });
 * ```
 */
export function nutsSampler(
  logDensity: (q: number[]) => number,
  gradLogDensity: (q: number[]) => number[],
  dimensions: number,
  options: {
    iterations?: number;
    burnIn?: number;
    seed?: number;
    maxTreeDepth?: number;
    targetAcceptRate?: number;
    initial?: number[];
    massMatrix?: number[];
    adaptMass?: boolean;
    divergenceThreshold?: number;
  } = {},
): HMCResult {
  const iterations = options.iterations ?? 10000;
  const burnIn = options.burnIn ?? Math.floor(iterations * 0.2);
  const maxTreeDepth = options.maxTreeDepth ?? 10;
  const targetAcceptRate = options.targetAcceptRate ?? 0.8;
  const rng = new SeededRng(options.seed ?? Math.floor(Math.random() * 2147483647));
  const adaptMassOpt = options.adaptMass ?? true;
  const divThreshold = options.divergenceThreshold ?? DIVERGENCE_THRESHOLD;

  let massMatrixInv: number[] | undefined = options.massMatrix
    ? options.massMatrix.map((m) => 1 / m)
    : undefined;
  const massDiag = options.massMatrix
    ? [...options.massMatrix]
    : new Array<number>(dimensions).fill(1);

  let q = options.initial ? [...options.initial] : new Array<number>(dimensions).fill(0);
  let currentLogDensity = logDensity(q);
  if (Number.isNaN(currentLogDensity)) currentLogDensity = -Infinity;

  // Find a reasonable initial step size
  let stepSize = findReasonableStepSize(q, logDensity, gradLogDensity, rng, massDiag, massMatrixInv);

  const da = dualAveraging(targetAcceptRate, { initialStepSize: stepSize });

  let accepted = 0;
  let divergences = 0;
  const chains: number[][] = [];
  const warmupSamples: number[][] = [];

  /** Check U-turn condition: returns true if U-turn detected. */
  function isUTurn(
    qMinus: number[],
    qPlus: number[],
    pMinus: number[],
    pPlus: number[],
  ): boolean {
    let dotMinus = 0;
    let dotPlus = 0;
    for (let i = 0; i < dimensions; i++) {
      const dq = qPlus[i] - qMinus[i];
      dotMinus += dq * pMinus[i];
      dotPlus += dq * pPlus[i];
    }
    return dotMinus < 0 || dotPlus < 0;
  }

  /**
   * Build tree recursively for NUTS.
   * Returns [qMinus, pMinus, qPlus, pPlus, qPrime, nPrime, sPrime, alphaPrime, nAlphaPrime]
   */
  function buildTree(
    qIn: number[],
    pIn: number[],
    u: number,
    v: number,
    j: number,
    eps: number,
    logDensity0: number,
  ): {
    qMinus: number[];
    pMinus: number[];
    qPlus: number[];
    pPlus: number[];
    qPrime: number[];
    nPrime: number;
    sPrime: boolean;
    alphaPrime: number;
    nAlphaPrime: number;
  } {
    if (j === 0) {
      // Base case: single leapfrog step
      const [qNew, pNew] = leapfrog(qIn, pIn, gradLogDensity, v * eps, 1, massMatrixInv);
      let logDensityNew = logDensity(qNew);
      if (Number.isNaN(logDensityNew)) logDensityNew = -Infinity;
      const keNew = kineticEnergy(pNew, massMatrixInv);
      const hamiltonianNew = -logDensityNew + keNew;
      const logJoint = -hamiltonianNew;

      // Check divergence
      const deltaH = hamiltonianNew - (-logDensity0 + kineticEnergy(pIn, massMatrixInv));
      if (Math.abs(deltaH) > divThreshold) {
        divergences++;
      }

      const nPrime = u <= Math.exp(logJoint) ? 1 : 0;
      const sPrime = Math.log(u) < logJoint + divThreshold;
      const acceptProb = Math.min(1, Math.exp(logDensityNew - logDensity0 - keNew + kineticEnergy(pIn, massMatrixInv)));

      return {
        qMinus: qNew,
        pMinus: pNew,
        qPlus: qNew,
        pPlus: pNew,
        qPrime: qNew,
        nPrime,
        sPrime,
        alphaPrime: Number.isNaN(acceptProb) ? 0 : acceptProb,
        nAlphaPrime: 1,
      };
    }

    // Recursion: build left and right subtrees
    const inner = buildTree(qIn, pIn, u, v, j - 1, eps, logDensity0);
    let {
      qMinus,
      pMinus,
      qPlus,
      pPlus,
      qPrime,
      nPrime,
      sPrime,
      alphaPrime,
      nAlphaPrime,
    } = inner;

    if (sPrime) {
      let inner2;
      if (v === -1) {
        inner2 = buildTree(qMinus, pMinus, u, v, j - 1, eps, logDensity0);
        qMinus = inner2.qMinus;
        pMinus = inner2.pMinus;
      } else {
        inner2 = buildTree(qPlus, pPlus, u, v, j - 1, eps, logDensity0);
        qPlus = inner2.qPlus;
        pPlus = inner2.pPlus;
      }

      // Multinomial sampling: accept new candidate with probability n'' / (n' + n'')
      const totalN = nPrime + inner2.nPrime;
      if (totalN > 0 && rng.next() < inner2.nPrime / totalN) {
        qPrime = inner2.qPrime;
      }

      sPrime = inner2.sPrime && !isUTurn(qMinus, qPlus, pMinus, pPlus);
      alphaPrime = alphaPrime + inner2.alphaPrime;
      nAlphaPrime = nAlphaPrime + inner2.nAlphaPrime;
      nPrime = totalN;
    }

    return {
      qMinus,
      pMinus,
      qPlus,
      pPlus,
      qPrime,
      nPrime,
      sPrime,
      alphaPrime,
      nAlphaPrime,
    };
  }

  for (let i = 0; i < iterations; i++) {
    // Sample momentum
    const p = new Array<number>(dimensions);
    for (let j = 0; j < dimensions; j++) {
      p[j] = rng.nextNormal(0, Math.sqrt(massDiag[j]));
    }

    const ke0 = kineticEnergy(p, massMatrixInv);
    const hamiltonian0 = -currentLogDensity + ke0;
    const jointLogProb = -hamiltonian0;

    // Slice variable
    const logu = jointLogProb - (-Math.log(rng.next()));
    const u = Math.exp(logu);

    // Initialize tree
    let qMinus = [...q];
    let pMinus = [...p];
    let qPlus = [...q];
    let pPlus = [...p];
    let j = 0;
    let n = 1;
    let s = true;
    let qCandidate = [...q];

    let treeAlpha = 0;
    let treeNAlpha = 0;

    while (s && j < maxTreeDepth) {
      // Choose direction uniformly
      const v = rng.next() < 0.5 ? -1 : 1;

      let result;
      if (v === -1) {
        result = buildTree(qMinus, pMinus, u, v, j, stepSize, currentLogDensity);
        qMinus = result.qMinus;
        pMinus = result.pMinus;
      } else {
        result = buildTree(qPlus, pPlus, u, v, j, stepSize, currentLogDensity);
        qPlus = result.qPlus;
        pPlus = result.pPlus;
      }

      if (result.sPrime && n > 0 && rng.next() < Math.min(1, result.nPrime / n)) {
        qCandidate = result.qPrime;
      }

      n += result.nPrime;
      s = result.sPrime && !isUTurn(qMinus, qPlus, pMinus, pPlus);
      j++;

      treeAlpha += result.alphaPrime;
      treeNAlpha += result.nAlphaPrime;
    }

    // Check if we moved
    let moved = false;
    for (let k = 0; k < dimensions; k++) {
      if (qCandidate[k] !== q[k]) {
        moved = true;
        break;
      }
    }
    if (moved) {
      q = qCandidate;
      currentLogDensity = logDensity(q);
      if (Number.isNaN(currentLogDensity)) currentLogDensity = -Infinity;
      accepted++;
    }

    // Adapt step size during warmup
    if (i < burnIn) {
      const avgAccept = treeNAlpha > 0 ? treeAlpha / treeNAlpha : 0;
      stepSize = da.update(avgAccept);
    } else if (i === burnIn) {
      stepSize = da.finalStepSize();
    }

    // Mass matrix adaptation during warmup
    if (i < burnIn && adaptMassOpt) {
      warmupSamples.push([...q]);
      if (i === Math.floor(burnIn / 2) || i === burnIn - 1) {
        massMatrixInv = adaptMassMatrix(warmupSamples);
        for (let k = 0; k < dimensions; k++) {
          massDiag[k] = 1 / massMatrixInv[k];
        }
      }
    }

    if (i >= burnIn) {
      chains.push([...q]);
    }
  }

  // Compute summaries
  const posteriorMeans = new Array<number>(dimensions);
  const posteriorStds = new Array<number>(dimensions);
  const effectiveSampleSizes = new Array<number>(dimensions);

  for (let j = 0; j < dimensions; j++) {
    const col = chains.map((row) => row[j]);
    posteriorMeans[j] = mean(col);
    posteriorStds[j] = Math.sqrt(variance(col));
    effectiveSampleSizes[j] = estimateESS(col);
  }

  return {
    chains,
    acceptanceRate: accepted / iterations,
    posteriorMeans,
    posteriorStds,
    totalIterations: iterations,
    burnIn,
    effectiveSampleSizes,
    divergences,
  };
}

/**
 * Find a reasonable initial step size for NUTS/HMC.
 *
 * Uses a heuristic that doubles or halves the step size until
 * the acceptance probability crosses 0.5.
 */
function findReasonableStepSize(
  q: number[],
  logDensity: (q: number[]) => number,
  gradLogDensity: (q: number[]) => number[],
  rng: SeededRng,
  massDiag: number[],
  massMatrixInv?: number[],
): number {
  const d = q.length;
  let eps = 1.0;

  // Sample a momentum
  const p = new Array<number>(d);
  for (let j = 0; j < d; j++) {
    p[j] = rng.nextNormal(0, Math.sqrt(massDiag[j]));
  }

  let ld = logDensity(q);
  if (Number.isNaN(ld)) ld = -Infinity;
  const ke0 = kineticEnergy(p, massMatrixInv);

  // One leapfrog step
  const [q1, p1] = leapfrog(q, p, gradLogDensity, eps, 1, massMatrixInv);
  let ld1 = logDensity(q1);
  if (Number.isNaN(ld1)) ld1 = -Infinity;
  const ke1 = kineticEnergy(p1, massMatrixInv);

  let logRatio = (ld1 - ke1) - (ld - ke0);
  if (Number.isNaN(logRatio)) logRatio = -Infinity;

  const a = logRatio > Math.log(0.5) ? 1 : -1;

  // Adjust step size
  let count = 0;
  while (count < 100) {
    eps = a === 1 ? eps * 2 : eps / 2;
    const [q2, p2] = leapfrog(q, p, gradLogDensity, eps, 1, massMatrixInv);
    let ld2 = logDensity(q2);
    if (Number.isNaN(ld2)) ld2 = -Infinity;
    const ke2 = kineticEnergy(p2, massMatrixInv);

    logRatio = (ld2 - ke2) - (ld - ke0);
    if (Number.isNaN(logRatio)) logRatio = -Infinity;

    if (a === 1 && logRatio <= Math.log(0.5)) break;
    if (a === -1 && logRatio >= Math.log(0.5)) break;
    count++;
  }

  return Math.max(eps, 1e-10);
}
