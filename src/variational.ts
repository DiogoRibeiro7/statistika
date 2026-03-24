import { SeededRng } from "./random";

// ============================================================================
// Interfaces
// ============================================================================

export interface VariationalResult {
  /** Optimized variational parameters (means) */
  means: number[];
  /** Optimized variational parameters (standard deviations) */
  stds: number[];
  /** Evidence Lower Bound trajectory over iterations */
  elboHistory: number[];
  /** Final ELBO value */
  finalElbo: number;
  /** Number of iterations performed */
  iterations: number;
  /** Whether the optimization converged */
  converged: boolean;
}

export interface ADVIResult extends VariationalResult {
  /** Transformation applied to each parameter */
  transforms: TransformType[];
  /** Samples from the approximate posterior */
  samples: number[][];
  /** Adam optimizer state at termination */
  adamState: {
    m: number[];
    v: number[];
    t: number;
  };
}

export interface ModelComparisonResult {
  /** ELBO values for each model */
  elbos: number[];
  /** Index of the best model (highest ELBO) */
  bestModelIndex: number;
  /** ELBO differences relative to the best model */
  elboDifferences: number[];
  /** Approximate model weights (softmax of ELBOs) */
  modelWeights: number[];
}

// ============================================================================
// Types
// ============================================================================

export type TransformType = "identity" | "log" | "logit";

export type LogDensityFn = (params: number[]) => number;

// ============================================================================
// Internal Helpers
// ============================================================================

function standardNormal(rng: SeededRng): number {
  // Box-Muller transform
  const u1 = rng.next();
  const u2 = rng.next();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function applyTransform(unconstrained: number, transform: TransformType): number {
  switch (transform) {
    case "identity":
      return unconstrained;
    case "log":
      return Math.exp(unconstrained);
    case "logit":
      return 1 / (1 + Math.exp(-unconstrained));
  }
}

function logDetJacobian(unconstrained: number, transform: TransformType): number {
  switch (transform) {
    case "identity":
      return 0;
    case "log":
      return unconstrained; // log(exp(eta)) = eta
    case "logit": {
      const sig = 1 / (1 + Math.exp(-unconstrained));
      return Math.log(sig) + Math.log(1 - sig);
    }
  }
}

function softplus(x: number): number {
  // Numerically stable softplus: log(1 + exp(x))
  if (x > 20) return x;
  if (x < -20) return Math.exp(x);
  return Math.log(1 + Math.exp(x));
}

function softplusInverse(y: number): number {
  // Inverse of softplus: log(exp(y) - 1)
  if (y > 20) return y;
  return Math.log(Math.exp(y) - 1);
}

// ============================================================================
// Core Functions
// ============================================================================

/**
 * Compute the Evidence Lower Bound (ELBO) via Monte Carlo estimation.
 *
 * ELBO = E_q[log p(z)] - E_q[log q(z)]
 *
 * @param logDensity - Log density of the target distribution (unnormalized log posterior)
 * @param means - Variational means
 * @param stds - Variational standard deviations (must be positive)
 * @param rng - Seeded random number generator
 * @param numSamples - Number of Monte Carlo samples for estimation
 * @param transforms - Transformations for each parameter
 * @returns Estimated ELBO value
 */
export function computeELBO(
  logDensity: LogDensityFn,
  means: number[],
  stds: number[],
  rng: SeededRng,
  numSamples: number = 100,
  transforms?: TransformType[]
): number {
  const dim = means.length;
  const trans = transforms || new Array(dim).fill("identity");
  let elbo = 0;

  for (let s = 0; s < numSamples; s++) {
    // Sample from q using reparameterization: z = mu + sigma * epsilon
    const epsilon: number[] = [];
    const unconstrained: number[] = [];
    for (let i = 0; i < dim; i++) {
      const eps = standardNormal(rng);
      epsilon.push(eps);
      unconstrained.push(means[i] + stds[i] * eps);
    }

    // Transform to constrained space
    const constrained = unconstrained.map((z, i) => applyTransform(z, trans[i]));

    // log p(constrained params) + log |det J|
    let logP = logDensity(constrained);
    for (let i = 0; i < dim; i++) {
      logP += logDetJacobian(unconstrained[i], trans[i]);
    }

    // Entropy of q (diagonal normal): -log q(z) = 0.5 * dim * log(2*pi*e) + sum(log(sigma))
    // We compute log q(z) for each sample
    let logQ = 0;
    for (let i = 0; i < dim; i++) {
      logQ += -0.5 * Math.log(2 * Math.PI) - Math.log(stds[i]) - 0.5 * epsilon[i] ** 2;
    }

    elbo += logP - logQ;
  }

  return elbo / numSamples;
}

/**
 * Mean-field Variational Inference with coordinate ascent.
 *
 * Assumes a fully factorized variational family q(z) = prod_i q_i(z_i)
 * where each q_i is a univariate Gaussian.
 *
 * @param logDensity - Log density of the target distribution
 * @param dim - Number of parameters
 * @param options - Configuration options
 * @returns VariationalResult with optimized parameters
 */
export function meanFieldVI(
  logDensity: LogDensityFn,
  dim: number,
  options: {
    rng: SeededRng;
    maxIterations?: number;
    learningRate?: number;
    numSamples?: number;
    tolerance?: number;
    initialMeans?: number[];
    initialStds?: number[];
  }
): VariationalResult {
  const {
    rng,
    maxIterations = 1000,
    learningRate = 0.01,
    numSamples = 50,
    tolerance = 1e-6,
    initialMeans,
    initialStds,
  } = options;

  const means = initialMeans ? [...initialMeans] : new Array(dim).fill(0);
  const logStds = initialStds
    ? initialStds.map((s) => Math.log(s))
    : new Array(dim).fill(Math.log(1));
  const elboHistory: number[] = [];
  let converged = false;
  let iter = 0;

  for (iter = 0; iter < maxIterations; iter++) {
    const stds = logStds.map((ls) => Math.exp(ls));

    // Compute ELBO for tracking
    const currentElbo = computeELBO(logDensity, means, stds, rng, numSamples);
    elboHistory.push(currentElbo);

    // Check convergence
    if (iter > 0 && Math.abs(currentElbo - elboHistory[iter - 1]) < tolerance) {
      converged = true;
      break;
    }

    // Monte Carlo gradient estimates using reparameterization trick
    const gradMeans = new Array(dim).fill(0);
    const gradLogStds = new Array(dim).fill(0);

    for (let s = 0; s < numSamples; s++) {
      const epsilon: number[] = [];
      const z: number[] = [];
      for (let i = 0; i < dim; i++) {
        const eps = standardNormal(rng);
        epsilon.push(eps);
        z.push(means[i] + stds[i] * eps);
      }

      const baseLogP = logDensity(z);

      // Numerical gradient for each parameter
      for (let i = 0; i < dim; i++) {
        const delta = 1e-5;

        // Gradient w.r.t. mean_i
        const zPlus = [...z];
        zPlus[i] += delta;
        const logPPlus = logDensity(zPlus);
        const dLogP_dz = (logPPlus - baseLogP) / delta;

        // dz/dmu = 1, dz/dsigma = epsilon
        gradMeans[i] += dLogP_dz;
        gradLogStds[i] += dLogP_dz * epsilon[i] * stds[i] + 1; // +1 from entropy term d/d(log sigma) [log sigma] = 1
      }
    }

    // Update parameters using stochastic gradient ascent
    for (let i = 0; i < dim; i++) {
      means[i] += learningRate * (gradMeans[i] / numSamples);
      logStds[i] += learningRate * (gradLogStds[i] / numSamples);
    }
  }

  const finalStds = logStds.map((ls) => Math.exp(ls));
  const finalElbo = elboHistory.length > 0 ? elboHistory[elboHistory.length - 1] : NaN;

  return {
    means: [...means],
    stds: finalStds,
    elboHistory,
    finalElbo,
    iterations: iter,
    converged,
  };
}

/**
 * Automatic Differentiation Variational Inference (ADVI).
 *
 * Uses the Adam optimizer with the reparameterization trick for gradient
 * estimation. Supports identity, log, and logit transformations for
 * constrained parameters.
 *
 * @param logDensity - Log density of the target distribution in the constrained space
 * @param dim - Number of parameters
 * @param options - Configuration options
 * @returns ADVIResult with optimized parameters and samples
 */
export function advi(
  logDensity: LogDensityFn,
  dim: number,
  options: {
    rng: SeededRng;
    transforms?: TransformType[];
    maxIterations?: number;
    numSamples?: number;
    tolerance?: number;
    initialMeans?: number[];
    initialStds?: number[];
    numPosteriorSamples?: number;
    adam?: {
      alpha?: number;
      beta1?: number;
      beta2?: number;
      epsilon?: number;
    };
  }
): ADVIResult {
  const {
    rng,
    transforms: inputTransforms,
    maxIterations = 2000,
    numSamples = 10,
    tolerance = 1e-8,
    initialMeans,
    initialStds,
    numPosteriorSamples = 1000,
    adam: adamConfig = {},
  } = options;

  const transforms = inputTransforms || new Array(dim).fill("identity");

  // Adam optimizer hyperparameters
  const alpha = adamConfig.alpha ?? 0.01;
  const beta1 = adamConfig.beta1 ?? 0.9;
  const beta2 = adamConfig.beta2 ?? 0.999;
  const eps = adamConfig.epsilon ?? 1e-8;

  // Variational parameters: means and log-stds (unconstrained parameterization of stds)
  const mu = initialMeans ? [...initialMeans] : new Array(dim).fill(0);
  const omega = initialStds
    ? initialStds.map((s) => softplusInverse(s))
    : new Array(dim).fill(softplusInverse(1)); // omega such that softplus(omega) = std

  // Adam state
  const mMu = new Array(dim).fill(0);
  const vMu = new Array(dim).fill(0);
  const mOmega = new Array(dim).fill(0);
  const vOmega = new Array(dim).fill(0);
  let t = 0;

  const elboHistory: number[] = [];
  let converged = false;
  let iterations = 0;

  for (iterations = 0; iterations < maxIterations; iterations++) {
    t++;
    const sigma = omega.map((w) => softplus(w));

    // Monte Carlo gradient estimation with reparameterization trick
    const gradMu = new Array(dim).fill(0);
    const gradOmega = new Array(dim).fill(0);
    let elboEstimate = 0;

    for (let s = 0; s < numSamples; s++) {
      // Reparameterization trick: z = mu + sigma * epsilon
      const epsilon: number[] = [];
      const eta: number[] = []; // unconstrained
      for (let i = 0; i < dim; i++) {
        const e = standardNormal(rng);
        epsilon.push(e);
        eta.push(mu[i] + sigma[i] * e);
      }

      // Transform to constrained space
      const theta = eta.map((z, i) => applyTransform(z, transforms[i]));

      // Compute log p(theta) + log |det J|
      let logJoint = logDensity(theta);
      for (let i = 0; i < dim; i++) {
        logJoint += logDetJacobian(eta[i], transforms[i]);
      }

      // Entropy contribution: log q(eta) for this sample
      let logQ = 0;
      for (let i = 0; i < dim; i++) {
        logQ +=
          -0.5 * Math.log(2 * Math.PI) - Math.log(sigma[i]) - 0.5 * epsilon[i] ** 2;
      }

      elboEstimate += logJoint - logQ;

      // Numerical gradients of log p w.r.t. eta
      const delta = 1e-5;
      for (let i = 0; i < dim; i++) {
        const etaPlus = [...eta];
        etaPlus[i] += delta;
        const thetaPlus = etaPlus.map((z, j) => applyTransform(z, transforms[j]));

        let logJointPlus = logDensity(thetaPlus);
        for (let j = 0; j < dim; j++) {
          logJointPlus += logDetJacobian(etaPlus[j], transforms[j]);
        }

        const dLogJoint_dEta = (logJointPlus - logJoint) / delta;

        // Gradients via chain rule:
        // d ELBO / d mu_i = d log_joint / d eta_i  (since d eta / d mu = 1)
        // d ELBO / d omega_i = d log_joint / d eta_i * epsilon_i * softplus'(omega_i)
        //                    + softplus'(omega_i) / softplus(omega_i)  (entropy gradient)
        const softplusDeriv = 1 / (1 + Math.exp(-omega[i])); // sigmoid(omega)

        gradMu[i] += dLogJoint_dEta;
        gradOmega[i] +=
          dLogJoint_dEta * epsilon[i] * softplusDeriv + softplusDeriv / sigma[i];
      }
    }

    // Average gradients
    for (let i = 0; i < dim; i++) {
      gradMu[i] /= numSamples;
      gradOmega[i] /= numSamples;
    }
    elboEstimate /= numSamples;
    elboHistory.push(elboEstimate);

    // Check convergence
    if (
      iterations > 0 &&
      Math.abs(elboEstimate - elboHistory[iterations - 1]) < tolerance
    ) {
      converged = true;
      iterations++;
      break;
    }

    // Adam update for mu
    for (let i = 0; i < dim; i++) {
      mMu[i] = beta1 * mMu[i] + (1 - beta1) * gradMu[i];
      vMu[i] = beta2 * vMu[i] + (1 - beta2) * gradMu[i] ** 2;
      const mHat = mMu[i] / (1 - beta1 ** t);
      const vHat = vMu[i] / (1 - beta2 ** t);
      mu[i] += alpha * mHat / (Math.sqrt(vHat) + eps);
    }

    // Adam update for omega
    for (let i = 0; i < dim; i++) {
      mOmega[i] = beta1 * mOmega[i] + (1 - beta1) * gradOmega[i];
      vOmega[i] = beta2 * vOmega[i] + (1 - beta2) * gradOmega[i] ** 2;
      const mHat = mOmega[i] / (1 - beta1 ** t);
      const vHat = vOmega[i] / (1 - beta2 ** t);
      omega[i] += alpha * mHat / (Math.sqrt(vHat) + eps);
    }
  }

  const finalSigma = omega.map((w) => softplus(w));
  const finalElbo = elboHistory.length > 0 ? elboHistory[elboHistory.length - 1] : NaN;

  // Generate posterior samples
  const samples: number[][] = [];
  for (let s = 0; s < numPosteriorSamples; s++) {
    const sample: number[] = [];
    for (let i = 0; i < dim; i++) {
      const eps_i = standardNormal(rng);
      const eta_i = mu[i] + finalSigma[i] * eps_i;
      sample.push(applyTransform(eta_i, transforms[i]));
    }
    samples.push(sample);
  }

  return {
    means: [...mu],
    stds: finalSigma,
    elboHistory,
    finalElbo,
    iterations,
    converged,
    transforms: [...transforms],
    samples,
    adamState: {
      m: [...mMu, ...mOmega],
      v: [...vMu, ...vOmega],
      t,
    },
  };
}

/**
 * Normal variational approximation for a single parameter.
 *
 * Fits a Gaussian q(z) = N(mu, sigma^2) to approximate the target density.
 *
 * @param logDensity - Log density of the target (univariate)
 * @param rng - Seeded random number generator
 * @param options - Configuration options
 * @returns VariationalResult with optimized mean and std
 */
export function normalVariational(
  logDensity: (x: number) => number,
  rng: SeededRng,
  options: {
    maxIterations?: number;
    learningRate?: number;
    numSamples?: number;
    tolerance?: number;
    initialMean?: number;
    initialStd?: number;
  } = {}
): VariationalResult {
  const wrappedLogDensity: LogDensityFn = (params) => logDensity(params[0]);

  const result = meanFieldVI(wrappedLogDensity, 1, {
    rng,
    maxIterations: options.maxIterations,
    learningRate: options.learningRate,
    numSamples: options.numSamples,
    tolerance: options.tolerance,
    initialMeans: options.initialMean !== undefined ? [options.initialMean] : undefined,
    initialStds: options.initialStd !== undefined ? [options.initialStd] : undefined,
  });

  return result;
}

/**
 * Log-normal variational approximation for a positive parameter.
 *
 * Fits q(z) such that log(z) ~ N(mu, sigma^2), ensuring z > 0.
 * Uses the log transform internally via ADVI.
 *
 * @param logDensity - Log density of the target in the constrained (positive) space
 * @param rng - Seeded random number generator
 * @param options - Configuration options
 * @returns ADVIResult with optimized parameters
 */
export function logNormalVariational(
  logDensity: (x: number) => number,
  rng: SeededRng,
  options: {
    maxIterations?: number;
    numSamples?: number;
    tolerance?: number;
    initialMean?: number;
    initialStd?: number;
    adam?: {
      alpha?: number;
      beta1?: number;
      beta2?: number;
      epsilon?: number;
    };
  } = {}
): ADVIResult {
  const wrappedLogDensity: LogDensityFn = (params) => logDensity(params[0]);

  return advi(wrappedLogDensity, 1, {
    rng,
    transforms: ["log"],
    maxIterations: options.maxIterations,
    numSamples: options.numSamples,
    tolerance: options.tolerance,
    initialMeans: options.initialMean !== undefined ? [options.initialMean] : undefined,
    initialStds: options.initialStd !== undefined ? [options.initialStd] : undefined,
    adam: options.adam,
  });
}

/**
 * Multivariate normal variational approximation (mean-field).
 *
 * Fits a factorized multivariate Gaussian q(z) = prod_i N(mu_i, sigma_i^2).
 * Each dimension is independent (diagonal covariance).
 *
 * @param logDensity - Log density of the target distribution
 * @param dim - Dimensionality of the parameter space
 * @param rng - Seeded random number generator
 * @param options - Configuration options
 * @returns VariationalResult with optimized parameters
 */
export function multivariateNormalVariational(
  logDensity: LogDensityFn,
  dim: number,
  rng: SeededRng,
  options: {
    maxIterations?: number;
    learningRate?: number;
    numSamples?: number;
    tolerance?: number;
    initialMeans?: number[];
    initialStds?: number[];
  } = {}
): VariationalResult {
  return meanFieldVI(logDensity, dim, {
    rng,
    maxIterations: options.maxIterations,
    learningRate: options.learningRate,
    numSamples: options.numSamples,
    tolerance: options.tolerance,
    initialMeans: options.initialMeans,
    initialStds: options.initialStds,
  });
}

/**
 * Compare multiple models using their ELBO values from variational inference.
 *
 * Computes approximate model weights using a softmax over the ELBO values,
 * providing a Bayesian model comparison framework.
 *
 * @param models - Array of objects, each containing a logDensity and VI configuration
 * @param rng - Seeded random number generator
 * @returns ModelComparisonResult with ELBO values, best model index, and model weights
 */
export function compareModelsVI(
  models: Array<{
    logDensity: LogDensityFn;
    dim: number;
    transforms?: TransformType[];
    options?: {
      maxIterations?: number;
      numSamples?: number;
      tolerance?: number;
      initialMeans?: number[];
      initialStds?: number[];
      adam?: {
        alpha?: number;
        beta1?: number;
        beta2?: number;
        epsilon?: number;
      };
    };
  }>,
  rng: SeededRng
): ModelComparisonResult {
  const elbos: number[] = [];

  for (const model of models) {
    const result = advi(model.logDensity, model.dim, {
      rng,
      transforms: model.transforms,
      maxIterations: model.options?.maxIterations ?? 1000,
      numSamples: model.options?.numSamples ?? 10,
      tolerance: model.options?.tolerance,
      initialMeans: model.options?.initialMeans,
      initialStds: model.options?.initialStds,
      adam: model.options?.adam,
    });
    elbos.push(result.finalElbo);
  }

  // Find best model
  let bestModelIndex = 0;
  let bestElbo = elbos[0];
  for (let i = 1; i < elbos.length; i++) {
    if (elbos[i] > bestElbo) {
      bestElbo = elbos[i];
      bestModelIndex = i;
    }
  }

  // Compute ELBO differences relative to best
  const elboDifferences = elbos.map((e) => e - bestElbo);

  // Compute model weights via softmax (shifted for numerical stability)
  const expDiffs = elboDifferences.map((d) => Math.exp(d));
  const sumExp = expDiffs.reduce((a, b) => a + b, 0);
  const modelWeights = expDiffs.map((e) => e / sumExp);

  return {
    elbos,
    bestModelIndex,
    elboDifferences,
    modelWeights,
  };
}
