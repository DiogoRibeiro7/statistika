/**
 * Item Response Theory (IRT) Module
 *
 * Provides 1PL (Rasch), 2PL, 3PL, and Graded Response Model (GRM)
 * implementations with ability estimation via EM algorithm using
 * Gauss-Hermite quadrature.
 */

// ---------------------------------------------------------------------------
// Interfaces
// ---------------------------------------------------------------------------

/** Parameters for a dichotomous IRT item (1PL / 2PL / 3PL). */
export interface IRTItemParams {
  /** Discrimination parameter (a). Default 1 for Rasch. */
  a?: number;
  /** Difficulty parameter (b). */
  b: number;
  /** Pseudo-guessing parameter (c). Default 0 for 1PL/2PL. */
  c?: number;
}

/** Result returned by dichotomous IRT model probability functions. */
export interface IRTResult {
  /** The ability value (theta) at which the probability was computed. */
  theta: number;
  /** Probability of a correct response. */
  probability: number;
}

/** Parameters for a single item in the Graded Response Model. */
export interface GRMItemParams {
  /** Discrimination parameter (a). */
  a: number;
  /** Category boundary (threshold) parameters, length = K-1 for K categories.
   *  Must be in ascending order. */
  b: number[];
}

/** Result returned by the Graded Response Model. */
export interface GRMResult {
  /** The ability value (theta). */
  theta: number;
  /** Probability for each ordinal category (sums to 1). */
  categoryProbabilities: number[];
}

/** Data point for an item information function. */
export interface ItemInformationData {
  /** The ability value (theta). */
  theta: number;
  /** Fisher information at that theta. */
  information: number;
}

/** Data point for an item characteristic curve. */
export interface ICCData {
  /** The ability value (theta). */
  theta: number;
  /** Probability of correct response at that theta. */
  probability: number;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Logistic function: 1 / (1 + exp(-x)).
 */
function logistic(x: number): number {
  if (x > 500) return 1;
  if (x < -500) return 0;
  return 1 / (1 + Math.exp(-x));
}

/**
 * Gauss-Hermite quadrature nodes and weights (21 points).
 * Precomputed for the standard normal weight function
 * w(x) = exp(-x^2). To use with a N(0,1) prior we transform
 * nodes by sqrt(2) and weights by 1/sqrt(pi).
 */
function gaussHermiteQuadrature(nPoints: number = 21): { nodes: number[]; weights: number[] } {
  // Use well-known 21-point Gauss-Hermite nodes/weights for exp(-x^2).
  // We provide a smaller hard-coded set and interpolate if needed,
  // but 21 points is standard for IRT.
  const rawNodes: number[] = [
    -5.38748089001, -4.60368244955, -3.94476404012, -3.34785456738,
    -2.78880605843, -2.25497400209, -1.73853771212, -1.23407621539,
    -0.73747372854, -0.24534070830, 0.24534070830, 0.73747372854,
    1.23407621539, 1.73853771212, 2.25497400209, 2.78880605843,
    3.34785456738, 3.94476404012, 4.60368244955, 5.38748089001,
    0.0
  ];

  const rawWeights: number[] = [
    2.2293936455e-13, 4.3993409922e-10, 1.0860693707e-7, 7.8025564785e-6,
    2.2833863601e-4, 3.2437733422e-3, 2.4810520887e-2, 1.0901720602e-1,
    2.8667550536e-1, 4.6224366960e-1, 4.6224366960e-1, 2.8667550536e-1,
    1.0901720602e-1, 2.4810520887e-2, 3.2437733422e-3, 2.2833863601e-4,
    7.8025564785e-6, 1.0860693707e-7, 4.3993409922e-10, 2.2293936455e-13,
    7.2023521560e-1
  ];

  // Sort by node value
  const paired = rawNodes.map((n, i) => ({ node: n, weight: rawWeights[i] }));
  paired.sort((a, b) => a.node - b.node);

  // Transform for N(0,1) prior: x_i = sqrt(2)*t_i, w_i = w_i / sqrt(pi)
  const sqrtPi = Math.sqrt(Math.PI);
  const sqrt2 = Math.sqrt(2);
  const nodes = paired.map(p => p.node * sqrt2);
  const weights = paired.map(p => p.weight / sqrtPi);

  if (nPoints >= nodes.length) {
    return { nodes, weights };
  }

  // If fewer points requested, take the central ones
  const start = Math.floor((nodes.length - nPoints) / 2);
  return {
    nodes: nodes.slice(start, start + nPoints),
    weights: weights.slice(start, start + nPoints),
  };
}

// ---------------------------------------------------------------------------
// IRT Model Probability Functions
// ---------------------------------------------------------------------------

/**
 * Rasch (1PL) model.
 * P(theta) = 1 / (1 + exp(-(theta - b)))
 */
export function raschModel(theta: number, params: IRTItemParams): IRTResult {
  const b = params.b;
  const probability = logistic(theta - b);
  return { theta, probability };
}

/**
 * Two-parameter logistic (2PL) model.
 * P(theta) = 1 / (1 + exp(-a*(theta - b)))
 */
export function twoPlModel(theta: number, params: IRTItemParams): IRTResult {
  const a = params.a ?? 1;
  const b = params.b;
  const probability = logistic(a * (theta - b));
  return { theta, probability };
}

/**
 * Three-parameter logistic (3PL) model.
 * P(theta) = c + (1-c) / (1 + exp(-a*(theta - b)))
 */
export function threePlModel(theta: number, params: IRTItemParams): IRTResult {
  const a = params.a ?? 1;
  const b = params.b;
  const c = params.c ?? 0;
  const probability = c + (1 - c) * logistic(a * (theta - b));
  return { theta, probability };
}

/**
 * Graded Response Model (GRM) — Samejima's cumulative logit model.
 *
 * For an item with K ordered categories (0..K-1) and K-1 threshold
 * parameters b_1 < b_2 < ... < b_{K-1}:
 *
 *   P*(k | theta) = 1 / (1 + exp(-a*(theta - b_k)))   cumulative prob
 *   P(category = k | theta) = P*(k) - P*(k+1)
 *
 * with P*(0) = 1 and P*(K) = 0.
 */
export function gradedResponseModel(theta: number, params: GRMItemParams): GRMResult {
  const { a, b } = params;
  const K = b.length + 1; // number of categories

  // Cumulative probabilities: P*(0)=1, P*(K)=0
  const cumProbs: number[] = new Array(K + 1);
  cumProbs[0] = 1;
  for (let k = 1; k <= b.length; k++) {
    cumProbs[k] = logistic(a * (theta - b[k - 1]));
  }
  cumProbs[K] = 0;

  // Category probabilities
  const categoryProbabilities: number[] = new Array(K);
  for (let k = 0; k < K; k++) {
    categoryProbabilities[k] = Math.max(0, cumProbs[k] - cumProbs[k + 1]);
  }

  // Normalise to handle floating-point drift
  const sum = categoryProbabilities.reduce((s, v) => s + v, 0);
  if (sum > 0) {
    for (let k = 0; k < K; k++) {
      categoryProbabilities[k] /= sum;
    }
  }

  return { theta, categoryProbabilities };
}

// ---------------------------------------------------------------------------
// Item & Test Information
// ---------------------------------------------------------------------------

/**
 * Compute item information for a dichotomous IRT item at a given theta.
 *
 * For the 3PL model the Fisher information is:
 *   I(theta) = a^2 * ((P - c)^2 / ((1-c)^2 * P)) * Q
 * where P = P(theta), Q = 1-P.
 *
 * For 1PL/2PL (c=0) this simplifies to a^2 * P * Q.
 */
export function itemInformation(theta: number, params: IRTItemParams): ItemInformationData {
  const a = params.a ?? 1;
  const c = params.c ?? 0;
  const P = threePlModel(theta, params).probability;
  const Q = 1 - P;

  let information: number;
  if (c === 0) {
    information = a * a * P * Q;
  } else {
    // 3PL information formula
    const numerator = a * a * Q * Math.pow(P - c, 2);
    const denominator = Math.pow(1 - c, 2) * P;
    information = denominator > 0 ? numerator / denominator : 0;
  }

  return { theta, information };
}

/**
 * Compute test information at a given theta as the sum of item
 * information values across all items.
 */
export function testInformation(theta: number, items: IRTItemParams[]): ItemInformationData {
  let totalInfo = 0;
  for (const item of items) {
    totalInfo += itemInformation(theta, item).information;
  }
  return { theta, information: totalInfo };
}

// ---------------------------------------------------------------------------
// Item Characteristic Curve
// ---------------------------------------------------------------------------

/**
 * Compute the ICC (Item Characteristic Curve) over a range of theta values.
 *
 * @param params   Item parameters.
 * @param thetaMin Lower bound of theta range (default -4).
 * @param thetaMax Upper bound of theta range (default  4).
 * @param nPoints  Number of equally-spaced evaluation points (default 81).
 * @returns Array of {theta, probability} data points.
 */
export function itemCharacteristicCurve(
  params: IRTItemParams,
  thetaMin: number = -4,
  thetaMax: number = 4,
  nPoints: number = 81,
): ICCData[] {
  const step = (thetaMax - thetaMin) / (nPoints - 1);
  const curve: ICCData[] = [];
  for (let i = 0; i < nPoints; i++) {
    const theta = thetaMin + i * step;
    const probability = threePlModel(theta, params).probability;
    curve.push({ theta, probability });
  }
  return curve;
}

// ---------------------------------------------------------------------------
// Ability Estimation — EM with Gauss-Hermite Quadrature
// ---------------------------------------------------------------------------

/**
 * Estimate a single examinee's latent ability (theta) given a vector of
 * dichotomous item responses and item parameters.
 *
 * Uses the EM (Expectation-Maximisation) algorithm with Gauss-Hermite
 * quadrature over a standard normal prior.
 *
 * @param responses  Array of 0/1 responses (one per item).
 * @param items      Corresponding item parameters.
 * @param options    Optional configuration.
 * @returns Estimated theta (EAP — Expected A Posteriori).
 */
export function estimateAbility(
  responses: number[],
  items: IRTItemParams[],
  options: {
    /** Maximum EM iterations (default 100). */
    maxIterations?: number;
    /** Convergence tolerance on theta change (default 1e-6). */
    tolerance?: number;
    /** Number of quadrature points (default 21). */
    quadPoints?: number;
    /** Prior mean (default 0). */
    priorMean?: number;
    /** Prior standard deviation (default 1). */
    priorSd?: number;
  } = {},
): number {
  const {
    maxIterations = 100,
    tolerance = 1e-6,
    quadPoints = 21,
    priorMean = 0,
    priorSd = 1,
  } = options;

  if (responses.length !== items.length) {
    throw new Error('responses and items must have the same length');
  }

  const { nodes, weights } = gaussHermiteQuadrature(quadPoints);

  // Shift/scale quadrature nodes to match prior N(priorMean, priorSd^2)
  const quadNodes = nodes.map(n => priorMean + priorSd * n);
  // Weights already incorporate the standard normal density via Gauss-Hermite
  // transform, so we can use them directly.

  let prevTheta = Infinity;

  // EAP estimation: iterate to allow for optional extensions (e.g.,
  // item parameter updates in a full EM). For pure EAP with fixed item
  // params this converges in one step, but we keep the loop for generality.
  let eapTheta = priorMean;

  for (let iter = 0; iter < maxIterations; iter++) {
    // E-step: compute posterior weights at each quadrature node
    const logLikelihoods: number[] = new Array(quadNodes.length);

    for (let q = 0; q < quadNodes.length; q++) {
      const thetaQ = quadNodes[q];
      let logL = 0;
      for (let j = 0; j < items.length; j++) {
        const p = threePlModel(thetaQ, items[j]).probability;
        const pClamped = Math.max(1e-15, Math.min(1 - 1e-15, p));
        if (responses[j] === 1) {
          logL += Math.log(pClamped);
        } else {
          logL += Math.log(1 - pClamped);
        }
      }
      logLikelihoods[q] = logL;
    }

    // Find max log-likelihood for numerical stability
    const maxLogL = Math.max(...logLikelihoods);

    // Posterior weights = prior_weight * likelihood
    const posteriorWeights: number[] = new Array(quadNodes.length);
    let sumWeights = 0;
    for (let q = 0; q < quadNodes.length; q++) {
      posteriorWeights[q] = weights[q] * Math.exp(logLikelihoods[q] - maxLogL);
      sumWeights += posteriorWeights[q];
    }

    // Normalise
    if (sumWeights > 0) {
      for (let q = 0; q < quadNodes.length; q++) {
        posteriorWeights[q] /= sumWeights;
      }
    }

    // EAP estimate: E[theta | data]
    eapTheta = 0;
    for (let q = 0; q < quadNodes.length; q++) {
      eapTheta += quadNodes[q] * posteriorWeights[q];
    }

    // Check convergence
    if (Math.abs(eapTheta - prevTheta) < tolerance) {
      break;
    }
    prevTheta = eapTheta;
  }

  return eapTheta;
}
