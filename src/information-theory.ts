import { Dataset } from "./types";

/**
 * Shannon entropy of a discrete probability distribution.
 *
 * H(X) = -Σ p(x) * log2(p(x))
 *
 * @param probabilities - Array of probabilities (must sum to ~1)
 * @param base - Logarithm base (default: 2 for bits)
 */
export function entropy(probabilities: Dataset, base = 2): number {
  validateDistribution(probabilities);
  let h = 0;
  for (const p of probabilities) {
    if (p > 0) h -= p * Math.log(p);
  }
  return h / Math.log(base);
}

/**
 * Shannon entropy estimated from observed data.
 *
 * Computes the empirical distribution from data values and returns its entropy.
 *
 * @param data - Array of categorical values (represented as numbers)
 * @param base - Logarithm base (default: 2 for bits)
 */
export function entropyFromData(data: Dataset, base = 2): number {
  if (data.length === 0) throw new Error("Dataset must not be empty");
  const probs = empiricalDistribution(data);
  return entropy(probs, base);
}

/**
 * Joint entropy of two discrete random variables.
 *
 * H(X, Y) = -Σ p(x, y) * log2(p(x, y))
 *
 * @param dataX - First variable observations
 * @param dataY - Second variable observations
 * @param base - Logarithm base (default: 2)
 */
export function jointEntropy(dataX: Dataset, dataY: Dataset, base = 2): number {
  if (dataX.length !== dataY.length) {
    throw new Error("Both datasets must have the same length");
  }
  if (dataX.length === 0) throw new Error("Datasets must not be empty");

  const jointCounts = new Map<string, number>();
  for (let i = 0; i < dataX.length; i++) {
    const key = `${dataX[i]},${dataY[i]}`;
    jointCounts.set(key, (jointCounts.get(key) ?? 0) + 1);
  }

  const n = dataX.length;
  const probs = Array.from(jointCounts.values()).map((c) => c / n);
  return entropy(probs, base);
}

/**
 * Conditional entropy H(Y|X).
 *
 * H(Y|X) = H(X, Y) - H(X)
 *
 * @param dataX - Conditioning variable
 * @param dataY - Target variable
 * @param base - Logarithm base (default: 2)
 */
export function conditionalEntropy(dataX: Dataset, dataY: Dataset, base = 2): number {
  return jointEntropy(dataX, dataY, base) - entropyFromData(dataX, base);
}

/**
 * Mutual information between two discrete random variables.
 *
 * I(X; Y) = H(X) + H(Y) - H(X, Y)
 *
 * Measures the amount of information shared between X and Y.
 *
 * @param dataX - First variable observations
 * @param dataY - Second variable observations
 * @param base - Logarithm base (default: 2)
 */
export function mutualInformation(dataX: Dataset, dataY: Dataset, base = 2): number {
  const hx = entropyFromData(dataX, base);
  const hy = entropyFromData(dataY, base);
  const hxy = jointEntropy(dataX, dataY, base);
  return Math.max(0, hx + hy - hxy);
}

/**
 * Normalized mutual information.
 *
 * NMI = 2 * I(X; Y) / (H(X) + H(Y))
 *
 * Returns a value between 0 and 1, making it easier to compare across
 * datasets of different sizes.
 *
 * @param dataX - First variable observations
 * @param dataY - Second variable observations
 * @param base - Logarithm base (default: 2)
 */
export function normalizedMutualInformation(dataX: Dataset, dataY: Dataset, base = 2): number {
  const hx = entropyFromData(dataX, base);
  const hy = entropyFromData(dataY, base);
  if (hx + hy === 0) return 0;
  const mi = mutualInformation(dataX, dataY, base);
  return (2 * mi) / (hx + hy);
}

/**
 * Kullback-Leibler divergence (relative entropy).
 *
 * D_KL(P || Q) = Σ p(x) * log(p(x) / q(x))
 *
 * Measures how distribution P diverges from reference distribution Q.
 * Not symmetric: D_KL(P||Q) ≠ D_KL(Q||P).
 *
 * @param p - The "true" distribution
 * @param q - The reference/model distribution
 * @param base - Logarithm base (default: 2)
 */
export function klDivergence(p: Dataset, q: Dataset, base = 2): number {
  if (p.length !== q.length) throw new Error("Distributions must have the same length");
  validateDistribution(p);
  validateDistribution(q);

  let kl = 0;
  for (let i = 0; i < p.length; i++) {
    if (p[i] > 0) {
      if (q[i] === 0) throw new Error("KL divergence undefined when q(x)=0 for any x where p(x)>0");
      kl += p[i] * Math.log(p[i] / q[i]);
    }
  }
  return kl / Math.log(base);
}

/**
 * Jensen-Shannon divergence.
 *
 * JSD(P || Q) = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M)
 * where M = 0.5 * (P + Q)
 *
 * A symmetric, bounded version of KL divergence. Always between 0 and 1
 * (when using base 2).
 *
 * @param p - First distribution
 * @param q - Second distribution
 * @param base - Logarithm base (default: 2)
 */
export function jsDivergence(p: Dataset, q: Dataset, base = 2): number {
  if (p.length !== q.length) throw new Error("Distributions must have the same length");
  validateDistribution(p);
  validateDistribution(q);

  const m = p.map((pi, i) => 0.5 * (pi + q[i]));
  return 0.5 * klDivergence(p, m, base) + 0.5 * klDivergence(q, m, base);
}

/**
 * Cross entropy.
 *
 * H(P, Q) = -Σ p(x) * log(q(x))
 *
 * Measures the average number of bits needed to encode data from
 * distribution P using a code optimized for distribution Q.
 *
 * @param p - The true distribution
 * @param q - The model distribution
 * @param base - Logarithm base (default: 2)
 */
export function crossEntropy(p: Dataset, q: Dataset, base = 2): number {
  if (p.length !== q.length) throw new Error("Distributions must have the same length");
  validateDistribution(p);
  validateDistribution(q);

  let h = 0;
  for (let i = 0; i < p.length; i++) {
    if (p[i] > 0) {
      if (q[i] === 0) throw new Error("Cross entropy undefined when q(x)=0 for any x where p(x)>0");
      h -= p[i] * Math.log(q[i]);
    }
  }
  return h / Math.log(base);
}

// ── Helpers ─────────────────────────────────────────────────────────────

function empiricalDistribution(data: Dataset): number[] {
  const counts = new Map<number, number>();
  for (const v of data) {
    counts.set(v, (counts.get(v) ?? 0) + 1);
  }
  const n = data.length;
  return Array.from(counts.values()).map((c) => c / n);
}

function validateDistribution(probs: number[]): void {
  if (probs.length === 0) throw new Error("Distribution must not be empty");
  for (const p of probs) {
    if (p < 0) throw new Error("Probabilities must be non-negative");
  }
  const sum = probs.reduce((a, b) => a + b, 0);
  if (Math.abs(sum - 1) > 1e-6) {
    throw new Error(`Probabilities must sum to 1, got ${sum}`);
  }
}
