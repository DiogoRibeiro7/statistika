/**
 * Multivariate t-distribution.
 *
 * A heavy-tailed generalization of the multivariate normal distribution.
 * Parameterized by a location vector `mu`, a scale matrix `sigma`, and
 * degrees of freedom `df`. As df -> Infinity, this converges to the
 * multivariate normal distribution.
 *
 * The multivariate t can be represented as a normal-variance mixture:
 *   X = mu + Z * sqrt(df / W)
 * where Z ~ MultivariateNormal(0, sigma) and W ~ Chi-squared(df).
 *
 * PDF: f(x) = Gamma((df+p)/2) / (Gamma(df/2) * df^{p/2} * pi^{p/2} * |sigma|^{1/2})
 *             * (1 + (x-mu)^T sigma^{-1} (x-mu) / df)^{-(df+p)/2}
 *
 * @example
 * ```ts
 * const dist = new MultivariateT([0, 0], [[1, 0], [0, 1]], 5);
 * dist.mean();       // [0, 0]
 * dist.pdf([0, 0]);  // density at the mode
 * dist.sample();     // random 2D vector
 * ```
 */

import { gammaLn } from "../../utils/math";
import { RandomFn } from "../../types";

/**
 * Computes the Cholesky decomposition of a symmetric positive-definite matrix.
 *
 * Returns the lower triangular matrix L such that A = L * L^T.
 *
 * @param A - A symmetric positive-definite matrix.
 * @returns The lower triangular Cholesky factor L.
 * @throws {Error} If the matrix is not positive definite.
 */
function cholesky(A: number[][]): number[][] {
  const n = A.length;
  const L: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));

  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = 0;
      for (let k = 0; k < j; k++) sum += L[i][k] * L[j][k];
      if (i === j) {
        const diag = A[i][i] - sum;
        if (diag <= 0) throw new Error("Matrix is not positive definite");
        L[i][j] = Math.sqrt(diag);
      } else {
        L[i][j] = (A[i][j] - sum) / L[j][j];
      }
    }
  }
  return L;
}

/**
 * Represents a Multivariate t-distribution.
 *
 * The multivariate t-distribution with `df` degrees of freedom, location `mu`,
 * and scale matrix `sigma` is commonly used in Bayesian inference and robust
 * statistics to model data with heavier tails than the multivariate normal.
 */
export class MultivariateT {
  readonly name: string;
  readonly dim: number;
  private readonly L: number[][]; // Cholesky of sigma
  private readonly logNormConst: number;
  private readonly logDetSigma: number;
  private rng: RandomFn;

  /**
   * Creates a new Multivariate t-distribution.
   *
   * @param mu - Location vector (p-dimensional).
   * @param sigma - Scale matrix (p x p, symmetric positive-definite).
   * @param df - Degrees of freedom (must be > 0).
   * @param rng - Optional random number generator; defaults to Math.random.
   * @throws {Error} If dimension is less than 1.
   * @throws {Error} If sigma is not square or dimensions mismatch mu.
   * @throws {Error} If df is not positive.
   * @throws {Error} If sigma is not positive definite.
   *
   * @example
   * ```ts
   * const dist = new MultivariateT([0, 0], [[1, 0.5], [0.5, 1]], 5);
   * ```
   */
  constructor(
    public readonly mu: number[],
    public readonly sigma: number[][],
    public readonly df: number,
    rng?: RandomFn,
  ) {
    this.rng = rng ?? Math.random;
    const p = mu.length;
    if (p < 1) throw new Error(`Invalid parameter 'mu': expected at least 1 dimension, received ${p}`);
    if (sigma.length !== p || sigma.some((r) => r.length !== p)) {
      throw new Error(`Invalid parameter 'sigma': expected a ${p}x${p} square matrix matching the dimension of mu, received ${sigma.length}x${sigma[0]?.length}`);
    }
    if (df <= 0) throw new Error(`Invalid parameter 'df': expected a positive number, received ${df}`);
    this.dim = p;
    this.name = `MultivariateT(dim=${p}, df=${df})`;
    this.L = cholesky(sigma);

    // Log determinant of sigma from Cholesky
    let logDet = 0;
    for (let i = 0; i < p; i++) logDet += 2 * Math.log(this.L[i][i]);
    this.logDetSigma = logDet;

    // Log normalization constant
    this.logNormConst =
      gammaLn((df + p) / 2) -
      gammaLn(df / 2) -
      (p / 2) * Math.log(df * Math.PI) -
      0.5 * logDet;
  }

  /**
   * Computes the mean vector of the distribution.
   *
   * Formula: E[X] = mu for df > 1.
   *
   * @returns A copy of the location vector.
   * @throws {Error} If df <= 1 (mean is undefined).
   */
  mean(): number[] {
    if (this.df <= 1) {
      throw new Error("Mean is undefined for df <= 1");
    }
    return [...this.mu];
  }

  /**
   * Computes the covariance matrix of the distribution.
   *
   * Formula: Cov(X) = (df / (df - 2)) * sigma for df > 2.
   *
   * @returns A p x p covariance matrix.
   * @throws {Error} If df <= 2 (covariance is undefined).
   */
  covariance(): number[][] {
    if (this.df <= 2) {
      throw new Error("Covariance is undefined for df <= 2");
    }
    const p = this.dim;
    const factor = this.df / (this.df - 2);
    const result: number[][] = Array.from({ length: p }, () => new Array(p));
    for (let i = 0; i < p; i++) {
      for (let j = 0; j < p; j++) {
        result[i][j] = factor * this.sigma[i][j];
      }
    }
    return result;
  }

  /**
   * Computes the log probability density function at a point x.
   *
   * @param x - A p-dimensional vector.
   * @returns The log-density at x.
   * @throws {Error} If x has incorrect dimension.
   */
  logPdf(x: number[]): number {
    const p = this.dim;
    if (x.length !== p) {
      throw new Error(`x must be a ${p}-dimensional vector`);
    }

    // Compute (x - mu)
    const diff = new Array(p);
    for (let i = 0; i < p; i++) diff[i] = x[i] - this.mu[i];

    // Solve L * z = diff (forward substitution)
    const z = new Array(p);
    for (let i = 0; i < p; i++) {
      let s = diff[i];
      for (let k = 0; k < i; k++) s -= this.L[i][k] * z[k];
      z[i] = s / this.L[i][i];
    }

    // Mahalanobis distance squared: z^T z
    let mahal = 0;
    for (let i = 0; i < p; i++) mahal += z[i] * z[i];

    return this.logNormConst - ((this.df + p) / 2) * Math.log(1 + mahal / this.df);
  }

  /**
   * Computes the probability density function at a point x.
   *
   * @param x - A p-dimensional vector.
   * @returns The density at x.
   */
  pdf(x: number[]): number {
    return Math.exp(this.logPdf(x));
  }

  /**
   * Draws a single random sample using the Normal/Chi-squared mixture representation.
   *
   * X = mu + Z * sqrt(df / W), where Z ~ N(0, sigma), W ~ Chi-squared(df).
   *
   * @returns A random p-dimensional vector.
   */
  sample(): number[] {
    const p = this.dim;

    // Sample Z ~ N(0, I_p)
    const z = new Array(p);
    for (let i = 0; i < p; i++) {
      const u1 = this.rng();
      const u2 = this.rng();
      z[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    // Transform: y = L * z (so y ~ N(0, sigma))
    const y = new Array(p).fill(0);
    for (let i = 0; i < p; i++) {
      for (let j = 0; j <= i; j++) {
        y[i] += this.L[i][j] * z[j];
      }
    }

    // Sample W ~ Chi-squared(df) = 2 * Gamma(df/2, 1)
    const w = sampleChiSquared(this.df, this.rng);
    const scale = Math.sqrt(this.df / w);

    // X = mu + y * sqrt(df / W)
    const result = new Array(p);
    for (let i = 0; i < p; i++) {
      result[i] = this.mu[i] + y[i] * scale;
    }
    return result;
  }

  /**
   * Draws n independent samples from the distribution.
   *
   * @param n - Number of samples to draw.
   * @returns An array of n random p-dimensional vectors.
   */
  sampleN(n: number): number[][] {
    const samples: number[][] = new Array(n);
    for (let i = 0; i < n; i++) samples[i] = this.sample();
    return samples;
  }
}

/**
 * Samples from the Chi-squared(df) distribution.
 */
function sampleChiSquared(df: number, rng: RandomFn): number {
  return 2 * sampleGamma(df / 2, rng);
}

/**
 * Samples from the Gamma(shape, 1) distribution using the Marsaglia-Tsang method.
 */
function sampleGamma(shape: number, rng: RandomFn): number {
  if (shape < 1) {
    return sampleGamma(shape + 1, rng) * Math.pow(rng(), 1 / shape);
  }
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  while (true) {
    let x: number, v: number;
    do {
      const u1 = rng();
      const u2 = rng();
      x = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = rng();
    if (
      u < 1 - 0.0331 * (x * x) * (x * x) ||
      Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))
    ) {
      return d * v;
    }
  }
}
