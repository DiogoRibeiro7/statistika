/**
 * Multivariate Normal (Gaussian) distribution.
 *
 * Parameterized by a mean vector mu and covariance matrix Sigma.
 * Supports PDF evaluation, sampling via Cholesky decomposition,
 * and log-likelihood computation.
 *
 * PDF: f(x; mu, Sigma) = (2pi)^{-k/2} |Sigma|^{-1/2} exp(-0.5 (x - mu)^T Sigma^{-1} (x - mu))
 *
 * @example
 * ```ts
 * const dist = new MultivariateNormal([0, 0], [[1, 0.5], [0.5, 1]]);
 * dist.pdf([0, 0]);    // density at the mean
 * dist.sample();       // random 2D vector
 * ```
 */

import { gammaLn } from "../../utils/math";
import { RandomFn } from "../../types";
import { resolveRng } from "../../random";

/**
 * Computes the Cholesky decomposition of a symmetric positive-definite matrix.
 *
 * Returns the lower triangular matrix L such that A = L * L^T.
 *
 * @param A - A symmetric positive-definite matrix (n x n).
 * @returns The lower triangular Cholesky factor L.
 * @throws {Error} If the matrix is not positive definite.
 */
function cholesky(A: number[][]): number[][] {
  const n = A.length;
  const L: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));

  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = 0;
      for (let k = 0; k < j; k++) {
        sum += L[i][k] * L[j][k];
      }
      if (i === j) {
        const diag = A[i][i] - sum;
        if (diag <= 0) {
          throw new Error(`Invalid parameter 'sigma': expected positive definite matrix, received non-positive diagonal at index ${i}`);
        }
        L[i][j] = Math.sqrt(diag);
      } else {
        L[i][j] = (A[i][j] - sum) / L[j][j];
      }
    }
  }
  return L;
}

/**
 * Computes the log-determinant of a positive-definite matrix from its Cholesky factor.
 *
 * Formula: log det(Sigma) = 2 * sum log(L_ii), since det(Sigma) = det(L)^2.
 *
 * @param L - Lower triangular Cholesky factor.
 * @returns The log-determinant of the original matrix.
 */
function logDetFromCholesky(L: number[][]): number {
  let sum = 0;
  for (let i = 0; i < L.length; i++) {
    sum += Math.log(L[i][i]);
  }
  return 2 * sum;
}

/**
 * Solves the linear system L * x = b where L is lower triangular (forward substitution).
 *
 * @param L - Lower triangular matrix.
 * @param b - Right-hand side vector.
 * @returns The solution vector x.
 */
function forwardSolve(L: number[][], b: number[]): number[] {
  const n = L.length;
  const x = new Array(n);
  for (let i = 0; i < n; i++) {
    let sum = b[i];
    for (let j = 0; j < i; j++) {
      sum -= L[i][j] * x[j];
    }
    x[i] = sum / L[i][i];
  }
  return x;
}

/**
 * Represents a Multivariate Normal (Gaussian) distribution.
 *
 * The distribution is parameterized by a mean vector mu of dimension k
 * and a k x k symmetric positive-definite covariance matrix Sigma.
 * Internally uses Cholesky decomposition for efficient sampling and
 * density evaluation.
 */
export class MultivariateNormal {
  readonly name: string;
  readonly dim: number;
  private readonly L: number[][];
  private readonly logDet: number;
  private rng: RandomFn;

  /**
   * Creates a new Multivariate Normal distribution.
   *
   * @param mean - Mean vector of length k (k >= 1).
   * @param covariance - Covariance matrix (k x k, symmetric positive-definite).
   * @param rng - Optional random number generator; defaults to Math.random.
   * @throws {Error} If dimension is less than 1.
   * @throws {Error} If covariance matrix dimensions do not match mean vector length.
   * @throws {Error} If covariance matrix is not positive definite.
   *
   * @example
   * ```ts
   * const dist = new MultivariateNormal(
   *   [0, 0],
   *   [[1, 0.5], [0.5, 1]]
   * );
   * ```
   */
  constructor(
    public readonly mean: number[],
    public readonly covariance: number[][],
    rng?: RandomFn,
  ) {
    this.rng = resolveRng(rng);
    const k = mean.length;
    if (k < 1) throw new Error(`Invalid parameter 'mean': expected at least 1 dimension, received ${k}`);
    if (covariance.length !== k || covariance.some((r) => r.length !== k)) {
      throw new Error(`Invalid parameter 'covariance': expected a ${k}x${k} matrix to match mean vector length, received ${covariance.length}x${covariance[0]?.length}`);
    }
    this.dim = k;
    this.name = `MultivariateNormal(dim=${k})`;
    this.L = cholesky(covariance);
    this.logDet = logDetFromCholesky(this.L);
  }

  /**
   * Computes the probability density function at point x.
   *
   * Formula: f(x) = exp(logPdf(x))
   *
   * @param x - A point in R^k.
   * @returns The density at x.
   * @throws {Error} If x has incorrect length.
   *
   * @example
   * ```ts
   * const dist = new MultivariateNormal([0, 0], [[1, 0], [0, 1]]);
   * dist.pdf([0, 0]); // ~0.1592 (peak density for 2D standard normal)
   * ```
   */
  pdf(x: number[]): number {
    return Math.exp(this.logPdf(x));
  }

  /**
   * Computes the log probability density function at point x.
   *
   * Formula: log f(x) = -0.5 * (k * log(2pi) + log|Sigma| + (x - mu)^T Sigma^{-1} (x - mu))
   *
   * The quadratic form is computed efficiently via the Cholesky factor:
   * solve L * z = (x - mu), then the quadratic form equals z^T * z.
   *
   * @param x - A point in R^k.
   * @returns The log-density at x.
   * @throws {Error} If x has incorrect length.
   *
   * @example
   * ```ts
   * const dist = new MultivariateNormal([0, 0], [[1, 0], [0, 1]]);
   * dist.logPdf([0, 0]); // ~-1.8379
   * ```
   */
  logPdf(x: number[]): number {
    if (x.length !== this.dim) {
      throw new Error(`Invalid parameter 'x': expected length ${this.dim}, received ${x.length}`);
    }
    const k = this.dim;
    const diff = x.map((v, i) => v - this.mean[i]);

    // Solve L * z = diff, then quadratic form = z^T z
    const z = forwardSolve(this.L, diff);
    let quad = 0;
    for (let i = 0; i < k; i++) quad += z[i] * z[i];

    return -0.5 * (k * Math.log(2 * Math.PI) + this.logDet + quad);
  }

  /**
   * Draws a single random sample from the distribution.
   *
   * Uses the Cholesky factor: X = mu + L * Z where Z ~ N(0, I),
   * with Z generated via the Box-Muller transform.
   *
   * @returns A random vector of length k from the multivariate normal distribution.
   *
   * @example
   * ```ts
   * const dist = new MultivariateNormal([0, 0], [[1, 0.5], [0.5, 1]]);
   * const point = dist.sample(); // e.g., [0.32, -0.17]
   * ```
   */
  sample(): number[] {
    const k = this.dim;
    const z = new Array(k);
    for (let i = 0; i < k; i++) {
      // Box-Muller
      const u1 = this.rng();
      const u2 = this.rng();
      z[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    const result = new Array(k);
    for (let i = 0; i < k; i++) {
      let sum = this.mean[i];
      for (let j = 0; j <= i; j++) {
        sum += this.L[i][j] * z[j];
      }
      result[i] = sum;
    }
    return result;
  }

  /**
   * Draws n independent samples from the distribution.
   *
   * @param n - Number of samples to draw.
   * @returns An array of n random vectors, each of length k.
   *
   * @example
   * ```ts
   * const dist = new MultivariateNormal([0, 0], [[1, 0], [0, 1]]);
   * const samples = dist.sampleN(1000); // 1000 random 2D points
   * ```
   */
  sampleN(n: number): number[][] {
    const samples: number[][] = new Array(n);
    for (let i = 0; i < n; i++) samples[i] = this.sample();
    return samples;
  }

  /**
   * Computes the log-likelihood of a set of observations.
   *
   * Formula: LL = sum_j log f(x_j; mu, Sigma)
   *
   * @param data - An array of observed points in R^k.
   * @returns The total log-likelihood.
   *
   * @example
   * ```ts
   * const dist = new MultivariateNormal([0, 0], [[1, 0], [0, 1]]);
   * const obs = dist.sampleN(100);
   * dist.logLikelihood(obs); // log-likelihood of the observations
   * ```
   */
  logLikelihood(data: number[][]): number {
    let ll = 0;
    for (const x of data) ll += this.logPdf(x);
    return ll;
  }
}
