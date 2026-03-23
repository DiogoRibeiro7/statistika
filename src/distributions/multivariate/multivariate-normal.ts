/**
 * Multivariate Normal (Gaussian) distribution.
 *
 * Parameterized by a mean vector μ and covariance matrix Σ.
 * Supports PDF evaluation, sampling via Cholesky decomposition,
 * and log-likelihood computation.
 */

import { gammaLn } from "../../utils/math";
import { RandomFn } from "../../types";

/**
 * Cholesky decomposition of a symmetric positive-definite matrix.
 * Returns the lower triangular matrix L such that A = L * L^T.
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
          throw new Error("Matrix is not positive definite");
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
 * Compute the log-determinant of a positive-definite matrix
 * from its Cholesky factor L (det(Σ) = det(L)^2, log det = 2 * sum log diag(L)).
 */
function logDetFromCholesky(L: number[][]): number {
  let sum = 0;
  for (let i = 0; i < L.length; i++) {
    sum += Math.log(L[i][i]);
  }
  return 2 * sum;
}

/**
 * Solve L * x = b where L is lower triangular (forward substitution).
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

export class MultivariateNormal {
  readonly name: string;
  readonly dim: number;
  private readonly L: number[][];
  private readonly logDet: number;
  private rng: RandomFn;

  /**
   * @param mean - Mean vector (length k)
   * @param covariance - Covariance matrix (k × k, symmetric positive-definite)
   * @param rng - Optional random number generator (defaults to Math.random).
   */
  constructor(
    public readonly mean: number[],
    public readonly covariance: number[][],
    rng?: RandomFn,
  ) {
    this.rng = rng ?? Math.random;
    const k = mean.length;
    if (k < 1) throw new Error("Dimension must be at least 1");
    if (covariance.length !== k || covariance.some((r) => r.length !== k)) {
      throw new Error("Covariance matrix dimensions must match mean vector length");
    }
    this.dim = k;
    this.name = `MultivariateNormal(dim=${k})`;
    this.L = cholesky(covariance);
    this.logDet = logDetFromCholesky(this.L);
  }

  /**
   * Probability density function at point x.
   */
  pdf(x: number[]): number {
    return Math.exp(this.logPdf(x));
  }

  /**
   * Log probability density function at point x.
   */
  logPdf(x: number[]): number {
    if (x.length !== this.dim) {
      throw new Error(`x must have length ${this.dim}`);
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
   * Draw a single sample from the distribution.
   * Uses the Cholesky factor: X = μ + L * Z where Z ~ N(0, I).
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
   * Draw n samples.
   */
  sampleN(n: number): number[][] {
    const samples: number[][] = new Array(n);
    for (let i = 0; i < n; i++) samples[i] = this.sample();
    return samples;
  }

  /**
   * Log-likelihood of a set of observations.
   */
  logLikelihood(data: number[][]): number {
    let ll = 0;
    for (const x of data) ll += this.logPdf(x);
    return ll;
  }
}
