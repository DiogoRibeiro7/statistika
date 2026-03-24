/**
 * Inverse Wishart distribution.
 *
 * A distribution over symmetric positive-definite matrices, defined as the
 * distribution of the inverse of a Wishart-distributed random matrix.
 * The Inverse Wishart is the conjugate prior for the covariance matrix
 * of a multivariate normal distribution.
 *
 * If X ~ Wishart(df, V^{-1}), then X^{-1} ~ InverseWishart(df, V).
 *
 * PDF: f(X; df, Psi) = |Psi|^{df/2} |X|^{-(df+p+1)/2} exp(-tr(Psi X^{-1})/2) / (2^{df*p/2} Gamma_p(df/2))
 *
 * @example
 * ```ts
 * const dist = new InverseWishart(5, [[1, 0], [0, 1]]);
 * dist.mean();     // Psi / (df - p - 1)
 * dist.sample();   // random 2x2 positive-definite matrix
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
 * Computes the log of the multivariate gamma function.
 *
 * Formula: log Gamma_p(a) = p(p-1)/4 * log(pi) + sum_{j=1}^{p} log Gamma(a + (1-j)/2)
 *
 * @param a - The argument (must be > (p-1)/2).
 * @param p - The dimension.
 * @returns The log of the multivariate gamma function.
 */
function logMultivariateGamma(a: number, p: number): number {
  let result = (p * (p - 1) / 4) * Math.log(Math.PI);
  for (let j = 1; j <= p; j++) {
    result += gammaLn(a + (1 - j) / 2);
  }
  return result;
}

/**
 * Inverts a symmetric positive-definite matrix using its Cholesky decomposition.
 *
 * @param A - A symmetric positive-definite matrix.
 * @returns The inverse matrix A^{-1}.
 */
function invertSPD(A: number[][]): number[][] {
  const n = A.length;
  const L = cholesky(A);

  // Invert L (lower triangular)
  const Linv: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    Linv[i][i] = 1 / L[i][i];
    for (let j = i + 1; j < n; j++) {
      let sum = 0;
      for (let k = i; k < j; k++) sum += L[j][k] * Linv[k][i];
      Linv[j][i] = -sum / L[j][j];
    }
  }

  // A^{-1} = L^{-T} * L^{-1}
  const Ainv: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = 0;
      for (let k = i; k < n; k++) sum += Linv[k][i] * Linv[k][j];
      Ainv[i][j] = sum;
      Ainv[j][i] = sum;
    }
  }
  return Ainv;
}

/**
 * Represents an Inverse Wishart distribution over p x p symmetric positive-definite matrices.
 *
 * The Inverse Wishart distribution is parameterized by degrees of freedom `df`
 * and a scale matrix `scale` (Psi). It is the conjugate prior for the covariance
 * matrix of a multivariate normal distribution.
 */
export class InverseWishart {
  readonly name: string;
  readonly dim: number;
  private readonly scaleInv: number[][];
  private readonly logNormConst: number;
  private rng: RandomFn;

  /**
   * Creates a new Inverse Wishart distribution.
   *
   * @param df - Degrees of freedom (must be > dim - 1).
   * @param scale - Scale matrix Psi (p x p, symmetric positive-definite).
   * @param rng - Optional random number generator; defaults to Math.random.
   * @throws {Error} If dimension is less than 1.
   * @throws {Error} If scale matrix is not square.
   * @throws {Error} If degrees of freedom is not greater than dim - 1.
   * @throws {Error} If scale matrix is not positive definite.
   *
   * @example
   * ```ts
   * const dist = new InverseWishart(10, [[2, 1], [1, 2]]);
   * ```
   */
  constructor(
    public readonly df: number,
    public readonly scale: number[][],
    rng?: RandomFn,
  ) {
    this.rng = rng ?? Math.random;
    const p = scale.length;
    if (p < 1) throw new Error(`Invalid parameter 'scale': expected at least 1 dimension, received ${p}`);
    if (scale.some((r) => r.length !== p)) {
      throw new Error(`Invalid parameter 'scale': expected a square matrix, received non-square matrix`);
    }
    if (df <= p - 1) {
      throw new Error(`Invalid parameter 'df': expected > ${p - 1} (dim - 1), received ${df}`);
    }
    this.dim = p;
    this.name = `InverseWishart(df=${df}, dim=${p})`;

    // Precompute inverse of scale for Wishart sampling
    this.scaleInv = invertSPD(scale);

    // Log normalization constant
    const L = cholesky(scale);
    let logDetPsi = 0;
    for (let i = 0; i < p; i++) logDetPsi += 2 * Math.log(L[i][i]);

    this.logNormConst =
      (df / 2) * logDetPsi -
      (df * p / 2) * Math.log(2) -
      logMultivariateGamma(df / 2, p);
  }

  /**
   * Computes the mean matrix of the distribution.
   *
   * Formula: E[X] = Psi / (df - p - 1) for df > p + 1.
   *
   * @returns A p x p matrix.
   * @throws {Error} If df <= p + 1 (mean is undefined).
   *
   * @example
   * ```ts
   * const dist = new InverseWishart(5, [[1, 0], [0, 1]]);
   * dist.mean(); // [[0.5, 0], [0, 0.5]]
   * ```
   */
  mean(): number[][] {
    const p = this.dim;
    if (this.df <= p + 1) {
      throw new Error("Mean is undefined for df <= dim + 1");
    }
    const denom = this.df - p - 1;
    const result: number[][] = Array.from({ length: p }, () => new Array(p));
    for (let i = 0; i < p; i++) {
      for (let j = 0; j < p; j++) {
        result[i][j] = this.scale[i][j] / denom;
      }
    }
    return result;
  }

  /**
   * Computes the log probability density function at a symmetric positive-definite matrix X.
   *
   * @param X - A p x p symmetric positive-definite matrix.
   * @returns The log-density at X. Returns -Infinity if X is not positive definite.
   * @throws {Error} If X has incorrect dimensions.
   */
  logPdf(X: number[][]): number {
    const p = this.dim;
    if (X.length !== p || X.some((r) => r.length !== p)) {
      throw new Error(`X must be a ${p}x${p} matrix`);
    }

    let Lx: number[][];
    try {
      Lx = cholesky(X);
    } catch {
      return -Infinity;
    }
    let logDetX = 0;
    for (let i = 0; i < p; i++) logDetX += 2 * Math.log(Lx[i][i]);

    // Compute trace(Psi * X^{-1})
    const Xinv = invertSPD(X);
    let trace = 0;
    for (let i = 0; i < p; i++) {
      for (let j = 0; j < p; j++) {
        trace += this.scale[i][j] * Xinv[j][i];
      }
    }

    return (
      this.logNormConst -
      ((this.df + p + 1) / 2) * logDetX -
      0.5 * trace
    );
  }

  /**
   * Computes the probability density function at a symmetric positive-definite matrix X.
   *
   * @param X - A p x p symmetric positive-definite matrix.
   * @returns The density at X.
   */
  pdf(X: number[][]): number {
    return Math.exp(this.logPdf(X));
  }

  /**
   * Draws a single random sample by inverting a Wishart sample.
   *
   * Generates W ~ Wishart(df, Psi^{-1}), then returns W^{-1} ~ InverseWishart(df, Psi).
   *
   * @returns A random p x p symmetric positive-definite matrix.
   */
  sample(): number[][] {
    // Sample W ~ Wishart(df, scaleInv) via Bartlett decomposition
    const W = this.sampleWishart();
    return invertSPD(W);
  }

  /**
   * Draws n independent samples from the distribution.
   *
   * @param n - Number of samples to draw.
   * @returns An array of n random p x p symmetric positive-definite matrices.
   */
  sampleN(n: number): number[][][] {
    const samples: number[][][] = new Array(n);
    for (let i = 0; i < n; i++) samples[i] = this.sample();
    return samples;
  }

  /**
   * Internal: generates a Wishart(df, scaleInv) sample using Bartlett decomposition.
   */
  private sampleWishart(): number[][] {
    const p = this.dim;
    const L = cholesky(this.scaleInv);

    // Bartlett decomposition
    const A: number[][] = Array.from({ length: p }, () => new Array(p).fill(0));
    for (let i = 0; i < p; i++) {
      A[i][i] = Math.sqrt(sampleChiSquared(this.df - i, this.rng));
      for (let j = 0; j < i; j++) {
        const u1 = this.rng();
        const u2 = this.rng();
        A[i][j] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      }
    }

    // LA = L * A
    const LA: number[][] = Array.from({ length: p }, () => new Array(p).fill(0));
    for (let i = 0; i < p; i++) {
      for (let j = 0; j <= i; j++) {
        let sum = 0;
        for (let k = j; k <= i; k++) sum += L[i][k] * A[k][j];
        LA[i][j] = sum;
      }
    }

    // W = LA * LA^T
    const W: number[][] = Array.from({ length: p }, () => new Array(p).fill(0));
    for (let i = 0; i < p; i++) {
      for (let j = 0; j <= i; j++) {
        let sum = 0;
        for (let k = 0; k < p; k++) sum += LA[i][k] * LA[j][k];
        W[i][j] = sum;
        W[j][i] = sum;
      }
    }
    return W;
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
