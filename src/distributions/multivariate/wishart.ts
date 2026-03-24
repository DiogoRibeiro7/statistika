/**
 * Wishart distribution.
 *
 * A distribution over symmetric positive-definite matrices,
 * parameterized by degrees of freedom nu and a scale matrix V.
 * The Wishart is the conjugate prior for the precision matrix
 * (inverse covariance) of a multivariate normal distribution.
 *
 * PDF: f(X; nu, V) = |X|^{(nu-p-1)/2} exp(-tr(V^{-1}X)/2) / (2^{nu*p/2} |V|^{nu/2} Gamma_p(nu/2))
 *
 * where p is the dimension, and Gamma_p is the multivariate gamma function.
 *
 * @example
 * ```ts
 * const dist = new Wishart(5, [[1, 0], [0, 1]]);
 * dist.mean();     // [[5, 0], [0, 5]]
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
 * @param a - The argument (must be > (p-1)/2 for the function to be defined).
 * @param p - The dimension.
 * @returns The log of the multivariate gamma function Gamma_p(a).
 */
function logMultivariateGamma(a: number, p: number): number {
  let result = (p * (p - 1) / 4) * Math.log(Math.PI);
  for (let j = 1; j <= p; j++) {
    result += gammaLn(a + (1 - j) / 2);
  }
  return result;
}

/**
 * Represents a Wishart distribution over p x p symmetric positive-definite matrices.
 *
 * The Wishart distribution with nu degrees of freedom and scale matrix V
 * arises as the distribution of the sample covariance matrix from nu
 * independent draws of a p-dimensional multivariate normal with covariance V.
 */
export class Wishart {
  readonly name: string;
  readonly dim: number;
  private readonly L: number[][]; // Cholesky of V
  private readonly logNormConst: number;
  private rng: RandomFn;

  /**
   * Creates a new Wishart distribution.
   *
   * @param df - Degrees of freedom (must be >= dim).
   * @param scale - Scale matrix V (p x p, symmetric positive-definite).
   * @param rng - Optional random number generator; defaults to Math.random.
   * @throws {Error} If dimension is less than 1.
   * @throws {Error} If scale matrix is not square.
   * @throws {Error} If degrees of freedom is less than the dimension.
   * @throws {Error} If scale matrix is not positive definite.
   *
   * @example
   * ```ts
   * const dist = new Wishart(10, [[2, 1], [1, 2]]);
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
    if (df < p) throw new Error(`Invalid parameter 'df': expected >= dimension (${p}), received ${df}`);
    this.dim = p;
    this.name = `Wishart(df=${df}, dim=${p})`;
    this.L = cholesky(scale);

    // Log determinant of V from Cholesky
    let logDetV = 0;
    for (let i = 0; i < p; i++) logDetV += 2 * Math.log(this.L[i][i]);

    // Normalization constant:
    // log C = (nu/2) log|V| + (nu*p/2) log 2 + log Gamma_p(nu/2)
    // log pdf = -C + ((nu-p-1)/2) log|X| - (1/2) tr(V^{-1} X)
    this.logNormConst =
      (df / 2) * logDetV +
      (df * p / 2) * Math.log(2) +
      logMultivariateGamma(df / 2, p);
  }

  /**
   * Computes the mean matrix of the distribution.
   *
   * Formula: E[X] = nu * V
   *
   * @returns A p x p matrix equal to df times the scale matrix.
   *
   * @example
   * ```ts
   * const dist = new Wishart(5, [[1, 0], [0, 1]]);
   * dist.mean(); // [[5, 0], [0, 5]]
   * ```
   */
  mean(): number[][] {
    const p = this.dim;
    const result: number[][] = Array.from({ length: p }, () => new Array(p));
    for (let i = 0; i < p; i++) {
      for (let j = 0; j < p; j++) {
        result[i][j] = this.df * this.scale[i][j];
      }
    }
    return result;
  }

  /**
   * Computes the log probability density function at a symmetric positive-definite matrix X.
   *
   * Formula: log f(X) = ((nu - p - 1) / 2) * log|X| - (1/2) * tr(V^{-1} X) - log C
   *
   * where log C is the log normalization constant involving |V|, nu, and Gamma_p.
   * The trace term tr(V^{-1} X) is computed efficiently using the Cholesky factor of V.
   *
   * @param X - A p x p symmetric positive-definite matrix.
   * @returns The log-density at X. Returns -Infinity if X is not positive definite.
   * @throws {Error} If X has incorrect dimensions.
   *
   * @example
   * ```ts
   * const dist = new Wishart(5, [[1, 0], [0, 1]]);
   * dist.logPdf([[5, 0], [0, 5]]); // log-density at the mean
   * ```
   */
  logPdf(X: number[][]): number {
    const p = this.dim;
    if (X.length !== p || X.some((r) => r.length !== p)) {
      throw new Error(`X must be a ${p}x${p} matrix`);
    }

    // Log determinant of X via Cholesky
    let Lx: number[][];
    try {
      Lx = cholesky(X);
    } catch {
      return -Infinity; // X is not positive definite
    }
    let logDetX = 0;
    for (let i = 0; i < p; i++) logDetX += 2 * Math.log(Lx[i][i]);

    // Compute trace(V^{-1} X) using Cholesky solve
    // V^{-1} X = (LL^T)^{-1} X; tr = sum of element-wise product of V^{-1} and X
    // More efficiently: tr(V^{-1} X) = tr(L^{-1} X L^{-T}) = ||L^{-1} X_cols||^2
    // Simple approach: solve L * Z = X column by column, then trace = sum of Z_ij^2
    let trace = 0;
    for (let j = 0; j < p; j++) {
      const col = new Array(p);
      for (let i = 0; i < p; i++) col[i] = X[i][j];
      // Forward solve L * z = col
      const z = new Array(p);
      for (let i = 0; i < p; i++) {
        let s = col[i];
        for (let k = 0; k < i; k++) s -= this.L[i][k] * z[k];
        z[i] = s / this.L[i][i];
      }
      for (let i = 0; i < p; i++) trace += z[i] * z[i];
    }

    return (
      ((this.df - p - 1) / 2) * logDetX -
      0.5 * trace -
      this.logNormConst
    );
  }

  /**
   * Computes the probability density function at a symmetric positive-definite matrix X.
   *
   * @param X - A p x p symmetric positive-definite matrix.
   * @returns The density at X. Returns 0 if X is not positive definite.
   * @throws {Error} If X has incorrect dimensions.
   *
   * @example
   * ```ts
   * const dist = new Wishart(5, [[1, 0], [0, 1]]);
   * dist.pdf([[5, 0], [0, 5]]); // density at the mean
   * ```
   */
  pdf(X: number[][]): number {
    return Math.exp(this.logPdf(X));
  }

  /**
   * Draws a single random sample using the Bartlett decomposition.
   *
   * Constructs a lower triangular matrix A where:
   *   - A[i][i] ~ sqrt(Chi-squared(nu - i)) for i = 0, ..., p-1
   *   - A[i][j] ~ N(0, 1) for i > j
   *
   * Then X = L * A * A^T * L^T ~ Wishart(nu, V), where L is the Cholesky
   * factor of the scale matrix V.
   *
   * @returns A random p x p symmetric positive-definite matrix.
   *
   * @example
   * ```ts
   * const dist = new Wishart(5, [[1, 0], [0, 1]]);
   * const matrix = dist.sample(); // random 2x2 positive-definite matrix
   * ```
   */
  sample(): number[][] {
    const p = this.dim;
    const A: number[][] = Array.from({ length: p }, () => new Array(p).fill(0));

    for (let i = 0; i < p; i++) {
      // Diagonal: sqrt of chi-squared(df - i) = sqrt of Gamma((df-i)/2, 1/2)
      A[i][i] = Math.sqrt(sampleChiSquared(this.df - i, this.rng));
      for (let j = 0; j < i; j++) {
        // Off-diagonal: standard normal
        const u1 = this.rng();
        const u2 = this.rng();
        A[i][j] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      }
    }

    // Compute LA = L * A
    const LA: number[][] = Array.from({ length: p }, () => new Array(p).fill(0));
    for (let i = 0; i < p; i++) {
      for (let j = 0; j <= i; j++) {
        let sum = 0;
        for (let k = j; k <= i; k++) {
          sum += this.L[i][k] * A[k][j];
        }
        LA[i][j] = sum;
      }
    }

    // X = LA * LA^T
    const X: number[][] = Array.from({ length: p }, () => new Array(p).fill(0));
    for (let i = 0; i < p; i++) {
      for (let j = 0; j <= i; j++) {
        let sum = 0;
        for (let k = 0; k < p; k++) {
          sum += LA[i][k] * LA[j][k];
        }
        X[i][j] = sum;
        X[j][i] = sum;
      }
    }

    return X;
  }

  /**
   * Draws n independent samples from the distribution.
   *
   * @param n - Number of samples to draw.
   * @returns An array of n random p x p symmetric positive-definite matrices.
   *
   * @example
   * ```ts
   * const dist = new Wishart(5, [[1, 0], [0, 1]]);
   * const samples = dist.sampleN(100); // 100 random matrices
   * ```
   */
  sampleN(n: number): number[][][] {
    const samples: number[][][] = new Array(n);
    for (let i = 0; i < n; i++) samples[i] = this.sample();
    return samples;
  }
}

/**
 * Samples from the Chi-squared(df) distribution.
 *
 * Uses the identity: Chi-squared(df) = 2 * Gamma(df/2, 1).
 *
 * @param df - Degrees of freedom.
 * @param rng - Random number generator.
 * @returns A random variate from Chi-squared(df).
 */
function sampleChiSquared(df: number, rng: RandomFn): number {
  return 2 * sampleGamma(df / 2, rng);
}

/**
 * Samples from the Gamma(shape, 1) distribution using the Marsaglia-Tsang method.
 *
 * For shape < 1, uses the identity: Gamma(shape) = Gamma(shape+1) * U^{1/shape}
 * where U ~ Uniform(0,1).
 *
 * @param shape - The shape parameter (must be positive).
 * @param rng - Random number generator.
 * @returns A random variate from Gamma(shape, 1).
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
