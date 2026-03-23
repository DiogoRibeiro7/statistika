/**
 * Wishart distribution.
 *
 * A distribution over symmetric positive-definite matrices,
 * parameterized by degrees of freedom ν and a scale matrix V.
 * The Wishart is the conjugate prior for the precision matrix
 * (inverse covariance) of a multivariate normal distribution.
 */

import { gammaLn } from "../../utils/math";
import { RandomFn } from "../../types";

/**
 * Cholesky decomposition of a symmetric positive-definite matrix.
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
 * Log of the multivariate gamma function:
 * Γ_p(a) = π^{p(p-1)/4} Π_{j=1}^{p} Γ(a + (1-j)/2)
 */
function logMultivariateGamma(a: number, p: number): number {
  let result = (p * (p - 1) / 4) * Math.log(Math.PI);
  for (let j = 1; j <= p; j++) {
    result += gammaLn(a + (1 - j) / 2);
  }
  return result;
}

export class Wishart {
  readonly name: string;
  readonly dim: number;
  private readonly L: number[][]; // Cholesky of V
  private readonly logNormConst: number;
  private rng: RandomFn;

  /**
   * @param df - Degrees of freedom (must be >= dim)
   * @param scale - Scale matrix V (p × p, symmetric positive-definite)
   * @param rng - Optional random number generator (defaults to Math.random).
   */
  constructor(
    public readonly df: number,
    public readonly scale: number[][],
    rng?: RandomFn,
  ) {
    this.rng = rng ?? Math.random;
    const p = scale.length;
    if (p < 1) throw new Error("Dimension must be at least 1");
    if (scale.some((r) => r.length !== p)) {
      throw new Error("Scale matrix must be square");
    }
    if (df < p) throw new Error("Degrees of freedom must be >= dimension");
    this.dim = p;
    this.name = `Wishart(df=${df}, dim=${p})`;
    this.L = cholesky(scale);

    // Log determinant of V from Cholesky
    let logDetV = 0;
    for (let i = 0; i < p; i++) logDetV += 2 * Math.log(this.L[i][i]);

    // Normalization constant:
    // log C = (ν/2) log|V| + (νp/2) log 2 + log Γ_p(ν/2)
    // log pdf = -C + ((ν-p-1)/2) log|X| - (1/2) tr(V^{-1} X)
    this.logNormConst =
      (df / 2) * logDetV +
      (df * p / 2) * Math.log(2) +
      logMultivariateGamma(df / 2, p);
  }

  /**
   * Mean: E[X] = ν * V
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
   * Log PDF at a symmetric positive-definite matrix X.
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
   * PDF at a symmetric positive-definite matrix X.
   */
  pdf(X: number[][]): number {
    return Math.exp(this.logPdf(X));
  }

  /**
   * Draw a sample using the Bartlett decomposition.
   *
   * If A is a lower triangular matrix where:
   *   A[i][i] ~ sqrt(Chi2(ν - i)) for i = 0..p-1
   *   A[i][j] ~ N(0,1) for i > j
   * Then X = L * A * A^T * L^T ~ Wishart(ν, V)
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
   * Draw n samples.
   */
  sampleN(n: number): number[][][] {
    const samples: number[][][] = new Array(n);
    for (let i = 0; i < n; i++) samples[i] = this.sample();
    return samples;
  }
}

/** Sample from Chi-squared(df) = Gamma(df/2, 1/2) * 2 = Gamma(df/2, 1) * 2 */
function sampleChiSquared(df: number, rng: RandomFn): number {
  return 2 * sampleGamma(df / 2, rng);
}

/** Marsaglia-Tsang method for Gamma(shape, 1). */
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
