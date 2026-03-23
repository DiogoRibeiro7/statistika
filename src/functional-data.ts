/**
 * Functional Data Analysis (FDA).
 *
 * - **Basis expansion** — B-spline, Fourier, polynomial basis systems.
 * - **Smoothing** — basis-penalised least squares (P-splines).
 * - **Functional PCA** — eigenfunction decomposition of covariance.
 * - **Functional mean and covariance** — pointwise estimates.
 * - **L² inner product and norm** — for functional objects.
 */

import { solveLinearSystem, symmetricEigen } from "./utils/linalg";
import { mean } from "./utils/descriptive";

// ── Types ─────────────────────────────────────────────────────────────────

/** A functional observation represented by its basis coefficients. */
export interface FunctionalObject {
  /** Basis system used. */
  basis: BasisSystem;
  /** Coefficients in the basis expansion. */
  coefficients: number[];
  /** Evaluate the function at a point t. */
  evaluate: (t: number) => number;
}

export interface BasisSystem {
  type: "bspline" | "fourier" | "polynomial";
  /** Number of basis functions. */
  nBasis: number;
  /** Domain [a, b]. */
  domain: [number, number];
  /** Evaluate all basis functions at t, returns vector of length nBasis. */
  evaluate: (t: number) => number[];
}

export interface FPCAResult {
  /** Eigenvalues (proportion of variance explained by each component). */
  eigenvalues: number[];
  /** Eigenfunctions (functional PCs): each is an array of coefficients. */
  eigenfunctions: number[][];
  /** Scores: scores[i][j] = score of observation i on component j. */
  scores: number[][];
  /** Proportion of variance explained by each component. */
  varianceExplained: number[];
  /** Cumulative proportion of variance explained. */
  cumulativeVariance: number[];
  /** Mean function coefficients. */
  meanCoefficients: number[];
}

// ── Basis Systems ─────────────────────────────────────────────────────────

/**
 * Create a polynomial basis system of degree p (p+1 basis functions).
 */
export function polynomialBasis(degree: number, domain: [number, number] = [0, 1]): BasisSystem {
  return {
    type: "polynomial",
    nBasis: degree + 1,
    domain,
    evaluate: (t: number) => {
      // Normalize to [0, 1]
      const s = (t - domain[0]) / (domain[1] - domain[0]);
      const result = new Array(degree + 1);
      result[0] = 1;
      for (let i = 1; i <= degree; i++) result[i] = s ** i;
      return result;
    },
  };
}

/**
 * Create a Fourier basis system with nBasis functions.
 * nBasis should be odd: 1 constant + (nBasis−1)/2 pairs of sin/cos.
 */
export function fourierBasis(nBasis: number, domain: [number, number] = [0, 1]): BasisSystem {
  if (nBasis < 1) throw new Error("nBasis must be at least 1");
  const nPairs = Math.floor((nBasis - 1) / 2);
  const actualN = 1 + 2 * nPairs;

  return {
    type: "fourier",
    nBasis: actualN,
    domain,
    evaluate: (t: number) => {
      const period = domain[1] - domain[0];
      const s = (t - domain[0]) / period;
      const result: number[] = [1]; // constant
      for (let k = 1; k <= nPairs; k++) {
        result.push(Math.sin(2 * Math.PI * k * s));
        result.push(Math.cos(2 * Math.PI * k * s));
      }
      return result;
    },
  };
}

/**
 * Create a B-spline basis system.
 *
 * Uses cubic B-splines with evenly spaced knots.
 *
 * @param nBasis  Number of basis functions (≥ 4).
 * @param domain  [a, b] domain.
 */
export function bsplineBasis(nBasis: number, domain: [number, number] = [0, 1]): BasisSystem {
  if (nBasis < 4) throw new Error("nBasis must be at least 4 for cubic B-splines");
  const order = 4; // cubic
  const nKnots = nBasis + order;
  const knots: number[] = [];

  // Extended knot sequence
  const [a, b] = domain;
  const nInner = nKnots - 2 * order;
  for (let i = 0; i < order; i++) knots.push(a);
  for (let i = 1; i <= nInner; i++) {
    knots.push(a + (i / (nInner + 1)) * (b - a));
  }
  for (let i = 0; i < order; i++) knots.push(b);

  return {
    type: "bspline",
    nBasis,
    domain,
    evaluate: (t: number) => {
      const result = new Array(nBasis).fill(0);
      // Cox–de Boor recursion
      for (let i = 0; i < nBasis; i++) {
        result[i] = bsplineValue(t, i, order, knots);
      }
      return result;
    },
  };
}

// ── Basis Expansion ───────────────────────────────────────────────────────

/**
 * Fit a functional object to discrete data via least squares.
 *
 * Finds coefficients c such that Σ(y(tⱼ) − Σ cᵢ φᵢ(tⱼ))² is minimized.
 *
 * @param tValues  Observation points (length m).
 * @param yValues  Observed values at those points (length m).
 * @param basis  Basis system to use.
 * @param lambda  Roughness penalty (default 0, no penalty).
 */
export function smoothBasisExpansion(
  tValues: number[],
  yValues: number[],
  basis: BasisSystem,
  lambda = 0,
): FunctionalObject {
  const m = tValues.length;
  if (m !== yValues.length) throw new Error("tValues and yValues must have same length");

  const k = basis.nBasis;
  // Build Φ matrix (m × k): Φ[j][i] = φᵢ(tⱼ)
  const PhitPhi: number[][] = Array.from({ length: k }, () => new Array(k).fill(0));
  const Phity = new Array(k).fill(0);

  for (let j = 0; j < m; j++) {
    const phi = basis.evaluate(tValues[j]);
    for (let i = 0; i < k; i++) {
      Phity[i] += phi[i] * yValues[j];
      for (let l = 0; l < k; l++) {
        PhitPhi[i][l] += phi[i] * phi[l];
      }
    }
  }

  // Add penalty: (Φ'Φ + λI) c = Φ'y
  if (lambda > 0) {
    for (let i = 0; i < k; i++) PhitPhi[i][i] += lambda;
  }

  const coefficients = solveLinearSystem(PhitPhi, Phity);

  return {
    basis,
    coefficients,
    evaluate: (t: number) => {
      const phi = basis.evaluate(t);
      let val = 0;
      for (let i = 0; i < k; i++) val += coefficients[i] * phi[i];
      return val;
    },
  };
}

// ── Functional PCA ────────────────────────────────────────────────────────

/**
 * Functional Principal Component Analysis.
 *
 * 1. Smooth each curve onto a common basis.
 * 2. Compute the covariance matrix of coefficients.
 * 3. Eigendecompose to get functional PCs.
 *
 * @param curves  Array of curves, each as { t: number[], y: number[] }.
 * @param basis  Common basis system.
 * @param nComponents  Number of PCs to retain (default: all).
 * @param lambda  Smoothing penalty (default 0).
 */
export function functionalPCA(
  curves: { t: number[]; y: number[] }[],
  basis: BasisSystem,
  nComponents?: number,
  lambda = 0,
): FPCAResult {
  const n = curves.length;
  if (n < 2) throw new Error("Need at least 2 curves");
  const k = basis.nBasis;
  const nComp = Math.min(nComponents ?? k, k, n);

  // Step 1: Smooth each curve
  const coefMatrix: number[][] = [];
  for (const curve of curves) {
    const fo = smoothBasisExpansion(curve.t, curve.y, basis, lambda);
    coefMatrix.push(fo.coefficients);
  }

  // Step 2: Compute mean coefficients
  const meanCoef = new Array(k).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < k; j++) meanCoef[j] += coefMatrix[i][j];
  }
  for (let j = 0; j < k; j++) meanCoef[j] /= n;

  // Step 3: Centre and compute covariance of coefficients
  const centred: number[][] = coefMatrix.map((row) =>
    row.map((v, j) => v - meanCoef[j]),
  );

  const covMatrix: number[][] = Array.from({ length: k }, () => new Array(k).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < k; j++) {
      for (let l = 0; l < k; l++) {
        covMatrix[j][l] += centred[i][j] * centred[i][l];
      }
    }
  }
  for (let j = 0; j < k; j++) {
    for (let l = 0; l < k; l++) {
      covMatrix[j][l] /= (n - 1);
    }
  }

  // Step 4: Eigendecompose
  const eigen = symmetricEigen(covMatrix);
  const eigenvalues = eigen.eigenvalues.slice(0, nComp);
  const eigenfunctions = eigen.eigenvectors.slice(0, nComp);

  // Step 5: Scores: score[i][j] = centred[i] · eigenvector[j]
  const scores: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < nComp; j++) {
      let dot = 0;
      for (let l = 0; l < k; l++) dot += centred[i][l] * eigenfunctions[j][l];
      row.push(dot);
    }
    scores.push(row);
  }

  // Variance explained
  const totalVar = eigenvalues.reduce((a, b) => a + Math.max(0, b), 0) +
    eigen.eigenvalues.slice(nComp).reduce((a, b) => a + Math.max(0, b), 0);
  const varianceExplained = eigenvalues.map((e) =>
    totalVar > 0 ? Math.max(0, e) / totalVar : 0,
  );
  const cumulativeVariance: number[] = [];
  let cumSum = 0;
  for (const ve of varianceExplained) {
    cumSum += ve;
    cumulativeVariance.push(cumSum);
  }

  return {
    eigenvalues,
    eigenfunctions,
    scores,
    varianceExplained,
    cumulativeVariance,
    meanCoefficients: meanCoef,
  };
}

// ── Functional Mean and Covariance ────────────────────────────────────────

/**
 * Compute the pointwise mean function from multiple curves.
 *
 * @param curves  Array of curves (each as { t, y } on a common grid).
 * @param tGrid  Common evaluation grid.
 */
export function functionalMean(
  curves: { t: number[]; y: number[] }[],
  tGrid: number[],
): number[] {
  const n = curves.length;
  const m = tGrid.length;
  const result = new Array(m).fill(0);

  for (const curve of curves) {
    // Simple interpolation at tGrid points
    for (let j = 0; j < m; j++) {
      result[j] += interpolate(curve.t, curve.y, tGrid[j]);
    }
  }

  for (let j = 0; j < m; j++) result[j] /= n;
  return result;
}

/**
 * Compute the L² inner product between two functional objects.
 *
 * ⟨f, g⟩ = ∫ f(t) g(t) dt (approximated by trapezoidal rule).
 */
export function l2InnerProduct(
  f: FunctionalObject,
  g: FunctionalObject,
  nPoints = 100,
): number {
  const [a, b] = f.basis.domain;
  const h = (b - a) / nPoints;
  let sum = 0;

  for (let i = 0; i <= nPoints; i++) {
    const t = a + i * h;
    const w = i === 0 || i === nPoints ? 0.5 : 1;
    sum += w * f.evaluate(t) * g.evaluate(t);
  }

  return sum * h;
}

/**
 * Compute the L² norm of a functional object.
 */
export function l2Norm(f: FunctionalObject, nPoints = 100): number {
  return Math.sqrt(l2InnerProduct(f, f, nPoints));
}

// ── Helpers ───────────────────────────────────────────────────────────────

/** Cox–de Boor recursion for B-spline basis function i of order p. */
function bsplineValue(t: number, i: number, p: number, knots: number[]): number {
  if (p === 1) {
    return t >= knots[i] && t < knots[i + 1] ? 1 : 0;
  }
  let left = 0;
  let right = 0;
  const denom1 = knots[i + p - 1] - knots[i];
  if (denom1 > 0) {
    left = ((t - knots[i]) / denom1) * bsplineValue(t, i, p - 1, knots);
  }
  const denom2 = knots[i + p] - knots[i + 1];
  if (denom2 > 0) {
    right = ((knots[i + p] - t) / denom2) * bsplineValue(t, i + 1, p - 1, knots);
  }
  return left + right;
}

/** Simple linear interpolation. */
function interpolate(t: number[], y: number[], target: number): number {
  const n = t.length;
  if (target <= t[0]) return y[0];
  if (target >= t[n - 1]) return y[n - 1];
  for (let i = 0; i < n - 1; i++) {
    if (target >= t[i] && target <= t[i + 1]) {
      const frac = (target - t[i]) / (t[i + 1] - t[i]);
      return y[i] + frac * (y[i + 1] - y[i]);
    }
  }
  return y[n - 1];
}
