/**
 * Generalized Additive Models (GAM) module.
 *
 * Provides spline basis construction, backfitting with GCV-based
 * smoothing parameter selection, and partial dependence computation.
 */

import { mean } from "./utils/descriptive";
import { solveLinearSystem } from "./utils/linalg";
import { GLMFamily, gaussian, poisson, binomial } from "./glm";

// ---------------------------------------------------------------------------
// Interfaces
// ---------------------------------------------------------------------------

/** Describes a single smooth term in the additive predictor. */
export interface SmoothTerm {
  /** Column index (or indices for tensor products) into the design matrix. */
  variables: number[];
  /** Type of basis to use for this term. */
  basisType: "cubic" | "thinPlate" | "tensor";
  /** Number of basis functions (knots) to use. */
  nBasis: number;
  /** Optional fixed penalty parameter; if omitted, selected via GCV. */
  lambda?: number;
}

/** Result returned by `gamFit`. */
export interface GAMResult {
  /** Fitted values on the response scale. */
  fitted: number[];
  /** Linear predictor values (before inverse-link). */
  linearPredictor: number[];
  /** Intercept term. */
  intercept: number;
  /** Smooth function evaluations for each term (nObs x nTerms). */
  smoothValues: number[][];
  /** Coefficients for each smooth term. */
  coefficients: number[][];
  /** Selected (or provided) penalty parameters per term. */
  lambdas: number[];
  /** Effective degrees of freedom per term. */
  edf: number[];
  /** GCV score at convergence. */
  gcv: number;
  /** Number of outer (backfitting) iterations performed. */
  iterations: number;
  /** Whether the algorithm converged. */
  converged: boolean;
}

/** Result from partial dependence evaluation. */
export interface PartialDependenceResult {
  /** Grid values at which the smooth was evaluated. */
  grid: number[];
  /** Predicted partial effect at each grid point. */
  values: number[];
  /** Lower bound of approximate confidence band. */
  lower: number[];
  /** Upper bound of approximate confidence band. */
  upper: number[];
}

// ---------------------------------------------------------------------------
// Basis construction helpers
// ---------------------------------------------------------------------------

/**
 * Truncated-power helper: max(0, x)^3.
 */
function truncPow3(x: number): number {
  return x > 0 ? x * x * x : 0;
}

/**
 * Build a cubic spline (B-spline-like truncated power) basis matrix.
 *
 * @param x      Vector of predictor values (length n).
 * @param nBasis Number of basis functions (including intercept & linear).
 * @returns      Basis matrix of shape n x nBasis stored row-major.
 */
export function cubicSplineBasis(x: number[], nBasis: number): number[][] {
  const n = x.length;
  const nKnots = Math.max(nBasis - 2, 1);

  // Place knots at quantiles of x.
  const sorted = [...x].sort((a, b) => a - b);
  const knots: number[] = [];
  for (let k = 0; k < nKnots; k++) {
    const p = (k + 1) / (nKnots + 1);
    const idx = Math.min(Math.floor(p * (n - 1)), n - 1);
    knots.push(sorted[idx]);
  }

  const basis: number[][] = new Array(n);
  for (let i = 0; i < n; i++) {
    const row = new Array(nBasis).fill(0);
    // Column 0: constant
    row[0] = 1;
    // Column 1: linear
    if (nBasis > 1) {
      row[1] = x[i];
    }
    // Remaining columns: truncated cubic at each knot
    for (let k = 0; k < nKnots && k + 2 < nBasis; k++) {
      row[k + 2] = truncPow3(x[i] - knots[k]);
    }
    basis[i] = row;
  }
  return basis;
}

/**
 * Build a thin-plate spline basis matrix using radial basis functions.
 *
 * For 1-D inputs the thin-plate radial basis is |x - knot|^2 * log(|x - knot|)
 * (with the convention 0 * log(0) = 0).
 *
 * @param x      Vector of predictor values (length n).
 * @param nBasis Number of basis functions (including polynomial null-space).
 * @returns      Basis matrix of shape n x nBasis stored row-major.
 */
export function thinPlateBasis(x: number[], nBasis: number): number[][] {
  const n = x.length;
  const nKnots = Math.max(nBasis - 2, 1);

  const sorted = [...x].sort((a, b) => a - b);
  const knots: number[] = [];
  for (let k = 0; k < nKnots; k++) {
    const p = (k + 1) / (nKnots + 1);
    const idx = Math.min(Math.floor(p * (n - 1)), n - 1);
    knots.push(sorted[idx]);
  }

  const basis: number[][] = new Array(n);
  for (let i = 0; i < n; i++) {
    const row = new Array(nBasis).fill(0);
    // Null-space: constant + linear
    row[0] = 1;
    if (nBasis > 1) {
      row[1] = x[i];
    }
    for (let k = 0; k < nKnots && k + 2 < nBasis; k++) {
      const r = Math.abs(x[i] - knots[k]);
      if (r > 0) {
        row[k + 2] = r * r * Math.log(r);
      } else {
        row[k + 2] = 0;
      }
    }
    basis[i] = row;
  }
  return basis;
}

/**
 * Build a tensor product basis from two marginal bases.
 *
 * Given marginal bases B1 (n x m1) and B2 (n x m2), the tensor product
 * basis has n rows and m1*m2 columns formed by row-wise Kronecker products.
 *
 * @param x1     First predictor vector.
 * @param x2     Second predictor vector.
 * @param nBasis Number of marginal basis functions for each predictor.
 * @returns      Tensor product basis matrix (n x nBasis^2).
 */
export function tensorProductBasis(
  x1: number[],
  x2: number[],
  nBasis: number
): number[][] {
  const B1 = cubicSplineBasis(x1, nBasis);
  const B2 = cubicSplineBasis(x2, nBasis);
  const n = x1.length;
  const m1 = nBasis;
  const m2 = nBasis;
  const totalCols = m1 * m2;

  const basis: number[][] = new Array(n);
  for (let i = 0; i < n; i++) {
    const row = new Array(totalCols);
    for (let j1 = 0; j1 < m1; j1++) {
      for (let j2 = 0; j2 < m2; j2++) {
        row[j1 * m2 + j2] = B1[i][j1] * B2[i][j2];
      }
    }
    basis[i] = row;
  }
  return basis;
}

// ---------------------------------------------------------------------------
// Internal linear-algebra utilities
// ---------------------------------------------------------------------------

/**
 * Multiply B^T * w * B  where w is a diagonal weight vector.
 * Returns a p x p matrix.
 */
function weightedCross(B: number[][], w: number[], p: number): number[][] {
  const n = B.length;
  const result: number[][] = Array.from({ length: p }, () =>
    new Array(p).fill(0)
  );
  for (let i = 0; i < n; i++) {
    const wi = w[i];
    for (let j = 0; j < p; j++) {
      const bj = B[i][j] * wi;
      for (let k = j; k < p; k++) {
        result[j][k] += bj * B[i][k];
      }
    }
  }
  // Symmetrise
  for (let j = 0; j < p; j++) {
    for (let k = 0; k < j; k++) {
      result[j][k] = result[k][j];
    }
  }
  return result;
}

/**
 * Multiply B^T * w * z  where w is diagonal and z is a vector.
 * Returns a vector of length p.
 */
function weightedCrossVec(
  B: number[][],
  w: number[],
  z: number[],
  p: number
): number[] {
  const n = B.length;
  const result = new Array(p).fill(0);
  for (let i = 0; i < n; i++) {
    const wz = w[i] * z[i];
    for (let j = 0; j < p; j++) {
      result[j] += B[i][j] * wz;
    }
  }
  return result;
}

/**
 * Compute the hat/smoother matrix trace  tr(H) for a single penalised
 * term so that effective degrees of freedom can be estimated.
 *
 * H = B (B^T W B + lambda * I)^{-1} B^T W
 * tr(H) = sum_j  (B (A^{-1}) B^T W)_{jj}  -- but we use the identity
 * tr(H) = tr(A^{-1} B^T W B) which avoids forming the full n x n matrix.
 */
function effectiveDf(
  BtWB: number[][],
  lambda: number,
  p: number
): number {
  // A = BtWB + lambda * I
  const A: number[][] = BtWB.map((row) => [...row]);
  for (let j = 0; j < p; j++) {
    A[j][j] += lambda;
  }

  // We need tr(A^{-1} BtWB).  Solve A X = BtWB column by column.
  let trace = 0;
  for (let col = 0; col < p; col++) {
    const rhs = BtWB.map((row) => row[col]);
    const x = solveLinearSystem(A, rhs);
    trace += x[col];
  }
  return trace;
}

// ---------------------------------------------------------------------------
// GCV score
// ---------------------------------------------------------------------------

function gcvScore(
  residuals: number[],
  weights: number[],
  totalEdf: number
): number {
  const n = residuals.length;
  let rss = 0;
  for (let i = 0; i < n; i++) {
    rss += weights[i] * residuals[i] * residuals[i];
  }
  const denom = (1 - totalEdf / n) * (1 - totalEdf / n);
  return rss / (n * denom);
}

// ---------------------------------------------------------------------------
// gamFit – main fitting routine using backfitting
// ---------------------------------------------------------------------------

/**
 * Fit a Generalized Additive Model via penalised iteratively re-weighted
 * least squares (P-IRLS) with backfitting.
 *
 * Algorithm outline (for the Gaussian case this simplifies to plain backfitting):
 *   1. Initialise all smooth functions to zero.
 *   2. Outer loop (IRLS): compute working weights and working response.
 *   3. Inner loop (backfitting): cycle through smooth terms, compute partial
 *      residuals and fit each term with a penalised (ridge) spline.
 *   4. Use GCV to choose lambda for any term that does not have a fixed lambda.
 *   5. Repeat until convergence.
 *
 * @param X       Design matrix: X[i][j] is the j-th predictor for observation i.
 * @param y       Response vector (length n).
 * @param terms   Array of SmoothTerm specifications.
 * @param family  GLM family (link / variance functions). Defaults to gaussian.
 * @param options Optional tuning parameters.
 * @returns       A GAMResult object.
 */
export function gamFit(
  X: number[][],
  y: number[],
  terms: SmoothTerm[],
  family?: GLMFamily,
  options?: {
    maxOuterIter?: number;
    maxInnerIter?: number;
    tol?: number;
    lambdaGrid?: number[];
  }
): GAMResult {
  const fam: GLMFamily = family ?? gaussian;
  const maxOuter = options?.maxOuterIter ?? 50;
  const maxInner = options?.maxInnerIter ?? 20;
  const tol = options?.tol ?? 1e-7;
  const lambdaGrid =
    options?.lambdaGrid ?? [1e-6, 1e-4, 1e-2, 0.1, 1, 10, 100, 1000];

  const n = y.length;
  const nTerms = terms.length;

  // --- Build basis matrices for each term -----------------------------------
  const bases: number[][][] = [];
  for (const term of terms) {
    let B: number[][];
    if (term.basisType === "tensor" && term.variables.length >= 2) {
      const x1 = X.map((row) => row[term.variables[0]]);
      const x2 = X.map((row) => row[term.variables[1]]);
      B = tensorProductBasis(x1, x2, term.nBasis);
    } else if (term.basisType === "thinPlate") {
      const x = X.map((row) => row[term.variables[0]]);
      B = thinPlateBasis(x, term.nBasis);
    } else {
      const x = X.map((row) => row[term.variables[0]]);
      B = cubicSplineBasis(x, term.nBasis);
    }
    bases.push(B);
  }

  // Number of basis columns per term
  const pTerms = bases.map((B) => B[0].length);

  // --- Initialisation -------------------------------------------------------
  const intercept_init = mean(y);
  let mu = new Array(n).fill(intercept_init);
  let eta = mu.map((m) => fam.link.link(m));

  // Smooth contributions per term, initialised to zero
  const smoothVals: number[][] = terms.map(() => new Array(n).fill(0));
  const coefs: number[][] = terms.map((_, t) =>
    new Array(pTerms[t]).fill(0)
  );
  const lambdas: number[] = terms.map((t) => t.lambda ?? 1.0);
  const edfArr: number[] = new Array(nTerms).fill(0);

  let converged = false;
  let outerIter = 0;

  // --- Outer loop (IRLS) ----------------------------------------------------
  for (outerIter = 0; outerIter < maxOuter; outerIter++) {
    // Working weights and working response (Fisher scoring step)
    const w = new Array(n);
    const z = new Array(n);
    for (let i = 0; i < n; i++) {
      const dmu = fam.link.derivative(eta[i]);
      const vi = fam.variance(mu[i]);
      w[i] = 1 / (dmu * dmu * vi + 1e-12);
      z[i] = eta[i] + (y[i] - mu[i]) * dmu;
    }

    // --- Backfitting inner loop ---------------------------------------------
    let interceptVal = mean(z);
    for (let inner = 0; inner < maxInner; inner++) {
      let maxChange = 0;

      for (let t = 0; t < nTerms; t++) {
        const B = bases[t];
        const p = pTerms[t];

        // Partial residuals: remove intercept and all other smooth terms
        const partialRes = new Array(n);
        for (let i = 0; i < n; i++) {
          let other = interceptVal;
          for (let s = 0; s < nTerms; s++) {
            if (s !== t) other += smoothVals[s][i];
          }
          partialRes[i] = z[i] - other;
        }

        // B^T W B and B^T W partialRes
        const BtWB = weightedCross(B, w, p);
        const BtWz = weightedCrossVec(B, w, partialRes, p);

        // --- Select lambda via GCV if not fixed -----------------------------
        let bestLambda = lambdas[t];
        if (terms[t].lambda === undefined) {
          let bestGcv = Infinity;
          for (const lam of lambdaGrid) {
            // Penalised normal equations: (B^T W B + lam I) beta = B^T W z
            const A: number[][] = BtWB.map((row) => [...row]);
            for (let j = 0; j < p; j++) {
              A[j][j] += lam;
            }
            const beta = solveLinearSystem(A, BtWz);

            // Residuals from this term
            const res = new Array(n);
            for (let i = 0; i < n; i++) {
              let fitted = 0;
              for (let j = 0; j < p; j++) fitted += B[i][j] * beta[j];
              res[i] = partialRes[i] - fitted;
            }
            const edfCandidate = effectiveDf(BtWB, lam, p);
            const g = gcvScore(res, w, edfCandidate);
            if (g < bestGcv) {
              bestGcv = g;
              bestLambda = lam;
            }
          }
        }
        lambdas[t] = bestLambda;

        // Solve penalised system with chosen lambda
        const A: number[][] = BtWB.map((row) => [...row]);
        for (let j = 0; j < p; j++) {
          A[j][j] += bestLambda;
        }
        const newCoefs = solveLinearSystem(A, BtWz);

        // Update smooth values
        const oldSmooth = smoothVals[t];
        const newSmooth = new Array(n);
        for (let i = 0; i < n; i++) {
          let val = 0;
          for (let j = 0; j < p; j++) val += B[i][j] * newCoefs[j];
          newSmooth[i] = val;
        }

        // Centre the smooth (identifiability constraint)
        const sm = mean(newSmooth);
        for (let i = 0; i < n; i++) newSmooth[i] -= sm;
        interceptVal += sm;

        // Check convergence
        let change = 0;
        for (let i = 0; i < n; i++) {
          const d = newSmooth[i] - oldSmooth[i];
          change += d * d;
        }
        maxChange = Math.max(maxChange, Math.sqrt(change / n));

        smoothVals[t] = newSmooth;
        coefs[t] = newCoefs;
        edfArr[t] = effectiveDf(BtWB, bestLambda, p);
      }

      if (maxChange < tol) break;
    }

    // Reconstruct eta and mu
    const etaOld = [...eta];
    for (let i = 0; i < n; i++) {
      eta[i] = interceptVal;
      for (let t = 0; t < nTerms; t++) {
        eta[i] += smoothVals[t][i];
      }
      mu[i] = fam.link.inverse(eta[i]);
    }

    // Check outer convergence
    let etaChange = 0;
    for (let i = 0; i < n; i++) {
      const d = eta[i] - etaOld[i];
      etaChange += d * d;
    }
    if (Math.sqrt(etaChange / n) < tol) {
      converged = true;
      outerIter++;
      break;
    }
  }

  // --- Final GCV score ------------------------------------------------------
  const residuals = new Array(n);
  const wFinal = new Array(n).fill(1);
  for (let i = 0; i < n; i++) {
    residuals[i] = y[i] - mu[i];
  }
  const totalEdf = edfArr.reduce((a, b) => a + b, 1); // +1 for intercept
  const gcvFinal = gcvScore(residuals, wFinal, totalEdf);

  // Reconstruct intercept (mean of eta minus smooth contributions)
  let finalIntercept = mean(eta);
  for (let t = 0; t < nTerms; t++) {
    finalIntercept -= mean(smoothVals[t]);
  }

  return {
    fitted: mu,
    linearPredictor: eta,
    intercept: finalIntercept,
    smoothValues: smoothVals,
    coefficients: coefs,
    lambdas,
    edf: edfArr,
    gcv: gcvFinal,
    iterations: outerIter,
    converged,
  };
}

// ---------------------------------------------------------------------------
// Partial dependence
// ---------------------------------------------------------------------------

/**
 * Compute the partial dependence of a fitted GAM smooth term over a grid.
 *
 * For each grid point the smooth is evaluated while all other predictors are
 * marginalised (averaged over the training data).
 *
 * @param gamResult  A fitted GAMResult.
 * @param termIndex  Index of the smooth term to evaluate.
 * @param X          Original design matrix used for fitting.
 * @param term       The SmoothTerm descriptor for this term.
 * @param gridSize   Number of evaluation points (default 100).
 * @returns          PartialDependenceResult with grid, values, and bands.
 */
export function partialDependence(
  gamResult: GAMResult,
  termIndex: number,
  X: number[][],
  term: SmoothTerm,
  gridSize: number = 100
): PartialDependenceResult {
  const n = X.length;
  const varIdx = term.variables[0];

  // Build evaluation grid spanning the range of the predictor
  const col = X.map((row) => row[varIdx]);
  const xMin = Math.min(...col);
  const xMax = Math.max(...col);
  const grid: number[] = [];
  for (let g = 0; g < gridSize; g++) {
    grid.push(xMin + (g / (gridSize - 1)) * (xMax - xMin));
  }

  const coefs = gamResult.coefficients[termIndex];
  const p = coefs.length;

  // Build basis for grid points
  let B: number[][];
  if (term.basisType === "thinPlate") {
    B = thinPlateBasis(grid, term.nBasis);
  } else {
    B = cubicSplineBasis(grid, term.nBasis);
  }

  // Evaluate smooth at each grid point
  const values: number[] = new Array(gridSize);
  for (let g = 0; g < gridSize; g++) {
    let val = 0;
    for (let j = 0; j < p; j++) {
      val += B[g][j] * coefs[j];
    }
    values[g] = val;
  }

  // Centre values
  const valMean = mean(values);
  for (let g = 0; g < gridSize; g++) {
    values[g] -= valMean;
  }

  // Approximate standard errors using the Bayesian posterior covariance
  // Vb ~ (B^T B + lambda I)^{-1} * scale.  We use a rough scale estimate
  // from the fitted model residuals.
  const lambda = gamResult.lambdas[termIndex];
  const BtB = weightedCross(
    B,
    new Array(gridSize).fill(1),
    p
  );

  // scale estimate from overall GCV
  const scale = gamResult.gcv;

  const lower: number[] = new Array(gridSize);
  const upper: number[] = new Array(gridSize);

  // For each grid point compute pointwise se = sqrt( b^T (BtB + lam I)^{-1} b * scale )
  const A: number[][] = BtB.map((row) => [...row]);
  for (let j = 0; j < p; j++) {
    A[j][j] += lambda;
  }

  for (let g = 0; g < gridSize; g++) {
    const bVec = B[g];
    const solved = solveLinearSystem(A, bVec);
    let varEst = 0;
    for (let j = 0; j < p; j++) {
      varEst += bVec[j] * solved[j];
    }
    const se = Math.sqrt(Math.max(0, varEst * scale));
    lower[g] = values[g] - 1.96 * se;
    upper[g] = values[g] + 1.96 * se;
  }

  return { grid, values, lower, upper };
}
