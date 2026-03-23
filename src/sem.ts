/**
 * Structural Equation Modeling (SEM).
 *
 * - **Path analysis** — direct, indirect, and total effects from a DAG.
 * - **Confirmatory Factor Analysis (CFA)** — factor loadings, communalities.
 * - **Fit indices** — χ², RMSEA, CFI, TLI, SRMR, AIC, BIC.
 * - **Covariance structure** — model-implied vs observed covariance.
 */

import { mean } from "./utils/descriptive";
import { solveLinearSystem, invertMatrix, symmetricEigen } from "./utils/linalg";

// ── Types ─────────────────────────────────────────────────────────────────

/**
 * A directed path (regression) in a Structural Equation Model.
 * Represents a causal relationship from one variable to another.
 */
export interface SEMPath {
  /** Source (predictor) variable name. */
  from: string;
  /** Target (outcome) variable name. */
  to: string;
  /** Fixed coefficient value. If undefined, estimated from data via OLS. */
  coefficient?: number;
}

/**
 * Factor-indicator loading specification for Confirmatory Factor Analysis.
 */
export interface CFALoading {
  /** Latent factor name. */
  factor: string;
  /** Observed indicator (variable) name. */
  indicator: string;
  /** Fixed loading value. If undefined, estimated from data. */
  loading?: number;
}

/**
 * Result of a path analysis.
 */
export interface PathAnalysisResult {
  /** Estimated path coefficients. */
  coefficients: Map<string, number>;
  /** Direct effects matrix (variable → variable → coefficient). */
  directEffects: Map<string, Map<string, number>>;
  /** Total effects matrix (including indirect). */
  totalEffects: Map<string, Map<string, number>>;
  /** R² for each endogenous variable. */
  rSquared: Map<string, number>;
}

/**
 * Result of a Confirmatory Factor Analysis.
 */
export interface CFAResult {
  /** Estimated factor loadings (factor → indicator → loading). */
  loadings: Map<string, Map<string, number>>;
  /** Communalities for each indicator: proportion of variance explained. */
  communalities: Map<string, number>;
  /** Unique variances (1 − communality). */
  uniquenesses: Map<string, number>;
  /** Factor correlations (if oblique). */
  factorCorrelations?: number[][];
  /** Model fit indices. */
  fit: SEMFitIndices;
}

/**
 * Fit indices for evaluating SEM / CFA model quality.
 */
export interface SEMFitIndices {
  /** Chi-squared test statistic. */
  chiSquared: number;
  /** Degrees of freedom. */
  df: number;
  /** p-value for chi-squared. */
  pValue: number;
  /** Root Mean Square Error of Approximation. */
  rmsea: number;
  /** Comparative Fit Index. */
  cfi: number;
  /** Tucker-Lewis Index. */
  tli: number;
  /** Standardized Root Mean Square Residual. */
  srmr: number;
  /** AIC. */
  aic: number;
  /** BIC. */
  bic: number;
}

// ── Path Analysis ─────────────────────────────────────────────────────────

/**
 * Path analysis via sequential OLS regressions.
 *
 * Estimates standardized direct effects from a set of directed paths (DAG),
 * then computes indirect and total effects via the Leontief inverse (I - B)^{-1},
 * where B is the matrix of direct effects.
 *
 * @param data - Map from variable name to numeric array (all same length)
 * @param paths - Directed paths defining the structural model
 * @returns PathAnalysisResult with coefficients, direct/total effects, and R-squared values
 * @throws Error if any variable has a length mismatch
 *
 * @example
 * ```ts
 * const data = { X: [1, 2, 3, 4], M: [2, 3, 5, 6], Y: [3, 5, 7, 9] };
 * const paths = [{ from: "X", to: "M" }, { from: "M", to: "Y" }];
 * const result = pathAnalysis(data, paths);
 * // result.coefficients — Map of "X -> M" => coefficient, etc.
 * // result.totalEffects — includes indirect effect of X on Y through M
 * ```
 */
export function pathAnalysis(
  data: Record<string, number[]>,
  paths: SEMPath[],
): PathAnalysisResult {
  const varNames = Object.keys(data);
  const n = data[varNames[0]].length;
  for (const v of varNames) {
    if (data[v].length !== n) throw new Error(`Variable "${v}" length mismatch`);
  }

  // Identify endogenous variables (those that appear as "to")
  const endogenous = new Set(paths.map((p) => p.to));
  const allVars = [...new Set([...varNames, ...paths.map((p) => p.from), ...paths.map((p) => p.to)])];
  const varIndex = new Map<string, number>();
  allVars.forEach((v, i) => varIndex.set(v, i));
  const k = allVars.length;

  // Build direct effects via OLS for each endogenous variable
  const directEffects = new Map<string, Map<string, number>>();
  const coefficients = new Map<string, number>();
  const rSquared = new Map<string, number>();

  for (const v of allVars) directEffects.set(v, new Map());

  for (const endoVar of endogenous) {
    const predictors = paths
      .filter((p) => p.to === endoVar)
      .map((p) => p.from);

    if (predictors.length === 0) continue;

    // Build X matrix and y vector (standardised)
    const yRaw = data[endoVar];
    if (!yRaw) continue;
    const { values: yStd, m: yM, s: yS } = standardize(yRaw);

    const xCols: number[][] = [];
    const xStats: { m: number; s: number }[] = [];
    for (const pred of predictors) {
      const raw = data[pred];
      if (!raw) continue;
      const { values: std, m, s } = standardize(raw);
      xCols.push(std);
      xStats.push({ m, s });
    }

    if (xCols.length === 0) continue;

    // OLS on standardised variables (no intercept → standardised coefficients)
    const pLen = xCols.length;
    const XtX: number[][] = Array.from({ length: pLen }, () => new Array(pLen).fill(0));
    const Xty = new Array(pLen).fill(0);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < pLen; j++) {
        Xty[j] += xCols[j][i] * yStd[i];
        for (let l = 0; l < pLen; l++) {
          XtX[j][l] += xCols[j][i] * xCols[l][i];
        }
      }
    }
    const beta = solveLinearSystem(XtX, Xty);

    // Store coefficients
    let ssReg = 0;
    for (let i = 0; i < n; i++) {
      let yHat = 0;
      for (let j = 0; j < pLen; j++) yHat += beta[j] * xCols[j][i];
      ssReg += yHat * yHat;
    }
    let ssTot = 0;
    for (let i = 0; i < n; i++) ssTot += yStd[i] * yStd[i];
    rSquared.set(endoVar, ssTot > 0 ? ssReg / ssTot : 0);

    for (let j = 0; j < pLen; j++) {
      const pred = predictors[j];
      const fixedPath = paths.find((p) => p.from === pred && p.to === endoVar);
      const coef = fixedPath?.coefficient ?? beta[j];
      directEffects.get(pred)!.set(endoVar, coef);
      coefficients.set(`${pred} -> ${endoVar}`, coef);
    }
  }

  // Total effects via (I − B)⁻¹
  // B is the matrix of direct effects: B[i][j] = effect of j on i
  const B: number[][] = Array.from({ length: k }, () => new Array(k).fill(0));
  for (const [from, targets] of directEffects) {
    for (const [to, coef] of targets) {
      const i = varIndex.get(to)!;
      const j = varIndex.get(from)!;
      B[i][j] = coef;
    }
  }

  // (I - B)
  const ImB: number[][] = Array.from({ length: k }, (_, i) =>
    Array.from({ length: k }, (_, j) => (i === j ? 1 : 0) - B[i][j]),
  );

  const ImBinv = invertMatrix(ImB);
  const totalEffects = new Map<string, Map<string, number>>();
  for (const v of allVars) totalEffects.set(v, new Map());

  if (ImBinv) {
    for (let i = 0; i < k; i++) {
      for (let j = 0; j < k; j++) {
        if (i !== j && Math.abs(ImBinv[i][j]) > 1e-10) {
          totalEffects.get(allVars[j])!.set(allVars[i], ImBinv[i][j]);
        }
      }
    }
  }

  return { coefficients, directEffects, totalEffects, rSquared };
}

// ── Confirmatory Factor Analysis ──────────────────────────────────────────

/**
 * Confirmatory Factor Analysis.
 *
 * Estimates factor loadings from the observed correlation matrix using
 * iterative principal axis factoring constrained to the specified structure.
 * The model assumes: Sigma = Lambda * Lambda' + Psi where Lambda is the
 * loadings matrix and Psi is the diagonal uniqueness matrix.
 *
 * @param data - Map from indicator name to numeric array (all same length)
 * @param loadingSpec - Factor-indicator loading specifications defining which indicators load on which factors
 * @param options - Configuration options
 * @param options.maxIterations - Maximum iterations for convergence (default 100)
 * @returns CFAResult with loadings, communalities, uniquenesses, and fit indices
 * @throws Error if any specified indicator is not found in data
 *
 * @example
 * ```ts
 * const data = { x1: [1,2,3,4], x2: [2,3,4,5], x3: [1,3,5,7] };
 * const spec = [
 *   { factor: "F1", indicator: "x1" },
 *   { factor: "F1", indicator: "x2" },
 *   { factor: "F1", indicator: "x3" },
 * ];
 * const result = cfa(data, spec);
 * // result.loadings — Map of factor => Map of indicator => loading
 * // result.fit — model fit indices (chi-squared, RMSEA, CFI, etc.)
 * ```
 */
export function cfa(
  data: Record<string, number[]>,
  loadingSpec: CFALoading[],
  options: { maxIterations?: number } = {},
): CFAResult {
  const { maxIterations = 100 } = options;
  const indicators = [...new Set(loadingSpec.map((l) => l.indicator))];
  const factors = [...new Set(loadingSpec.map((l) => l.factor))];
  const nInd = indicators.length;
  const nFac = factors.length;
  const indIndex = new Map<string, number>();
  indicators.forEach((v, i) => indIndex.set(v, i));
  const facIndex = new Map<string, number>();
  factors.forEach((f, i) => facIndex.set(f, i));

  // Compute observed correlation matrix
  for (const ind of indicators) {
    if (!data[ind]) throw new Error(`Indicator "${ind}" not found in data`);
  }
  const n = data[indicators[0]].length;
  const stdData: number[][] = [];
  for (const ind of indicators) {
    stdData.push(standardize(data[ind]).values);
  }

  const R: number[][] = Array.from({ length: nInd }, () => new Array(nInd).fill(0));
  for (let i = 0; i < nInd; i++) {
    for (let j = 0; j <= i; j++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += stdData[i][k] * stdData[j][k];
      R[i][j] = s / (n - 1);
      R[j][i] = R[i][j];
    }
  }

  // Build loading pattern matrix (which factor loads on which indicator)
  const pattern: boolean[][] = Array.from({ length: nInd }, () =>
    new Array(nFac).fill(false),
  );
  for (const spec of loadingSpec) {
    const i = indIndex.get(spec.indicator)!;
    const f = facIndex.get(spec.factor)!;
    pattern[i][f] = true;
  }

  // Iterative estimation of loadings via reduced correlation
  const Lambda: number[][] = Array.from({ length: nInd }, () =>
    new Array(nFac).fill(0),
  );

  // Initialize loadings from correlations with first indicator of each factor
  for (let f = 0; f < nFac; f++) {
    const factorInds = indicators
      .map((ind, i) => ({ ind, i }))
      .filter(({ i }) => pattern[i][f]);
    if (factorInds.length === 0) continue;
    const refIdx = factorInds[0].i;
    for (const { i } of factorInds) {
      Lambda[i][f] = i === refIdx ? 0.7 : R[i][refIdx] * 0.8;
    }
  }

  // Iterate: update communalities, re-extract loadings
  for (let iter = 0; iter < maxIterations; iter++) {
    const oldLambda = Lambda.map((row) => [...row]);

    // Compute model-implied correlation: Σ = ΛΛ' + Ψ
    // Update communalities (diagonal of ΛΛ')
    const communalities = new Array(nInd).fill(0);
    for (let i = 0; i < nInd; i++) {
      for (let f = 0; f < nFac; f++) {
        communalities[i] += Lambda[i][f] ** 2;
      }
    }

    // Reduced correlation: R - Ψ (replace diagonal with communalities)
    const Rred: number[][] = R.map((row) => [...row]);
    for (let i = 0; i < nInd; i++) {
      Rred[i][i] = Math.min(communalities[i], 0.999);
    }

    // For each factor, extract loadings from reduced R restricted to its indicators
    for (let f = 0; f < nFac; f++) {
      const fInds = indicators
        .map((_, i) => i)
        .filter((i) => pattern[i][f]);
      if (fInds.length < 2) {
        if (fInds.length === 1) Lambda[fInds[0]][f] = Math.sqrt(Math.max(0, communalities[fInds[0]]));
        continue;
      }

      // Extract sub-matrix
      const subR: number[][] = fInds.map((i) =>
        fInds.map((j) => Rred[i][j]),
      );

      // First eigenvector of sub-matrix (power iteration)
      const ev = firstEigenvector(subR);
      const eigenvalue = eigenvalueEstimate(subR, ev);

      for (let k = 0; k < fInds.length; k++) {
        Lambda[fInds[k]][f] = ev[k] * Math.sqrt(Math.max(0, eigenvalue));
      }
    }

    // Check convergence
    let maxDiff = 0;
    for (let i = 0; i < nInd; i++) {
      for (let f = 0; f < nFac; f++) {
        maxDiff = Math.max(maxDiff, Math.abs(Lambda[i][f] - oldLambda[i][f]));
      }
    }
    if (maxDiff < 1e-6) break;
  }

  // Apply fixed loadings
  for (const spec of loadingSpec) {
    if (spec.loading !== undefined) {
      Lambda[indIndex.get(spec.indicator)!][facIndex.get(spec.factor)!] = spec.loading;
    }
  }

  // Build results
  const loadings = new Map<string, Map<string, number>>();
  for (let f = 0; f < nFac; f++) {
    const facMap = new Map<string, number>();
    for (let i = 0; i < nInd; i++) {
      if (pattern[i][f]) {
        facMap.set(indicators[i], Lambda[i][f]);
      }
    }
    loadings.set(factors[f], facMap);
  }

  const communalityMap = new Map<string, number>();
  const uniquenessMap = new Map<string, number>();
  for (let i = 0; i < nInd; i++) {
    let h2 = 0;
    for (let f = 0; f < nFac; f++) h2 += Lambda[i][f] ** 2;
    h2 = Math.min(h2, 1);
    communalityMap.set(indicators[i], h2);
    uniquenessMap.set(indicators[i], 1 - h2);
  }

  // Model-implied covariance
  const sigmaModel: number[][] = Array.from({ length: nInd }, () =>
    new Array(nInd).fill(0),
  );
  for (let i = 0; i < nInd; i++) {
    for (let j = 0; j < nInd; j++) {
      for (let f = 0; f < nFac; f++) {
        sigmaModel[i][j] += Lambda[i][f] * Lambda[j][f];
      }
    }
    sigmaModel[i][i] += uniquenessMap.get(indicators[i])!;
  }

  const fit = computeFitIndices(R, sigmaModel, n, nInd, nFac, loadingSpec.length);

  return {
    loadings,
    communalities: communalityMap,
    uniquenesses: uniquenessMap,
    fit,
  };
}

// ── Fit Indices ───────────────────────────────────────────────────────────

/**
 * Compute SEM fit indices from observed and model-implied covariance matrices.
 *
 * Computes the maximum likelihood discrepancy function
 * F_ML = tr(S * Sigma^{-1}) + ln|Sigma| - ln|S| - p, then derives
 * chi-squared = (n-1) * F_ML. Also computes RMSEA, CFI, TLI, SRMR, AIC, and BIC.
 *
 * @param observed - Observed covariance (or correlation) matrix (p x p)
 * @param implied - Model-implied covariance matrix (p x p)
 * @param n - Number of observations
 * @param nIndicators - Number of observed indicators (p)
 * @param nFactors - Number of latent factors
 * @param nEstimatedParams - Number of freely estimated parameters
 * @returns SEMFitIndices with chi-squared, df, p-value, RMSEA, CFI, TLI, SRMR, AIC, and BIC
 */
export function computeFitIndices(
  observed: number[][],
  implied: number[][],
  n: number,
  nIndicators: number,
  nFactors: number,
  nEstimatedParams: number,
): SEMFitIndices {
  const p = nIndicators;

  // SRMR: standardized root mean square residual
  let srmrSum = 0;
  let srmrCount = 0;
  for (let i = 0; i < p; i++) {
    for (let j = 0; j <= i; j++) {
      const residual = observed[i][j] - implied[i][j];
      const denom = Math.sqrt(observed[i][i] * observed[j][j]);
      if (denom > 0) {
        srmrSum += (residual / denom) ** 2;
        srmrCount++;
      }
    }
  }
  const srmr = Math.sqrt(srmrSum / Math.max(1, srmrCount));

  // Chi-squared (ML discrepancy)
  const impliedInv = invertMatrix(implied);
  let fML = 0;
  if (impliedInv) {
    // F = tr(S Σ⁻¹) - ln|S Σ⁻¹| - p
    let trSinvSigma = 0;
    for (let i = 0; i < p; i++) {
      for (let j = 0; j < p; j++) {
        trSinvSigma += observed[i][j] * impliedInv[i][j];
      }
    }

    let logDetObs = 0;
    let logDetImpl = 0;
    // Approximate log determinants from diagonal (for positive-definite)
    for (let i = 0; i < p; i++) {
      logDetObs += Math.log(Math.max(1e-10, observed[i][i]));
      logDetImpl += Math.log(Math.max(1e-10, implied[i][i]));
    }

    fML = Math.max(0, trSinvSigma + logDetImpl - logDetObs - p);
  }

  const chiSquared = Math.max(0, (n - 1) * fML);
  const dfModel = (p * (p + 1)) / 2 - nEstimatedParams;
  const df = Math.max(1, dfModel);

  const pValue = 1 - chiSquaredCdf(chiSquared, df);

  // RMSEA
  const rmsea = Math.sqrt(Math.max(0, (chiSquared / df - 1) / (n - 1)));

  // Null model chi-squared (independence model)
  let chiSqNull = 0;
  for (let i = 0; i < p; i++) {
    for (let j = 0; j < p; j++) {
      if (i !== j) chiSqNull += observed[i][j] ** 2;
    }
  }
  chiSqNull *= (n - 1);
  const dfNull = (p * (p - 1)) / 2;

  // CFI
  const cfi = dfNull > 0
    ? Math.min(1, Math.max(0, 1 - (chiSquared - df) / (chiSqNull - dfNull)))
    : 1;

  // TLI
  const tli = dfNull > 0 && df > 0
    ? Math.min(1, Math.max(0, (chiSqNull / dfNull - chiSquared / df) / (chiSqNull / dfNull - 1)))
    : 1;

  // AIC and BIC
  const ll = -0.5 * (n - 1) * fML;
  const aic = -2 * ll + 2 * nEstimatedParams;
  const bic = -2 * ll + nEstimatedParams * Math.log(n);

  return { chiSquared, df, pValue, rmsea, cfi, tli, srmr, aic, bic };
}

// ── Helpers ───────────────────────────────────────────────────────────────

function standardize(data: number[]): { values: number[]; m: number; s: number } {
  const n = data.length;
  const m = mean(data);
  let ss = 0;
  for (const x of data) ss += (x - m) ** 2;
  const s = Math.sqrt(ss / (n - 1));
  if (s === 0) return { values: new Array(n).fill(0), m, s: 1 };
  return { values: data.map((x) => (x - m) / s), m, s };
}

function firstEigenvector(A: number[][]): number[] {
  const n = A.length;
  let v = new Array(n).fill(1 / Math.sqrt(n));
  for (let iter = 0; iter < 100; iter++) {
    const vNew = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) vNew[i] += A[i][j] * v[j];
    }
    let norm = 0;
    for (let i = 0; i < n; i++) norm += vNew[i] * vNew[i];
    norm = Math.sqrt(norm);
    if (norm === 0) break;
    let maxDiff = 0;
    for (let i = 0; i < n; i++) {
      vNew[i] /= norm;
      maxDiff = Math.max(maxDiff, Math.abs(vNew[i] - v[i]));
    }
    v = vNew;
    if (maxDiff < 1e-10) break;
  }
  return v;
}

function eigenvalueEstimate(A: number[][], v: number[]): number {
  const n = v.length;
  const Av = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) Av[i] += A[i][j] * v[j];
  }
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) { num += v[i] * Av[i]; den += v[i] * v[i]; }
  return den > 0 ? num / den : 0;
}

function chiSquaredCdf(x: number, k: number): number {
  if (x <= 0) return 0;
  return regularizedGammaLower(k / 2, x / 2);
}

function regularizedGammaLower(a: number, x: number): number {
  if (x <= 0) return 0;
  let sum = 0;
  let term = 1 / a;
  for (let n = 0; n < 200; n++) {
    sum += term;
    term *= x / (a + n + 1);
    if (Math.abs(term) < 1e-14) break;
  }
  return sum * Math.exp(-x + a * Math.log(x) - gammaLnLocal(a));
}

function gammaLnLocal(x: number): number {
  const c = [
    76.18009172947146, -86.50532032941677, 24.01409824083091,
    -1.231739572450155, 0.001208650973866179, -0.000005395239384953,
  ];
  let y = x;
  let tmp = x + 5.5;
  tmp -= (x + 0.5) * Math.log(tmp);
  let ser = 1.000000000190015;
  for (let j = 0; j < 6; j++) ser += c[j] / ++y;
  return -tmp + Math.log((2.5066282746310005 * ser) / x);
}
