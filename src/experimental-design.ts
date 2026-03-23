/**
 * Experimental design and analysis.
 *
 * - **Factorial designs** — full factorial generation, main effects, interactions.
 * - **Latin square** — construction and analysis.
 * - **Response surface methodology** — central composite design, quadratic model fitting.
 * - **Randomised block design** — block-adjusted ANOVA.
 * - **Power/sample size** — minimum sample size for factorial experiments.
 */

import { solveLinearSystem } from "./utils/linalg";
import { mean } from "./utils/descriptive";

// ── Types ─────────────────────────────────────────────────────────────────

export interface FactorialDesign {
  /** Factor names. */
  factors: string[];
  /** Levels per factor. */
  levels: number[][];
  /** Design matrix: each row is a treatment combination. */
  runs: number[][];
  /** Number of runs. */
  nRuns: number;
}

export interface FactorialAnalysisResult {
  /** Main effects: factor name → effect size. */
  mainEffects: Map<string, number>;
  /** Two-way interactions: "A:B" → effect size. */
  interactions: Map<string, number>;
  /** Sum of squares for each term. */
  sumOfSquares: Map<string, number>;
  /** F-statistics for each term. */
  fStatistics: Map<string, number>;
  /** Residual sum of squares. */
  residualSS: number;
  /** Residual degrees of freedom. */
  residualDF: number;
  /** R². */
  rSquared: number;
}

export interface LatinSquare {
  /** n × n square of treatment assignments (0-indexed). */
  square: number[][];
  /** Order n. */
  order: number;
}

export interface LatinSquareAnalysisResult {
  /** Row effect sum of squares. */
  ssRows: number;
  /** Column effect sum of squares. */
  ssCols: number;
  /** Treatment effect sum of squares. */
  ssTreatment: number;
  /** Residual sum of squares. */
  ssResidual: number;
  /** F-statistic for treatment effects. */
  fTreatment: number;
  /** Treatment means. */
  treatmentMeans: number[];
  /** Grand mean. */
  grandMean: number;
}

export interface ResponseSurfaceDesign {
  /** Design points (rows = runs, cols = factors). */
  runs: number[][];
  /** Factor names. */
  factors: string[];
  /** Design type. */
  type: "central-composite" | "box-behnken";
  /** Number of runs. */
  nRuns: number;
}

export interface ResponseSurfaceResult {
  /** Quadratic model coefficients: [intercept, linear..., interaction..., quadratic...]. */
  coefficients: number[];
  /** Coefficient names for reference. */
  coefficientNames: string[];
  /** R². */
  rSquared: number;
  /** Adjusted R². */
  adjRSquared: number;
  /** Predicted response at a given point. */
  predict: (x: number[]) => number;
  /** Estimated stationary point (if it exists within design space). */
  stationaryPoint?: number[];
}

// ── Full Factorial Design ─────────────────────────────────────────────────

/**
 * Generate a full factorial design.
 *
 * @param factors  Map of factor name → array of levels.
 */
export function fullFactorial(
  factors: Record<string, number[]>,
): FactorialDesign {
  const names = Object.keys(factors);
  const levels = names.map((f) => factors[f]);
  const nFactors = names.length;

  // Generate all combinations
  const runs: number[][] = [[]];
  for (let f = 0; f < nFactors; f++) {
    const newRuns: number[][] = [];
    for (const run of runs) {
      for (const level of levels[f]) {
        newRuns.push([...run, level]);
      }
    }
    runs.length = 0;
    runs.push(...newRuns);
  }

  return {
    factors: names,
    levels,
    runs,
    nRuns: runs.length,
  };
}

/**
 * Analyse a factorial experiment.
 *
 * Computes main effects, two-way interactions, and ANOVA-like F-tests.
 *
 * @param design  The factorial design.
 * @param response  Observed response for each run (length = nRuns × replicates).
 * @param replicates  Number of replicates per treatment (default 1).
 */
export function analyzeFactorial(
  design: FactorialDesign,
  response: number[],
  replicates = 1,
): FactorialAnalysisResult {
  const { factors, runs, nRuns } = design;
  const nObs = nRuns * replicates;
  if (response.length !== nObs) {
    throw new Error(`Expected ${nObs} observations, got ${response.length}`);
  }

  const grandMean = mean(response);
  let ssTot = 0;
  for (const y of response) ssTot += (y - grandMean) ** 2;

  const mainEffects = new Map<string, number>();
  const interactions = new Map<string, number>();
  const sumOfSquares = new Map<string, number>();
  const fStatistics = new Map<string, number>();
  const nFactors = factors.length;

  // Compute treatment means
  const treatmentMeans = new Map<string, number>();
  for (let r = 0; r < nRuns; r++) {
    let sum = 0;
    for (let rep = 0; rep < replicates; rep++) {
      sum += response[r * replicates + rep];
    }
    treatmentMeans.set(runs[r].join(","), sum / replicates);
  }

  // Main effects
  for (let f = 0; f < nFactors; f++) {
    const uniqueLevels = [...new Set(runs.map((r) => r[f]))];
    const levelMeans: number[] = [];
    let ss = 0;

    for (const level of uniqueLevels) {
      const matchingRuns = runs
        .map((r, i) => ({ r, i }))
        .filter(({ r }) => r[f] === level);
      let sum = 0;
      let count = 0;
      for (const { i } of matchingRuns) {
        for (let rep = 0; rep < replicates; rep++) {
          sum += response[i * replicates + rep];
          count++;
        }
      }
      const levelMean = sum / count;
      levelMeans.push(levelMean);
      ss += count * (levelMean - grandMean) ** 2;
    }

    const effect = levelMeans.length >= 2
      ? levelMeans[levelMeans.length - 1] - levelMeans[0]
      : 0;
    mainEffects.set(factors[f], effect);
    sumOfSquares.set(factors[f], ss);
  }

  // Two-way interactions
  for (let f1 = 0; f1 < nFactors; f1++) {
    for (let f2 = f1 + 1; f2 < nFactors; f2++) {
      const key = `${factors[f1]}:${factors[f2]}`;
      const levels1 = [...new Set(runs.map((r) => r[f1]))];
      const levels2 = [...new Set(runs.map((r) => r[f2]))];

      let ssInteraction = 0;
      for (const l1 of levels1) {
        for (const l2 of levels2) {
          const matching = runs
            .map((r, i) => ({ r, i }))
            .filter(({ r }) => r[f1] === l1 && r[f2] === l2);
          if (matching.length === 0) continue;
          let sum = 0;
          let count = 0;
          for (const { i } of matching) {
            for (let rep = 0; rep < replicates; rep++) {
              sum += response[i * replicates + rep];
              count++;
            }
          }
          const cellMean = sum / count;
          ssInteraction += count * (cellMean - grandMean) ** 2;
        }
      }

      // Subtract main effects
      const ssMain1 = sumOfSquares.get(factors[f1]) ?? 0;
      const ssMain2 = sumOfSquares.get(factors[f2]) ?? 0;
      ssInteraction = Math.max(0, ssInteraction - ssMain1 - ssMain2);

      interactions.set(key, ssInteraction / Math.max(1, nObs));
      sumOfSquares.set(key, ssInteraction);
    }
  }

  // Residual
  let ssModel = 0;
  for (const [, ss] of sumOfSquares) ssModel += ss;
  const residualSS = Math.max(0, ssTot - ssModel);
  const modelDF = sumOfSquares.size;
  const residualDF = Math.max(1, nObs - modelDF - 1);
  const msResid = residualSS / residualDF;

  // F-statistics
  for (const [term, ss] of sumOfSquares) {
    const dfTerm = term.includes(":") ? 1 : (new Set(runs.map((r) => r[factors.indexOf(term.split(":")[0])])).size - 1) || 1;
    fStatistics.set(term, msResid > 0 ? (ss / dfTerm) / msResid : 0);
  }

  const rSquared = ssTot > 0 ? 1 - residualSS / ssTot : 0;

  return {
    mainEffects,
    interactions,
    sumOfSquares,
    fStatistics,
    residualSS,
    residualDF,
    rSquared,
  };
}

// ── Latin Square ──────────────────────────────────────────────────────────

/**
 * Generate a standard Latin square of order n.
 *
 * Treatments are labeled 0..n−1. Row i, col j gets treatment (i+j) mod n.
 */
export function latinSquare(n: number): LatinSquare {
  if (n < 2) throw new Error("Order must be at least 2");
  const square: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < n; j++) {
      row.push((i + j) % n);
    }
    square.push(row);
  }
  return { square, order: n };
}

/**
 * Analyse a Latin square experiment.
 *
 * @param square  The Latin square layout.
 * @param response  Observed values: response[i][j] for row i, col j.
 */
export function analyzeLatinSquare(
  square: LatinSquare,
  response: number[][],
): LatinSquareAnalysisResult {
  const n = square.order;
  if (response.length !== n || response.some((r) => r.length !== n)) {
    throw new Error(`Response must be ${n} × ${n}`);
  }

  const N = n * n;
  let grandSum = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) grandSum += response[i][j];
  }
  const grandMean = grandSum / N;

  // Row means
  const rowMeans: number[] = [];
  for (let i = 0; i < n; i++) {
    rowMeans.push(mean(response[i]));
  }

  // Column means
  const colMeans: number[] = [];
  for (let j = 0; j < n; j++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += response[i][j];
    colMeans.push(s / n);
  }

  // Treatment means
  const treatmentSums = new Array(n).fill(0);
  const treatmentCounts = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const t = square.square[i][j];
      treatmentSums[t] += response[i][j];
      treatmentCounts[t]++;
    }
  }
  const treatmentMeans = treatmentSums.map((s, i) => s / treatmentCounts[i]);

  // Sum of squares
  let ssRows = 0;
  for (let i = 0; i < n; i++) ssRows += n * (rowMeans[i] - grandMean) ** 2;

  let ssCols = 0;
  for (let j = 0; j < n; j++) ssCols += n * (colMeans[j] - grandMean) ** 2;

  let ssTreatment = 0;
  for (let t = 0; t < n; t++) ssTreatment += n * (treatmentMeans[t] - grandMean) ** 2;

  let ssTot = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) ssTot += (response[i][j] - grandMean) ** 2;
  }

  const ssResidual = Math.max(0, ssTot - ssRows - ssCols - ssTreatment);
  const dfResidual = Math.max(1, (n - 1) * (n - 2));
  const msResidual = ssResidual / dfResidual;
  const fTreatment = msResidual > 0 ? (ssTreatment / (n - 1)) / msResidual : 0;

  return {
    ssRows,
    ssCols,
    ssTreatment,
    ssResidual,
    fTreatment,
    treatmentMeans,
    grandMean,
  };
}

// ── Response Surface Methodology ──────────────────────────────────────────

/**
 * Generate a Central Composite Design (CCD).
 *
 * Includes:  2^k factorial points, 2k axial (star) points, n₀ centre points.
 *
 * @param k  Number of factors (2 to 5).
 * @param alpha  Axial distance (default √k for rotatability).
 * @param nCenter  Number of centre points (default 3).
 */
export function centralCompositeDesign(
  k: number,
  alpha?: number,
  nCenter = 3,
): ResponseSurfaceDesign {
  if (k < 2 || k > 5) throw new Error("k must be between 2 and 5");
  const a = alpha ?? Math.sqrt(k);

  const runs: number[][] = [];
  const factors = Array.from({ length: k }, (_, i) => `x${i + 1}`);

  // Factorial points (2^k)
  const nFact = 1 << k;
  for (let i = 0; i < nFact; i++) {
    const point: number[] = [];
    for (let j = 0; j < k; j++) {
      point.push((i >> j) & 1 ? 1 : -1);
    }
    runs.push(point);
  }

  // Axial (star) points (2k)
  for (let j = 0; j < k; j++) {
    const plus = new Array(k).fill(0);
    const minus = new Array(k).fill(0);
    plus[j] = a;
    minus[j] = -a;
    runs.push(plus);
    runs.push(minus);
  }

  // Centre points
  for (let i = 0; i < nCenter; i++) {
    runs.push(new Array(k).fill(0));
  }

  return { runs, factors, type: "central-composite", nRuns: runs.length };
}

/**
 * Fit a full quadratic response surface model.
 *
 * y = β₀ + Σ βᵢ xᵢ + Σ βᵢⱼ xᵢxⱼ + Σ βᵢᵢ xᵢ² + ε
 *
 * @param design  Response surface design (or any numeric matrix of runs).
 * @param response  Observed responses (length = nRuns).
 */
export function fitResponseSurface(
  design: ResponseSurfaceDesign | { runs: number[][] },
  response: number[],
): ResponseSurfaceResult {
  const runs = design.runs;
  const n = runs.length;
  const k = runs[0].length;
  if (response.length !== n) {
    throw new Error(`Expected ${n} responses, got ${response.length}`);
  }

  const factors = "factors" in design
    ? design.factors
    : Array.from({ length: k }, (_, i) => `x${i + 1}`);

  // Build model matrix: [1, x₁, ..., xₖ, x₁x₂, ..., x₁², ..., xₖ²]
  const names: string[] = ["intercept"];
  for (let i = 0; i < k; i++) names.push(factors[i]);
  for (let i = 0; i < k; i++) {
    for (let j = i + 1; j < k; j++) {
      names.push(`${factors[i]}:${factors[j]}`);
    }
  }
  for (let i = 0; i < k; i++) names.push(`${factors[i]}^2`);

  const p = names.length;
  const X: number[][] = [];

  for (let r = 0; r < n; r++) {
    const row = [1];
    for (let i = 0; i < k; i++) row.push(runs[r][i]);
    for (let i = 0; i < k; i++) {
      for (let j = i + 1; j < k; j++) {
        row.push(runs[r][i] * runs[r][j]);
      }
    }
    for (let i = 0; i < k; i++) row.push(runs[r][i] ** 2);
    X.push(row);
  }

  // OLS: β = (X'X)⁻¹ X'y
  const XtX: number[][] = Array.from({ length: p }, () => new Array(p).fill(0));
  const Xty = new Array(p).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < p; j++) {
      Xty[j] += X[i][j] * response[i];
      for (let l = 0; l < p; l++) {
        XtX[j][l] += X[i][j] * X[i][l];
      }
    }
  }

  const coefficients = solveLinearSystem(XtX, Xty);

  // R²
  const yMean = mean(response);
  let ssTot = 0;
  let ssRes = 0;
  for (let i = 0; i < n; i++) {
    let yHat = 0;
    for (let j = 0; j < p; j++) yHat += X[i][j] * coefficients[j];
    ssRes += (response[i] - yHat) ** 2;
    ssTot += (response[i] - yMean) ** 2;
  }
  const rSquared = ssTot > 0 ? 1 - ssRes / ssTot : 0;
  const adjRSquared = 1 - (1 - rSquared) * (n - 1) / Math.max(1, n - p);

  // Stationary point: solve ∂ŷ/∂x = 0
  // For quadratic model: b + Bx = 0 where b = linear coeffs, B = matrix of quadratic/interaction
  let stationaryPoint: number[] | undefined;
  try {
    const b = coefficients.slice(1, 1 + k);
    const B: number[][] = Array.from({ length: k }, () => new Array(k).fill(0));
    // Quadratic terms on diagonal
    for (let i = 0; i < k; i++) {
      B[i][i] = 2 * coefficients[1 + k + (k * (k - 1)) / 2 + i]; // index of x_i^2
    }
    // Interaction terms
    let idx = 1 + k;
    for (let i = 0; i < k; i++) {
      for (let j = i + 1; j < k; j++) {
        B[i][j] = coefficients[idx] / 2;
        B[j][i] = coefficients[idx] / 2;
        idx++;
      }
    }
    const negB = b.map((v) => -v);
    stationaryPoint = solveLinearSystem(B, negB);
  } catch {
    stationaryPoint = undefined;
  }

  const predict = (x: number[]) => {
    let val = coefficients[0];
    for (let i = 0; i < k; i++) val += coefficients[1 + i] * x[i];
    let idx = 1 + k;
    for (let i = 0; i < k; i++) {
      for (let j = i + 1; j < k; j++) {
        val += coefficients[idx++] * x[i] * x[j];
      }
    }
    for (let i = 0; i < k; i++) {
      val += coefficients[1 + k + (k * (k - 1)) / 2 + i] * x[i] ** 2;
    }
    return val;
  };

  return {
    coefficients,
    coefficientNames: names,
    rSquared,
    adjRSquared,
    predict,
    stationaryPoint,
  };
}

// ── Randomised Complete Block Design ──────────────────────────────────────

export interface RCBDResult {
  /** Treatment sum of squares. */
  ssTreatment: number;
  /** Block sum of squares. */
  ssBlock: number;
  /** Residual (error) sum of squares. */
  ssResidual: number;
  /** F-statistic for treatment effects. */
  fTreatment: number;
  /** Treatment means. */
  treatmentMeans: number[];
  /** Block means. */
  blockMeans: number[];
  /** Grand mean. */
  grandMean: number;
  /** Residual degrees of freedom. */
  residualDF: number;
}

/**
 * Analyse a Randomised Complete Block Design.
 *
 * @param response  Response matrix: response[block][treatment] (b × t).
 */
export function analyzeRCBD(response: number[][]): RCBDResult {
  const nBlocks = response.length;
  if (nBlocks < 2) throw new Error("Need at least 2 blocks");
  const nTreatments = response[0].length;
  if (nTreatments < 2) throw new Error("Need at least 2 treatments");
  for (const row of response) {
    if (row.length !== nTreatments) throw new Error("All blocks must have the same number of treatments");
  }

  const N = nBlocks * nTreatments;
  let grandSum = 0;
  for (let i = 0; i < nBlocks; i++) {
    for (let j = 0; j < nTreatments; j++) grandSum += response[i][j];
  }
  const grandMean = grandSum / N;

  // Treatment means
  const treatmentMeans: number[] = [];
  for (let j = 0; j < nTreatments; j++) {
    let s = 0;
    for (let i = 0; i < nBlocks; i++) s += response[i][j];
    treatmentMeans.push(s / nBlocks);
  }

  // Block means
  const blockMeans = response.map((row) => mean(row));

  // Sum of squares
  let ssTreatment = 0;
  for (let j = 0; j < nTreatments; j++) {
    ssTreatment += nBlocks * (treatmentMeans[j] - grandMean) ** 2;
  }

  let ssBlock = 0;
  for (let i = 0; i < nBlocks; i++) {
    ssBlock += nTreatments * (blockMeans[i] - grandMean) ** 2;
  }

  let ssTot = 0;
  for (let i = 0; i < nBlocks; i++) {
    for (let j = 0; j < nTreatments; j++) {
      ssTot += (response[i][j] - grandMean) ** 2;
    }
  }

  const ssResidual = Math.max(0, ssTot - ssTreatment - ssBlock);
  const residualDF = (nBlocks - 1) * (nTreatments - 1);
  const msResidual = residualDF > 0 ? ssResidual / residualDF : 0;
  const fTreatment = msResidual > 0 ? (ssTreatment / (nTreatments - 1)) / msResidual : 0;

  return {
    ssTreatment,
    ssBlock,
    ssResidual,
    fTreatment,
    treatmentMeans,
    blockMeans,
    grandMean,
    residualDF,
  };
}
