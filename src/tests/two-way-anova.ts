import { Dataset } from "../types";
import { mean } from "../utils/descriptive";
import { regularizedBeta } from "../utils/math";

/** Result of a two-way ANOVA. */
export interface TwoWayAnovaResult {
  /** Factor A main effect */
  factorA: {
    fStatistic: number;
    pValue: number;
    df: number;
    ss: number;
    ms: number;
    rejected: boolean;
  };
  /** Factor B main effect */
  factorB: {
    fStatistic: number;
    pValue: number;
    df: number;
    ss: number;
    ms: number;
    rejected: boolean;
  };
  /** A × B interaction effect */
  interaction: {
    fStatistic: number;
    pValue: number;
    df: number;
    ss: number;
    ms: number;
    rejected: boolean;
  };
  /** Residual (error) */
  residual: {
    df: number;
    ss: number;
    ms: number;
  };
  /** Total */
  total: {
    df: number;
    ss: number;
  };
}

/**
 * Upper-tail p-value for the F-distribution.
 */
function fPValue(f: number, d1: number, d2: number): number {
  if (f <= 0) return 1;
  const x = (d1 * f) / (d1 * f + d2);
  return 1 - regularizedBeta(x, d1 / 2, d2 / 2);
}

/**
 * Two-way ANOVA with interaction.
 *
 * Expects a balanced or unbalanced design specified as a flat array of observations
 * with their factor levels.
 *
 * @param data - Array of { a, b, value } where a and b are factor level indices
 * @param alpha - Significance level
 */
export function twoWayAnova(
  data: Array<{ a: number; b: number; value: number }>,
  alpha = 0.05,
): TwoWayAnovaResult {
  if (data.length < 4) {
    throw new Error("Need at least 4 observations");
  }

  // Identify factor levels
  const aLevels = [...new Set(data.map((d) => d.a))].sort((x, y) => x - y);
  const bLevels = [...new Set(data.map((d) => d.b))].sort((x, y) => x - y);
  const a = aLevels.length;
  const b = bLevels.length;

  if (a < 2) throw new Error("Factor A must have at least 2 levels");
  if (b < 2) throw new Error("Factor B must have at least 2 levels");

  // Build cell structure
  const aIndex = new Map(aLevels.map((v, i) => [v, i]));
  const bIndex = new Map(bLevels.map((v, i) => [v, i]));

  const cells: number[][][] = Array.from({ length: a }, () =>
    Array.from({ length: b }, () => []),
  );

  for (const d of data) {
    const ai = aIndex.get(d.a)!;
    const bi = bIndex.get(d.b)!;
    cells[ai][bi].push(d.value);
  }

  // Check each cell has observations
  for (let i = 0; i < a; i++) {
    for (let j = 0; j < b; j++) {
      if (cells[i][j].length === 0) {
        throw new Error(
          `Empty cell at factor A level ${aLevels[i]}, factor B level ${bLevels[j]}`,
        );
      }
    }
  }

  const N = data.length;
  const grandMean = data.reduce((s, d) => s + d.value, 0) / N;

  // Cell means, marginal means
  const cellMeans: number[][] = cells.map((row) =>
    row.map((cell) => mean(cell)),
  );
  const cellN: number[][] = cells.map((row) => row.map((cell) => cell.length));

  // Marginal means for factor A (weighted)
  const aMeans: number[] = [];
  const aN: number[] = [];
  for (let i = 0; i < a; i++) {
    let sum = 0;
    let count = 0;
    for (let j = 0; j < b; j++) {
      for (const v of cells[i][j]) {
        sum += v;
        count++;
      }
    }
    aMeans.push(sum / count);
    aN.push(count);
  }

  // Marginal means for factor B (weighted)
  const bMeans: number[] = [];
  const bN: number[] = [];
  for (let j = 0; j < b; j++) {
    let sum = 0;
    let count = 0;
    for (let i = 0; i < a; i++) {
      for (const v of cells[i][j]) {
        sum += v;
        count++;
      }
    }
    bMeans.push(sum / count);
    bN.push(count);
  }

  // Type III-like sums of squares (works well for balanced designs)
  // SS_A = sum_i n_i. * (mean_i. - grandMean)^2
  let ssA = 0;
  for (let i = 0; i < a; i++) {
    ssA += aN[i] * (aMeans[i] - grandMean) ** 2;
  }

  // SS_B = sum_j n_.j * (mean_.j - grandMean)^2
  let ssB = 0;
  for (let j = 0; j < b; j++) {
    ssB += bN[j] * (bMeans[j] - grandMean) ** 2;
  }

  // SS_Total = sum (y_ijk - grandMean)^2
  let ssTotal = 0;
  for (const d of data) {
    ssTotal += (d.value - grandMean) ** 2;
  }

  // SS_Within (error) = sum (y_ijk - cellMean_ij)^2
  let ssWithin = 0;
  for (let i = 0; i < a; i++) {
    for (let j = 0; j < b; j++) {
      for (const v of cells[i][j]) {
        ssWithin += (v - cellMeans[i][j]) ** 2;
      }
    }
  }

  // SS_Cells = sum n_ij * (cellMean_ij - grandMean)^2
  let ssCells = 0;
  for (let i = 0; i < a; i++) {
    for (let j = 0; j < b; j++) {
      ssCells += cellN[i][j] * (cellMeans[i][j] - grandMean) ** 2;
    }
  }

  // SS_AB = SS_Cells - SS_A - SS_B
  const ssAB = Math.max(0, ssCells - ssA - ssB);

  // Degrees of freedom
  const dfA = a - 1;
  const dfB = b - 1;
  const dfAB = dfA * dfB;
  const dfWithin = N - a * b;
  const dfTotal = N - 1;

  if (dfWithin <= 0) {
    throw new Error(
      "Insufficient degrees of freedom for error term. Need replication within cells.",
    );
  }

  // Mean squares
  const msA = ssA / dfA;
  const msB = ssB / dfB;
  const msAB = ssAB / dfAB;
  const msWithin = ssWithin / dfWithin;

  // F-statistics
  const fA = msA / msWithin;
  const fB = msB / msWithin;
  const fAB = msAB / msWithin;

  // P-values
  const pA = fPValue(fA, dfA, dfWithin);
  const pB = fPValue(fB, dfB, dfWithin);
  const pAB = fPValue(fAB, dfAB, dfWithin);

  return {
    factorA: {
      fStatistic: fA,
      pValue: pA,
      df: dfA,
      ss: ssA,
      ms: msA,
      rejected: pA < alpha,
    },
    factorB: {
      fStatistic: fB,
      pValue: pB,
      df: dfB,
      ss: ssB,
      ms: msB,
      rejected: pB < alpha,
    },
    interaction: {
      fStatistic: fAB,
      pValue: pAB,
      df: dfAB,
      ss: ssAB,
      ms: msAB,
      rejected: pAB < alpha,
    },
    residual: {
      df: dfWithin,
      ss: ssWithin,
      ms: msWithin,
    },
    total: {
      df: dfTotal,
      ss: ssTotal,
    },
  };
}
