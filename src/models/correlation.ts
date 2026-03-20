import { Dataset, CorrelationResult } from "../types";
import { mean } from "../utils/descriptive";
import { regularizedBeta } from "../utils/math";

/**
 * Pearson correlation coefficient with two-tailed p-value.
 *
 * Uses the t-distribution approximation: t = r * sqrt((n-2)/(1-r²))
 */
export function pearsonCorrelation(x: Dataset, y: Dataset): CorrelationResult {
  validateInputs(x, y);
  const n = x.length;
  const xMean = mean(x);
  const yMean = mean(y);

  let ssXY = 0;
  let ssXX = 0;
  let ssYY = 0;

  for (let i = 0; i < n; i++) {
    const dx = x[i] - xMean;
    const dy = y[i] - yMean;
    ssXY += dx * dy;
    ssXX += dx * dx;
    ssYY += dy * dy;
  }

  if (ssXX === 0 || ssYY === 0) {
    return { coefficient: NaN, pValue: NaN };
  }

  const r = ssXY / Math.sqrt(ssXX * ssYY);
  const pValue = correlationPValue(r, n);

  return { coefficient: r, pValue };
}

/**
 * Spearman rank correlation coefficient with two-tailed p-value.
 *
 * Computes ranks (with tie averaging) and then applies Pearson to the ranks.
 */
export function spearmanCorrelation(
  x: Dataset,
  y: Dataset,
): CorrelationResult {
  validateInputs(x, y);

  const rankX = computeRanks(x);
  const rankY = computeRanks(y);

  return pearsonCorrelation(rankX, rankY);
}

/**
 * Kendall's tau-b correlation coefficient with two-tailed p-value.
 *
 * Handles ties using the tau-b formula.
 */
export function kendallCorrelation(
  x: Dataset,
  y: Dataset,
): CorrelationResult {
  validateInputs(x, y);
  const n = x.length;

  let concordant = 0;
  let discordant = 0;
  let tiedX = 0;
  let tiedY = 0;

  for (let i = 0; i < n - 1; i++) {
    for (let j = i + 1; j < n; j++) {
      const dx = Math.sign(x[j] - x[i]);
      const dy = Math.sign(y[j] - y[i]);

      if (dx === 0 && dy === 0) {
        tiedX++;
        tiedY++;
      } else if (dx === 0) {
        tiedX++;
      } else if (dy === 0) {
        tiedY++;
      } else if (dx === dy) {
        concordant++;
      } else {
        discordant++;
      }
    }
  }

  const totalPairs = (n * (n - 1)) / 2;
  const denom = Math.sqrt(
    (totalPairs - tiedX) * (totalPairs - tiedY),
  );

  if (denom === 0) {
    return { coefficient: NaN, pValue: NaN };
  }

  const tau = (concordant - discordant) / denom;

  // Normal approximation for p-value (valid for n >= 10)
  const variance = (2 * (2 * n + 5)) / (9 * n * (n - 1));
  const z = tau / Math.sqrt(variance);
  const pValue = 2 * (1 - normalCDF(Math.abs(z)));

  return { coefficient: tau, pValue };
}

function validateInputs(x: Dataset, y: Dataset): void {
  if (x.length !== y.length) {
    throw new Error("Datasets must have the same length");
  }
  if (x.length < 3) {
    throw new Error("Datasets must have at least 3 elements");
  }
}

/** Compute ranks with tie averaging. */
function computeRanks(data: Dataset): number[] {
  const indexed = data.map((value, index) => ({ value, index }));
  indexed.sort((a, b) => a.value - b.value);

  const ranks = new Array(data.length);
  let i = 0;
  while (i < indexed.length) {
    let j = i;
    while (j < indexed.length && indexed[j].value === indexed[i].value) {
      j++;
    }
    // Average rank for tied values
    const avgRank = (i + j + 1) / 2; // 1-based average
    for (let k = i; k < j; k++) {
      ranks[indexed[k].index] = avgRank;
    }
    i = j;
  }

  return ranks;
}

/** Two-tailed p-value from Pearson r using the t-distribution via regularized beta. */
function correlationPValue(r: number, n: number): number {
  if (Math.abs(r) >= 1) return 0;
  const df = n - 2;
  const t2 = (r * r * df) / (1 - r * r);
  // P(T² > t²) = 1 - I_{df/(df+t²)}(df/2, 1/2) using regularized beta
  const x = df / (df + t2);
  return regularizedBeta(x, df / 2, 0.5);
}

/** Standard normal CDF approximation (used for Kendall p-value). */
function normalCDF(x: number): number {
  // Abramowitz and Stegun approximation 26.2.17
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;

  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x) / Math.SQRT2;

  const t = 1.0 / (1.0 + p * x);
  const y =
    1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

  return 0.5 * (1.0 + sign * y);
}
