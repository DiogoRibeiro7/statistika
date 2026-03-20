import { Dataset, EffectSizeResult, OddsRatioResult, AnovaResult } from "./types";
import { mean, variance } from "./utils/descriptive";
import { Normal } from "./distributions/continuous/normal";

/**
 * Cohen's d for two independent samples.
 * Uses the pooled standard deviation as the denominator.
 *
 * @param data1 - First sample
 * @param data2 - Second sample
 */
export function cohensD(data1: Dataset, data2: Dataset): EffectSizeResult {
  if (data1.length < 2 || data2.length < 2) {
    throw new Error("Both datasets must have at least 2 elements");
  }

  const n1 = data1.length;
  const n2 = data2.length;
  const m1 = mean(data1);
  const m2 = mean(data2);
  const v1 = variance(data1);
  const v2 = variance(data2);

  // Pooled standard deviation
  const sp = Math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2));
  const d = (m1 - m2) / sp;

  return {
    measure: "Cohen's d",
    value: d,
    interpretation: interpretCohensD(Math.abs(d)),
  };
}

/**
 * Glass's delta for two independent samples.
 * Uses the standard deviation of the control group as the denominator.
 *
 * @param treatment - Treatment group data
 * @param control - Control group data
 */
export function glassDelta(
  treatment: Dataset,
  control: Dataset,
): EffectSizeResult {
  if (treatment.length < 2 || control.length < 2) {
    throw new Error("Both datasets must have at least 2 elements");
  }

  const mt = mean(treatment);
  const mc = mean(control);
  const sc = Math.sqrt(variance(control));
  const delta = (mt - mc) / sc;

  return {
    measure: "Glass's delta",
    value: delta,
    interpretation: interpretCohensD(Math.abs(delta)),
  };
}

/**
 * Hedges' g — bias-corrected version of Cohen's d for small samples.
 *
 * @param data1 - First sample
 * @param data2 - Second sample
 */
export function hedgesG(data1: Dataset, data2: Dataset): EffectSizeResult {
  const d = cohensD(data1, data2);
  const n = data1.length + data2.length;

  // Approximate correction factor: 1 - 3/(4*df - 1)
  const df = n - 2;
  const correction = 1 - 3 / (4 * df - 1);
  const g = d.value * correction;

  return {
    measure: "Hedges' g",
    value: g,
    interpretation: interpretCohensD(Math.abs(g)),
  };
}

/**
 * Cohen's d for a paired / one-sample design.
 * Standardizes the mean difference by the standard deviation of the differences.
 *
 * @param data1 - First sample (or paired observations)
 * @param data2 - Second sample (paired with data1)
 */
export function pairedCohensD(
  data1: Dataset,
  data2: Dataset,
): EffectSizeResult {
  if (data1.length !== data2.length) {
    throw new Error("Paired datasets must have the same length");
  }
  if (data1.length < 2) {
    throw new Error("Datasets must have at least 2 elements");
  }

  const diffs = data1.map((v, i) => v - data2[i]);
  const md = mean(diffs);
  const sd = Math.sqrt(variance(diffs));
  const d = md / sd;

  return {
    measure: "Cohen's d (paired)",
    value: d,
    interpretation: interpretCohensD(Math.abs(d)),
  };
}

/**
 * Eta-squared (η²) from ANOVA results.
 * Proportion of total variance explained by the grouping variable.
 *
 * @param anovaResult - Result from oneWayAnova
 */
export function etaSquared(anovaResult: AnovaResult): EffectSizeResult {
  const eta2 =
    anovaResult.ssBetween / (anovaResult.ssBetween + anovaResult.ssWithin);

  return {
    measure: "Eta-squared",
    value: eta2,
    interpretation: interpretEtaSquared(eta2),
  };
}

/**
 * Partial eta-squared (η²p) from ANOVA results.
 * Same as eta-squared for one-way ANOVA but the standard partial form.
 *
 * @param anovaResult - Result from oneWayAnova
 */
export function partialEtaSquared(anovaResult: AnovaResult): EffectSizeResult {
  const eta2p =
    anovaResult.ssBetween / (anovaResult.ssBetween + anovaResult.ssWithin);

  return {
    measure: "Partial eta-squared",
    value: eta2p,
    interpretation: interpretEtaSquared(eta2p),
  };
}

/**
 * Omega-squared (ω²) from ANOVA results.
 * Less biased estimator of the population effect size than eta-squared.
 *
 * @param anovaResult - Result from oneWayAnova
 */
export function omegaSquared(anovaResult: AnovaResult): EffectSizeResult {
  const ssTotal = anovaResult.ssBetween + anovaResult.ssWithin;
  const omega2 =
    (anovaResult.ssBetween - anovaResult.dfBetween * anovaResult.msWithin) /
    (ssTotal + anovaResult.msWithin);

  return {
    measure: "Omega-squared",
    value: Math.max(0, omega2),
    interpretation: interpretEtaSquared(Math.max(0, omega2)),
  };
}

/**
 * Cohen's f from ANOVA results.
 * f = sqrt(eta² / (1 - eta²))
 *
 * @param anovaResult - Result from oneWayAnova
 */
export function cohensF(anovaResult: AnovaResult): EffectSizeResult {
  const eta2 =
    anovaResult.ssBetween / (anovaResult.ssBetween + anovaResult.ssWithin);
  const f = Math.sqrt(eta2 / (1 - eta2));

  return {
    measure: "Cohen's f",
    value: f,
    interpretation: interpretCohensF(f),
  };
}

/**
 * Cramér's V for a contingency table.
 * Measures the association between two categorical variables.
 *
 * @param table - Contingency table (2D array of counts)
 */
export function cramersV(table: number[][]): EffectSizeResult {
  const nRows = table.length;
  if (nRows < 2) throw new Error("Table must have at least 2 rows");
  const nCols = table[0].length;
  if (nCols < 2) throw new Error("Table must have at least 2 columns");

  const rowTotals = table.map((row) => row.reduce((a, b) => a + b, 0));
  const colTotals = new Array(nCols).fill(0);
  for (let j = 0; j < nCols; j++) {
    for (let i = 0; i < nRows; i++) {
      colTotals[j] += table[i][j];
    }
  }
  const n = rowTotals.reduce((a: number, b: number) => a + b, 0);
  if (n === 0) throw new Error("Table total must be positive");

  // Chi-squared statistic
  let chi2 = 0;
  for (let i = 0; i < nRows; i++) {
    for (let j = 0; j < nCols; j++) {
      const expected = (rowTotals[i] * colTotals[j]) / n;
      if (expected > 0) {
        chi2 += (table[i][j] - expected) ** 2 / expected;
      }
    }
  }

  const k = Math.min(nRows, nCols);
  const v = Math.sqrt(chi2 / (n * (k - 1)));

  return {
    measure: "Cramér's V",
    value: v,
    interpretation: interpretCramersV(v, k),
  };
}

/**
 * Phi coefficient (φ) for a 2×2 contingency table.
 * Special case of Cramér's V for 2×2 tables.
 *
 * @param table - 2×2 contingency table
 */
export function phiCoefficient(table: number[][]): EffectSizeResult {
  if (table.length !== 2 || table[0].length !== 2 || table[1].length !== 2) {
    throw new Error("Phi coefficient requires a 2×2 table");
  }

  const a = table[0][0];
  const b = table[0][1];
  const c = table[1][0];
  const d = table[1][1];

  const denom = Math.sqrt((a + b) * (c + d) * (a + c) * (b + d));
  if (denom === 0) {
    return { measure: "Phi coefficient", value: 0, interpretation: "negligible" };
  }

  const phi = (a * d - b * c) / denom;

  return {
    measure: "Phi coefficient",
    value: phi,
    interpretation: interpretCohensD(Math.abs(phi)),
  };
}

/**
 * Odds ratio for a 2×2 contingency table.
 * With Woolf logit confidence interval.
 *
 * Table layout:
 *   [[a, b], [c, d]]
 * where a=exposed+outcome, b=exposed+no_outcome,
 *       c=unexposed+outcome, d=unexposed+no_outcome
 *
 * @param table - 2×2 contingency table
 * @param confidence - Confidence level for the interval (default 0.95)
 */
export function oddsRatio(
  table: number[][],
  confidence = 0.95,
): OddsRatioResult {
  if (table.length !== 2 || table[0].length !== 2 || table[1].length !== 2) {
    throw new Error("Odds ratio requires a 2×2 table");
  }
  if (confidence <= 0 || confidence >= 1) {
    throw new Error("Confidence level must be between 0 and 1 (exclusive)");
  }

  const a = table[0][0];
  const b = table[0][1];
  const c = table[1][0];
  const d = table[1][1];

  if (a < 0 || b < 0 || c < 0 || d < 0) {
    throw new Error("Cell counts must be non-negative");
  }

  // Handle zero cells with Haldane-Anscombe correction (+0.5)
  const useCorrection = a === 0 || b === 0 || c === 0 || d === 0;
  const aa = useCorrection ? a + 0.5 : a;
  const bb = useCorrection ? b + 0.5 : b;
  const cc = useCorrection ? c + 0.5 : c;
  const dd = useCorrection ? d + 0.5 : d;

  const or = (aa * dd) / (bb * cc);
  const logOR = Math.log(or);

  // Standard error of log odds ratio (Woolf)
  const se = Math.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd);

  const z = new Normal().quantile(1 - (1 - confidence) / 2);

  return {
    oddsRatio: or,
    logOddsRatio: logOR,
    standardError: se,
    lower: Math.exp(logOR - z * se),
    upper: Math.exp(logOR + z * se),
    confidenceLevel: confidence,
  };
}

/**
 * Relative risk (risk ratio) for a 2×2 contingency table.
 *
 * Table layout:
 *   [[a, b], [c, d]]
 * where a=exposed+outcome, b=exposed+no_outcome,
 *       c=unexposed+outcome, d=unexposed+no_outcome
 *
 * @param table - 2×2 contingency table
 * @param confidence - Confidence level for the interval (default 0.95)
 */
export function relativeRisk(
  table: number[][],
  confidence = 0.95,
): OddsRatioResult {
  if (table.length !== 2 || table[0].length !== 2 || table[1].length !== 2) {
    throw new Error("Relative risk requires a 2×2 table");
  }
  if (confidence <= 0 || confidence >= 1) {
    throw new Error("Confidence level must be between 0 and 1 (exclusive)");
  }

  const a = table[0][0];
  const b = table[0][1];
  const c = table[1][0];
  const d = table[1][1];

  if (a < 0 || b < 0 || c < 0 || d < 0) {
    throw new Error("Cell counts must be non-negative");
  }

  const r1 = a + b; // exposed total
  const r2 = c + d; // unexposed total
  if (r1 === 0 || r2 === 0) {
    throw new Error("Row totals must be positive");
  }

  const p1 = a / r1;
  const p2 = c / r2;
  if (p2 === 0) {
    throw new Error("Cannot compute relative risk when baseline risk is zero");
  }

  const rr = p1 / p2;
  const logRR = Math.log(rr);
  const se = Math.sqrt(b / (a * r1) + d / (c * r2));

  const z = new Normal().quantile(1 - (1 - confidence) / 2);

  return {
    oddsRatio: rr,
    logOddsRatio: logRR,
    standardError: se,
    lower: Math.exp(logRR - z * se),
    upper: Math.exp(logRR + z * se),
    confidenceLevel: confidence,
  };
}

/**
 * Point-biserial correlation as an effect size for a t-test.
 * Computed from the t-statistic and degrees of freedom.
 *
 * @param tStatistic - The t-statistic from a t-test
 * @param df - Degrees of freedom
 */
export function pointBiserialR(
  tStatistic: number,
  df: number,
): EffectSizeResult {
  if (df <= 0) throw new Error("Degrees of freedom must be positive");

  const r = Math.sqrt(tStatistic ** 2 / (tStatistic ** 2 + df));
  const signed = tStatistic >= 0 ? r : -r;

  return {
    measure: "Point-biserial r",
    value: signed,
    interpretation: interpretR(Math.abs(signed)),
  };
}

// ---- Interpretation helpers ----

function interpretCohensD(d: number): EffectSizeResult["interpretation"] {
  if (d < 0.2) return "negligible";
  if (d < 0.5) return "small";
  if (d < 0.8) return "medium";
  return "large";
}

function interpretEtaSquared(
  eta2: number,
): EffectSizeResult["interpretation"] {
  if (eta2 < 0.01) return "negligible";
  if (eta2 < 0.06) return "small";
  if (eta2 < 0.14) return "medium";
  return "large";
}

function interpretCohensF(f: number): EffectSizeResult["interpretation"] {
  if (f < 0.1) return "negligible";
  if (f < 0.25) return "small";
  if (f < 0.4) return "medium";
  return "large";
}

function interpretCramersV(
  v: number,
  k: number,
): EffectSizeResult["interpretation"] {
  // Cohen's guidelines adjusted by df
  if (k === 2) return interpretCohensD(v); // same thresholds as phi
  if (k === 3) {
    if (v < 0.07) return "negligible";
    if (v < 0.21) return "small";
    if (v < 0.35) return "medium";
    return "large";
  }
  // k >= 4
  if (v < 0.06) return "negligible";
  if (v < 0.17) return "small";
  if (v < 0.29) return "medium";
  return "large";
}

function interpretR(r: number): EffectSizeResult["interpretation"] {
  if (r < 0.1) return "negligible";
  if (r < 0.3) return "small";
  if (r < 0.5) return "medium";
  return "large";
}
