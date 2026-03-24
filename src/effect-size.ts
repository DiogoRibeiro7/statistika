import { Dataset, EffectSizeResult, OddsRatioResult, AnovaResult } from "./types";
import { mean, variance } from "./utils/descriptive";
import { Normal } from "./distributions/continuous/normal";

/**
 * Cohen's d for two independent samples.
 *
 * Computes the standardized mean difference using the pooled standard deviation
 * as the denominator: d = (M1 - M2) / Sp, where Sp = sqrt(((n1-1)*s1^2 + (n2-1)*s2^2) / (n1+n2-2)).
 *
 * @param data1 - First sample array of numeric observations
 * @param data2 - Second sample array of numeric observations
 * @returns An {@link EffectSizeResult} with the Cohen's d value and qualitative interpretation
 *   (negligible if |d| < 0.2, small if < 0.5, medium if < 0.8, large otherwise)
 * @throws {Error} If either dataset has fewer than 2 elements
 *
 * @example
 * ```ts
 * const treatment = [5.1, 6.2, 7.3, 5.8, 6.5];
 * const control = [3.2, 4.1, 3.8, 4.5, 3.9];
 * const result = cohensD(treatment, control);
 * console.log(result.value);          // positive d indicating treatment > control
 * console.log(result.interpretation); // e.g. "large"
 * ```
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
 *
 * Uses the standard deviation of the control group as the denominator:
 * delta = (Mt - Mc) / Sc. Preferred over Cohen's d when group variances
 * are unequal and the control group variance is the more appropriate reference.
 *
 * @param treatment - Treatment group data array
 * @param control - Control group data array (its SD is used as the denominator)
 * @returns An {@link EffectSizeResult} with the Glass's delta value and interpretation
 * @throws {Error} If either dataset has fewer than 2 elements
 *
 * @example
 * ```ts
 * const result = glassDelta([8, 9, 10], [4, 5, 6]);
 * console.log(result.value); // positive delta
 * ```
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
 * Hedges' g -- bias-corrected version of Cohen's d for small samples.
 *
 * Applies the approximate correction factor J = 1 - 3/(4*df - 1) to Cohen's d,
 * reducing positive bias in small-sample effect size estimation.
 *
 * @param data1 - First sample array
 * @param data2 - Second sample array
 * @returns An {@link EffectSizeResult} with the Hedges' g value and interpretation
 * @throws {Error} If either dataset has fewer than 2 elements
 *
 * @example
 * ```ts
 * const g = hedgesG([2, 3, 4], [5, 6, 7]);
 * console.log(g.value); // bias-corrected effect size
 * ```
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
 *
 * Computes d = Md / Sd, where Md is the mean of the pairwise differences
 * and Sd is their standard deviation. Appropriate for repeated-measures
 * or matched-pairs designs.
 *
 * @param data1 - First sample (or pre-treatment observations)
 * @param data2 - Second sample (paired with data1, e.g. post-treatment)
 * @returns An {@link EffectSizeResult} with the paired Cohen's d value and interpretation
 * @throws {Error} If datasets have different lengths
 * @throws {Error} If datasets have fewer than 2 elements
 *
 * @example
 * ```ts
 * const pre  = [10, 12, 14, 16];
 * const post = [12, 15, 17, 20];
 * const result = pairedCohensD(pre, post);
 * console.log(result.value); // negative d (pre < post)
 * ```
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
 * Eta-squared (eta^2) from ANOVA results.
 *
 * Computes the proportion of total variance explained by the grouping variable:
 * eta^2 = SS_between / (SS_between + SS_within).
 *
 * Interpretation thresholds: < 0.01 negligible, < 0.06 small, < 0.14 medium, >= 0.14 large.
 *
 * @param anovaResult - Result object from a one-way ANOVA (must include ssBetween and ssWithin)
 * @returns An {@link EffectSizeResult} with the eta-squared value and interpretation
 *
 * @example
 * ```ts
 * const anova = oneWayAnova([group1, group2, group3]);
 * const es = etaSquared(anova);
 * console.log(es.value); // proportion of variance explained
 * ```
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
 * Partial eta-squared (eta^2_p) from ANOVA results.
 *
 * For one-way ANOVA this is identical to eta-squared:
 * eta^2_p = SS_between / (SS_between + SS_within).
 * In factorial designs, partial eta-squared isolates the variance attributable
 * to one factor relative to that factor's SS plus residual SS.
 *
 * @param anovaResult - Result object from a one-way ANOVA
 * @returns An {@link EffectSizeResult} with the partial eta-squared value and interpretation
 *
 * @example
 * ```ts
 * const anova = oneWayAnova([group1, group2, group3]);
 * const es = partialEtaSquared(anova);
 * console.log(es.value); // proportion of variance explained (same as eta-squared for one-way)
 * ```
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
 * Omega-squared (omega^2) from ANOVA results.
 *
 * A less biased estimator of the population effect size than eta-squared:
 * omega^2 = (SS_between - df_between * MS_within) / (SS_total + MS_within).
 * The result is floored at 0 to avoid negative estimates.
 *
 * @param anovaResult - Result object from a one-way ANOVA
 * @returns An {@link EffectSizeResult} with the omega-squared value (>= 0) and interpretation
 *
 * @example
 * ```ts
 * const anova = oneWayAnova([group1, group2, group3]);
 * const es = omegaSquared(anova);
 * console.log(es.value); // less biased than eta-squared
 * ```
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
 *
 * Computed as f = sqrt(eta^2 / (1 - eta^2)). Interpretation thresholds:
 * < 0.1 negligible, < 0.25 small, < 0.4 medium, >= 0.4 large.
 *
 * @param anovaResult - Result object from a one-way ANOVA
 * @returns An {@link EffectSizeResult} with the Cohen's f value and interpretation
 *
 * @example
 * ```ts
 * const anova = oneWayAnova([group1, group2]);
 * const f = cohensF(anova);
 * console.log(f.interpretation); // e.g. "medium"
 * ```
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
 * Cramer's V for a contingency table.
 *
 * Measures the association between two categorical variables:
 * V = sqrt(chi^2 / (n * (k - 1))), where k = min(nRows, nCols).
 * Ranges from 0 (no association) to 1 (perfect association).
 *
 * @param table - Contingency table as a 2D array of observed counts (at least 2x2)
 * @returns An {@link EffectSizeResult} with the Cramer's V value and interpretation
 * @throws {Error} If the table has fewer than 2 rows or 2 columns
 * @throws {Error} If the table total is zero
 *
 * @example
 * ```ts
 * const table = [[10, 20], [30, 40]];
 * const result = cramersV(table);
 * console.log(result.value); // association strength
 * ```
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
 * Phi coefficient (phi) for a 2x2 contingency table.
 *
 * A special case of Cramer's V for 2x2 tables:
 * phi = (ad - bc) / sqrt((a+b)(c+d)(a+c)(b+d)).
 * Ranges from -1 to +1, where the sign indicates direction of association.
 *
 * @param table - A 2x2 contingency table as `[[a, b], [c, d]]`
 * @returns An {@link EffectSizeResult} with the phi value and interpretation
 * @throws {Error} If the table is not exactly 2x2
 *
 * @example
 * ```ts
 * const result = phiCoefficient([[20, 5], [10, 15]]);
 * console.log(result.value); // positive association
 * ```
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
 * Odds ratio for a 2x2 contingency table with Woolf logit confidence interval.
 *
 * OR = (a*d) / (b*c). When any cell is zero, the Haldane-Anscombe correction
 * (+0.5 to all cells) is applied. The confidence interval is computed on the
 * log-odds scale using SE = sqrt(1/a + 1/b + 1/c + 1/d) and then exponentiated.
 *
 * Table layout:
 *   `[[a, b], [c, d]]`
 * where a = exposed+outcome, b = exposed+no_outcome,
 *       c = unexposed+outcome, d = unexposed+no_outcome.
 *
 * @param table - A 2x2 contingency table of non-negative counts
 * @param confidence - Confidence level for the interval, in (0, 1) (default 0.95)
 * @returns An {@link OddsRatioResult} with the odds ratio, log-OR, SE, and CI bounds
 * @throws {Error} If the table is not exactly 2x2
 * @throws {Error} If confidence level is not in (0, 1)
 * @throws {Error} If any cell count is negative
 *
 * @example
 * ```ts
 * const result = oddsRatio([[30, 10], [20, 40]]);
 * console.log(result.oddsRatio); // 6.0
 * console.log(result.lower, result.upper); // 95% CI bounds
 * ```
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
 * Relative risk (risk ratio) for a 2x2 contingency table.
 *
 * RR = (a/(a+b)) / (c/(c+d)). The confidence interval is computed on the
 * log scale using SE = sqrt(b/(a*(a+b)) + d/(c*(c+d))) and then exponentiated.
 *
 * Table layout:
 *   `[[a, b], [c, d]]`
 * where a = exposed+outcome, b = exposed+no_outcome,
 *       c = unexposed+outcome, d = unexposed+no_outcome.
 *
 * @param table - A 2x2 contingency table of non-negative counts
 * @param confidence - Confidence level for the interval, in (0, 1) (default 0.95)
 * @returns An {@link OddsRatioResult} with the risk ratio, log-RR, SE, and CI bounds
 * @throws {Error} If the table is not exactly 2x2
 * @throws {Error} If confidence level is not in (0, 1)
 * @throws {Error} If any cell count is negative
 * @throws {Error} If row totals are zero or baseline risk is zero
 *
 * @example
 * ```ts
 * const result = relativeRisk([[30, 70], [10, 90]]);
 * console.log(result.oddsRatio); // risk ratio = 3.0
 * ```
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
 *
 * Converts a t-statistic to a correlation coefficient:
 * r = sqrt(t^2 / (t^2 + df)), with the sign preserved from the t-statistic.
 * Interpretation: < 0.1 negligible, < 0.3 small, < 0.5 medium, >= 0.5 large.
 *
 * @param tStatistic - The t-statistic from a t-test
 * @param df - Degrees of freedom (must be positive)
 * @returns An {@link EffectSizeResult} with the signed point-biserial r and interpretation
 * @throws {Error} If degrees of freedom is not positive
 *
 * @example
 * ```ts
 * const r = pointBiserialR(2.5, 30);
 * console.log(r.value);          // ~0.415
 * console.log(r.interpretation); // "medium"
 * ```
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

/** Interpret a Cohen's d magnitude using standard thresholds (0.2, 0.5, 0.8). */
function interpretCohensD(d: number): EffectSizeResult["interpretation"] {
  if (d < 0.2) return "negligible";
  if (d < 0.5) return "small";
  if (d < 0.8) return "medium";
  return "large";
}

/** Interpret an eta-squared value using standard thresholds (0.01, 0.06, 0.14). */
function interpretEtaSquared(
  eta2: number,
): EffectSizeResult["interpretation"] {
  if (eta2 < 0.01) return "negligible";
  if (eta2 < 0.06) return "small";
  if (eta2 < 0.14) return "medium";
  return "large";
}

/** Interpret a Cohen's f value using standard thresholds (0.1, 0.25, 0.4). */
function interpretCohensF(f: number): EffectSizeResult["interpretation"] {
  if (f < 0.1) return "negligible";
  if (f < 0.25) return "small";
  if (f < 0.4) return "medium";
  return "large";
}

/** Interpret a Cramer's V value with thresholds adjusted by min(rows, cols). */
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

/** Interpret a correlation magnitude using standard thresholds (0.1, 0.3, 0.5). */
function interpretR(r: number): EffectSizeResult["interpretation"] {
  if (r < 0.1) return "negligible";
  if (r < 0.3) return "small";
  if (r < 0.5) return "medium";
  return "large";
}
