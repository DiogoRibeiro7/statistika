import { mean } from "./utils/descriptive";
import { normalCdf, normalQuantile } from "./utils/linalg";
import { regularizedGammaP } from "./utils/math";

// ---------------------------------------------------------------------------
// Interfaces
// ---------------------------------------------------------------------------

export interface MetaAnalysisResult {
  pooledEffect: number;
  standardError: number;
  confidenceInterval: { lower: number; upper: number };
  zStatistic: number;
  pValue: number;
  weights: number[];
  method: "fixed" | "random";
}

export interface RandomEffectsResult extends MetaAnalysisResult {
  tau2: number;
  i2: number;
  q: number;
  qPValue: number;
  h2: number;
}

export interface ForestPlotData {
  studies: {
    label: string;
    effect: number;
    lower: number;
    upper: number;
    weight: number;
  }[];
  overall: { effect: number; lower: number; upper: number };
  method: string;
}

export interface FunnelPlotData {
  points: { effect: number; se: number }[];
  pooledEffect: number;
  pseudoCI: { se: number; lower: number; upper: number }[];
}

export interface PublicationBiasTest {
  statistic: number;
  pValue: number;
  reject: boolean;
  method: string;
}

export interface TrimAndFillResult {
  originalPooled: number;
  adjustedPooled: number;
  adjustedCI: { lower: number; upper: number };
  nMissing: number;
  filledEffects: number[];
  filledVariances: number[];
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Validates that effects and variances arrays are non-empty, equal length,
 * and that all variances are positive.
 */
function validateInputs(effects: number[], variances: number[]): void {
  if (effects.length === 0) {
    throw new Error("Effects array must not be empty");
  }
  if (effects.length !== variances.length) {
    throw new Error("Effects and variances arrays must have the same length");
  }
  if (effects.length < 2) {
    throw new Error("At least 2 studies are required for meta-analysis");
  }
  for (let i = 0; i < variances.length; i++) {
    if (variances[i] <= 0) {
      throw new Error("All variances must be positive");
    }
  }
}

/**
 * Computes the chi-squared survival function P(X > x) for df degrees of freedom
 * using the regularized incomplete gamma function.
 */
function chiSquaredSf(x: number, df: number): number {
  if (x <= 0) return 1;
  return 1 - regularizedGammaP(df / 2, x / 2);
}

/**
 * Computes Cochran's Q statistic and associated heterogeneity measures.
 */
function computeHeterogeneity(
  effects: number[],
  weights: number[],
  pooledEffect: number,
): { q: number; qPValue: number; i2: number; h2: number; tau2: number } {
  const k = effects.length;
  const df = k - 1;

  let q = 0;
  let sumW = 0;
  let sumW2 = 0;
  for (let i = 0; i < k; i++) {
    q += weights[i] * (effects[i] - pooledEffect) ** 2;
    sumW += weights[i];
    sumW2 += weights[i] ** 2;
  }

  const qPValue = chiSquaredSf(q, df);

  // DerSimonian-Laird tau² estimator
  const c = sumW - sumW2 / sumW;
  const tau2 = Math.max(0, (q - df) / c);

  // I² (percentage of variability due to heterogeneity)
  const i2 = df > 0 ? Math.max(0, ((q - df) / q) * 100) : 0;

  // H² statistic
  const h2 = df > 0 ? Math.max(1, q / df) : 1;

  return { q, qPValue, i2, h2, tau2 };
}

// ---------------------------------------------------------------------------
// Fixed-effects model
// ---------------------------------------------------------------------------

/**
 * Performs a fixed-effects meta-analysis using inverse variance weighting.
 *
 * Each study's weight is the inverse of its variance (wi = 1/vi). The pooled
 * effect is the weighted mean of the individual study effects:
 * θ_pooled = Σ(wi * θi) / Σ(wi).
 *
 * @param effects - Array of effect sizes from individual studies
 * @param variances - Array of within-study variances corresponding to each effect
 * @returns A {@link MetaAnalysisResult} containing the pooled effect estimate,
 *   standard error, 95% confidence interval, z-test statistic, and p-value
 * @throws {Error} If arrays are empty, unequal length, fewer than 2 studies,
 *   or any variance is non-positive
 *
 * @example
 * ```ts
 * const effects = [0.5, 0.3, 0.7, 0.4];
 * const variances = [0.04, 0.06, 0.05, 0.03];
 * const result = fixedEffectsMeta(effects, variances);
 * console.log(result.pooledEffect); // weighted average effect
 * console.log(result.pValue);       // p-value for test of pooled effect = 0
 * ```
 */
export function fixedEffectsMeta(
  effects: number[],
  variances: number[],
): MetaAnalysisResult {
  validateInputs(effects, variances);

  const k = effects.length;
  const weights: number[] = new Array(k);
  let sumW = 0;
  let sumWY = 0;

  for (let i = 0; i < k; i++) {
    weights[i] = 1 / variances[i];
    sumW += weights[i];
    sumWY += weights[i] * effects[i];
  }

  const pooledEffect = sumWY / sumW;
  const se = Math.sqrt(1 / sumW);
  const z = pooledEffect / se;
  const pValue = 2 * (1 - normalCdf(Math.abs(z)));
  const zCrit = normalQuantile(0.975);

  return {
    pooledEffect,
    standardError: se,
    confidenceInterval: {
      lower: pooledEffect - zCrit * se,
      upper: pooledEffect + zCrit * se,
    },
    zStatistic: z,
    pValue,
    weights,
    method: "fixed",
  };
}

// ---------------------------------------------------------------------------
// Random-effects model
// ---------------------------------------------------------------------------

/**
 * Performs a random-effects meta-analysis using the DerSimonian-Laird method.
 *
 * First computes a fixed-effects estimate to derive Cochran's Q statistic,
 * then estimates the between-study variance (τ²) via the DerSimonian-Laird
 * estimator. Study weights are recalculated as wi* = 1/(vi + τ²) and the
 * pooled effect re-estimated.
 *
 * @param effects - Array of effect sizes from individual studies
 * @param variances - Array of within-study variances corresponding to each effect
 * @returns A {@link RandomEffectsResult} extending the base result with
 *   heterogeneity statistics: τ², I², Cochran's Q, Q p-value, and H²
 * @throws {Error} If arrays are empty, unequal length, fewer than 2 studies,
 *   or any variance is non-positive
 *
 * @example
 * ```ts
 * const effects = [0.5, 0.3, 0.7, 0.4, 0.6];
 * const variances = [0.04, 0.06, 0.05, 0.03, 0.07];
 * const result = randomEffectsMeta(effects, variances);
 * console.log(result.tau2); // between-study variance
 * console.log(result.i2);   // I-squared percentage
 * console.log(result.q);    // Cochran's Q
 * ```
 */
export function randomEffectsMeta(
  effects: number[],
  variances: number[],
): RandomEffectsResult {
  validateInputs(effects, variances);

  const k = effects.length;

  // Step 1: fixed-effects weights for Q computation
  const fixedWeights: number[] = new Array(k);
  let sumW = 0;
  let sumWY = 0;
  for (let i = 0; i < k; i++) {
    fixedWeights[i] = 1 / variances[i];
    sumW += fixedWeights[i];
    sumWY += fixedWeights[i] * effects[i];
  }
  const fixedPooled = sumWY / sumW;

  // Step 2: heterogeneity from fixed-effects weights
  const het = computeHeterogeneity(effects, fixedWeights, fixedPooled);

  // Step 3: random-effects weights incorporating tau²
  const randomWeights: number[] = new Array(k);
  let sumRW = 0;
  let sumRWY = 0;
  for (let i = 0; i < k; i++) {
    randomWeights[i] = 1 / (variances[i] + het.tau2);
    sumRW += randomWeights[i];
    sumRWY += randomWeights[i] * effects[i];
  }

  const pooledEffect = sumRWY / sumRW;
  const se = Math.sqrt(1 / sumRW);
  const z = pooledEffect / se;
  const pValue = 2 * (1 - normalCdf(Math.abs(z)));
  const zCrit = normalQuantile(0.975);

  return {
    pooledEffect,
    standardError: se,
    confidenceInterval: {
      lower: pooledEffect - zCrit * se,
      upper: pooledEffect + zCrit * se,
    },
    zStatistic: z,
    pValue,
    weights: randomWeights,
    method: "random",
    tau2: het.tau2,
    i2: het.i2,
    q: het.q,
    qPValue: het.qPValue,
    h2: het.h2,
  };
}

// ---------------------------------------------------------------------------
// Forest plot data
// ---------------------------------------------------------------------------

/**
 * Generates structured data for rendering a forest plot.
 *
 * Performs a random-effects meta-analysis to obtain the pooled estimate and
 * per-study weights, then assembles per-study effect sizes with 95% CIs and
 * relative weights suitable for visualization.
 *
 * @param effects - Array of effect sizes from individual studies
 * @param variances - Array of within-study variances corresponding to each effect
 * @param labels - Array of study labels (e.g., "Author 2020")
 * @returns A {@link ForestPlotData} object containing per-study and overall data
 * @throws {Error} If arrays are empty, unequal length, fewer than 2 studies,
 *   or any variance is non-positive. Also throws if labels length does not match.
 *
 * @example
 * ```ts
 * const effects = [0.5, 0.3, 0.7];
 * const variances = [0.04, 0.06, 0.05];
 * const labels = ["Study A", "Study B", "Study C"];
 * const plot = forestPlotData(effects, variances, labels);
 * console.log(plot.studies[0].label);  // "Study A"
 * console.log(plot.overall.effect);    // pooled effect
 * ```
 */
export function forestPlotData(
  effects: number[],
  variances: number[],
  labels: string[],
): ForestPlotData {
  validateInputs(effects, variances);
  if (labels.length !== effects.length) {
    throw new Error("Labels array must have the same length as effects");
  }

  const meta = randomEffectsMeta(effects, variances);
  const zCrit = normalQuantile(0.975);
  const totalWeight = meta.weights.reduce((s, w) => s + w, 0);

  const studies = effects.map((eff, i) => {
    const se = Math.sqrt(variances[i]);
    return {
      label: labels[i],
      effect: eff,
      lower: eff - zCrit * se,
      upper: eff + zCrit * se,
      weight: (meta.weights[i] / totalWeight) * 100,
    };
  });

  return {
    studies,
    overall: {
      effect: meta.pooledEffect,
      lower: meta.confidenceInterval.lower,
      upper: meta.confidenceInterval.upper,
    },
    method: "random",
  };
}

// ---------------------------------------------------------------------------
// Funnel plot data
// ---------------------------------------------------------------------------

/**
 * Generates structured data for rendering a funnel plot.
 *
 * A funnel plot displays each study's effect size against its standard error.
 * In the absence of publication bias, the plot should resemble an inverted
 * funnel centered on the pooled effect. Pseudo-confidence interval lines
 * (at 95%) are included to aid visual inspection.
 *
 * @param effects - Array of effect sizes from individual studies
 * @param standardErrors - Array of standard errors corresponding to each effect
 * @returns A {@link FunnelPlotData} object with study points, pooled effect,
 *   and pseudo-CI boundary lines at various SE levels
 * @throws {Error} If arrays are empty, unequal length, fewer than 2 studies,
 *   or any standard error is non-positive
 *
 * @example
 * ```ts
 * const effects = [0.5, 0.3, 0.7, 0.4];
 * const ses = [0.2, 0.25, 0.22, 0.18];
 * const funnel = funnelPlotData(effects, ses);
 * console.log(funnel.pooledEffect); // fixed-effects pooled estimate
 * console.log(funnel.pseudoCI);     // boundary lines for funnel shape
 * ```
 */
export function funnelPlotData(
  effects: number[],
  standardErrors: number[],
): FunnelPlotData {
  if (effects.length === 0) {
    throw new Error("Effects array must not be empty");
  }
  if (effects.length !== standardErrors.length) {
    throw new Error(
      "Effects and standard errors arrays must have the same length",
    );
  }
  if (effects.length < 2) {
    throw new Error("At least 2 studies are required");
  }
  for (let i = 0; i < standardErrors.length; i++) {
    if (standardErrors[i] <= 0) {
      throw new Error("All standard errors must be positive");
    }
  }

  // Compute fixed-effects pooled estimate for the funnel center
  const variances = standardErrors.map((se) => se * se);
  const fixed = fixedEffectsMeta(effects, variances);

  const points = effects.map((eff, i) => ({
    effect: eff,
    se: standardErrors[i],
  }));

  // Pseudo-CI lines at various SE levels
  const maxSE = Math.max(...standardErrors);
  const zCrit = normalQuantile(0.975);
  const nSteps = 50;
  const pseudoCI: { se: number; lower: number; upper: number }[] = [];

  for (let i = 0; i <= nSteps; i++) {
    const se = (maxSE * i) / nSteps;
    pseudoCI.push({
      se,
      lower: fixed.pooledEffect - zCrit * se,
      upper: fixed.pooledEffect + zCrit * se,
    });
  }

  return {
    points,
    pooledEffect: fixed.pooledEffect,
    pseudoCI,
  };
}

// ---------------------------------------------------------------------------
// Egger's test
// ---------------------------------------------------------------------------

/**
 * Performs Egger's regression test for funnel plot asymmetry.
 *
 * Tests for publication bias by regressing the standardized effect
 * (effect / SE) on precision (1 / SE). Under the null hypothesis of no
 * funnel plot asymmetry, the intercept of this regression is zero.
 * A significant intercept suggests small-study effects / publication bias.
 *
 * @param effects - Array of effect sizes from individual studies
 * @param standardErrors - Array of standard errors corresponding to each effect
 * @returns A {@link PublicationBiasTest} with the t-statistic for the intercept,
 *   p-value, rejection decision at α = 0.10, and method name
 * @throws {Error} If arrays are empty, unequal length, fewer than 3 studies,
 *   or any standard error is non-positive
 *
 * @example
 * ```ts
 * const effects = [0.5, 0.3, 0.7, 0.4, 0.6];
 * const ses = [0.2, 0.25, 0.22, 0.18, 0.3];
 * const test = eggersTest(effects, ses);
 * console.log(test.pValue);  // p-value for asymmetry
 * console.log(test.reject);  // true if significant at α = 0.10
 * ```
 */
export function eggersTest(
  effects: number[],
  standardErrors: number[],
): PublicationBiasTest {
  if (effects.length === 0) {
    throw new Error("Effects array must not be empty");
  }
  if (effects.length !== standardErrors.length) {
    throw new Error(
      "Effects and standard errors arrays must have the same length",
    );
  }
  if (effects.length < 3) {
    throw new Error("At least 3 studies are required for Egger's test");
  }
  for (let i = 0; i < standardErrors.length; i++) {
    if (standardErrors[i] <= 0) {
      throw new Error("All standard errors must be positive");
    }
  }

  const k = effects.length;

  // Regress standardized effect (y = effect/SE) on precision (x = 1/SE)
  // y_i = a + b * x_i + error
  const x: number[] = new Array(k);
  const y: number[] = new Array(k);
  for (let i = 0; i < k; i++) {
    x[i] = 1 / standardErrors[i];
    y[i] = effects[i] / standardErrors[i];
  }

  const xMean = mean(x);
  const yMean = mean(y);

  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < k; i++) {
    const dx = x[i] - xMean;
    sxx += dx * dx;
    sxy += dx * (y[i] - yMean);
  }

  const slope = sxy / sxx;
  const intercept = yMean - slope * xMean;

  // Residual standard error
  let rss = 0;
  for (let i = 0; i < k; i++) {
    const residual = y[i] - (intercept + slope * x[i]);
    rss += residual * residual;
  }
  const residualSE = Math.sqrt(rss / (k - 2));

  // SE of intercept
  const interceptSE = residualSE * Math.sqrt(1 / k + (xMean * xMean) / sxx);

  // t-statistic for intercept
  const tStat = intercept / interceptSE;
  const df = k - 2;

  // Approximate p-value for t-distribution using normal approximation
  // (conservative for small df, but standard in many implementations)
  const pValue = 2 * (1 - normalCdf(Math.abs(tStat) * Math.sqrt(df / (df + tStat * tStat)) * Math.sqrt((df + 1) / df)));

  return {
    statistic: tStat,
    pValue,
    reject: pValue < 0.1,
    method: "Egger's regression test",
  };
}

// ---------------------------------------------------------------------------
// Begg's test
// ---------------------------------------------------------------------------

/**
 * Performs Begg and Mazumdar's rank correlation test for publication bias.
 *
 * Computes Kendall's tau between the standardized effect sizes and their
 * variances. A significant correlation suggests funnel plot asymmetry
 * indicative of publication bias.
 *
 * @param effects - Array of effect sizes from individual studies
 * @param variances - Array of within-study variances corresponding to each effect
 * @returns A {@link PublicationBiasTest} with Kendall's tau statistic,
 *   two-sided p-value (normal approximation), rejection at α = 0.10, and method
 * @throws {Error} If arrays are empty, unequal length, fewer than 3 studies,
 *   or any variance is non-positive
 *
 * @example
 * ```ts
 * const effects = [0.5, 0.3, 0.7, 0.4, 0.6];
 * const variances = [0.04, 0.06, 0.05, 0.03, 0.07];
 * const test = beggsTest(effects, variances);
 * console.log(test.statistic); // Kendall's tau
 * console.log(test.pValue);    // two-sided p-value
 * ```
 */
export function beggsTest(
  effects: number[],
  variances: number[],
): PublicationBiasTest {
  if (effects.length === 0) {
    throw new Error("Effects array must not be empty");
  }
  if (effects.length !== variances.length) {
    throw new Error("Effects and variances arrays must have the same length");
  }
  if (effects.length < 3) {
    throw new Error("At least 3 studies are required for Begg's test");
  }
  for (let i = 0; i < variances.length; i++) {
    if (variances[i] <= 0) {
      throw new Error("All variances must be positive");
    }
  }

  const k = effects.length;

  // Standardize effects by subtracting pooled estimate and dividing by SE
  const fixed = fixedEffectsMeta(effects, variances);
  const standardized: number[] = new Array(k);
  for (let i = 0; i < k; i++) {
    standardized[i] =
      (effects[i] - fixed.pooledEffect) / Math.sqrt(variances[i]);
  }

  // Kendall's tau between standardized effects and variances
  let concordant = 0;
  let discordant = 0;
  for (let i = 0; i < k - 1; i++) {
    for (let j = i + 1; j < k; j++) {
      const dx = standardized[i] - standardized[j];
      const dy = variances[i] - variances[j];
      const product = dx * dy;
      if (product > 0) concordant++;
      else if (product < 0) discordant++;
    }
  }

  const nPairs = (k * (k - 1)) / 2;
  const tau = (concordant - discordant) / nPairs;

  // Normal approximation for Kendall's tau
  const tauVar = (2 * (2 * k + 5)) / (9 * k * (k - 1));
  const z = tau / Math.sqrt(tauVar);
  const pValue = 2 * (1 - normalCdf(Math.abs(z)));

  return {
    statistic: tau,
    pValue,
    reject: pValue < 0.1,
    method: "Begg's rank correlation test",
  };
}

// ---------------------------------------------------------------------------
// Trim-and-fill
// ---------------------------------------------------------------------------

/**
 * Performs the trim-and-fill method for estimating and adjusting for
 * publication bias.
 *
 * Uses the L0 estimator (Duval & Tweedie, 2000) to estimate the number of
 * missing studies on the side of the funnel plot showing fewer studies.
 * Missing studies are imputed by reflecting existing extreme studies across
 * the pooled effect, and the meta-analytic estimate is recalculated.
 *
 * @param effects - Array of effect sizes from individual studies
 * @param variances - Array of within-study variances corresponding to each effect
 * @returns A {@link TrimAndFillResult} with the original pooled estimate,
 *   adjusted estimate after imputation, number of imputed studies, and the
 *   filled effect/variance arrays
 * @throws {Error} If arrays are empty, unequal length, fewer than 3 studies,
 *   or any variance is non-positive
 *
 * @example
 * ```ts
 * const effects = [0.5, 0.3, 0.7, 0.4, 0.6, 0.9];
 * const variances = [0.04, 0.06, 0.05, 0.03, 0.07, 0.04];
 * const result = trimAndFill(effects, variances);
 * console.log(result.nMissing);       // estimated missing studies
 * console.log(result.adjustedPooled); // adjusted pooled effect
 * ```
 */
export function trimAndFill(
  effects: number[],
  variances: number[],
): TrimAndFillResult {
  if (effects.length === 0) {
    throw new Error("Effects array must not be empty");
  }
  if (effects.length !== variances.length) {
    throw new Error("Effects and variances arrays must have the same length");
  }
  if (effects.length < 3) {
    throw new Error("At least 3 studies are required for trim-and-fill");
  }
  for (let i = 0; i < variances.length; i++) {
    if (variances[i] <= 0) {
      throw new Error("All variances must be positive");
    }
  }

  const k = effects.length;

  // Step 1: original pooled estimate
  const originalMeta = fixedEffectsMeta(effects, variances);
  const originalPooled = originalMeta.pooledEffect;

  // Step 2: estimate number of missing studies using the L0 estimator
  // Sort effects by distance from pooled estimate
  const indexed = effects.map((e, i) => ({
    effect: e,
    variance: variances[i],
    dist: Math.abs(e - originalPooled),
    sign: e - originalPooled >= 0 ? 1 : -1,
  }));
  indexed.sort((a, b) => a.dist - b.dist);

  // Determine which side has fewer studies (the "missing" side)
  let nRight = 0;
  let nLeft = 0;
  for (let i = 0; i < k; i++) {
    if (effects[i] >= originalPooled) nRight++;
    else nLeft++;
  }
  const missingOnRight = nLeft > nRight;

  // Rank by distance from pooled, count studies on the asymmetric side
  // that are further than the furthest study on the opposite side
  const sorted = indexed.slice();
  sorted.sort((a, b) => a.dist - b.dist);

  // Assign ranks (1-based) by distance from center
  const ranks = sorted.map((_, i) => i + 1);

  // Count studies on the side with more studies
  // For each study on the "fat" side, count how many lack a mirror
  let sn = 0;
  for (let i = k - 1; i >= 0; i--) {
    const s = sorted[i];
    const onFatSide = missingOnRight ? s.sign < 0 : s.sign > 0;
    if (onFatSide) {
      // Check if there's a matching study on the other side
      const mirrorEffect = 2 * originalPooled - s.effect;
      const hasMatch = sorted.some(
        (other) =>
          other !== s &&
          Math.abs(other.effect - mirrorEffect) < s.dist * 0.5,
      );
      if (!hasMatch) sn++;
    }
  }

  // L0 estimator: R0 = max(0, round(Sn))
  // Using iterative approach for robustness
  let nMissing = 0;
  let currentEffects = [...effects];
  let currentVariances = [...variances];

  for (let iter = 0; iter < 10; iter++) {
    const meta = fixedEffectsMeta(currentEffects, currentVariances);
    const center = meta.pooledEffect;

    // Sort original studies by distance from current center
    const dists: { idx: number; dist: number; sign: number }[] = [];
    for (let i = 0; i < k; i++) {
      dists.push({
        idx: i,
        dist: Math.abs(effects[i] - center),
        sign: effects[i] >= center ? 1 : -1,
      });
    }
    dists.sort((a, b) => a.dist - b.dist);

    // Count asymmetric studies on the side with more mass
    let rightCount = 0;
    let leftCount = 0;
    for (const d of dists) {
      if (d.sign >= 0) rightCount++;
      else leftCount++;
    }
    const excessOnRight = rightCount > leftCount;

    // Rank from most extreme inward; count unmatched on excess side
    let tn = 0;
    for (let i = k - 1; i >= 0; i--) {
      const d = dists[i];
      const onExcessSide = excessOnRight ? d.sign > 0 : d.sign < 0;
      if (onExcessSide) tn++;
      else break;
    }

    // L0 estimator
    const r0 = Math.max(0, Math.round((4 * tn - k) / 2));

    if (r0 === nMissing) break;
    nMissing = r0;

    // Impute missing studies by reflecting the most extreme studies
    currentEffects = [...effects];
    currentVariances = [...variances];

    // Get the most extreme studies on the excess side
    const extremeStudies: { effect: number; variance: number }[] = [];
    for (let i = k - 1; i >= 0 && extremeStudies.length < nMissing; i--) {
      const d = dists[i];
      const onExcessSide = excessOnRight ? d.sign > 0 : d.sign < 0;
      if (onExcessSide) {
        extremeStudies.push({
          effect: effects[d.idx],
          variance: variances[d.idx],
        });
      }
    }

    // Mirror extreme studies across the center
    for (const study of extremeStudies) {
      const mirroredEffect = 2 * center - study.effect;
      currentEffects.push(mirroredEffect);
      currentVariances.push(study.variance);
    }
  }

  // Step 3: compute adjusted pooled estimate
  const adjustedMeta = fixedEffectsMeta(currentEffects, currentVariances);
  const zCrit = normalQuantile(0.975);

  return {
    originalPooled,
    adjustedPooled: adjustedMeta.pooledEffect,
    adjustedCI: {
      lower:
        adjustedMeta.pooledEffect - zCrit * adjustedMeta.standardError,
      upper:
        adjustedMeta.pooledEffect + zCrit * adjustedMeta.standardError,
    },
    nMissing,
    filledEffects: currentEffects,
    filledVariances: currentVariances,
  };
}
