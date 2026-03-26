/**
 * Model comparison and selection utilities.
 *
 * - **Information criteria** — AIC, AICc (corrected), BIC for any model
 *   given its log-likelihood and parameter count.
 * - **Likelihood ratio test (LRT)** — compare nested models via the
 *   chi-squared distribution of −2 Δℓ.
 * - **Vuong test** — compare non-nested models on the same data using
 *   the individual log-likelihood contributions.
 */

import { ChiSquared } from "./distributions/continuous/chi-squared";
import { normalCdf } from "./utils/linalg";

// ── Information criteria ──────────────────────────────────────────────────

/**
 * Information criteria for model comparison.
 *
 * Lower values indicate better model fit (penalised for complexity).
 */
export interface InformationCriteria {
  /** Akaike Information Criterion: -2*logLik + 2*k. */
  aic: number;
  /** Corrected AIC for small samples: AIC + 2*k*(k+1)/(n-k-1). Infinity if n <= k+1. */
  aicc: number;
  /** Bayesian Information Criterion: -2*logLik + k*ln(n). */
  bic: number;
}

/**
 * Compute AIC, AICc, and BIC for a fitted model.
 *
 * - AIC = -2*logLik + 2*k
 * - AICc = AIC + 2*k*(k+1)/(n-k-1)  (corrected for small samples)
 * - BIC = -2*logLik + k*ln(n)
 *
 * @param logLikelihood - Maximised log-likelihood (must be finite)
 * @param k - Number of estimated parameters, including intercept (must be >= 1)
 * @param n - Number of observations (must be >= 1)
 * @returns An {@link InformationCriteria} object with AIC, AICc, and BIC
 * @throws {Error} If k < 1, n < 1, or logLikelihood is not finite
 *
 * @example
 * ```ts
 * const ic = informationCriteria(-150, 3, 100);
 * console.log(ic.aic);  // 306
 * console.log(ic.bic);  // ~313.8
 * ```
 */
export function informationCriteria(
  logLikelihood: number,
  k: number,
  n: number,
): InformationCriteria {
  if (k < 1) throw new Error(`Invalid parameter 'k': expected at least 1, received ${k}`);
  if (n < 1) throw new Error(`Invalid parameter 'n': expected at least 1, received ${n}`);
  if (!Number.isFinite(logLikelihood)) {
    throw new Error(`Invalid parameter 'logLikelihood': expected a finite number, received ${logLikelihood}`);
  }

  const aic = -2 * logLikelihood + 2 * k;
  const aicc =
    n - k - 1 > 0
      ? aic + (2 * k * (k + 1)) / (n - k - 1)
      : Infinity;
  const bic = -2 * logLikelihood + k * Math.log(n);

  return { aic, aicc, bic };
}

// ── Likelihood ratio test ─────────────────────────────────────────────────

/**
 * Result of a likelihood ratio test comparing two nested models.
 */
export interface LRTResult {
  /** Test statistic D = -2*(logLik_restricted - logLik_full), clamped to >= 0. */
  statistic: number;
  /** Degrees of freedom (difference in parameter counts between models). */
  degreesOfFreedom: number;
  /** p-value from chi-squared distribution under H0. */
  pValue: number;
  /** Whether the null hypothesis (restricted model is adequate) is rejected at the given alpha. */
  rejected: boolean;
}

/**
 * Likelihood ratio test for nested models.
 *
 * Tests H0: the restricted (simpler) model is adequate,
 * against H1: the full model fits significantly better.
 *
 * The test statistic D = -2*(logLik_restricted - logLik_full) follows a chi-squared
 * distribution with df = k_full - k_restricted under H0.
 *
 * @param logLikRestricted - Log-likelihood of the restricted (null) model
 * @param logLikFull - Log-likelihood of the full (alternative) model
 * @param dfRestricted - Number of parameters in the restricted model
 * @param dfFull - Number of parameters in the full model (must be > dfRestricted)
 * @param alpha - Significance level for rejection decision (default: 0.05)
 * @returns An {@link LRTResult} with test statistic, df, p-value, and rejection decision
 * @throws {Error} If log-likelihood values are not finite
 * @throws {Error} If dfFull <= dfRestricted
 *
 * @example
 * ```ts
 * const result = likelihoodRatioTest(-120, -115, 3, 5);
 * console.log(result.statistic); // 10
 * console.log(result.rejected);  // true if p < 0.05
 * ```
 */
export function likelihoodRatioTest(
  logLikRestricted: number,
  logLikFull: number,
  dfRestricted: number,
  dfFull: number,
  alpha = 0.05,
): LRTResult {
  if (!Number.isFinite(logLikRestricted) || !Number.isFinite(logLikFull)) {
    throw new Error(`Invalid parameters 'logLikRestricted', 'logLikFull': expected finite numbers, received logLikRestricted=${logLikRestricted}, logLikFull=${logLikFull}`);
  }
  if (dfFull <= dfRestricted) {
    throw new Error(
      `Invalid parameters 'dfRestricted', 'dfFull': expected dfFull > dfRestricted, received dfRestricted=${dfRestricted}, dfFull=${dfFull}`,
    );
  }

  const df = dfFull - dfRestricted;
  const statistic = -2 * (logLikRestricted - logLikFull);

  // Clamp at zero (numerical rounding can yield tiny negatives)
  const D = Math.max(0, statistic);

  const chi2 = new ChiSquared(df);
  const pValue = 1 - chi2.cdf(D);

  return {
    statistic: D,
    degreesOfFreedom: df,
    pValue,
    rejected: pValue < alpha,
  };
}

// ── Vuong test ────────────────────────────────────────────────────────────

/**
 * Result of a Vuong test comparing two non-nested models.
 */
export interface VuongResult {
  /** Vuong test statistic (z-score). Positive favours model 1, negative favours model 2. */
  statistic: number;
  /** Two-sided p-value. */
  pValue: number;
  /**
   * Interpretation of which model is preferred:
   * - `"model1"` -- Model 1 fits significantly better.
   * - `"model2"` -- Model 2 fits significantly better.
   * - `"indistinguishable"` -- Neither model is significantly better at the given alpha.
   */
  preferred: "model1" | "model2" | "indistinguishable";
}

/**
 * Vuong test for comparing non-nested models.
 *
 * Given pointwise log-likelihood values from two models fitted on the
 * same data, tests whether one model is closer to the true data-generating
 * process than the other.
 *
 * The test statistic is V = (sqrt(n) * mean(m)) / sd(m), where
 * m_i = logLik1[i] - logLik2[i]. Under H0 (models are equally close
 * to the truth), V ~ N(0, 1).
 *
 * An optional Schwarz correction adjusts for different parameter counts:
 * the mean of m_i is shifted by -(k1 - k2) * ln(n) / (2n).
 *
 * @param logLik1 - Pointwise log-likelihoods from model 1 (length n)
 * @param logLik2 - Pointwise log-likelihoods from model 2 (length n)
 * @param options - Test configuration
 * @param options.k1 - Number of parameters in model 1 (for Schwarz correction)
 * @param options.k2 - Number of parameters in model 2 (for Schwarz correction)
 * @param options.alpha - Significance level (default: 0.05)
 * @returns A {@link VuongResult} with z-statistic, p-value, and preferred model
 * @throws {Error} If logLik1 and logLik2 have different lengths
 * @throws {Error} If fewer than 2 observations
 *
 * @example
 * ```ts
 * const result = vuongTest(logLiks1, logLiks2, { k1: 3, k2: 5 });
 * console.log(result.preferred); // "model1", "model2", or "indistinguishable"
 * ```
 */
export function vuongTest(
  logLik1: number[],
  logLik2: number[],
  options: {
    k1?: number;
    k2?: number;
    alpha?: number;
  } = {},
): VuongResult {
  const { alpha = 0.05 } = options;
  const n = logLik1.length;

  if (n !== logLik2.length) {
    throw new Error(
      `Invalid parameters 'logLik1', 'logLik2': expected same length, received logLik1.length=${n}, logLik2.length=${logLik2.length}`,
    );
  }
  if (n < 2) {
    throw new Error(`Invalid parameter 'logLik1': expected at least 2 observations, received ${n}`);
  }

  // Pointwise differences
  const m = new Array<number>(n);
  let sumM = 0;
  for (let i = 0; i < n; i++) {
    m[i] = logLik1[i] - logLik2[i];
    sumM += m[i];
  }

  let meanM = sumM / n;

  // Apply Schwarz correction if parameter counts provided
  if (options.k1 !== undefined && options.k2 !== undefined) {
    const correction =
      ((options.k1 - options.k2) * Math.log(n)) / (2 * n);
    meanM -= correction;
  }

  // Sample standard deviation of m
  let sumSq = 0;
  for (let i = 0; i < n; i++) {
    sumSq += (m[i] - sumM / n) ** 2;
  }
  const sdM = Math.sqrt(sumSq / (n - 1));

  if (sdM < 1e-15) {
    if (Math.abs(meanM) < 1e-15) {
      // Models are identical in fit
      return {
        statistic: 0,
        pValue: 1,
        preferred: "indistinguishable",
      };
    }
    // Zero variance but non-zero mean: one model is deterministically better
    const sign = meanM > 0 ? 1 : -1;
    return {
      statistic: sign * Infinity,
      pValue: 0,
      preferred: sign > 0 ? "model1" : "model2",
    };
  }

  const statistic = (Math.sqrt(n) * meanM) / sdM;
  const pValue = 2 * (1 - normalCdf(Math.abs(statistic)));

  let preferred: VuongResult["preferred"];
  if (pValue < alpha) {
    preferred = statistic > 0 ? "model1" : "model2";
  } else {
    preferred = "indistinguishable";
  }

  return { statistic, pValue, preferred };
}

// ── Model comparison helpers ──────────────────────────────────────────────

/**
 * A single entry in a model comparison table.
 *
 * Includes information criteria, delta-AIC from the best model,
 * and the Akaike weight (evidence ratio).
 */
export interface ModelComparisonEntry {
  /** Model name or label. */
  name: string;
  /** Maximised log-likelihood. */
  logLikelihood: number;
  /** Number of estimated parameters. */
  k: number;
  /** Akaike Information Criterion. */
  aic: number;
  /** Corrected AIC for small samples. */
  aicc: number;
  /** Bayesian Information Criterion. */
  bic: number;
  /** Difference in AIC from the best (lowest-AIC) model. */
  deltaAIC: number;
  /** Akaike weight: exp(-0.5*deltaAIC) / sum(exp(-0.5*deltaAIC)). Weights sum to 1. */
  weight: number;
}

/**
 * Compare multiple models by information criteria.
 *
 * Returns a table sorted by AIC with Akaike weights (evidence ratios),
 * making it easy to see how much support each model has relative to
 * the best model. The weight represents the probability that a given
 * model is the best approximating model in the set.
 *
 * @param models - Array of model descriptors with name, logLikelihood, and k (parameter count)
 * @param n - Number of observations (shared across all models)
 * @returns Array of {@link ModelComparisonEntry} sorted by AIC (best first)
 * @throws {Error} If no models are provided or n < 1
 *
 * @example
 * ```ts
 * const table = compareModels([
 *   { name: "linear", logLikelihood: -120, k: 2 },
 *   { name: "quadratic", logLikelihood: -115, k: 3 },
 * ], 100);
 * console.log(table[0].name);   // best model by AIC
 * console.log(table[0].weight); // Akaike weight
 * ```
 */
export function compareModels(
  models: Array<{ name: string; logLikelihood: number; k: number }>,
  n: number,
): ModelComparisonEntry[] {
  if (models.length === 0) throw new Error(`Invalid parameter 'models': expected at least one model, received length 0`);
  if (n < 1) throw new Error(`Invalid parameter 'n': expected at least 1, received ${n}`);

  const entries: ModelComparisonEntry[] = models.map((m) => {
    const ic = informationCriteria(m.logLikelihood, m.k, n);
    return {
      name: m.name,
      logLikelihood: m.logLikelihood,
      k: m.k,
      aic: ic.aic,
      aicc: ic.aicc,
      bic: ic.bic,
      deltaAIC: 0,
      weight: 0,
    };
  });

  // Sort by AIC
  entries.sort((a, b) => a.aic - b.aic);
  const bestAIC = entries[0].aic;

  // Compute delta AIC and Akaike weights
  let sumExp = 0;
  for (const e of entries) {
    e.deltaAIC = e.aic - bestAIC;
    sumExp += Math.exp(-0.5 * e.deltaAIC);
  }
  for (const e of entries) {
    e.weight = Math.exp(-0.5 * e.deltaAIC) / sumExp;
  }

  return entries;
}
