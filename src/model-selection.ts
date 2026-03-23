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

export interface InformationCriteria {
  /** Akaike Information Criterion: −2ℓ + 2k. */
  aic: number;
  /** Corrected AIC for small samples: AIC + 2k(k+1)/(n−k−1). */
  aicc: number;
  /** Bayesian Information Criterion: −2ℓ + k ln(n). */
  bic: number;
}

/**
 * Compute AIC, AICc, and BIC for a fitted model.
 *
 * @param logLikelihood  Maximised log-likelihood ℓ.
 * @param k  Number of estimated parameters (including intercept).
 * @param n  Number of observations.
 */
export function informationCriteria(
  logLikelihood: number,
  k: number,
  n: number,
): InformationCriteria {
  if (k < 1) throw new Error("k must be at least 1");
  if (n < 1) throw new Error("n must be at least 1");
  if (!Number.isFinite(logLikelihood)) {
    throw new Error("logLikelihood must be finite");
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

export interface LRTResult {
  /** Test statistic: −2(ℓ_restricted − ℓ_full). */
  statistic: number;
  /** Degrees of freedom (difference in parameter counts). */
  degreesOfFreedom: number;
  /** p-value from chi-squared distribution. */
  pValue: number;
  /** Whether the null (restricted model is adequate) is rejected at α = 0.05. */
  rejected: boolean;
}

/**
 * Likelihood ratio test for nested models.
 *
 * Tests H₀: the restricted (simpler) model is adequate
 * against H₁: the full model fits significantly better.
 *
 * The test statistic D = −2(ℓ_restricted − ℓ_full) follows a χ²
 * distribution with df = k_full − k_restricted under H₀.
 *
 * @param logLikRestricted  Log-likelihood of the restricted (null) model.
 * @param logLikFull  Log-likelihood of the full (alternative) model.
 * @param dfRestricted  Number of parameters in the restricted model.
 * @param dfFull  Number of parameters in the full model.
 * @param alpha  Significance level (default 0.05).
 */
export function likelihoodRatioTest(
  logLikRestricted: number,
  logLikFull: number,
  dfRestricted: number,
  dfFull: number,
  alpha = 0.05,
): LRTResult {
  if (!Number.isFinite(logLikRestricted) || !Number.isFinite(logLikFull)) {
    throw new Error("Log-likelihood values must be finite");
  }
  if (dfFull <= dfRestricted) {
    throw new Error(
      "Full model must have more parameters than restricted model",
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

export interface VuongResult {
  /** Vuong test statistic (z-score). */
  statistic: number;
  /** Two-sided p-value. */
  pValue: number;
  /**
   * Interpretation:
   * - `"model1"` — Model 1 fits significantly better.
   * - `"model2"` — Model 2 fits significantly better.
   * - `"indistinguishable"` — Neither model is significantly better.
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
 * The test statistic is  V = (1/√n) Σ mᵢ / s_m,  where
 * mᵢ = log f₁(yᵢ | θ̂₁) − log f₂(yᵢ | θ̂₂)  and  s_m is the sample
 * standard deviation of the mᵢ.  Under H₀ (models are equally close
 * to the truth), V ~ N(0, 1).
 *
 * An optional Schwarz correction adjusts for different numbers of
 * parameters: the mean of mᵢ is shifted by −(k₁ − k₂) ln(n) / (2n).
 *
 * @param logLik1  Pointwise log-likelihoods from model 1 (length n).
 * @param logLik2  Pointwise log-likelihoods from model 2 (length n).
 * @param options.k1  Number of parameters in model 1 (for Schwarz correction).
 * @param options.k2  Number of parameters in model 2 (for Schwarz correction).
 * @param options.alpha  Significance level (default 0.05).
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
      "logLik1 and logLik2 must have the same length",
    );
  }
  if (n < 2) {
    throw new Error("Need at least 2 observations");
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

export interface ModelComparisonEntry {
  /** Model name/label. */
  name: string;
  /** Log-likelihood. */
  logLikelihood: number;
  /** Number of parameters. */
  k: number;
  /** AIC. */
  aic: number;
  /** AICc. */
  aicc: number;
  /** BIC. */
  bic: number;
  /** Δ AIC from best model. */
  deltaAIC: number;
  /** Akaike weight (evidence ratio). */
  weight: number;
}

/**
 * Compare multiple models by information criteria.
 *
 * Returns a table sorted by AIC with Akaike weights (evidence ratios),
 * making it easy to see how much support each model has relative to
 * the best model.
 *
 * @param models  Array of { name, logLikelihood, k }.
 * @param n  Number of observations (shared across all models).
 */
export function compareModels(
  models: Array<{ name: string; logLikelihood: number; k: number }>,
  n: number,
): ModelComparisonEntry[] {
  if (models.length === 0) throw new Error("Must provide at least one model");
  if (n < 1) throw new Error("n must be at least 1");

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
