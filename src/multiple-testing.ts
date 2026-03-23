/**
 * Result of a multiple testing correction procedure.
 */
export interface MultipleTestingResult {
  /** Original p-values. */
  originalPValues: number[];
  /** Adjusted p-values. */
  adjustedPValues: number[];
  /** Which hypotheses are rejected at the given alpha. */
  rejected: boolean[];
  /** The correction method used. */
  method: string;
  /** Significance level. */
  alpha: number;
}

/**
 * Bonferroni correction for multiple comparisons.
 *
 * Multiplies each p-value by the number of tests. The most conservative
 * method, controlling the family-wise error rate (FWER).
 *
 * @param pValues - Array of p-values from individual tests
 * @param alpha - Significance level (default: 0.05)
 * @returns MultipleTestingResult with adjusted p-values and rejection decisions
 * @throws Error if p-values array is empty, contains NaN, or values outside [0, 1]
 * @throws Error if alpha is not in (0, 1)
 *
 * @example
 * ```ts
 * const result = bonferroni([0.01, 0.04, 0.03, 0.005], 0.05);
 * // result.adjustedPValues ≈ [0.04, 0.16, 0.12, 0.02]
 * // result.rejected ≈ [true, false, false, true]
 * ```
 */
export function bonferroni(pValues: number[], alpha = 0.05): MultipleTestingResult {
  validatePValues(pValues);
  validateAlpha(alpha);
  const m = pValues.length;
  const adjustedPValues = pValues.map((p) => Math.min(p * m, 1));
  return {
    originalPValues: [...pValues],
    adjustedPValues,
    rejected: adjustedPValues.map((p) => p < alpha),
    method: "Bonferroni",
    alpha,
  };
}

/**
 * Sidak correction for multiple comparisons.
 *
 * Uses the formula: p_adj = 1 - (1 - p)^m instead of p * m.
 * Slightly less conservative than Bonferroni when tests are independent.
 *
 * @param pValues - Array of p-values from individual tests
 * @param alpha - Significance level (default: 0.05)
 * @returns MultipleTestingResult with adjusted p-values and rejection decisions
 * @throws Error if p-values array is empty, contains NaN, or values outside [0, 1]
 * @throws Error if alpha is not in (0, 1)
 *
 * @example
 * ```ts
 * const result = sidak([0.01, 0.04, 0.03, 0.005], 0.05);
 * // result.adjustedPValues — each adjusted via 1 - (1 - p)^4
 * // result.rejected — which hypotheses to reject
 * ```
 */
export function sidak(pValues: number[], alpha = 0.05): MultipleTestingResult {
  validatePValues(pValues);
  validateAlpha(alpha);
  const m = pValues.length;
  const adjustedPValues = pValues.map((p) => Math.min(1 - Math.pow(1 - p, m), 1));
  return {
    originalPValues: [...pValues],
    adjustedPValues,
    rejected: adjustedPValues.map((p) => p < alpha),
    method: "Šidák",
    alpha,
  };
}

/**
 * Holm-Bonferroni step-down procedure.
 *
 * Sequentially rejects hypotheses from smallest to largest p-value,
 * adjusting thresholds. Less conservative than Bonferroni while still
 * controlling FWER.
 *
 * @param pValues - Array of p-values from individual tests
 * @param alpha - Significance level (default: 0.05)
 * @returns MultipleTestingResult with adjusted p-values and rejection decisions
 * @throws Error if p-values array is empty, contains NaN, or values outside [0, 1]
 * @throws Error if alpha is not in (0, 1)
 *
 * @example
 * ```ts
 * const result = holm([0.01, 0.04, 0.03, 0.005], 0.05);
 * // Step-down adjusted p-values with FWER control
 * ```
 */
export function holm(pValues: number[], alpha = 0.05): MultipleTestingResult {
  validatePValues(pValues);
  validateAlpha(alpha);
  const m = pValues.length;

  // Create indexed array and sort by p-value
  const indexed = pValues.map((p, i) => ({ p, i }));
  indexed.sort((a, b) => a.p - b.p);

  const adjustedPValues = new Array<number>(m);

  // Step-down: enforce monotonicity (adjusted p-values must be non-decreasing)
  let runningMax = 0;
  for (let k = 0; k < m; k++) {
    const adjusted = indexed[k].p * (m - k);
    runningMax = Math.max(runningMax, adjusted);
    adjustedPValues[indexed[k].i] = Math.min(runningMax, 1);
  }

  return {
    originalPValues: [...pValues],
    adjustedPValues,
    rejected: adjustedPValues.map((p) => p < alpha),
    method: "Holm-Bonferroni",
    alpha,
  };
}

/**
 * Hochberg step-up procedure.
 *
 * Similar to Holm but works in reverse (step-up). Less conservative than
 * Holm when tests are independent, but requires the assumption of
 * independence or positive regression dependency.
 *
 * @param pValues - Array of p-values from individual tests
 * @param alpha - Significance level (default: 0.05)
 * @returns MultipleTestingResult with adjusted p-values and rejection decisions
 * @throws Error if p-values array is empty, contains NaN, or values outside [0, 1]
 * @throws Error if alpha is not in (0, 1)
 *
 * @example
 * ```ts
 * const result = hochberg([0.01, 0.04, 0.03, 0.005], 0.05);
 * // Step-up adjusted p-values
 * ```
 */
export function hochberg(pValues: number[], alpha = 0.05): MultipleTestingResult {
  validatePValues(pValues);
  validateAlpha(alpha);
  const m = pValues.length;

  const indexed = pValues.map((p, i) => ({ p, i }));
  indexed.sort((a, b) => a.p - b.p);

  const adjustedPValues = new Array<number>(m);

  // Step-up: enforce monotonicity from the top down
  let runningMin = 1;
  for (let k = m - 1; k >= 0; k--) {
    const adjusted = indexed[k].p * (m - k);
    runningMin = Math.min(runningMin, adjusted);
    adjustedPValues[indexed[k].i] = Math.min(runningMin, 1);
  }

  return {
    originalPValues: [...pValues],
    adjustedPValues,
    rejected: adjustedPValues.map((p) => p < alpha),
    method: "Hochberg",
    alpha,
  };
}

/**
 * Benjamini-Hochberg procedure for controlling the false discovery rate (FDR).
 *
 * Controls the expected proportion of false positives among rejected
 * hypotheses. Less conservative than FWER-controlling methods when many
 * tests are performed. Adjusted p-value for rank k: p_adj(k) = p(k) * m / k,
 * enforced to be non-increasing from the largest rank downward.
 *
 * @param pValues - Array of p-values from individual tests
 * @param alpha - Target FDR level (default: 0.05)
 * @returns MultipleTestingResult with adjusted p-values and rejection decisions
 * @throws Error if p-values array is empty, contains NaN, or values outside [0, 1]
 * @throws Error if alpha is not in (0, 1)
 *
 * @example
 * ```ts
 * const result = benjaminiHochberg([0.01, 0.04, 0.03, 0.005], 0.05);
 * // result.adjustedPValues — FDR-adjusted p-values
 * // result.rejected — which hypotheses to reject at FDR = 0.05
 * ```
 */
export function benjaminiHochberg(pValues: number[], alpha = 0.05): MultipleTestingResult {
  validatePValues(pValues);
  validateAlpha(alpha);
  const m = pValues.length;

  const indexed = pValues.map((p, i) => ({ p, i }));
  indexed.sort((a, b) => a.p - b.p);

  const adjustedPValues = new Array<number>(m);

  // Step-up: start from the largest p-value
  let runningMin = 1;
  for (let k = m - 1; k >= 0; k--) {
    const rank = k + 1;
    const adjusted = (indexed[k].p * m) / rank;
    runningMin = Math.min(runningMin, adjusted);
    adjustedPValues[indexed[k].i] = Math.min(runningMin, 1);
  }

  return {
    originalPValues: [...pValues],
    adjustedPValues,
    rejected: adjustedPValues.map((p) => p < alpha),
    method: "Benjamini-Hochberg",
    alpha,
  };
}

/**
 * Benjamini-Yekutieli procedure for controlling FDR under arbitrary dependence.
 *
 * More conservative than Benjamini-Hochberg but valid regardless of the
 * dependence structure among test statistics. Uses the correction factor
 * c(m) = sum(1/i, i=1..m), the m-th harmonic number.
 *
 * @param pValues - Array of p-values from individual tests
 * @param alpha - Target FDR level (default: 0.05)
 * @returns MultipleTestingResult with adjusted p-values and rejection decisions
 * @throws Error if p-values array is empty, contains NaN, or values outside [0, 1]
 * @throws Error if alpha is not in (0, 1)
 *
 * @example
 * ```ts
 * const result = benjaminiYekutieli([0.01, 0.04, 0.03, 0.005], 0.05);
 * // More conservative than Benjamini-Hochberg, safe for dependent tests
 * ```
 */
export function benjaminiYekutieli(pValues: number[], alpha = 0.05): MultipleTestingResult {
  validatePValues(pValues);
  validateAlpha(alpha);
  const m = pValues.length;

  // Harmonic number c(m) = sum(1/i for i=1..m)
  let cm = 0;
  for (let i = 1; i <= m; i++) cm += 1 / i;

  const indexed = pValues.map((p, i) => ({ p, i }));
  indexed.sort((a, b) => a.p - b.p);

  const adjustedPValues = new Array<number>(m);

  let runningMin = 1;
  for (let k = m - 1; k >= 0; k--) {
    const rank = k + 1;
    const adjusted = (indexed[k].p * m * cm) / rank;
    runningMin = Math.min(runningMin, adjusted);
    adjustedPValues[indexed[k].i] = Math.min(runningMin, 1);
  }

  return {
    originalPValues: [...pValues],
    adjustedPValues,
    rejected: adjustedPValues.map((p) => p < alpha),
    method: "Benjamini-Yekutieli",
    alpha,
  };
}

function validateAlpha(alpha: number): void {
  if (alpha <= 0 || alpha >= 1 || Number.isNaN(alpha)) {
    throw new Error(`alpha must be between 0 and 1 (exclusive), got ${alpha}`);
  }
}

function validatePValues(pValues: number[]): void {
  if (pValues.length === 0) {
    throw new Error("p-values array must not be empty");
  }
  for (const p of pValues) {
    if (Number.isNaN(p)) {
      throw new Error("p-values must not contain NaN");
    }
    if (p < 0 || p > 1) {
      throw new Error(`p-values must be between 0 and 1, got ${p}`);
    }
  }
}
