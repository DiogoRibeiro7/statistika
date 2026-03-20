import { Dataset, KSTestResult } from "../types";

/**
 * Kolmogorov-Smirnov survival function approximation.
 * Computes P(D_n >= d) using the asymptotic formula.
 */
function ksSurvival(d: number, n: number): number {
  // Effective sample size adjusted statistic
  const lambda = (Math.sqrt(n) + 0.12 + 0.11 / Math.sqrt(n)) * d;
  // Asymptotic series (Marsaglia et al. 2003)
  let sum = 0;
  for (let k = 1; k <= 100; k++) {
    const term = (-1) ** (k - 1) * Math.exp(-2 * k * k * lambda * lambda);
    sum += term;
    if (Math.abs(term) < 1e-14) break;
  }
  return Math.min(1, Math.max(0, 2 * sum));
}

/**
 * Two-sample Kolmogorov-Smirnov survival function.
 */
function ksSurvival2(d: number, n1: number, n2: number): number {
  const ne = (n1 * n2) / (n1 + n2);
  const lambda = (Math.sqrt(ne) + 0.12 + 0.11 / Math.sqrt(ne)) * d;
  let sum = 0;
  for (let k = 1; k <= 100; k++) {
    const term = (-1) ** (k - 1) * Math.exp(-2 * k * k * lambda * lambda);
    sum += term;
    if (Math.abs(term) < 1e-14) break;
  }
  return Math.min(1, Math.max(0, 2 * sum));
}

/**
 * One-sample Kolmogorov-Smirnov test.
 * Tests whether data follows a given continuous CDF.
 */
export function ksTest(
  data: Dataset,
  cdf: (x: number) => number,
  alpha: number = 0.05,
): KSTestResult {
  if (data.length < 1) throw new Error("Need at least 1 observation");
  const n = data.length;
  const sorted = [...data].sort((a, b) => a - b);

  let dMax = 0;
  for (let i = 0; i < n; i++) {
    const fx = cdf(sorted[i]);
    // D+ = max(i/n - F(x_i))
    const dPlus = (i + 1) / n - fx;
    // D- = max(F(x_i) - (i-1)/n)
    const dMinus = fx - i / n;
    dMax = Math.max(dMax, dPlus, dMinus);
  }

  const pValue = ksSurvival(dMax, n);
  return { statistic: dMax, pValue, rejected: pValue < alpha };
}

/**
 * Two-sample Kolmogorov-Smirnov test.
 * Tests whether two samples come from the same distribution.
 */
export function ksTwoSampleTest(
  data1: Dataset,
  data2: Dataset,
  alpha: number = 0.05,
): KSTestResult {
  if (data1.length < 1 || data2.length < 1) throw new Error("Need at least 1 observation per sample");
  const n1 = data1.length;
  const n2 = data2.length;
  const sorted1 = [...data1].sort((a, b) => a - b);
  const sorted2 = [...data2].sort((a, b) => a - b);

  // Combine and sort all unique values, then compute ECDFs
  const all = [...new Set([...sorted1, ...sorted2])].sort((a, b) => a - b);

  let dMax = 0;
  for (const x of all) {
    // F1(x) = #{x_i <= x} / n1, F2(x) = #{y_j <= x} / n2
    let c1 = 0;
    let c2 = 0;
    for (let k = 0; k < n1; k++) {
      if (sorted1[k] <= x) c1++;
      else break;
    }
    for (let k = 0; k < n2; k++) {
      if (sorted2[k] <= x) c2++;
      else break;
    }
    const diff = Math.abs(c1 / n1 - c2 / n2);
    if (diff > dMax) dMax = diff;
  }

  const pValue = ksSurvival2(dMax, n1, n2);
  return { statistic: dMax, pValue, rejected: pValue < alpha };
}
