/**
 * Survival analysis: Kaplan-Meier estimator, log-rank test, and Nelson-Aalen estimator.
 */

/** A single survival observation. */
export interface SurvivalObservation {
  /** Time to event or censoring */
  time: number;
  /** Whether the event occurred (true) or was censored (false) */
  event: boolean;
}

/** A point on the Kaplan-Meier survival curve. */
export interface SurvivalPoint {
  time: number;
  survival: number;
  standardError: number;
  lower: number;
  upper: number;
  nRisk: number;
  nEvents: number;
  nCensored: number;
}

/** Result of a Kaplan-Meier estimation. */
export interface KaplanMeierResult {
  /** Survival curve points */
  curve: SurvivalPoint[];
  /** Median survival time (NaN if not reached) */
  medianSurvival: number;
  /** Number of observations */
  n: number;
  /** Number of events */
  nEvents: number;
  /** Number of censored */
  nCensored: number;
  /** Evaluate S(t) at any time t */
  survivalAt: (t: number) => number;
}

/** Result of a log-rank test comparing two survival curves. */
export interface LogRankResult {
  chiSquared: number;
  pValue: number;
  degreesOfFreedom: number;
  rejected: boolean;
}

/** A point on the Nelson-Aalen cumulative hazard curve. */
export interface CumulativeHazardPoint {
  time: number;
  hazard: number;
  standardError: number;
  nRisk: number;
  nEvents: number;
}

/** Result of a Nelson-Aalen estimation. */
export interface NelsonAalenResult {
  curve: CumulativeHazardPoint[];
  n: number;
  nEvents: number;
  hazardAt: (t: number) => number;
}

/**
 * Kaplan-Meier survival estimator.
 *
 * Computes the non-parametric survival function from time-to-event data
 * with right censoring. Confidence intervals use Greenwood's formula
 * with log transformation.
 *
 * @param observations - Array of {time, event} observations
 * @param confidence - Confidence level (default 0.95)
 */
export function kaplanMeier(
  observations: SurvivalObservation[],
  confidence = 0.95,
): KaplanMeierResult {
  if (observations.length === 0) {
    throw new Error("Need at least 1 observation");
  }
  if (confidence <= 0 || confidence >= 1) {
    throw new Error("Confidence level must be between 0 and 1");
  }

  // Sort by time, events before censored at the same time
  const sorted = [...observations].sort((a, b) => {
    if (a.time !== b.time) return a.time - b.time;
    return a.event === b.event ? 0 : a.event ? -1 : 1;
  });

  const n = sorted.length;
  const totalEvents = sorted.filter((o) => o.event).length;
  const totalCensored = n - totalEvents;

  // z-value for CI
  const z = normalQuantile(1 - (1 - confidence) / 2);

  // Aggregate events at each unique time
  const times: {
    time: number;
    events: number;
    censored: number;
  }[] = [];

  let i = 0;
  while (i < sorted.length) {
    const t = sorted[i].time;
    let events = 0;
    let censored = 0;
    while (i < sorted.length && sorted[i].time === t) {
      if (sorted[i].event) events++;
      else censored++;
      i++;
    }
    times.push({ time: t, events, censored });
  }

  // Compute survival function
  const curve: SurvivalPoint[] = [];
  let nRisk = n;
  let survival = 1.0;
  let greenwoodSum = 0; // Greenwood's formula accumulator

  for (const { time, events, censored } of times) {
    if (events > 0) {
      survival *= 1 - events / nRisk;

      if (survival > 0 && nRisk > events) {
        greenwoodSum += events / (nRisk * (nRisk - events));
      }

      // SE via Greenwood's formula
      const se = survival * Math.sqrt(greenwoodSum);

      // Log-transformed CI for better coverage
      let lower: number;
      let upper: number;
      if (survival > 0 && survival < 1) {
        const logS = Math.log(survival);
        const logSE = se / (survival * Math.abs(logS));
        lower = Math.max(0, Math.exp(logS * Math.exp(z * logSE)));
        upper = Math.min(1, Math.exp(logS * Math.exp(-z * logSE)));
      } else {
        lower = Math.max(0, survival - z * se);
        upper = Math.min(1, survival + z * se);
      }

      curve.push({
        time,
        survival,
        standardError: se,
        lower,
        upper,
        nRisk,
        nEvents: events,
        nCensored: censored,
      });
    }

    nRisk -= events + censored;
  }

  // Median survival: smallest time where S(t) <= 0.5
  let medianSurvival = NaN;
  for (const point of curve) {
    if (point.survival <= 0.5) {
      medianSurvival = point.time;
      break;
    }
  }

  // S(t) evaluator
  const survivalAt = (t: number): number => {
    if (t < 0) return 1;
    if (curve.length === 0) return 1;

    let s = 1.0;
    for (const point of curve) {
      if (point.time > t) break;
      s = point.survival;
    }
    return s;
  };

  return {
    curve,
    medianSurvival,
    n,
    nEvents: totalEvents,
    nCensored: totalCensored,
    survivalAt,
  };
}

/**
 * Nelson-Aalen estimator of the cumulative hazard function.
 *
 * H(t) = sum_{t_i <= t} d_i / n_i
 *
 * @param observations - Array of {time, event} observations
 */
export function nelsonAalen(
  observations: SurvivalObservation[],
): NelsonAalenResult {
  if (observations.length === 0) {
    throw new Error("Need at least 1 observation");
  }

  const sorted = [...observations].sort((a, b) => {
    if (a.time !== b.time) return a.time - b.time;
    return a.event === b.event ? 0 : a.event ? -1 : 1;
  });

  const n = sorted.length;
  const totalEvents = sorted.filter((o) => o.event).length;

  // Aggregate at unique event times
  const times: { time: number; events: number; censored: number }[] = [];
  let idx = 0;
  while (idx < sorted.length) {
    const t = sorted[idx].time;
    let events = 0;
    let censored = 0;
    while (idx < sorted.length && sorted[idx].time === t) {
      if (sorted[idx].event) events++;
      else censored++;
      idx++;
    }
    times.push({ time: t, events, censored });
  }

  const curve: CumulativeHazardPoint[] = [];
  let nRisk = n;
  let cumHazard = 0;
  let varSum = 0;

  for (const { time, events, censored } of times) {
    if (events > 0) {
      cumHazard += events / nRisk;
      varSum += events / (nRisk * nRisk);

      curve.push({
        time,
        hazard: cumHazard,
        standardError: Math.sqrt(varSum),
        nRisk,
        nEvents: events,
      });
    }
    nRisk -= events + censored;
  }

  const hazardAt = (t: number): number => {
    if (t < 0) return 0;
    let h = 0;
    for (const point of curve) {
      if (point.time > t) break;
      h = point.hazard;
    }
    return h;
  };

  return { curve, n, nEvents: totalEvents, hazardAt };
}

/**
 * Log-rank test for comparing two survival curves.
 *
 * Tests H0: the two groups have identical survival functions.
 *
 * @param group1 - Survival data for group 1
 * @param group2 - Survival data for group 2
 * @param alpha - Significance level
 */
export function logRankTest(
  group1: SurvivalObservation[],
  group2: SurvivalObservation[],
  alpha = 0.05,
): LogRankResult {
  if (group1.length === 0 || group2.length === 0) {
    throw new Error("Both groups must have at least 1 observation");
  }

  // Collect all unique event times across both groups
  const allTimes = new Set<number>();
  for (const o of group1) {
    if (o.event) allTimes.add(o.time);
  }
  for (const o of group2) {
    if (o.event) allTimes.add(o.time);
  }

  const sortedTimes = [...allTimes].sort((a, b) => a - b);

  // Sort observations
  const s1 = [...group1].sort((a, b) => a.time - b.time);
  const s2 = [...group2].sort((a, b) => a.time - b.time);

  // Count at-risk and events at each time
  let n1 = s1.length;
  let n2 = s2.length;
  let idx1 = 0;
  let idx2 = 0;

  let sumOE = 0; // sum of (observed - expected) for group 1
  let sumVar = 0;

  for (const t of sortedTimes) {
    // Remove those censored or events before this time
    while (idx1 < s1.length && s1[idx1].time < t) {
      n1--;
      idx1++;
    }
    while (idx2 < s2.length && s2[idx2].time < t) {
      n2--;
      idx2++;
    }

    // Count events at this time
    let d1 = 0;
    let tempIdx1 = idx1;
    while (tempIdx1 < s1.length && s1[tempIdx1].time === t) {
      if (s1[tempIdx1].event) d1++;
      tempIdx1++;
    }

    let d2 = 0;
    let tempIdx2 = idx2;
    while (tempIdx2 < s2.length && s2[tempIdx2].time === t) {
      if (s2[tempIdx2].event) d2++;
      tempIdx2++;
    }

    const nTotal = n1 + n2;
    const dTotal = d1 + d2;

    if (nTotal > 1 && dTotal > 0) {
      const expected1 = (n1 * dTotal) / nTotal;
      sumOE += d1 - expected1;

      const v =
        (n1 * n2 * dTotal * (nTotal - dTotal)) / (nTotal * nTotal * (nTotal - 1));
      sumVar += v;
    }

    // Now advance past this time's observations
    while (idx1 < s1.length && s1[idx1].time === t) {
      n1--;
      idx1++;
    }
    while (idx2 < s2.length && s2[idx2].time === t) {
      n2--;
      idx2++;
    }
  }

  const chiSquared = sumVar > 0 ? (sumOE * sumOE) / sumVar : 0;

  // Chi-squared with 1 df
  const pValue = 1 - chiSquaredCdf1(chiSquared);

  return {
    chiSquared,
    pValue,
    degreesOfFreedom: 1,
    rejected: pValue < alpha,
  };
}

// ---- Helpers ----

/**
 * CDF of chi-squared with 1 df: P(X <= x) = erf(sqrt(x/2))
 */
function chiSquaredCdf1(x: number): number {
  if (x <= 0) return 0;
  return erf(Math.sqrt(x / 2));
}

/**
 * Error function approximation (Abramowitz and Stegun 7.1.26).
 */
function erf(x: number): number {
  const sign = x >= 0 ? 1 : -1;
  x = Math.abs(x);

  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;

  const t = 1 / (1 + p * x);
  const y = 1 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return sign * y;
}

/**
 * Standard normal quantile (rational approximation).
 */
function normalQuantile(p: number): number {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  if (p === 0.5) return 0;

  if (p < 0.5) return -normalQuantile(1 - p);

  // Rational approximation for upper half
  const t = Math.sqrt(-2 * Math.log(1 - p));
  const c0 = 2.515517;
  const c1 = 0.802853;
  const c2 = 0.010328;
  const d1 = 1.432788;
  const d2 = 0.189269;
  const d3 = 0.001308;

  return t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t);
}
