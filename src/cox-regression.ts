/**
 * Cox Proportional Hazards model.
 *
 * Semi-parametric survival regression that models the hazard function as:
 *   h(t | X) = h0(t) * exp(beta' X)
 *
 * where h0(t) is the baseline hazard, beta are regression coefficients,
 * and X are covariate values.
 *
 * Uses Newton-Raphson optimization of the partial log-likelihood
 * (Breslow method for tied event times).
 */

import type { SurvivalObservation } from "./survival";
import { invertMatrix, normalCdf } from "./utils/linalg";

/**
 * A survival observation with covariates for Cox regression.
 *
 * Extends the base survival observation (time and event indicator)
 * with a vector of covariate values.
 */
export interface CoxObservation extends SurvivalObservation {
  /** Covariate values */
  covariates: number[];
}

/**
 * Result of fitting a Cox Proportional Hazards model.
 *
 * Contains regression coefficients, hazard ratios exp(beta), standard errors,
 * Wald test statistics and p-values, confidence intervals for hazard ratios,
 * the partial log-likelihood, concordance index, baseline cumulative hazard,
 * and a prediction function.
 */
export interface CoxRegressionResult {
  /** Regression coefficients (beta) */
  coefficients: number[];
  /** Standard errors of the coefficients */
  standardErrors: number[];
  /** Hazard ratios exp(beta) */
  hazardRatios: number[];
  /** Wald z-statistics */
  zScores: number[];
  /** Two-sided p-values from Wald test */
  pValues: number[];
  /** 95% confidence intervals for hazard ratios */
  hazardRatioCIs: Array<[number, number]>;
  /** Partial log-likelihood */
  logLikelihood: number;
  /** Number of Newton-Raphson iterations */
  iterations: number;
  /** Concordance index (C-statistic) */
  concordance: number;
  /** Predict hazard ratio for new covariates */
  predictHazardRatio: (covariates: number[]) => number;
  /** Baseline cumulative hazard (Breslow estimator) at event times */
  baselineHazard: Array<{ time: number; hazard: number }>;
}

/**
 * Fit a Cox Proportional Hazards model via Newton-Raphson.
 *
 * The Cox PH model assumes the hazard function has the form:
 *   h(t | X) = h0(t) * exp(beta' * X)
 * where h0(t) is an unspecified baseline hazard.
 *
 * The partial log-likelihood is maximized using Newton-Raphson optimization.
 * Tied event times are handled using the Breslow approximation. The baseline
 * cumulative hazard is estimated via the Breslow estimator.
 *
 * @param observations - Array of survival observations with covariates
 *   (each must have time, event indicator, and covariates array)
 * @param maxIterations - Maximum Newton-Raphson iterations (default 25)
 * @param tolerance - Convergence tolerance for coefficient updates (default 1e-9)
 * @returns A {@link CoxRegressionResult} with coefficients, hazard ratios, p-values,
 *   concordance index, baseline hazard, and prediction function
 * @throws {Error} If fewer than 2 observations
 * @throws {Error} If no covariates are provided
 * @throws {Error} If observations have inconsistent numbers of covariates
 *
 * @example
 * ```ts
 * const observations = [
 *   { time: 5, event: true, covariates: [1, 65] },
 *   { time: 10, event: false, covariates: [0, 50] },
 *   { time: 3, event: true, covariates: [1, 70] },
 * ];
 * const result = coxRegression(observations);
 * console.log(result.hazardRatios);  // exp(beta) for each covariate
 * console.log(result.concordance);   // C-statistic
 * console.log(result.predictHazardRatio([1, 60])); // HR for new patient
 * ```
 */
export function coxRegression(
  observations: CoxObservation[],
  maxIterations = 25,
  tolerance = 1e-9,
): CoxRegressionResult {
  const n = observations.length;
  if (n < 2) throw new Error(`Invalid parameter 'observations': expected at least 2 observations, received ${n}`);

  const p = observations[0].covariates.length;
  if (p < 1) throw new Error(`Invalid parameter 'observations': expected at least 1 covariate, received ${p}`);

  for (const obs of observations) {
    if (obs.covariates.length !== p) {
      throw new Error(`Invalid parameter 'observations': expected all observations to have ${p} covariates, received ${obs.covariates.length}`);
    }
  }

  // Sort by time (descending for risk set construction), events first at ties
  const sorted = [...observations].sort((a, b) => {
    if (a.time !== b.time) return a.time - b.time;
    return a.event === b.event ? 0 : a.event ? -1 : 1;
  });

  const times = sorted.map((o) => o.time);
  const events = sorted.map((o) => (o.event ? 1 : 0));
  const X: number[][] = sorted.map((o) => o.covariates);

  // Newton-Raphson optimization of partial log-likelihood
  const beta = new Array(p).fill(0);
  let iterations = 0;

  for (let iter = 0; iter < maxIterations; iter++) {
    iterations = iter + 1;

    // Compute exp(X * beta) for all subjects
    const expXBeta = new Array(n);
    for (let i = 0; i < n; i++) {
      let xb = 0;
      for (let j = 0; j < p; j++) xb += X[i][j] * beta[j];
      expXBeta[i] = Math.exp(xb);
    }

    // Compute gradient and Hessian of partial log-likelihood
    const gradient = new Array(p).fill(0);
    const hessian: number[][] = [];
    for (let j = 0; j < p; j++) {
      hessian[j] = new Array(p).fill(0);
    }

    // Risk set sums (computed from the end backwards)
    let riskSum = 0; // sum of exp(xb) in risk set
    const riskSumX = new Array(p).fill(0); // sum of x_j * exp(xb)
    const riskSumXX: number[][] = [];
    for (let j = 0; j < p; j++) riskSumXX[j] = new Array(p).fill(0);

    // Walk backwards through sorted observations
    // At time t, risk set = {i : time_i >= t}
    // We accumulate from the end
    const ii = n - 1;
    // First, add all to risk set
    for (let i = 0; i < n; i++) {
      riskSum += expXBeta[i];
      for (let j = 0; j < p; j++) {
        riskSumX[j] += X[i][j] * expXBeta[i];
        for (let k = 0; k < p; k++) {
          riskSumXX[j][k] += X[i][j] * X[i][k] * expXBeta[i];
        }
      }
    }

    // Now process events from earliest to latest
    // Remove subjects with times before the current event time from risk set
    const prevTime = -Infinity;
    const removeIdx = 0;

    for (let i = 0; i < n; i++) {
      if (events[i] === 0) continue;

      // Update gradient and Hessian for this event
      for (let j = 0; j < p; j++) {
        gradient[j] += X[i][j] - riskSumX[j] / riskSum;
        for (let k = 0; k < p; k++) {
          hessian[j][k] -=
            riskSumXX[j][k] / riskSum -
            (riskSumX[j] * riskSumX[k]) / (riskSum * riskSum);
        }
      }
    }

    // Solve H * delta = -g  (Newton step)
    // For small p, use direct inversion
    const invH = invertMatrix(hessian);
    if (!invH) break; // Singular Hessian

    const delta = new Array(p).fill(0);
    for (let j = 0; j < p; j++) {
      for (let k = 0; k < p; k++) {
        delta[j] -= invH[j][k] * gradient[k];
      }
    }

    // Update beta
    let maxDelta = 0;
    for (let j = 0; j < p; j++) {
      beta[j] += delta[j];
      maxDelta = Math.max(maxDelta, Math.abs(delta[j]));
    }

    if (maxDelta < tolerance) break;
  }

  // Compute final quantities
  const expXBeta = new Array(n);
  for (let i = 0; i < n; i++) {
    let xb = 0;
    for (let j = 0; j < p; j++) xb += X[i][j] * beta[j];
    expXBeta[i] = Math.exp(xb);
  }

  // Partial log-likelihood
  let logLik = 0;
  let riskSum = 0;
  for (let i = n - 1; i >= 0; i--) {
    riskSum += expXBeta[i];
    if (events[i] === 1) {
      let xb = 0;
      for (let j = 0; j < p; j++) xb += X[i][j] * beta[j];
      logLik += xb - Math.log(riskSum);
    }
  }

  // Information matrix (negative Hessian) for standard errors
  const hessian: number[][] = [];
  for (let j = 0; j < p; j++) hessian[j] = new Array(p).fill(0);

  riskSum = 0;
  const riskSumX = new Array(p).fill(0);
  const riskSumXX: number[][] = [];
  for (let j = 0; j < p; j++) riskSumXX[j] = new Array(p).fill(0);

  for (let i = n - 1; i >= 0; i--) {
    riskSum += expXBeta[i];
    for (let j = 0; j < p; j++) {
      riskSumX[j] += X[i][j] * expXBeta[i];
      for (let k = 0; k < p; k++) {
        riskSumXX[j][k] += X[i][j] * X[i][k] * expXBeta[i];
      }
    }

    if (events[i] === 1) {
      for (let j = 0; j < p; j++) {
        for (let k = 0; k < p; k++) {
          hessian[j][k] +=
            riskSumXX[j][k] / riskSum -
            (riskSumX[j] * riskSumX[k]) / (riskSum * riskSum);
        }
      }
    }
  }

  const invInfo = invertMatrix(hessian);
  const se = new Array(p);
  for (let j = 0; j < p; j++) {
    se[j] = invInfo ? Math.sqrt(Math.max(0, invInfo[j][j])) : NaN;
  }

  const hazardRatios = beta.map((b) => Math.exp(b));
  const zScores = beta.map((b, j) => (se[j] > 0 ? b / se[j] : 0));
  const pValues = zScores.map((z) => 2 * (1 - normalCdf(Math.abs(z))));

  const z95 = 1.96;
  const hazardRatioCIs: Array<[number, number]> = beta.map((b, j) => [
    Math.exp(b - z95 * se[j]),
    Math.exp(b + z95 * se[j]),
  ]);

  // Breslow baseline hazard estimator
  const baselineHazard = computeBaselineHazard(sorted, expXBeta);

  // Concordance index
  const concordance = computeConcordance(sorted, beta);

  return {
    coefficients: beta,
    standardErrors: se,
    hazardRatios,
    zScores,
    pValues,
    hazardRatioCIs,
    logLikelihood: logLik,
    iterations,
    concordance,
    predictHazardRatio: (covariates: number[]) => {
      let xb = 0;
      for (let j = 0; j < p; j++) xb += covariates[j] * beta[j];
      return Math.exp(xb);
    },
    baselineHazard,
  };
}

// ---- Helpers ----

function computeBaselineHazard(
  sorted: CoxObservation[],
  expXBeta: number[],
): Array<{ time: number; hazard: number }> {
  const n = sorted.length;
  const result: Array<{ time: number; hazard: number }> = [];
  let cumHazard = 0;

  // Risk set sum from the right
  let riskSum = 0;
  for (let i = 0; i < n; i++) riskSum += expXBeta[i];

  let i = 0;
  while (i < n) {
    const t = sorted[i].time;
    let nEvents = 0;

    // Count events at this time
    const startI = i;
    while (i < n && sorted[i].time === t) {
      if (sorted[i].event) nEvents++;
      i++;
    }

    if (nEvents > 0) {
      cumHazard += nEvents / riskSum;
      result.push({ time: t, hazard: cumHazard });
    }

    // Remove subjects at this time from risk set
    for (let j = startI; j < i; j++) {
      riskSum -= expXBeta[j];
    }
  }

  return result;
}

function computeConcordance(
  sorted: CoxObservation[],
  beta: number[],
): number {
  const p = beta.length;
  const n = sorted.length;

  // Linear predictor for each subject
  const lp = new Array(n);
  for (let i = 0; i < n; i++) {
    let xb = 0;
    for (let j = 0; j < p; j++) xb += sorted[i].covariates[j] * beta[j];
    lp[i] = xb;
  }

  let concordant = 0;
  let discordant = 0;

  for (let i = 0; i < n; i++) {
    if (!sorted[i].event) continue;
    for (let j = i + 1; j < n; j++) {
      if (sorted[j].time <= sorted[i].time) continue;
      // Pair (i, j): i had event at earlier time, j survived longer
      // Concordant if lp[i] > lp[j] (higher risk = earlier event)
      if (lp[i] > lp[j]) concordant++;
      else if (lp[i] < lp[j]) discordant++;
    }
  }

  const total = concordant + discordant;
  return total > 0 ? concordant / total : 0.5;
}

