/**
 * State-Space Models and Kalman Filter.
 *
 * Provides Kalman filtering, RTS smoothing, EM parameter estimation,
 * and convenience constructors for local level and local linear trend models.
 */

// ── Interfaces ──────────────────────────────────────────────────────────

/**
 * Defines a linear Gaussian state-space model:
 *   x_t = F * x_{t-1} + w_t,   w_t ~ N(0, Q)
 *   y_t = H * x_t     + v_t,   v_t ~ N(0, R)
 */
export interface StateSpaceModel {
  /** State transition matrix (m x m) */
  F: number[][];
  /** Observation matrix (p x m) */
  H: number[][];
  /** State noise covariance (m x m) */
  Q: number[][];
  /** Observation noise covariance (p x p) */
  R: number[][];
  /** Initial state mean (m x 1) */
  x0: number[];
  /** Initial state covariance (m x m) */
  P0: number[][];
}

/**
 * Result of the Kalman filter forward pass.
 */
export interface KalmanFilterResult {
  /** Filtered state estimates [t][state] */
  states: number[][];
  /** Filtered covariance matrices [t][i][j] */
  covariances: number[][][];
  /** One-step-ahead observation predictions [t][obs] */
  predictions: number[][];
  /** Total log-likelihood of the observations */
  logLikelihood: number;
  /** Innovation (prediction error) vectors [t][obs] */
  innovations: number[][];
  /** Innovation covariance matrices [t][i][j] */
  innovationCovariances: number[][][];
}

/**
 * Result of the Kalman smoother (RTS), extending the filter result.
 */
export interface KalmanSmootherResult extends KalmanFilterResult {
  /** Smoothed state estimates [t][state] */
  smoothedStates: number[][];
  /** Smoothed covariance matrices [t][i][j] */
  smoothedCovariances: number[][][];
}

/**
 * Forecast result from state-space prediction.
 */
export interface StateSpacePrediction {
  /** Forecasted state means [h][state] */
  stateForecast: number[][];
  /** Forecasted observation means [h][obs] */
  observationForecast: number[][];
  /** Forecasted state covariances [h][i][j] */
  stateCovariances: number[][][];
  /** Forecasted observation covariances [h][i][j] */
  observationCovariances: number[][][];
  /** Lower 95% confidence bound [h][obs] */
  lowerBound: number[][];
  /** Upper 95% confidence bound [h][obs] */
  upperBound: number[][];
}

/**
 * Options for the EM algorithm.
 */
export interface StateSpaceEMOptions {
  /** Maximum number of EM iterations (default: 100) */
  maxIterations?: number;
  /** Convergence tolerance on log-likelihood change (default: 1e-6) */
  tolerance?: number;
  /** State dimension for the model (default: 1 for local level) */
  stateDim?: number;
}

import {
  hasNativeKalman,
  kalmanFilterUnivariate as nativeKalmanFilter,
} from "./utils/native-kalman";

// ── Matrix helpers (small-matrix operations) ────────────────────────────

/** Create an m x n zero matrix. */
function zeros(m: number, n: number): number[][] {
  return Array.from({ length: m }, () => new Array<number>(n).fill(0));
}

/** Create an n x n identity matrix. */
function eye(n: number): number[][] {
  const I = zeros(n, n);
  for (let i = 0; i < n; i++) I[i][i] = 1;
  return I;
}

/** Matrix multiply: A (m x k) * B (k x n) -> (m x n). */
function matMul(A: number[][], B: number[][]): number[][] {
  const m = A.length;
  const k = A[0].length;
  const n = B[0].length;
  const C = zeros(m, n);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let l = 0; l < k; l++) s += A[i][l] * B[l][j];
      C[i][j] = s;
    }
  }
  return C;
}

/** Matrix-vector multiply: A (m x n) * v (n) -> (m). */
function matVec(A: number[][], v: number[]): number[] {
  const m = A.length;
  const n = A[0].length;
  const r = new Array<number>(m).fill(0);
  for (let i = 0; i < m; i++) {
    let s = 0;
    for (let j = 0; j < n; j++) s += A[i][j] * v[j];
    r[i] = s;
  }
  return r;
}

/** Transpose of A. */
function transpose(A: number[][]): number[][] {
  const m = A.length;
  const n = A[0].length;
  const T = zeros(n, m);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) T[j][i] = A[i][j];
  }
  return T;
}

/** Element-wise matrix addition: A + B. */
function matAdd(A: number[][], B: number[][]): number[][] {
  const m = A.length;
  const n = A[0].length;
  const C = zeros(m, n);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) C[i][j] = A[i][j] + B[i][j];
  }
  return C;
}

/** Element-wise matrix subtraction: A - B. */
function matSub(A: number[][], B: number[][]): number[][] {
  const m = A.length;
  const n = A[0].length;
  const C = zeros(m, n);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) C[i][j] = A[i][j] - B[i][j];
  }
  return C;
}

/** Scalar multiply: c * A. */
function matScale(A: number[][], c: number): number[][] {
  const m = A.length;
  const n = A[0].length;
  const C = zeros(m, n);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) C[i][j] = c * A[i][j];
  }
  return C;
}

/** Outer product: u (m) * v (n)^T -> (m x n). */
function outer(u: number[], v: number[]): number[][] {
  const m = u.length;
  const n = v.length;
  const M = zeros(m, n);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) M[i][j] = u[i] * v[j];
  }
  return M;
}

/**
 * Invert a small square matrix via Gauss-Jordan elimination.
 * Throws if the matrix is singular.
 */
function matInv(A: number[][]): number[][] {
  const n = A.length;
  // Augmented matrix [A | I]
  const aug: number[][] = Array.from({ length: n }, (_, i) => {
    const row = new Array<number>(2 * n).fill(0);
    for (let j = 0; j < n; j++) row[j] = A[i][j];
    row[n + i] = 1;
    return row;
  });

  for (let col = 0; col < n; col++) {
    // Partial pivoting
    let maxVal = Math.abs(aug[col][col]);
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(aug[row][col]) > maxVal) {
        maxVal = Math.abs(aug[row][col]);
        maxRow = row;
      }
    }
    if (maxVal < 1e-14) {
      throw new Error("Invalid state: expected non-singular matrix for inversion, received singular matrix");
    }
    if (maxRow !== col) {
      const tmp = aug[col];
      aug[col] = aug[maxRow];
      aug[maxRow] = tmp;
    }

    const pivot = aug[col][col];
    for (let j = 0; j < 2 * n; j++) aug[col][j] /= pivot;

    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = aug[row][col];
      for (let j = 0; j < 2 * n; j++) {
        aug[row][j] -= factor * aug[col][j];
      }
    }
  }

  // Extract inverse
  const inv = zeros(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) inv[i][j] = aug[i][n + j];
  }
  return inv;
}

/** Deep-copy a matrix. */
function matCopy(A: number[][]): number[][] {
  return A.map((row) => row.slice());
}

/** Vector subtraction: a - b. */
function vecSub(a: number[], b: number[]): number[] {
  return a.map((v, i) => v - b[i]);
}

// ── Validation helpers ──────────────────────────────────────────────────

/**
 * Validate dimensions of a state-space model.
 * @throws If any matrix has inconsistent dimensions or contains NaN/Infinity.
 */
function validateModel(model: StateSpaceModel): { m: number; p: number } {
  const { F, H, Q, R, x0, P0 } = model;

  const m = F.length;
  if (m === 0) throw new Error("Invalid parameter 'F': expected at least 1 state dimension, received 0");

  // F: m x m
  for (let i = 0; i < m; i++) {
    if (F[i].length !== m) throw new Error(`Invalid parameter 'F': expected square ${m}x${m} matrix, received row ${i} with ${F[i].length} columns`);
  }

  // H: p x m
  const p = H.length;
  if (p === 0) throw new Error("Invalid parameter 'H': expected at least 1 observation dimension, received 0");
  for (let i = 0; i < p; i++) {
    if (H[i].length !== m) throw new Error(`Invalid parameter 'H': expected ${m} columns at row ${i}, received ${H[i].length}`);
  }

  // Q: m x m
  if (Q.length !== m) throw new Error(`Invalid parameter 'Q': expected ${m}x${m} matrix, received ${Q.length} rows`);
  for (let i = 0; i < m; i++) {
    if (Q[i].length !== m) throw new Error(`Invalid parameter 'Q': expected ${m} columns at row ${i}, received ${Q[i].length}`);
  }

  // R: p x p
  if (R.length !== p) throw new Error(`Invalid parameter 'R': expected ${p}x${p} matrix, received ${R.length} rows`);
  for (let i = 0; i < p; i++) {
    if (R[i].length !== p) throw new Error(`Invalid parameter 'R': expected ${p} columns at row ${i}, received ${R[i].length}`);
  }

  // x0: m
  if (x0.length !== m) throw new Error(`Invalid parameter 'x0': expected length ${m}, received ${x0.length}`);

  // P0: m x m
  if (P0.length !== m) throw new Error(`Invalid parameter 'P0': expected ${m}x${m} matrix, received ${P0.length} rows`);
  for (let i = 0; i < m; i++) {
    if (P0[i].length !== m) throw new Error(`Invalid parameter 'P0': expected ${m} columns at row ${i}, received ${P0[i].length}`);
  }

  // NaN guard on all matrices
  const allEntries: { name: string; mat: number[][] | number[] }[] = [
    { name: "F", mat: F },
    { name: "H", mat: H },
    { name: "Q", mat: Q },
    { name: "R", mat: R },
    { name: "P0", mat: P0 },
  ];
  for (const { name, mat } of allEntries) {
    for (let i = 0; i < mat.length; i++) {
      const row = mat[i] as number[] | number;
      if (Array.isArray(row)) {
        for (let j = 0; j < row.length; j++) {
          if (!Number.isFinite(row[j])) {
            throw new Error(`Invalid parameter '${name}': expected finite number at [${i}][${j}], received ${row[j]}`);
          }
        }
      }
    }
  }
  for (let i = 0; i < x0.length; i++) {
    if (!Number.isFinite(x0[i])) throw new Error(`Invalid parameter 'x0': expected finite number at index ${i}, received ${x0[i]}`);
  }

  return { m, p };
}

/**
 * Check whether an observation vector contains any NaN (missing).
 */
function isMissing(obs: number[]): boolean {
  for (let i = 0; i < obs.length; i++) {
    if (!Number.isFinite(obs[i])) return true;
  }
  return false;
}

// ── Kalman filter ───────────────────────────────────────────────────────

/**
 * Run the Kalman filter (forward pass) on a set of observations.
 *
 * Handles missing observations (NaN values) by skipping the update step,
 * propagating the predicted state and covariance forward.
 *
 * @param model - The state-space model specification
 * @param observations - Array of observation vectors [t][obs]. Each element
 *   is a number array of length p, or may contain NaN for missing data.
 * @returns The filtered states, covariances, predictions, innovations, and log-likelihood
 * @throws If model dimensions are inconsistent
 * @throws If observation dimension does not match H
 *
 * @example
 * ```ts
 * const model: StateSpaceModel = {
 *   F: [[1]], H: [[1]], Q: [[0.5]], R: [[1]],
 *   x0: [0], P0: [[1]]
 * };
 * const obs = [[1.2], [NaN], [2.1], [3.0]];
 * const result = kalmanFilter(model, obs);
 * // result.states — filtered state at each time step
 * // result.logLikelihood — total log-likelihood
 * ```
 */
export function kalmanFilter(
  model: StateSpaceModel,
  observations: number[][],
): KalmanFilterResult {
  const { m, p } = validateModel(model);
  const { F, H, Q, R, x0, P0 } = model;
  const T = observations.length;

  if (T === 0) throw new Error("Invalid parameter 'observations': expected non-empty array, received length 0");

  for (let t = 0; t < T; t++) {
    if (observations[t].length !== p) {
      throw new Error(
        `Invalid parameter 'observations': expected ${p} dimensions at t=${t}, received ${observations[t].length}`,
      );
    }
  }

  const Ft = transpose(F);
  const Ht = transpose(H);

  const states: number[][] = [];
  const covariances: number[][][] = [];
  const predictions: number[][] = [];
  const innovations: number[][] = [];
  const innovationCovariances: number[][][] = [];
  let logLikelihood = 0;

  let x = x0.slice();
  let P = matCopy(P0);

  for (let t = 0; t < T; t++) {
    // ── Predict ──
    const xPred = matVec(F, x);
    const PPred = matAdd(matMul(matMul(F, P), Ft), Q);

    // ── Innovation ──
    const yPred = matVec(H, xPred);
    const innov = vecSub(observations[t], yPred);
    const S = matAdd(matMul(matMul(H, PPred), Ht), R);

    predictions.push(yPred);
    innovations.push(innov);
    innovationCovariances.push(matCopy(S));

    if (isMissing(observations[t])) {
      // Skip update; carry prediction forward
      x = xPred;
      P = PPred;
    } else {
      // ── Update ──
      const Sinv = matInv(S);
      // Kalman gain: K = PPred * H^T * S^{-1}
      const K = matMul(matMul(PPred, Ht), Sinv);

      x = xPred.map((v, i) => {
        let correction = 0;
        for (let j = 0; j < p; j++) correction += K[i][j] * innov[j];
        return v + correction;
      });

      P = matSub(PPred, matMul(matMul(K, H), PPred));

      // Log-likelihood contribution: -0.5 * (p*log(2pi) + log|S| + v^T S^{-1} v)
      const detS = matDet(S);
      if (detS > 0) {
        const SinvV = matVec(Sinv, innov);
        let quadForm = 0;
        for (let j = 0; j < p; j++) quadForm += innov[j] * SinvV[j];
        logLikelihood += -0.5 * (p * Math.log(2 * Math.PI) + Math.log(detS) + quadForm);
      }
    }

    states.push(x.slice());
    covariances.push(matCopy(P));
  }

  return {
    states,
    covariances,
    predictions,
    logLikelihood,
    innovations,
    innovationCovariances,
  };
}

/**
 * Compute the determinant of a small matrix via LU-like decomposition.
 */
function matDet(A: number[][]): number {
  const n = A.length;
  if (n === 1) return A[0][0];
  if (n === 2) return A[0][0] * A[1][1] - A[0][1] * A[1][0];

  // Gaussian elimination with partial pivoting
  const M = matCopy(A);
  let det = 1;
  for (let col = 0; col < n; col++) {
    let maxVal = Math.abs(M[col][col]);
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(M[row][col]) > maxVal) {
        maxVal = Math.abs(M[row][col]);
        maxRow = row;
      }
    }
    if (maxVal < 1e-14) return 0;
    if (maxRow !== col) {
      const tmp = M[col];
      M[col] = M[maxRow];
      M[maxRow] = tmp;
      det *= -1;
    }
    det *= M[col][col];
    for (let row = col + 1; row < n; row++) {
      const factor = M[row][col] / M[col][col];
      for (let j = col + 1; j < n; j++) {
        M[row][j] -= factor * M[col][j];
      }
    }
  }
  return det;
}

// ── Kalman smoother (RTS) ───────────────────────────────────────────────

/**
 * Run the Rauch-Tung-Striebel (RTS) smoother on a set of observations.
 *
 * Performs a forward Kalman filter pass followed by a backward smoothing
 * pass to obtain optimally smoothed state estimates.
 *
 * @param model - The state-space model specification
 * @param observations - Array of observation vectors [t][obs]
 * @returns The filtered results plus smoothed states and covariances
 * @throws If model dimensions are inconsistent
 * @throws If observation dimension does not match H
 *
 * @example
 * ```ts
 * const model: StateSpaceModel = {
 *   F: [[1]], H: [[1]], Q: [[0.5]], R: [[1]],
 *   x0: [0], P0: [[1]]
 * };
 * const obs = [[1.2], [1.8], [2.1], [3.0]];
 * const result = kalmanSmoother(model, obs);
 * // result.smoothedStates — optimally smoothed states
 * ```
 */
export function kalmanSmoother(
  model: StateSpaceModel,
  observations: number[][],
): KalmanSmootherResult {
  const { F, Q } = model;
  const Ft = transpose(F);

  // Forward pass
  const filterResult = kalmanFilter(model, observations);
  const { states, covariances } = filterResult;
  const T = observations.length;

  const smoothedStates: number[][] = new Array(T);
  const smoothedCovariances: number[][][] = new Array(T);

  // Initialize at the last time step
  smoothedStates[T - 1] = states[T - 1].slice();
  smoothedCovariances[T - 1] = matCopy(covariances[T - 1]);

  // Backward pass
  for (let t = T - 2; t >= 0; t--) {
    const PPred = matAdd(matMul(matMul(F, covariances[t]), Ft), Q);
    const PPredInv = matInv(PPred);
    // Smoother gain: L_t = P_t * F^T * PPred^{-1}
    const L = matMul(matMul(covariances[t], Ft), PPredInv);

    // Predicted state at t+1 (from filtered state at t)
    const xPred = matVec(F, states[t]);

    // Smoothed state
    const diff = vecSub(smoothedStates[t + 1], xPred);
    smoothedStates[t] = states[t].map((v, i) => {
      let correction = 0;
      for (let j = 0; j < diff.length; j++) correction += L[i][j] * diff[j];
      return v + correction;
    });

    // Smoothed covariance
    const covDiff = matSub(smoothedCovariances[t + 1], PPred);
    smoothedCovariances[t] = matAdd(
      covariances[t],
      matMul(matMul(L, covDiff), transpose(L)),
    );
  }

  return {
    ...filterResult,
    smoothedStates,
    smoothedCovariances,
  };
}

// ── EM algorithm for state-space models ─────────────────────────────────

/**
 * Estimate state-space model parameters using the Expectation-Maximization
 * (EM) algorithm for a local level model.
 *
 * The EM algorithm alternates between:
 * - E-step: run the Kalman smoother to get expected states
 * - M-step: update Q and R based on the smoothed estimates
 *
 * The model structure is a local level model (random walk + noise):
 *   x_t = x_{t-1} + w_t,   w_t ~ N(0, Q)
 *   y_t = x_t + v_t,        v_t ~ N(0, R)
 *
 * @param observations - Array of scalar observations (may contain NaN for missing)
 * @param options - EM configuration options
 * @returns The estimated StateSpaceModel and the final smoother result
 * @throws If observations is empty
 * @throws If all observations are NaN
 *
 * @example
 * ```ts
 * const obs = [1.1, 2.0, 1.8, 2.5, 3.1];
 * const { model, smootherResult } = stateSpaceEM(obs);
 * // model.Q, model.R — estimated noise covariances
 * // smootherResult.smoothedStates — smoothed level estimates
 * ```
 */
export function stateSpaceEM(
  observations: number[],
  options?: StateSpaceEMOptions,
): { model: StateSpaceModel; smootherResult: KalmanSmootherResult } {
  const maxIter = options?.maxIterations ?? 100;
  const tol = options?.tolerance ?? 1e-6;
  const stateDim = options?.stateDim ?? 1;

  if (observations.length === 0) throw new Error("Invalid parameter 'observations': expected non-empty array, received length 0");

  // Convert scalar observations to [t][1] format
  const obs: number[][] = observations.map((v) => [v]);
  const T = obs.length;

  // Count valid observations
  const validObs = observations.filter((v) => Number.isFinite(v));
  if (validObs.length === 0) throw new Error("Invalid parameter 'observations': expected at least one finite value, received all NaN");

  const obsMean = validObs.reduce((a, b) => a + b, 0) / validObs.length;
  const obsVar = validObs.reduce((a, b) => a + (b - obsMean) ** 2, 0) / validObs.length;

  // Initialize model parameters
  let qVal = obsVar * 0.1;
  let rVal = obsVar * 0.9;

  // Build model matrices
  function buildModel(q: number, r: number): StateSpaceModel {
    if (stateDim === 1) {
      return {
        F: [[1]],
        H: [[1]],
        Q: [[q]],
        R: [[r]],
        x0: [obsMean],
        P0: [[obsVar]],
      };
    }
    // For stateDim === 2 (local linear trend)
    return {
      F: [
        [1, 1],
        [0, 1],
      ],
      H: [[1, 0]],
      Q: [
        [q, 0],
        [0, q * 0.1],
      ],
      R: [[r]],
      x0: [obsMean, 0],
      P0: [
        [obsVar, 0],
        [0, obsVar * 0.1],
      ],
    };
  }

  let model = buildModel(qVal, rVal);
  let prevLogLik = -Infinity;
  let smootherResult: KalmanSmootherResult;

  for (let iter = 0; iter < maxIter; iter++) {
    // E-step: Kalman smoother
    smootherResult = kalmanSmoother(model, obs);

    // Check convergence
    if (
      Math.abs(smootherResult.logLikelihood - prevLogLik) < tol &&
      iter > 0
    ) {
      return { model, smootherResult };
    }
    prevLogLik = smootherResult.logLikelihood;

    // M-step: re-estimate Q and R
    const { smoothedStates, smoothedCovariances } = smootherResult;

    if (stateDim === 1) {
      // Estimate R: average (y_t - x_t)^2 + P_t
      let sumR = 0;
      let countR = 0;
      for (let t = 0; t < T; t++) {
        if (!isMissing(obs[t])) {
          const resid = obs[t][0] - smoothedStates[t][0];
          sumR += resid * resid + smoothedCovariances[t][0][0];
          countR++;
        }
      }
      rVal = countR > 0 ? sumR / countR : rVal;

      // Estimate Q: average (x_t - x_{t-1})^2 + P_t + P_{t-1} - 2*cov(x_t, x_{t-1})
      // Approximate cross-covariance using smoother gain
      let sumQ = 0;
      for (let t = 1; t < T; t++) {
        const diff = smoothedStates[t][0] - smoothedStates[t - 1][0];
        sumQ +=
          diff * diff +
          smoothedCovariances[t][0][0] +
          smoothedCovariances[t - 1][0][0];
        // Note: the cross-covariance term is approximated as part of the
        // smoother covariance already for the local level model
      }
      qVal = T > 1 ? sumQ / (T - 1) : qVal;
      // Ensure positivity
      qVal = Math.max(qVal, 1e-10);
      rVal = Math.max(rVal, 1e-10);
    } else {
      // stateDim === 2: local linear trend M-step
      let sumR = 0;
      let countR = 0;
      for (let t = 0; t < T; t++) {
        if (!isMissing(obs[t])) {
          const resid = obs[t][0] - smoothedStates[t][0];
          sumR += resid * resid + smoothedCovariances[t][0][0];
          countR++;
        }
      }
      rVal = countR > 0 ? sumR / countR : rVal;

      let sumQ = 0;
      for (let t = 1; t < T; t++) {
        const diff0 = smoothedStates[t][0] - smoothedStates[t - 1][0] - smoothedStates[t - 1][1];
        sumQ += diff0 * diff0 + smoothedCovariances[t][0][0] + smoothedCovariances[t - 1][0][0];
      }
      qVal = T > 1 ? sumQ / (T - 1) : qVal;
      qVal = Math.max(qVal, 1e-10);
      rVal = Math.max(rVal, 1e-10);
    }

    model = buildModel(qVal, rVal);
  }

  // Final smoother run with converged parameters
  smootherResult = kalmanSmoother(model, obs);
  return { model, smootherResult: smootherResult! };
}

// ── Local level model ───────────────────────────────────────────────────

/**
 * Build and fit a local level (random walk plus noise) state-space model.
 *
 * The local level model is:
 *   x_t = x_{t-1} + w_t,   w_t ~ N(0, Q)
 *   y_t = x_t + v_t,        v_t ~ N(0, R)
 *
 * Parameters Q and R are estimated via the EM algorithm.
 *
 * @param observations - Array of scalar observations (may contain NaN for missing)
 * @returns Object containing the fitted model, smoother result, and filtered result
 * @throws If observations is empty or all NaN
 *
 * @example
 * ```ts
 * const obs = [1.0, 1.5, 2.0, 1.8, 2.3, 2.5];
 * const result = localLevelModel(obs);
 * // result.smoothedStates — smoothed level
 * // result.model — estimated model with Q and R
 * ```
 */
export function localLevelModel(observations: number[]): {
  model: StateSpaceModel;
  smootherResult: KalmanSmootherResult;
  filterResult: KalmanFilterResult;
} {
  if (observations.length === 0) {
    throw new Error("Invalid parameter 'observations': expected non-empty array, received length 0");
  }

  const { model, smootherResult } = stateSpaceEM(observations, { stateDim: 1 });

  // Also run filter for convenience
  const obs2d = observations.map((v) => [v]);
  const filterResult = kalmanFilter(model, obs2d);

  return { model, smootherResult, filterResult };
}

// ── Local linear trend model ────────────────────────────────────────────

/**
 * Build and fit a local linear trend state-space model.
 *
 * The local linear trend model has two state components (level and slope):
 *   level_t = level_{t-1} + slope_{t-1} + w1_t
 *   slope_t = slope_{t-1} + w2_t
 *   y_t     = level_t + v_t
 *
 * Parameters are estimated via the EM algorithm.
 *
 * @param observations - Array of scalar observations (may contain NaN for missing)
 * @returns Object containing the fitted model, smoother result, and filtered result
 * @throws If observations is empty or all NaN
 *
 * @example
 * ```ts
 * const obs = [1.0, 2.1, 3.0, 4.2, 5.1, 5.9];
 * const result = localLinearTrendModel(obs);
 * // result.smootherResult.smoothedStates[t][0] — smoothed level
 * // result.smootherResult.smoothedStates[t][1] — smoothed slope
 * ```
 */
export function localLinearTrendModel(observations: number[]): {
  model: StateSpaceModel;
  smootherResult: KalmanSmootherResult;
  filterResult: KalmanFilterResult;
} {
  if (observations.length === 0) {
    throw new Error("Invalid parameter 'observations': expected non-empty array, received length 0");
  }

  const { model, smootherResult } = stateSpaceEM(observations, { stateDim: 2 });

  const obs2d = observations.map((v) => [v]);
  const filterResult = kalmanFilter(model, obs2d);

  return { model, smootherResult, filterResult };
}

// ── State-space prediction ──────────────────────────────────────────────

/**
 * Generate multi-step-ahead forecasts from a fitted state-space model.
 *
 * Starting from the last filtered state and covariance, propagates the
 * state-space model forward to produce observation forecasts with 95%
 * prediction intervals.
 *
 * @param model - The state-space model
 * @param filterResult - A Kalman filter result (must have at least one time step)
 * @param horizon - Number of steps to forecast ahead (must be >= 1)
 * @returns Forecasted states, observations, covariances, and 95% confidence bounds
 * @throws If horizon is less than 1
 * @throws If filterResult has no states
 *
 * @example
 * ```ts
 * const model: StateSpaceModel = {
 *   F: [[1]], H: [[1]], Q: [[0.5]], R: [[1]],
 *   x0: [0], P0: [[1]]
 * };
 * const obs = [[1.2], [1.8], [2.1], [3.0]];
 * const filtered = kalmanFilter(model, obs);
 * const forecast = stateSpacePredict(model, filtered, 5);
 * // forecast.observationForecast — predicted observations
 * // forecast.lowerBound, forecast.upperBound — 95% CI
 * ```
 */
export function stateSpacePredict(
  model: StateSpaceModel,
  filterResult: KalmanFilterResult,
  horizon: number,
): StateSpacePrediction {
  if (horizon < 1) throw new Error(`Invalid parameter 'horizon': expected at least 1, received ${horizon}`);
  if (filterResult.states.length === 0) {
    throw new Error("Invalid parameter 'filterResult': expected at least one state, received 0");
  }

  validateModel(model);

  const { F, H, Q, R } = model;
  const Ft = transpose(F);
  const Ht = transpose(H);
  const p = H.length;

  const T = filterResult.states.length;
  let x = filterResult.states[T - 1].slice();
  let P = matCopy(filterResult.covariances[T - 1]);

  const z975 = 1.96; // approximate z_{0.975}

  const stateForecast: number[][] = [];
  const observationForecast: number[][] = [];
  const stateCovariances: number[][][] = [];
  const observationCovariances: number[][][] = [];
  const lowerBound: number[][] = [];
  const upperBound: number[][] = [];

  for (let h = 0; h < horizon; h++) {
    // Propagate state
    x = matVec(F, x);
    P = matAdd(matMul(matMul(F, P), Ft), Q);

    // Observation forecast
    const yHat = matVec(H, x);
    const S = matAdd(matMul(matMul(H, P), Ht), R);

    stateForecast.push(x.slice());
    observationForecast.push(yHat.slice());
    stateCovariances.push(matCopy(P));
    observationCovariances.push(matCopy(S));

    // 95% prediction interval
    const lower: number[] = new Array(p);
    const upper: number[] = new Array(p);
    for (let j = 0; j < p; j++) {
      const se = Math.sqrt(Math.max(0, S[j][j]));
      lower[j] = yHat[j] - z975 * se;
      upper[j] = yHat[j] + z975 * se;
    }
    lowerBound.push(lower);
    upperBound.push(upper);
  }

  return {
    stateForecast,
    observationForecast,
    stateCovariances,
    observationCovariances,
    lowerBound,
    upperBound,
  };
}
