/**
 * Native Kalman filter acceleration layer.
 *
 * Exposes Fortran-accelerated Kalman filter for the common case of
 * univariate observations. Falls back to pure TypeScript when the
 * native addon is unavailable.
 */

import { nativeAddon } from "./native-addon";

// ── Native interface ────────────────────────────────────────────────────

interface NativeKalman {
  kalmanFilterUnivariate(
    F: number[][],
    H: number[],
    Q: number[][],
    R: number,
    y: number[],
    x0: number[],
    P0: number[][],
    T: number,
    m: number,
  ): { states: number[][]; logLikelihood: number };
}

// Probe for native Kalman availability
let native: NativeKalman | null = null;
try {
  if (nativeAddon && typeof nativeAddon.kalmanFilterUnivariate === "function") {
    native = nativeAddon as unknown as NativeKalman;
  }
} catch {
  // Fallback to TypeScript
}

/** Whether native Kalman filter acceleration is available. */
export const hasNativeKalman = native !== null;

// ── Kalman Filter (univariate observations) ─────────────────────────────

/**
 * Run the Kalman filter forward pass for a univariate observation model.
 * Uses Fortran when available for tight matrix-per-timestep acceleration.
 *
 * @param F - State transition matrix (m x m)
 * @param H - Observation row vector (1 x m, passed as flat array)
 * @param Q - Process noise covariance (m x m)
 * @param R - Observation noise variance (scalar)
 * @param y - Observations (T-length)
 * @param x0 - Initial state (m-length)
 * @param P0 - Initial covariance (m x m)
 * @returns Filtered states (T x m) and total log-likelihood
 */
export function kalmanFilterUnivariate(
  F: number[][],
  H: number[],
  Q: number[][],
  R: number,
  y: number[],
  x0: number[],
  P0: number[][],
): { states: number[][]; logLikelihood: number } {
  const T = y.length;
  const m = x0.length;

  if (native) {
    return native.kalmanFilterUnivariate(F, H, Q, R, y, x0, P0, T, m);
  }

  return tsKalmanFilterUnivariate(F, H, Q, R, y, x0, P0, T, m);
}

function tsKalmanFilterUnivariate(
  F: number[][],
  H: number[],
  Q: number[][],
  R: number,
  y: number[],
  x0: number[],
  P0: number[][],
  T: number,
  m: number,
): { states: number[][]; logLikelihood: number } {
  const LOG2PI = Math.log(2 * Math.PI);

  // Working state
  let x = x0.slice();
  let P = P0.map((r) => r.slice());

  const states: number[][] = [];
  let ll = 0;

  for (let t = 0; t < T; t++) {
    // Predict: x_pred = F * x
    const xPred = new Array(m).fill(0);
    for (let i = 0; i < m; i++) {
      for (let k = 0; k < m; k++) {
        xPred[i] += F[i][k] * x[k];
      }
    }

    // P_pred = F * P * F^T + Q
    const FP = Array.from({ length: m }, () => new Array(m).fill(0));
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < m; j++) {
        for (let k = 0; k < m; k++) {
          FP[i][j] += F[i][k] * P[k][j];
        }
      }
    }
    const PPred = Array.from({ length: m }, () => new Array(m).fill(0));
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < m; j++) {
        PPred[i][j] = Q[i][j];
        for (let k = 0; k < m; k++) {
          PPred[i][j] += FP[i][k] * F[j][k];
        }
      }
    }

    // Update: scalar observation
    let Hx = 0;
    for (let k = 0; k < m; k++) Hx += H[k] * xPred[k];
    const innov = y[t] - Hx;

    // PH = P_pred * H^T (m-vector)
    const PH = new Array(m).fill(0);
    for (let i = 0; i < m; i++) {
      for (let k = 0; k < m; k++) {
        PH[i] += PPred[i][k] * H[k];
      }
    }

    // S = H * PH + R
    let S = R;
    for (let k = 0; k < m; k++) S += H[k] * PH[k];

    const Sinv = S > 0 ? 1 / S : 0;
    ll += -0.5 * (LOG2PI + Math.log(Math.max(S, 1e-300)) + innov * innov * Sinv);

    // x = xPred + K * innov
    x = new Array(m);
    for (let i = 0; i < m; i++) {
      x[i] = xPred[i] + PH[i] * Sinv * innov;
    }

    // P = PPred - PH * PH^T * Sinv
    P = Array.from({ length: m }, () => new Array(m).fill(0));
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < m; j++) {
        P[i][j] = PPred[i][j] - PH[i] * PH[j] * Sinv;
      }
    }

    states.push(x.slice());
  }

  return { states, logLikelihood: ll };
}
