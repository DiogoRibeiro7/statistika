import { Dataset } from "./types";
import { mean } from "./utils/descriptive";
import { solveLinearSystem } from "./utils/linalg";
import { Normal } from "./distributions/continuous/normal";

/**
 * Link function for GLM.
 */
export interface LinkFunction {
  /** Link function g(μ). */
  link(mu: number): number;
  /** Inverse link g^{-1}(η). */
  inverse(eta: number): number;
  /** Derivative of inverse link. */
  derivative(eta: number): number;
}

/**
 * GLM family (distribution + canonical link).
 */
export interface GLMFamily {
  name: string;
  link: LinkFunction;
  /** Variance function V(μ). */
  variance(mu: number): number;
  /** Log-likelihood contribution for one observation. */
  logLikelihood(y: number, mu: number): number;
}

// ── Link Functions ──────────────────────────────────────────────────────

export const identityLink: LinkFunction = {
  link: (mu) => mu,
  inverse: (eta) => eta,
  derivative: () => 1,
};

export const logLink: LinkFunction = {
  link: (mu) => Math.log(mu),
  inverse: (eta) => Math.exp(eta),
  derivative: (eta) => Math.exp(eta),
};

export const logitLink: LinkFunction = {
  link: (mu) => Math.log(mu / (1 - mu)),
  inverse: (eta) => 1 / (1 + Math.exp(-eta)),
  derivative: (eta) => {
    const p = 1 / (1 + Math.exp(-eta));
    return p * (1 - p);
  },
};

export const probitLink: LinkFunction = {
  link: (mu) => new Normal().quantile(mu),
  inverse: (eta) => new Normal().cdf(eta),
  derivative: (eta) => new Normal().pdf(eta),
};

export const inverseLink: LinkFunction = {
  link: (mu) => 1 / mu,
  inverse: (eta) => 1 / eta,
  derivative: (eta) => -1 / (eta * eta),
};

// ── GLM Families ────────────────────────────────────────────────────────

export const gaussian: GLMFamily = {
  name: "gaussian",
  link: identityLink,
  variance: () => 1,
  logLikelihood: (y, mu) => -0.5 * (y - mu) ** 2,
};

export const binomial: GLMFamily = {
  name: "binomial",
  link: logitLink,
  variance: (mu) => mu * (1 - mu),
  logLikelihood: (y, mu) => {
    const p = Math.max(1e-10, Math.min(1 - 1e-10, mu));
    return y * Math.log(p) + (1 - y) * Math.log(1 - p);
  },
};

export const poisson: GLMFamily = {
  name: "poisson",
  link: logLink,
  variance: (mu) => mu,
  logLikelihood: (y, mu) => {
    const m = Math.max(1e-10, mu);
    return y * Math.log(m) - m;
  },
};

export const gamma: GLMFamily = {
  name: "gamma",
  link: inverseLink,
  variance: (mu) => mu * mu,
  logLikelihood: (y, mu) => {
    const m = Math.max(1e-10, mu);
    return -y / m - Math.log(m);
  },
};

/**
 * Result of a GLM fit.
 */
export interface GLMResult {
  /** Estimated coefficients (including intercept as first element). */
  coefficients: number[];
  /** Standard errors of coefficients. */
  standardErrors: number[];
  /** z/t statistics. */
  zValues: number[];
  /** p-values for coefficients. */
  pValues: number[];
  /** Log-likelihood. */
  logLikelihood: number;
  /** Deviance. */
  deviance: number;
  /** AIC. */
  aic: number;
  /** Number of IRLS iterations. */
  iterations: number;
  /** Family name. */
  family: string;
  /** Prediction function. */
  predict: (x: number[]) => number;
}

/**
 * Fit a Generalized Linear Model using IRLS (Iteratively Reweighted Least Squares).
 *
 * @param X - Design matrix (n x p), without intercept column
 * @param y - Response variable
 * @param family - GLM family (gaussian, binomial, poisson, gamma)
 * @param options - Configuration
 */
export function glm(
  X: number[][],
  y: Dataset,
  family: GLMFamily,
  options: {
    maxIterations?: number;
    tol?: number;
  } = {},
): GLMResult {
  const n = X.length;
  const p = X[0].length;
  if (n !== y.length) throw new Error("X and y must have the same length");
  if (n <= p + 1) throw new Error("Need more observations than parameters");

  const maxIter = options.maxIterations ?? 25;
  const tol = options.tol ?? 1e-8;
  const cols = p + 1; // +1 for intercept

  // Initialize coefficients
  const beta = new Array<number>(cols).fill(0);
  // Initialize intercept to link of mean(y)
  const yMean = mean(y);
  beta[0] = family.link.link(Math.max(1e-6, Math.min(1 - 1e-6, yMean)));

  let iter = 0;
  for (; iter < maxIter; iter++) {
    // Compute linear predictor η = Xβ
    const eta = new Array<number>(n);
    const mu = new Array<number>(n);
    for (let i = 0; i < n; i++) {
      eta[i] = beta[0];
      for (let j = 0; j < p; j++) eta[i] += beta[j + 1] * X[i][j];
      mu[i] = family.link.inverse(eta[i]);
    }

    // IRLS: compute working weights and adjusted response
    const W = new Array<number>(n);
    const z = new Array<number>(n);
    for (let i = 0; i < n; i++) {
      const dmu = family.link.derivative(eta[i]);
      const v = family.variance(mu[i]);
      W[i] = (dmu * dmu) / Math.max(v, 1e-10);
      z[i] = eta[i] + (y[i] - mu[i]) / Math.max(dmu, 1e-10);
    }

    // Weighted least squares: solve (X^T W X) β = X^T W z
    const XtWX = Array.from({ length: cols }, () => new Array<number>(cols).fill(0));
    const XtWz = new Array<number>(cols).fill(0);

    for (let i = 0; i < n; i++) {
      const row = [1, ...X[i]];
      for (let j = 0; j < cols; j++) {
        XtWz[j] += row[j] * W[i] * z[i];
        for (let k = 0; k < cols; k++) {
          XtWX[j][k] += row[j] * W[i] * row[k];
        }
      }
    }

    const betaNew = solveLinearSystem(XtWX, XtWz);

    // Check convergence
    let maxDiff = 0;
    for (let j = 0; j < cols; j++) {
      maxDiff = Math.max(maxDiff, Math.abs(betaNew[j] - beta[j]));
    }

    for (let j = 0; j < cols; j++) beta[j] = betaNew[j];

    if (maxDiff < tol) {
      iter++;
      break;
    }
  }

  // Final predictions
  const eta = new Array<number>(n);
  const mu = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    eta[i] = beta[0];
    for (let j = 0; j < p; j++) eta[i] += beta[j + 1] * X[i][j];
    mu[i] = family.link.inverse(eta[i]);
  }

  // Log-likelihood
  let ll = 0;
  for (let i = 0; i < n; i++) ll += family.logLikelihood(y[i], mu[i]);

  // Deviance (null deviance - residual deviance approximation)
  let deviance = 0;
  for (let i = 0; i < n; i++) {
    const saturated = family.logLikelihood(y[i], y[i] === 0 ? 1e-10 : y[i]);
    deviance += 2 * (saturated - family.logLikelihood(y[i], mu[i]));
  }

  // Standard errors via (X^T W X)^{-1}
  const W = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    const dmu = family.link.derivative(eta[i]);
    const v = family.variance(mu[i]);
    W[i] = (dmu * dmu) / Math.max(v, 1e-10);
  }

  const XtWX = Array.from({ length: cols }, () => new Array<number>(cols).fill(0));
  for (let i = 0; i < n; i++) {
    const row = [1, ...X[i]];
    for (let j = 0; j < cols; j++) {
      for (let k = 0; k < cols; k++) {
        XtWX[j][k] += row[j] * W[i] * row[k];
      }
    }
  }

  const covMatrix = invertMatrix(XtWX);
  const standardErrors = new Array<number>(cols);
  const zValues = new Array<number>(cols);
  const pValues = new Array<number>(cols);
  const normal = new Normal();

  for (let j = 0; j < cols; j++) {
    standardErrors[j] = Math.sqrt(Math.max(0, covMatrix[j][j]));
    zValues[j] = standardErrors[j] > 0 ? beta[j] / standardErrors[j] : 0;
    pValues[j] = 2 * (1 - normal.cdf(Math.abs(zValues[j])));
  }

  const aic = -2 * ll + 2 * cols;

  return {
    coefficients: beta,
    standardErrors,
    zValues,
    pValues,
    logLikelihood: ll,
    deviance,
    aic,
    iterations: iter,
    family: family.name,
    predict: (x: number[]) => {
      let eta = beta[0];
      for (let j = 0; j < p; j++) eta += beta[j + 1] * x[j];
      return family.link.inverse(eta);
    },
  };
}

function invertMatrix(matrix: number[][]): number[][] {
  const n = matrix.length;
  const aug = matrix.map((row, i) => {
    const r = [...row];
    for (let j = 0; j < n; j++) r.push(i === j ? 1 : 0);
    return r;
  });

  for (let col = 0; col < n; col++) {
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(aug[row][col]) > Math.abs(aug[maxRow][col])) maxRow = row;
    }
    [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];

    const pivot = aug[col][col];
    if (Math.abs(pivot) < 1e-12) return Array.from({ length: n }, () => new Array(n).fill(0));

    for (let j = 0; j < 2 * n; j++) aug[col][j] /= pivot;
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = aug[row][col];
      for (let j = 0; j < 2 * n; j++) aug[row][j] -= factor * aug[col][j];
    }
  }

  return aug.map((row) => row.slice(n));
}
