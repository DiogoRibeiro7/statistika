import { Dataset } from "./types";
import { mean } from "./utils/descriptive";
import { solveLinearSystem, invertMatrix, normalCdf, normalQuantile } from "./utils/linalg";
import { weightedCrossProducts } from "./utils/native-stats";

/**
 * Link function for a Generalized Linear Model.
 *
 * Defines the relationship between the linear predictor eta and the
 * conditional mean mu: g(mu) = eta, or equivalently mu = g^{-1}(eta).
 */
export interface LinkFunction {
  /** Link function g(mu) mapping the mean to the linear predictor scale. */
  link(mu: number): number;
  /** Inverse link g^{-1}(eta) mapping the linear predictor to the mean scale. */
  inverse(eta: number): number;
  /** Derivative of the inverse link d(g^{-1})/d(eta), used in IRLS weighting. */
  derivative(eta: number): number;
}

/**
 * GLM family specifying the response distribution and canonical link function.
 *
 * Combines a link function with the variance function V(mu) and the
 * log-likelihood contribution for the exponential family distribution.
 */
export interface GLMFamily {
  /** Name of the family (e.g. "gaussian", "binomial", "poisson", "gamma"). */
  name: string;
  /** The link function relating the linear predictor to the mean. */
  link: LinkFunction;
  /** Variance function V(mu) defining how variance depends on the mean. */
  variance(mu: number): number;
  /** Log-likelihood contribution for a single observation (y, mu). */
  logLikelihood(y: number, mu: number): number;
}

// -- Link Functions ----------------------------------------------------------

/** Identity link: g(mu) = mu. Canonical link for the Gaussian family. */
export const identityLink: LinkFunction = {
  link: (mu) => mu,
  inverse: (eta) => eta,
  derivative: () => 1,
};

/** Log link: g(mu) = ln(mu). Canonical link for the Poisson family. */
export const logLink: LinkFunction = {
  link: (mu) => Math.log(mu),
  inverse: (eta) => Math.exp(eta),
  derivative: (eta) => Math.exp(eta),
};

/** Logit link: g(mu) = ln(mu/(1-mu)). Canonical link for the Binomial family. */
export const logitLink: LinkFunction = {
  link: (mu) => Math.log(mu / (1 - mu)),
  inverse: (eta) => 1 / (1 + Math.exp(-eta)),
  derivative: (eta) => {
    const p = 1 / (1 + Math.exp(-eta));
    return p * (1 - p);
  },
};

/** Probit link: g(mu) = Phi^{-1}(mu), where Phi is the standard normal CDF. */
export const probitLink: LinkFunction = {
  link: (mu) => normalQuantile(mu),
  inverse: (eta) => normalCdf(eta),
  derivative: (eta) => {
    // Standard normal PDF: (1/sqrt(2*pi)) * exp(-0.5 * eta^2)
    return (1 / Math.sqrt(2 * Math.PI)) * Math.exp(-0.5 * eta * eta);
  },
};

/** Inverse (reciprocal) link: g(mu) = 1/mu. Canonical link for the Gamma family. */
export const inverseLink: LinkFunction = {
  link: (mu) => 1 / mu,
  inverse: (eta) => 1 / eta,
  derivative: (eta) => -1 / (eta * eta),
};

// -- GLM Families ------------------------------------------------------------

/** Gaussian (normal) family with identity link. V(mu) = 1. */
export const gaussian: GLMFamily = {
  name: "gaussian",
  link: identityLink,
  variance: () => 1,
  logLikelihood: (y, mu) => -0.5 * (y - mu) ** 2,
};

/** Binomial family with logit link. V(mu) = mu*(1-mu). For binary/proportion responses. */
export const binomial: GLMFamily = {
  name: "binomial",
  link: logitLink,
  variance: (mu) => mu * (1 - mu),
  logLikelihood: (y, mu) => {
    const p = Math.max(1e-10, Math.min(1 - 1e-10, mu));
    return y * Math.log(p) + (1 - y) * Math.log(1 - p);
  },
};

/** Poisson family with log link. V(mu) = mu. For count data. */
export const poisson: GLMFamily = {
  name: "poisson",
  link: logLink,
  variance: (mu) => mu,
  logLikelihood: (y, mu) => {
    const m = Math.max(1e-10, mu);
    return y * Math.log(m) - m;
  },
};

/** Gamma family with inverse link. V(mu) = mu^2. For positive continuous data. */
export const gammaFamily: GLMFamily = {
  name: "gamma",
  link: inverseLink,
  variance: (mu) => mu * mu,
  logLikelihood: (y, mu) => {
    const m = Math.max(1e-10, mu);
    return -y / m - Math.log(m);
  },
};

/**
 * Result of fitting a Generalized Linear Model.
 *
 * Contains estimated coefficients, standard errors, significance tests,
 * goodness-of-fit measures, and a prediction function.
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
 * @returns A {@link GLMResult} with coefficients, standard errors, z-values, p-values,
 *   log-likelihood, deviance, AIC, iteration count, and a predict function
 * @throws If X and y have mismatched lengths, X has inconsistent row lengths,
 *   inputs contain NaN/Infinity, or there are not enough observations
 *
 * @example
 * ```ts
 * const X = [[1], [2], [3], [4], [5]];
 * const y = [0, 0, 1, 1, 1];
 * const result = glm(X, y, binomial);
 * console.log(result.coefficients); // [intercept, slope]
 * console.log(result.predict([3])); // probability near 0.5
 * ```
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
  if (n === 0) throw new Error(`Invalid parameter 'X': expected a non-empty matrix, received 0 rows`);
  const p = X[0].length;
  if (n !== y.length) throw new Error(`Invalid parameters 'X', 'y': expected same length, received X.length=${n}, y.length=${y.length}`);
  if (n <= p + 1) throw new Error(`Invalid parameters 'X', 'y': expected more observations than parameters, received n=${n}, p=${p + 1}`);

  // Validate consistent row lengths
  for (let i = 0; i < n; i++) {
    if (X[i].length !== p) {
      throw new Error(
        `Invalid parameter 'X': expected ${p} columns at row ${i}, received ${X[i].length}`,
      );
    }
  }

  // NaN / Infinity guards
  for (let i = 0; i < n; i++) {
    if (!Number.isFinite(y[i])) {
      throw new Error(`Invalid parameter 'y': expected finite number at index ${i}, received ${y[i]}`);
    }
    for (let j = 0; j < p; j++) {
      if (!Number.isFinite(X[i][j])) {
        throw new Error(`Invalid parameter 'X': expected finite number at [${i}][${j}], received ${X[i][j]}`);
      }
    }
  }

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
    // Compute linear predictor eta = X*beta
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

    // Weighted least squares: solve (X^T W X) beta = X^T W z
    // Uses Fortran-accelerated cross-products when native addon is available
    const designMatrix = X.map((row) => [1, ...row]);
    const { XtWX, XtWz } = weightedCrossProducts(designMatrix, W, z);

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

  // Standard errors via (X^T W X)^{-1} (uses LAPACK when available)
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

  for (let j = 0; j < cols; j++) {
    standardErrors[j] = Math.sqrt(Math.max(0, covMatrix ? covMatrix[j][j] : 0));
    zValues[j] = standardErrors[j] > 0 ? beta[j] / standardErrors[j] : 0;
    pValues[j] = 2 * (1 - normalCdf(Math.abs(zValues[j])));
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

// -- Additional Link Functions ------------------------------------------------

/** Complementary log-log link: g(mu) = ln(-ln(1-mu)). For asymmetric binary responses. */
export const cloglogLink: LinkFunction = {
  link: (mu) => Math.log(-Math.log(1 - mu)),
  inverse: (eta) => 1 - Math.exp(-Math.exp(eta)),
  derivative: (eta) => Math.exp(eta - Math.exp(eta)),
};

// -- Additional GLM Families --------------------------------------------------

/**
 * Negative Binomial family with log link.
 *
 * V(mu) = mu + mu^2 / theta (quadratic variance function).
 * Suitable for overdispersed count data where the Poisson assumption is too restrictive.
 *
 * @param theta - Dispersion parameter (shape parameter). Larger theta means closer to Poisson.
 * @returns A {@link GLMFamily} for negative binomial regression.
 */
export function negativeBinomialFamily(theta: number): GLMFamily {
  return {
    name: "negativeBinomial",
    link: logLink,
    variance: (mu) => mu + (mu * mu) / theta,
    logLikelihood: (y, mu) => {
      const m = Math.max(1e-10, mu);
      // Log-likelihood kernel for NB(theta, mu):
      // y*ln(mu/(mu+theta)) + theta*ln(theta/(mu+theta)) + lnGamma(y+theta) - lnGamma(theta) - lnGamma(y+1)
      // We omit constant terms involving only y and theta since they don't affect fitting.
      return y * Math.log(m / (m + theta)) + theta * Math.log(theta / (m + theta));
    },
  };
}

/**
 * Tweedie family with log link.
 *
 * V(mu) = mu^p where 1 < p < 2, corresponding to compound Poisson-Gamma distributions.
 * Particularly useful for insurance claims and other zero-inflated continuous data.
 *
 * @param p - Variance power parameter, must satisfy 1 < p < 2.
 * @returns A {@link GLMFamily} for Tweedie regression.
 * @throws If p is not in the open interval (1, 2).
 */
export function tweedieFamily(p: number): GLMFamily {
  if (p <= 1 || p >= 2) {
    throw new Error(`Invalid parameter 'p': expected a value in (1, 2), received ${p}`);
  }
  return {
    name: "tweedie",
    link: logLink,
    variance: (mu) => Math.pow(Math.max(1e-10, mu), p),
    logLikelihood: (y, mu) => {
      const m = Math.max(1e-10, mu);
      // Tweedie deviance unit: 2 * ( y*mu^(1-p)/(1-p) - mu^(2-p)/(2-p) )
      // We use the log-likelihood kernel form for IRLS fitting.
      return -(
        (Math.pow(m, 2 - p) / (2 - p)) -
        (y * Math.pow(m, 1 - p) / (1 - p))
      );
    },
  };
}

/**
 * Quasi-likelihood family with user-specified variance function and link.
 *
 * Quasi-likelihood does not correspond to a true probability distribution,
 * so there is no proper log-likelihood. The deviance is computed from the
 * quasi-likelihood equations instead.
 *
 * @param varianceFn - Variance function V(mu) defining how variance depends on the mean.
 * @param link - Link function relating the linear predictor to the mean.
 * @returns A {@link GLMFamily} for quasi-likelihood regression.
 */
export function quasiFamily(varianceFn: (mu: number) => number, link: LinkFunction): GLMFamily {
  return {
    name: "quasi",
    link,
    variance: varianceFn,
    logLikelihood: (y, mu) => {
      // No true log-likelihood for quasi families.
      // Return the negative quasi-deviance contribution: -0.5 * (y - mu)^2 / V(mu)
      const v = Math.max(1e-10, varianceFn(mu));
      return -0.5 * ((y - mu) ** 2) / v;
    },
  };
}

// -- Dispersion Estimation ----------------------------------------------------

/**
 * Estimate the dispersion parameter using the Pearson estimator.
 *
 * phi = sum((y_i - mu_i)^2 / V(mu_i)) / (n - p)
 *
 * @param X - Design matrix (n x p), without intercept column.
 * @param y - Response variable.
 * @param family - GLM family used for fitting.
 * @param coefficients - Estimated coefficients (including intercept as first element).
 * @returns The estimated dispersion parameter.
 */
export function estimateDispersion(
  X: number[][],
  y: Dataset,
  family: GLMFamily,
  coefficients: number[],
): number {
  const n = X.length;
  const p = coefficients.length; // includes intercept

  let pearsonChi2 = 0;
  for (let i = 0; i < n; i++) {
    let eta = coefficients[0];
    for (let j = 0; j < X[i].length; j++) {
      eta += coefficients[j + 1] * X[i][j];
    }
    const mu = family.link.inverse(eta);
    const v = family.variance(mu);
    pearsonChi2 += ((y[i] - mu) ** 2) / Math.max(v, 1e-10);
  }

  return pearsonChi2 / (n - p);
}

// -- GLM with Dispersion ------------------------------------------------------

/**
 * Extended GLM result that includes dispersion-adjusted inference.
 */
export interface GLMDispersionResult extends GLMResult {
  /** Estimated dispersion parameter (Pearson estimator). */
  dispersion: number;
  /** Standard errors adjusted by sqrt(dispersion). */
  adjustedStandardErrors: number[];
  /** p-values recomputed using adjusted standard errors. */
  adjustedPValues: number[];
}

/**
 * Fit a GLM and estimate the dispersion parameter, adjusting standard errors accordingly.
 *
 * First fits the model using IRLS via {@link glm}, then computes the Pearson
 * dispersion estimate and scales the standard errors by sqrt(dispersion).
 *
 * @param X - Design matrix (n x p), without intercept column.
 * @param y - Response variable.
 * @param family - GLM family.
 * @param options - Configuration (same as {@link glm} options).
 * @returns A {@link GLMDispersionResult} with dispersion-adjusted inference.
 *
 * @example
 * ```ts
 * const X = [[1], [2], [3], [4], [5]];
 * const y = [2.1, 3.9, 6.2, 7.8, 10.1];
 * const result = glmWithDispersion(X, y, gaussian);
 * console.log(result.dispersion); // estimated dispersion
 * console.log(result.adjustedStandardErrors); // scaled standard errors
 * ```
 */
export function glmWithDispersion(
  X: number[][],
  y: Dataset,
  family: GLMFamily,
  options: {
    maxIterations?: number;
    tol?: number;
  } = {},
): GLMDispersionResult {
  const result = glm(X, y, family, options);

  const dispersion = estimateDispersion(X, y, family, result.coefficients);
  const sqrtDisp = Math.sqrt(dispersion);

  const adjustedStandardErrors = result.standardErrors.map((se) => se * sqrtDisp);
  const adjustedPValues = adjustedStandardErrors.map((ase, j) => {
    const z = ase > 0 ? result.coefficients[j] / ase : 0;
    return 2 * (1 - normalCdf(Math.abs(z)));
  });

  return {
    ...result,
    dispersion,
    adjustedStandardErrors,
    adjustedPValues,
  };
}
