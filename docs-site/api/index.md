# API Reference

The full API reference for node_stats is generated using [TypeDoc](https://typedoc.org/) from the TypeScript source code.

## Generating API Docs

To generate the API documentation locally:

```bash
yarn docs
```

This creates HTML documentation in the `docs/` directory using the TypeDoc configuration in `typedoc.json`.

## Module Overview

node_stats is organized into the following modules, each available as a sub-path export:

### Core

| Module | Import Path | Description |
|---|---|---|
| **Distributions** | `node_stats/distributions` | 23 probability distributions (Normal, Beta, Poisson, etc.) |
| **Models** | `node_stats/models` | Regression models (linear, multiple, polynomial, logistic, robust) |
| **Tests** | `node_stats/tests` | Hypothesis tests (t-test, chi-squared, ANOVA, KS, etc.) |
| **Utils** | `node_stats/utils` | Descriptive statistics, linear algebra, special math functions |

### Statistical Modeling

| Module | Import Path | Description |
|---|---|---|
| **GLM** | `node_stats/glm` | Generalized linear models (Gaussian, Binomial, Poisson, Gamma) |
| **GAM** | `node_stats/gam` | Generalized additive models |
| **Bayesian** | `node_stats/bayesian` | Conjugate models, Bayes factors |
| **MCMC** | `node_stats/mcmc` | Metropolis-Hastings, Gelman-Rubin, ESS |

### Time Series

| Module | Import Path | Description |
|---|---|---|
| **Time Series** | `node_stats/time-series` | ARIMA, auto-ARIMA, ACF/PACF, seasonal decomposition, ADF test |
| **GARCH** | `node_stats/garch` | ARCH, GARCH, EGARCH, GJR-GARCH volatility models |
| **VAR** | `node_stats/var` | Vector autoregression, Granger causality, impulse response |
| **State Space** | `node_stats/state-space` | Kalman filter, state space models |
| **Changepoint** | `node_stats/changepoint` | Changepoint detection |

### Specialized Analysis

| Module | Import Path | Description |
|---|---|---|
| **Survival** | `node_stats/survival` | Kaplan-Meier, Nelson-Aalen, log-rank test, Cox regression |
| **Causal Inference** | `node_stats/causal-inference` | Propensity scores, matching, DiD, RDD, 2SLS |
| **Resampling** | `node_stats/resampling` | Cross-validation, bootstrap, jackknife |
| **Diagnostics** | `node_stats/diagnostics` | Regression diagnostics, VIF, residual analysis |
| **Meta-analysis** | `node_stats/meta-analysis` | Fixed/random effects meta-analysis |
| **Spatial** | `node_stats/spatial` | Spatial statistics |
| **Dimensionality** | `node_stats/dimensionality` | PCA, factor analysis, t-SNE |

## Key Types

All result types are exported from their respective modules. Here are the most commonly used:

```typescript
// Hypothesis tests return TestResult
interface TestResult {
  statistic: number;
  pValue: number;
  rejected: boolean;
}

// Distributions share a common interface
interface ContinuousDistribution {
  mean(): number;
  variance(): number;
  stdDev(): number;
  pdf(x: number): number;
  cdf(x: number): number;
  sf(x: number): number;
  quantile(p: number): number;
  sample(): number;
  sampleN(n: number): number[];
}

// Regression results
interface LinearRegressionResult {
  slope: number;
  intercept: number;
  rSquared: number;
  predict(x: number): number;
}
```

For complete type definitions, refer to the TypeDoc-generated documentation or the TypeScript source files in `src/`.
