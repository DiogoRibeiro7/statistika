# API Reference

The full API reference for statistika is generated using [TypeDoc](https://typedoc.org/) from the TypeScript source code.

## Generating API Docs

To generate the API documentation locally:

```bash
yarn docs
```

This creates HTML documentation in the `docs/` directory using the TypeDoc configuration in `typedoc.json`.

## Module Overview

statistika is organized into the following modules, each available as a sub-path export:

### Core

| Module | Import Path | Description |
|---|---|---|
| **Distributions** | `statistika/distributions` | 23 probability distributions (Normal, Beta, Poisson, etc.) |
| **Models** | `statistika/models` | Regression models (linear, multiple, polynomial, logistic, robust) |
| **Tests** | `statistika/tests` | Hypothesis tests (t-test, chi-squared, ANOVA, KS, etc.) |
| **Utils** | `statistika/utils` | Descriptive statistics, linear algebra, special math functions |

### Statistical Modeling

| Module | Import Path | Description |
|---|---|---|
| **GLM** | `statistika/glm` | Generalized linear models (Gaussian, Binomial, Poisson, Gamma) |
| **GAM** | `statistika/gam` | Generalized additive models |
| **Bayesian** | `statistika/bayesian` | Conjugate models, Bayes factors |
| **MCMC** | `statistika/mcmc` | Metropolis-Hastings, Gelman-Rubin, ESS |

### Time Series

| Module | Import Path | Description |
|---|---|---|
| **Time Series** | `statistika/time-series` | ARIMA, auto-ARIMA, ACF/PACF, seasonal decomposition, ADF test |
| **GARCH** | `statistika/garch` | ARCH, GARCH, EGARCH, GJR-GARCH volatility models |
| **VAR** | `statistika/var` | Vector autoregression, Granger causality, impulse response |
| **State Space** | `statistika/state-space` | Kalman filter, state space models |
| **Changepoint** | `statistika/changepoint` | Changepoint detection |

### Specialized Analysis

| Module | Import Path | Description |
|---|---|---|
| **Survival** | `statistika/survival` | Kaplan-Meier, Nelson-Aalen, log-rank test, Cox regression |
| **Causal Inference** | `statistika/causal-inference` | Propensity scores, matching, DiD, RDD, 2SLS |
| **Resampling** | `statistika/resampling` | Cross-validation, bootstrap, jackknife |
| **Diagnostics** | `statistika/diagnostics` | Regression diagnostics, VIF, residual analysis |
| **Meta-analysis** | `statistika/meta-analysis` | Fixed/random effects meta-analysis |
| **Spatial** | `statistika/spatial` | Spatial statistics |
| **Dimensionality** | `statistika/dimensionality` | PCA, factor analysis, t-SNE |

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
