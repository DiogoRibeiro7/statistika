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
| **Distributions** | `@diogoribeiro7/statistika/distributions` | 37 probability distributions (Normal, Beta, Poisson, etc.) |
| **Models** | `@diogoribeiro7/statistika/models` | Regression models (linear, multiple, polynomial, logistic, robust) |
| **Tests** | `@diogoribeiro7/statistika/tests` | Hypothesis tests (t-test, chi-squared, ANOVA, KS, etc.) |
| **Utils** | `@diogoribeiro7/statistika/utils` | Descriptive statistics, linear algebra, special math functions |

### Statistical Modeling

| Module | Import Path | Description |
|---|---|---|
| **GLM** | `@diogoribeiro7/statistika/glm` | Generalized linear models (Gaussian, Binomial, Poisson, Gamma) |
| **GAM** | `@diogoribeiro7/statistika/gam` | Generalized additive models |
| **Bayesian** | `@diogoribeiro7/statistika/bayesian` | Conjugate models, Bayes factors |
| **MCMC** | `@diogoribeiro7/statistika/mcmc` | Metropolis-Hastings, Gelman-Rubin, ESS |

### Time Series

| Module | Import Path | Description |
|---|---|---|
| **Time Series** | `@diogoribeiro7/statistika/time-series` | ARIMA, auto-ARIMA, ACF/PACF, seasonal decomposition, ADF test |
| **GARCH** | `@diogoribeiro7/statistika/garch` | ARCH, GARCH, EGARCH, GJR-GARCH volatility models |
| **VAR** | `@diogoribeiro7/statistika/var` | Vector autoregression, Granger causality, impulse response |
| **State Space** | `@diogoribeiro7/statistika/state-space` | Kalman filter, state space models |
| **Changepoint** | `@diogoribeiro7/statistika/changepoint` | Changepoint detection |

### Specialized Analysis

| Module | Import Path | Description |
|---|---|---|
| **Survival** | `@diogoribeiro7/statistika/survival` | Kaplan-Meier, Nelson-Aalen, log-rank test, Cox regression |
| **Causal Inference** | `@diogoribeiro7/statistika/causal-inference` | Propensity scores, matching, DiD, RDD, 2SLS |
| **Resampling** | `@diogoribeiro7/statistika/resampling` | Cross-validation, bootstrap, jackknife |
| **Diagnostics** | `@diogoribeiro7/statistika/diagnostics` | Regression diagnostics, VIF, residual analysis |
| **Meta-analysis** | `@diogoribeiro7/statistika/meta-analysis` | Fixed/random effects meta-analysis |
| **Spatial** | `@diogoribeiro7/statistika/spatial` | Spatial statistics |
| **Dimensionality** | `@diogoribeiro7/statistika/dimensionality` | PCA, factor analysis, t-SNE |

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
