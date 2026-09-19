# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0] - 2026-09-19

### Added
- A substantially broader statistical modelling surface, including multiple, polynomial, logistic, robust, and quantile regression; GLMs and GAMs; mixed models; SEM; survival analysis and Cox regression.
- Bayesian inference tooling including conjugate models, Bayes factors, multidimensional Metropolis-Hastings MCMC, convergence diagnostics, variational inference, and Bayesian networks.
- Time-series and forecasting methods including ARIMA, automatic ARIMA, GARCH-family models, VAR, state-space models, Kalman filtering, seasonal decomposition, and changepoint detection.
- Causal inference methods including propensity scores, matching, inverse-probability weighting, difference-in-differences, regression discontinuity, and 2SLS.
- Meta-analysis, spatial statistics, dimensionality reduction, clustering, functional and compositional data methods, missing-data tools, resampling, experimental design, effect sizes, power analysis, and multiple-testing corrections.
- Streaming and observable statistics, lazy data-frame operations, worker-thread utilities, WASM support, and expanded native Fortran acceleration.
- VitePress and TypeDoc documentation, benchmark regression checks, GitHub Pages deployment, GitHub Packages publishing, release validation, package-content verification, and clean consumer-install smoke tests.

### Changed
- The package is published as `@diogoribeiro7/statistika`.
- Node.js 22 or newer is now required and CI validates Node.js 22 and 24.
- CI and release automation use Yarn 4 with immutable installs and current Node 24-based GitHub Actions.
- Native Fortran acceleration is optional. Environments without a usable native backend fall back to the TypeScript implementation.
- Native linear-algebra acceleration is capability-checked so incomplete no-LAPACK builds are not loaded as valid backends.
- `SeededRng` now uses SplitMix32-seeded `xoshiro128**`. Seeded sequences therefore differ from earlier releases.

### Fixed
- Corrected the seeded RNG so its uniform output has the expected first two moments, removing bias that could distort Monte Carlo and MCMC results.
- Hardened no-LAPACK native builds and runtime fallback behavior.
- Made benchmark comparison fail closed on suite errors and material performance regressions.
- Hardened native build command execution by removing shell-interpolated command strings.

### Security
- Native build subprocesses now use explicit executable/argument invocation rather than shell execution.

## [1.0.0] - 2025-01-01

### Added
- Descriptive statistics: mean, median, variance, standard deviation, describe
- Simple linear regression with R² and prediction
- Probability distributions (continuous): Normal, Exponential, Uniform, Gamma, Beta, StudentT, ChiSquared, F, Cauchy, Pareto, LogNormal
- Probability distributions (discrete): Binomial, Poisson, Geometric, DiscreteUniform, NegativeBinomial, Hypergeometric
- Extreme value distributions: GEV, Gumbel, Fréchet, Weibull, GPD
- Hypothesis tests: one-sample/two-sample/Welch's/paired t-tests, chi-squared goodness-of-fit and independence, one-way ANOVA, KS test (one-sample and two-sample)
- Special math functions: gamma, gammaLn, factorial, binomialCoeff, beta, erf, erfc, regularizedGammaP, regularizedBeta
- Optional native Fortran acceleration for special math functions
- Full TypeScript type declarations
