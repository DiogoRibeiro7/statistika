# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Multiple linear regression via OLS normal equations
- Polynomial regression (arbitrary degree)
- Binary logistic regression via IRLS (Newton's method)
- Pearson, Spearman, and Kendall correlation with p-values
- Mann-Whitney U test (non-parametric, two independent samples)
- Wilcoxon signed-rank test (non-parametric, paired samples)
- Fisher's exact test for 2x2 contingency tables
- Improved package publishing readiness (metadata, optional native deps, changelog)

### Changed
- Native Fortran addon is now optional — `npm install` no longer fails without gfortran
- Moved `node-addon-api` and `node-gyp` from `dependencies` to `optionalDependencies`

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
