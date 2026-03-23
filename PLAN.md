# node_stats — Feature Plan

## Repository Summary

**node_stats** is a comprehensive statistical library for Node.js written in TypeScript (~15k lines of source) with optional native Fortran acceleration. It currently ships:

- **27 probability distributions** (17 continuous, 7 discrete, 3 multivariate)
- **Hypothesis tests** — t-tests, chi-squared, ANOVA, KS, Mann-Whitney, Wilcoxon, Fisher's exact
- **Regression** — linear, multiple, polynomial, logistic, robust, quantile, Cox PH
- **GLM** — Gaussian, Binomial, Poisson, Gamma families
- **Bayesian** — conjugate models, Metropolis-Hastings MCMC, Bayes factors
- **Time series** — ARIMA, auto-ARIMA, seasonal decomposition, ADF test
- **Machine learning** — PCA, factor analysis, k-means, hierarchical clustering, t-SNE, GMM
- **Survival** — Kaplan-Meier, Nelson-Aalen, log-rank, Cox regression
- **Plus**: nonparametric methods, resampling/CV, robust statistics, information theory, distance metrics, missing data, streaming stats, multiple testing corrections, smoothing, effect sizes, power analysis, confidence intervals

**Health**: 69 test suites, 1,405 tests all passing. CI/CD with GitHub Actions. Dual CJS/ESM build.

---

## Proposed New Features

### Priority 1 — High Impact, Fills Clear Gaps

#### 1. Matrix / Linear Algebra Module (`src/matrix.ts`)
The library relies on hand-rolled array operations throughout (e.g., in PCA, GLM, regression). A proper `Matrix` class would:
- Provide creation helpers (`zeros`, `identity`, `fromArray`, `diagonal`)
- Core ops: add, multiply, transpose, inverse, determinant
- Decompositions: LU, QR, Cholesky, SVD, eigenvalue
- Solve linear systems (`Ax = b`)
- This would simplify and speed up regression, GLM, PCA, factor analysis, and multivariate distributions

#### 2. Feature Selection & Variable Importance (`src/feature-selection.ts`)
- Forward/backward stepwise selection (AIC, BIC criteria)
- LASSO (L1) and Ridge (L2) regularized regression
- Elastic Net
- Variable importance scores (permutation-based)
- Mutual information-based feature ranking (leveraging existing info-theory module)

#### 3. Spatial Statistics (`src/spatial.ts`)
- Spatial autocorrelation (Moran's I, Geary's C)
- Variogram estimation and kriging
- Nearest neighbor analysis
- Point pattern analysis (Ripley's K)

#### 4. Causal Inference (`src/causal.ts`)
- Propensity score matching
- Difference-in-differences estimator
- Instrumental variables (2SLS)
- Regression discontinuity design
- Average treatment effect (ATE/ATT/ATU) estimation

### Priority 2 — Valuable Extensions

#### 5. Network / Graph Statistics (`src/graph-stats.ts`)
- Centrality measures (degree, betweenness, closeness, eigenvector)
- Community detection (modularity-based)
- Graph density, diameter, clustering coefficient
- Random graph models (Erdős-Rényi, Barabási-Albert)

#### 6. Functional Data Analysis (`src/functional.ts`)
- Basis expansion (B-spline, Fourier)
- Functional PCA
- Functional linear models
- Curve registration and alignment

#### 7. Survey Statistics (`src/survey.ts`)
- Weighted descriptive statistics
- Design effects (DEFF)
- Stratified and cluster sampling estimators
- Ratio and regression estimators
- Finite population correction
- Horvitz-Thompson estimator

#### 8. Compositional Data Analysis (`src/compositional.ts`)
- Aitchison geometry (CLR, ALR, ILR transforms)
- Compositional mean, variation matrix
- Compositional regression
- Ternary plot data generation

### Priority 3 — Polish & Infrastructure

#### 9. Data Frame / Table Abstraction (`src/dataframe.ts`)
- Column-oriented typed data container
- Group-by / aggregate operations
- Join, filter, sort, pivot
- Integration point: pass directly to regression/GLM/tests
- CSV/JSON import helpers

#### 10. Statistical Visualization Data (`src/viz-data.ts`)
- Generate plot-ready data structures (not rendering, just data)
- Histogram bins, KDE curves, Q-Q plot coordinates
- Box plot statistics (quartiles, whiskers, outliers)
- Scatter plot with regression line data
- Correlation heatmap matrix

#### 11. Model Comparison & Selection (`src/model-selection.ts`)
- AIC, BIC, AICc computation for any fitted model
- Likelihood ratio tests
- Vuong test for non-nested models
- Cross-validation wrappers for model comparison
- Standardized model result interface

#### 12. Expanded Distribution Toolkit
- **New distributions**: Multinomial, Negative Hypergeometric, Zero-Inflated Poisson, Truncated Normal, Mixture distributions (arbitrary)
- **Distribution fitting**: MLE parameter estimation from data for all distributions
- **Goodness-of-fit**: Anderson-Darling, Cramér-von Mises, Lilliefors tests
- **QQ-plot data generation** (ties into viz-data)

### Priority 4 — Advanced / Research-Grade

#### 13. Structural Equation Modeling (`src/sem.ts`)
- Path analysis
- Confirmatory factor analysis (CFA)
- Model fit indices (CFI, TLI, RMSEA, SRMR)
- Mediation and moderation analysis

#### 14. Bayesian Networks (`src/bayesian-networks.ts`)
- DAG structure representation
- Conditional probability tables
- Exact inference (variable elimination)
- Structure learning (PC algorithm, score-based)

#### 15. Multi-Level / Mixed Effects Models (`src/mixed-models.ts`)
- Random intercepts and slopes
- Nested and crossed random effects
- REML estimation
- ICC (intra-class correlation)

#### 16. Experimental Design (`src/experimental-design.ts`)
- Factorial design generation (full and fractional)
- Latin squares, Graeco-Latin squares
- Block design
- Response surface methodology
- Optimal design (D-optimal, A-optimal)

---

## Implementation Order Recommendation

| Phase | Features | Rationale |
|-------|----------|-----------|
| **Phase 1** | Matrix module (#1) | Foundation for everything else; unblocks improvements across all modules |
| **Phase 2** | Feature selection (#2), Model selection (#11), Distribution toolkit (#12) | High user demand, leverages existing code |
| **Phase 3** | Causal inference (#4), Survey statistics (#7), Data frame (#9) | Domain-specific value, attracts new user segments |
| **Phase 4** | Spatial (#3), Network (#5), Viz data (#10) | Broadens library scope |
| **Phase 5** | Mixed models (#15), SEM (#13), Experimental design (#16) | Research-grade features |
| **Phase 6** | Functional data (#6), Compositional (#8), Bayesian networks (#14) | Specialized domains |

---

## Non-Functional Improvements

- **Bundle size**: Add tree-shaking friendly sub-path exports (`node_stats/distributions`, `node_stats/regression`, etc.)
- **Performance**: Profile hot paths (matrix ops, distribution sampling) and add WASM or Fortran acceleration
- **Documentation**: Add a docs site with tutorials, not just API reference
- **Benchmarks**: Expand benchmark suite to cover regression, GLM, ML modules
- **TypeDoc**: Ensure all public APIs have JSDoc with `@example` tags
