# node_stats — Feature Plan

## Current State (v1.0.0)

**node_stats** is a comprehensive statistical library for Node.js written in TypeScript (~18.5k lines of source) with optional native Fortran/LAPACK acceleration. All 16 originally planned features have been implemented:

### What's Shipped

| Module | Features |
|--------|----------|
| **Distributions** (27) | Normal, Uniform, Exponential, Gamma, Beta, Chi-squared, Student-t, F, Log-Normal, Weibull, GEV, Gumbel, Frechet, GPD, Cauchy, Pareto, Bernoulli, Binomial, Poisson, Geometric, Discrete Uniform, Negative Binomial, Hypergeometric, Zero-Inflated Poisson, Multivariate Normal, Dirichlet, Wishart, Multinomial |
| **Matrix/LinAlg** | Matrix class with LU, QR, Cholesky, SVD, eigenvalue decomposition; native LAPACK acceleration |
| **Regression** | Linear, multiple, polynomial, logistic, robust (Huber, RANSAC), quantile, Ridge, LASSO, Elastic Net |
| **Feature Selection** | Forward/backward stepwise (AIC/BIC), mutual information ranking |
| **GLM** | Gaussian, Binomial, Poisson, Gamma families with identity/log/logit/probit/inverse links |
| **Hypothesis Tests** | t-tests, chi-squared, ANOVA (one/two-way), KS, Mann-Whitney U, Wilcoxon, Fisher's exact |
| **Bayesian** | Conjugate models (Beta-Binomial, Normal-Normal, Gamma-Poisson), Metropolis-Hastings MCMC, Bayes factors |
| **Time Series** | ACF/PACF, ARIMA, SARIMA, auto-ARIMA, seasonal decomposition, ADF test, Prophet-style decomposition |
| **Survival** | Kaplan-Meier, Nelson-Aalen, log-rank test, Cox proportional hazards |
| **ML/Dimensionality** | PCA, factor analysis, k-means, hierarchical clustering, t-SNE, GMM |
| **Causal Inference** | Propensity scores, IPW, matching, Difference-in-Differences, 2SLS, RDD |
| **Spatial** | Moran's I, Geary's C, variogram, kriging |
| **Graph/Network** | Centrality measures, community detection, random graph models |
| **Survey** | Weighted stats, design effects, stratified/cluster sampling, Horvitz-Thompson |
| **SEM** | Path analysis, CFA, fit indices (CFI, TLI, RMSEA, SRMR) |
| **Mixed Models** | Random intercepts/slopes, REML, ICC, BLUPs |
| **Other** | Bayesian networks, functional data analysis, compositional data, experimental design, streaming stats, resampling/CV, robust statistics, information theory, missing data, smoothing, effect sizes, power analysis, DataFrame, viz-data |

**Health**: 87 test suites, all passing. CI/CD with GitHub Actions. Dual CJS/ESM build.

---

## Proposed New Features (v2.0 Roadmap)

### Phase 1 — High Impact, Fills Clear Gaps

#### 1. GARCH / Volatility Models (`src/garch.ts`)
Essential for financial time series. Currently missing entirely.
- ARCH(q) and GARCH(p,q) parameter estimation (MLE)
- EGARCH (exponential GARCH) for asymmetric volatility
- GJR-GARCH (threshold effects)
- Volatility forecasting with confidence intervals
- Standardized residual diagnostics
- ARCH-LM test for conditional heteroscedasticity

#### 2. Vector Autoregression (`src/var.ts`)
Multivariate time series is a major gap — only univariate ARIMA exists.
- VAR(p) model estimation (OLS, equation by equation)
- Lag order selection (AIC, BIC, HQ criteria)
- Granger causality tests
- Impulse response functions (IRF) with bootstrapped CIs
- Forecast error variance decomposition (FEVD)
- VAR forecasting with fan charts
- Cointegration testing (Johansen trace/max-eigenvalue)

#### 3. Gibbs Sampling (`src/mcmc.ts` extension)
Only Metropolis-Hastings exists. Gibbs is the most-used MCMC method for hierarchical models.
- Component-wise Gibbs sampler
- Block Gibbs sampling
- Integration with conjugate priors for efficient conditionals
- Hierarchical model support (e.g., Normal-Inverse-Gamma)
- Convergence diagnostics (trace plots data, autocorrelation of chain)

#### 4. Additional Distributions
Fill gaps in the distribution library for common applied use cases.
- **Truncated Normal** — widely used in applied statistics, censored data
- **Laplace (Double Exponential)** — robust statistics, Bayesian LASSO prior
- **Inverse Gamma** — conjugate prior for Normal variance
- **Inverse Wishart** — conjugate prior for covariance matrices
- **Multivariate t-distribution** — robust multivariate analysis
- **Rayleigh** — wind speed, wave height modeling
- **Log-Logistic** — survival analysis alternative to Weibull
- **von Mises** — directional/circular statistics

#### 5. Generalized Additive Models (`src/gam.ts`)
Bridges the gap between GLM and fully nonparametric methods.
- Penalized regression splines (cubic, thin-plate)
- Backfitting algorithm for additive model estimation
- Smoothness selection (GCV, REML)
- Partial dependence plots data
- Support for all existing GLM families/links
- Tensor product smooths for interaction terms

### Phase 2 — Valuable Extensions

#### 6. State-Space Models & Kalman Filter (`src/state-space.ts`)
Powerful framework that unifies many time series methods.
- Linear Gaussian state-space model representation
- Kalman filter (forward pass)
- Kalman smoother (backward pass)
- EM algorithm for parameter estimation
- Missing observation handling
- Local level, local linear trend, and structural TS models
- Prediction intervals

#### 7. Changepoint Detection (`src/changepoint.ts`)
Detect structural breaks in time series — commonly needed in monitoring/alerting.
- CUSUM (cumulative sum) test
- PELT algorithm (Pruned Exact Linear Time)
- Binary segmentation
- Change in mean, variance, or both
- Multiple changepoint detection with penalty selection (BIC, mBIC)
- Online changepoint detection (BOCPD — Bayesian Online CPD)

#### 8. Expanded Regression Diagnostics (`src/diagnostics.ts` extension)
The current diagnostics module covers basics; production use needs more.
- **Breusch-Pagan test** — heteroscedasticity
- **White's test** — heteroscedasticity (general form)
- **Ramsey RESET test** — functional form specification
- **Cook's distance** — influential observation detection
- **DFBETAS / DFFITS** — per-observation influence measures
- **Leverage (hat) values** — high-leverage point detection
- **Added variable (partial regression) plots** data
- **Condition number** — numerical stability of design matrix

#### 9. Negative Binomial & Tweedie GLM Families (`src/glm.ts` extension)
The two most-requested GLM families missing from the current implementation.
- **Negative Binomial** — overdispersed count data (very common in genomics, ecology)
- **Tweedie** — compound Poisson-Gamma for insurance claims, zero-inflated continuous data
- **Complementary log-log link** — asymmetric binary response
- **Quasi-likelihood** — when the distribution is unknown but variance function is specified
- **Dispersion parameter estimation** — currently assumed fixed

#### 10. Advanced Resampling Methods (`src/resampling.ts` extension)
Current resampling is solid but missing time-series and robust variants.
- **Block bootstrap** — moving block, circular block, stationary bootstrap for time series
- **Wild bootstrap** — for heteroscedastic regression residuals
- **Bayesian bootstrap** — Dirichlet-weighted resampling
- **Time-series cross-validation** — expanding window, rolling window, blocked
- **Nested cross-validation** — for hyperparameter tuning + model evaluation
- **Monte Carlo cross-validation** — repeated random splits

### Phase 3 — Advanced / Research-Grade

#### 11. Hamiltonian Monte Carlo (`src/mcmc.ts` extension)
More efficient than MH for high-dimensional posteriors.
- Leapfrog integrator
- NUTS (No-U-Turn Sampler) for automatic tuning
- Mass matrix adaptation (diagonal and dense)
- Dual averaging for step size tuning
- Divergence detection and reporting

#### 12. Variational Inference (`src/variational.ts`)
Fast approximate Bayesian inference — alternative to MCMC for large datasets.
- Mean-field variational Bayes (CAVI algorithm)
- ELBO monitoring for convergence
- Automatic differentiation variational inference (ADVI) style transforms
- Variational families: Normal, log-Normal, multivariate Normal
- Model comparison via ELBO

#### 13. Meta-Analysis (`src/meta-analysis.ts`)
Combining results across studies — used heavily in medicine, social science.
- Fixed-effects model (inverse variance weighting)
- Random-effects model (DerSimonian-Laird)
- Heterogeneity statistics (Q, I², tau²)
- Forest plot data generation
- Funnel plot data (publication bias detection)
- Egger's test, Begg's test for publication bias
- Trim-and-fill method

#### 14. Item Response Theory (`src/irt.ts`)
Foundational for psychometrics, educational testing, survey research.
- 1PL (Rasch) model
- 2PL model (discrimination + difficulty)
- 3PL model (+ guessing parameter)
- Graded Response Model (ordinal)
- MLE and MAP ability estimation
- Item and test information functions
- Item characteristic curves data

#### 15. DBSCAN & Spectral Clustering (`src/dimensionality.ts` extension)
K-means and hierarchical are implemented; density-based and spectral methods are missing.
- **DBSCAN** — density-based clustering, no k required, handles noise
- **OPTICS** — ordered points, more flexible than DBSCAN
- **Spectral clustering** — using graph Laplacian eigenvectors
- **Silhouette score** — cluster quality evaluation
- **UMAP** — modern dimensionality reduction (often preferred over t-SNE)

#### 16. Sensitivity Analysis (`src/sensitivity.ts`)
Quantify how model outputs respond to input variation — essential for simulation and uncertainty.
- **Sobol indices** (first-order, total-order)
- **Morris method** (elementary effects screening)
- **FAST (Fourier Amplitude Sensitivity Testing)**
- **Scatter plot / correlation-based screening**
- Works with any model callable as a function

---

## Implementation Order Recommendation

| Phase | Features | Rationale |
|-------|----------|-----------|
| **Phase 1** | GARCH (#1), VAR (#2), Gibbs (#3), Distributions (#4), GAM (#5) | Biggest user-facing gaps; finance & Bayesian users need these |
| **Phase 2** | State-space (#6), Changepoint (#7), Diagnostics (#8), NB/Tweedie GLM (#9), Resampling (#10) | Completes time-series and regression stories |
| **Phase 3** | HMC (#11), Variational (#12), Meta-analysis (#13), IRT (#14), Clustering (#15), Sensitivity (#16) | Research-grade features for specialized domains |

---

## Non-Functional Improvements

### Performance
- **WASM acceleration**: Compile hot-path matrix ops to WebAssembly for environments without gfortran
- **Typed arrays everywhere**: Migrate remaining `number[][]` matrix ops to `Float64Array` for cache-friendly access
- **Lazy evaluation in DataFrame**: Only compute columns when accessed
- **Worker thread support**: Offload large MCMC chains and bootstrap runs to worker threads

### Developer Experience
- **Documentation site**: Tutorials with worked examples (not just API reference)
- **Expanded benchmarks**: Cover regression, GLM, MCMC, and clustering modules (currently only special functions and linalg)
- **JSDoc @example tags**: Ensure every public function has a runnable example
- **Error messages**: Include parameter names and valid ranges in all validation errors

### Ecosystem
- **Sub-path exports**: Already partially done (`node_stats/distributions`, etc.) — extend to all modules
- **Standalone packages**: Consider publishing distributions, time-series, and bayesian as standalone npm packages
- **Observable/RxJS integration**: Streaming stats adapter for Observable streams
- **JSON schema**: Publish JSON schemas for result types (useful for API responses)

### Testing
- **Property-based testing**: Add generative tests for distributions (e.g., CDF(quantile(p)) ≈ p)
- **Numerical accuracy benchmarks**: Compare against R/SciPy reference values
- **Fuzz testing**: Random inputs to catch edge cases in special functions
- **Performance regression tests**: Track benchmark results in CI
