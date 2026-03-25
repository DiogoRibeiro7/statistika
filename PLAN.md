# node_stats — Feature Plan

## Current State (v2.0.0 → v3.0 in progress)

**node_stats** is a comprehensive statistical library for Node.js written in TypeScript with optional native Fortran/LAPACK acceleration and WASM support. All originally planned v1.0 and v2.0 features have been implemented. v3.0 work is underway.

### What's Shipped

| Module | Features |
|--------|----------|
| **Distributions** (36+) | Normal, Uniform, Exponential, Gamma, Beta, Chi-squared, Student-t, F, Log-Normal, Weibull, GEV, Gumbel, Frechet, GPD, Cauchy, Pareto, Laplace, Inverse Gamma, Log-Logistic, Truncated Normal, Rayleigh, von Mises, **Lévy**, Bernoulli, Binomial, Poisson, Geometric, Discrete Uniform, Negative Binomial, Hypergeometric, Zero-Inflated Poisson, Multivariate Normal, Multivariate t, Dirichlet, Wishart, Inverse Wishart, Multinomial |
| **Matrix/LinAlg** | Matrix class with LU, QR, Cholesky, SVD, eigenvalue decomposition; native LAPACK acceleration; WASM acceleration |
| **Regression** | Linear, multiple, polynomial, logistic, robust (Huber, RANSAC), quantile, Ridge, LASSO, Elastic Net |
| **Feature Selection** | Forward/backward stepwise (AIC/BIC), mutual information ranking |
| **GLM** | Gaussian, Binomial, Poisson, Gamma, Negative Binomial, Tweedie families; identity/log/logit/probit/inverse/cloglog links; quasi-likelihood; dispersion estimation |
| **GAM** | Penalized regression splines (cubic, thin-plate), backfitting, GCV smoothness selection, partial dependence plots |
| **Hypothesis Tests** | t-tests, chi-squared, ANOVA (one/two-way), KS, Mann-Whitney U, Wilcoxon, Fisher's exact |
| **Regression Diagnostics** | Breusch-Pagan, White's test, Ramsey RESET, Cook's distance, DFBETAS/DFFITS, leverage, condition number |
| **Bayesian** | Conjugate models, Metropolis-Hastings, Gibbs (component/block/hierarchical), HMC, NUTS, Bayes factors |
| **Variational Inference** | Mean-field CAVI, ELBO monitoring, ADVI transforms, model comparison |
| **Time Series** | ACF/PACF, ARIMA, SARIMA, auto-ARIMA, seasonal decomposition, ADF test, Prophet-style decomposition |
| **GARCH** | ARCH, GARCH, EGARCH, GJR-GARCH, volatility forecasting, ARCH-LM test |
| **VAR** | VAR(p) estimation, lag selection, Granger causality, IRF, FEVD, Johansen cointegration |
| **State-Space** | Kalman filter/smoother, EM estimation, missing data handling, structural TS models, native Fortran acceleration |
| **Changepoint** | CUSUM, PELT, binary segmentation, mean/variance shifts, BOCPD online detection |
| **Survival** | Kaplan-Meier, Nelson-Aalen, log-rank test, Cox proportional hazards |
| **ML/Dimensionality** | PCA, factor analysis, k-means, hierarchical clustering, t-SNE, UMAP, GMM, DBSCAN, OPTICS, spectral clustering, silhouette score |
| **Causal Inference** | Propensity scores, IPW, matching, Difference-in-Differences, 2SLS, RDD |
| **Spatial** | Moran's I, Geary's C, variogram, kriging |
| **Meta-Analysis** | Fixed/random effects, heterogeneity (Q, I², τ²), forest/funnel plots, Egger's/Begg's tests, trim-and-fill |
| **IRT** | 1PL, 2PL, 3PL, Graded Response Model, ability estimation, information functions |
| **Sensitivity Analysis** | Sobol indices, Morris method, FAST, correlation-based screening |
| **Graph/Network** | Centrality measures, community detection, random graph models |
| **Survey** | Weighted stats, design effects, stratified/cluster sampling, Horvitz-Thompson |
| **SEM** | Path analysis, CFA, fit indices (CFI, TLI, RMSEA, SRMR) |
| **Mixed Models** | Random intercepts/slopes, REML, ICC, BLUPs |
| **Resampling** | k-fold CV, LOO-CV, jackknife, bootstrap, permutation tests, block/wild/Bayesian bootstrap, time-series CV, nested CV, Monte Carlo CV |
| **Worker Threads** | Parallel MCMC chains, parallel bootstrap, parallel cross-validation via worker_threads |
| **WASM Acceleration** | Special functions (gammaLn, gamma, erf, erfc, betaFn) and linear algebra (matMul, solve, cholesky) via WebAssembly with TS fallback |
| **Other** | Bayesian networks, functional data analysis, compositional data, experimental design, streaming stats, robust statistics, information theory, missing data, smoothing, effect sizes, power analysis, DataFrame, viz-data |

**Health**: 105+ test suites, all passing. CI/CD with GitHub Actions (CodeQL, dependency review, benchmarks). Dual CJS/ESM build. Native Fortran acceleration (optional). WASM acceleration (optional).

---

## v3.0 Roadmap — Progress

### Performance

- [x] **WASM acceleration**: C source for special functions and linear algebra, TS fallback, integrated into math.ts/linalg.ts fallback chain (Fortran → WASM → TS)
- [ ] **Typed arrays everywhere**: Migrate remaining `number[][]` matrix ops to `Float64Array` for cache-friendly access
- [x] **Lazy evaluation in DataFrame**: Only compute columns when accessed
- [x] **Worker thread support**: parallelMCMC, parallelBootstrap, parallelCrossValidation with WorkerPool

### Developer Experience

- [x] **Documentation site**: VitePress site with 11 tutorials (distributions, regression, bayesian, time-series, survival, causal-inference, dimensionality, mixed-models, meta-analysis, changepoint, spatial) + guides + API reference
- [x] **Expanded benchmarks**: Cover special functions, linalg, regression, GLM, MCMC, and clustering modules
- [x] **JSDoc @example tags**: Added to all major public functions
- [x] **Error messages**: Include parameter names and valid ranges in all validation errors

### Ecosystem

- [x] **Sub-path exports**: 50+ sub-path exports (`node_stats/distributions`, `node_stats/bayesian`, etc.)
- [ ] **Standalone packages**: Consider publishing distributions, time-series, and bayesian as standalone npm packages
- [x] **Observable/RxJS integration**: Streaming stats adapter for Observable streams
- [x] **JSON schema**: JSON schemas for result types published

### Testing

- [x] **Property-based testing**: 34+ generative tests across all distributions (CDF(quantile(p)) ≈ p, monotonicity, etc.)
- [x] **Numerical accuracy benchmarks**: Reference values from R/SciPy for 14+ distributions and special functions
- [x] **Fuzz testing**: 68 tests with random/extreme inputs for all special functions
- [x] **Performance regression tests**: Benchmark CI with >10% regression detection, baselines tracked

### Completed

- [x] **Lévy distribution**: Full implementation with pdf, cdf, quantile, sample, sf, mean, variance

---

## Remaining Work

- Publish compiled WASM binary in npm package (currently requires local Emscripten build)
- Browser/serverless WASM loader (fetch-based instead of fs.readFile)
- Expand benchmark baselines for all modules
- Migrate matrix internals to Float64Array
- Consider standalone npm packages for reduced bundle size
