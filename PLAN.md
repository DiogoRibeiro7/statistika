# node_stats — Feature Plan

## Current State (v2.0.0)

**node_stats** is a comprehensive statistical library for Node.js written in TypeScript with optional native Fortran/LAPACK acceleration. All originally planned v1.0 and v2.0 features have been implemented.

### What's Shipped

| Module | Features |
|--------|----------|
| **Distributions** (35+) | Normal, Uniform, Exponential, Gamma, Beta, Chi-squared, Student-t, F, Log-Normal, Weibull, GEV, Gumbel, Frechet, GPD, Cauchy, Pareto, Laplace, Inverse Gamma, Log-Logistic, Truncated Normal, Rayleigh, von Mises, Bernoulli, Binomial, Poisson, Geometric, Discrete Uniform, Negative Binomial, Hypergeometric, Zero-Inflated Poisson, Multivariate Normal, Multivariate t, Dirichlet, Wishart, Inverse Wishart, Multinomial |
| **Matrix/LinAlg** | Matrix class with LU, QR, Cholesky, SVD, eigenvalue decomposition; native LAPACK acceleration |
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
| **Other** | Bayesian networks, functional data analysis, compositional data, experimental design, streaming stats, robust statistics, information theory, missing data, smoothing, effect sizes, power analysis, DataFrame, viz-data |

**Health**: 98 test suites, all passing. CI/CD with GitHub Actions (CodeQL, dependency review, benchmarks). Dual CJS/ESM build. Native Fortran acceleration (optional).

---

## Next Steps (v3.0 Roadmap)

### Performance

- **WASM acceleration**: Compile hot-path matrix ops to WebAssembly for environments without gfortran — enables browser and serverless use
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

### Missing Distribution

- **Levy distribution** — stable distribution used in finance and anomalous diffusion modeling
