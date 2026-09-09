# statistika

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.9-blue.svg)](https://www.typescriptlang.org/)
[![Node.js](https://img.shields.io/badge/Node.js-%E2%89%A518-green.svg)](https://nodejs.org/)
[![Jest](https://img.shields.io/badge/Tests-Jest-red.svg)](https://jestjs.io/)

A comprehensive statistical modeling and probability distribution library for Node.js, written in TypeScript with optional native Fortran acceleration.

## Features

- **37 probability distributions** with PDF/PMF, CDF, quantile, survival function, and random sampling
- **Hypothesis testing** — t-tests, chi-squared, ANOVA, KS, Mann-Whitney U, Wilcoxon, Fisher's exact
- **Regression models** — linear, multiple, polynomial, logistic, robust, quantile, Cox PH
- **Generalized Linear Models** — Gaussian, Binomial, Poisson, and Gamma families with link functions
- **Bayesian inference** — conjugate models, Metropolis-Hastings MCMC, Bayes factors
- **Time series & forecasting** — ARIMA, auto-ARIMA, seasonal decomposition, ADF test
- **Machine learning** — PCA, factor analysis, k-means, hierarchical clustering, t-SNE, GMM
- **Survival analysis** — Kaplan-Meier, Nelson-Aalen, log-rank test, Cox regression
- **Nonparametric methods** — KDE, bootstrap CI, permutation tests
- **Resampling & cross-validation** — k-fold CV, LOOCV, jackknife, stratified sampling
- **Robust statistics** — MAD, trimmed/winsorized mean, Huber M-estimate, outlier detection
- **Information theory** — entropy, mutual information, KL/JS divergence
- **Distance & similarity** — Euclidean, Manhattan, Mahalanobis, cosine, Jaccard
- **Missing data** — imputation (mean, median, mode, interpolation), pattern analysis
- **Streaming statistics** — online mean/variance, covariance, quantile estimation
- **Multiple testing corrections** — Bonferroni, Holm, Benjamini-Hochberg, and more
- **Descriptive statistics** — mean, median, variance, standard deviation, correlation
- **Confidence intervals** — mean, proportion, regression, paired/two-sample
- **Effect sizes** — Cohen's d, Hedges' g, eta-squared, Cramér's V, odds ratio
- **Power analysis** — sample size and power for t-tests, ANOVA, chi-squared, proportions
- **Special math functions** — gamma, beta, erf, regularized incomplete functions
- **Native Fortran acceleration** via N-API with automatic TypeScript fallback
- **Fully typed** — strict TypeScript with exported interfaces

## Installation

```bash
yarn add statistika
```

### Optional: Native Fortran acceleration

For best performance on special math functions, install with native compilation (requires `gfortran` and build tools):

```bash
yarn build
```

If `gfortran` is not available, the library falls back to pure TypeScript implementations automatically.

On Windows, `yarn build` will now attempt a native Fortran build if `gfortran` is installed and LAPACK is available. This generally requires a compatible Windows native toolchain (MSYS2/MinGW or Visual Studio build tools) and LAPACK on `PATH`. If the native build cannot be completed, the TypeScript fallback remains usable.

### TypeScript-only build

```bash
yarn build:ts
```

## Quick start

```typescript
import { Normal, mean, stdDev, oneSampleTTest } from "statistika";

// Create a normal distribution
const dist = new Normal(0, 1);
console.log(dist.pdf(0));        // 0.3989...
console.log(dist.cdf(1.96));     // 0.975...
console.log(dist.quantile(0.5)); // 0
console.log(dist.sample());      // random draw

// Descriptive statistics
const data = [2.3, 1.8, 3.1, 2.7, 2.5];
console.log(mean(data));   // 2.48
console.log(stdDev(data)); // 0.473...

// Hypothesis testing
const result = oneSampleTTest(data, 2.0);
console.log(result.pValue);   // p-value
console.log(result.rejected); // true/false at alpha = 0.05
```

## Distributions

Every distribution provides: `mean()`, `variance()`, `stdDev()`, `sample()`, `sampleN(n)`.

### Continuous distributions

Continuous distributions additionally provide: `pdf(x)`, `cdf(x)`, `quantile(p)`, `sf(x)`.

| Class | Constructor | Parameters |
|---|---|---|
| `Normal` | `new Normal(mu, sigma)` | mean, standard deviation |
| `Uniform` | `new Uniform(a, b)` | lower bound, upper bound |
| `Exponential` | `new Exponential(lambda)` | rate |
| `GammaDistribution` | `new GammaDistribution(alpha, beta)` | shape, rate |
| `BetaDistribution` | `new BetaDistribution(alpha, beta)` | shape, shape |
| `ChiSquared` | `new ChiSquared(df)` | degrees of freedom |
| `StudentT` | `new StudentT(df)` | degrees of freedom |
| `FDistribution` | `new FDistribution(d1, d2)` | numerator df, denominator df |
| `LogNormal` | `new LogNormal(mu, sigma)` | log-mean, log-std |
| `Weibull` | `new Weibull(k, lambda)` | shape, scale |
| `Pareto` | `new Pareto(xm, alpha)` | minimum value, shape |
| `Cauchy` | `new Cauchy(x0, gamma)` | location, scale |
| `GEV` | `new GEV(mu, sigma, xi)` | location, scale, shape |
| `Gumbel` | `new Gumbel(mu, beta)` | location, scale |
| `Frechet` | `new Frechet(alpha, s)` | shape, scale |
| `GPD` | `new GPD(xi, sigma)` | shape, scale |

### Discrete distributions

Discrete distributions additionally provide: `pmf(k)`, `cdf(k)`, `quantile(p)`, `sf(k)`.

| Class | Constructor | Parameters |
|---|---|---|
| `Bernoulli` | `new Bernoulli(p)` | probability of success |
| `Binomial` | `new Binomial(n, p)` | trials, probability |
| `Poisson` | `new Poisson(lambda)` | rate |
| `Geometric` | `new Geometric(p)` | probability of success |
| `DiscreteUniform` | `new DiscreteUniform(a, b)` | min, max |
| `NegativeBinomial` | `new NegativeBinomial(r, p)` | successes needed, probability |
| `Hypergeometric` | `new Hypergeometric(N, K, n)` | population, successes in population, draws |

## Hypothesis testing

All tests return a result object with `statistic`, `pValue`, and `rejected` (at the given alpha, default 0.05).

```typescript
import {
  oneSampleTTest,
  twoSampleTTest,
  welchTTest,
  pairedTTest,
  chiSquaredGoodnessOfFit,
  chiSquaredIndependence,
  oneWayAnova,
  ksTest,
  ksTwoSampleTest,
  mannWhitneyU,
  wilcoxonSignedRank,
  fisherExactTest,
} from "statistika";
```

### t-tests

```typescript
// One-sample: test if population mean equals mu
oneSampleTTest(data, mu, alpha?);

// Two-sample: equal variances assumed
twoSampleTTest(data1, data2, alpha?);

// Welch's t-test: unequal variances
welchTTest(data1, data2, alpha?);

// Paired t-test
pairedTTest(before, after, alpha?);
```

### Chi-squared tests

```typescript
// Goodness of fit
chiSquaredGoodnessOfFit(observed, expected, alpha?);

// Test of independence (contingency table)
chiSquaredIndependence(table, alpha?);
```

### ANOVA

```typescript
// One-way ANOVA — returns AnovaResult with F-statistic, SS, MS, df
oneWayAnova([group1, group2, group3], alpha?);
```

### Kolmogorov-Smirnov test

```typescript
// One-sample KS test against a theoretical CDF
ksTest(data, cdfFunction, alpha?);

// Two-sample KS test
ksTwoSampleTest(data1, data2, alpha?);
```

### Non-parametric tests

```typescript
// Mann-Whitney U test (two independent samples)
mannWhitneyU(data1, data2, alpha?);

// Wilcoxon signed-rank test (paired samples)
wilcoxonSignedRank(before, after, alpha?);
```

### Fisher's exact test

```typescript
// 2x2 contingency table — exact p-value via hypergeometric distribution
fisherExactTest([[a, b], [c, d]], alpha?);
```

## Descriptive statistics

```typescript
import { mean, median, variance, stdDev, describe } from "statistika";

const data = [4, 8, 15, 16, 23, 42];

mean(data);     // arithmetic mean
median(data);   // median
variance(data); // sample variance (pass false for population)
stdDev(data);   // sample standard deviation

describe(data);
// { count: 6, mean: 18, median: 15.5, variance: 177.2, stdDev: 13.31, min: 4, max: 42 }
```

## Regression models

### Simple linear regression

```typescript
import { linearRegression } from "statistika";

const x = [1, 2, 3, 4, 5];
const y = [2.1, 3.9, 6.2, 7.8, 10.1];

const fit = linearRegression(x, y);
console.log(fit.slope);      // ~2.0
console.log(fit.intercept);  // ~0.06
console.log(fit.rSquared);   // ~0.999
console.log(fit.predict(6)); // predict y for x = 6
```

### Multiple linear regression

```typescript
import { multipleRegression } from "statistika";

const X = [
  [1, 5], [2, 3], [3, 1], [4, 6], [5, 2],
];
const y = [14.0, 11.0, 8.0, 17.0, 13.0];

const fit = multipleRegression(X, y);
console.log(fit.coefficients); // [coeff_x1, coeff_x2]
console.log(fit.intercept);    // intercept
console.log(fit.rSquared);     // R²
console.log(fit.predict([3, 4])); // predict for new observation
```

### Polynomial regression

```typescript
import { polynomialRegression } from "statistika";

const x = [-2, -1, 0, 1, 2, 3];
const y = x.map((xi) => 1 + 2 * xi + 3 * xi ** 2);

const fit = polynomialRegression(x, y, 2); // degree 2
console.log(fit.coefficients); // [1, 2, 3] — a0 + a1*x + a2*x²
console.log(fit.rSquared);     // 1.0
console.log(fit.predict(4));   // 57
```

### Logistic regression

```typescript
import { logisticRegression } from "statistika";

const X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]];
const y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1];

const fit = logisticRegression(X, y);
console.log(fit.coefficients); // feature weights
console.log(fit.intercept);    // intercept
console.log(fit.predict([3])); // P(y=1) — close to 0
console.log(fit.predict([8])); // P(y=1) — close to 1
```

### Generalized Linear Models

```typescript
import { glm, poisson, gaussian, binomial, gamma } from "statistika";

// Poisson regression for count data
const X = [[1], [2], [3], [4], [5]];
const counts = [2, 5, 8, 15, 30];
const fit = glm(X, counts, poisson());
console.log(fit.coefficients); // log-linear coefficients
console.log(fit.predict([3])); // predicted count
```

## Correlation

```typescript
import {
  pearsonCorrelation,
  spearmanCorrelation,
  kendallCorrelation,
} from "statistika";

const x = [1, 2, 3, 4, 5];
const y = [2, 4, 6, 8, 10];

const r = pearsonCorrelation(x, y);
console.log(r.coefficient); // 1.0
console.log(r.pValue);      // ~0

const rho = spearmanCorrelation(x, y);
console.log(rho.coefficient); // 1.0

const tau = kendallCorrelation(x, y);
console.log(tau.coefficient); // 1.0
```

## Confidence intervals

```typescript
import { meanCI, proportionCI, linearRegressionCI } from "statistika";

// CI for population mean
const ci = meanCI([2.3, 1.8, 3.1, 2.7, 2.5], 0.95);
console.log(ci.lower, ci.upper);

// Wilson score CI for proportion
const pci = proportionCI(45, 200, 0.95);
console.log(pci.lower, pci.upper);

// CI for regression coefficients
const rci = linearRegressionCI([1, 2, 3, 4, 5], [2, 4, 5, 8, 10]);
console.log(rci.slope); // { lower, upper }
```

## Effect sizes

```typescript
import { cohensD, hedgesG, etaSquared, cramersV, oddsRatio } from "statistika";

// Cohen's d for two groups
const d = cohensD([1, 2, 3], [4, 5, 6]);
console.log(d.d, d.interpretation); // effect size + "small"/"medium"/"large"

// Cramér's V for contingency tables
const v = cramersV([[10, 20], [30, 40]]);

// Odds ratio with CI
const or = oddsRatio([[15, 85], [30, 70]]);
console.log(or.oddsRatio, or.ci);
```

## Power analysis

```typescript
import { tTestSampleSize, tTestPower, anovaSampleSize } from "statistika";

// How many subjects for a medium effect at 80% power?
const n = tTestSampleSize(0.5, 0.8, 0.05, 2);
console.log(n); // required sample size per group

// Power of a test with n=30
const power = tTestPower(0.5, 30, 0.05, 2);
console.log(power); // achieved power
```

## Survival analysis

```typescript
import { kaplanMeier, nelsonAalen, logRankTest, coxRegression } from "statistika";

const obs = [
  { time: 1, event: true },
  { time: 3, event: false }, // censored
  { time: 4, event: true },
  { time: 6, event: true },
];

// Kaplan-Meier survival curve
const km = kaplanMeier(obs);
console.log(km.survivalFunction); // [{ time, survival, ci }]
console.log(km.medianSurvival);

// Log-rank test comparing two groups
const lr = logRankTest(group1, group2);
console.log(lr.pValue);
```

## Time series & forecasting

```typescript
import {
  autocorrelation, arima, difference,
  autoArima, forecastWithIntervals, seasonalDecompose, adfTest,
} from "statistika";

// ACF/PACF
const acf = autocorrelation(series, 10);

// ARIMA model
const model = arima(series, 1, 1, 1);
console.log(model.arCoefficients, model.maCoefficients);

// Auto-ARIMA: automatic model selection by AIC
const best = autoArima(series, { maxP: 3, maxD: 2, maxQ: 3 });
console.log(best.order); // [p, d, q]

// Forecasting with prediction intervals
const fc = forecastWithIntervals(best, 10, 0.95);
console.log(fc.forecasts, fc.lower, fc.upper);

// Seasonal decomposition
const decomp = seasonalDecompose(series, 12);
console.log(decomp.trend, decomp.seasonal, decomp.residual);

// Augmented Dickey-Fuller stationarity test
const adf = adfTest(series);
console.log(adf.statistic, adf.pValue);
```

## Bayesian inference

```typescript
import {
  betaBinomial, normalNormal, gammaPoisson,
  metropolisHastings, bayesFactor,
} from "statistika";

// Beta-Binomial conjugate model
const posterior = betaBinomial(45, 500);
console.log(posterior.posteriorMean);     // 0.0908...
console.log(posterior.credibleInterval); // [0.068, 0.117]

// Metropolis-Hastings MCMC
const mcmc = metropolisHastings(logPosteriorFn, {
  nSamples: 10000, burnIn: 2000, proposalStd: 1,
});
console.log(mcmc.mean, mcmc.std, mcmc.acceptanceRate);
```

## MCMC sampling

```typescript
import {
  metropolisHastingsND, gelmanRubin, estimateESS,
} from "statistika";

// Multi-dimensional MCMC
const result = metropolisHastingsND(logDensity, 2, {
  nSamples: 50000, burnIn: 5000,
});
console.log(result.means, result.acceptanceRate);

// Convergence diagnostics
const rhat = gelmanRubin([chain1, chain2]);
console.log(rhat); // should be < 1.1

const ess = estimateESS(chain);
console.log(ess); // effective sample size
```

## Machine learning

### Principal Component Analysis

```typescript
import { pca } from "statistika";

const data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]];
const result = pca(data, { nComponents: 2 });
console.log(result.components);         // principal components
console.log(result.explainedVariance);  // variance explained
console.log(result.projected);          // projected data
```

### Clustering

```typescript
import { kMeans, hierarchicalClustering } from "statistika";

// K-means with k-means++ initialization
const km = kMeans(data, 3);
console.log(km.labels);    // cluster assignments
console.log(km.centroids); // cluster centers

// Agglomerative clustering
const hc = hierarchicalClustering(data, 3, { linkage: "complete" });
console.log(hc.labels);
```

### Dimensionality reduction & cluster validation

```typescript
import { tsne, silhouetteScore, daviesBouldinIndex, adjustedRandIndex } from "statistika";

// t-SNE for 2D visualization
const embedded = tsne(highDimData, { perplexity: 30 });
console.log(embedded.coordinates); // 2D coordinates

// Evaluate clustering quality
const sil = silhouetteScore(data, labels);      // [-1, 1], higher is better
const dbi = daviesBouldinIndex(data, labels);    // lower is better
const ari = adjustedRandIndex(labels1, labels2); // agreement between clusterings
```

### Gaussian Mixture Models

```typescript
import { gaussianMixture, selectComponents } from "statistika";

const result = gaussianMixture(data, 3);
console.log(result.means);       // component means
console.log(result.weights);     // mixing weights
console.log(result.assignments); // cluster labels

// Automatic component selection via BIC
const best = selectComponents(data, 5);
console.log(best.k); // optimal number of components
```

## Generalized Linear Models

```typescript
import {
  glm, gaussian, binomial, poisson, gamma,
  identityLink, logLink, logitLink, probitLink,
} from "statistika";

// Poisson regression
const fit = glm(X, y, poisson());
console.log(fit.coefficients);
console.log(fit.deviance);
console.log(fit.aic);
```

## Nonparametric methods

```typescript
import { kernelDensity, bootstrapCI, permutationTest } from "statistika";

// Kernel density estimation
const kde = kernelDensity(data);
console.log(kde.estimate(2.5)); // density at x=2.5

// Bootstrap confidence interval
const ci = bootstrapCI(data, mean, { nResamples: 10000, method: "bca" });
console.log(ci.lower, ci.upper);

// Permutation test
const pt = permutationTest(group1, group2, { nPermutations: 10000 });
console.log(pt.pValue);
```

## Resampling & cross-validation

```typescript
import { kFoldCV, loocv, jackknife, stratifiedSample, mse, r2Score } from "statistika";

// K-fold cross-validation
const scores = kFoldCV(X, y, fitPredictFn, { k: 5, scorer: mse });
console.log(scores.mean, scores.scores);

// Leave-one-out cross-validation
const loo = loocv(X, y, fitPredictFn, mse);
console.log(loo.mean);

// Jackknife bias and SE estimation
const jk = jackknife(data, mean);
console.log(jk.estimate, jk.bias, jk.standardError);
```

## Robust statistics

```typescript
import {
  mad, trimmedMean, winsorizedMean, huberMean,
  detectOutliers, iqr, biweightMidvariance,
} from "statistika";

const data = [1, 2, 3, 4, 5, 100]; // contains outlier

mad(data);            // Median Absolute Deviation
trimmedMean(data);    // mean after trimming extremes
winsorizedMean(data); // mean after Winsorizing
huberMean(data);      // Huber M-estimate
detectOutliers(data); // { outliers: [100], indices: [5], bounds }
```

## Multiple testing corrections

```typescript
import { bonferroni, holm, benjaminiHochberg } from "statistika";

const pValues = [0.01, 0.04, 0.03, 0.20, 0.005];

// Family-wise error rate control
const bonf = bonferroni(pValues, 0.05);
console.log(bonf.rejected); // [true, false, false, false, true]

// Step-down procedure (more powerful)
const holmResult = holm(pValues, 0.05);

// False discovery rate control
const bh = benjaminiHochberg(pValues, 0.05);
console.log(bh.rejected);
```

## Information theory

```typescript
import {
  entropy, mutualInformation, klDivergence, jsDivergence,
} from "statistika";

// Shannon entropy
entropy([0.5, 0.5]);      // 1.0 (max entropy for binary)
entropy([0.9, 0.1]);      // 0.469...

// Mutual information between variables
mutualInformation(dataX, dataY);

// KL and JS divergence
klDivergence([0.5, 0.5], [0.9, 0.1]);
jsDivergence([0.5, 0.5], [0.9, 0.1]); // symmetric
```

## Distance & similarity

```typescript
import {
  euclidean, manhattan, cosineDistance,
  mahalanobis, distanceMatrix,
} from "statistika";

euclidean([1, 0], [0, 1]);     // 1.414...
manhattan([1, 0], [0, 1]);     // 2
cosineDistance([1, 0], [0, 1]); // 1.0

// Mahalanobis distance from a dataset
mahalanobis([2, 3], dataset);

// Pairwise distance matrix
const dm = distanceMatrix(vectors, euclidean);
```

## Missing data handling

```typescript
import {
  analyzeMissing, meanImputation, medianImputation,
  linearInterpolation, listwiseDeletion,
} from "statistika";

const data = [1, null, 3, undefined, 5, NaN, 7];

// Analyze missing patterns
const report = analyzeMissing(data);
console.log(report.missingCount, report.missingRate);

// Imputation
meanImputation(data);           // replace with mean
medianImputation(data);         // replace with median
linearInterpolation(data);      // interpolate between neighbors
```

## Streaming statistics

```typescript
import { OnlineStats, OnlineCovariance, OnlineQuantile } from "statistika";

// Online mean/variance (Welford's algorithm)
const stats = new OnlineStats();
stats.push(1.5);
stats.push(2.3);
stats.push(3.7);
console.log(stats.mean, stats.variance, stats.count);

// Online covariance/correlation
const cov = new OnlineCovariance();
cov.push(1, 2);
cov.push(2, 4);
console.log(cov.correlation);

// Online quantile estimation (P² algorithm)
const q = new OnlineQuantile(0.5); // median
for (const x of largeStream) q.push(x);
console.log(q.quantile);
```

## Smoothing & interpolation

```typescript
import { loess, cubicSpline, sma, ema } from "statistika";

// LOESS smoothing
const smoothed = loess(x, y, 0.3);
console.log(smoothed); // smoothed y values

// Cubic spline interpolation
const spline = cubicSpline(x, y);
console.log(spline(2.5)); // interpolated value

// Moving averages
sma(data, 5);    // simple moving average
ema(data, 0.3);  // exponential moving average
```

## Categorical data analysis

```typescript
import {
  contingencyTable, mcnemarsTest, cochranMantelHaenszel, gTest,
} from "statistika";

// Contingency table with expected counts
const ct = contingencyTable([[10, 20], [30, 40]]);
console.log(ct.expected);

// McNemar's test for paired categorical data
mcnemarsTest([[50, 10], [5, 35]]);

// G-test (log-likelihood ratio)
gTest([[10, 20, 30], [15, 25, 35]]);
```

## Regression diagnostics

```typescript
import { regressionSummary, residualDiagnostics, vif } from "statistika";

// Full regression summary (similar to R's summary(lm()))
const summary = regressionSummary(X, y);
console.log(summary.coefficients); // estimates, SE, t, p-value
console.log(summary.rSquared);
console.log(summary.fStatistic);

// Residual diagnostics
const diag = residualDiagnostics(observed, predicted);
console.log(diag.durbinWatson);  // autocorrelation
console.log(diag.jarqueBera);    // normality

// Variance Inflation Factor
const vifs = vif(X);
console.log(vifs); // values > 10 indicate multicollinearity
```

## Random number generation

```typescript
import { SeededRng, haltonSequence, latinHypercube } from "statistika";

// Reproducible pseudo-random numbers
const rng = new SeededRng(42);
console.log(rng.next());      // [0, 1)
console.log(rng.nextInt(100)); // integer in [0, 99]

// Low-discrepancy sequences
const seq = haltonSequence(2, 100); // base-2 Halton, 100 points

// Latin Hypercube Sampling
const lhs = latinHypercube(3, 50); // 3 dimensions, 50 samples
```

## Special math functions

These are accelerated by native Fortran when available, with pure TypeScript fallbacks.

```typescript
import {
  gamma, gammaLn, factorial, logFactorial,
  binomialCoeff, betaFn, erf, erfc,
  regularizedGammaP, regularizedBeta,
} from "statistika";

gamma(5);              // 24
factorial(10);         // 3628800
binomialCoeff(10, 3);  // 120
erf(1);                // 0.8427...
```

## Development

```bash
# Install dependencies
yarn install

# Build everything (Fortran + TypeScript)
yarn build

# Build TypeScript only
yarn build:ts

# Run tests
yarn test

# Run tests with coverage
yarn test:coverage

# Lint
yarn lint

# Format
yarn format

# Generate API documentation
yarn docs

# Run benchmarks
yarn bench
```

## License

[MIT](LICENSE)
