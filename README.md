# node_stats

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.9-blue.svg)](https://www.typescriptlang.org/)
[![Node.js](https://img.shields.io/badge/Node.js-%E2%89%A518-green.svg)](https://nodejs.org/)
[![Jest](https://img.shields.io/badge/Tests-Jest-red.svg)](https://jestjs.io/)

A comprehensive statistical modeling and probability distribution library for Node.js, written in TypeScript with optional native Fortran acceleration for special mathematical functions.

## Features

- **23 probability distributions** with PDF/PMF, CDF, quantile, survival function, and random sampling
- **Hypothesis testing** — t-tests, chi-squared, ANOVA, Kolmogorov-Smirnov
- **Descriptive statistics** — mean, median, variance, standard deviation
- **Linear regression** with R-squared and prediction
- **Special math functions** — gamma, beta, erf, regularized incomplete functions
- **Native Fortran acceleration** via N-API with automatic TypeScript fallback
- **Fully typed** — strict TypeScript with exported interfaces

## Installation

```bash
npm install node_stats
```

### Optional: Native Fortran acceleration

For best performance on special math functions, install with native compilation (requires `gfortran` and build tools):

```bash
npm run build
```

If `gfortran` is not available, the library falls back to pure TypeScript implementations automatically.

### TypeScript-only build

```bash
npm run build:ts
```

## Quick start

```typescript
import { Normal, mean, stdDev, oneSampleTTest } from "node_stats";

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
} from "node_stats";
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

## Descriptive statistics

```typescript
import { mean, median, variance, stdDev, describe } from "node_stats";

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
import { linearRegression } from "node_stats";

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
import { multipleRegression } from "node_stats";

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
import { polynomialRegression } from "node_stats";

const x = [-2, -1, 0, 1, 2, 3];
const y = x.map((xi) => 1 + 2 * xi + 3 * xi ** 2);

const fit = polynomialRegression(x, y, 2); // degree 2
console.log(fit.coefficients); // [1, 2, 3] — a0 + a1*x + a2*x²
console.log(fit.rSquared);     // 1.0
console.log(fit.predict(4));   // 57
```

### Logistic regression

```typescript
import { logisticRegression } from "node_stats";

const X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]];
const y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1];

const fit = logisticRegression(X, y);
console.log(fit.coefficients); // feature weights
console.log(fit.intercept);    // intercept
console.log(fit.predict([3])); // P(y=1) — close to 0
console.log(fit.predict([8])); // P(y=1) — close to 1
```

## Correlation

```typescript
import {
  pearsonCorrelation,
  spearmanCorrelation,
  kendallCorrelation,
} from "node_stats";

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

## Special math functions

These are accelerated by native Fortran when available, with pure TypeScript fallbacks.

```typescript
import {
  gamma, gammaLn, factorial, logFactorial,
  binomialCoeff, betaFn, erf, erfc,
  regularizedGammaP, regularizedBeta,
} from "node_stats";

gamma(5);              // 24
factorial(10);         // 3628800
binomialCoeff(10, 3);  // 120
erf(1);                // 0.8427...
```

## Development

```bash
# Install dependencies
npm install

# Build everything (Fortran + TypeScript)
npm run build

# Build TypeScript only
npm run build:ts

# Run tests
npm test

# Run tests with coverage
npm run test:coverage

# Lint
npm run lint

# Format
npm run format
```

## License

[MIT](LICENSE)
