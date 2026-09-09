# Getting Started

statistika is a comprehensive statistical modeling and probability distribution library for Node.js, written in TypeScript with optional native Fortran acceleration.

## Prerequisites

- Node.js 18 or later
- A package manager: yarn, npm, or pnpm

## Install

```bash
yarn add @diogoribeiro7/statistika
```

Or with npm:

```bash
npm install @diogoribeiro7/statistika
```

## Basic Usage

Import what you need from the top-level package:

```typescript
import { Normal, mean, stdDev, oneSampleTTest } from 'statistika';

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

## Sub-path Exports

statistika uses sub-path exports so you can import from specific modules. This enables better tree-shaking and makes it clear which part of the library you depend on.

```typescript
// Import everything from the top level
import { Normal, linearRegression } from 'statistika';

// Or import from specific sub-paths
import { Normal, BetaDistribution } from 'statistika/distributions';
import { linearRegression, multipleRegression } from 'statistika/models';
import { oneSampleTTest, chiSquaredIndependence } from 'statistika/tests';
import { mean, variance, describe } from 'statistika/utils';

// Specialized modules
import { betaBinomial, normalNormal } from 'statistika/bayesian';
import { metropolisHastings, gelmanRubin } from 'statistika/mcmc';
import { arima, autoArima, seasonalDecompose } from 'statistika/time-series';
import { garchFit, egarchFit } from 'statistika/garch';
import { varFit, grangerCausality } from 'statistika/var';
import { glm, poisson, binomial } from 'statistika/glm';
import { kaplanMeier, logRankTest } from 'statistika/survival';
import { propensityScore, differenceInDifferences, rdd } from 'statistika/causal-inference';
```

Available sub-paths:

| Sub-path | Contents |
|---|---|
| `statistika/distributions` | All 37 probability distributions |
| `statistika/models` | Regression models (linear, multiple, polynomial, logistic, robust) |
| `statistika/tests` | Hypothesis tests (t-test, chi-squared, ANOVA, KS, etc.) |
| `statistika/utils` | Descriptive statistics, linear algebra, special functions |
| `statistika/bayesian` | Conjugate models and Bayes factors |
| `statistika/mcmc` | Metropolis-Hastings, Gelman-Rubin, ESS |
| `statistika/time-series` | ARIMA, auto-ARIMA, ACF/PACF, seasonal decompose, ADF |
| `statistika/garch` | ARCH, GARCH, EGARCH, GJR-GARCH |
| `statistika/var` | VAR models, Granger causality, impulse response |
| `statistika/glm` | Generalized linear models |
| `statistika/gam` | Generalized additive models |
| `statistika/survival` | Kaplan-Meier, Nelson-Aalen, log-rank test |
| `statistika/causal-inference` | Propensity scores, matching, DiD, RDD, 2SLS |
| `statistika/resampling` | Cross-validation, bootstrap, jackknife |
| `statistika/diagnostics` | Regression diagnostics, VIF, residual analysis |

## TypeScript Support

statistika is written in TypeScript with strict typing. All public interfaces and result types are exported:

```typescript
import type { TestResult, AnovaResult } from 'statistika';
import type { KaplanMeierResult } from 'statistika/survival';
import type { GARCHResult } from 'statistika/garch';
import type { VARResult, GrangerCausalityResult } from 'statistika/var';
```

## What's Next?

- [Installation](./installation) -- detailed setup including native acceleration
- [Quick Start](./quick-start) -- code examples covering major features
- [Tutorials](/tutorials/distributions) -- in-depth guides for each domain
