# Getting Started

node_stats is a comprehensive statistical modeling and probability distribution library for Node.js, written in TypeScript with optional native Fortran acceleration.

## Prerequisites

- Node.js 18 or later
- A package manager: yarn, npm, or pnpm

## Install

```bash
yarn add @diogoribeiro7/node_stats
```

Or with npm:

```bash
npm install @diogoribeiro7/node_stats
```

## Basic Usage

Import what you need from the top-level package:

```typescript
import { Normal, mean, stdDev, oneSampleTTest } from 'node_stats';

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

node_stats uses sub-path exports so you can import from specific modules. This enables better tree-shaking and makes it clear which part of the library you depend on.

```typescript
// Import everything from the top level
import { Normal, linearRegression } from 'node_stats';

// Or import from specific sub-paths
import { Normal, BetaDistribution } from 'node_stats/distributions';
import { linearRegression, multipleRegression } from 'node_stats/models';
import { oneSampleTTest, chiSquaredIndependence } from 'node_stats/tests';
import { mean, variance, describe } from 'node_stats/utils';

// Specialized modules
import { betaBinomial, normalNormal } from 'node_stats/bayesian';
import { metropolisHastings, gelmanRubin } from 'node_stats/mcmc';
import { arima, autoArima, seasonalDecompose } from 'node_stats/time-series';
import { garchFit, egarchFit } from 'node_stats/garch';
import { varFit, grangerCausality } from 'node_stats/var';
import { glm, poisson, binomial } from 'node_stats/glm';
import { kaplanMeier, logRankTest } from 'node_stats/survival';
import { propensityScore, differenceInDifferences, rdd } from 'node_stats/causal-inference';
```

Available sub-paths:

| Sub-path | Contents |
|---|---|
| `node_stats/distributions` | All 23 probability distributions |
| `node_stats/models` | Regression models (linear, multiple, polynomial, logistic, robust) |
| `node_stats/tests` | Hypothesis tests (t-test, chi-squared, ANOVA, KS, etc.) |
| `node_stats/utils` | Descriptive statistics, linear algebra, special functions |
| `node_stats/bayesian` | Conjugate models and Bayes factors |
| `node_stats/mcmc` | Metropolis-Hastings, Gelman-Rubin, ESS |
| `node_stats/time-series` | ARIMA, auto-ARIMA, ACF/PACF, seasonal decompose, ADF |
| `node_stats/garch` | ARCH, GARCH, EGARCH, GJR-GARCH |
| `node_stats/var` | VAR models, Granger causality, impulse response |
| `node_stats/glm` | Generalized linear models |
| `node_stats/gam` | Generalized additive models |
| `node_stats/survival` | Kaplan-Meier, Nelson-Aalen, log-rank test |
| `node_stats/causal-inference` | Propensity scores, matching, DiD, RDD, 2SLS |
| `node_stats/resampling` | Cross-validation, bootstrap, jackknife |
| `node_stats/diagnostics` | Regression diagnostics, VIF, residual analysis |

## TypeScript Support

node_stats is written in TypeScript with strict typing. All public interfaces and result types are exported:

```typescript
import type { TestResult, AnovaResult } from 'node_stats';
import type { KaplanMeierResult } from 'node_stats/survival';
import type { GARCHResult } from 'node_stats/garch';
import type { VARResult, GrangerCausalityResult } from 'node_stats/var';
```

## What's Next?

- [Installation](./installation) -- detailed setup including native acceleration
- [Quick Start](./quick-start) -- code examples covering major features
- [Tutorials](/tutorials/distributions) -- in-depth guides for each domain
