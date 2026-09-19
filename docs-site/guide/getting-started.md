# Getting Started

statistika is a comprehensive statistical modeling and probability distribution library for Node.js, written in TypeScript with optional native Fortran acceleration.

## Prerequisites

- Node.js 22 or later
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
import { Normal, mean, stdDev, oneSampleTTest } from '@diogoribeiro7/statistika';

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
import { Normal, linearRegression } from '@diogoribeiro7/statistika';

// Or import from specific sub-paths
import { Normal, BetaDistribution } from '@diogoribeiro7/statistika/distributions';
import { linearRegression, multipleRegression } from '@diogoribeiro7/statistika/models';
import { oneSampleTTest, chiSquaredIndependence } from '@diogoribeiro7/statistika/tests';
import { mean, variance, describe } from '@diogoribeiro7/statistika/utils';

// Specialized modules
import { betaBinomial, normalNormal } from '@diogoribeiro7/statistika/bayesian';
import { metropolisHastings, gelmanRubin } from '@diogoribeiro7/statistika/mcmc';
import { arima, autoArima, seasonalDecompose } from '@diogoribeiro7/statistika/time-series';
import { garchFit, egarchFit } from '@diogoribeiro7/statistika/garch';
import { varFit, grangerCausality } from '@diogoribeiro7/statistika/var';
import { glm, poisson, binomial } from '@diogoribeiro7/statistika/glm';
import { kaplanMeier, logRankTest } from '@diogoribeiro7/statistika/survival';
import { propensityScore, differenceInDifferences, rdd } from '@diogoribeiro7/statistika/causal-inference';
```

Available sub-paths:

| Sub-path | Contents |
|---|---|
| `@diogoribeiro7/statistika/distributions` | All 37 probability distributions |
| `@diogoribeiro7/statistika/models` | Regression models (linear, multiple, polynomial, logistic, robust) |
| `@diogoribeiro7/statistika/tests` | Hypothesis tests (t-test, chi-squared, ANOVA, KS, etc.) |
| `@diogoribeiro7/statistika/utils` | Descriptive statistics, linear algebra, special functions |
| `@diogoribeiro7/statistika/bayesian` | Conjugate models and Bayes factors |
| `@diogoribeiro7/statistika/mcmc` | Metropolis-Hastings, Gelman-Rubin, ESS |
| `@diogoribeiro7/statistika/time-series` | ARIMA, auto-ARIMA, ACF/PACF, seasonal decompose, ADF |
| `@diogoribeiro7/statistika/garch` | ARCH, GARCH, EGARCH, GJR-GARCH |
| `@diogoribeiro7/statistika/var` | VAR models, Granger causality, impulse response |
| `@diogoribeiro7/statistika/glm` | Generalized linear models |
| `@diogoribeiro7/statistika/gam` | Generalized additive models |
| `@diogoribeiro7/statistika/survival` | Kaplan-Meier, Nelson-Aalen, log-rank test |
| `@diogoribeiro7/statistika/causal-inference` | Propensity scores, matching, DiD, RDD, 2SLS |
| `@diogoribeiro7/statistika/resampling` | Cross-validation, bootstrap, jackknife |
| `@diogoribeiro7/statistika/diagnostics` | Regression diagnostics, VIF, residual analysis |

## TypeScript Support

statistika is written in TypeScript with strict typing. All public interfaces and result types are exported:

```typescript
import type { TestResult, AnovaResult } from '@diogoribeiro7/statistika';
import type { KaplanMeierResult } from '@diogoribeiro7/statistika/survival';
import type { GARCHResult } from '@diogoribeiro7/statistika/garch';
import type { VARResult, GrangerCausalityResult } from '@diogoribeiro7/statistika/var';
```

## What's Next?

- [Installation](./installation) -- detailed setup including native acceleration
- [Quick Start](./quick-start) -- code examples covering major features
- [Tutorials](/tutorials/distributions) -- in-depth guides for each domain
