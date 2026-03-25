# Tutorial: Regression Modeling

This tutorial covers regression modeling in node_stats, from simple linear regression to generalized linear models, including diagnostics and model selection.

## Simple Linear Regression

Fit a straight line through bivariate data:

```typescript
import { linearRegression } from 'node_stats';

const height = [150, 160, 165, 170, 175, 180, 185, 190];
const weight = [50, 60, 62, 68, 72, 78, 82, 90];

const fit = linearRegression(height, weight);
console.log(fit.slope);      // ~0.97
console.log(fit.intercept);  // ~-94.5
console.log(fit.rSquared);   // ~0.99
console.log(fit.predict(172)); // predicted weight at 172cm
```

The result object contains `slope`, `intercept`, `rSquared`, and a `predict(x)` method.

## Multiple Linear Regression

When you have multiple predictors, use `multipleRegression`:

```typescript
import { multipleRegression } from 'node_stats';

// Predict salary (in $k) from years of experience and education level
const features = [
  [2, 12], [4, 14], [6, 16], [8, 12],
  [10, 14], [12, 16], [14, 18], [16, 14],
];
const salary = [35, 45, 55, 50, 60, 70, 85, 75];

const fit = multipleRegression(features, salary);
console.log(fit.coefficients); // [coeff_experience, coeff_education]
console.log(fit.intercept);    // intercept
console.log(fit.rSquared);     // R-squared

// Predict for a new observation: 5 years experience, 16 years education
console.log(fit.predict([5, 16]));
```

## Polynomial Regression

Fit higher-degree polynomials with `polynomialRegression`:

```typescript
import { polynomialRegression } from 'node_stats';

// Free-fall distance: d = 0.5 * g * t^2 (g ~ 9.8)
const time = [0, 1, 2, 3, 4, 5, 6, 7, 8];
const distance = [0, 4.9, 19.6, 44.1, 78.4, 122.5, 176.4, 240.1, 313.6];

const fit = polynomialRegression(time, distance, 2);
console.log(fit.coefficients);
// [~0, ~0, ~4.9] — coefficients for a0 + a1*x + a2*x^2
console.log(fit.rSquared);   // ~1.0
console.log(fit.predict(10)); // distance at t=10
```

## Logistic Regression

For binary outcomes, use `logisticRegression`:

```typescript
import { logisticRegression } from 'node_stats';

// Predict exam pass/fail from study hours
const hours = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]];
const pass = [0, 0, 0, 0, 1, 0, 1, 1, 1, 1];

const fit = logisticRegression(hours, pass);
console.log(fit.coefficients[0]); // positive: more hours -> higher P(pass)
console.log(fit.intercept);

// Predicted probabilities
console.log(fit.predict([3]));  // P(pass | 3 hours) — low
console.log(fit.predict([5]));  // P(pass | 5 hours) — moderate
console.log(fit.predict([8]));  // P(pass | 8 hours) — high
```

## Generalized Linear Models

The `glm` function supports multiple distribution families: `gaussian`, `binomial`, `poisson`, and `gamma`. Each family has a canonical link function.

### Poisson Regression for Count Data

```typescript
import { glm, poisson } from 'node_stats';

// Number of insurance claims by driver age group
const X = [[1], [2], [3], [4], [5], [6], [7], [8]];
const claims = [1, 2, 3, 5, 8, 13, 21, 34];

const fit = glm(X, claims, poisson);
console.log(fit.coefficients); // log-linear coefficients
console.log(fit.deviance);     // model deviance
console.log(fit.aic);          // Akaike Information Criterion
```

### Binomial GLM

```typescript
import { glm, binomial } from 'node_stats';

const X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]];
const y = [0, 0, 0, 0, 1, 0, 1, 1, 1, 1];

const fit = glm(X, y, binomial);
console.log(fit.coefficients);
console.log(fit.deviance);
```

### Gaussian GLM

The Gaussian family with identity link is equivalent to ordinary linear regression:

```typescript
import { glm, gaussian } from 'node_stats';

const X = [[1], [2], [3], [4], [5]];
const y = [2.1, 3.9, 6.2, 7.8, 10.1];

const fit = glm(X, y, gaussian);
console.log(fit.coefficients);
console.log(fit.deviance);
```

## Robust Regression

When your data contains outliers, robust methods resist their influence:

```typescript
import { huberRegression, ransacRegression } from 'node_stats';

// True relationship: y = 2x + 1, with outliers
const x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
const y = x.map(xi => 2 * xi + 1);
y[4] = 200;   // outlier
y[11] = -100; // outlier

// Huber M-estimation — downweights outliers
const huber = huberRegression(x, y);
console.log(huber.slopes[0]);   // ~2.0 (close to true value)
console.log(huber.intercept);   // ~1.0
console.log(huber.scale);       // estimated scale
console.log(huber.iterations);  // IRLS iterations

// RANSAC — fits on inlier subset only
const ransac = ransacRegression(x, y);
console.log(ransac.slopes[0]);  // ~2.0
console.log(ransac.intercept);  // ~1.0
console.log(ransac.nInliers);   // number of inlier points
```

## Regression Diagnostics

Evaluate the quality of a regression fit:

```typescript
import { regressionSummary, residualDiagnostics, vif } from 'node_stats';

// Full regression summary (similar to R's summary(lm()))
const X = [
  [2, 12], [4, 14], [6, 16], [8, 12],
  [10, 14], [12, 16], [14, 18], [16, 14],
];
const y = [35, 45, 55, 50, 60, 70, 85, 75];

const summary = regressionSummary(X, y);
console.log(summary.coefficients);
// Each coefficient has: estimate, standardError, tStatistic, pValue
console.log(summary.rSquared);
console.log(summary.fStatistic);

// Residual diagnostics
const predicted = X.map(row =>
  summary.coefficients.reduce((s, c, i) =>
    i === 0 ? s + c.estimate : s + c.estimate * row[i - 1], 0));
const diag = residualDiagnostics(y, predicted);
console.log(diag.durbinWatson);  // test for autocorrelation (~2 is good)
console.log(diag.jarqueBera);    // test for normality of residuals

// Variance Inflation Factor for multicollinearity
const vifs = vif(X);
console.log(vifs);
// Values > 10 indicate problematic multicollinearity
```

## Model Selection with Cross-Validation

Use cross-validation to compare models:

```typescript
import { kFoldCV, mse } from 'node_stats';
import { linearRegression, polynomialRegression } from 'node_stats';

const x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
const y = [2.1, 3.9, 6.2, 7.8, 10.1, 12.0, 14.2, 15.8, 18.1, 20.0];

// Compare linear vs quadratic fit using 5-fold CV
const X = x.map(xi => [xi]);

const linearFitPredict = (Xtrain: number[][], ytrain: number[]) => {
  const fit = linearRegression(
    Xtrain.map(r => r[0]), ytrain
  );
  return (Xtest: number[][]) => Xtest.map(r => fit.predict(r[0]));
};

const scores = kFoldCV(X, y, linearFitPredict, { k: 5, scorer: mse });
console.log(scores.mean);   // average MSE across folds
console.log(scores.scores); // per-fold MSE values
```
