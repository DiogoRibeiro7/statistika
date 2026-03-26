# Quick Start

Here are six examples covering the major features of statistika.

## 1. Descriptive Statistics

```typescript
import { mean, median, variance, stdDev, describe } from 'statistika';

const data = [4, 8, 15, 16, 23, 42];

console.log(mean(data));     // 18
console.log(median(data));   // 15.5
console.log(variance(data)); // 177.2 (sample variance)
console.log(stdDev(data));   // 13.31...

const summary = describe(data);
// { count: 6, mean: 18, median: 15.5, variance: 177.2,
//   stdDev: 13.31, min: 4, max: 42 }
```

## 2. Probability Distributions

```typescript
import { Normal, Poisson, BetaDistribution } from 'statistika';

// Normal distribution — IQ scores
const iq = new Normal(100, 15);
console.log(iq.cdf(130));       // P(IQ < 130) = 0.9772
console.log(iq.sf(85));         // P(IQ > 85) = 0.8413
console.log(iq.quantile(0.95)); // Top 5% threshold = 124.7
console.log(iq.sampleN(5));     // 5 random IQ scores

// Poisson distribution — events per hour
const events = new Poisson(4.5);
console.log(events.pmf(3));     // P(X = 3) = 0.1687
console.log(events.cdf(6));     // P(X <= 6) = 0.8311

// Beta distribution — conversion rate prior
const beta = new BetaDistribution(2, 5);
console.log(beta.mean());       // 0.2857
console.log(beta.cdf(0.5));     // P(rate < 0.5) = 0.9844
```

## 3. Hypothesis Testing

```typescript
import {
  oneSampleTTest, welchTTest, chiSquaredIndependence, oneWayAnova,
} from 'statistika';

// One-sample t-test: is the mean different from 50?
const sample = [48, 52, 51, 49, 53, 50, 47, 54];
const t = oneSampleTTest(sample, 50);
console.log(t.statistic); // t-statistic
console.log(t.pValue);    // p-value
console.log(t.rejected);  // false (not significant at alpha=0.05)

// Welch's t-test: compare two groups with unequal variance
const group1 = [22, 25, 28, 24, 26];
const group2 = [30, 33, 29, 35, 31];
const w = welchTTest(group1, group2);
console.log(w.pValue);    // significant difference

// Chi-squared test of independence
const table = [[10, 20], [30, 40]];
const chi = chiSquaredIndependence(table);
console.log(chi.statistic, chi.pValue);

// One-way ANOVA
const anova = oneWayAnova([
  [85, 90, 88], [78, 82, 80], [92, 95, 93],
]);
console.log(anova.fStatistic, anova.pValue);
```

## 4. Regression Models

```typescript
import { linearRegression, logisticRegression, glm, poisson } from 'statistika';

// Simple linear regression
const x = [1, 2, 3, 4, 5];
const y = [2.1, 3.9, 6.2, 7.8, 10.1];
const lr = linearRegression(x, y);
console.log(lr.slope);      // ~2.0
console.log(lr.intercept);  // ~0.06
console.log(lr.rSquared);   // ~0.999
console.log(lr.predict(6)); // predict y for x=6

// Logistic regression
const hours = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]];
const pass = [0, 0, 0, 0, 1, 0, 1, 1, 1, 1];
const logit = logisticRegression(hours, pass);
console.log(logit.predict([5])); // P(pass | 5 hours)
console.log(logit.predict([8])); // P(pass | 8 hours)

// Poisson GLM for count data
const X = [[1], [2], [3], [4], [5], [6], [7], [8]];
const counts = [1, 2, 3, 5, 8, 13, 21, 34];
const pfit = glm(X, counts, poisson);
console.log(pfit.coefficients);
console.log(pfit.deviance);
console.log(pfit.aic);
```

## 5. Time Series Forecasting

```typescript
import { autocorrelation, arima, autoArima, forecastWithIntervals } from 'statistika';

const sales = [
  120, 135, 150, 145, 160, 175, 170, 185, 200, 195, 210, 225,
  220, 235, 250, 245, 260, 275, 270, 285, 300, 295, 310, 325,
];

// ACF/PACF analysis
const acf = autocorrelation(sales, 12);
console.log(acf.acf);  // autocorrelation values
console.log(acf.pacf); // partial autocorrelation values

// Fit an ARIMA model
const model = arima(sales, 2, 1, 0);
console.log(model.arCoefficients);
console.log(model.aic);

// Auto-ARIMA: automatic order selection by AIC
const best = autoArima(sales, { maxP: 3, maxD: 2, maxQ: 3 });
console.log(best.selectedOrder); // { p, d, q }

// Forecast with prediction intervals
const fc = forecastWithIntervals(best, 6, 0.95);
console.log(fc.point); // 6-step point forecasts
console.log(fc.lower); // lower 95% bounds
console.log(fc.upper); // upper 95% bounds
```

## 6. Bayesian Inference

```typescript
import { betaBinomial, normalNormal, metropolisHastings } from 'statistika';

// A/B testing with Beta-Binomial conjugate model
const variantA = betaBinomial(45, 500);
console.log(variantA.posteriorMean);     // 0.0908
console.log(variantA.credibleInterval); // [0.068, 0.117]

// Normal-Normal conjugate model
const temps = [21.2, 19.8, 22.1, 20.5, 21.7, 20.3, 22.0, 19.5, 21.3, 20.8];
const result = normalNormal(temps, 4, 20, 100);
console.log(result.posteriorMean);     // updated mean
console.log(result.credibleInterval); // 95% credible interval

// Metropolis-Hastings MCMC for custom posteriors
const logPosterior = (x: number) => -0.5 * x * x; // standard normal
const mcmc = metropolisHastings(logPosterior, {
  nSamples: 10000, burnIn: 2000, proposalStd: 1, initial: 0,
});
console.log(mcmc.mean);           // ~0
console.log(mcmc.std);            // ~1
console.log(mcmc.acceptanceRate); // acceptance rate
```
