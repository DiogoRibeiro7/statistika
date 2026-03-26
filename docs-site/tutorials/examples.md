# Worked Examples

These end-to-end examples show how to combine multiple statistika modules to solve real-world statistical problems.

## Example 1: A/B Test Analysis

Compare two website variants using frequentist and Bayesian methods.

```typescript
import {
  mean, stdDev, twoSampleTTest, BetaDistribution, betaBinomial,
} from 'statistika';

// Variant A: 1000 visitors, 45 conversions
// Variant B: 1000 visitors, 62 conversions
const rateA = 45 / 1000;
const rateB = 62 / 1000;

// ── Frequentist: two-proportion z-test ──
// Generate binary arrays (1 = converted, 0 = not)
const a = Array(45).fill(1).concat(Array(955).fill(0));
const b = Array(62).fill(1).concat(Array(938).fill(0));

const test = twoSampleTTest(a, b);
console.log('t-statistic:', test.statistic.toFixed(3));
console.log('p-value:', test.pValue.toFixed(4));
console.log('Significant at 5%?', test.pValue < 0.05);

// ── Bayesian: Beta-Binomial conjugate model ──
const postA = betaBinomial(45, 1000);
const postB = betaBinomial(62, 1000);

console.log('Posterior mean A:', postA.posteriorMean.toFixed(4));
console.log('Posterior mean B:', postB.posteriorMean.toFixed(4));
console.log('95% CI for A:', postA.credibleInterval);
console.log('95% CI for B:', postB.credibleInterval);

// Monte Carlo estimate: P(B > A)
const distA = new BetaDistribution(postA.posteriorAlpha, postA.posteriorBeta);
const distB = new BetaDistribution(postB.posteriorAlpha, postB.posteriorBeta);
let bWins = 0;
const N = 100000;
for (let i = 0; i < N; i++) {
  if (distB.sample() > distA.sample()) bWins++;
}
console.log(`P(B > A) ≈ ${(bWins / N).toFixed(3)}`);
```

## Example 2: Regression with Diagnostics

Fit a model, check assumptions, and compute confidence intervals.

```typescript
import {
  linearRegression, multipleRegression,
  mean, stdDev,
} from 'statistika';

// Predict house prices from square footage
const sqft = [850, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600];
const price = [150, 210, 260, 320, 370, 430, 475, 540, 590, 640];

const fit = linearRegression(sqft, price);
console.log(`Price ≈ ${fit.intercept.toFixed(0)} + ${fit.slope.toFixed(2)} × sqft`);
console.log(`R² = ${fit.rSquared.toFixed(4)}`);

// Predict for a 2000 sqft house
const predicted = fit.predict(2000);
console.log(`Predicted price for 2000 sqft: $${predicted.toFixed(0)}k`);

// Compute residuals and check for patterns
const residuals = sqft.map((x, i) => price[i] - fit.predict(x));
console.log('Mean residual:', mean(residuals).toFixed(4)); // should be ~0
console.log('StdDev of residuals:', stdDev(residuals).toFixed(2));
```

## Example 3: Time Series Forecasting

Analyze and forecast a seasonal time series.

```typescript
import { arima, autocorrelation, adfTest, difference } from 'statistika';

// Monthly sales data (2 years)
const sales = [
  120, 135, 150, 145, 160, 175, 170, 185, 200, 195, 210, 225,
  220, 235, 250, 245, 260, 275, 270, 285, 300, 295, 310, 325,
];

// Step 1: Check stationarity
const adf = adfTest(sales);
console.log('ADF statistic:', adf.statistic.toFixed(4));
console.log('Is stationary?', adf.isStationary);

// Step 2: Difference if needed
const diffSales = difference(sales, 1);

// Step 3: Check ACF of differenced series
const acf = autocorrelation(diffSales, 6);
console.log('ACF lags 1-3:', acf.acf.slice(1, 4).map(v => v.toFixed(3)));

// Step 4: Fit ARIMA(1,1,0) model
const model = arima(sales, 1, 1, 0);
console.log('AR(1) coefficient:', model.arCoefficients[0].toFixed(4));
console.log('AIC:', model.aic.toFixed(2));

// Step 5: Forecast next 6 months
const forecast = model.forecast(6);
console.log('Next 6 months forecast:', forecast.map(v => v.toFixed(0)));
```

## Example 4: Survival Curve Comparison

Compare survival between treatment and control groups.

```typescript
import { kaplanMeier, logRankTest } from 'statistika';

// Treatment group
const treatment = [
  { time: 6, event: true }, { time: 7, event: true },
  { time: 10, event: false }, { time: 15, event: true },
  { time: 16, event: true }, { time: 22, event: false },
  { time: 23, event: true }, { time: 30, event: false },
  { time: 35, event: true }, { time: 40, event: false },
];

// Control group
const control = [
  { time: 1, event: true }, { time: 3, event: true },
  { time: 4, event: true }, { time: 5, event: false },
  { time: 8, event: true }, { time: 10, event: true },
  { time: 12, event: false }, { time: 14, event: true },
  { time: 18, event: true }, { time: 20, event: false },
];

// Kaplan-Meier estimates
const kmTreat = kaplanMeier(treatment);
const kmCtrl = kaplanMeier(control);

console.log('Treatment median survival:', kmTreat.medianSurvival);
console.log('Control median survival:', kmCtrl.medianSurvival);

// Log-rank test: is there a significant difference?
const lr = logRankTest(treatment, control);
console.log('Log-rank chi²:', lr.statistic.toFixed(3));
console.log('p-value:', lr.pValue.toFixed(4));
console.log('Significant at 5%?', lr.pValue < 0.05);
```

## Example 5: Distribution Fitting & Model Selection

Find the best-fitting distribution for a dataset.

```typescript
import {
  fitNormal, fitGamma, fitLogNormal, fitWeibull,
  andersonDarling,
} from 'statistika';

// Observed waiting times (minutes)
const waitTimes = [
  2.1, 0.5, 3.4, 1.2, 4.5, 0.8, 2.7, 1.9, 3.1, 0.3,
  5.2, 1.5, 2.3, 0.7, 3.8, 1.1, 4.1, 2.5, 1.8, 0.6,
  3.5, 2.0, 1.4, 4.8, 0.9, 2.8, 1.6, 3.2, 0.4, 5.0,
];

// Fit candidates
const fits = [
  { name: 'Normal', fit: fitNormal(waitTimes) },
  { name: 'Gamma', fit: fitGamma(waitTimes) },
  { name: 'LogNormal', fit: fitLogNormal(waitTimes) },
  { name: 'Weibull', fit: fitWeibull(waitTimes) },
];

// Compare by AIC
fits.sort((a, b) => a.fit.aic - b.fit.aic);
console.log('Distribution ranking by AIC:');
for (const { name, fit } of fits) {
  console.log(`  ${name}: AIC = ${fit.aic.toFixed(2)}, logLik = ${fit.logLikelihood.toFixed(2)}`);
}

// Anderson-Darling test on the best fit
const best = fits[0];
console.log(`\nBest fit: ${best.name}`);
```

## Example 6: Bayesian Parameter Estimation with MCMC

Estimate a parameter using Metropolis-Hastings sampling.

```typescript
import { metropolisHastingsND, gelmanRubin, estimateESS } from 'statistika';

// Estimate the mean of normally distributed data
const data = [4.2, 3.8, 4.5, 4.1, 3.9, 4.3, 4.0, 4.4, 3.7, 4.6];

// Log-posterior: Normal likelihood + flat prior
function logPosterior(params: number[]): number {
  const [mu] = params;
  const sigma = 0.3; // assume known
  let logLik = 0;
  for (const x of data) {
    logLik += -0.5 * ((x - mu) / sigma) ** 2;
  }
  return logLik; // flat prior contributes 0
}

// Run MCMC
const result = metropolisHastingsND(
  logPosterior,
  [4.0],       // initial values
  10000,       // iterations
  [0.1],       // proposal standard deviations
);

// Discard burn-in
const burnIn = 2000;
const samples = result.samples.slice(burnIn);
const muSamples = samples.map(s => s[0]);

// Posterior summary
const postMean = muSamples.reduce((a, b) => a + b) / muSamples.length;
const sorted = [...muSamples].sort((a, b) => a - b);
const ci025 = sorted[Math.floor(sorted.length * 0.025)];
const ci975 = sorted[Math.floor(sorted.length * 0.975)];

console.log(`Posterior mean: ${postMean.toFixed(3)}`);
console.log(`95% credible interval: [${ci025.toFixed(3)}, ${ci975.toFixed(3)}]`);
console.log(`ESS: ${estimateESS(muSamples).toFixed(0)}`);
console.log(`Acceptance rate: ${result.acceptanceRate.toFixed(3)}`);
```
