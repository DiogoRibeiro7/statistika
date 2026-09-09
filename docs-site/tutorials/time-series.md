# Tutorial: Time Series Analysis

This tutorial covers time series analysis in statistika: exploratory data analysis, ARIMA modeling, GARCH volatility models, VAR multivariate models, and forecasting.

## Exploratory Data Analysis

### Autocorrelation and Partial Autocorrelation

The ACF and PACF are the primary tools for identifying the order of ARIMA models:

```typescript
import { autocorrelation } from 'statistika';

const sales = [
  120, 135, 150, 145, 160, 175, 170, 185, 200, 195, 210, 225,
  220, 235, 250, 245, 260, 275, 270, 285, 300, 295, 310, 325,
];

const acf = autocorrelation(sales, 12);

console.log('95% confidence bound:', acf.confidenceBound);
// Correlations outside this bound are significant.

for (let k = 0; k <= 6; k++) {
  const sig = k > 0 && Math.abs(acf.acf[k]) > acf.confidenceBound ? '*' : '';
  console.log(`Lag ${k}: ACF=${acf.acf[k].toFixed(4)} PACF=${acf.pacf[k].toFixed(4)} ${sig}`);
}
```

### Stationarity Testing

Use the Augmented Dickey-Fuller test to check whether a series is stationary:

```typescript
import { adfTest } from 'statistika';

const adf = adfTest(sales);
console.log(adf.statistic);    // ADF test statistic
console.log(adf.pValue);       // p-value
console.log(adf.lags);         // number of lags used
console.log(adf.isStationary); // true if p < 0.05
```

If the series is non-stationary, differencing is needed before fitting ARMA models.

### Differencing

```typescript
import { difference } from 'statistika';

const diff1 = difference(sales, 1); // first difference
console.log(diff1);
// [15, 15, -5, 15, 15, -5, 15, 15, -5, ...]

// Test stationarity of the differenced series
const adfDiff = adfTest(diff1);
console.log(adfDiff.isStationary); // likely true now
```

### Moving Averages

Smooth the series to reveal the underlying trend:

```typescript
import { simpleMovingAverage, exponentialMovingAverage } from 'statistika';

const sma = simpleMovingAverage(sales, 3);
console.log(sma.values); // smoothed values (2 fewer points)

const ema = exponentialMovingAverage(sales, 0.3);
console.log(ema.values); // same length as input
```

## ARIMA Models

### Fitting an ARIMA Model

The `arima(series, p, d, q)` function fits an ARIMA model:

```typescript
import { arima } from 'statistika';

// ARIMA(2,1,0): 2 AR terms, 1 differencing, 0 MA terms
const model = arima(sales, 2, 1, 0);

console.log(model.arCoefficients);  // AR coefficients
console.log(model.maCoefficients);  // MA coefficients (empty for q=0)
console.log(model.intercept);       // intercept
console.log(model.sigma2);          // residual variance
console.log(model.aic);             // AIC for model comparison
```

### Forecasting

Generate point forecasts from the fitted model:

```typescript
const forecast = model.forecast(6);
for (let i = 0; i < forecast.length; i++) {
  console.log(`Month ${sales.length + i + 1}: ${forecast[i].toFixed(1)}`);
}
```

### Auto-ARIMA

Let the library select the best (p, d, q) order automatically by minimizing AIC:

```typescript
import { autoArima } from 'statistika';

const best = autoArima(sales, { maxP: 3, maxD: 2, maxQ: 3 });
console.log(best.selectedOrder);
// { p: ..., d: ..., q: ... } — the order that minimizes AIC

console.log(best.arCoefficients);
console.log(best.maCoefficients);
```

### Forecasting with Prediction Intervals

```typescript
import { forecastWithIntervals } from 'statistika';

const fc = forecastWithIntervals(best, 6, 0.95);
for (let i = 0; i < fc.point.length; i++) {
  console.log(
    `Step ${i + 1}: ${fc.point[i].toFixed(1)} ` +
    `[${fc.lower[i].toFixed(1)}, ${fc.upper[i].toFixed(1)}]`
  );
}
```

## Seasonal Decomposition

Decompose a series into trend, seasonal, and residual components:

```typescript
import { seasonalDecompose } from 'statistika';

// Monthly data with period 12
const monthly = Array.from({ length: 48 }, (_, i) =>
  50 + 2 * i + 10 * Math.sin((2 * Math.PI * i) / 12) + Math.random() * 2
);

const decomp = seasonalDecompose(monthly, 12);
console.log(decomp.trend);    // trend component (nulls at edges)
console.log(decomp.seasonal); // repeating seasonal pattern
console.log(decomp.residual); // remainder after removing trend + season
```

## GARCH Volatility Models

GARCH models capture time-varying volatility -- essential for financial time series.

### GARCH(1,1)

```typescript
import { garchFit, garchForecast } from 'statistika/garch';

// Daily stock returns
const returns = [0.01, -0.02, 0.015, -0.005, 0.03, -0.025, 0.008,
  -0.012, 0.022, -0.018, 0.005, -0.008, 0.012, -0.015, 0.028,
  -0.022, 0.009, -0.011, 0.018, -0.014, 0.006, -0.003, 0.017,
  -0.019, 0.013, -0.007, 0.021, -0.016, 0.004, -0.009];

const fit = garchFit(returns, 1, 1);
console.log(fit.omega);                // variance intercept
console.log(fit.alpha);               // ARCH coefficients
console.log(fit.beta);                // GARCH coefficients
console.log(fit.logLikelihood);
console.log(fit.aic);
console.log(fit.conditionalVariance); // time-varying variance series

// Forecast future volatility
const volForecast = garchForecast(fit, 5);
console.log(volForecast); // 5-step-ahead variance forecasts
```

### EGARCH for Asymmetric Volatility

EGARCH captures leverage effects -- negative returns often increase volatility more than positive returns:

```typescript
import { egarchFit } from 'statistika/garch';

const efit = egarchFit(returns, 1, 1);
console.log(efit.omega);  // log-variance intercept
console.log(efit.alpha);  // ARCH effect
console.log(efit.beta);   // persistence
console.log(efit.gamma);  // leverage/asymmetry coefficients
```

### ARCH-LM Test

Test whether ARCH effects are present in the residuals:

```typescript
import { archLMTest } from 'statistika/garch';

const test = archLMTest(returns, 5);
console.log(test.fStatistic);
console.log(test.pValue);
// A significant p-value indicates ARCH effects are present,
// justifying the use of GARCH models.
```

## VAR: Multivariate Time Series

Vector Autoregression models the interdependencies between multiple time series.

### Fitting a VAR Model

```typescript
import { varFit, varLagSelect } from 'statistika/var';

// Two interrelated series: GDP growth and inflation (quarterly, 40 obs)
const gdp = Array.from({ length: 40 }, (_, i) => 2 + 0.5 * Math.sin(i / 4) + Math.random() * 0.3);
const inflation = Array.from({ length: 40 }, (_, i) => 3 + 0.3 * Math.cos(i / 4) + Math.random() * 0.2);
const data = gdp.map((g, i) => [g, inflation[i]]);

// Select optimal lag order
const lagSel = varLagSelect(data, 8);
console.log('AIC-optimal lag:', lagSel.aicLag);
console.log('BIC-optimal lag:', lagSel.bicLag);

// Fit VAR with selected lag
const model = varFit(data, lagSel.bicLag);
console.log(model.coefficients); // coefficient matrices
console.log(model.intercept);    // intercept vector
console.log(model.aic);
console.log(model.bic);
```

### Granger Causality

Test whether one variable helps predict another:

```typescript
import { grangerCausality } from 'statistika/var';

const result = grangerCausality(data, lagSel.bicLag, 0, 1);
// Does variable 0 (GDP) Granger-cause variable 1 (inflation)?
console.log(result.fStatistic);
console.log(result.pValue);
```

### Impulse Response and Variance Decomposition

```typescript
import { impulseResponse, varianceDecomposition, varForecast } from 'statistika/var';

// How does a shock to GDP affect inflation over 10 periods?
const irf = impulseResponse(model, 10);
console.log(irf.responses);
// irf.responses[shock_var][response_var] = array of responses

// Forecast error variance decomposition
const fevd = varianceDecomposition(model, 10);
console.log(fevd.decomposition);

// VAR forecast
const fc = varForecast(model, 5);
console.log(fc.forecasts); // 5-step-ahead forecasts for each variable
```

## Complete Workflow Example

```typescript
import { adfTest, autoArima, forecastWithIntervals, seasonalDecompose } from 'statistika';

const series = Array.from({ length: 100 }, (_, i) =>
  10 + 0.5 * i + Math.random() * 3
);

// Step 1: Test for stationarity
const adf = adfTest(series);
console.log('Stationary:', adf.isStationary, 'p=', adf.pValue);

// Step 2: Select and fit the best ARIMA model
const best = autoArima(series, { maxP: 3, maxD: 2, maxQ: 3 });
console.log('Order:', best.selectedOrder);

// Step 3: Forecast with 95% prediction intervals
const fc = forecastWithIntervals(best, 10, 0.95);
for (let h = 0; h < fc.point.length; h++) {
  console.log(
    `t+${h + 1}: ${fc.point[h].toFixed(2)} [${fc.lower[h].toFixed(2)}, ${fc.upper[h].toFixed(2)}]`
  );
}
```
