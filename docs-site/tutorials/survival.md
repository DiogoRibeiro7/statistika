# Tutorial: Survival Analysis

Survival analysis studies time-to-event data where some observations may be censored (the event was not observed during the study period). node_stats provides Kaplan-Meier estimation, Nelson-Aalen hazard estimation, log-rank tests, and Cox proportional hazards regression.

## Data Format

Survival data is represented as an array of objects with `time` and `event` fields:

```typescript
interface SurvivalObservation {
  time: number;   // observed time
  event: boolean; // true = event occurred, false = censored
}
```

## Kaplan-Meier Estimator

The Kaplan-Meier estimator is the standard nonparametric method for estimating the survival function.

```typescript
import { kaplanMeier } from 'node_stats';

// Clinical trial: time to relapse (months)
const treatment = [
  { time: 6, event: true },
  { time: 7, event: true },
  { time: 10, event: false },  // censored (lost to follow-up)
  { time: 15, event: true },
  { time: 16, event: true },
  { time: 22, event: false },  // censored
  { time: 23, event: true },
  { time: 24, event: false },  // censored
  { time: 28, event: true },
  { time: 30, event: false },  // censored
];

const km = kaplanMeier(treatment);

// Summary statistics
console.log(km.n);             // 10 total observations
console.log(km.nEvents);       // number of events
console.log(km.nCensored);     // number of censored observations
console.log(km.medianSurvival); // median survival time
```

### Survival Function

The survival function S(t) = P(T > t) gives the probability of surviving past time t:

```typescript
// Probability of surviving past 12 months
console.log(km.survivalAt(12));
// e.g., 0.6857

// Probability of surviving past 24 months
console.log(km.survivalAt(24));
// e.g., 0.3086
```

### Survival Curve with Confidence Intervals

The `curve` property contains the step-function survival curve with Greenwood confidence intervals:

```typescript
for (const pt of km.curve) {
  console.log(
    `t=${pt.time}: S=${pt.survival.toFixed(3)} ` +
    `95% CI [${pt.lower.toFixed(3)}, ${pt.upper.toFixed(3)}] ` +
    `(${pt.nRisk} at risk)`
  );
}
// t= 6: S=0.900  95% CI [0.714, 1.000]  (10 at risk)
// t= 7: S=0.800  95% CI [0.596, 1.000]  (9 at risk)
// t=15: S=0.686  95% CI [0.456, 0.915]  (7 at risk)
// ...
```

## Nelson-Aalen Estimator

The Nelson-Aalen estimator provides an alternative estimate of the cumulative hazard function H(t):

```typescript
import { nelsonAalen } from 'node_stats';

const na = nelsonAalen(treatment);
for (const pt of na.curve) {
  console.log(`t=${pt.time}: H=${pt.cumHazard.toFixed(4)}`);
}
// The survival function can be estimated as S(t) = exp(-H(t))
```

## Log-Rank Test

The log-rank test compares survival curves between two groups to determine if there is a statistically significant difference:

```typescript
import { logRankTest } from 'node_stats';

const control = [
  { time: 1, event: true },
  { time: 3, event: true },
  { time: 5, event: true },
  { time: 8, event: false },
  { time: 9, event: true },
  { time: 12, event: true },
  { time: 13, event: false },
  { time: 14, event: true },
  { time: 18, event: true },
  { time: 20, event: false },
];

const logrank = logRankTest(treatment, control);
console.log(logrank.chiSquared); // chi-squared statistic
console.log(logrank.pValue);     // p-value
console.log(logrank.rejected);   // true if p < 0.05
```

A significant result indicates that the survival distributions of the two groups are different. In this example, the treatment group has longer survival times.

## Cox Proportional Hazards Regression

Cox regression models the hazard as a function of covariates without specifying the baseline hazard:

h(t | X) = h_0(t) * exp(beta_1 * X_1 + beta_2 * X_2 + ...)

```typescript
import { coxRegression, CoxObservation } from 'node_stats';

// Model: hazard depends on age and treatment group (0=control, 1=treatment)
const data: CoxObservation[] = [
  { time: 1, event: true, covariates: [65, 0] },
  { time: 3, event: true, covariates: [58, 0] },
  { time: 5, event: true, covariates: [72, 0] },
  { time: 6, event: true, covariates: [60, 1] },
  { time: 7, event: true, covariates: [55, 1] },
  { time: 9, event: true, covariates: [68, 0] },
  { time: 10, event: false, covariates: [50, 1] },
  { time: 12, event: true, covariates: [70, 0] },
  { time: 15, event: true, covariates: [45, 1] },
  { time: 16, event: true, covariates: [62, 1] },
  { time: 22, event: false, covariates: [48, 1] },
  { time: 23, event: true, covariates: [53, 1] },
  { time: 28, event: true, covariates: [67, 0] },
  { time: 30, event: false, covariates: [42, 1] },
];

const cox = coxRegression(data);
```

### Interpreting Results

```typescript
const labels = ['Age', 'Treatment'];

console.log('Concordance:', cox.concordance);
console.log('Log-likelihood:', cox.logLikelihood);

for (let i = 0; i < labels.length; i++) {
  console.log(
    `${labels[i]}: ` +
    `beta=${cox.coefficients[i].toFixed(4)} ` +
    `HR=${cox.hazardRatios[i].toFixed(3)} ` +
    `95% CI [${cox.hazardRatioCIs[i][0].toFixed(3)}, ${cox.hazardRatioCIs[i][1].toFixed(3)}] ` +
    `p=${cox.pValues[i].toFixed(4)}`
  );
}
// Age:       beta=0.0312  HR=1.032  95% CI [0.98, 1.09]  p=0.23
// Treatment: beta=-0.891  HR=0.410  95% CI [0.14, 1.17]  p=0.09
```

Key interpretations:
- **Hazard Ratio (HR)**: HR > 1 means increased hazard (worse survival); HR < 1 means decreased hazard (better survival).
- **Concordance**: Measures predictive ability; 0.5 = random, 1.0 = perfect discrimination.

### Predicting Hazard Ratios

You can predict the relative hazard for new covariate profiles:

```typescript
// Hazard ratio for a 60-year-old on treatment vs baseline
console.log(cox.predictHazardRatio([60, 1]));

// Compare a 45-year-old on treatment vs a 70-year-old on control
const hr45treat = cox.predictHazardRatio([45, 1]);
const hr70ctrl = cox.predictHazardRatio([70, 0]);
console.log('Relative hazard:', hr45treat / hr70ctrl);
```

## Complete Workflow

A typical survival analysis workflow:

```typescript
import { kaplanMeier, logRankTest, coxRegression } from 'node_stats';

// Step 1: Kaplan-Meier for each group
const kmTreatment = kaplanMeier(treatment);
const kmControl = kaplanMeier(control);
console.log('Treatment median survival:', kmTreatment.medianSurvival);
console.log('Control median survival:', kmControl.medianSurvival);

// Step 2: Log-rank test to compare groups
const lr = logRankTest(treatment, control);
console.log('Log-rank p-value:', lr.pValue);

// Step 3: Cox regression to adjust for covariates
const cox = coxRegression(data);
console.log('Treatment HR:', cox.hazardRatios[1].toFixed(3));
console.log('Treatment p-value:', cox.pValues[1].toFixed(4));

// Step 4: Report
// - Kaplan-Meier curves show visual separation
// - Log-rank test p < 0.05 confirms statistical significance
// - Cox regression HR < 1 for treatment confirms protective effect
//   after adjusting for age
```
