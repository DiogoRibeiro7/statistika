# Tutorial: Mixed-Effects Models

Mixed-effects models handle data with a hierarchical or grouped structure -- for example, students nested within schools, or repeated measurements on the same subjects. node_stats provides random intercept models, random intercept + slope models, ICC computation, and likelihood ratio tests for model comparison.

## When to Use Mixed Models

Use a mixed model when:

- Observations are grouped (e.g., patients within hospitals, measurements within subjects)
- You want to account for correlation within groups
- You need group-specific predictions (BLUPs)
- You want to generalize beyond the specific groups in your sample

## Random Intercept Model

The simplest mixed model allows each group to have its own intercept while sharing fixed-effect slopes across groups.

```typescript
import { lmmRandomIntercept } from 'node_stats';

// Student test scores nested within schools
// y = scores, X = study hours, groups = school ID
const scores = [
  85, 90, 78, 92,   // School A
  70, 75, 68, 72,   // School B
  95, 88, 92, 97,   // School C
];
const studyHours = [
  [3], [5], [2], [6],
  [3], [5], [2], [6],
  [3], [5], [2], [6],
];
const schools = [
  'A', 'A', 'A', 'A',
  'B', 'B', 'B', 'B',
  'C', 'C', 'C', 'C',
];

const result = lmmRandomIntercept(scores, studyHours, schools);

console.log(result.fixedEffects);
// [intercept, slope for study hours]
// e.g., [73.5, 3.2] — each extra hour adds ~3.2 points

console.log(result.randomEffects.interceptVariance);
// Between-school variance in intercepts

console.log(result.residualVariance);
// Within-school (residual) variance
```

### Interpreting BLUPs

BLUPs (Best Linear Unbiased Predictors) estimate how each group deviates from the overall intercept:

```typescript
for (const [school, blup] of result.blups) {
  console.log(`School ${school}: intercept shift = ${blup[0].toFixed(2)}`);
}
// School A: intercept shift = +2.15
// School B: intercept shift = -8.40
// School C: intercept shift = +6.25
// School C performs above average; School B below
```

### Prediction

Predict for new data, optionally incorporating group-specific effects:

```typescript
// Population-average prediction (no group effect)
const avgPred = result.predict([4]);
console.log(`Average student with 4 hours: ${avgPred.toFixed(1)}`);

// Group-specific prediction for School A
const schoolAPred = result.predict([4], 'A');
console.log(`School A student with 4 hours: ${schoolAPred.toFixed(1)}`);
```

## Random Intercept + Slope Model

When the effect of a covariate varies by group, add a random slope. For example, the benefit of study hours may differ across schools:

```typescript
import { lmmRandomSlope } from 'node_stats';

const result = lmmRandomSlope(scores, studyHours, schools);

console.log(result.fixedEffects);
// [intercept, average slope]

console.log(result.randomEffects.interceptVariance);
// Between-school intercept variance

console.log(result.randomEffects.slopeVariance);
// Between-school slope variance

console.log(result.randomEffects.interceptSlopeCovariance);
// Covariance between random intercepts and slopes
```

### Group-Specific Slopes

Each group now gets both an intercept and a slope adjustment:

```typescript
for (const [school, blup] of result.blups) {
  console.log(
    `School ${school}: intercept shift = ${blup[0].toFixed(2)}, ` +
    `slope shift = ${blup[1].toFixed(2)}`
  );
}
// School A: intercept shift = +1.50, slope shift = +0.30
// School B: intercept shift = -7.20, slope shift = -0.15
// School C: intercept shift = +5.70, slope shift = -0.15
```

## Intraclass Correlation Coefficient (ICC)

The ICC measures what proportion of total variance is attributable to between-group differences. A high ICC means groups differ substantially.

```typescript
import { icc } from 'node_stats';

const result = icc(scores, schools);

console.log(result.icc);
// e.g., 0.72 — 72% of variance is between schools

console.log(result.betweenVariance);
// Estimated between-group variance

console.log(result.withinVariance);
// Estimated within-group variance
```

An ICC near 0 suggests grouping has little effect, and a standard regression may suffice. An ICC near 1 means nearly all variation is between groups.

## Model Comparison with Likelihood Ratio Test

Compare a simpler model (e.g., random intercept only) to a more complex one (random intercept + slope) using a likelihood ratio test:

```typescript
import { lmmRandomIntercept, lmmRandomSlope, lrtTest } from 'node_stats';

const simple = lmmRandomIntercept(scores, studyHours, schools);
const full = lmmRandomSlope(scores, studyHours, schools);

const test = lrtTest(simple.logLikelihood, full.logLikelihood, 2);
// dfDiff = 2: slope variance + intercept-slope covariance

console.log(test.statistic);
// Chi-squared test statistic

console.log(test.pValue);
// p < 0.05 suggests the random slope significantly improves fit
```

### Using AIC and BIC

You can also compare models with information criteria -- lower is better:

```typescript
console.log(`Random intercept — AIC: ${simple.aic.toFixed(1)}, BIC: ${simple.bic.toFixed(1)}`);
console.log(`Random slope     — AIC: ${full.aic.toFixed(1)}, BIC: ${full.bic.toFixed(1)}`);
// Choose the model with lower AIC/BIC
```

## Complete Example: Classroom Data

```typescript
import { lmmRandomIntercept, icc } from 'node_stats';

// Reading scores for 20 students across 5 classrooms
const readingScores = [
  72, 78, 75, 80,   // Classroom 1
  65, 62, 68, 60,   // Classroom 2
  88, 85, 90, 92,   // Classroom 3
  55, 58, 52, 60,   // Classroom 4
  70, 74, 72, 68,   // Classroom 5
];
const priorGPA = [
  [3.0], [3.5], [3.2], [3.8],
  [3.0], [2.8], [3.3], [2.5],
  [3.5], [3.2], [3.7], [3.9],
  [2.5], [2.8], [2.2], [3.0],
  [3.0], [3.3], [3.1], [2.9],
];
const classrooms = [
  1, 1, 1, 1,
  2, 2, 2, 2,
  3, 3, 3, 3,
  4, 4, 4, 4,
  5, 5, 5, 5,
];

// Step 1: Check if mixed model is warranted
const iccResult = icc(readingScores, classrooms);
console.log(`ICC = ${iccResult.icc.toFixed(3)}`);
// High ICC confirms substantial between-classroom variation

// Step 2: Fit the model
const model = lmmRandomIntercept(readingScores, priorGPA, classrooms);
console.log(`Fixed intercept: ${model.fixedEffects[0].toFixed(2)}`);
console.log(`GPA effect: ${model.fixedEffects[1].toFixed(2)}`);
console.log(`Between-class variance: ${model.randomEffects.interceptVariance.toFixed(2)}`);
console.log(`Residual variance: ${model.residualVariance.toFixed(2)}`);

// Step 3: Predict for a new student
const prediction = model.predict([3.5], 3);
console.log(`Predicted score (Classroom 3, GPA 3.5): ${prediction.toFixed(1)}`);
```

## Function Reference

| Function | Description |
|---|---|
| `lmmRandomIntercept(y, X, groups, options?)` | Random intercept LMM via REML/EM |
| `lmmRandomSlope(y, X, groups, options?)` | Random intercept + slope LMM via REML/EM |
| `icc(values, groups)` | Intraclass Correlation Coefficient |
| `lrtTest(restricted, full, dfDiff)` | Likelihood ratio test for nested models |
