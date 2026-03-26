# Tutorial: Causal Inference

This tutorial covers the causal inference methods in statistika: propensity score estimation, matching, inverse probability weighting (IPW), difference-in-differences (DiD), regression discontinuity design (RDD), and two-stage least squares (2SLS).

## Propensity Scores

The propensity score e(X) = P(T=1 | X) is the probability of receiving treatment given observed covariates. It is estimated via logistic regression.

```typescript
import { propensityScore } from 'statistika/causal-inference';

// Covariates: [age, income_in_thousands]
const X = [
  [25, 30], [30, 45], [35, 50], [40, 60], [45, 55],
  [28, 35], [33, 48], [38, 52], [42, 65], [50, 70],
  [27, 32], [31, 42], [36, 58], [41, 62], [48, 68],
  [26, 28], [34, 46], [39, 54], [43, 61], [47, 72],
];

// Treatment assignment (1 = treated, 0 = control)
const treatment = [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1];

const ps = propensityScore(X, treatment);
console.log(ps.scores);        // estimated P(T=1|X) for each observation
console.log(ps.coefficients);  // logistic regression coefficients
console.log(ps.iterations);    // IRLS iterations
```

## Propensity Score Matching

Match treated units to control units with similar propensity scores:

```typescript
import { propensityScore, propensityMatching } from 'statistika/causal-inference';

// Outcome variable (e.g., health score improvement)
const outcomes = [2, 3, 8, 10, 9, 1, 4, 7, 11, 12, 2, 3, 9, 10, 11, 1, 5, 8, 10, 13];

const ps = propensityScore(X, treatment);
const match = propensityMatching(outcomes, treatment, ps.scores);

console.log(match.att);          // Average Treatment Effect on the Treated
console.log(match.treatedMean);  // mean outcome for matched treated
console.log(match.controlMean);  // mean outcome for matched controls
console.log(match.matches);     // array of [treated_idx, control_idx] pairs
```

## Inverse Probability Weighting (IPW)

IPW uses the propensity scores to create a pseudo-population where treatment is independent of covariates:

```typescript
import { propensityScore, ipw } from 'statistika/causal-inference';

const ps = propensityScore(X, treatment);
const result = ipw(outcomes, treatment, ps.scores);

console.log(result.ate);    // Average Treatment Effect
console.log(result.att);    // Average Treatment Effect on the Treated
```

The ATE estimates the causal effect of treatment across the entire population. The ATT estimates the effect specifically for those who were treated.

## Difference-in-Differences (DiD)

DiD compares the change in outcomes over time between a treatment group and a control group:

```typescript
import { differenceInDifferences } from 'statistika/causal-inference';

// Outcome values for all units across both periods
const y = [
  // Control group, pre-period
  10, 12, 11, 13, 10,
  // Control group, post-period
  11, 13, 12, 14, 11,
  // Treatment group, pre-period
  10, 11, 12, 13, 11,
  // Treatment group, post-period
  15, 17, 16, 19, 16,
];

// Treatment indicator (0 = control, 1 = treated)
const treat = [
  0, 0, 0, 0, 0,
  0, 0, 0, 0, 0,
  1, 1, 1, 1, 1,
  1, 1, 1, 1, 1,
];

// Post-period indicator (0 = pre, 1 = post)
const post = [
  0, 0, 0, 0, 0,
  1, 1, 1, 1, 1,
  0, 0, 0, 0, 0,
  1, 1, 1, 1, 1,
];

const did = differenceInDifferences(y, treat, post);
console.log(did.estimate);    // DiD estimate: (treat_post - treat_pre) - (ctrl_post - ctrl_pre)
console.log(did.treatPre);    // mean outcome, treatment group, pre-period
console.log(did.treatPost);   // mean outcome, treatment group, post-period
console.log(did.controlPre);  // mean outcome, control group, pre-period
console.log(did.controlPost); // mean outcome, control group, post-period
console.log(did.tStatistic);  // t-statistic for significance
```

The key identification assumption is the **parallel trends** assumption: in the absence of treatment, the treatment and control groups would have followed the same trend.

## Regression Discontinuity Design (RDD)

RDD exploits a cutoff in a running variable that determines treatment assignment. Units just above and just below the cutoff are compared.

### Sharp RDD

In a sharp RDD, treatment is a deterministic function of the running variable:

```typescript
import { rdd } from 'statistika/causal-inference';

// Running variable: test score (treatment if score >= 50)
const n = 100;
const running: number[] = [];
const outcomes: number[] = [];

for (let i = 0; i < n; i++) {
  const score = 20 + 60 * (i / n) + (Math.random() - 0.5) * 5;
  running.push(score);
  // True treatment effect = 5 points at the cutoff
  const treated = score >= 50 ? 1 : 0;
  outcomes.push(30 + 0.5 * score + 5 * treated + (Math.random() - 0.5) * 3);
}

const result = rdd(outcomes, running, 50);
console.log(result.estimate);      // estimated treatment effect (~5)
console.log(result.standardError); // standard error
console.log(result.interceptBelow); // intercept for units below cutoff
console.log(result.interceptAbove); // intercept for units above cutoff
console.log(result.slopeBelow);    // slope below cutoff
console.log(result.slopeAbove);    // slope above cutoff

// With bandwidth restriction (only use observations near the cutoff)
const bw = rdd(outcomes, running, 50, { bandwidth: 10 });
console.log(bw.estimate); // more local estimate
```

### Fuzzy RDD

In a fuzzy RDD, crossing the cutoff increases the probability of treatment but does not guarantee it. This uses 2SLS internally:

```typescript
import { fuzzyRDD } from 'statistika/causal-inference';

// Not everyone above the cutoff actually receives treatment
const actualTreatment = running.map((r, i) => {
  if (r >= 50) return Math.random() > 0.2 ? 1 : 0;  // 80% compliance above
  return Math.random() > 0.9 ? 1 : 0;                // 10% contamination below
});

const fuzzy = fuzzyRDD(outcomes, actualTreatment, running, 50);
console.log(fuzzy.estimate);    // LATE at the cutoff
console.log(fuzzy.firstStageF); // instrument strength (want > 10)
console.log(fuzzy.nUsed);       // observations used
```

## Two-Stage Least Squares (2SLS)

2SLS addresses endogeneity by using instrumental variables. An instrument Z must be:
1. **Relevant**: correlated with the endogenous variable X
2. **Exogenous**: uncorrelated with the error term

```typescript
import { twoSLS } from 'statistika/causal-inference';

// Classic example: returns to schooling
// - Y: wages (outcome)
// - X: years of education (endogenous)
// - Z: instruments (parent education, proximity to college)
const wages = [25, 30, 35, 28, 40, 45, 32, 38, 50, 42,
               27, 33, 36, 41, 48, 29, 34, 39, 44, 52];
const education = [12, 14, 16, 13, 18, 20, 14, 16, 22, 18,
                   12, 14, 16, 17, 20, 13, 15, 17, 19, 22];
const instruments = wages.map((_, i) => [
  10 + Math.floor(education[i] * 0.8 + Math.random() * 4),  // parent education
  Math.random() > 0.5 ? 1 : 0,                                // proximity to college
]);

const result = twoSLS(wages, education, instruments);
console.log(result.slopes[0]);     // causal effect of education on wages
console.log(result.intercept);     // intercept
console.log(result.firstStageF);  // instrument strength (want > 10)
console.log(result.coefficients); // full coefficient vector
```

A first-stage F-statistic below 10 suggests weak instruments, which can lead to biased estimates.

## Choosing the Right Method

| Method | When to Use | Key Assumption |
|---|---|---|
| **Matching/IPW** | Observational data, selection on observables | No unmeasured confounders |
| **DiD** | Panel data, before/after treatment | Parallel trends |
| **Sharp RDD** | Treatment determined by a cutoff | Continuity at cutoff |
| **Fuzzy RDD** | Cutoff affects treatment probability | LATE assumptions |
| **2SLS** | Endogenous regressors | Valid instruments |

## Complete Workflow: Propensity Score Analysis

```typescript
import { propensityScore, propensityMatching, ipw } from 'statistika/causal-inference';

// Step 1: Estimate propensity scores
const ps = propensityScore(X, treatment);

// Step 2: Check overlap
const treated = ps.scores.filter((_, i) => treatment[i] === 1);
const control = ps.scores.filter((_, i) => treatment[i] === 0);
console.log('Treated PS range:', Math.min(...treated).toFixed(3), '-', Math.max(...treated).toFixed(3));
console.log('Control PS range:', Math.min(...control).toFixed(3), '-', Math.max(...control).toFixed(3));
// Good overlap means these ranges substantially intersect.

// Step 3: Estimate treatment effects via matching
const matchResult = propensityMatching(outcomes, treatment, ps.scores);
console.log('ATT (matching):', matchResult.att.toFixed(3));

// Step 4: Estimate via IPW for comparison
const ipwResult = ipw(outcomes, treatment, ps.scores);
console.log('ATE (IPW):', ipwResult.ate.toFixed(3));
console.log('ATT (IPW):', ipwResult.att.toFixed(3));

// Step 5: Compare estimates
// If matching and IPW give similar results, this increases
// confidence in the findings.
```
