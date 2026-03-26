# Tutorial: Meta-Analysis

Meta-analysis combines results from multiple independent studies to produce a single, more precise estimate. statistika provides fixed-effects and random-effects models, heterogeneity statistics, forest and funnel plot data, and publication bias tests.

## Fixed-Effects Meta-Analysis

A fixed-effects model assumes all studies estimate the same true effect. Each study is weighted by the inverse of its variance:

```typescript
import { fixedEffectsMeta } from 'statistika';

// Five clinical trials reporting standardized mean differences
const effects   = [0.52, 0.31, 0.67, 0.45, 0.58];
const variances = [0.04, 0.06, 0.05, 0.03, 0.07];

const result = fixedEffectsMeta(effects, variances);

console.log(result.pooledEffect.toFixed(3));
// Weighted average effect size

console.log(result.standardError.toFixed(3));
// SE of the pooled estimate

console.log(result.confidenceInterval);
// { lower: ..., upper: ... } — 95% CI

console.log(result.pValue.toFixed(4));
// Test of H0: pooled effect = 0

console.log(result.weights);
// Inverse-variance weights for each study
```

## Random-Effects Meta-Analysis

When studies may estimate different (but related) true effects, use a random-effects model. It incorporates between-study variance using the DerSimonian-Laird estimator:

```typescript
import { randomEffectsMeta } from 'statistika';

const result = randomEffectsMeta(effects, variances);

console.log(result.pooledEffect.toFixed(3));
// Pooled effect (typically wider CI than fixed-effects)

console.log(result.confidenceInterval);
// Wider CI reflecting between-study heterogeneity
```

## Heterogeneity Statistics

The random-effects result includes a full set of heterogeneity measures:

```typescript
console.log(result.q.toFixed(2));
// Cochran's Q — tests H0: all studies share the same effect

console.log(result.qPValue.toFixed(4));
// p-value for Q — significant means heterogeneity is present

console.log(result.i2.toFixed(1));
// I-squared (%) — proportion of variability due to heterogeneity
// 0% = no heterogeneity, 25% = low, 50% = moderate, 75% = high

console.log(result.tau2.toFixed(4));
// Tau-squared — estimated between-study variance

console.log(result.h2.toFixed(2));
// H-squared — ratio of Q to its degrees of freedom
```

### Interpreting I-squared

| I-squared | Heterogeneity |
|---|---|
| 0--25% | Low |
| 25--50% | Moderate |
| 50--75% | Substantial |
| 75--100% | Considerable |

## Forest Plot Data

Generate structured data for rendering a forest plot. Each study is shown with its effect size, confidence interval, and relative weight:

```typescript
import { forestPlotData } from 'statistika';

const labels = [
  'Smith 2018',
  'Jones 2019',
  'Wang 2020',
  'Garcia 2021',
  'Kim 2022',
];

const plot = forestPlotData(effects, variances, labels);

// Per-study data
for (const study of plot.studies) {
  console.log(
    `${study.label}: effect=${study.effect.toFixed(2)} ` +
    `[${study.lower.toFixed(2)}, ${study.upper.toFixed(2)}] ` +
    `weight=${study.weight.toFixed(1)}%`
  );
}

// Overall pooled estimate
console.log(
  `Overall: ${plot.overall.effect.toFixed(2)} ` +
  `[${plot.overall.lower.toFixed(2)}, ${plot.overall.upper.toFixed(2)}]`
);

console.log(plot.method); // "random"
```

## Funnel Plot Data

Funnel plots help detect publication bias visually. In the absence of bias, studies should form a symmetric inverted funnel around the pooled effect:

```typescript
import { funnelPlotData } from 'statistika';

const ses = variances.map(v => Math.sqrt(v));
const funnel = funnelPlotData(effects, ses);

console.log(funnel.pooledEffect.toFixed(3));
// Center line of the funnel

console.log(funnel.points);
// [{ effect, se }, ...] — one per study

console.log(funnel.pseudoCI.length);
// 51 boundary points for the 95% pseudo-CI funnel shape
```

## Publication Bias Tests

### Egger's Regression Test

Egger's test checks for funnel plot asymmetry by regressing standardized effects on precision. A significant intercept suggests small-study bias:

```typescript
import { eggersTest } from 'statistika';

const test = eggersTest(effects, ses);

console.log(test.statistic.toFixed(3));
// t-statistic for the intercept

console.log(test.pValue.toFixed(4));
// p-value — significant at alpha = 0.10 suggests bias

console.log(test.reject);
// true if p < 0.10

console.log(test.method);
// "Egger's regression test"
```

### Begg's Rank Correlation Test

Begg's test uses Kendall's tau to check for correlation between effect sizes and their variances:

```typescript
import { beggsTest } from 'statistika';

const begg = beggsTest(effects, variances);

console.log(begg.statistic.toFixed(3));
// Kendall's tau

console.log(begg.pValue.toFixed(4));
// Two-sided p-value

console.log(begg.method);
// "Begg's rank correlation test"
```

### Trim-and-Fill

The trim-and-fill method estimates the number of missing studies and adjusts the pooled estimate:

```typescript
import { trimAndFill } from 'statistika';

const tf = trimAndFill(effects, variances);

console.log(tf.nMissing);
// Estimated number of suppressed studies

console.log(tf.originalPooled.toFixed(3));
// Original pooled effect

console.log(tf.adjustedPooled.toFixed(3));
// Adjusted pooled effect after imputing missing studies

console.log(tf.adjustedCI);
// { lower, upper } — adjusted confidence interval
```

## Complete Example: Combining Clinical Trials

```typescript
import {
  randomEffectsMeta,
  forestPlotData,
  eggersTest,
  trimAndFill,
} from 'statistika';

// Six RCTs reporting odds ratios (log scale)
const logOR     = [0.22, 0.35, 0.18, 0.40, 0.28, 0.50];
const variances = [0.03, 0.05, 0.04, 0.02, 0.06, 0.03];
const labels    = [
  'Trial A (n=200)', 'Trial B (n=150)',
  'Trial C (n=300)', 'Trial D (n=500)',
  'Trial E (n=120)', 'Trial F (n=250)',
];

// Step 1: Pool the results
const meta = randomEffectsMeta(logOR, variances);
console.log(`Pooled log-OR: ${meta.pooledEffect.toFixed(3)}`);
console.log(`95% CI: [${meta.confidenceInterval.lower.toFixed(3)}, ${meta.confidenceInterval.upper.toFixed(3)}]`);
console.log(`I-squared: ${meta.i2.toFixed(1)}%`);

// Step 2: Generate forest plot data
const forest = forestPlotData(logOR, variances, labels);

// Step 3: Check for publication bias
const ses = variances.map(v => Math.sqrt(v));
const egger = eggersTest(logOR, ses);
console.log(`Egger's test p-value: ${egger.pValue.toFixed(4)}`);

// Step 4: Adjust for bias if present
const tf = trimAndFill(logOR, variances);
console.log(`Missing studies: ${tf.nMissing}`);
console.log(`Adjusted log-OR: ${tf.adjustedPooled.toFixed(3)}`);
```

## Function Reference

| Function | Description |
|---|---|
| `fixedEffectsMeta(effects, variances)` | Inverse-variance weighted fixed-effects model |
| `randomEffectsMeta(effects, variances)` | DerSimonian-Laird random-effects model |
| `forestPlotData(effects, variances, labels)` | Structured data for forest plot rendering |
| `funnelPlotData(effects, standardErrors)` | Structured data for funnel plot rendering |
| `eggersTest(effects, standardErrors)` | Egger's regression test for asymmetry |
| `beggsTest(effects, variances)` | Begg's rank correlation test |
| `trimAndFill(effects, variances)` | Trim-and-fill bias adjustment |
