# Tutorial: Changepoint Detection

Changepoint detection identifies points in a time series where the statistical properties (mean, variance, or both) shift abruptly. statistika provides four algorithms: CUSUM for single changepoints, PELT and Binary Segmentation for multiple changepoints, and BOCPD for online Bayesian detection.

## CUSUM Test

The CUSUM (Cumulative Sum) test detects a single change in mean. It computes the maximum deviation of the cumulative sum of centered residuals.

```typescript
import { cusumTest } from 'statistika';

// Stable process that shifts at index 50
const data = [
  ...Array.from({ length: 50 }, () => 10 + Math.random() * 2),
  ...Array.from({ length: 50 }, () => 15 + Math.random() * 2),
];

const result = cusumTest(data);

console.log(result.changepoint);
// ~50 — estimated location of the shift

console.log(result.statistic.toFixed(3));
// CUSUM test statistic

console.log(result.pValue.toFixed(4));
// p < 0.05 rejects the null of no changepoint

console.log(result.reject);
// true — there is a significant change

console.log(result.cusumValues.length);
// n+1 — the full CUSUM path for plotting
```

### Plotting the CUSUM Path

The `cusumValues` array contains the normalized CUSUM at each index. The peak corresponds to the estimated changepoint:

```typescript
const peak = result.cusumValues.reduce(
  (max, val, i) => (val > max.val ? { val, idx: i } : max),
  { val: -Infinity, idx: 0 },
);
console.log(`Peak CUSUM at index ${peak.idx}: ${peak.val.toFixed(3)}`);
```

## PELT Algorithm

PELT (Pruned Exact Linear Time) finds the optimal set of multiple changepoints by minimizing a penalized cost function. It is exact and typically runs in O(n) expected time.

```typescript
import { pelt } from 'statistika';

// Three distinct regimes
const data = [
  ...Array(30).fill(0).map(() => 5 + Math.random()),   // mean ~5
  ...Array(30).fill(0).map(() => 15 + Math.random()),  // mean ~15
  ...Array(30).fill(0).map(() => 8 + Math.random()),   // mean ~8
];

const result = pelt(data, {
  costFunction: 'meanvar', // detect changes in mean and variance
  penalty: 'BIC',          // automatic BIC penalty
  minSegmentLength: 5,     // segments must be at least 5 points
});

console.log(result.changepoints);
// [30, 60] — detected changepoint indices

console.log(result.segments);
// [
//   { start: 0, end: 30, mean: 5.5, variance: 0.08 },
//   { start: 30, end: 60, mean: 15.5, variance: 0.08 },
//   { start: 60, end: 90, mean: 8.5, variance: 0.08 },
// ]

console.log(result.cost.toFixed(2));
// Total cost across all segments

console.log(result.penalty);
// "BIC"
```

### Cost Functions

Choose the cost function based on what kind of change you expect:

| Cost function | Detects changes in |
|---|---|
| `"mean"` | Mean only (assumes constant variance) |
| `"variance"` | Variance only (assumes constant mean) |
| `"meanvar"` | Both mean and variance |

```typescript
// Detect only variance changes
const varResult = pelt(data, { costFunction: 'variance' });

// Use a custom numeric penalty instead of BIC
const customResult = pelt(data, { penalty: 10.0 });
```

### Penalty Selection

- **BIC** (default): `k * log(n)` where k is the number of parameters per segment. Conservative, fewer false positives.
- **mBIC**: Modified BIC with an extra `log(log(n))` term. More conservative for large datasets.
- **Numeric**: Supply your own penalty value for fine-grained control.

## Binary Segmentation

Binary Segmentation is a greedy, top-down approach. It is faster than PELT for very long series but may miss some changepoints because it splits recursively rather than optimizing globally.

```typescript
import { binarySegmentation } from 'statistika';

const result = binarySegmentation(data, {
  costFunction: 'meanvar',
  penalty: 'BIC',
  minSegmentLength: 5,
  maxChangepoints: 10,  // stop after finding at most 10
});

console.log(result.changepoints);
// Similar to PELT but found greedily

console.log(result.segments.length);
// Number of segments = changepoints + 1
```

## Bayesian Online Changepoint Detection (BOCPD)

BOCPD processes data sequentially and maintains a probability distribution over run lengths (time since the last changepoint). It is ideal for streaming data.

```typescript
import { bocpd } from 'statistika';

// Simulate a mean shift from 0 to 5 at index 40
const data = [
  ...Array.from({ length: 40 }, () => Math.random() - 0.5),
  ...Array.from({ length: 40 }, () => 5 + Math.random() - 0.5),
];

const result = bocpd(data, {
  hazardLambda: 1 / 20,  // expect changepoints every ~20 steps
  priorMu: 0,
  priorKappa: 1,
  priorAlpha: 1,
  priorBeta: 1,
  threshold: 0.5,  // declare changepoint when P(run_length=0) > 0.5
});

console.log(result.changepoints);
// [~40] — detected changepoint indices

console.log(result.maxRunLengthProb);
// Most probable run length at each time step
// Drops to 0 at changepoints, then grows linearly

console.log(result.runLengthProbabilities.length);
// n — full posterior over run lengths at each step
```

### Tuning the Hazard Rate

The hazard rate `lambda` controls how frequently changepoints are expected:

- **Higher lambda** (e.g., 1/10): expects frequent changes, more sensitive
- **Lower lambda** (e.g., 1/1000): expects rare changes, fewer false alarms

```typescript
// Sensitive detector for frequently changing signals
const sensitive = bocpd(data, { hazardLambda: 1 / 10, threshold: 0.3 });

// Conservative detector for stable processes
const conservative = bocpd(data, { hazardLambda: 1 / 500, threshold: 0.7 });
```

## Comparing Methods

| Method | Type | Changepoints | Best for |
|---|---|---|---|
| `cusumTest` | Offline | Single | Quick hypothesis test for one shift |
| `pelt` | Offline | Multiple | Optimal segmentation of moderate-length series |
| `binarySegmentation` | Offline | Multiple | Fast approximate segmentation of long series |
| `bocpd` | Online | Multiple | Streaming data, real-time monitoring |

## Complete Example: Sensor Monitoring

```typescript
import { pelt, cusumTest, bocpd } from 'statistika';

// Simulated temperature sensor readings (one per minute)
const readings: number[] = [];
for (let i = 0; i < 200; i++) {
  if (i < 60) readings.push(22 + Math.random() * 0.5);
  else if (i < 130) readings.push(28 + Math.random() * 0.5);
  else readings.push(22 + Math.random() * 0.5);
}

// Detect all changepoints with PELT
const result = pelt(readings, { costFunction: 'mean', penalty: 'BIC' });

console.log('Changepoints:', result.changepoints);
// [60, 130]

for (const seg of result.segments) {
  console.log(
    `Minutes ${seg.start}–${seg.end}: ` +
    `mean=${seg.mean.toFixed(1)}°C, var=${seg.variance.toFixed(3)}`
  );
}
// Minutes 0–60: mean=22.3°C, var=0.021
// Minutes 60–130: mean=28.3°C, var=0.020
// Minutes 130–200: mean=22.2°C, var=0.022
```

## Function Reference

| Function | Description |
|---|---|
| `cusumTest(data, alpha?)` | CUSUM test for a single mean shift |
| `pelt(data, options?)` | Optimal multiple changepoint detection |
| `binarySegmentation(data, options?)` | Greedy top-down segmentation |
| `bocpd(data, options?)` | Bayesian online changepoint detection |
| `costMean(data, start, end)` | Segment cost for change-in-mean |
| `costVariance(data, start, end)` | Segment cost for change-in-variance |
| `costMeanVar(data, start, end)` | Segment cost for change in both |
