# Tutorial: Spatial Statistics

Spatial statistics analyzes data that is tied to geographic or spatial locations. statistika provides tools for measuring spatial autocorrelation (Moran's I, Geary's C), modeling spatial dependence (variograms), and performing spatial interpolation (kriging).

## Spatial Weights

Most spatial analyses require a weights matrix that encodes which locations are "neighbors." statistika provides two common approaches:

### Distance Band Weights

Locations within a threshold distance are neighbors:

```typescript
import { distanceBandWeights } from 'statistika';

const locations = [
  [0, 0], [1, 0], [2, 0],
  [0, 1], [1, 1], [2, 1],
  [0, 2], [1, 2], [2, 2],
];

const W = distanceBandWeights(locations, 1.5);
// W[i][j] = 1 if distance(i, j) <= 1.5, else 0
// Diagonal is always 0
```

### k-Nearest Neighbor Weights

Each location's k closest points are its neighbors:

```typescript
import { knnWeights } from 'statistika';

const W = knnWeights(locations, 4);
// Each location has exactly 4 neighbors
// Note: the resulting matrix may be asymmetric
```

### Distance Matrix

You can also compute the full pairwise distance matrix directly:

```typescript
import { spatialDistanceMatrix } from 'statistika';

const D = spatialDistanceMatrix(locations);
// D[i][j] = Euclidean distance between locations i and j
console.log(D[0][4]); // distance from (0,0) to (1,1) = sqrt(2)
```

## Moran's I

Moran's I is the most widely used measure of global spatial autocorrelation. It tests whether nearby locations have similar values (positive autocorrelation) or dissimilar values (negative autocorrelation).

```typescript
import { moranI, distanceBandWeights } from 'statistika';

// Property values at 9 grid locations
const values = [
  100, 105, 110,
  102, 108, 115,
  95, 100, 112,
];

const W = distanceBandWeights(locations, 1.5);
const result = moranI(values, W);

console.log(result.I.toFixed(4));
// Moran's I statistic
// Positive: similar values cluster together
// Negative: dissimilar values are adjacent
// Near E[I]: random spatial pattern

console.log(result.expectedI.toFixed(4));
// Expected I under H0 (no autocorrelation): -1/(n-1)

console.log(result.zScore.toFixed(3));
// z-score for significance testing

console.log(result.pValue.toFixed(4));
// Two-sided p-value
// p < 0.05 indicates significant spatial autocorrelation
```

### Interpreting Moran's I

| Value | Interpretation |
|---|---|
| I > E[I] | Positive autocorrelation (clustering of similar values) |
| I ~ E[I] | Random spatial pattern |
| I < E[I] | Negative autocorrelation (checkerboard pattern) |

## Geary's C

Geary's C is an alternative to Moran's I that focuses on pairwise differences rather than deviations from the mean. It is more sensitive to local spatial autocorrelation.

```typescript
import { gearyC } from 'statistika';

const result = gearyC(values, W);

console.log(result.C.toFixed(4));
// Geary's C statistic
// C < 1: positive autocorrelation
// C = 1: no autocorrelation
// C > 1: negative autocorrelation

console.log(result.zScore.toFixed(3));
console.log(result.pValue.toFixed(4));
```

## Empirical Variogram

The variogram describes how spatial dependence changes with distance. The semivariance increases with distance until reaching a "sill," beyond which points are no longer correlated.

```typescript
import { empiricalVariogram } from 'statistika';

// Soil contamination measurements at scattered locations
const points = [
  [0, 0], [1, 0], [2, 0], [3, 0],
  [0, 1], [1, 1], [2, 1], [3, 1],
  [0, 2], [1, 2], [2, 2], [3, 2],
];
const contamination = [
  2.1, 2.5, 3.8, 4.2,
  2.3, 2.9, 3.5, 4.0,
  1.8, 2.2, 3.0, 3.8,
];

const bins = empiricalVariogram(points, contamination, 10);

for (const bin of bins) {
  console.log(
    `Distance ${bin.distance.toFixed(2)}: ` +
    `semivariance = ${bin.semivariance.toFixed(3)} ` +
    `(${bin.count} pairs)`
  );
}
```

## Fitting a Variogram Model

Fit a parametric model to the empirical variogram using weighted least squares. The fitted model has three key parameters:

- **Nugget**: discontinuity at distance 0 (measurement error or micro-scale variation)
- **Sill**: total variance at which the variogram levels off
- **Range**: distance at which spatial correlation effectively reaches zero

```typescript
import { fitVariogramModel } from 'statistika';

const model = fitVariogramModel(bins, 'spherical');
// Also available: 'exponential', 'gaussian', 'linear'

console.log(`Nugget: ${model.nugget.toFixed(3)}`);
console.log(`Sill:   ${model.sill.toFixed(3)}`);
console.log(`Range:  ${model.range.toFixed(3)}`);

// Evaluate the model at any distance
console.log(model.evaluate(0));    // 0 (by convention)
console.log(model.evaluate(1.5));  // semivariance at distance 1.5
console.log(model.evaluate(10));   // near nugget + sill (beyond range)
```

### Variogram Model Types

| Model | Shape | Use case |
|---|---|---|
| `spherical` | Linear near origin, flattens at range | Most common default |
| `exponential` | Approaches sill asymptotically | Smooth transitions |
| `gaussian` | Parabolic near origin | Very smooth phenomena |
| `linear` | Straight line to range | Simple, unbounded |

## Ordinary Kriging

Kriging uses the fitted variogram to interpolate values at unobserved locations. It provides both a prediction and an estimate of uncertainty (kriging variance).

```typescript
import { ordinaryKriging, empiricalVariogram, fitVariogramModel } from 'statistika';

// Fit the variogram
const bins = empiricalVariogram(points, contamination, 10);
const model = fitVariogramModel(bins, 'spherical');

// Predict at new locations
const queryPoints = [
  [0.5, 0.5],
  [1.5, 1.5],
  [2.5, 0.5],
];

const result = ordinaryKriging(points, contamination, queryPoints, model);

for (let i = 0; i < queryPoints.length; i++) {
  const [x, y] = queryPoints[i];
  console.log(
    `(${x}, ${y}): predicted = ${result.predictions[i].toFixed(2)}, ` +
    `variance = ${result.variances[i].toFixed(3)}`
  );
}
// (0.5, 0.5): predicted = 2.35, variance = 0.082
// (1.5, 1.5): predicted = 2.85, variance = 0.045
// (2.5, 0.5): predicted = 3.72, variance = 0.078
```

Kriging variance is lowest near observed points and increases as you move away from the data.

## Complete Example: Analyzing Geographic Data

```typescript
import {
  moranI,
  gearyC,
  distanceBandWeights,
  empiricalVariogram,
  fitVariogramModel,
  ordinaryKriging,
} from 'statistika';

// Rainfall measurements at 12 weather stations
const stations = [
  [10, 20], [15, 25], [20, 15], [25, 30],
  [30, 10], [35, 20], [40, 25], [45, 15],
  [10, 40], [20, 35], [30, 40], [40, 35],
];
const rainfall = [
  52, 55, 48, 60, 45, 50, 58, 42,
  65, 62, 68, 55,
];

// Step 1: Test for spatial autocorrelation
const W = distanceBandWeights(stations, 15);
const moran = moranI(rainfall, W);
console.log(`Moran's I = ${moran.I.toFixed(3)}, p = ${moran.pValue.toFixed(4)}`);

const geary = gearyC(rainfall, W);
console.log(`Geary's C = ${geary.C.toFixed(3)}, p = ${geary.pValue.toFixed(4)}`);

// Step 2: Model the spatial structure
const bins = empiricalVariogram(stations, rainfall, 8);
const model = fitVariogramModel(bins, 'exponential');
console.log(`Variogram — nugget: ${model.nugget.toFixed(1)}, sill: ${model.sill.toFixed(1)}, range: ${model.range.toFixed(1)}`);

// Step 3: Interpolate at ungauged locations
const newLocations = [
  [22, 22], [32, 28], [18, 38],
];
const kriged = ordinaryKriging(stations, rainfall, newLocations, model);
for (let i = 0; i < newLocations.length; i++) {
  console.log(
    `Location (${newLocations[i]}): ${kriged.predictions[i].toFixed(1)} mm ` +
    `+/- ${Math.sqrt(kriged.variances[i]).toFixed(1)}`
  );
}
```

## Function Reference

| Function | Description |
|---|---|
| `spatialDistanceMatrix(points)` | Pairwise Euclidean distance matrix |
| `distanceBandWeights(points, threshold)` | Binary weights from distance threshold |
| `knnWeights(points, k)` | k-nearest-neighbor weights |
| `moranI(values, W)` | Global Moran's I test |
| `gearyC(values, W)` | Geary's C test |
| `empiricalVariogram(points, values, nBins?, maxDist?)` | Empirical semivariogram |
| `fitVariogramModel(bins, type?)` | Fit parametric variogram model |
| `ordinaryKriging(points, values, queryPoints, model)` | Kriging interpolation |
