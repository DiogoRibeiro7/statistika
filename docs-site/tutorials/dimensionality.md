# Tutorial: Dimensionality Reduction & Clustering

node_stats provides a full suite of dimensionality reduction techniques (PCA, t-SNE, UMAP) and clustering algorithms (k-means, hierarchical, DBSCAN, GMM) along with cluster validation metrics.

## Principal Component Analysis (PCA)

PCA projects high-dimensional data onto orthogonal axes of maximum variance. Use it to visualize data, reduce feature sets, or remove noise.

```typescript
import { pca } from 'node_stats';

// Iris-like data: 4 features per observation
const data = [
  [5.1, 3.5, 1.4, 0.2],
  [4.9, 3.0, 1.4, 0.2],
  [7.0, 3.2, 4.7, 1.4],
  [6.4, 3.2, 4.5, 1.5],
  [6.3, 3.3, 6.0, 2.5],
  [5.8, 2.7, 5.1, 1.9],
];

const result = pca(data, { nComponents: 2, center: true, scale: true });

console.log(result.explainedVariance);
// [0.92, 0.05] — first PC captures ~92% of variance

console.log(result.cumulativeVariance);
// [0.92, 0.97] — first two PCs capture ~97%

console.log(result.scores);
// n x 2 matrix of projected coordinates
```

### Interpreting Loadings

Each principal component is a linear combination of the original features. Inspect the components to understand which features drive each axis:

```typescript
console.log(result.components[0]);
// Loadings for PC1 — large values indicate influential features

console.log(result.eigenvalues);
// Eigenvalues in descending order — magnitude indicates importance
```

## t-SNE

t-SNE is a nonlinear technique best suited for visualizing high-dimensional data in 2D. It preserves local neighborhood structure but does not preserve global distances.

```typescript
import { tsne } from 'node_stats';

const highDimData = [
  [1, 2, 3, 4, 5],
  [1, 2, 3, 4, 6],
  [10, 20, 30, 40, 50],
  [10, 20, 30, 40, 51],
  [100, 200, 300, 400, 500],
  [100, 200, 300, 400, 501],
];

const result = tsne(highDimData, {
  perplexity: 2,
  iterations: 500,
  learningRate: 200,
  seed: 42,
});

console.log(result.embedding);
// [[x1, y1], [x2, y2], ...] — 2D coordinates

console.log(result.klDivergence);
// Lower values indicate a better embedding

console.log(result.iterations);
// Number of gradient descent iterations performed
```

## UMAP

UMAP produces embeddings similar to t-SNE but often runs faster and better preserves global structure.

```typescript
import { umap } from 'node_stats';

const result = umap(highDimData, {
  nNeighbors: 3,
  minDist: 0.1,
  nComponents: 2,
  seed: 42,
});

console.log(result.embedding);
// 2D coordinates, similar to t-SNE output

console.log(result.nEpochs);
// Optimization epochs performed
```

## k-Means Clustering

k-Means partitions data into k groups by minimizing within-cluster sum of squares. It uses k-means++ initialization and multiple restarts for robustness.

```typescript
import { kMeans } from 'node_stats';

const data = [
  [1, 2], [1.5, 1.8], [1.2, 2.1],
  [8, 8], [8.5, 7.5], [9, 8],
  [5, 5], [5.5, 5.2], [4.8, 4.9],
];

const result = kMeans(data, 3, { seed: 42, nInit: 10 });

console.log(result.assignments);
// [0, 0, 0, 1, 1, 1, 2, 2, 2]

console.log(result.centroids);
// [[1.23, 1.97], [8.5, 7.83], [5.1, 5.03]]

console.log(result.totalWCSS);
// Total within-cluster sum of squares — lower is tighter
```

## Hierarchical Clustering

Agglomerative clustering builds a tree of merges. You choose a linkage criterion and the desired number of clusters.

```typescript
import { hierarchicalClustering } from 'node_stats';

const result = hierarchicalClustering(data, 3, {
  linkage: 'complete', // also: 'single', 'average'
});

console.log(result.assignments);
// Cluster labels for each observation

console.log(result.merges);
// Merge history: [[i, j, distance], ...] for dendrogram
```

## DBSCAN

DBSCAN discovers clusters of arbitrary shape based on point density. It automatically determines the number of clusters and labels sparse points as noise.

```typescript
import { dbscan } from 'node_stats';

const data = [
  [0, 0], [0.1, 0], [0, 0.1],
  [10, 10], [10.1, 10], [10, 10.1],
  [50, 50], // isolated noise point
];

const result = dbscan(data, { epsilon: 1, minPoints: 2 });

console.log(result.labels);
// [0, 0, 0, 1, 1, 1, -1] — noise is labeled -1

console.log(result.nClusters); // 2
console.log(result.nNoise);    // 1
console.log(result.corePoints);
// Indices of core points (those with >= minPoints neighbors)
```

## Gaussian Mixture Models (GMM)

GMMs fit a probabilistic model of k Gaussian components using the EM algorithm. Each observation gets soft (probabilistic) assignments.

```typescript
import { gaussianMixture } from 'node_stats';

// Bimodal 1D data
const data = [1.0, 1.2, 0.9, 1.1, 5.0, 5.2, 4.8, 5.1];

const result = gaussianMixture(data, 2, { seed: 42 });

console.log(result.means);      // ~[1.05, 5.03]
console.log(result.variances);  // variance of each component
console.log(result.weights);    // mixing proportions (sum to 1)
console.log(result.labels);     // hard cluster assignments
console.log(result.bic);        // use BIC to choose k
```

### Selecting the Number of Components

Use BIC to compare models with different numbers of components:

```typescript
import { selectComponents } from 'node_stats';

const best = selectComponents(data, 1, 5, { seed: 42 });
// Fits GMMs with k = 1..5 and returns the one with the lowest BIC
```

## Cluster Validation

### Silhouette Score

The silhouette score measures how similar each point is to its own cluster compared to other clusters. Values range from -1 to +1; higher is better.

```typescript
import { silhouetteScore, kMeans } from 'node_stats';

const clusters = kMeans(data, 3, { seed: 42 });
const score = silhouetteScore(data, clusters.assignments);

console.log(score);
// Close to 1.0 for well-separated clusters
```

### Davies-Bouldin Index

Lower values indicate better clustering. Unlike silhouette, it does not require pairwise distance computation for every point.

```typescript
import { daviesBouldinIndex } from 'node_stats';

const dbi = daviesBouldinIndex(data, clusters.assignments);
console.log(dbi);
// Lower is better — 0 is perfect separation
```

### Adjusted Rand Index

Compare two clustering solutions (e.g., predicted vs. true labels):

```typescript
import { adjustedRandIndex } from 'node_stats';

const trueLabels = [0, 0, 0, 1, 1, 1, 2, 2, 2];
const predLabels = clusters.assignments;

const ari = adjustedRandIndex(trueLabels, predLabels);
console.log(ari);
// 1.0 for perfect agreement, 0 for random
```

## Putting It All Together

A typical workflow: reduce dimensionality with PCA, cluster in the reduced space, and validate the result.

```typescript
import { pca, kMeans, silhouetteScore, tsne } from 'node_stats';

// 1. Reduce to 3 principal components
const pcaResult = pca(rawData, { nComponents: 3 });

// 2. Cluster in PC space
const clusters = kMeans(pcaResult.scores, 3, { seed: 42 });

// 3. Validate
const score = silhouetteScore(pcaResult.scores, clusters.assignments);
console.log(`Silhouette score: ${score.toFixed(3)}`);

// 4. Visualize with t-SNE
const viz = tsne(rawData, { perplexity: 15, seed: 42 });
// Plot viz.embedding colored by clusters.assignments
```
