/**
 * Example: Clustering and dimensionality reduction.
 *
 * Run: npx ts-node examples/clustering.ts
 */
import {
  kMeans,
  hierarchicalClustering,
  pca,
  silhouetteScore,
  daviesBouldinIndex,
  gaussianMixture,
  selectComponents,
} from "../src";

// Generate 3 clusters in 2D
let seed = 123;
const rand = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return (seed / 0x7fffffff - 0.5) * 2; };

const cluster1 = Array.from({ length: 20 }, () => [0 + rand(), 0 + rand()]);
const cluster2 = Array.from({ length: 20 }, () => [5 + rand(), 5 + rand()]);
const cluster3 = Array.from({ length: 20 }, () => [10 + rand(), 0 + rand()]);
const data = [...cluster1, ...cluster2, ...cluster3];
const trueLabels = [...Array(20).fill(0), ...Array(20).fill(1), ...Array(20).fill(2)];

console.log("=== K-Means Clustering ===");
const km = kMeans(data, 3, { seed: 42 });
console.log(`  Iterations: ${km.iterations}`);
console.log(`  Centroids:`);
km.centroids.forEach((c, i) => console.log(`    Cluster ${i}: [${c.map(v => v.toFixed(2)).join(", ")}]`));
const sil = silhouetteScore(data, km.assignments);
console.log(`  Silhouette score: ${sil.toFixed(4)}`);
const dbi = daviesBouldinIndex(data, km.assignments);
console.log(`  Davies-Bouldin index: ${dbi.toFixed(4)}`);
console.log();

console.log("=== Hierarchical Clustering ===");
const hc = hierarchicalClustering(data, 3, { linkage: "complete" });
const hcSil = silhouetteScore(data, hc.assignments);
console.log(`  Silhouette score: ${hcSil.toFixed(4)}`);
console.log();

console.log("=== PCA ===");
const pcaResult = pca(data, { nComponents: 2 });
console.log(`  Explained variance: [${pcaResult.explainedVariance.map(v => v.toFixed(4)).join(", ")}]`);
console.log(`  Cumulative variance: [${pcaResult.cumulativeVariance.map(v => (v * 100).toFixed(1) + "%").join(", ")}]`);
console.log();

console.log("=== Gaussian Mixture Model ===");
const gmm = gaussianMixture(data.map(d => d[0]), 3, { seed: 42 });
console.log(`  Means: [${gmm.means.map(m => m.toFixed(2)).join(", ")}]`);
console.log(`  Weights: [${gmm.weights.map(w => w.toFixed(3)).join(", ")}]`);
console.log(`  Iterations: ${gmm.iterations}`);
console.log();

console.log("=== Automatic Component Selection ===");
const sel = selectComponents(data.map(d => d[0]), 5, { seed: 42 });
console.log(`  Best K: ${sel.bestK}`);
console.log(`  BIC values: [${sel.bicValues.map(b => b.toFixed(1)).join(", ")}]`);
