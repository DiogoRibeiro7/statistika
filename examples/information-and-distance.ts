/**
 * Example: Information theory and distance metrics.
 *
 * Run: npx ts-node examples/information-and-distance.ts
 */
import {
  entropy, entropyFromData, mutualInformation,
  klDivergence, jsDivergence,
  euclidean, manhattan, cosineDistance, cosineSimilarity,
  mahalanobis, distanceMatrix,
} from "../src";

// --- Information theory ---
console.log("=== Shannon Entropy ===");
console.log(`  Fair coin [0.5, 0.5]: ${entropy([0.5, 0.5]).toFixed(4)} bits`);
console.log(`  Biased coin [0.9, 0.1]: ${entropy([0.9, 0.1]).toFixed(4)} bits`);
console.log(`  Certain [1.0, 0.0]: ${entropy([1.0, 0.0]).toFixed(4)} bits`);
console.log(`  Fair die [1/6 x 6]: ${entropy([1/6, 1/6, 1/6, 1/6, 1/6, 1/6]).toFixed(4)} bits`);
console.log();

console.log("=== Empirical Entropy ===");
const labels = [0, 0, 0, 1, 1, 2, 2, 2, 2, 2];
console.log(`  Data: [${labels.join(", ")}]`);
console.log(`  Entropy: ${entropyFromData(labels).toFixed(4)} bits`);
console.log();

console.log("=== KL and JS Divergence ===");
const p = [0.4, 0.3, 0.2, 0.1];
const q = [0.25, 0.25, 0.25, 0.25];
console.log(`  P: [${p.join(", ")}]`);
console.log(`  Q: [${q.join(", ")}]`);
console.log(`  KL(P || Q): ${klDivergence(p, q).toFixed(4)}`);
console.log(`  JS(P, Q): ${jsDivergence(p, q).toFixed(4)}`);
console.log();

// --- Distance metrics ---
console.log("=== Distance Metrics ===");
const a = [1, 0, 0];
const b = [0, 1, 0];
const c = [1, 1, 0];
console.log(`  a=[${a}], b=[${b}], c=[${c}]`);
console.log(`  Euclidean(a, b): ${euclidean(a, b).toFixed(4)}`);
console.log(`  Manhattan(a, b): ${manhattan(a, b).toFixed(4)}`);
console.log(`  Cosine similarity(a, c): ${cosineSimilarity(a, c).toFixed(4)}`);
console.log(`  Cosine distance(a, b): ${cosineDistance(a, b).toFixed(4)}`);
console.log();

console.log("=== Distance Matrix ===");
const vectors = [[0, 0], [1, 0], [0, 1], [1, 1]];
const dm = distanceMatrix(vectors);
console.log("  Euclidean distances between 4 points:");
for (let i = 0; i < dm.length; i++) {
  console.log(`    [${dm[i].map(d => d.toFixed(2)).join(", ")}]`);
}
