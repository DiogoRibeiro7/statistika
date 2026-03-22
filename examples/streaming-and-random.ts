/**
 * Example: Streaming statistics and random number generation.
 *
 * Run: npx ts-node examples/streaming-and-random.ts
 */
import {
  OnlineStats, OnlineCovariance, OnlineQuantile,
  SeededRng, haltonSequence, latinHypercube,
} from "../src";

// --- Streaming statistics ---
console.log("=== Online Statistics (Welford's Algorithm) ===");
const stats = new OnlineStats();
const values = [4.2, 3.8, 5.1, 4.7, 3.9, 5.3, 4.1, 4.6];
for (const v of values) stats.push(v);
console.log(`  Values: [${values.join(", ")}]`);
console.log(`  Count: ${stats.count}`);
console.log(`  Mean: ${stats.mean.toFixed(4)}`);
console.log(`  Variance: ${stats.variance.toFixed(4)}`);
console.log(`  Std Dev: ${stats.stdDev.toFixed(4)}`);
console.log(`  Min: ${stats.min}, Max: ${stats.max}`);
console.log();

console.log("=== Online Covariance ===");
const cov = new OnlineCovariance();
const xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
const ys = xs.map(x => 2 * x + 1);
for (let i = 0; i < xs.length; i++) cov.push(xs[i], ys[i]);
console.log(`  Covariance: ${cov.covariance.toFixed(4)}`);
console.log(`  Correlation: ${cov.correlation.toFixed(4)}`);
console.log();

console.log("=== Online Quantile (P² Algorithm) ===");
const q50 = new OnlineQuantile(0.5);
const q90 = new OnlineQuantile(0.9);
const data = [3, 6, 7, 8, 8, 10, 13, 15, 16, 20, 1, 5, 9, 11, 14];
for (const v of data) { q50.push(v); q90.push(v); }
console.log(`  Data: [${data.join(", ")}]`);
console.log(`  Estimated median: ${q50.estimate.toFixed(2)}`);
console.log(`  Estimated 90th percentile: ${q90.estimate.toFixed(2)}`);
console.log();

// --- Random number generation ---
console.log("=== Seeded Random Number Generator ===");
const rng = new SeededRng(42);
console.log(`  5 uniform: [${Array.from({ length: 5 }, () => rng.next().toFixed(4)).join(", ")}]`);
const rng2 = new SeededRng(42);
console.log(`  Same seed: [${Array.from({ length: 5 }, () => rng2.next().toFixed(4)).join(", ")}]`);
console.log();

console.log("=== Halton Sequence (Low-Discrepancy) ===");
const halton = haltonSequence(2, 8);
console.log(`  Base-2, 8 points: [${halton.map(v => v.toFixed(4)).join(", ")}]`);
console.log();

console.log("=== Latin Hypercube Sampling ===");
const lhs = latinHypercube(2, 5, 42);
console.log(`  2D, 5 samples:`);
for (const pt of lhs) {
  console.log(`    [${pt.map(v => v.toFixed(4)).join(", ")}]`);
}
