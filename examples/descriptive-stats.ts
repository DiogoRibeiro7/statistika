/**
 * Example: Descriptive statistics and confidence intervals.
 *
 * Run: npx ts-node examples/descriptive-stats.ts
 */

import {
  mean,
  median,
  variance,
  stdDev,
  skewness,
  kurtosis,
  percentile,
  describe,
  meanCI,
  proportionCI,
  pearsonCorrelation,
  spearmanCorrelation,
} from "../src";

// Dataset: exam scores
const scores = [
  72, 85, 90, 68, 94, 78, 88, 76, 82, 95, 70, 87, 91, 65, 80, 73, 89, 84,
  77, 93, 86, 79, 81, 92, 75, 88, 71, 83, 96, 74,
];

console.log("=== Descriptive Statistics (exam scores) ===");
const stats = describe(scores);
console.log(`  N: ${stats.count}`);
console.log(`  Mean: ${stats.mean.toFixed(2)}`);
console.log(`  Median: ${stats.median.toFixed(2)}`);
console.log(`  Variance: ${stats.variance.toFixed(2)}`);
console.log(`  Std Dev: ${stats.stdDev.toFixed(2)}`);
console.log(`  Skewness: ${skewness(scores).toFixed(4)}`);
console.log(`  Kurtosis: ${kurtosis(scores).toFixed(4)}`);
console.log(`  Min: ${stats.min}, Max: ${stats.max}`);
console.log(`  Q1: ${percentile(scores, 25).toFixed(1)}`);
console.log(`  Q3: ${percentile(scores, 75).toFixed(1)}`);
console.log(`  IQR: ${(percentile(scores, 75) - percentile(scores, 25)).toFixed(1)}`);
console.log(`  P10: ${percentile(scores, 10).toFixed(1)}`);
console.log(`  P90: ${percentile(scores, 90).toFixed(1)}`);
console.log();

// --- Confidence intervals ---
console.log("=== Confidence Intervals ===");
const ci = meanCI(scores);
console.log(`  Mean 95% CI: [${ci.lower.toFixed(2)}, ${ci.upper.toFixed(2)}]`);
console.log(`  Margin of error: +/- ${ci.marginOfError.toFixed(2)}`);

// Proportion: 18 out of 30 scored above 80
const propCI = proportionCI(18, 30);
console.log(`  Proportion >80: ${((18 / 30) * 100).toFixed(1)}%`);
console.log(
  `  95% CI: [${(propCI.lower * 100).toFixed(1)}%, ${(propCI.upper * 100).toFixed(1)}%]`,
);
console.log();

// --- Correlation ---
// Study hours vs exam scores (subset)
const studyHours = [
  2, 3, 5, 1, 7, 4, 6, 3, 4, 8, 2, 5, 6, 1, 4, 3, 6, 5, 3, 7, 5, 3, 4, 7,
  3, 6, 2, 4, 8, 3,
];

console.log("=== Correlation (study hours vs scores) ===");
const pearson = pearsonCorrelation(studyHours, scores);
console.log(
  `  Pearson r: ${pearson.coefficient.toFixed(4)}, p = ${pearson.pValue.toFixed(4)}`,
);

const spearman = spearmanCorrelation(studyHours, scores);
console.log(
  `  Spearman rho: ${spearman.coefficient.toFixed(4)}, p = ${spearman.pValue.toFixed(4)}`,
);
