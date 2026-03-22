/**
 * Example: Robust statistics and missing data handling.
 *
 * Run: npx ts-node examples/robust-and-missing.ts
 */
import {
  mad, trimmedMean, winsorizedMean, huberMean,
  detectOutliers, iqr, biweightMidvariance,
  analyzeMissing, meanImputation, medianImputation,
  linearInterpolation, forwardFill,
  mean, stdDev,
} from "../src";

// --- Robust statistics ---
console.log("=== Robust Statistics ===");
const clean = [2.1, 2.5, 2.3, 2.7, 2.4, 2.6, 2.2, 2.8];
const contaminated = [...clean, 50, -30]; // with outliers

console.log(`  Data with outliers: [${contaminated.join(", ")}]`);
console.log(`  Regular mean: ${mean(contaminated).toFixed(2)}`);
console.log(`  Regular stdDev: ${stdDev(contaminated).toFixed(2)}`);
console.log(`  Trimmed mean (10%): ${trimmedMean(contaminated, 0.1).toFixed(2)}`);
console.log(`  Winsorized mean (10%): ${winsorizedMean(contaminated, 0.1).toFixed(2)}`);
console.log(`  Huber M-estimate: ${huberMean(contaminated).toFixed(2)}`);
console.log(`  MAD: ${mad(contaminated).toFixed(2)}`);
console.log(`  IQR: ${iqr(contaminated).toFixed(2)}`);
console.log(`  Biweight midvariance: ${biweightMidvariance(contaminated).toFixed(2)}`);
console.log();

console.log("=== Outlier Detection ===");
const outliers = detectOutliers(contaminated);
console.log(`  Outliers: [${outliers.outliers.join(", ")}]`);
console.log(`  At indices: [${outliers.indices.join(", ")}]`);
console.log(`  Bounds: [${outliers.lower.toFixed(2)}, ${outliers.upper.toFixed(2)}]`);
console.log();

// --- Missing data ---
console.log("=== Missing Data Analysis ===");
const withMissing: (number | null | undefined)[] = [1, null, 3, undefined, 5, NaN, 7, 8, null, 10];

const report = analyzeMissing(withMissing as number[]);
console.log(`  Total values: ${report.totalValues}`);
console.log(`  Missing: ${report.missingCount}`);
console.log(`  Missing rate: ${(report.missingProportion * 100).toFixed(1)}%`);
console.log();

console.log("=== Imputation Methods ===");
console.log(`  Original: [${withMissing.map(v => v ?? "?").join(", ")}]`);
console.log(`  Mean imputation: [${meanImputation(withMissing as number[]).map(v => v.toFixed(1)).join(", ")}]`);
console.log(`  Median imputation: [${medianImputation(withMissing as number[]).map(v => v.toFixed(1)).join(", ")}]`);
console.log(`  Linear interpolation: [${linearInterpolation(withMissing as number[]).map(v => v.toFixed(1)).join(", ")}]`);
console.log(`  Forward fill: [${forwardFill(withMissing as number[]).map(v => v.toFixed(1)).join(", ")}]`);
