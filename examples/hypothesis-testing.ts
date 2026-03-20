/**
 * Example: Hypothesis testing.
 *
 * Run: npx ts-node examples/hypothesis-testing.ts
 */

import {
  oneSampleTTest,
  twoSampleTTest,
  chiSquaredGoodnessOfFit,
  mannWhitneyU,
  oneWayAnova,
} from "../src";

// --- One-sample t-test ---
// Test if mean study time differs from 5 hours
const studyHours = [5.2, 4.8, 6.1, 5.5, 5.9, 4.7, 6.3, 5.1, 5.8, 6.0];
const oneSample = oneSampleTTest(studyHours, 5.0);
console.log("=== One-Sample t-Test (H0: mu = 5 hours) ===");
console.log(`  t-statistic: ${oneSample.statistic.toFixed(4)}`);
console.log(`  p-value: ${oneSample.pValue.toFixed(4)}`);
console.log(`  df: ${oneSample.degreesOfFreedom}`);
console.log(`  Reject H0 at alpha=0.05? ${oneSample.rejected}`);
console.log();

// --- Two-sample t-test ---
// Compare test scores between two teaching methods
const methodA = [78, 85, 92, 88, 76, 95, 89, 84];
const methodB = [65, 72, 80, 68, 75, 70, 77, 73];
const twoSample = twoSampleTTest(methodA, methodB);
console.log("=== Two-Sample t-Test (teaching methods) ===");
console.log(`  t-statistic: ${twoSample.statistic.toFixed(4)}`);
console.log(`  p-value: ${twoSample.pValue.toFixed(4)}`);
console.log(`  Reject H0 at alpha=0.05? ${twoSample.rejected}`);
console.log();

// --- Chi-squared goodness of fit ---
// Test if a die is fair
const observed = [18, 22, 15, 20, 25, 20]; // 120 rolls
const expected = [20, 20, 20, 20, 20, 20];
const chiSq = chiSquaredGoodnessOfFit(observed, expected);
console.log("=== Chi-Squared Goodness of Fit (fair die?) ===");
console.log(`  Chi-squared: ${chiSq.statistic.toFixed(4)}`);
console.log(`  p-value: ${chiSq.pValue.toFixed(4)}`);
console.log(`  df: ${chiSq.degreesOfFreedom}`);
console.log(`  Reject H0 at alpha=0.05? ${chiSq.rejected}`);
console.log();

// --- Mann-Whitney U test ---
// Non-parametric comparison of two independent samples
const groupX = [45, 52, 48, 41, 56, 50];
const groupY = [38, 42, 35, 40, 36, 44];
const mw = mannWhitneyU(groupX, groupY);
console.log("=== Mann-Whitney U Test ===");
console.log(`  U-statistic: ${mw.statistic.toFixed(4)}`);
console.log(`  p-value: ${mw.pValue.toFixed(4)}`);
console.log(`  Reject H0 at alpha=0.05? ${mw.rejected}`);
console.log();

// --- One-way ANOVA ---
// Compare three fertilizer treatments
const fert1 = [20.1, 19.5, 21.3, 20.8, 22.0];
const fert2 = [22.4, 23.1, 21.8, 24.0, 23.5];
const fert3 = [18.2, 17.5, 19.0, 18.8, 17.9];
const anova = oneWayAnova([fert1, fert2, fert3]);
console.log("=== One-way ANOVA (fertilizer comparison) ===");
console.log(`  F-statistic: ${anova.fStatistic.toFixed(4)}`);
console.log(`  p-value: ${anova.pValue.toFixed(6)}`);
console.log(`  df between: ${anova.dfBetween}, df within: ${anova.dfWithin}`);
console.log(`  Reject H0 at alpha=0.05? ${anova.rejected}`);
