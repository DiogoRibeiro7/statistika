/**
 * Example: Regression models.
 *
 * Run: npx ts-node examples/regression.ts
 */

import {
  linearRegression,
  multipleRegression,
  polynomialRegression,
  logisticRegression,
  huberRegression,
  ransacRegression,
} from "../src";

// --- Simple linear regression ---
const height = [150, 160, 165, 170, 175, 180, 185, 190];
const weight = [50, 60, 62, 68, 72, 78, 82, 90];

const lr = linearRegression(height, weight);
console.log("=== Linear Regression (height -> weight) ===");
console.log(`  Slope: ${lr.slope.toFixed(4)}`);
console.log(`  Intercept: ${lr.intercept.toFixed(4)}`);
console.log(`  R-squared: ${lr.rSquared.toFixed(4)}`);
console.log(`  Predicted weight at 172cm: ${lr.predict(172).toFixed(1)} kg`);
console.log();

// --- Multiple regression ---
// Predict salary from years of experience and education level
const features = [
  [2, 12],
  [4, 14],
  [6, 16],
  [8, 12],
  [10, 14],
  [12, 16],
  [14, 18],
  [16, 14],
];
const salary = [35, 45, 55, 50, 60, 70, 85, 75];

const mr = multipleRegression(features, salary);
console.log("=== Multiple Regression (experience + education -> salary) ===");
console.log(`  Coefficients: [${mr.coefficients.map((c) => c.toFixed(2))}]`);
console.log(`  Intercept: ${mr.intercept.toFixed(2)}`);
console.log(`  R-squared: ${mr.rSquared.toFixed(4)}`);
console.log(`  Predict (5 yrs, 16 edu): ${mr.predict([5, 16]).toFixed(1)}k`);
console.log();

// --- Polynomial regression ---
// Fit a quadratic to acceleration data
const time = [0, 1, 2, 3, 4, 5, 6, 7, 8];
const distance = [0, 4.9, 19.6, 44.1, 78.4, 122.5, 176.4, 240.1, 313.6];

const poly = polynomialRegression(time, distance, 2);
console.log("=== Polynomial Regression (free-fall distance, degree=2) ===");
console.log(`  Coefficients: [${poly.coefficients.map((c) => c.toFixed(2))}]`);
console.log(`  R-squared: ${poly.rSquared.toFixed(6)}`);
console.log(`  Distance at t=10: ${poly.predict(10).toFixed(1)} m`);
console.log();

// --- Logistic regression ---
// Predict pass/fail from study hours
const hours = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]];
const pass = [0, 0, 0, 0, 1, 0, 1, 1, 1, 1];

const logit = logisticRegression(hours, pass);
console.log("=== Logistic Regression (study hours -> pass/fail) ===");
console.log(`  Coefficient: ${logit.coefficients[0].toFixed(4)}`);
console.log(`  Intercept: ${logit.intercept.toFixed(4)}`);
console.log(`  P(pass | 5 hours): ${logit.predict([5]).toFixed(4)}`);
console.log(`  P(pass | 8 hours): ${logit.predict([8]).toFixed(4)}`);
console.log();

// --- Huber robust regression ---
// y = 2x + 1, with outliers
const x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
const y = x.map((xi) => 2 * xi + 1);
y[4] = 200; // outlier
y[11] = -100; // outlier

const huber = huberRegression(x, y);
console.log("=== Huber Robust Regression (with outliers) ===");
console.log(`  Slope: ${huber.slopes[0].toFixed(4)} (true: 2.0)`);
console.log(`  Intercept: ${huber.intercept.toFixed(4)} (true: 1.0)`);
console.log(`  Scale: ${huber.scale.toFixed(4)}`);
console.log(`  Iterations: ${huber.iterations}`);
console.log();

// --- RANSAC regression ---
let seed = 42;
const random = () => {
  seed = (seed * 1103515245 + 12345) & 0x7fffffff;
  return seed / 0x7fffffff;
};

const ransac = ransacRegression(x, y, undefined, 200, undefined, random);
console.log("=== RANSAC Regression (with outliers) ===");
console.log(`  Slope: ${ransac.slopes[0].toFixed(4)} (true: 2.0)`);
console.log(`  Intercept: ${ransac.intercept.toFixed(4)} (true: 1.0)`);
console.log(`  Inliers: ${ransac.nInliers} / ${x.length}`);
console.log(`  Outlier indices: ${x.filter((_, i) => !ransac.inlierIndices.includes(i)).length} detected`);
