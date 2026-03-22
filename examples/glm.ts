/**
 * Example: Generalized Linear Models.
 *
 * Run: npx ts-node examples/glm.ts
 */
import { glm, poisson, gaussian, binomial } from "../src";

console.log("=== Gaussian GLM (Linear Regression) ===");
const X1 = [[1], [2], [3], [4], [5]];
const y1 = [2.1, 3.9, 6.2, 7.8, 10.1];
const gfit = glm(X1, y1, gaussian);
console.log(`  Coefficients: [${gfit.coefficients.map(c => c.toFixed(4)).join(", ")}]`);
console.log(`  Deviance: ${gfit.deviance.toFixed(4)}`);
console.log();

console.log("=== Poisson GLM (Count Regression) ===");
const X2 = [[1], [2], [3], [4], [5], [6], [7], [8]];
const counts = [1, 2, 3, 5, 8, 13, 21, 34];
const pfit = glm(X2, counts, poisson);
console.log(`  Coefficients: [${pfit.coefficients.map(c => c.toFixed(4)).join(", ")}]`);
console.log(`  Deviance: ${pfit.deviance.toFixed(4)}`);
console.log(`  AIC: ${pfit.aic.toFixed(4)}`);
console.log();

console.log("=== Binomial GLM (Logistic via GLM) ===");
const X3 = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]];
const y3 = [0, 0, 0, 0, 1, 0, 1, 1, 1, 1];
const bfit = glm(X3, y3, binomial);
console.log(`  Coefficients: [${bfit.coefficients.map(c => c.toFixed(4)).join(", ")}]`);
console.log(`  Deviance: ${bfit.deviance.toFixed(4)}`);
