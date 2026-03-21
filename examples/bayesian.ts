/**
 * Example: Bayesian methods.
 *
 * Run: npx ts-node examples/bayesian.ts
 */

import {
  betaBinomial,
  normalNormal,
  gammaPoisson,
  metropolisHastings,
} from "../src";

// --- Beta-Binomial: A/B testing ---
console.log("=== A/B Test with Bayesian Inference ===");

// Variant A: 45 conversions out of 500 visitors
const variantA = betaBinomial(45, 500);
console.log(`  Variant A (45/500):`);
console.log(`    Posterior: Beta(${variantA.posteriorAlpha}, ${variantA.posteriorBeta})`);
console.log(`    Conversion rate: ${(variantA.posteriorMean * 100).toFixed(2)}%`);
console.log(`    95% CI: [${(variantA.credibleInterval[0] * 100).toFixed(2)}%, ${(variantA.credibleInterval[1] * 100).toFixed(2)}%]`);

// Variant B: 60 conversions out of 500 visitors
const variantB = betaBinomial(60, 500);
console.log(`  Variant B (60/500):`);
console.log(`    Posterior: Beta(${variantB.posteriorAlpha}, ${variantB.posteriorBeta})`);
console.log(`    Conversion rate: ${(variantB.posteriorMean * 100).toFixed(2)}%`);
console.log(`    95% CI: [${(variantB.credibleInterval[0] * 100).toFixed(2)}%, ${(variantB.credibleInterval[1] * 100).toFixed(2)}%]`);
console.log();

// --- Normal-Normal: estimating a mean ---
console.log("=== Normal-Normal: Estimating Mean Temperature ===");
// Prior: historical average is 20°C with high uncertainty
// Data: 10 measurements, known measurement variance = 4
const temps = [21.2, 19.8, 22.1, 20.5, 21.7, 20.3, 22.0, 19.5, 21.3, 20.8];
const result = normalNormal(temps, 4, 20, 100);
console.log(`  Prior: N(${result.priorMean}, ${result.priorVariance})`);
console.log(`  Posterior mean: ${result.posteriorMean.toFixed(4)}°C`);
console.log(`  Posterior variance: ${result.posteriorVariance.toFixed(4)}`);
console.log(`  95% CI: [${result.credibleInterval[0].toFixed(2)}, ${result.credibleInterval[1].toFixed(2)}]°C`);
console.log();

// --- Gamma-Poisson: event rate estimation ---
console.log("=== Gamma-Poisson: Estimating Bug Rate ===");
// Weekly bug counts over 8 weeks
const bugs = [3, 5, 2, 4, 6, 3, 4, 5];
const bugRate = gammaPoisson(bugs, 1, 0.1); // weakly informative prior
console.log(`  Data: [${bugs.join(", ")}] bugs/week`);
console.log(`  Posterior: Gamma(${bugRate.posteriorAlpha}, ${bugRate.posteriorBeta.toFixed(1)})`);
console.log(`  Estimated rate: ${bugRate.posteriorMean.toFixed(2)} bugs/week`);
console.log(`  95% CI: [${bugRate.credibleInterval[0].toFixed(2)}, ${bugRate.credibleInterval[1].toFixed(2)}]`);
console.log();

// --- MCMC: Metropolis-Hastings ---
console.log("=== MCMC: Sampling from a Mixture ===");

// Target: mixture of two normals N(-2, 0.5) and N(3, 1)
const logPosterior = (x: number) => {
  const comp1 = Math.exp(-((x + 2) ** 2) / 1);
  const comp2 = Math.exp(-((x - 3) ** 2) / 2);
  return Math.log(0.4 * comp1 + 0.6 * comp2);
};

let seed = 42;
const random = () => {
  seed = (seed * 1103515245 + 12345) & 0x7fffffff;
  return seed / 0x7fffffff;
};

const mcmc = metropolisHastings(logPosterior, {
  nSamples: 10000,
  burnIn: 2000,
  proposalStd: 2,
  initial: 0,
  random,
});

console.log(`  Samples drawn: ${mcmc.samples.length}`);
console.log(`  Acceptance rate: ${(mcmc.acceptanceRate * 100).toFixed(1)}%`);
console.log(`  Posterior mean: ${mcmc.mean.toFixed(4)}`);
console.log(`  Posterior std: ${mcmc.std.toFixed(4)}`);
console.log(`  95% CI: [${mcmc.credibleInterval[0].toFixed(2)}, ${mcmc.credibleInterval[1].toFixed(2)}]`);

// Histogram of samples
const bins = 10;
const min = Math.min(...mcmc.samples);
const max = Math.max(...mcmc.samples);
const binWidth = (max - min) / bins;
const counts = new Array(bins).fill(0);
for (const s of mcmc.samples) {
  const idx = Math.min(Math.floor((s - min) / binWidth), bins - 1);
  counts[idx]++;
}
const maxCount = Math.max(...counts);
console.log("\n  Posterior histogram:");
for (let i = 0; i < bins; i++) {
  const lo = (min + i * binWidth).toFixed(1);
  const hi = (min + (i + 1) * binWidth).toFixed(1);
  const bar = "#".repeat(Math.round((counts[i] / maxCount) * 30));
  console.log(`    [${lo.padStart(6)}, ${hi.padStart(6)}] ${bar}`);
}
