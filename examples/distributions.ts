/**
 * Example: Working with probability distributions.
 *
 * Run: npx ts-node examples/distributions.ts
 */

import {
  Normal,
  BetaDistribution,
  Poisson,
  Binomial,
  Exponential,
  StudentT,
} from "../src";

// --- Normal distribution ---
const normal = new Normal(100, 15); // IQ scores: mean=100, sd=15
console.log("=== Normal Distribution (IQ scores) ===");
console.log(`  Mean: ${normal.mean()}`);
console.log(`  Std Dev: ${normal.stdDev()}`);
console.log(`  P(IQ < 130): ${normal.cdf(130).toFixed(4)}`);
console.log(`  P(IQ > 85): ${normal.sf(85).toFixed(4)}`);
console.log(`  Top 5% threshold: ${normal.quantile(0.95).toFixed(1)}`);
console.log(`  Random sample: ${normal.sampleN(5).map((x) => x.toFixed(1))}`);
console.log();

// --- Beta distribution ---
const beta = new BetaDistribution(2, 5);
console.log("=== Beta Distribution (conversion rate prior) ===");
console.log(`  Mean: ${beta.mean().toFixed(4)}`);
console.log(`  Variance: ${beta.variance().toFixed(4)}`);
console.log(`  P(rate < 0.5): ${beta.cdf(0.5).toFixed(4)}`);
console.log(`  Median: ${beta.quantile(0.5).toFixed(4)}`);
console.log();

// --- Poisson distribution ---
const poisson = new Poisson(4.5); // avg 4.5 events per hour
console.log("=== Poisson Distribution (events per hour, lambda=4.5) ===");
console.log(`  Mean: ${poisson.mean()}`);
console.log(`  Variance: ${poisson.variance()}`);
console.log(`  P(X = 3): ${poisson.pmf(3).toFixed(4)}`);
console.log(`  P(X <= 6): ${poisson.cdf(6).toFixed(4)}`);
console.log(`  P(X > 8): ${poisson.sf(8).toFixed(4)}`);
console.log();

// --- Binomial distribution ---
const binom = new Binomial(20, 0.3);
console.log("=== Binomial Distribution (20 trials, p=0.3) ===");
console.log(`  Mean: ${binom.mean()}`);
console.log(`  P(X = 6): ${binom.pmf(6).toFixed(4)}`);
console.log(`  P(X <= 8): ${binom.cdf(8).toFixed(4)}`);
console.log(`  Median: ${binom.quantile(0.5)}`);
console.log();

// --- Exponential distribution ---
const expo = new Exponential(0.5); // mean time to failure = 2 hours
console.log("=== Exponential Distribution (failure rate=0.5/hr) ===");
console.log(`  Mean time to failure: ${expo.mean()} hours`);
console.log(`  P(survive > 3 hrs): ${expo.sf(3).toFixed(4)}`);
console.log(`  Median lifetime: ${expo.quantile(0.5).toFixed(2)} hours`);
console.log();

// --- Student-t distribution ---
const t = new StudentT(10);
console.log("=== Student-t Distribution (10 df) ===");
console.log(`  Mean: ${t.mean()}`);
console.log(`  Variance: ${t.variance().toFixed(4)}`);
console.log(`  95% critical value: ${t.quantile(0.975).toFixed(4)}`);
console.log(`  P(|T| > 2): ${(2 * t.sf(2)).toFixed(4)}`);
