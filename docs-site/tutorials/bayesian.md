# Tutorial: Bayesian Inference

This tutorial covers the Bayesian inference workflow in node_stats: specifying priors, computing posteriors with conjugate models, sampling from arbitrary posteriors with MCMC, and assessing convergence.

## Conjugate Models

Conjugate models provide exact, closed-form posterior distributions. No sampling is needed.

### Beta-Binomial: A/B Testing

The Beta-Binomial model is ideal for estimating conversion rates, click-through rates, or any binary outcome.

```typescript
import { betaBinomial } from 'node_stats';

// Variant A: 45 conversions out of 500 visitors
const variantA = betaBinomial(45, 500);
console.log(variantA.posteriorAlpha);  // 46 (prior alpha=1 + 45 successes)
console.log(variantA.posteriorBeta);   // 456 (prior beta=1 + 455 failures)
console.log(variantA.posteriorMean);   // 0.0908
console.log(variantA.credibleInterval);
// [0.068, 0.117] — 95% credible interval

// Variant B: 60 conversions out of 500 visitors
const variantB = betaBinomial(60, 500);
console.log(variantB.posteriorMean);   // 0.1178
console.log(variantB.credibleInterval);
// [0.092, 0.147]
```

The default prior is Beta(1, 1), which is uniform on [0, 1]. You can specify a different prior by passing prior alpha and beta values.

### Normal-Normal: Estimating a Mean

When the data variance is known, the Normal-Normal conjugate model updates your belief about the population mean:

```typescript
import { normalNormal } from 'node_stats';

// Prior: historical average temperature is 20C with high uncertainty
// Data: 10 temperature measurements; known measurement variance = 4
const temps = [21.2, 19.8, 22.1, 20.5, 21.7, 20.3, 22.0, 19.5, 21.3, 20.8];
const result = normalNormal(temps, 4, 20, 100);

console.log(result.priorMean);         // 20
console.log(result.priorVariance);     // 100
console.log(result.posteriorMean);     // ~20.92 (data-dominated)
console.log(result.posteriorVariance); // much smaller than prior
console.log(result.credibleInterval);  // tight 95% interval
```

Because the prior variance (100) is large relative to the data precision, the posterior is pulled strongly toward the data mean.

### Gamma-Poisson: Rate Estimation

For count data where you want to estimate an event rate:

```typescript
import { gammaPoisson } from 'node_stats';

// Weekly bug counts over 8 weeks
const bugs = [3, 5, 2, 4, 6, 3, 4, 5];
const result = gammaPoisson(bugs, 1, 0.1); // weakly informative prior

console.log(result.posteriorAlpha); // 1 + sum(bugs) = 33
console.log(result.posteriorBeta);  // 0.1 + 8 = 8.1
console.log(result.posteriorMean);  // ~4.07 bugs/week
console.log(result.credibleInterval);
// 95% credible interval for the bug rate
```

## MCMC Sampling

When you have a custom posterior that does not have a conjugate form, use Metropolis-Hastings MCMC.

### Single-Dimensional MCMC

```typescript
import { metropolisHastings } from 'node_stats';

// Target: mixture of two normals
const logPosterior = (x: number) => {
  const comp1 = Math.exp(-((x + 2) ** 2) / 1);
  const comp2 = Math.exp(-((x - 3) ** 2) / 2);
  return Math.log(0.4 * comp1 + 0.6 * comp2);
};

const mcmc = metropolisHastings(logPosterior, {
  nSamples: 10000,
  burnIn: 2000,
  proposalStd: 2,
  initial: 0,
});

console.log(mcmc.samples.length); // 10000
console.log(mcmc.mean);           // posterior mean
console.log(mcmc.std);            // posterior standard deviation
console.log(mcmc.acceptanceRate); // aim for 0.2-0.5
console.log(mcmc.credibleInterval);
// 95% credible interval from the samples
```

### Multi-Dimensional MCMC

For posteriors with multiple parameters:

```typescript
import { metropolisHastingsND } from 'node_stats';

// 2D target: bivariate normal with correlation
const logDensity = (x: number[]) => {
  const rho = 0.7;
  const z = x[0] ** 2 - 2 * rho * x[0] * x[1] + x[1] ** 2;
  return -z / (2 * (1 - rho ** 2));
};

const result = metropolisHastingsND(logDensity, 2, {
  nSamples: 50000,
  burnIn: 5000,
});

console.log(result.means);          // [~0, ~0]
console.log(result.acceptanceRate); // acceptance rate
```

## Convergence Diagnostics

After running MCMC, you should check that the chains have converged.

### Gelman-Rubin Diagnostic

Run multiple chains from different starting points and compute the R-hat statistic:

```typescript
import { metropolisHastings, gelmanRubin } from 'node_stats';

const logPost = (x: number) => -0.5 * x * x;

// Run 4 chains from different starting points
const chains = [-5, 0, 3, 7].map(init =>
  metropolisHastings(logPost, {
    nSamples: 5000, burnIn: 1000, proposalStd: 1, initial: init,
  }).samples
);

const rhat = gelmanRubin(chains);
console.log(rhat);
// Should be < 1.1 for convergence.
// Values above 1.2 suggest the chains have not mixed well.
```

### Effective Sample Size

Autocorrelation in MCMC chains means consecutive samples are not independent. The effective sample size (ESS) estimates the equivalent number of independent samples:

```typescript
import { estimateESS } from 'node_stats';

const ess = estimateESS(mcmc.samples);
console.log(ess);
// If ESS is much smaller than the number of samples,
// your chain has high autocorrelation. Try:
// - Increasing proposalStd
// - Running longer chains
// - Thinning the chain
```

## Bayes Factors

Compare two hypotheses by computing a Bayes factor:

```typescript
import { bayesFactor } from 'node_stats';

// Compare marginal likelihoods of two models
const bf = bayesFactor(logMarginalH1, logMarginalH0);
console.log(bf);
// bf > 3: substantial evidence for H1
// bf > 10: strong evidence for H1
// bf > 100: decisive evidence for H1
```

## Complete Workflow Example

Putting it all together -- a Bayesian analysis of customer churn rate:

```typescript
import { betaBinomial, metropolisHastings } from 'node_stats';

// Step 1: Quick conjugate analysis
// 23 churns out of 200 customers this quarter
const conjugate = betaBinomial(23, 200);
console.log('Conjugate posterior mean:', conjugate.posteriorMean);
console.log('95% CI:', conjugate.credibleInterval);

// Step 2: MCMC for a more flexible model
// Log-posterior with an informative prior: Beta(2, 10)
const logPost = (p: number) => {
  if (p <= 0 || p >= 1) return -Infinity;
  // Beta(2, 10) prior
  const logPrior = Math.log(p) + 9 * Math.log(1 - p);
  // Binomial likelihood: 23 successes in 200 trials
  const logLik = 23 * Math.log(p) + 177 * Math.log(1 - p);
  return logPrior + logLik;
};

const mcmc = metropolisHastings(logPost, {
  nSamples: 20000,
  burnIn: 5000,
  proposalStd: 0.02,
  initial: 0.1,
});

console.log('MCMC posterior mean:', mcmc.mean);
console.log('MCMC 95% CI:', mcmc.credibleInterval);
console.log('Acceptance rate:', mcmc.acceptanceRate);

// Step 3: Assess convergence
// The acceptance rate should be between 0.2 and 0.5.
// For a single chain, examine the trace for stationarity.
```
