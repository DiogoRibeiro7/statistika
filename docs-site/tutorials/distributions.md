# Tutorial: Working with Distributions

node_stats provides 31 probability distributions -- 23 continuous and 8 discrete. Every distribution shares a common interface for computing moments, sampling, and probability calculations.

## Creating a Distribution

Each distribution is a class you instantiate with its parameters:

```typescript
import {
  Normal, Exponential, BetaDistribution, GammaDistribution,
  Poisson, Binomial, Uniform,
} from 'node_stats';

const normal = new Normal(0, 1);          // mean=0, sd=1
const expo = new Exponential(0.5);        // rate=0.5
const beta = new BetaDistribution(2, 5);  // alpha=2, beta=5
const gam = new GammaDistribution(3, 2);  // shape=3, rate=2
const pois = new Poisson(4.5);            // rate=4.5
const binom = new Binomial(20, 0.3);      // n=20, p=0.3
const unif = new Uniform(0, 10);          // a=0, b=10
```

## Common Interface

Every distribution provides these methods:

```typescript
const dist = new Normal(100, 15);

// Moments
dist.mean();     // 100
dist.variance(); // 225
dist.stdDev();   // 15

// Random sampling
dist.sample();      // single random draw
dist.sampleN(1000); // array of 1000 draws
```

Continuous distributions additionally provide:

```typescript
dist.pdf(100);       // probability density at x=100
dist.cdf(115);       // P(X <= 115)
dist.sf(115);        // P(X > 115) = 1 - cdf(115)
dist.quantile(0.95); // value where P(X <= x) = 0.95
```

Discrete distributions use `pmf` instead of `pdf`:

```typescript
const pois = new Poisson(4.5);
pois.pmf(3);       // P(X = 3)
pois.cdf(6);       // P(X <= 6)
pois.sf(6);        // P(X > 6)
pois.quantile(0.5); // median
```

## PDF and CDF Evaluation

The PDF (or PMF) gives the relative likelihood at a point. The CDF gives cumulative probability.

```typescript
const normal = new Normal(100, 15);

// What fraction of the population has IQ below 130?
console.log(normal.cdf(130));
// 0.9772 — about 97.7%

// What fraction has IQ above 85?
console.log(normal.sf(85));
// 0.8413 — about 84.1%

// What is the IQ threshold for the top 5%?
console.log(normal.quantile(0.95));
// 124.67
```

For the Poisson distribution, you can compute exact probabilities:

```typescript
const arrivals = new Poisson(4.5);

// Exactly 3 arrivals in an hour
console.log(arrivals.pmf(3));
// 0.1687

// At most 6 arrivals
console.log(arrivals.cdf(6));
// 0.8311

// More than 8 arrivals
console.log(arrivals.sf(8));
// 0.0317
```

## Random Sampling

All distributions support random sampling with `sample()` and `sampleN(n)`:

```typescript
const expo = new Exponential(0.5);

// Generate 10000 samples and verify the empirical mean
const samples = expo.sampleN(10000);
const empiricalMean = samples.reduce((a, b) => a + b, 0) / samples.length;
console.log(expo.mean());     // 2.0 (theoretical)
console.log(empiricalMean);   // ~2.0 (empirical, varies by run)
```

## Comparing Distributions

You can compare distributions by overlaying their CDFs or computing statistical tests:

```typescript
import { Normal, StudentT } from 'node_stats';

const normal = new Normal(0, 1);
const t10 = new StudentT(10);

// Compare tails — Student-t has heavier tails
const points = [-3, -2, -1, 0, 1, 2, 3];
for (const x of points) {
  console.log(
    `x=${x}: Normal CDF=${normal.cdf(x).toFixed(4)}, ` +
    `t(10) CDF=${t10.cdf(x).toFixed(4)}`
  );
}
// At x=-3: Normal CDF=0.0013, t(10) CDF=0.0067
// The t-distribution puts more probability in the tails.
```

## Working with the Beta Distribution

The Beta distribution is useful for modeling proportions and probabilities, such as conversion rates:

```typescript
import { BetaDistribution } from 'node_stats';

// Prior belief: conversion rate is around 10% (Beta(2, 18))
const prior = new BetaDistribution(2, 18);
console.log(prior.mean());          // 0.1
console.log(prior.quantile(0.025)); // lower 95% bound
console.log(prior.quantile(0.975)); // upper 95% bound

// After observing 15 conversions in 100 trials,
// the posterior is Beta(2 + 15, 18 + 85) = Beta(17, 103)
const posterior = new BetaDistribution(17, 103);
console.log(posterior.mean());          // 0.1417
console.log(posterior.quantile(0.025)); // tighter lower bound
console.log(posterior.quantile(0.975)); // tighter upper bound
```

## Extreme Value Distributions

node_stats includes GEV, Gumbel, Frechet, and GPD for modeling rare events:

```typescript
import { GEV, Gumbel, GPD } from 'node_stats';

// Gumbel distribution for annual maximum temperatures
const gumbel = new Gumbel(35, 3); // location=35, scale=3
console.log(gumbel.mean());          // 36.73
console.log(gumbel.quantile(0.99)); // 100-year return level

// GEV with shape parameter for heavier tail
const gev = new GEV(35, 3, 0.2);
console.log(gev.quantile(0.99));  // higher than Gumbel due to heavy tail

// GPD for exceedances above a threshold
const gpd = new GPD(0.1, 2); // shape=0.1, scale=2
console.log(gpd.sf(10));     // P(excess > 10)
```

## Full Distribution Reference

### Continuous Distributions

| Class | Constructor | Parameters |
|---|---|---|
| `Normal` | `new Normal(mu, sigma)` | mean, standard deviation |
| `Uniform` | `new Uniform(a, b)` | lower bound, upper bound |
| `Exponential` | `new Exponential(lambda)` | rate |
| `GammaDistribution` | `new GammaDistribution(alpha, beta)` | shape, rate |
| `BetaDistribution` | `new BetaDistribution(alpha, beta)` | shape, shape |
| `ChiSquared` | `new ChiSquared(df)` | degrees of freedom |
| `StudentT` | `new StudentT(df)` | degrees of freedom |
| `FDistribution` | `new FDistribution(d1, d2)` | numerator df, denominator df |
| `LogNormal` | `new LogNormal(mu, sigma)` | log-mean, log-std |
| `Weibull` | `new Weibull(k, lambda)` | shape, scale |
| `Pareto` | `new Pareto(xm, alpha)` | minimum value, shape |
| `Cauchy` | `new Cauchy(x0, gamma)` | location, scale |
| `GEV` | `new GEV(mu, sigma, xi)` | location, scale, shape |
| `Gumbel` | `new Gumbel(mu, beta)` | location, scale |
| `Frechet` | `new Frechet(alpha, s)` | shape, scale |
| `GPD` | `new GPD(xi, sigma)` | shape, scale |

### Discrete Distributions

| Class | Constructor | Parameters |
|---|---|---|
| `Bernoulli` | `new Bernoulli(p)` | probability of success |
| `Binomial` | `new Binomial(n, p)` | trials, probability |
| `Poisson` | `new Poisson(lambda)` | rate |
| `Geometric` | `new Geometric(p)` | probability of success |
| `DiscreteUniform` | `new DiscreteUniform(a, b)` | min, max |
| `NegativeBinomial` | `new NegativeBinomial(r, p)` | successes needed, probability |
| `Hypergeometric` | `new Hypergeometric(N, K, n)` | population, successes, draws |
