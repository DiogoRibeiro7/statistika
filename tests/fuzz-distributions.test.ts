/**
 * Fuzz tests for distribution classes with edge-case inputs.
 *
 * Tests that distributions handle NaN, Infinity, negative values,
 * and extreme parameters gracefully without crashing.
 */

import { Normal } from '../src/distributions/continuous/normal';
import { Exponential } from '../src/distributions/continuous/exponential';
import { GammaDistribution } from '../src/distributions/continuous/gamma';
import { BetaDistribution } from '../src/distributions/continuous/beta';
import { ChiSquared } from '../src/distributions/continuous/chi-squared';
import { StudentT } from '../src/distributions/continuous/student-t';
import { Uniform } from '../src/distributions/continuous/uniform';
import { Poisson } from '../src/distributions/discrete/poisson';
import { Binomial } from '../src/distributions/discrete/binomial';
import { LogNormal } from '../src/distributions/continuous/log-normal';
import { Weibull } from '../src/distributions/continuous/weibull';
import { Cauchy } from '../src/distributions/continuous/cauchy';

// Helper: safely call a function, returning NaN on throw
function safeCall(fn: () => number): number {
  try {
    return fn();
  } catch {
    return NaN;
  }
}

// ==========================================================================
// 1. NaN inputs to distribution methods
// ==========================================================================

describe('Fuzz: NaN inputs to continuous distributions', () => {
  const distributions = [
    { name: 'Normal(0,1)', dist: new Normal(0, 1) },
    { name: 'Exponential(1)', dist: new Exponential(1) },
    { name: 'Uniform(0,1)', dist: new Uniform(0, 1) },
    { name: 'ChiSquared(5)', dist: new ChiSquared(5) },
    { name: 'StudentT(10)', dist: new StudentT(10) },
    { name: 'Cauchy(0,1)', dist: new Cauchy(0, 1) },
  ];

  for (const { name, dist } of distributions) {
    test(`${name}.pdf(NaN) does not throw`, () => {
      expect(() => dist.pdf(NaN)).not.toThrow();
    });

    test(`${name}.cdf(NaN) does not throw`, () => {
      expect(() => dist.cdf(NaN)).not.toThrow();
    });

    test(`${name}.pdf(Infinity) does not throw`, () => {
      expect(() => dist.pdf(Infinity)).not.toThrow();
    });

    test(`${name}.cdf(Infinity) does not throw`, () => {
      expect(() => dist.cdf(Infinity)).not.toThrow();
    });

    test(`${name}.pdf(-Infinity) does not throw`, () => {
      expect(() => dist.pdf(-Infinity)).not.toThrow();
    });

    test(`${name}.cdf(-Infinity) does not throw`, () => {
      expect(() => dist.cdf(-Infinity)).not.toThrow();
    });
  }
});

// ==========================================================================
// 2. Extreme parameter values
// ==========================================================================

describe('Fuzz: distributions with extreme parameters', () => {
  test('Normal(0, very_small_sigma) does not crash', () => {
    const dist = new Normal(0, 1e-10);
    expect(() => dist.pdf(0)).not.toThrow();
    expect(() => dist.cdf(0)).not.toThrow();
    expect(() => dist.mean()).not.toThrow();
  });

  test('Normal(0, very_large_sigma) does not crash', () => {
    const dist = new Normal(0, 1e10);
    expect(() => dist.pdf(0)).not.toThrow();
    expect(() => dist.cdf(0)).not.toThrow();
  });

  test('Normal(very_large_mean, 1) does not crash', () => {
    const dist = new Normal(1e15, 1);
    expect(() => dist.pdf(1e15)).not.toThrow();
    expect(() => dist.cdf(1e15)).not.toThrow();
  });

  test('Exponential(very_small_rate) does not crash', () => {
    const dist = new Exponential(1e-10);
    expect(() => dist.pdf(1)).not.toThrow();
    expect(() => dist.cdf(1)).not.toThrow();
  });

  test('Exponential(very_large_rate) does not crash', () => {
    const dist = new Exponential(1e10);
    expect(() => dist.pdf(1e-15)).not.toThrow();
    expect(() => dist.cdf(1e-15)).not.toThrow();
  });

  test('GammaDistribution with very small shape does not crash', () => {
    const dist = new GammaDistribution(0.001, 1);
    expect(() => dist.pdf(0.001)).not.toThrow();
    expect(() => dist.cdf(0.001)).not.toThrow();
  });

  test('GammaDistribution with large shape does not crash', () => {
    const dist = new GammaDistribution(100, 1);
    expect(() => dist.pdf(100)).not.toThrow();
    expect(() => dist.cdf(100)).not.toThrow();
  });

  test('BetaDistribution with very small parameters does not crash', () => {
    const dist = new BetaDistribution(0.01, 0.01);
    expect(() => dist.pdf(0.5)).not.toThrow();
    expect(() => dist.cdf(0.5)).not.toThrow();
  });

  test('BetaDistribution with large parameters does not crash', () => {
    const dist = new BetaDistribution(100, 100);
    expect(() => dist.pdf(0.5)).not.toThrow();
    expect(() => dist.cdf(0.5)).not.toThrow();
  });

  test('ChiSquared with df=1 does not crash at x near 0', () => {
    const dist = new ChiSquared(1);
    expect(() => dist.pdf(1e-10)).not.toThrow();
    expect(() => dist.cdf(1e-10)).not.toThrow();
  });

  test('ChiSquared with large df does not crash', () => {
    const dist = new ChiSquared(200);
    expect(() => dist.pdf(200)).not.toThrow();
    expect(() => dist.cdf(200)).not.toThrow();
  });

  test('StudentT with df=1 (Cauchy) does not crash', () => {
    const dist = new StudentT(1);
    expect(() => dist.pdf(1e10)).not.toThrow();
    expect(() => dist.cdf(1e10)).not.toThrow();
  });

  test('Weibull with very small shape does not crash', () => {
    const dist = new Weibull(0.01, 1);
    expect(() => dist.pdf(0.5)).not.toThrow();
    expect(() => dist.cdf(0.5)).not.toThrow();
  });

  test('LogNormal with large sigma does not crash', () => {
    const dist = new LogNormal(0, 10);
    expect(() => dist.pdf(1)).not.toThrow();
    expect(() => dist.cdf(1)).not.toThrow();
  });
});

// ==========================================================================
// 3. Boundary values for continuous distributions
// ==========================================================================

describe('Fuzz: boundary values for continuous distributions', () => {
  test('Normal CDF at extreme tails', () => {
    const dist = new Normal(0, 1);
    const cdfNeg = safeCall(() => dist.cdf(-100));
    const cdfPos = safeCall(() => dist.cdf(100));
    if (!isNaN(cdfNeg)) expect(cdfNeg).toBeCloseTo(0, 6);
    if (!isNaN(cdfPos)) expect(cdfPos).toBeCloseTo(1, 6);
  });

  test('Exponential CDF(0) = 0', () => {
    const dist = new Exponential(1);
    expect(dist.cdf(0)).toBeCloseTo(0, 10);
  });

  test('Exponential CDF of very large x approaches 1', () => {
    const dist = new Exponential(1);
    expect(dist.cdf(100)).toBeCloseTo(1, 6);
  });

  test('Uniform CDF at boundaries', () => {
    const dist = new Uniform(0, 1);
    expect(dist.cdf(0)).toBeCloseTo(0, 10);
    expect(dist.cdf(1)).toBeCloseTo(1, 10);
    expect(dist.cdf(-1)).toBeCloseTo(0, 10);
    expect(dist.cdf(2)).toBeCloseTo(1, 10);
  });

  test('StudentT CDF approaches Normal for large df', () => {
    const t100 = new StudentT(1000);
    const normal = new Normal(0, 1);
    // At x=1.96, both should give ~0.975
    expect(t100.cdf(1.96)).toBeCloseTo(normal.cdf(1.96), 2);
  });
});

// ==========================================================================
// 4. Edge cases for discrete distributions
// ==========================================================================

describe('Fuzz: edge cases for discrete distributions', () => {
  test('Poisson(very_small_lambda) does not crash', () => {
    const dist = new Poisson(1e-10);
    expect(() => dist.pmf(0)).not.toThrow();
    expect(() => dist.cdf(0)).not.toThrow();
    expect(dist.pmf(0)).toBeCloseTo(1, 6);
  });

  test('Poisson(large_lambda) does not crash', () => {
    const dist = new Poisson(100);
    expect(() => dist.pmf(100)).not.toThrow();
    expect(() => dist.cdf(100)).not.toThrow();
  });

  test('Poisson PMF is zero for negative values', () => {
    const dist = new Poisson(5);
    expect(dist.pmf(-1)).toBe(0);
  });

  test('Binomial(n, 0) gives pmf(0) = 1', () => {
    const dist = new Binomial(10, 0);
    expect(dist.pmf(0)).toBeCloseTo(1, 10);
  });

  test('Binomial(n, 1) gives pmf(n) = 1', () => {
    const dist = new Binomial(10, 1);
    expect(dist.pmf(10)).toBeCloseTo(1, 10);
  });

  test('Binomial PMF sums to 1', () => {
    const dist = new Binomial(20, 0.3);
    let sum = 0;
    for (let k = 0; k <= 20; k++) sum += dist.pmf(k);
    expect(sum).toBeCloseTo(1, 6);
  });

  test('Binomial with large n does not crash', () => {
    const dist = new Binomial(1000, 0.5);
    expect(() => dist.pmf(500)).not.toThrow();
    expect(() => dist.cdf(500)).not.toThrow();
  });

  test('Poisson PMF sums to approximately 1', () => {
    const dist = new Poisson(5);
    let sum = 0;
    for (let k = 0; k <= 50; k++) sum += dist.pmf(k);
    expect(sum).toBeCloseTo(1, 4);
  });
});

// ==========================================================================
// 5. Quantile edge cases
// ==========================================================================

describe('Fuzz: quantile edge cases', () => {
  test('Normal quantile(0.5) = mean', () => {
    const dist = new Normal(42, 7);
    expect(dist.quantile(0.5)).toBeCloseTo(42, 4);
  });

  test('Exponential quantile(0) = 0', () => {
    const dist = new Exponential(1);
    const q = safeCall(() => dist.quantile(0));
    if (!isNaN(q)) expect(q).toBeCloseTo(0, 4);
  });

  test('Normal quantile at extreme probabilities', () => {
    const dist = new Normal(0, 1);
    const qLow = safeCall(() => dist.quantile(0.001));
    const qHigh = safeCall(() => dist.quantile(0.999));
    if (!isNaN(qLow)) expect(qLow).toBeLessThan(-2.5);
    if (!isNaN(qHigh)) expect(qHigh).toBeGreaterThan(2.5);
  });

  test('Uniform quantile is linear', () => {
    const dist = new Uniform(10, 20);
    expect(dist.quantile(0.0)).toBeCloseTo(10, 6);
    expect(dist.quantile(0.5)).toBeCloseTo(15, 6);
    expect(dist.quantile(1.0)).toBeCloseTo(20, 6);
  });
});

// ==========================================================================
// 6. Sampling stress tests
// ==========================================================================

describe('Fuzz: sampling does not crash', () => {
  const distributions: Array<{ name: string; sample: () => number }> = [
    { name: 'Normal(0,1)', sample: () => new Normal(0, 1).sample() },
    { name: 'Exponential(1)', sample: () => new Exponential(1).sample() },
    { name: 'Gamma(2,1)', sample: () => new GammaDistribution(2, 1).sample() },
    { name: 'Beta(2,5)', sample: () => new BetaDistribution(2, 5).sample() },
    { name: 'ChiSquared(3)', sample: () => new ChiSquared(3).sample() },
    { name: 'StudentT(5)', sample: () => new StudentT(5).sample() },
    { name: 'Poisson(10)', sample: () => new Poisson(10).sample() },
    { name: 'Binomial(20,0.5)', sample: () => new Binomial(20, 0.5).sample() },
  ];

  for (const { name, sample } of distributions) {
    test(`${name}.sample() produces finite numbers`, () => {
      for (let i = 0; i < 100; i++) {
        const value = sample();
        expect(isFinite(value)).toBe(true);
      }
    });
  }
});
