/**
 * Property-based tests for ALL distributions in statistika.
 *
 * Tests verify universal statistical invariants:
 *  1. CDF-Quantile roundtrip
 *  2. CDF monotonicity
 *  3. PDF/PMF non-negativity
 *  4. CDF bounds [0,1]
 *  5. SF complement: SF(x) = 1 - CDF(x)
 *  6. Sample mean convergence
 *  7. Sample variance convergence
 *  8. Multivariate structural properties
 */

// --- Continuous distributions ---
import { Normal } from '../src/distributions/continuous/normal';
import { Uniform } from '../src/distributions/continuous/uniform';
import { Exponential } from '../src/distributions/continuous/exponential';
import { GammaDistribution } from '../src/distributions/continuous/gamma';
import { BetaDistribution } from '../src/distributions/continuous/beta';
import { ChiSquared } from '../src/distributions/continuous/chi-squared';
import { StudentT } from '../src/distributions/continuous/student-t';
import { FDistribution } from '../src/distributions/continuous/f-distribution';
import { LogNormal } from '../src/distributions/continuous/log-normal';
import { Weibull } from '../src/distributions/continuous/weibull';
import { GEV } from '../src/distributions/continuous/gev';
import { Gumbel } from '../src/distributions/continuous/gumbel';
import { Frechet } from '../src/distributions/continuous/frechet';
import { GPD } from '../src/distributions/continuous/gpd';
import { Cauchy } from '../src/distributions/continuous/cauchy';
import { Pareto } from '../src/distributions/continuous/pareto';
import { Laplace } from '../src/distributions/continuous/laplace';
import { InverseGamma } from '../src/distributions/continuous/inverse-gamma';
import { LogLogistic } from '../src/distributions/continuous/log-logistic';
import { TruncatedNormal } from '../src/distributions/continuous/truncated-normal';
import { Rayleigh } from '../src/distributions/continuous/rayleigh';
import { VonMises } from '../src/distributions/continuous/von-mises';
import { Levy } from '../src/distributions/continuous/levy';

// --- Discrete distributions ---
import { Bernoulli } from '../src/distributions/discrete/bernoulli';
import { Binomial } from '../src/distributions/discrete/binomial';
import { Poisson } from '../src/distributions/discrete/poisson';
import { Geometric } from '../src/distributions/discrete/geometric';
import { DiscreteUniform } from '../src/distributions/discrete/discrete-uniform';
import { NegativeBinomial } from '../src/distributions/discrete/negative-binomial';
import { Hypergeometric } from '../src/distributions/discrete/hypergeometric';
import { ZeroInflatedPoisson } from '../src/distributions/discrete/zero-inflated-poisson';

// --- Multivariate distributions ---
import { MultivariateNormal } from '../src/distributions/multivariate/multivariate-normal';
import { MultivariateT } from '../src/distributions/multivariate/multivariate-t';
import { Dirichlet } from '../src/distributions/multivariate/dirichlet';
import { Multinomial } from '../src/distributions/multivariate/multinomial';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Seedable LCG PRNG for reproducible tests. */
function seededRng(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) & 0x7fffffff;
    return s / 0x7fffffff;
  };
}

const QUANTILE_PROBS = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99];
const N_SAMPLES = 10000;
const ROUNDTRIP_TOL = 0.02;
const MONOTONICITY_POINTS = 20;

/** Generate evenly-spaced test points spanning most of the distribution range. */
function testPoints(dist: { quantile(p: number): number }, n = MONOTONICITY_POINTS): number[] {
  const pts: number[] = [];
  for (let i = 1; i <= n; i++) {
    pts.push(dist.quantile(i / (n + 1)));
  }
  return pts;
}

function sampleMean(values: number[]): number {
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function sampleVariance(values: number[]): number {
  const m = sampleMean(values);
  return values.reduce((s, x) => s + (x - m) ** 2, 0) / (values.length - 1);
}

// ---------------------------------------------------------------------------
// Distribution registry
// ---------------------------------------------------------------------------

interface ContinuousEntry {
  name: string;
  factory: () => { mean(): number; variance(): number; pdf(x: number): number; cdf(x: number): number; quantile(p: number): number; sf(x: number): number; sample(): number; sampleN(n: number): number[] };
  finiteMean: boolean;
  finiteVariance: boolean;
  skipRoundtrip?: boolean; // e.g., VonMises (circular)
}

const rng = seededRng(42);

const continuousDistributions: ContinuousEntry[] = [
  { name: 'Normal(0,1)', factory: () => new Normal(0, 1, rng), finiteMean: true, finiteVariance: true },
  { name: 'Normal(5,2)', factory: () => new Normal(5, 2, rng), finiteMean: true, finiteVariance: true },
  { name: 'Uniform(0,1)', factory: () => new Uniform(0, 1, rng), finiteMean: true, finiteVariance: true },
  { name: 'Uniform(-3,7)', factory: () => new Uniform(-3, 7, rng), finiteMean: true, finiteVariance: true },
  { name: 'Exponential(1)', factory: () => new Exponential(1, rng), finiteMean: true, finiteVariance: true },
  { name: 'Exponential(0.5)', factory: () => new Exponential(0.5, rng), finiteMean: true, finiteVariance: true },
  { name: 'Gamma(2,1)', factory: () => new GammaDistribution(2, 1, rng), finiteMean: true, finiteVariance: true },
  { name: 'Gamma(5,2)', factory: () => new GammaDistribution(5, 2, rng), finiteMean: true, finiteVariance: true },
  { name: 'Beta(2,5)', factory: () => new BetaDistribution(2, 5, rng), finiteMean: true, finiteVariance: true },
  { name: 'Beta(0.5,0.5)', factory: () => new BetaDistribution(0.5, 0.5, rng), finiteMean: true, finiteVariance: true },
  { name: 'ChiSquared(3)', factory: () => new ChiSquared(3, rng), finiteMean: true, finiteVariance: true },
  { name: 'ChiSquared(10)', factory: () => new ChiSquared(10, rng), finiteMean: true, finiteVariance: true },
  { name: 'StudentT(5)', factory: () => new StudentT(5, rng), finiteMean: true, finiteVariance: true },
  { name: 'StudentT(30)', factory: () => new StudentT(30, rng), finiteMean: true, finiteVariance: true },
  { name: 'StudentT(1) [Cauchy-like]', factory: () => new StudentT(1, rng), finiteMean: false, finiteVariance: false },
  { name: 'FDistribution(5,10)', factory: () => new FDistribution(5, 10, rng), finiteMean: true, finiteVariance: true },
  { name: 'LogNormal(0,1)', factory: () => new LogNormal(0, 1, rng), finiteMean: true, finiteVariance: true },
  { name: 'LogNormal(1,0.5)', factory: () => new LogNormal(1, 0.5, rng), finiteMean: true, finiteVariance: true },
  { name: 'Weibull(2,1)', factory: () => new Weibull(2, 1, rng), finiteMean: true, finiteVariance: true },
  { name: 'Weibull(1,2)', factory: () => new Weibull(1, 2, rng), finiteMean: true, finiteVariance: true },
  { name: 'GEV(0,1,0)', factory: () => new GEV(0, 1, 0, rng), finiteMean: true, finiteVariance: true },
  { name: 'GEV(0,1,0.3)', factory: () => new GEV(0, 1, 0.3, rng), finiteMean: true, finiteVariance: true },
  { name: 'Gumbel(0,1)', factory: () => new Gumbel(0, 1, rng), finiteMean: true, finiteVariance: true },
  { name: 'Frechet(3,1,0)', factory: () => new Frechet(3, 1, 0, rng), finiteMean: true, finiteVariance: false },
  { name: 'Frechet(1.5,1,0)', factory: () => new Frechet(1.5, 1, 0, rng), finiteMean: true, finiteVariance: false },
  { name: 'GPD(0,1,0)', factory: () => new GPD(0, 1, 0, rng), finiteMean: true, finiteVariance: true },
  { name: 'GPD(0,1,0.3)', factory: () => new GPD(0, 1, 0.3, rng), finiteMean: true, finiteVariance: true },
  { name: 'Cauchy(0,1)', factory: () => new Cauchy(0, 1, rng), finiteMean: false, finiteVariance: false },
  { name: 'Pareto(3,1)', factory: () => new Pareto(3, 1, rng), finiteMean: true, finiteVariance: false },
  { name: 'Pareto(1.5,1)', factory: () => new Pareto(1.5, 1, rng), finiteMean: true, finiteVariance: false },
  { name: 'Laplace(0,1)', factory: () => new Laplace(0, 1, rng), finiteMean: true, finiteVariance: true },
  { name: 'InverseGamma(3,1)', factory: () => new InverseGamma(3, 1, rng), finiteMean: true, finiteVariance: true },
  { name: 'LogLogistic(1,4)', factory: () => new LogLogistic(1, 4, rng), finiteMean: true, finiteVariance: false },
  { name: 'LogLogistic(1,1.5)', factory: () => new LogLogistic(1, 1.5, rng), finiteMean: true, finiteVariance: false },
  { name: 'TruncatedNormal(0,1,-2,2)', factory: () => new TruncatedNormal(0, 1, -2, 2, rng), finiteMean: true, finiteVariance: true },
  { name: 'Rayleigh(1)', factory: () => new Rayleigh(1, rng), finiteMean: true, finiteVariance: true },
  { name: 'Rayleigh(3)', factory: () => new Rayleigh(3, rng), finiteMean: true, finiteVariance: true },
  { name: 'VonMises(0,2)', factory: () => new VonMises(0, 2, rng), finiteMean: true, finiteVariance: false, skipRoundtrip: true },
  { name: 'Levy(0,1)', factory: () => new Levy(0, 1, rng), finiteMean: false, finiteVariance: false },
];

interface DiscreteEntry {
  name: string;
  factory: () => { mean(): number; variance(): number; pmf(k: number): number; cdf(k: number): number; quantile(p: number): number; sf(k: number): number; sample(): number; sampleN(n: number): number[] };
  finiteMean: boolean;
  finiteVariance: boolean;
  testRange: number[]; // integer values to test PDF/CDF on
}

const discreteDistributions: DiscreteEntry[] = [
  { name: 'Bernoulli(0.3)', factory: () => new Bernoulli(0.3, rng), finiteMean: true, finiteVariance: true, testRange: [0, 1] },
  { name: 'Bernoulli(0.7)', factory: () => new Bernoulli(0.7, rng), finiteMean: true, finiteVariance: true, testRange: [0, 1] },
  { name: 'Binomial(20,0.4)', factory: () => new Binomial(20, 0.4, rng), finiteMean: true, finiteVariance: true, testRange: Array.from({ length: 21 }, (_, i) => i) },
  { name: 'Binomial(10,0.5)', factory: () => new Binomial(10, 0.5, rng), finiteMean: true, finiteVariance: true, testRange: Array.from({ length: 11 }, (_, i) => i) },
  { name: 'Poisson(3)', factory: () => new Poisson(3, rng), finiteMean: true, finiteVariance: true, testRange: Array.from({ length: 15 }, (_, i) => i) },
  { name: 'Poisson(10)', factory: () => new Poisson(10, rng), finiteMean: true, finiteVariance: true, testRange: Array.from({ length: 25 }, (_, i) => i) },
  { name: 'Geometric(0.3)', factory: () => new Geometric(0.3, rng), finiteMean: true, finiteVariance: true, testRange: Array.from({ length: 20 }, (_, i) => i) },
  { name: 'DiscreteUniform(1,6)', factory: () => new DiscreteUniform(1, 6, rng), finiteMean: true, finiteVariance: true, testRange: [1, 2, 3, 4, 5, 6] },
  { name: 'NegativeBinomial(5,0.4)', factory: () => new NegativeBinomial(5, 0.4, rng), finiteMean: true, finiteVariance: true, testRange: Array.from({ length: 30 }, (_, i) => i) },
  { name: 'Hypergeometric(50,10,15)', factory: () => new Hypergeometric(50, 10, 15, rng), finiteMean: true, finiteVariance: true, testRange: Array.from({ length: 11 }, (_, i) => i) },
  { name: 'ZeroInflatedPoisson(3,0.2)', factory: () => new ZeroInflatedPoisson(3, 0.2, rng), finiteMean: true, finiteVariance: true, testRange: Array.from({ length: 15 }, (_, i) => i) },
];

// ===========================================================================
// CONTINUOUS DISTRIBUTION TESTS
// ===========================================================================

describe('Property-based tests: Continuous distributions', () => {
  // ---- 1. CDF-Quantile roundtrip ----
  describe('CDF-Quantile roundtrip: CDF(quantile(p)) ≈ p', () => {
    for (const entry of continuousDistributions) {
      if (entry.skipRoundtrip) continue;
      it(entry.name, () => {
        const dist = entry.factory();
        for (const p of QUANTILE_PROBS) {
          const x = dist.quantile(p);
          const recovered = dist.cdf(x);
          expect(recovered).toBeCloseTo(p, 1); // within 0.05
          expect(Math.abs(recovered - p)).toBeLessThan(ROUNDTRIP_TOL);
        }
      });
    }
  });

  // ---- 2. CDF monotonicity ----
  describe('CDF monotonicity: x1 < x2 => CDF(x1) <= CDF(x2)', () => {
    for (const entry of continuousDistributions) {
      it(entry.name, () => {
        const dist = entry.factory();
        const pts = testPoints(dist);
        for (let i = 1; i < pts.length; i++) {
          if (pts[i] > pts[i - 1]) {
            expect(dist.cdf(pts[i])).toBeGreaterThanOrEqual(dist.cdf(pts[i - 1]));
          }
        }
      });
    }
  });

  // ---- 3. PDF non-negativity ----
  describe('PDF non-negativity: pdf(x) >= 0', () => {
    for (const entry of continuousDistributions) {
      it(entry.name, () => {
        const dist = entry.factory();
        const pts = testPoints(dist);
        for (const x of pts) {
          expect(dist.pdf(x)).toBeGreaterThanOrEqual(0);
        }
      });
    }
  });

  // ---- 4. CDF bounds ----
  describe('CDF bounds: 0 <= CDF(x) <= 1', () => {
    for (const entry of continuousDistributions) {
      it(entry.name, () => {
        const dist = entry.factory();
        const pts = testPoints(dist);
        for (const x of pts) {
          const c = dist.cdf(x);
          expect(c).toBeGreaterThanOrEqual(-1e-12);
          expect(c).toBeLessThanOrEqual(1 + 1e-12);
        }
      });
    }
  });

  // ---- 5. SF complement ----
  describe('SF complement: SF(x) ≈ 1 - CDF(x)', () => {
    for (const entry of continuousDistributions) {
      it(entry.name, () => {
        const dist = entry.factory();
        const pts = testPoints(dist);
        for (const x of pts) {
          expect(dist.sf(x)).toBeCloseTo(1 - dist.cdf(x), 10);
        }
      });
    }
  });

  // ---- 6. Sample mean convergence ----
  describe('Sample mean convergence', () => {
    for (const entry of continuousDistributions) {
      if (!entry.finiteMean) continue;
      it(entry.name, () => {
        const dist = entry.factory();
        const samples = dist.sampleN(N_SAMPLES);
        const sMean = sampleMean(samples);
        const thMean = dist.mean();
        if (Math.abs(thMean) < 0.01) {
          // For near-zero means, use absolute tolerance
          expect(Math.abs(sMean - thMean)).toBeLessThan(0.3);
        } else {
          // Relative error within 20%
          expect(Math.abs((sMean - thMean) / thMean)).toBeLessThan(0.2);
        }
      });
    }
  });

  // ---- 7. Sample variance convergence ----
  describe('Sample variance convergence', () => {
    for (const entry of continuousDistributions) {
      if (!entry.finiteVariance) continue;
      it(entry.name, () => {
        const dist = entry.factory();
        const samples = dist.sampleN(N_SAMPLES);
        const sVar = sampleVariance(samples);
        const thVar = dist.variance();
        if (thVar < 0.01) {
          expect(Math.abs(sVar - thVar)).toBeLessThan(0.1);
        } else {
          expect(Math.abs((sVar - thVar) / thVar)).toBeLessThan(0.2);
        }
      });
    }
  });
});

// ===========================================================================
// DISCRETE DISTRIBUTION TESTS
// ===========================================================================

describe('Property-based tests: Discrete distributions', () => {
  // ---- 1. CDF-Quantile roundtrip ----
  describe('CDF-Quantile roundtrip: CDF(quantile(p)) >= p', () => {
    for (const entry of discreteDistributions) {
      it(entry.name, () => {
        const dist = entry.factory();
        for (const p of QUANTILE_PROBS) {
          const k = dist.quantile(p);
          const recovered = dist.cdf(k);
          // For discrete dists, CDF(quantile(p)) >= p (quantile returns smallest k with CDF(k) >= p)
          expect(recovered).toBeGreaterThanOrEqual(p - 1e-9);
        }
      });
    }
  });

  // ---- 2. CDF monotonicity ----
  describe('CDF monotonicity: k1 < k2 => CDF(k1) <= CDF(k2)', () => {
    for (const entry of discreteDistributions) {
      it(entry.name, () => {
        const dist = entry.factory();
        const range = entry.testRange;
        for (let i = 1; i < range.length; i++) {
          expect(dist.cdf(range[i])).toBeGreaterThanOrEqual(dist.cdf(range[i - 1]) - 1e-12);
        }
      });
    }
  });

  // ---- 3. PMF non-negativity ----
  describe('PMF non-negativity: pmf(k) >= 0', () => {
    for (const entry of discreteDistributions) {
      it(entry.name, () => {
        const dist = entry.factory();
        for (const k of entry.testRange) {
          expect(dist.pmf(k)).toBeGreaterThanOrEqual(-1e-12);
        }
      });
    }
  });

  // ---- 4. CDF bounds ----
  describe('CDF bounds: 0 <= CDF(k) <= 1', () => {
    for (const entry of discreteDistributions) {
      it(entry.name, () => {
        const dist = entry.factory();
        for (const k of entry.testRange) {
          const c = dist.cdf(k);
          expect(c).toBeGreaterThanOrEqual(-1e-12);
          expect(c).toBeLessThanOrEqual(1 + 1e-12);
        }
      });
    }
  });

  // ---- 5. PMF sums to ~1 over support ----
  describe('PMF sums to approximately 1 over test range', () => {
    for (const entry of discreteDistributions) {
      it(entry.name, () => {
        const dist = entry.factory();
        const total = entry.testRange.reduce((s, k) => s + dist.pmf(k), 0);
        // For bounded distributions the sum should be very close to 1;
        // for unbounded ones the test range may not cover everything, so relax
        expect(total).toBeLessThanOrEqual(1 + 1e-9);
        expect(total).toBeGreaterThan(0.9);
      });
    }
  });

  // ---- 6. SF complement ----
  describe('SF complement: SF(k) ≈ 1 - CDF(k)', () => {
    for (const entry of discreteDistributions) {
      it(entry.name, () => {
        const dist = entry.factory();
        for (const k of entry.testRange) {
          expect(dist.sf(k)).toBeCloseTo(1 - dist.cdf(k), 10);
        }
      });
    }
  });

  // ---- 7. Sample mean convergence ----
  describe('Sample mean convergence', () => {
    for (const entry of discreteDistributions) {
      if (!entry.finiteMean) continue;
      it(entry.name, () => {
        const dist = entry.factory();
        const samples = dist.sampleN(N_SAMPLES);
        const sMean = sampleMean(samples);
        const thMean = dist.mean();
        if (Math.abs(thMean) < 0.5) {
          expect(Math.abs(sMean - thMean)).toBeLessThan(0.3);
        } else {
          expect(Math.abs((sMean - thMean) / thMean)).toBeLessThan(0.15);
        }
      });
    }
  });

  // ---- 8. Sample variance convergence ----
  describe('Sample variance convergence', () => {
    for (const entry of discreteDistributions) {
      if (!entry.finiteVariance) continue;
      it(entry.name, () => {
        const dist = entry.factory();
        const samples = dist.sampleN(N_SAMPLES);
        const sVar = sampleVariance(samples);
        const thVar = dist.variance();
        if (thVar < 0.1) {
          expect(Math.abs(sVar - thVar)).toBeLessThan(0.15);
        } else {
          expect(Math.abs((sVar - thVar) / thVar)).toBeLessThan(0.2);
        }
      });
    }
  });
});

// ===========================================================================
// MULTIVARIATE DISTRIBUTION TESTS
// ===========================================================================

describe('Property-based tests: Multivariate distributions', () => {
  const mvnRng = seededRng(123);

  describe('MultivariateNormal', () => {
    const mu = [1, 2, 3];
    const cov = [
      [2, 0.5, 0],
      [0.5, 3, 0.5],
      [0, 0.5, 1],
    ];

    it('sample returns correct dimensions', () => {
      const dist = new MultivariateNormal(mu, cov, mvnRng);
      const s = dist.sample();
      expect(s).toHaveLength(3);
    });

    it('mean returns correct dimensions and values', () => {
      const dist = new MultivariateNormal(mu, cov, mvnRng);
      // MultivariateNormal.mean is a property, not a method
      expect(dist.mean).toHaveLength(3);
      expect(dist.mean).toEqual(mu);
    });

    it('sampleN returns correct shape', () => {
      const dist = new MultivariateNormal(mu, cov, mvnRng);
      const samples = dist.sampleN(100);
      expect(samples).toHaveLength(100);
      for (const s of samples) {
        expect(s).toHaveLength(3);
      }
    });

    it('sample mean converges to theoretical mean', () => {
      const dist = new MultivariateNormal(mu, cov, mvnRng);
      const samples = dist.sampleN(N_SAMPLES);
      for (let d = 0; d < 3; d++) {
        const componentMean = samples.reduce((s, x) => s + x[d], 0) / N_SAMPLES;
        expect(Math.abs((componentMean - mu[d]) / (mu[d] || 1))).toBeLessThan(0.15);
      }
    });
  });

  describe('MultivariateT', () => {
    const mu = [0, 1];
    const sigma = [
      [1, 0.3],
      [0.3, 1],
    ];
    const df = 5;

    it('sample returns correct dimensions', () => {
      const dist = new MultivariateT(mu, sigma, df, mvnRng);
      const s = dist.sample();
      expect(s).toHaveLength(2);
    });

    it('mean returns correct dimensions (df > 1)', () => {
      const dist = new MultivariateT(mu, sigma, df, mvnRng);
      const m = dist.mean();
      expect(m).toHaveLength(2);
      expect(m[0]).toBeCloseTo(mu[0], 10);
      expect(m[1]).toBeCloseTo(mu[1], 10);
    });

    it('sampleN returns correct shape', () => {
      const dist = new MultivariateT(mu, sigma, df, mvnRng);
      const samples = dist.sampleN(50);
      expect(samples).toHaveLength(50);
      for (const s of samples) {
        expect(s).toHaveLength(2);
      }
    });
  });

  describe('Dirichlet', () => {
    const alpha = [2, 3, 5];

    it('sample returns correct dimensions', () => {
      const dist = new Dirichlet(alpha, mvnRng);
      const s = dist.sample();
      expect(s).toHaveLength(3);
    });

    it('mean returns correct dimensions', () => {
      const dist = new Dirichlet(alpha, mvnRng);
      const m = dist.mean();
      expect(m).toHaveLength(3);
    });

    it('mean values are correct', () => {
      const dist = new Dirichlet(alpha, mvnRng);
      const m = dist.mean();
      const alphaSum = alpha.reduce((a, b) => a + b, 0);
      for (let i = 0; i < alpha.length; i++) {
        expect(m[i]).toBeCloseTo(alpha[i] / alphaSum, 10);
      }
    });

    it('samples sum to 1', () => {
      const dist = new Dirichlet(alpha, mvnRng);
      const samples = dist.sampleN(100);
      for (const s of samples) {
        const total = s.reduce((a, b) => a + b, 0);
        expect(total).toBeCloseTo(1, 8);
      }
    });

    it('all sample components are non-negative', () => {
      const dist = new Dirichlet(alpha, mvnRng);
      const samples = dist.sampleN(100);
      for (const s of samples) {
        for (const v of s) {
          expect(v).toBeGreaterThanOrEqual(0);
        }
      }
    });

    it('sample component means converge to theoretical means', () => {
      const dist = new Dirichlet(alpha, mvnRng);
      const samples = dist.sampleN(N_SAMPLES);
      const thMean = dist.mean();
      for (let d = 0; d < alpha.length; d++) {
        const compMean = samples.reduce((s, x) => s + x[d], 0) / N_SAMPLES;
        expect(Math.abs((compMean - thMean[d]) / thMean[d])).toBeLessThan(0.15);
      }
    });
  });

  describe('Multinomial', () => {
    const n = 20;
    const probs = [0.2, 0.3, 0.5];

    it('sample returns correct dimensions', () => {
      const dist = new Multinomial(n, probs, mvnRng);
      const s = dist.sample();
      expect(s).toHaveLength(3);
    });

    it('mean returns correct dimensions', () => {
      const dist = new Multinomial(n, probs, mvnRng);
      const m = dist.mean();
      expect(m).toHaveLength(3);
    });

    it('mean values are correct', () => {
      const dist = new Multinomial(n, probs, mvnRng);
      const m = dist.mean();
      for (let i = 0; i < probs.length; i++) {
        expect(m[i]).toBeCloseTo(n * probs[i], 10);
      }
    });

    it('samples sum to n', () => {
      const dist = new Multinomial(n, probs, mvnRng);
      const samples = dist.sampleN(200);
      for (const s of samples) {
        const total = s.reduce((a, b) => a + b, 0);
        expect(total).toBe(n);
      }
    });

    it('all sample components are non-negative integers', () => {
      const dist = new Multinomial(n, probs, mvnRng);
      const samples = dist.sampleN(100);
      for (const s of samples) {
        for (const v of s) {
          expect(v).toBeGreaterThanOrEqual(0);
          expect(Number.isInteger(v)).toBe(true);
        }
      }
    });

    it('sample component means converge to theoretical means', () => {
      const dist = new Multinomial(n, probs, mvnRng);
      const samples = dist.sampleN(N_SAMPLES);
      const thMean = dist.mean();
      for (let d = 0; d < probs.length; d++) {
        const compMean = samples.reduce((s, x) => s + x[d], 0) / N_SAMPLES;
        expect(Math.abs((compMean - thMean[d]) / thMean[d])).toBeLessThan(0.1);
      }
    });
  });
});
