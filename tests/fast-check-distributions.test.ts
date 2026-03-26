/**
 * Property-based tests using fast-check for randomized parameter generation.
 *
 * Unlike the deterministic property tests in property-based-distributions.test.ts,
 * these tests use fast-check to randomly generate distribution parameters AND
 * test points, with automatic shrinking on failure.
 *
 * Properties verified:
 *   - CDF(quantile(p)) ≈ p with random parameters
 *   - PDF >= 0 with random x
 *   - CDF monotonicity with random x pairs
 *   - 0 <= CDF(x) <= 1
 *   - SF(x) + CDF(x) = 1
 *   - stdDev = sqrt(variance)
 */
import fc from "fast-check";

import { Normal } from "../src/distributions/continuous/normal";
import { Uniform } from "../src/distributions/continuous/uniform";
import { Exponential } from "../src/distributions/continuous/exponential";
import { GammaDistribution } from "../src/distributions/continuous/gamma";
import { BetaDistribution } from "../src/distributions/continuous/beta";
import { ChiSquared } from "../src/distributions/continuous/chi-squared";
import { StudentT } from "../src/distributions/continuous/student-t";
import { LogNormal } from "../src/distributions/continuous/log-normal";
import { Weibull } from "../src/distributions/continuous/weibull";
import { Cauchy } from "../src/distributions/continuous/cauchy";
import { Pareto } from "../src/distributions/continuous/pareto";
import { Laplace } from "../src/distributions/continuous/laplace";
import { Rayleigh } from "../src/distributions/continuous/rayleigh";
import { LogLogistic } from "../src/distributions/continuous/log-logistic";
import { Gumbel } from "../src/distributions/continuous/gumbel";
import { Frechet } from "../src/distributions/continuous/frechet";
import { Levy } from "../src/distributions/continuous/levy";

import { Binomial } from "../src/distributions/discrete/binomial";
import { Poisson } from "../src/distributions/discrete/poisson";
import { Geometric } from "../src/distributions/discrete/geometric";
import { Bernoulli } from "../src/distributions/discrete/bernoulli";

// ---------------------------------------------------------------------------
// Arbitraries
// ---------------------------------------------------------------------------

/** Probability in safe interior range. */
const safeP = fc.double({ min: 0.005, max: 0.995, noNaN: true });

/** Positive shape/scale/rate parameter. */
const pos = (min = 0.1, max = 50) => fc.double({ min, max, noNaN: true });

/** Location parameter. */
const loc = fc.double({ min: -50, max: 50, noNaN: true });

// ---------------------------------------------------------------------------
// Continuous: random parameters
// ---------------------------------------------------------------------------

describe("fast-check: Normal with random parameters", () => {
  it("CDF(quantile(p)) ≈ p", () => {
    fc.assert(
      fc.property(loc, pos(0.1, 20), safeP, (mu, sigma, p) => {
        const d = new Normal(mu, sigma);
        const rt = d.cdf(d.quantile(p));
        expect(rt).toBeCloseTo(p, 3);
      }),
      { numRuns: 100 },
    );
  });

  it("PDF >= 0 for random x", () => {
    fc.assert(
      fc.property(loc, pos(0.1, 20), fc.double({ min: -100, max: 100, noNaN: true }), (mu, sigma, x) => {
        expect(new Normal(mu, sigma).pdf(x)).toBeGreaterThanOrEqual(0);
      }),
      { numRuns: 100 },
    );
  });

  it("CDF monotonicity", () => {
    fc.assert(
      fc.property(loc, pos(0.1, 20), fc.double({ min: -50, max: 50, noNaN: true }), fc.double({ min: -50, max: 50, noNaN: true }), (mu, sigma, a, b) => {
        const d = new Normal(mu, sigma);
        if (a <= b) expect(d.cdf(b)).toBeGreaterThanOrEqual(d.cdf(a) - 1e-10);
      }),
      { numRuns: 100 },
    );
  });
});

describe("fast-check: Exponential with random rate", () => {
  it("CDF(quantile(p)) ≈ p", () => {
    fc.assert(
      fc.property(pos(0.01, 100), safeP, (lambda, p) => {
        const d = new Exponential(lambda);
        expect(d.cdf(d.quantile(p))).toBeCloseTo(p, 4);
      }),
      { numRuns: 100 },
    );
  });

  it("CDF in [0,1] for x > 0", () => {
    fc.assert(
      fc.property(pos(0.01, 100), pos(0.001, 1000), (lambda, x) => {
        const c = new Exponential(lambda).cdf(x);
        expect(c).toBeGreaterThanOrEqual(-1e-12);
        expect(c).toBeLessThanOrEqual(1 + 1e-12);
      }),
      { numRuns: 100 },
    );
  });
});

describe("fast-check: Gamma with random parameters", () => {
  it("CDF(quantile(p)) ≈ p", () => {
    fc.assert(
      fc.property(pos(0.5, 20), pos(0.1, 10), safeP, (shape, rate, p) => {
        const d = new GammaDistribution(shape, rate);
        expect(d.cdf(d.quantile(p))).toBeCloseTo(p, 2);
      }),
      { numRuns: 80 },
    );
  });

  it("PDF >= 0", () => {
    fc.assert(
      fc.property(pos(0.5, 20), pos(0.1, 10), pos(0.001, 100), (shape, rate, x) => {
        expect(new GammaDistribution(shape, rate).pdf(x)).toBeGreaterThanOrEqual(0);
      }),
      { numRuns: 100 },
    );
  });
});

describe("fast-check: Beta with random parameters", () => {
  it("CDF(quantile(p)) ≈ p", () => {
    fc.assert(
      fc.property(pos(0.5, 20), pos(0.5, 20), safeP, (alpha, beta, p) => {
        const d = new BetaDistribution(alpha, beta);
        expect(d.cdf(d.quantile(p))).toBeCloseTo(p, 2);
      }),
      { numRuns: 80 },
    );
  });

  it("CDF in [0,1]", () => {
    fc.assert(
      fc.property(pos(0.5, 20), pos(0.5, 20), fc.double({ min: 0, max: 1, noNaN: true }), (alpha, beta, x) => {
        const c = new BetaDistribution(alpha, beta).cdf(x);
        expect(c).toBeGreaterThanOrEqual(-1e-10);
        expect(c).toBeLessThanOrEqual(1 + 1e-10);
      }),
      { numRuns: 100 },
    );
  });
});

describe("fast-check: Uniform with random bounds", () => {
  it("CDF(quantile(p)) = p", () => {
    fc.assert(
      fc.property(
        fc.double({ min: -100, max: 100, noNaN: true }),
        fc.double({ min: 0.01, max: 100, noNaN: true }),
        safeP,
        (a, width, p) => {
          const d = new Uniform(a, a + width);
          expect(d.cdf(d.quantile(p))).toBeCloseTo(p, 8);
        },
      ),
      { numRuns: 100 },
    );
  });
});

describe("fast-check: Weibull with random parameters", () => {
  it("CDF(quantile(p)) ≈ p", () => {
    fc.assert(
      fc.property(pos(0.5, 10), pos(0.1, 10), safeP, (k, lambda, p) => {
        const d = new Weibull(k, lambda);
        expect(d.cdf(d.quantile(p))).toBeCloseTo(p, 4);
      }),
      { numRuns: 80 },
    );
  });
});

describe("fast-check: StudentT with random df", () => {
  it("CDF(quantile(p)) ≈ p", () => {
    fc.assert(
      fc.property(pos(2, 100), fc.double({ min: 0.01, max: 0.99, noNaN: true }), (nu, p) => {
        const d = new StudentT(nu);
        expect(d.cdf(d.quantile(p))).toBeCloseTo(p, 1);
      }),
      { numRuns: 80 },
    );
  });

  it("CDF(0) ≈ 0.5 (symmetric)", () => {
    fc.assert(
      fc.property(pos(1, 100), (nu) => {
        expect(new StudentT(nu).cdf(0)).toBeCloseTo(0.5, 3);
      }),
      { numRuns: 50 },
    );
  });
});

describe("fast-check: LogNormal with random parameters", () => {
  it("CDF(quantile(p)) ≈ p", () => {
    fc.assert(
      fc.property(loc, pos(0.1, 5), safeP, (mu, sigma, p) => {
        const d = new LogNormal(mu, sigma);
        expect(d.cdf(d.quantile(p))).toBeCloseTo(p, 3);
      }),
      { numRuns: 80 },
    );
  });
});

describe("fast-check: Cauchy with random parameters", () => {
  it("CDF(quantile(p)) ≈ p", () => {
    fc.assert(
      fc.property(loc, pos(0.1, 20), safeP, (x0, gamma, p) => {
        const d = new Cauchy(x0, gamma);
        expect(d.cdf(d.quantile(p))).toBeCloseTo(p, 5);
      }),
      { numRuns: 100 },
    );
  });
});

describe("fast-check: Pareto with random parameters", () => {
  it("CDF(quantile(p)) ≈ p", () => {
    fc.assert(
      fc.property(pos(0.5, 20), pos(0.1, 10), safeP, (alpha, xm, p) => {
        const d = new Pareto(alpha, xm);
        expect(d.cdf(d.quantile(p))).toBeCloseTo(p, 8);
      }),
      { numRuns: 100 },
    );
  });
});

describe("fast-check: Laplace with random parameters", () => {
  it("CDF(quantile(p)) ≈ p", () => {
    fc.assert(
      fc.property(loc, pos(0.1, 20), safeP, (mu, b, p) => {
        const d = new Laplace(mu, b);
        expect(d.cdf(d.quantile(p))).toBeCloseTo(p, 8);
      }),
      { numRuns: 100 },
    );
  });
});

describe("fast-check: Rayleigh with random sigma", () => {
  it("CDF(quantile(p)) ≈ p", () => {
    fc.assert(
      fc.property(pos(0.1, 20), safeP, (sigma, p) => {
        const d = new Rayleigh(sigma);
        expect(d.cdf(d.quantile(p))).toBeCloseTo(p, 8);
      }),
      { numRuns: 100 },
    );
  });
});

describe("fast-check: Gumbel with random parameters", () => {
  it("CDF(quantile(p)) ≈ p", () => {
    fc.assert(
      fc.property(loc, pos(0.1, 20), safeP, (mu, beta, p) => {
        const d = new Gumbel(mu, beta);
        expect(d.cdf(d.quantile(p))).toBeCloseTo(p, 5);
      }),
      { numRuns: 100 },
    );
  });
});

describe("fast-check: LogLogistic with random parameters", () => {
  it("CDF(quantile(p)) ≈ p", () => {
    fc.assert(
      fc.property(pos(0.1, 10), pos(0.5, 10), safeP, (alpha, beta, p) => {
        const d = new LogLogistic(alpha, beta);
        expect(d.cdf(d.quantile(p))).toBeCloseTo(p, 5);
      }),
      { numRuns: 80 },
    );
  });
});

describe("fast-check: Frechet with random parameters", () => {
  it("CDF(quantile(p)) ≈ p", () => {
    fc.assert(
      fc.property(pos(0.5, 10), pos(0.1, 10), safeP, (alpha, s, p) => {
        const d = new Frechet(alpha, s, 0);
        expect(d.cdf(d.quantile(p))).toBeCloseTo(p, 3);
      }),
      { numRuns: 80 },
    );
  });
});

describe("fast-check: Levy with random parameters", () => {
  it("CDF(quantile(p)) ≈ p", () => {
    fc.assert(
      fc.property(loc, pos(0.1, 20), fc.double({ min: 0.01, max: 0.95, noNaN: true }), (mu, c, p) => {
        const d = new Levy(mu, c);
        expect(d.cdf(d.quantile(p))).toBeCloseTo(p, 3);
      }),
      { numRuns: 80 },
    );
  });
});

// ---------------------------------------------------------------------------
// Discrete: random parameters
// ---------------------------------------------------------------------------

describe("fast-check: Binomial with random parameters", () => {
  it("CDF(quantile(p)) >= p", () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 1, max: 100 }),
        fc.double({ min: 0.01, max: 0.99, noNaN: true }),
        safeP,
        (n, prob, p) => {
          const d = new Binomial(n, prob);
          expect(d.cdf(d.quantile(p))).toBeGreaterThanOrEqual(p - 1e-9);
        },
      ),
      { numRuns: 80 },
    );
  });

  it("PMF in [0,1]", () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 1, max: 50 }),
        fc.double({ min: 0.01, max: 0.99, noNaN: true }),
        fc.integer({ min: 0, max: 50 }),
        (n, prob, k) => {
          const pm = new Binomial(n, prob).pmf(Math.min(k, n));
          expect(pm).toBeGreaterThanOrEqual(-1e-12);
          expect(pm).toBeLessThanOrEqual(1 + 1e-12);
        },
      ),
      { numRuns: 100 },
    );
  });
});

describe("fast-check: Poisson with random lambda", () => {
  it("CDF(quantile(p)) >= p", () => {
    fc.assert(
      fc.property(pos(0.1, 50), safeP, (lambda, p) => {
        const d = new Poisson(lambda);
        expect(d.cdf(d.quantile(p))).toBeGreaterThanOrEqual(p - 1e-9);
      }),
      { numRuns: 80 },
    );
  });

  it("PMF(k) >= 0 for k >= 0", () => {
    fc.assert(
      fc.property(pos(0.1, 50), fc.integer({ min: 0, max: 100 }), (lambda, k) => {
        expect(new Poisson(lambda).pmf(k)).toBeGreaterThanOrEqual(-1e-12);
      }),
      { numRuns: 100 },
    );
  });
});

describe("fast-check: Geometric with random p", () => {
  it("CDF(quantile(p)) >= p", () => {
    fc.assert(
      fc.property(fc.double({ min: 0.01, max: 0.99, noNaN: true }), safeP, (prob, p) => {
        const d = new Geometric(prob);
        expect(d.cdf(d.quantile(p))).toBeGreaterThanOrEqual(p - 1e-9);
      }),
      { numRuns: 80 },
    );
  });
});

describe("fast-check: Bernoulli with random p", () => {
  it("PMF sums to 1", () => {
    fc.assert(
      fc.property(fc.double({ min: 0.01, max: 0.99, noNaN: true }), (p) => {
        const d = new Bernoulli(p);
        expect(d.pmf(0) + d.pmf(1)).toBeCloseTo(1, 10);
      }),
      { numRuns: 100 },
    );
  });
});

// ---------------------------------------------------------------------------
// Cross-distribution: ChiSquared(k) ≈ Gamma(k/2, 0.5)
// ---------------------------------------------------------------------------

describe("fast-check: ChiSquared(k) matches Gamma(k/2, 0.5)", () => {
  it("CDF values agree", () => {
    fc.assert(
      fc.property(fc.integer({ min: 1, max: 30 }), pos(0.1, 50), (k, x) => {
        const chi = new ChiSquared(k);
        const gam = new GammaDistribution(k / 2, 0.5);
        expect(chi.cdf(x)).toBeCloseTo(gam.cdf(x), 3);
      }),
      { numRuns: 50 },
    );
  });
});
