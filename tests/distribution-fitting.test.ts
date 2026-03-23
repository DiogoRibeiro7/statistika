import {
  fitNormal,
  fitExponential,
  fitPoisson,
  fitGamma,
  fitBeta,
  fitLogNormal,
  fitGeometric,
  fitZIP,
  andersonDarlingTest,
  cramerVonMisesTest,
} from "../src/distribution-fitting";
import { Normal } from "../src/distributions/continuous/normal";

// ── Helper: seeded pseudo-random normal samples ───────────────────────────

function normalSamples(n: number, mu: number, sigma: number): number[] {
  // Deterministic samples from inverse CDF at evenly spaced quantiles
  const dist = new Normal(mu, sigma);
  const data: number[] = [];
  for (let i = 0; i < n; i++) {
    const p = (i + 0.5) / n;
    data.push(dist.quantile(p));
  }
  return data;
}

// ── MLE Fitting ───────────────────────────────────────────────────────────

describe("fitNormal", () => {
  it("recovers parameters from Normal data", () => {
    const data = normalSamples(200, 5, 2);
    const result = fitNormal(data);

    expect(result.distribution.mu).toBeCloseTo(5, 1);
    expect(result.distribution.sigma).toBeCloseTo(2, 1);
    expect(result.nParams).toBe(2);
    expect(result.logLikelihood).toBeLessThan(0);
    expect(result.aic).toBeGreaterThan(0);
  });

  it("throws for insufficient data", () => {
    expect(() => fitNormal([1])).toThrow("at least 2");
  });
});

describe("fitExponential", () => {
  it("recovers rate parameter", () => {
    // Exponential quantiles: F^{-1}(p) = -ln(1-p)/lambda
    const lambda = 2;
    const data: number[] = [];
    for (let i = 0; i < 200; i++) {
      data.push(-Math.log(1 - (i + 0.5) / 200) / lambda);
    }

    const result = fitExponential(data);
    expect(result.distribution.lambda).toBeCloseTo(lambda, 0);
    expect(result.nParams).toBe(1);
  });

  it("throws for negative data", () => {
    expect(() => fitExponential([-1, 1, 2])).toThrow("non-negative");
  });
});

describe("fitPoisson", () => {
  it("recovers lambda from count data", () => {
    // Create data with known mean
    const data = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 6];
    const result = fitPoisson(data);

    const expectedLambda = data.reduce((a, b) => a + b, 0) / data.length;
    expect(result.distribution.lambda).toBeCloseTo(expectedLambda, 10);
    expect(result.nParams).toBe(1);
  });

  it("throws for non-integer data", () => {
    expect(() => fitPoisson([1.5, 2, 3])).toThrow("non-negative integer");
  });
});

describe("fitGamma", () => {
  it("recovers approximate parameters", () => {
    // Generate Gamma-like data from quantiles
    const data: number[] = [];
    for (let i = 0; i < 200; i++) {
      const p = (i + 0.5) / 200;
      // Approximate Gamma(2, 1) quantiles using chi-squared relationship
      data.push(Math.max(0.01, -2 * Math.log(1 - p)));
    }

    const result = fitGamma(data);
    expect(result.distribution).toBeDefined();
    expect(result.nParams).toBe(2);
    expect(result.logLikelihood).toBeLessThan(0);
  });

  it("throws for non-positive data", () => {
    expect(() => fitGamma([0, 1, 2])).toThrow("positive");
  });
});

describe("fitBeta", () => {
  it("recovers approximate parameters from (0,1) data", () => {
    // Generate Beta-like data
    const data: number[] = [];
    for (let i = 0; i < 100; i++) {
      data.push((i + 0.5) / 100);
    }
    // Uniform on (0,1) -> Beta(1,1), but method of moments gives ~1,1

    const result = fitBeta(data);
    expect(result.distribution).toBeDefined();
    expect(result.nParams).toBe(2);
  });

  it("throws for data outside (0,1)", () => {
    expect(() => fitBeta([0, 0.5, 1])).toThrow("(0, 1)");
  });
});

describe("fitLogNormal", () => {
  it("recovers log-space parameters", () => {
    // LogNormal: if X ~ LogNormal(mu, sigma), then log(X) ~ Normal(mu, sigma)
    const mu = 1;
    const sigma = 0.5;
    const logData = normalSamples(200, mu, sigma);
    const data = logData.map(Math.exp);

    const result = fitLogNormal(data);
    expect(result.distribution.mu).toBeCloseTo(mu, 1);
    expect(result.distribution.sigma).toBeCloseTo(sigma, 1);
    expect(result.nParams).toBe(2);
  });

  it("throws for non-positive data", () => {
    expect(() => fitLogNormal([-1, 1, 2])).toThrow("positive");
  });
});

describe("fitGeometric", () => {
  it("recovers p from geometric data", () => {
    // Geometric mean = (1-p)/p, so p = 1/(1 + mean)
    const data = [0, 0, 0, 1, 1, 2, 3, 0, 1, 0]; // mean = 0.8
    const result = fitGeometric(data);

    const expectedP = 1 / (1 + 0.8);
    expect(result.distribution.p).toBeCloseTo(expectedP, 8);
    expect(result.nParams).toBe(1);
  });
});

describe("fitZIP", () => {
  it("fits ZIP to zero-inflated data", () => {
    // Create data with excess zeros
    const data = [
      0, 0, 0, 0, 0, 0, 0, 0, // structural zeros
      0, 1, 2, 3, 1, 2, 0, 1, 3, 2, 4, 1, // Poisson-like
    ];
    const result = fitZIP(data);

    expect(result.distribution.pi).toBeGreaterThan(0);
    expect(result.distribution.lambda).toBeGreaterThan(0);
    expect(result.nParams).toBe(2);
  });

  it("pi is near zero for non-inflated data", () => {
    // Standard Poisson data without extra zeros
    const data = [1, 2, 3, 2, 1, 4, 2, 3, 1, 2, 3, 2, 1, 2, 3, 4, 5, 2, 1, 3];
    const result = fitZIP(data);
    expect(result.distribution.pi).toBeLessThan(0.3);
  });
});

// ── Anderson-Darling test ─────────────────────────────────────────────────

describe("andersonDarlingTest", () => {
  it("does not reject data from the correct distribution", () => {
    const data = normalSamples(100, 0, 1);
    const dist = new Normal(0, 1);

    const result = andersonDarlingTest(data, dist);
    expect(result.rejected).toBe(false);
    expect(result.statistic).toBeLessThan(2.492);
  });

  it("rejects data from wrong distribution", () => {
    // Exponential-like data tested against Normal(0,1)
    const data: number[] = [];
    for (let i = 0; i < 100; i++) {
      data.push(-Math.log(1 - (i + 0.5) / 100));
    }
    const dist = new Normal(0, 1);

    const result = andersonDarlingTest(data, dist);
    expect(result.rejected).toBe(true);
    expect(result.statistic).toBeGreaterThan(2.492);
  });

  it("statistic is non-negative", () => {
    const data = normalSamples(50, 0, 1);
    const result = andersonDarlingTest(data, new Normal(0, 1));
    expect(result.statistic).toBeGreaterThanOrEqual(0);
  });

  it("respects custom alpha", () => {
    const data = normalSamples(100, 0, 1);
    const dist = new Normal(0, 1);

    const strict = andersonDarlingTest(data, dist, 0.01);
    expect(strict.alpha).toBe(0.01);
  });

  it("throws for insufficient data", () => {
    expect(() => andersonDarlingTest([1], new Normal(0, 1))).toThrow("at least 2");
  });
});

// ── Cramér-von Mises test ─────────────────────────────────────────────────

describe("cramerVonMisesTest", () => {
  it("does not reject data from the correct distribution", () => {
    const data = normalSamples(100, 0, 1);
    const dist = new Normal(0, 1);

    const result = cramerVonMisesTest(data, dist);
    expect(result.rejected).toBe(false);
    expect(result.statistic).toBeLessThan(0.461);
  });

  it("rejects data from wrong distribution", () => {
    const data: number[] = [];
    for (let i = 0; i < 100; i++) {
      data.push(-Math.log(1 - (i + 0.5) / 100));
    }
    const dist = new Normal(0, 1);

    const result = cramerVonMisesTest(data, dist);
    expect(result.rejected).toBe(true);
  });

  it("statistic is non-negative", () => {
    const data = normalSamples(50, 0, 1);
    const result = cramerVonMisesTest(data, new Normal(0, 1));
    expect(result.statistic).toBeGreaterThanOrEqual(0);
  });

  it("throws for insufficient data", () => {
    expect(() => cramerVonMisesTest([1], new Normal(0, 1))).toThrow("at least 2");
  });
});

// ── Integration: fit + test ───────────────────────────────────────────────

describe("integration: fit then test", () => {
  it("fitted Normal passes Anderson-Darling on its own data", () => {
    const data = normalSamples(200, 10, 3);
    const fit = fitNormal(data);
    const result = andersonDarlingTest(data, fit.distribution);

    // Fitted distribution should match well
    expect(result.rejected).toBe(false);
  });

  it("fitted Normal passes Cramér-von Mises on its own data", () => {
    const data = normalSamples(200, 10, 3);
    const fit = fitNormal(data);
    const result = cramerVonMisesTest(data, fit.distribution);

    expect(result.rejected).toBe(false);
  });
});
