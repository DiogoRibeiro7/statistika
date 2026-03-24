import { TruncatedNormal } from "../../src/distributions/continuous/truncated-normal";
import { Laplace } from "../../src/distributions/continuous/laplace";
import { InverseGamma } from "../../src/distributions/continuous/inverse-gamma";
import { Rayleigh } from "../../src/distributions/continuous/rayleigh";
import { LogLogistic } from "../../src/distributions/continuous/log-logistic";
import { VonMises } from "../../src/distributions/continuous/von-mises";
import { InverseWishart } from "../../src/distributions/multivariate/inverse-wishart";
import { MultivariateT } from "../../src/distributions/multivariate/multivariate-t";

describe("TruncatedNormal distribution", () => {
  const dist = new TruncatedNormal(0, 1, -1, 1);

  it("has correct mean (0 by symmetry)", () => {
    expect(dist.mean()).toBeCloseTo(0, 5);
  });

  it("pdf is zero outside truncation bounds", () => {
    expect(dist.pdf(-2)).toBe(0);
    expect(dist.pdf(2)).toBe(0);
  });

  it("pdf is positive inside bounds", () => {
    expect(dist.pdf(0)).toBeGreaterThan(0);
    expect(dist.pdf(0.5)).toBeGreaterThan(0);
  });

  it("cdf(lower) = 0 and cdf(upper) = 1", () => {
    expect(dist.cdf(-1)).toBeCloseTo(0, 5);
    expect(dist.cdf(1)).toBeCloseTo(1, 5);
  });

  it("cdf(0) = 0.5 by symmetry", () => {
    expect(dist.cdf(0)).toBeCloseTo(0.5, 2);
  });

  it("sample returns values within bounds", () => {
    const samples = dist.sampleN(100);
    for (const s of samples) {
      expect(s).toBeGreaterThanOrEqual(-1);
      expect(s).toBeLessThanOrEqual(1);
    }
  });

  it("throws on invalid parameters", () => {
    expect(() => new TruncatedNormal(0, -1, -1, 1)).toThrow();
    expect(() => new TruncatedNormal(0, 1, 1, -1)).toThrow();
  });
});

describe("Laplace distribution", () => {
  const dist = new Laplace(0, 1);

  it("has correct mean and variance", () => {
    expect(dist.mean()).toBe(0);
    expect(dist.variance()).toBe(2);
  });

  it("pdf at mean equals 0.5", () => {
    expect(dist.pdf(0)).toBeCloseTo(0.5, 8);
  });

  it("pdf is symmetric", () => {
    expect(dist.pdf(1)).toBeCloseTo(dist.pdf(-1), 8);
  });

  it("cdf(0) = 0.5", () => {
    expect(dist.cdf(0)).toBeCloseTo(0.5, 8);
  });

  it("cdf and quantile are inverses", () => {
    for (const p of [0.1, 0.25, 0.5, 0.75, 0.9]) {
      expect(dist.cdf(dist.quantile(p))).toBeCloseTo(p, 4);
    }
  });

  it("throws on non-positive scale", () => {
    expect(() => new Laplace(0, 0)).toThrow();
    expect(() => new Laplace(0, -1)).toThrow();
  });
});

describe("InverseGamma distribution", () => {
  const dist = new InverseGamma(3, 2);

  it("has correct mean: beta / (alpha - 1)", () => {
    expect(dist.mean()).toBeCloseTo(1, 5);
  });

  it("has correct variance: beta^2 / ((alpha-1)^2 * (alpha-2))", () => {
    expect(dist.variance()).toBeCloseTo(1, 5); // 4 / (4 * 1) = 1
  });

  it("mean is Infinity for alpha <= 1", () => {
    const d = new InverseGamma(1, 1);
    expect(d.mean()).toBe(Infinity);
  });

  it("pdf is 0 for x <= 0", () => {
    expect(dist.pdf(0)).toBe(0);
    expect(dist.pdf(-1)).toBe(0);
  });

  it("pdf is positive for x > 0", () => {
    expect(dist.pdf(1)).toBeGreaterThan(0);
    expect(dist.pdf(0.5)).toBeGreaterThan(0);
  });

  it("cdf is 0 for x <= 0 and approaches 1 for large x", () => {
    expect(dist.cdf(0)).toBe(0);
    expect(dist.cdf(100)).toBeCloseTo(1, 2);
  });

  it("throws on invalid parameters", () => {
    expect(() => new InverseGamma(0, 1)).toThrow();
    expect(() => new InverseGamma(1, 0)).toThrow();
  });
});

describe("Rayleigh distribution", () => {
  const dist = new Rayleigh(1);

  it("has correct mean: sigma * sqrt(pi/2)", () => {
    expect(dist.mean()).toBeCloseTo(Math.sqrt(Math.PI / 2), 5);
  });

  it("has correct variance: (4 - pi)/2 * sigma^2", () => {
    expect(dist.variance()).toBeCloseTo((4 - Math.PI) / 2, 5);
  });

  it("pdf is 0 for x < 0", () => {
    expect(dist.pdf(-1)).toBe(0);
  });

  it("cdf is 0 for x < 0 and approaches 1 for large x", () => {
    expect(dist.cdf(-1)).toBe(0);
    expect(dist.cdf(10)).toBeCloseTo(1, 5);
  });

  it("quantile and cdf are inverses", () => {
    for (const p of [0.1, 0.5, 0.9]) {
      expect(dist.cdf(dist.quantile(p))).toBeCloseTo(p, 4);
    }
  });

  it("throws on non-positive sigma", () => {
    expect(() => new Rayleigh(0)).toThrow();
    expect(() => new Rayleigh(-1)).toThrow();
  });
});

describe("LogLogistic distribution", () => {
  const dist = new LogLogistic(1, 3);

  it("median equals alpha", () => {
    expect(dist.quantile(0.5)).toBeCloseTo(1, 4);
  });

  it("mean is finite for beta > 1", () => {
    expect(Number.isFinite(dist.mean())).toBe(true);
    expect(dist.mean()).toBeGreaterThan(0);
  });

  it("mean is Infinity for beta <= 1", () => {
    const d = new LogLogistic(1, 1);
    expect(d.mean()).toBe(Infinity);
  });

  it("cdf(alpha) = 0.5", () => {
    expect(dist.cdf(1)).toBeCloseTo(0.5, 5);
  });

  it("pdf is 0 for x < 0", () => {
    expect(dist.pdf(-1)).toBe(0);
  });

  it("throws on invalid parameters", () => {
    expect(() => new LogLogistic(0, 1)).toThrow();
    expect(() => new LogLogistic(1, 0)).toThrow();
  });
});

describe("VonMises distribution", () => {
  const dist = new VonMises(0, 2);

  it("mean is the location parameter", () => {
    expect(dist.mean()).toBe(0);
  });

  it("variance is in [0, 1]", () => {
    expect(dist.variance()).toBeGreaterThanOrEqual(0);
    expect(dist.variance()).toBeLessThanOrEqual(1);
  });

  it("variance is 1 for kappa = 0 (uniform)", () => {
    const uniform = new VonMises(0, 0);
    expect(uniform.variance()).toBeCloseTo(1, 5);
  });

  it("pdf is maximum at the mean", () => {
    const atMean = dist.pdf(0);
    const away = dist.pdf(Math.PI / 2);
    expect(atMean).toBeGreaterThan(away);
  });

  it("pdf is positive everywhere on the circle", () => {
    for (const x of [-Math.PI, -1, 0, 1, Math.PI]) {
      expect(dist.pdf(x)).toBeGreaterThan(0);
    }
  });

  it("throws on negative kappa", () => {
    expect(() => new VonMises(0, -1)).toThrow();
  });
});

describe("InverseWishart distribution", () => {
  const scale = [[2, 0.5], [0.5, 2]];
  const dist = new InverseWishart(5, scale);

  it("mean is Psi / (df - p - 1)", () => {
    const m = dist.mean();
    expect(m).toHaveLength(2);
    expect(m[0]).toHaveLength(2);
    // mean[0][0] = 2 / (5 - 2 - 1) = 1
    expect(m[0][0]).toBeCloseTo(1, 5);
    // mean[0][1] = 0.5 / 2 = 0.25
    expect(m[0][1]).toBeCloseTo(0.25, 5);
  });

  it("sample returns a symmetric matrix", () => {
    const S = dist.sample();
    expect(S).toHaveLength(2);
    expect(S[0]).toHaveLength(2);
    expect(S[0][1]).toBeCloseTo(S[1][0], 8);
  });

  it("sampleN returns the requested number of samples", () => {
    const samples = dist.sampleN(5);
    expect(samples).toHaveLength(5);
    for (const S of samples) {
      expect(S).toHaveLength(2);
    }
  });

  it("logPdf is finite for a valid positive-definite matrix", () => {
    const X = [[1, 0], [0, 1]];
    const lp = dist.logPdf(X);
    expect(Number.isFinite(lp)).toBe(true);
  });

  it("throws when df is too small", () => {
    expect(() => new InverseWishart(1, [[1, 0], [0, 1]])).toThrow();
  });
});

describe("MultivariateT distribution", () => {
  const dist = new MultivariateT([0, 0], [[1, 0], [0, 1]], 5);

  it("mean is the location vector (for df > 1)", () => {
    const m = dist.mean();
    expect(m).toHaveLength(2);
    expect(m[0]).toBeCloseTo(0, 8);
    expect(m[1]).toBeCloseTo(0, 8);
  });

  it("covariance is sigma * df/(df-2) for df > 2", () => {
    const cov = dist.covariance();
    expect(cov).toHaveLength(2);
    // sigma * 5/3 = 5/3 for diagonal
    expect(cov[0][0]).toBeCloseTo(5 / 3, 4);
    expect(cov[1][1]).toBeCloseTo(5 / 3, 4);
  });

  it("pdf is positive at the mean", () => {
    expect(dist.pdf([0, 0])).toBeGreaterThan(0);
  });

  it("logPdf is finite at the mean", () => {
    expect(Number.isFinite(dist.logPdf([0, 0]))).toBe(true);
  });

  it("pdf is symmetric about the mean", () => {
    const p1 = dist.pdf([1, 0]);
    const p2 = dist.pdf([-1, 0]);
    expect(p1).toBeCloseTo(p2, 8);
  });

  it("sample returns correct dimension", () => {
    const s = dist.sample();
    expect(s).toHaveLength(2);
  });

  it("sampleN returns requested number of samples", () => {
    const samples = dist.sampleN(10);
    expect(samples).toHaveLength(10);
    for (const s of samples) {
      expect(s).toHaveLength(2);
    }
  });

  it("throws on invalid parameters", () => {
    expect(() => new MultivariateT([0], [[1, 0], [0, 1]], 5)).toThrow();
    expect(() => new MultivariateT([0, 0], [[1, 0], [0, 1]], 0)).toThrow();
  });
});
