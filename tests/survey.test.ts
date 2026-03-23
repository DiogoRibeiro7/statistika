import {
  weightedStats,
  weightedQuantile,
  horvitzThompson,
  designEffect,
  ratioEstimator,
  postStratify,
} from "../src/survey";

// ── weightedStats ─────────────────────────────────────────────────────────

describe("weightedStats", () => {
  it("equal weights reproduce unweighted statistics", () => {
    const data = [2, 4, 6, 8, 10];
    const weights = [1, 1, 1, 1, 1];
    const result = weightedStats(data, weights);

    expect(result.mean).toBeCloseTo(6, 10);
    expect(result.effectiveSampleSize).toBeCloseTo(5, 10);
  });

  it("computes correct weighted mean", () => {
    const data = [10, 20];
    const weights = [3, 1];
    const result = weightedStats(data, weights);
    // Weighted mean: (3*10 + 1*20) / 4 = 12.5
    expect(result.mean).toBeCloseTo(12.5, 10);
  });

  it("effective sample size decreases with unequal weights", () => {
    const data = [1, 2, 3, 4, 5];
    const equal = weightedStats(data, [1, 1, 1, 1, 1]);
    const unequal = weightedStats(data, [10, 1, 1, 1, 1]);

    expect(unequal.effectiveSampleSize).toBeLessThan(equal.effectiveSampleSize);
  });

  it("variance and stdDev are consistent", () => {
    const result = weightedStats([1, 2, 3, 4, 5], [1, 2, 1, 2, 1]);
    expect(result.stdDev).toBeCloseTo(Math.sqrt(result.variance), 10);
  });

  it("throws for mismatched lengths", () => {
    expect(() => weightedStats([1, 2], [1])).toThrow("same length");
  });

  it("throws for negative weights", () => {
    expect(() => weightedStats([1], [-1])).toThrow("non-negative");
  });
});

// ── weightedQuantile ──────────────────────────────────────────────────────

describe("weightedQuantile", () => {
  it("median of equal weights matches regular median", () => {
    const data = [1, 2, 3, 4, 5];
    const weights = [1, 1, 1, 1, 1];
    expect(weightedQuantile(data, weights, 0.5)).toBe(3);
  });

  it("heavily weighted value dominates", () => {
    const data = [10, 20, 30];
    const weights = [100, 1, 1];
    // The median should be 10 since it has 98% of the weight
    expect(weightedQuantile(data, weights, 0.5)).toBe(10);
  });

  it("p=0 returns minimum", () => {
    expect(weightedQuantile([3, 1, 2], [1, 1, 1], 0)).toBe(1);
  });

  it("p=1 returns maximum", () => {
    expect(weightedQuantile([3, 1, 2], [1, 1, 1], 1)).toBe(3);
  });

  it("throws for invalid p", () => {
    expect(() => weightedQuantile([1], [1], -0.1)).toThrow("p must be");
  });
});

// ── horvitzThompson ───────────────────────────────────────────────────────

describe("horvitzThompson", () => {
  it("estimates population total", () => {
    // 5 observations from population of 100
    // Each sampled with probability 0.05 (= 5/100)
    const y = [10, 20, 30, 40, 50];
    const pi = [0.05, 0.05, 0.05, 0.05, 0.05];

    const result = horvitzThompson(y, pi, 100);
    // Total = sum(yi/pi) = 20*(10+20+30+40+50) = 3000
    expect(result.total).toBeCloseTo(3000, 8);
    expect(result.mean).toBeCloseTo(30, 8);
  });

  it("unequal inclusion probabilities", () => {
    const y = [10, 50];
    const pi = [0.5, 0.1]; // second unit harder to sample

    const result = horvitzThompson(y, pi, 10);
    // Total = 10/0.5 + 50/0.1 = 20 + 500 = 520
    expect(result.total).toBeCloseTo(520, 8);
  });

  it("returns confidence interval", () => {
    const y = [10, 20, 30];
    const pi = [0.1, 0.1, 0.1];

    const result = horvitzThompson(y, pi, 100);
    expect(result.ciTotal[0]).toBeLessThan(result.total);
    expect(result.ciTotal[1]).toBeGreaterThan(result.total);
  });

  it("throws for invalid inclusion probabilities", () => {
    expect(() => horvitzThompson([1], [0], 10)).toThrow("(0, 1]");
    expect(() => horvitzThompson([1], [1.5], 10)).toThrow("(0, 1]");
  });

  it("throws when population < sample", () => {
    expect(() => horvitzThompson([1, 2, 3], [0.5, 0.5, 0.5], 2)).toThrow("at least");
  });
});

// ── designEffect ──────────────────────────────────────────────────────────

describe("designEffect", () => {
  it("DEFF = 1 for equal weights", () => {
    const data = [1, 2, 3, 4, 5];
    const weights = [1, 1, 1, 1, 1];
    const result = designEffect(data, weights);

    expect(result.deff).toBeCloseTo(1, 8);
    expect(result.effectiveSampleSize).toBeCloseTo(5, 8);
  });

  it("DEFF > 1 for unequal weights", () => {
    const data = [1, 2, 3, 4, 5];
    const weights = [10, 1, 1, 1, 1];
    const result = designEffect(data, weights);

    expect(result.deff).toBeGreaterThan(1);
    expect(result.effectiveSampleSize).toBeLessThan(5);
  });

  it("throws for fewer than 2 observations", () => {
    expect(() => designEffect([1], [1])).toThrow("at least 2");
  });
});

// ── ratioEstimator ────────────────────────────────────────────────────────

describe("ratioEstimator", () => {
  it("estimates ratio and population total", () => {
    const y = [10, 20, 30, 40, 50];
    const x = [5, 10, 15, 20, 25]; // y = 2*x
    const weights = [1, 1, 1, 1, 1];
    const xTotal = 1000;

    const result = ratioEstimator(y, x, weights, xTotal);
    expect(result.ratio).toBeCloseTo(2, 8);
    expect(result.total).toBeCloseTo(2000, 8);
  });

  it("standard error is non-negative", () => {
    const y = [10, 12, 14];
    const x = [5, 6, 7];
    const weights = [1, 2, 1];

    const result = ratioEstimator(y, x, weights, 100);
    expect(result.standardError).toBeGreaterThanOrEqual(0);
  });

  it("throws for mismatched lengths", () => {
    expect(() => ratioEstimator([1], [1, 2], [1], 100)).toThrow("same length");
  });
});

// ── postStratify ──────────────────────────────────────────────────────────

describe("postStratify", () => {
  it("adjusts weights to match population counts", () => {
    const weights = [1, 1, 1, 1]; // 2 in stratum 0, 2 in stratum 1
    const strata = [0, 0, 1, 1];
    const popCounts = { 0: 100, 1: 200 }; // stratum 1 is twice as large

    const adjusted = postStratify(weights, strata, popCounts);
    // Stratum 0: 2 obs, pop 100, factor = 100/2 = 50
    // Stratum 1: 2 obs, pop 200, factor = 200/2 = 100
    expect(adjusted[0]).toBeCloseTo(50, 8);
    expect(adjusted[1]).toBeCloseTo(50, 8);
    expect(adjusted[2]).toBeCloseTo(100, 8);
    expect(adjusted[3]).toBeCloseTo(100, 8);
  });

  it("preserves relative weights within strata", () => {
    const weights = [1, 2, 3, 4]; // stratum 0: weights 1,2; stratum 1: weights 3,4
    const strata = [0, 0, 1, 1];
    const popCounts = { 0: 60, 1: 140 };

    const adjusted = postStratify(weights, strata, popCounts);
    // Within stratum 0: ratio should be preserved (1:2)
    expect(adjusted[1] / adjusted[0]).toBeCloseTo(2, 8);
  });

  it("throws for unknown stratum", () => {
    expect(() =>
      postStratify([1, 1], [0, 1], { 0: 100 }),
    ).toThrow("No population count for stratum 1");
  });
});
