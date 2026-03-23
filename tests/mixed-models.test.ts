import {
  lmmRandomIntercept,
  lmmRandomSlope,
  icc,
  lrtTest,
} from "../src/mixed-models";

// ── Random Intercept Model ────────────────────────────────────────────────

describe("lmmRandomIntercept", () => {
  // Data: 3 groups, each with different intercepts
  const groups = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
  const X = [[1], [2], [3], [4], [1], [2], [3], [4], [1], [2], [3], [4]];
  // y = 2*x + group_offset + noise
  // group 0: offset 0, group 1: offset 5, group 2: offset 10
  const y = [2, 4, 6, 8, 7, 9, 11, 13, 12, 14, 16, 18];

  it("recovers fixed effects approximately", () => {
    const result = lmmRandomIntercept(y, X, groups);
    // Fixed slope should be close to 2
    expect(result.fixedEffects[1]).toBeCloseTo(2, 0);
    expect(result.fixedEffects.length).toBe(2); // intercept + slope
  });

  it("estimates non-zero random intercept variance", () => {
    const result = lmmRandomIntercept(y, X, groups);
    expect(result.randomEffects.interceptVariance).toBeGreaterThan(0);
  });

  it("produces BLUPs for each group", () => {
    const result = lmmRandomIntercept(y, X, groups);
    expect(result.blups.size).toBe(3);
    // Group 2 should have highest BLUP, group 0 lowest
    const b0 = result.blups.get(0)![0];
    const b2 = result.blups.get(2)![0];
    expect(b2).toBeGreaterThan(b0);
  });

  it("predict works with and without group", () => {
    const result = lmmRandomIntercept(y, X, groups);
    const predNoGroup = result.predict([3]);
    const predGroup0 = result.predict([3], 0);
    const predGroup2 = result.predict([3], 2);
    // Group predictions should differ
    expect(predGroup2).toBeGreaterThan(predGroup0);
    // Population prediction is between group predictions
    expect(predNoGroup).toBeDefined();
  });

  it("returns valid AIC and BIC", () => {
    const result = lmmRandomIntercept(y, X, groups);
    expect(Number.isFinite(result.aic)).toBe(true);
    expect(Number.isFinite(result.bic)).toBe(true);
    expect(result.nVarParams).toBe(2);
  });

  it("intercept-only model works (X = null)", () => {
    const result = lmmRandomIntercept(y, null, groups);
    expect(result.fixedEffects.length).toBe(1); // just intercept
    expect(result.blups.size).toBe(3);
  });

  it("throws for mismatched lengths", () => {
    expect(() => lmmRandomIntercept([1, 2], null, [0])).toThrow("same length");
  });
});

// ── Random Slope Model ────────────────────────────────────────────────────

describe("lmmRandomSlope", () => {
  // Groups with different slopes
  const groups = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
  const X = [[1], [2], [3], [4], [1], [2], [3], [4], [1], [2], [3], [4]];
  // group 0: y = 1*x, group 1: y = 2*x, group 2: y = 3*x
  const y = [1, 2, 3, 4, 2, 4, 6, 8, 3, 6, 9, 12];

  it("estimates both intercept and slope variance", () => {
    const result = lmmRandomSlope(y, X, groups);
    expect(result.randomEffects.interceptVariance).toBeGreaterThanOrEqual(0);
    expect(result.randomEffects.slopeVariance).toBeDefined();
    expect(result.randomEffects.slopeVariance!).toBeGreaterThanOrEqual(0);
  });

  it("produces 2-element BLUPs (intercept + slope)", () => {
    const result = lmmRandomSlope(y, X, groups);
    for (const [, blup] of result.blups) {
      expect(blup.length).toBe(2);
    }
  });

  it("predict includes random slope adjustment", () => {
    const result = lmmRandomSlope(y, X, groups);
    const pred0 = result.predict([3], 0);
    const pred2 = result.predict([3], 2);
    // Group 2 has steeper slope, so at x=3, prediction should be higher
    expect(pred2).toBeGreaterThan(pred0);
  });

  it("returns slope covariance", () => {
    const result = lmmRandomSlope(y, X, groups);
    expect(result.randomEffects.interceptSlopeCovariance).toBeDefined();
    expect(Number.isFinite(result.randomEffects.interceptSlopeCovariance!)).toBe(true);
  });

  it("has more variance params than intercept-only model", () => {
    const result = lmmRandomSlope(y, X, groups);
    expect(result.nVarParams).toBe(4); // sigma2, g00, g11, g01
  });
});

// ── ICC ───────────────────────────────────────────────────────────────────

describe("icc", () => {
  it("ICC ≈ 1 for pure between-group variation", () => {
    const values = [10, 10, 10, 20, 20, 20, 30, 30, 30];
    const groups = [0, 0, 0, 1, 1, 1, 2, 2, 2];
    const result = icc(values, groups);
    expect(result.icc).toBeCloseTo(1, 1);
  });

  it("ICC ≈ 0 for pure within-group variation", () => {
    // Same mean, high within-group variance
    const values = [1, 5, 9, 2, 6, 8, 3, 4, 7];
    const groups = [0, 0, 0, 1, 1, 1, 2, 2, 2];
    const result = icc(values, groups);
    expect(result.icc).toBeLessThan(0.3);
  });

  it("ICC is between 0 and 1", () => {
    const values = [1, 2, 3, 5, 6, 7, 10, 11, 12];
    const groups = [0, 0, 0, 1, 1, 1, 2, 2, 2];
    const result = icc(values, groups);
    expect(result.icc).toBeGreaterThanOrEqual(0);
    expect(result.icc).toBeLessThanOrEqual(1);
  });

  it("returns between and within variance", () => {
    const values = [1, 2, 3, 10, 11, 12];
    const groups = [0, 0, 0, 1, 1, 1];
    const result = icc(values, groups);
    expect(result.betweenVariance).toBeGreaterThan(0);
    expect(result.withinVariance).toBeGreaterThan(0);
  });

  it("throws for fewer than 2 groups", () => {
    expect(() => icc([1, 2, 3], [0, 0, 0])).toThrow("at least 2 groups");
  });
});

// ── Likelihood Ratio Test ─────────────────────────────────────────────────

describe("lrtTest", () => {
  it("positive statistic when full model is better", () => {
    const result = lrtTest(-100, -80, 2);
    expect(result.statistic).toBe(40);
    expect(result.df).toBe(2);
    expect(result.pValue).toBeGreaterThanOrEqual(0);
    expect(result.pValue).toBeLessThanOrEqual(1);
  });

  it("small statistic → large p-value", () => {
    const result = lrtTest(-100, -99.5, 1);
    expect(result.statistic).toBe(1);
    expect(result.pValue).toBeGreaterThan(0.1);
  });

  it("large statistic → small p-value", () => {
    const result = lrtTest(-100, -50, 1);
    expect(result.pValue).toBeLessThan(0.01);
  });

  it("throws for invalid dfDiff", () => {
    expect(() => lrtTest(-100, -80, 0)).toThrow("positive");
  });
});
