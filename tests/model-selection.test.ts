import {
  informationCriteria,
  likelihoodRatioTest,
  vuongTest,
  compareModels,
} from "../src/model-selection";

// ── informationCriteria ───────────────────────────────────────────────────

describe("informationCriteria", () => {
  it("computes AIC = -2*ll + 2*k", () => {
    const ic = informationCriteria(-100, 3, 50);
    expect(ic.aic).toBeCloseTo(-2 * -100 + 2 * 3, 10);
    expect(ic.aic).toBeCloseTo(206, 10);
  });

  it("computes BIC = -2*ll + k*ln(n)", () => {
    const ic = informationCriteria(-100, 3, 50);
    expect(ic.bic).toBeCloseTo(-2 * -100 + 3 * Math.log(50), 10);
  });

  it("computes AICc with small-sample correction", () => {
    const ic = informationCriteria(-50, 3, 10);
    const expectedAIC = -2 * -50 + 2 * 3;
    const correction = (2 * 3 * 4) / (10 - 3 - 1); // 2k(k+1)/(n-k-1)
    expect(ic.aicc).toBeCloseTo(expectedAIC + correction, 10);
  });

  it("AICc = Infinity when n - k - 1 <= 0", () => {
    const ic = informationCriteria(-50, 5, 5);
    expect(ic.aicc).toBe(Infinity);
  });

  it("AICc converges to AIC for large n", () => {
    const ic = informationCriteria(-500, 3, 10000);
    expect(Math.abs(ic.aicc - ic.aic)).toBeLessThan(0.01);
  });

  it("lower AIC is better (more parameters penalised)", () => {
    const simple = informationCriteria(-100, 2, 50);
    const complex = informationCriteria(-100, 5, 50);
    expect(simple.aic).toBeLessThan(complex.aic);
  });

  it("better fit offsets parameter penalty", () => {
    const simple = informationCriteria(-100, 2, 50);
    const betterFit = informationCriteria(-90, 3, 50);
    expect(betterFit.aic).toBeLessThan(simple.aic);
  });

  it("BIC penalises more heavily than AIC for large n", () => {
    const ic = informationCriteria(-100, 5, 1000);
    // BIC penalty: k*ln(n) = 5*ln(1000) ≈ 34.5
    // AIC penalty: 2*k = 10
    expect(ic.bic).toBeGreaterThan(ic.aic);
  });

  it("throws for invalid inputs", () => {
    expect(() => informationCriteria(-100, 0, 50)).toThrow("k must be at least 1");
    expect(() => informationCriteria(-100, 3, 0)).toThrow("n must be at least 1");
    expect(() => informationCriteria(NaN, 3, 50)).toThrow("finite");
  });
});

// ── likelihoodRatioTest ───────────────────────────────────────────────────

describe("likelihoodRatioTest", () => {
  it("rejects when full model is much better", () => {
    // Restricted model: ll = -200, k = 2
    // Full model: ll = -180, k = 5
    // D = -2(-200 - (-180)) = 40, df = 3
    const result = likelihoodRatioTest(-200, -180, 2, 5);

    expect(result.statistic).toBeCloseTo(40, 10);
    expect(result.degreesOfFreedom).toBe(3);
    expect(result.pValue).toBeLessThan(0.001);
    expect(result.rejected).toBe(true);
  });

  it("does not reject when models are similar", () => {
    // Very small improvement
    const result = likelihoodRatioTest(-100, -99.9, 2, 3);

    expect(result.statistic).toBeCloseTo(0.2, 8);
    expect(result.degreesOfFreedom).toBe(1);
    expect(result.pValue).toBeGreaterThan(0.05);
    expect(result.rejected).toBe(false);
  });

  it("handles df = 1 correctly", () => {
    const result = likelihoodRatioTest(-100, -97, 3, 4);
    expect(result.degreesOfFreedom).toBe(1);
    expect(result.statistic).toBeCloseTo(6, 10);
    // chi2(1) cdf at 6 ≈ 0.986, so p ≈ 0.014
    expect(result.pValue).toBeLessThan(0.05);
    expect(result.rejected).toBe(true);
  });

  it("statistic is clamped at zero", () => {
    // Due to numerical issues, restricted can sometimes appear slightly better
    const result = likelihoodRatioTest(-100, -100.01, 2, 3);
    expect(result.statistic).toBe(0);
    expect(result.pValue).toBe(1);
    expect(result.rejected).toBe(false);
  });

  it("respects custom alpha", () => {
    // p ≈ 0.03 -- rejected at 0.05 but not at 0.01
    const result005 = likelihoodRatioTest(-100, -97.7, 3, 4, 0.05);
    const result001 = likelihoodRatioTest(-100, -97.7, 3, 4, 0.01);
    expect(result005.rejected).toBe(true);
    expect(result001.rejected).toBe(false);
  });

  it("throws when full model has fewer params", () => {
    expect(() => likelihoodRatioTest(-100, -95, 5, 3)).toThrow("more parameters");
  });

  it("throws for non-finite log-likelihoods", () => {
    expect(() => likelihoodRatioTest(NaN, -95, 2, 3)).toThrow("finite");
    expect(() => likelihoodRatioTest(-100, Infinity, 2, 3)).toThrow("finite");
  });
});

// ── vuongTest ─────────────────────────────────────────────────────────────

describe("vuongTest", () => {
  it("prefers model 1 when it has consistently higher log-likelihoods", () => {
    const n = 100;
    // Model 1 is better on average, with some variation
    const logLik1 = new Array(n).fill(0).map((_, i) => -2 + 0.5 * Math.sin(i));
    const logLik2 = new Array(n).fill(0).map((_, i) => -5 + 0.5 * Math.cos(i));

    const result = vuongTest(logLik1, logLik2);
    expect(result.statistic).toBeGreaterThan(0);
    expect(result.pValue).toBeLessThan(0.05);
    expect(result.preferred).toBe("model1");
  });

  it("prefers model 2 when it has consistently higher log-likelihoods", () => {
    const n = 100;
    const logLik1 = new Array(n).fill(0).map((_, i) => -10 + 0.3 * Math.sin(i));
    const logLik2 = new Array(n).fill(0).map((_, i) => -5 + 0.3 * Math.cos(i));

    const result = vuongTest(logLik1, logLik2);
    expect(result.statistic).toBeLessThan(0);
    expect(result.pValue).toBeLessThan(0.05);
    expect(result.preferred).toBe("model2");
  });

  it("returns indistinguishable for identical models", () => {
    const logLik = [-1, -2, -3, -4, -5];
    const result = vuongTest(logLik, logLik);
    expect(result.statistic).toBe(0);
    expect(result.pValue).toBe(1);
    expect(result.preferred).toBe("indistinguishable");
  });

  it("returns indistinguishable for mixed log-likelihoods with no clear winner", () => {
    // Alternating which model is better -- high variance, no signal
    const n = 50;
    const logLik1 = new Array(n).fill(0).map((_, i) => (i % 2 === 0 ? -1 : -3));
    const logLik2 = new Array(n).fill(0).map((_, i) => (i % 2 === 0 ? -3 : -1));

    const result = vuongTest(logLik1, logLik2);
    expect(result.preferred).toBe("indistinguishable");
  });

  it("applies Schwarz correction when k1 and k2 are provided", () => {
    const n = 100;
    // Model 1 is slightly better but has more parameters
    const logLik1 = new Array(n).fill(0).map((_, i) => -2 + 0.3 * Math.sin(i));
    const logLik2 = new Array(n).fill(0).map((_, i) => -2.3 + 0.3 * Math.cos(i));

    const withoutCorrection = vuongTest(logLik1, logLik2);
    const withCorrection = vuongTest(logLik1, logLik2, { k1: 10, k2: 3 });

    // Correction penalises model 1 (more params), reducing its advantage
    expect(withCorrection.statistic).toBeLessThan(withoutCorrection.statistic);
  });

  it("throws for mismatched lengths", () => {
    expect(() => vuongTest([1, 2], [1, 2, 3])).toThrow("same length");
  });

  it("throws for fewer than 2 observations", () => {
    expect(() => vuongTest([1], [2])).toThrow("at least 2");
  });
});

// ── compareModels ─────────────────────────────────────────────────────────

describe("compareModels", () => {
  it("sorts models by AIC (best first)", () => {
    const table = compareModels(
      [
        { name: "complex", logLikelihood: -90, k: 6 },
        { name: "simple", logLikelihood: -100, k: 2 },
        { name: "medium", logLikelihood: -95, k: 4 },
      ],
      100,
    );

    // AIC: complex = 192, medium = 198, simple = 204
    expect(table[0].name).toBe("complex");
    expect(table[1].name).toBe("medium");
    expect(table[2].name).toBe("simple");
  });

  it("computes deltaAIC relative to best model", () => {
    const table = compareModels(
      [
        { name: "A", logLikelihood: -90, k: 3 },
        { name: "B", logLikelihood: -100, k: 3 },
      ],
      50,
    );

    expect(table[0].deltaAIC).toBe(0);
    expect(table[1].deltaAIC).toBeCloseTo(20, 10); // 2 * (100 - 90)
  });

  it("Akaike weights sum to 1", () => {
    const table = compareModels(
      [
        { name: "A", logLikelihood: -90, k: 3 },
        { name: "B", logLikelihood: -95, k: 3 },
        { name: "C", logLikelihood: -100, k: 3 },
      ],
      50,
    );

    const totalWeight = table.reduce((s, e) => s + e.weight, 0);
    expect(totalWeight).toBeCloseTo(1, 10);
  });

  it("best model has highest Akaike weight", () => {
    const table = compareModels(
      [
        { name: "A", logLikelihood: -90, k: 3 },
        { name: "B", logLikelihood: -100, k: 3 },
      ],
      50,
    );

    expect(table[0].weight).toBeGreaterThan(table[1].weight);
  });

  it("includes all information criteria", () => {
    const table = compareModels(
      [{ name: "only", logLikelihood: -100, k: 3 }],
      50,
    );

    expect(table[0].aic).toBeCloseTo(206, 10);
    expect(table[0].bic).toBeCloseTo(-2 * -100 + 3 * Math.log(50), 10);
    expect(table[0].aicc).toBeDefined();
    expect(table[0].weight).toBe(1);
    expect(table[0].deltaAIC).toBe(0);
  });

  it("throws for empty model list", () => {
    expect(() => compareModels([], 50)).toThrow("at least one model");
  });
});

// ── Integration: LRT with informationCriteria ─────────────────────────────

describe("integration", () => {
  it("LRT and IC agree on model preference for nested models", () => {
    const llRestricted = -150;
    const llFull = -130;
    const kRestricted = 3;
    const kFull = 6;
    const n = 200;

    // LRT should reject (big improvement in ll)
    const lrt = likelihoodRatioTest(llRestricted, llFull, kRestricted, kFull);
    expect(lrt.rejected).toBe(true);

    // AIC should also prefer the full model
    const icRestricted = informationCriteria(llRestricted, kRestricted, n);
    const icFull = informationCriteria(llFull, kFull, n);
    expect(icFull.aic).toBeLessThan(icRestricted.aic);
  });

  it("LRT and IC can disagree when improvement is marginal", () => {
    // Very slight improvement does not overcome parameter penalty in AIC
    const llRestricted = -100;
    const llFull = -98;
    const kRestricted = 2;
    const kFull = 8;
    const n = 50;

    const lrt = likelihoodRatioTest(llRestricted, llFull, kRestricted, kFull);
    // D = 4, df = 6, p is large -> not rejected
    expect(lrt.rejected).toBe(false);

    // AIC penalises heavily: full = 212, restricted = 204
    const icRestricted = informationCriteria(llRestricted, kRestricted, n);
    const icFull = informationCriteria(llFull, kFull, n);
    expect(icRestricted.aic).toBeLessThan(icFull.aic);
  });
});
