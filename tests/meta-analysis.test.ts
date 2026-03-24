import {
  fixedEffectsMeta,
  randomEffectsMeta,
  forestPlotData,
  funnelPlotData,
  eggersTest,
  beggsTest,
  trimAndFill,
} from "../src/meta-analysis";

const effects = [0.2, 0.3, 0.25, 0.35, 0.15];
const variances = [0.01, 0.02, 0.015, 0.01, 0.025];

describe("fixedEffectsMeta", () => {
  it("computes a pooled effect size", () => {
    const result = fixedEffectsMeta(effects, variances);
    expect(result.method).toBe("fixed");
    expect(result.pooledEffect).toBeGreaterThan(0);
    expect(result.pooledEffect).toBeLessThan(0.5);
    expect(result.standardError).toBeGreaterThan(0);
    expect(result.weights).toHaveLength(5);
  });

  it("confidence interval brackets the pooled effect", () => {
    const result = fixedEffectsMeta(effects, variances);
    expect(result.confidenceInterval.lower).toBeLessThan(result.pooledEffect);
    expect(result.confidenceInterval.upper).toBeGreaterThan(result.pooledEffect);
  });

  it("p-value is between 0 and 1", () => {
    const result = fixedEffectsMeta(effects, variances);
    expect(result.pValue).toBeGreaterThanOrEqual(0);
    expect(result.pValue).toBeLessThanOrEqual(1);
  });

  it("throws on empty arrays", () => {
    expect(() => fixedEffectsMeta([], [])).toThrow();
  });

  it("throws on mismatched array lengths", () => {
    expect(() => fixedEffectsMeta([0.1, 0.2], [0.01])).toThrow();
  });

  it("throws on non-positive variance", () => {
    expect(() => fixedEffectsMeta([0.1, 0.2], [0.01, -0.01])).toThrow();
  });
});

describe("randomEffectsMeta", () => {
  it("computes a random-effects pooled estimate", () => {
    const result = randomEffectsMeta(effects, variances);
    expect(result.method).toBe("random");
    expect(result.pooledEffect).toBeGreaterThan(0);
    expect(result.tau2).toBeGreaterThanOrEqual(0);
    expect(result.i2).toBeGreaterThanOrEqual(0);
    expect(result.i2).toBeLessThanOrEqual(100);
  });

  it("Cochran Q statistic and p-value are valid", () => {
    const result = randomEffectsMeta(effects, variances);
    expect(result.q).toBeGreaterThanOrEqual(0);
    expect(result.qPValue).toBeGreaterThanOrEqual(0);
    expect(result.qPValue).toBeLessThanOrEqual(1);
  });

  it("h2 is at least 1", () => {
    const result = randomEffectsMeta(effects, variances);
    expect(result.h2).toBeGreaterThanOrEqual(1);
  });
});

describe("forestPlotData", () => {
  it("generates forest plot data for all studies", () => {
    const result = forestPlotData(effects, variances, effects.map((_, i) => `Study ${i + 1}`));
    expect(result.studies).toHaveLength(5);
    expect(result.overall).toBeDefined();
    expect(result.overall.effect).toBeGreaterThan(0);
    for (const study of result.studies) {
      expect(study.lower).toBeLessThan(study.effect);
      expect(study.upper).toBeGreaterThan(study.effect);
      expect(study.weight).toBeGreaterThan(0);
    }
  });

  it("study labels are present", () => {
    const result = forestPlotData(effects, variances, effects.map((_, i) => `Study ${i + 1}`));
    for (const study of result.studies) {
      expect(typeof study.label).toBe("string");
    }
  });
});

describe("funnelPlotData", () => {
  it("generates funnel plot data", () => {
    const result = funnelPlotData(effects, variances);
    expect(result.points).toHaveLength(5);
    expect(Number.isFinite(result.pooledEffect)).toBe(true);
    for (const pt of result.points) {
      expect(pt.se).toBeGreaterThan(0);
    }
  });

  it("pseudo CI lines are provided", () => {
    const result = funnelPlotData(effects, variances);
    expect(result.pseudoCI.length).toBeGreaterThan(0);
    for (const ci of result.pseudoCI) {
      expect(ci.lower).toBeLessThanOrEqual(ci.upper);
    }
  });
});

describe("eggersTest", () => {
  it("returns a valid publication bias test result", () => {
    const result = eggersTest(effects, variances);
    expect(result.method).toContain("Egger");
    expect(Number.isFinite(result.statistic)).toBe(true);
    expect(result.pValue).toBeGreaterThanOrEqual(0);
    expect(result.pValue).toBeLessThanOrEqual(1);
  });
});

describe("beggsTest", () => {
  it("returns a valid rank correlation test result", () => {
    const result = beggsTest(effects, variances);
    expect(result.method).toContain("Begg");
    expect(Number.isFinite(result.statistic)).toBe(true);
    expect(result.pValue).toBeGreaterThanOrEqual(0);
    expect(result.pValue).toBeLessThanOrEqual(1);
  });
});

describe("trimAndFill", () => {
  it("returns adjusted estimates", () => {
    const result = trimAndFill(effects, variances);
    expect(Number.isFinite(result.originalPooled)).toBe(true);
    expect(Number.isFinite(result.adjustedPooled)).toBe(true);
    expect(result.nMissing).toBeGreaterThanOrEqual(0);
    expect(result.adjustedCI.lower).toBeLessThan(result.adjustedCI.upper);
  });

  it("filled effects include original effects", () => {
    const result = trimAndFill(effects, variances);
    expect(result.filledEffects.length).toBeGreaterThanOrEqual(effects.length);
    expect(result.filledVariances.length).toBeGreaterThanOrEqual(variances.length);
  });
});
