import {
  cohensD,
  glassDelta,
  hedgesG,
  pairedCohensD,
  etaSquared,
  partialEtaSquared,
  omegaSquared,
  cohensF,
  cramersV,
  phiCoefficient,
  oddsRatio,
  relativeRisk,
  pointBiserialR,
} from "../src/effect-size";
import { oneWayAnova } from "../src/tests/anova";

describe("cohensD", () => {
  it("returns ~0 for identical distributions", () => {
    const data1 = [1, 2, 3, 4, 5];
    const data2 = [1, 2, 3, 4, 5];
    const result = cohensD(data1, data2);

    expect(result.measure).toBe("Cohen's d");
    expect(result.value).toBeCloseTo(0);
    expect(result.interpretation).toBe("negligible");
  });

  it("computes a known effect size", () => {
    // group1 mean=10, group2 mean=8, pooled sd ≈ 1.58 => d ≈ 1.265
    const data1 = [8, 9, 10, 11, 12];
    const data2 = [6, 7, 8, 9, 10];
    const result = cohensD(data1, data2);

    expect(result.value).toBeCloseTo(1.265, 1);
    expect(result.interpretation).toBe("large");
  });

  it("returns a negative value when group1 < group2", () => {
    const data1 = [1, 2, 3, 4, 5];
    const data2 = [6, 7, 8, 9, 10];
    const result = cohensD(data1, data2);

    expect(result.value).toBeLessThan(0);
  });

  it("correctly classifies small effect", () => {
    // Shift of 0.3 sd
    const data1 = [10, 10.3, 9.7, 10.1, 9.9, 10.2, 10.15, 9.85, 10.05, 9.95];
    const data2 = [10, 10.3, 9.7, 10.1, 9.9, 10.2, 10.15, 9.85, 10.05, 9.95].map(
      (x) => x - 0.05,
    );
    const result = cohensD(data1, data2);

    expect(result.interpretation).toBe("small");
  });

  it("throws with insufficient data", () => {
    expect(() => cohensD([1], [1, 2, 3])).toThrow();
    expect(() => cohensD([1, 2, 3], [1])).toThrow();
  });
});

describe("glassDelta", () => {
  it("uses control group sd as denominator", () => {
    // Control: mean=5, sd=1. Treatment: mean=8, sd=3
    const control = [4, 5, 5, 5, 6];
    const treatment = [5, 7, 8, 9, 11];
    const result = glassDelta(treatment, control);

    const controlSD = Math.sqrt(
      control.reduce((s, x) => s + (x - 5) ** 2, 0) / (control.length - 1),
    );
    const expected = (8 - 5) / controlSD;
    expect(result.value).toBeCloseTo(expected, 1);
    expect(result.measure).toBe("Glass's delta");
  });
});

describe("hedgesG", () => {
  it("applies small-sample bias correction", () => {
    const data1 = [8, 9, 10, 11, 12];
    const data2 = [6, 7, 8, 9, 10];
    const d = cohensD(data1, data2);
    const g = hedgesG(data1, data2);

    // Hedges' g should have smaller absolute value than Cohen's d
    expect(Math.abs(g.value)).toBeLessThan(Math.abs(d.value));
    expect(g.measure).toBe("Hedges' g");
  });

  it("converges to Cohen's d for large samples", () => {
    const rng = (n: number, offset: number) =>
      Array.from({ length: n }, (_, i) => offset + (i % 10));
    const data1 = rng(200, 10);
    const data2 = rng(200, 8);
    const d = cohensD(data1, data2);
    const g = hedgesG(data1, data2);

    expect(Math.abs(d.value - g.value)).toBeLessThan(0.02);
  });
});

describe("pairedCohensD", () => {
  it("computes effect size from paired differences", () => {
    const before = [200, 210, 220, 230, 240];
    const after = [190, 195, 205, 220, 230];
    const result = pairedCohensD(before, after);

    expect(result.measure).toBe("Cohen's d (paired)");
    expect(result.value).toBeGreaterThan(0);
  });

  it("throws on mismatched lengths", () => {
    expect(() => pairedCohensD([1, 2], [1, 2, 3])).toThrow();
  });

  it("throws on too few elements", () => {
    expect(() => pairedCohensD([1], [2])).toThrow();
  });
});

describe("etaSquared", () => {
  it("computes eta-squared from ANOVA results", () => {
    const groups = [
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ];
    const anova = oneWayAnova(groups);
    const result = etaSquared(anova);

    expect(result.measure).toBe("Eta-squared");
    expect(result.value).toBeGreaterThan(0);
    expect(result.value).toBeLessThanOrEqual(1);
    // Large separation between groups
    expect(result.interpretation).toBe("large");
  });

  it("returns small value for overlapping groups", () => {
    const groups = [
      [5, 5.1, 4.9, 5.05, 4.95],
      [5, 5.1, 4.9, 5.05, 4.95],
    ];
    const anova = oneWayAnova(groups);
    const result = etaSquared(anova);

    expect(result.value).toBeLessThan(0.06);
  });
});

describe("partialEtaSquared", () => {
  it("equals eta-squared for one-way ANOVA", () => {
    const groups = [[1, 2, 3], [4, 5, 6]];
    const anova = oneWayAnova(groups);
    const eta2 = etaSquared(anova);
    const peta2 = partialEtaSquared(anova);

    expect(peta2.value).toBeCloseTo(eta2.value);
    expect(peta2.measure).toBe("Partial eta-squared");
  });
});

describe("omegaSquared", () => {
  it("is less biased than eta-squared", () => {
    const groups = [
      [1, 2, 3, 4, 5],
      [3, 4, 5, 6, 7],
    ];
    const anova = oneWayAnova(groups);
    const eta2 = etaSquared(anova);
    const omega2 = omegaSquared(anova);

    // Omega-squared is always ≤ eta-squared
    expect(omega2.value).toBeLessThanOrEqual(eta2.value);
    expect(omega2.measure).toBe("Omega-squared");
  });

  it("does not return negative values", () => {
    const groups = [
      [5, 5.01, 4.99, 5.005],
      [5, 5.01, 4.99, 5.005],
    ];
    const anova = oneWayAnova(groups);
    const omega2 = omegaSquared(anova);

    expect(omega2.value).toBeGreaterThanOrEqual(0);
  });
});

describe("cohensF", () => {
  it("computes Cohen's f from ANOVA results", () => {
    const groups = [
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ];
    const anova = oneWayAnova(groups);
    const result = cohensF(anova);

    expect(result.measure).toBe("Cohen's f");
    expect(result.value).toBeGreaterThan(0);
    expect(result.interpretation).toBe("large");
  });
});

describe("cramersV", () => {
  it("computes Cramér's V for a contingency table", () => {
    // Strong association
    const table = [
      [50, 10],
      [10, 50],
    ];
    const result = cramersV(table);

    expect(result.measure).toBe("Cramér's V");
    expect(result.value).toBeGreaterThan(0.5);
    expect(result.value).toBeLessThanOrEqual(1);
  });

  it("returns ~0 for no association", () => {
    const table = [
      [25, 25],
      [25, 25],
    ];
    const result = cramersV(table);

    expect(result.value).toBeCloseTo(0, 1);
    expect(result.interpretation).toBe("negligible");
  });

  it("works with larger tables", () => {
    const table = [
      [30, 10, 5],
      [5, 30, 10],
      [5, 5, 30],
    ];
    const result = cramersV(table);

    expect(result.value).toBeGreaterThan(0);
    expect(result.value).toBeLessThanOrEqual(1);
  });

  it("throws on invalid table dimensions", () => {
    expect(() => cramersV([[1, 2]])).toThrow();
    expect(() => cramersV([[1], [2]])).toThrow();
  });

  it("throws on empty table", () => {
    expect(() =>
      cramersV([
        [0, 0],
        [0, 0],
      ]),
    ).toThrow();
  });
});

describe("phiCoefficient", () => {
  it("computes phi for a 2×2 table", () => {
    const table = [
      [40, 10],
      [10, 40],
    ];
    const result = phiCoefficient(table);

    expect(result.measure).toBe("Phi coefficient");
    expect(result.value).toBeGreaterThan(0.5);
  });

  it("returns negative phi for inverse association", () => {
    const table = [
      [10, 40],
      [40, 10],
    ];
    const result = phiCoefficient(table);

    expect(result.value).toBeLessThan(0);
  });

  it("throws for non-2×2 tables", () => {
    expect(() =>
      phiCoefficient([
        [1, 2, 3],
        [4, 5, 6],
      ]),
    ).toThrow();
  });
});

describe("oddsRatio", () => {
  it("computes odds ratio for a 2×2 table", () => {
    const table = [
      [30, 10],
      [10, 50],
    ];
    const result = oddsRatio(table);

    // OR = (30*50)/(10*10) = 15
    expect(result.oddsRatio).toBeCloseTo(15);
    expect(result.logOddsRatio).toBeCloseTo(Math.log(15));
    expect(result.lower).toBeLessThan(result.oddsRatio);
    expect(result.upper).toBeGreaterThan(result.oddsRatio);
    expect(result.confidenceLevel).toBe(0.95);
  });

  it("returns OR=1 for no association", () => {
    const table = [
      [25, 25],
      [25, 25],
    ];
    const result = oddsRatio(table);

    expect(result.oddsRatio).toBeCloseTo(1);
    expect(result.lower).toBeLessThan(1);
    expect(result.upper).toBeGreaterThan(1);
  });

  it("handles zero cells with Haldane-Anscombe correction", () => {
    const table = [
      [10, 0],
      [5, 10],
    ];
    const result = oddsRatio(table);

    expect(result.oddsRatio).toBeGreaterThan(1);
    expect(isFinite(result.oddsRatio)).toBe(true);
  });

  it("supports custom confidence levels", () => {
    const table = [
      [30, 10],
      [10, 50],
    ];
    const ci90 = oddsRatio(table, 0.90);
    const ci99 = oddsRatio(table, 0.99);

    expect(ci99.upper - ci99.lower).toBeGreaterThan(ci90.upper - ci90.lower);
  });

  it("throws on invalid table", () => {
    expect(() => oddsRatio([[1, 2]])).toThrow();
  });

  it("throws on negative counts", () => {
    expect(() =>
      oddsRatio([
        [-1, 2],
        [3, 4],
      ]),
    ).toThrow();
  });
});

describe("relativeRisk", () => {
  it("computes relative risk for a 2×2 table", () => {
    const table = [
      [30, 70],
      [10, 90],
    ];
    const result = relativeRisk(table);

    // RR = (30/100) / (10/100) = 3
    expect(result.oddsRatio).toBeCloseTo(3);
    expect(result.lower).toBeLessThan(3);
    expect(result.upper).toBeGreaterThan(3);
  });

  it("returns RR≈1 for no association", () => {
    const table = [
      [20, 80],
      [20, 80],
    ];
    const result = relativeRisk(table);

    expect(result.oddsRatio).toBeCloseTo(1);
  });

  it("throws when baseline risk is zero", () => {
    expect(() =>
      relativeRisk([
        [5, 10],
        [0, 15],
      ]),
    ).toThrow();
  });

  it("throws on invalid table dimensions", () => {
    expect(() => relativeRisk([[1, 2]])).toThrow();
  });
});

describe("pointBiserialR", () => {
  it("computes r from a t-statistic", () => {
    // t=3, df=20 => r = sqrt(9/29) ≈ 0.557
    const result = pointBiserialR(3, 20);

    expect(result.measure).toBe("Point-biserial r");
    expect(result.value).toBeCloseTo(Math.sqrt(9 / 29), 5);
    expect(result.interpretation).toBe("large");
  });

  it("returns negative r for negative t", () => {
    const result = pointBiserialR(-2.5, 30);
    expect(result.value).toBeLessThan(0);
  });

  it("returns ~0 for t≈0", () => {
    const result = pointBiserialR(0.01, 100);
    expect(Math.abs(result.value)).toBeLessThan(0.01);
    expect(result.interpretation).toBe("negligible");
  });

  it("throws with non-positive df", () => {
    expect(() => pointBiserialR(2, 0)).toThrow();
    expect(() => pointBiserialR(2, -1)).toThrow();
  });
});
