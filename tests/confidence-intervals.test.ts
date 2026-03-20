import {
  meanCI,
  twoSampleMeanCI,
  pairedMeanCI,
  proportionCI,
  twoProportionCI,
  linearRegressionCI,
  multipleRegressionCI,
} from "../src/confidence-intervals";

describe("meanCI", () => {
  it("computes a 95% CI for a sample mean", () => {
    // Data from a known example: 10 values with mean = 5, stdDev ~ 1.58
    const data = [3, 4, 4, 5, 5, 5, 6, 6, 6, 7];
    const ci = meanCI(data);

    expect(ci.confidenceLevel).toBe(0.95);
    expect(ci.estimate).toBeCloseTo(5.1, 1);
    expect(ci.lower).toBeLessThan(ci.estimate);
    expect(ci.upper).toBeGreaterThan(ci.estimate);
    expect(ci.marginOfError).toBeGreaterThan(0);
    expect(ci.upper - ci.lower).toBeCloseTo(2 * ci.marginOfError, 10);
  });

  it("produces a narrower interval at 90% confidence", () => {
    const data = [3, 4, 4, 5, 5, 5, 6, 6, 6, 7];
    const ci95 = meanCI(data, 0.95);
    const ci90 = meanCI(data, 0.90);

    expect(ci90.marginOfError).toBeLessThan(ci95.marginOfError);
  });

  it("produces a wider interval at 99% confidence", () => {
    const data = [3, 4, 4, 5, 5, 5, 6, 6, 6, 7];
    const ci95 = meanCI(data, 0.95);
    const ci99 = meanCI(data, 0.99);

    expect(ci99.marginOfError).toBeGreaterThan(ci95.marginOfError);
  });

  it("narrows with larger sample size", () => {
    const small = [2, 4, 6, 8, 10];
    const large = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 4, 5, 6, 7, 8];
    const ciSmall = meanCI(small);
    const ciLarge = meanCI(large);

    // Larger sample -> narrower CI (assuming similar variance)
    expect(ciLarge.marginOfError).toBeLessThan(ciSmall.marginOfError);
  });

  it("throws on fewer than 2 elements", () => {
    expect(() => meanCI([5])).toThrow();
  });

  it("throws on invalid confidence level", () => {
    expect(() => meanCI([1, 2, 3], 0)).toThrow();
    expect(() => meanCI([1, 2, 3], 1)).toThrow();
    expect(() => meanCI([1, 2, 3], 1.5)).toThrow();
    expect(() => meanCI([1, 2, 3], -0.1)).toThrow();
  });

  it("matches known t-interval computation", () => {
    // n=5, mean=10, s=2 => se = 2/sqrt(5) = 0.8944
    // t(4, 0.025) ≈ 2.776 => moe ≈ 2.483
    const data = [8, 9, 10, 11, 12]; // mean=10, s=Math.sqrt(2.5)
    const ci = meanCI(data);

    expect(ci.estimate).toBeCloseTo(10, 10);
    // t critical for df=4 at 97.5% ≈ 2.776
    const se = Math.sqrt(2.5 / 5);
    expect(ci.marginOfError).toBeCloseTo(2.776 * se, 1);
  });
});

describe("twoSampleMeanCI", () => {
  it("computes CI for difference of two means", () => {
    const data1 = [10, 12, 14, 16, 18];
    const data2 = [8, 9, 10, 11, 12];
    const ci = twoSampleMeanCI(data1, data2);

    expect(ci.confidenceLevel).toBe(0.95);
    // Mean1 = 14, Mean2 = 10, diff = 4
    expect(ci.estimate).toBeCloseTo(4);
    expect(ci.lower).toBeLessThan(ci.estimate);
    expect(ci.upper).toBeGreaterThan(ci.estimate);
  });

  it("contains zero when means are similar", () => {
    const data1 = [10, 11, 12, 13, 14];
    const data2 = [10, 11, 12, 13, 14];
    const ci = twoSampleMeanCI(data1, data2);

    expect(ci.estimate).toBeCloseTo(0);
    expect(ci.lower).toBeLessThanOrEqual(0);
    expect(ci.upper).toBeGreaterThanOrEqual(0);
  });

  it("throws with insufficient data", () => {
    expect(() => twoSampleMeanCI([1], [1, 2, 3])).toThrow();
    expect(() => twoSampleMeanCI([1, 2, 3], [1])).toThrow();
  });
});

describe("pairedMeanCI", () => {
  it("computes CI for paired differences", () => {
    const before = [200, 210, 220, 230, 240];
    const after = [190, 195, 205, 220, 230];
    const ci = pairedMeanCI(before, after);

    // Diffs: 10, 15, 15, 10, 10 => mean diff = 12
    expect(ci.estimate).toBeCloseTo(12);
    expect(ci.lower).toBeLessThan(ci.estimate);
    expect(ci.upper).toBeGreaterThan(ci.estimate);
  });

  it("throws when arrays have different lengths", () => {
    expect(() => pairedMeanCI([1, 2], [1, 2, 3])).toThrow();
  });
});

describe("proportionCI", () => {
  it("computes Wilson score CI for a proportion", () => {
    const ci = proportionCI(60, 100);

    expect(ci.confidenceLevel).toBe(0.95);
    expect(ci.estimate).toBeCloseTo(0.6);
    expect(ci.lower).toBeLessThan(0.6);
    expect(ci.upper).toBeGreaterThan(0.6);
    expect(ci.lower).toBeGreaterThan(0);
    expect(ci.upper).toBeLessThan(1);
  });

  it("handles proportion near 0", () => {
    const ci = proportionCI(1, 100);

    expect(ci.estimate).toBeCloseTo(0.01);
    // Wilson interval should not go below 0
    expect(ci.lower).toBeGreaterThanOrEqual(0);
  });

  it("handles proportion near 1", () => {
    const ci = proportionCI(99, 100);

    expect(ci.estimate).toBeCloseTo(0.99);
    // Wilson interval should not go above 1
    expect(ci.upper).toBeLessThanOrEqual(1);
  });

  it("handles proportion of 0", () => {
    const ci = proportionCI(0, 50);

    expect(ci.estimate).toBe(0);
    expect(ci.lower).toBeGreaterThanOrEqual(-1e-15);
  });

  it("handles proportion of 1", () => {
    const ci = proportionCI(50, 50);

    expect(ci.estimate).toBe(1);
    expect(ci.upper).toBeLessThanOrEqual(1);
  });

  it("throws on invalid inputs", () => {
    expect(() => proportionCI(-1, 100)).toThrow();
    expect(() => proportionCI(101, 100)).toThrow();
    expect(() => proportionCI(5, 0)).toThrow();
    expect(() => proportionCI(5.5, 10)).toThrow();
  });

  it("narrows with larger sample size", () => {
    const ciSmall = proportionCI(6, 10);
    const ciLarge = proportionCI(600, 1000);

    const widthSmall = ciSmall.upper - ciSmall.lower;
    const widthLarge = ciLarge.upper - ciLarge.lower;
    expect(widthLarge).toBeLessThan(widthSmall);
  });
});

describe("twoProportionCI", () => {
  it("computes CI for difference of two proportions", () => {
    const ci = twoProportionCI(60, 100, 40, 100);

    expect(ci.estimate).toBeCloseTo(0.2);
    expect(ci.lower).toBeLessThan(0.2);
    expect(ci.upper).toBeGreaterThan(0.2);
  });

  it("contains zero when proportions are equal", () => {
    const ci = twoProportionCI(50, 100, 50, 100);

    expect(ci.estimate).toBeCloseTo(0);
    expect(ci.lower).toBeLessThanOrEqual(0);
    expect(ci.upper).toBeGreaterThanOrEqual(0);
  });

  it("throws with invalid inputs", () => {
    expect(() => twoProportionCI(5, 0, 5, 10)).toThrow();
    expect(() => twoProportionCI(5, 10, 5, 0)).toThrow();
    expect(() => twoProportionCI(-1, 10, 5, 10)).toThrow();
  });
});

describe("linearRegressionCI", () => {
  it("computes CIs for slope and intercept", () => {
    const x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const y = [2.1, 3.9, 6.2, 7.8, 10.1, 12.0, 14.1, 15.8, 18.2, 19.9];
    const cis = linearRegressionCI(x, y);

    expect(cis).toHaveLength(2);
    expect(cis[0].name).toBe("intercept");
    expect(cis[1].name).toBe("slope");

    // Slope should be close to 2
    expect(cis[1].estimate).toBeCloseTo(2, 0);
    expect(cis[1].lower).toBeLessThan(cis[1].estimate);
    expect(cis[1].upper).toBeGreaterThan(cis[1].estimate);

    // Each CI should have valid structure
    for (const ci of cis) {
      expect(ci.standardError).toBeGreaterThan(0);
      expect(ci.pValue).toBeGreaterThanOrEqual(0);
      expect(ci.pValue).toBeLessThanOrEqual(1);
    }
  });

  it("slope CI excludes zero for significant relationship", () => {
    const x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]; // perfect y = 2x
    const cis = linearRegressionCI(x, y);

    // Slope CI should not contain zero
    expect(cis[1].lower).toBeGreaterThan(0);
    expect(cis[1].pValue).toBeLessThan(0.05);
  });

  it("slope CI contains zero for no relationship", () => {
    // Flat line with noise
    const x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const y = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]; // no relationship
    const cis = linearRegressionCI(x, y);

    expect(cis[1].estimate).toBeCloseTo(0);
  });

  it("throws with too few data points", () => {
    expect(() => linearRegressionCI([1, 2], [1, 2])).toThrow();
  });

  it("throws with mismatched lengths", () => {
    expect(() => linearRegressionCI([1, 2, 3], [1, 2])).toThrow();
  });
});

describe("multipleRegressionCI", () => {
  it("computes CIs for intercept and all coefficients", () => {
    // y ≈ 1 + 2*x1 + 3*x2
    const X = [
      [1, 1],
      [2, 1],
      [3, 2],
      [4, 2],
      [5, 3],
      [6, 3],
      [7, 4],
      [8, 4],
      [9, 5],
      [10, 5],
    ];
    const y = X.map((row) => 1 + 2 * row[0] + 3 * row[1] + (Math.random() - 0.5) * 0.01);
    const cis = multipleRegressionCI(X, y);

    expect(cis).toHaveLength(3);
    expect(cis[0].name).toBe("intercept");
    expect(cis[1].name).toBe("x1");
    expect(cis[2].name).toBe("x2");

    // Coefficients should be close to true values
    expect(cis[0].estimate).toBeCloseTo(1, 0);
    expect(cis[1].estimate).toBeCloseTo(2, 0);
    expect(cis[2].estimate).toBeCloseTo(3, 0);

    for (const ci of cis) {
      expect(ci.standardError).toBeGreaterThan(0);
      expect(ci.lower).toBeLessThan(ci.upper);
    }
  });

  it("uses custom feature names when provided", () => {
    const X = [
      [1, 2],
      [2, 3],
      [3, 1],
      [4, 5],
      [5, 4],
    ];
    const y = [5, 8, 6, 14, 13];
    const cis = multipleRegressionCI(X, y, 0.95, ["height", "weight"]);

    expect(cis[0].name).toBe("intercept");
    expect(cis[1].name).toBe("height");
    expect(cis[2].name).toBe("weight");
  });

  it("throws with insufficient observations", () => {
    const X = [[1, 2], [3, 4]];
    const y = [1, 2];
    expect(() => multipleRegressionCI(X, y)).toThrow();
  });

  it("throws with mismatched dimensions", () => {
    const X = [[1, 2], [3, 4], [5, 6]];
    const y = [1, 2];
    expect(() => multipleRegressionCI(X, y)).toThrow();
  });
});
