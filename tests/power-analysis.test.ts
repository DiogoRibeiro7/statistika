import {
  tTestPower,
  tTestSampleSize,
  oneSampleTTestPower,
  oneSampleTTestSampleSize,
  pairedTTestPower,
  pairedTTestSampleSize,
  anovaPower,
  anovaSampleSize,
  chiSquaredPower,
  chiSquaredSampleSize,
  proportionTestPower,
  proportionTestSampleSize,
  correlationPower,
  correlationSampleSize,
} from "../src/power-analysis";

// ====================================================================
//  TWO-SAMPLE T-TEST
// ====================================================================

describe("tTestPower", () => {
  it("returns high power for large effect + large n", () => {
    const result = tTestPower(0.8, 50);
    expect(result.power).toBeGreaterThan(0.95);
    expect(result.test).toBe("Two-sample t-test");
  });

  it("returns low power for small effect + small n", () => {
    const result = tTestPower(0.2, 10);
    expect(result.power).toBeLessThan(0.20);
  });

  it("power increases with sample size", () => {
    const p1 = tTestPower(0.5, 20);
    const p2 = tTestPower(0.5, 50);
    const p3 = tTestPower(0.5, 100);
    expect(p2.power).toBeGreaterThan(p1.power);
    expect(p3.power).toBeGreaterThan(p2.power);
  });

  it("power increases with effect size", () => {
    const p1 = tTestPower(0.2, 50);
    const p2 = tTestPower(0.5, 50);
    const p3 = tTestPower(0.8, 50);
    expect(p2.power).toBeGreaterThan(p1.power);
    expect(p3.power).toBeGreaterThan(p2.power);
  });

  it("one-tailed has more power than two-tailed", () => {
    const twoTailed = tTestPower(0.5, 30, 0.05, 2);
    const oneTailed = tTestPower(0.5, 30, 0.05, 1);
    expect(oneTailed.power).toBeGreaterThan(twoTailed.power);
  });

  // G*Power reference: d=0.5, n=64 per group, alpha=0.05, two-tailed => power ≈ 0.80
  it("matches G*Power reference for medium effect", () => {
    const result = tTestPower(0.5, 64);
    expect(result.power).toBeGreaterThan(0.75);
    expect(result.power).toBeLessThan(0.85);
  });

  it("power approaches alpha when effect size is 0", () => {
    const result = tTestPower(0, 100);
    expect(result.power).toBeCloseTo(0.05, 1);
  });

  it("throws on invalid inputs", () => {
    expect(() => tTestPower(0.5, 1)).toThrow(); // n < 2
    expect(() => tTestPower(0.5, 50, 0)).toThrow(); // alpha = 0
    expect(() => tTestPower(0.5, 50, 1)).toThrow(); // alpha = 1
  });
});

describe("tTestSampleSize", () => {
  // Classic: d=0.5, power=0.80, alpha=0.05 => n ≈ 64 per group
  it("returns ~64 per group for medium effect, 80% power", () => {
    const result = tTestSampleSize(0.5);
    expect(result.sampleSize).toBeGreaterThanOrEqual(60);
    expect(result.sampleSize).toBeLessThanOrEqual(70);
    expect(result.achievedPower).toBeGreaterThanOrEqual(0.80);
  });

  // d=0.2, power=0.80, alpha=0.05 => n ≈ 394
  it("returns large n for small effect", () => {
    const result = tTestSampleSize(0.2);
    expect(result.sampleSize).toBeGreaterThanOrEqual(380);
    expect(result.sampleSize).toBeLessThanOrEqual(410);
  });

  // d=0.8, power=0.80, alpha=0.05 => n ≈ 26
  it("returns small n for large effect", () => {
    const result = tTestSampleSize(0.8);
    expect(result.sampleSize).toBeGreaterThanOrEqual(24);
    expect(result.sampleSize).toBeLessThanOrEqual(30);
  });

  it("achieved power is >= target", () => {
    const result = tTestSampleSize(0.5, 0.90);
    expect(result.achievedPower).toBeGreaterThanOrEqual(0.90);
  });

  it("throws on zero effect size", () => {
    expect(() => tTestSampleSize(0)).toThrow();
  });
});

// ====================================================================
//  ONE-SAMPLE T-TEST
// ====================================================================

describe("oneSampleTTestPower", () => {
  it("returns high power for large effect + large n", () => {
    const result = oneSampleTTestPower(0.8, 30);
    expect(result.power).toBeGreaterThan(0.90);
  });

  it("power increases with n", () => {
    const p1 = oneSampleTTestPower(0.5, 10);
    const p2 = oneSampleTTestPower(0.5, 50);
    expect(p2.power).toBeGreaterThan(p1.power);
  });
});

describe("oneSampleTTestSampleSize", () => {
  // d=0.5, power=0.80 => n ≈ 34
  it("returns ~34 for medium effect, 80% power", () => {
    const result = oneSampleTTestSampleSize(0.5);
    expect(result.sampleSize).toBeGreaterThanOrEqual(30);
    expect(result.sampleSize).toBeLessThanOrEqual(38);
    expect(result.achievedPower).toBeGreaterThanOrEqual(0.80);
  });
});

// ====================================================================
//  PAIRED T-TEST
// ====================================================================

describe("pairedTTestPower", () => {
  it("delegates to one-sample with correct test name", () => {
    const result = pairedTTestPower(0.5, 30);
    expect(result.test).toBe("Paired t-test");
    expect(result.power).toBeGreaterThan(0.5);
  });
});

describe("pairedTTestSampleSize", () => {
  it("returns same n as one-sample", () => {
    const paired = pairedTTestSampleSize(0.5);
    const oneSample = oneSampleTTestSampleSize(0.5);
    expect(paired.sampleSize).toBe(oneSample.sampleSize);
    expect(paired.test).toBe("Paired t-test");
  });
});

// ====================================================================
//  ONE-WAY ANOVA
// ====================================================================

describe("anovaPower", () => {
  it("returns high power for large effect", () => {
    const result = anovaPower(0.4, 3, 30);
    expect(result.power).toBeGreaterThan(0.80);
  });

  it("power increases with n and effect size", () => {
    const p1 = anovaPower(0.25, 3, 20);
    const p2 = anovaPower(0.25, 3, 50);
    const p3 = anovaPower(0.40, 3, 50);
    expect(p2.power).toBeGreaterThan(p1.power);
    expect(p3.power).toBeGreaterThan(p2.power);
  });

  it("returns near-alpha power for zero effect", () => {
    const result = anovaPower(0, 3, 50);
    expect(result.power).toBeCloseTo(0.05, 1);
  });

  it("throws for k < 2", () => {
    expect(() => anovaPower(0.25, 1, 30)).toThrow();
  });
});

describe("anovaSampleSize", () => {
  // f=0.25, k=3, power=0.80 => n ≈ 53 per group (G*Power)
  it("returns ~53 per group for medium effect, 3 groups", () => {
    const result = anovaSampleSize(0.25, 3);
    expect(result.sampleSize).toBeGreaterThanOrEqual(45);
    expect(result.sampleSize).toBeLessThanOrEqual(60);
    expect(result.achievedPower).toBeGreaterThanOrEqual(0.80);
  });

  // f=0.40, k=4, power=0.80 => n ≈ 18 per group
  it("returns smaller n for large effect", () => {
    const result = anovaSampleSize(0.40, 4);
    expect(result.sampleSize).toBeGreaterThanOrEqual(14);
    expect(result.sampleSize).toBeLessThanOrEqual(22);
  });
});

// ====================================================================
//  CHI-SQUARED TEST
// ====================================================================

describe("chiSquaredPower", () => {
  it("returns high power for large w and n", () => {
    const result = chiSquaredPower(0.5, 1, 100);
    expect(result.power).toBeGreaterThan(0.90);
  });

  it("power increases with n", () => {
    const p1 = chiSquaredPower(0.3, 1, 30);
    const p2 = chiSquaredPower(0.3, 1, 100);
    expect(p2.power).toBeGreaterThan(p1.power);
  });

  it("returns near-alpha for zero effect", () => {
    const result = chiSquaredPower(0, 1, 100);
    expect(result.power).toBeCloseTo(0.05, 1);
  });
});

describe("chiSquaredSampleSize", () => {
  // w=0.3, df=1, power=0.80 => n ≈ 88 (G*Power)
  it("returns ~88 for medium effect, df=1", () => {
    const result = chiSquaredSampleSize(0.3, 1);
    expect(result.sampleSize).toBeGreaterThanOrEqual(80);
    expect(result.sampleSize).toBeLessThanOrEqual(100);
    expect(result.achievedPower).toBeGreaterThanOrEqual(0.80);
  });

  // w=0.1, df=4, power=0.80 => n ≈ 1091
  it("returns large n for small effect", () => {
    const result = chiSquaredSampleSize(0.1, 4);
    expect(result.sampleSize).toBeGreaterThanOrEqual(950);
    expect(result.sampleSize).toBeLessThanOrEqual(1200);
  });
});

// ====================================================================
//  TWO-PROPORTION Z-TEST
// ====================================================================

describe("proportionTestPower", () => {
  it("returns high power when proportions differ greatly", () => {
    const result = proportionTestPower(0.6, 0.3, 80);
    expect(result.power).toBeGreaterThan(0.90);
  });

  it("returns low power when proportions are close", () => {
    const result = proportionTestPower(0.51, 0.50, 30);
    expect(result.power).toBeLessThan(0.10);
  });

  it("power increases with n", () => {
    const p1 = proportionTestPower(0.6, 0.4, 30);
    const p2 = proportionTestPower(0.6, 0.4, 100);
    expect(p2.power).toBeGreaterThan(p1.power);
  });

  it("throws on invalid proportions", () => {
    expect(() => proportionTestPower(-0.1, 0.5, 50)).toThrow();
    expect(() => proportionTestPower(0.5, 1.1, 50)).toThrow();
  });
});

describe("proportionTestSampleSize", () => {
  // p1=0.6, p2=0.4, power=0.80 => n ≈ 97 per group
  it("returns reasonable n for p1=0.6, p2=0.4", () => {
    const result = proportionTestSampleSize(0.6, 0.4);
    expect(result.sampleSize).toBeGreaterThanOrEqual(85);
    expect(result.sampleSize).toBeLessThanOrEqual(110);
    expect(result.achievedPower).toBeGreaterThanOrEqual(0.80);
  });

  it("throws when proportions are equal", () => {
    expect(() => proportionTestSampleSize(0.5, 0.5)).toThrow();
  });
});

// ====================================================================
//  CORRELATION
// ====================================================================

describe("correlationPower", () => {
  it("returns high power for strong correlation + large n", () => {
    const result = correlationPower(0.5, 50);
    expect(result.power).toBeGreaterThan(0.90);
  });

  it("power increases with n", () => {
    const p1 = correlationPower(0.3, 20);
    const p2 = correlationPower(0.3, 80);
    expect(p2.power).toBeGreaterThan(p1.power);
  });

  it("works with negative correlation", () => {
    const pos = correlationPower(0.3, 50);
    const neg = correlationPower(-0.3, 50);
    // Power should be similar (symmetric)
    expect(Math.abs(pos.power - neg.power)).toBeLessThan(0.05);
  });
});

describe("correlationSampleSize", () => {
  // r=0.3, power=0.80 => n ≈ 85 (G*Power)
  it("returns ~85 for r=0.3, 80% power", () => {
    const result = correlationSampleSize(0.3);
    expect(result.sampleSize).toBeGreaterThanOrEqual(78);
    expect(result.sampleSize).toBeLessThanOrEqual(95);
    expect(result.achievedPower).toBeGreaterThanOrEqual(0.80);
  });

  // r=0.5, power=0.80 => n ≈ 29
  it("returns ~29 for r=0.5", () => {
    const result = correlationSampleSize(0.5);
    expect(result.sampleSize).toBeGreaterThanOrEqual(25);
    expect(result.sampleSize).toBeLessThanOrEqual(35);
  });

  it("throws on zero correlation", () => {
    expect(() => correlationSampleSize(0)).toThrow();
  });
});

// ====================================================================
//  ROUND-TRIP CONSISTENCY
// ====================================================================

describe("round-trip consistency", () => {
  it("tTest: power(sampleSize(d)) >= target", () => {
    const ss = tTestSampleSize(0.5, 0.80);
    const p = tTestPower(0.5, ss.sampleSize);
    expect(p.power).toBeGreaterThanOrEqual(0.80);
  });

  it("anova: power(sampleSize(f)) >= target", () => {
    const ss = anovaSampleSize(0.25, 3, 0.80);
    const p = anovaPower(0.25, 3, ss.sampleSize);
    expect(p.power).toBeGreaterThanOrEqual(0.80);
  });

  it("chiSquared: power(sampleSize(w)) >= target", () => {
    const ss = chiSquaredSampleSize(0.3, 1, 0.80);
    const p = chiSquaredPower(0.3, 1, ss.sampleSize);
    expect(p.power).toBeGreaterThanOrEqual(0.80);
  });

  it("correlation: power(sampleSize(r)) >= target", () => {
    const ss = correlationSampleSize(0.3, 0.80);
    const p = correlationPower(0.3, ss.sampleSize);
    expect(p.power).toBeGreaterThanOrEqual(0.80);
  });

  it("proportion: power(sampleSize(p1,p2)) >= target", () => {
    const ss = proportionTestSampleSize(0.6, 0.4, 0.80);
    const p = proportionTestPower(0.6, 0.4, ss.sampleSize);
    expect(p.power).toBeGreaterThanOrEqual(0.80);
  });
});
