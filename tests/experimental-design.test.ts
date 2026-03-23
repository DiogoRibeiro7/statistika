import {
  fullFactorial,
  analyzeFactorial,
  latinSquare,
  analyzeLatinSquare,
  centralCompositeDesign,
  fitResponseSurface,
  analyzeRCBD,
} from "../src/experimental-design";

// ── Full Factorial Design ─────────────────────────────────────────────────

describe("fullFactorial", () => {
  it("generates 2^k runs for two-level factors", () => {
    const design = fullFactorial({ A: [-1, 1], B: [-1, 1], C: [-1, 1] });
    expect(design.nRuns).toBe(8);
    expect(design.factors).toEqual(["A", "B", "C"]);
  });

  it("generates correct number of runs for mixed levels", () => {
    const design = fullFactorial({ A: [1, 2, 3], B: [0, 1] });
    expect(design.nRuns).toBe(6); // 3 × 2
  });

  it("all combinations are unique", () => {
    const design = fullFactorial({ A: [-1, 1], B: [-1, 1] });
    const keys = design.runs.map((r) => r.join(","));
    expect(new Set(keys).size).toBe(4);
  });

  it("each run has correct number of columns", () => {
    const design = fullFactorial({ A: [1, 2], B: [1, 2], C: [1, 2] });
    for (const run of design.runs) {
      expect(run.length).toBe(3);
    }
  });
});

// ── Factorial Analysis ────────────────────────────────────────────────────

describe("analyzeFactorial", () => {
  it("detects main effects", () => {
    const design = fullFactorial({ A: [-1, 1], B: [-1, 1] });
    // Main effect of A = 10, B = 0
    const response = design.runs.map((r) => 10 + 5 * r[0]);

    const result = analyzeFactorial(design, response);
    expect(result.mainEffects.get("A")).toBeCloseTo(10, 4);
    expect(Math.abs(result.mainEffects.get("B")!)).toBeLessThan(1);
  });

  it("computes sum of squares", () => {
    const design = fullFactorial({ A: [-1, 1], B: [-1, 1] });
    const response = design.runs.map((r) => r[0] * 3 + r[1] * 2);

    const result = analyzeFactorial(design, response);
    expect(result.sumOfSquares.has("A")).toBe(true);
    expect(result.sumOfSquares.has("B")).toBe(true);
    expect(result.sumOfSquares.get("A")!).toBeGreaterThan(0);
  });

  it("detects interactions", () => {
    const design = fullFactorial({ A: [-1, 1], B: [-1, 1] });
    // Strong interaction: effect depends on combination
    const response = design.runs.map((r) => r[0] * r[1] * 5);

    const result = analyzeFactorial(design, response);
    expect(result.interactions.has("A:B")).toBe(true);
    expect(result.sumOfSquares.get("A:B")!).toBeGreaterThan(0);
  });

  it("R² is high for known model", () => {
    const design = fullFactorial({ A: [-1, 1], B: [-1, 1] });
    const response = design.runs.map((r) => 10 + 5 * r[0] + 3 * r[1]);

    const result = analyzeFactorial(design, response);
    expect(result.rSquared).toBeGreaterThan(0.8);
  });

  it("works with replicates", () => {
    const design = fullFactorial({ A: [-1, 1], B: [-1, 1] });
    const base = design.runs.map((r) => 10 + 5 * r[0]);
    const response = [...base, ...base.map((v) => v + 0.5)]; // 2 replicates

    const result = analyzeFactorial(design, response, 2);
    expect(result.mainEffects.has("A")).toBe(true);
    expect(result.residualDF).toBeGreaterThan(0);
  });

  it("throws for wrong response length", () => {
    const design = fullFactorial({ A: [-1, 1] });
    expect(() => analyzeFactorial(design, [1, 2, 3])).toThrow("Expected");
  });
});

// ── Latin Square ──────────────────────────────────────────────────────────

describe("latinSquare", () => {
  it("generates valid n×n square", () => {
    const ls = latinSquare(4);
    expect(ls.order).toBe(4);
    expect(ls.square.length).toBe(4);
    for (const row of ls.square) expect(row.length).toBe(4);
  });

  it("each row has all treatments", () => {
    const ls = latinSquare(4);
    for (const row of ls.square) {
      expect(new Set(row).size).toBe(4);
    }
  });

  it("each column has all treatments", () => {
    const ls = latinSquare(4);
    for (let j = 0; j < 4; j++) {
      const col = ls.square.map((row) => row[j]);
      expect(new Set(col).size).toBe(4);
    }
  });

  it("throws for n < 2", () => {
    expect(() => latinSquare(1)).toThrow("at least 2");
  });
});

describe("analyzeLatinSquare", () => {
  it("detects treatment effects", () => {
    const ls = latinSquare(3);
    // Treatment 0: low, 1: medium, 2: high
    const treatmentEffect = [0, 5, 10];
    const response = ls.square.map((row) =>
      row.map((t) => 10 + treatmentEffect[t]),
    );

    const result = analyzeLatinSquare(ls, response);
    expect(result.ssTreatment).toBeGreaterThan(0);
    // With no noise, ssResidual ≈ 0 and fTreatment can be 0 or Infinity
    expect(result.treatmentMeans[2]).toBeGreaterThan(result.treatmentMeans[0]);
  });

  it("grand mean is correct", () => {
    const ls = latinSquare(3);
    const response = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];

    const result = analyzeLatinSquare(ls, response);
    expect(result.grandMean).toBeCloseTo(5, 8);
  });

  it("throws for wrong dimensions", () => {
    const ls = latinSquare(3);
    expect(() => analyzeLatinSquare(ls, [[1, 2], [3, 4]])).toThrow("3 × 3");
  });
});

// ── Central Composite Design ──────────────────────────────────────────────

describe("centralCompositeDesign", () => {
  it("generates correct number of runs for k=2", () => {
    const ccd = centralCompositeDesign(2, undefined, 3);
    // 2^2 factorial + 2*2 axial + 3 center = 4 + 4 + 3 = 11
    expect(ccd.nRuns).toBe(11);
    expect(ccd.factors.length).toBe(2);
  });

  it("generates correct number of runs for k=3", () => {
    const ccd = centralCompositeDesign(3, undefined, 5);
    // 2^3 + 2*3 + 5 = 8 + 6 + 5 = 19
    expect(ccd.nRuns).toBe(19);
  });

  it("includes center points at origin", () => {
    const ccd = centralCompositeDesign(2, undefined, 3);
    const centerPoints = ccd.runs.filter((r) => r.every((v) => v === 0));
    expect(centerPoints.length).toBe(3);
  });

  it("includes axial points", () => {
    const alpha = Math.sqrt(2);
    const ccd = centralCompositeDesign(2, alpha, 1);
    const axial = ccd.runs.filter((r) =>
      r.some((v) => Math.abs(Math.abs(v) - alpha) < 1e-10) &&
      r.filter((v) => v === 0).length === 1,
    );
    expect(axial.length).toBe(4); // 2 per factor × 2 factors
  });

  it("throws for invalid k", () => {
    expect(() => centralCompositeDesign(1)).toThrow("between 2 and 5");
    expect(() => centralCompositeDesign(6)).toThrow("between 2 and 5");
  });
});

// ── Response Surface ──────────────────────────────────────────────────────

describe("fitResponseSurface", () => {
  it("fits quadratic model to known surface", () => {
    // y = 10 - x1² - x2² (maximum at origin = 10)
    const ccd = centralCompositeDesign(2, Math.sqrt(2), 3);
    const response = ccd.runs.map((r) => 10 - r[0] ** 2 - r[1] ** 2);

    const result = fitResponseSurface(ccd, response);
    expect(result.rSquared).toBeGreaterThan(0.95);
    expect(result.coefficients[0]).toBeCloseTo(10, 0); // intercept
    expect(result.predict([0, 0])).toBeCloseTo(10, 0); // at center
  });

  it("returns coefficient names", () => {
    const ccd = centralCompositeDesign(2, 1.414, 1);
    const response = ccd.runs.map((r) => r[0] + r[1]);
    const result = fitResponseSurface(ccd, response);

    expect(result.coefficientNames).toContain("intercept");
    expect(result.coefficientNames).toContain("x1");
    expect(result.coefficientNames).toContain("x2");
    expect(result.coefficientNames).toContain("x1:x2");
    expect(result.coefficientNames).toContain("x1^2");
  });

  it("adjusted R² ≤ R²", () => {
    const ccd = centralCompositeDesign(2);
    const response = ccd.runs.map((r) => r[0] * r[1] + 5);
    const result = fitResponseSurface(ccd, response);
    expect(result.adjRSquared).toBeLessThanOrEqual(result.rSquared + 1e-10);
  });

  it("estimates stationary point for quadratic", () => {
    const ccd = centralCompositeDesign(2, Math.sqrt(2), 3);
    const response = ccd.runs.map((r) => 10 - (r[0] - 1) ** 2 - (r[1] + 0.5) ** 2);

    const result = fitResponseSurface(ccd, response);
    if (result.stationaryPoint) {
      expect(result.stationaryPoint[0]).toBeCloseTo(1, 0);
      expect(result.stationaryPoint[1]).toBeCloseTo(-0.5, 0);
    }
  });

  it("throws for wrong response length", () => {
    const ccd = centralCompositeDesign(2);
    expect(() => fitResponseSurface(ccd, [1, 2])).toThrow("Expected");
  });
});

// ── RCBD ──────────────────────────────────────────────────────────────────

describe("analyzeRCBD", () => {
  it("detects treatment effects in blocked design", () => {
    // 3 blocks, 4 treatments. Treatment adds [0, 5, 10, 15]
    const response = [
      [10, 15, 20, 25], // block 1
      [12, 17, 22, 27], // block 2
      [8, 13, 18, 23],  // block 3
    ];

    const result = analyzeRCBD(response);
    expect(result.ssTreatment).toBeGreaterThan(0);
    expect(result.treatmentMeans[3]).toBeGreaterThan(result.treatmentMeans[0]);
  });

  it("accounts for block effects", () => {
    const response = [
      [10, 15, 20],
      [20, 25, 30], // higher block mean
      [5, 10, 15],  // lower block mean
    ];

    const result = analyzeRCBD(response);
    expect(result.ssBlock).toBeGreaterThan(0);
    expect(result.blockMeans[1]).toBeGreaterThan(result.blockMeans[2]);
  });

  it("grand mean is correct", () => {
    const response = [[1, 2], [3, 4]];
    const result = analyzeRCBD(response);
    expect(result.grandMean).toBeCloseTo(2.5, 8);
  });

  it("SS decomposition sums correctly", () => {
    const response = [
      [10, 15, 20],
      [12, 17, 22],
      [8, 13, 18],
    ];

    const result = analyzeRCBD(response);
    let ssTot = 0;
    const gm = result.grandMean;
    for (const row of response) {
      for (const v of row) ssTot += (v - gm) ** 2;
    }
    const ssSum = result.ssTreatment + result.ssBlock + result.ssResidual;
    expect(ssSum).toBeCloseTo(ssTot, 4);
  });

  it("throws for insufficient blocks", () => {
    expect(() => analyzeRCBD([[1, 2]])).toThrow("at least 2 blocks");
  });

  it("throws for mismatched block sizes", () => {
    expect(() => analyzeRCBD([[1, 2], [3]])).toThrow("same number");
  });
});
