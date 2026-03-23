import { pathAnalysis, cfa, computeFitIndices } from "../src/sem";

// ── Path Analysis ─────────────────────────────────────────────────────────

describe("pathAnalysis", () => {
  // Simple mediation: X → M → Y and X → Y
  const n = 100;
  const xData: number[] = [];
  const mData: number[] = [];
  const yData: number[] = [];
  for (let i = 0; i < n; i++) {
    const x = (i - 50) / 10;
    const m = 0.6 * x + (i % 3 - 1) * 0.1;
    const y = 0.3 * x + 0.5 * m + (i % 5 - 2) * 0.1;
    xData.push(x);
    mData.push(m);
    yData.push(y);
  }

  it("estimates direct effects", () => {
    const result = pathAnalysis(
      { X: xData, M: mData, Y: yData },
      [
        { from: "X", to: "M" },
        { from: "X", to: "Y" },
        { from: "M", to: "Y" },
      ],
    );

    // X → M should be positive
    const xToM = result.directEffects.get("X")?.get("M");
    expect(xToM).toBeDefined();
    expect(xToM!).toBeGreaterThan(0);

    // M → Y should be positive
    const mToY = result.directEffects.get("M")?.get("Y");
    expect(mToY).toBeDefined();
    expect(mToY!).toBeGreaterThan(0);
  });

  it("computes total effects including indirect", () => {
    const result = pathAnalysis(
      { X: xData, M: mData, Y: yData },
      [
        { from: "X", to: "M" },
        { from: "X", to: "Y" },
        { from: "M", to: "Y" },
      ],
    );

    // Total effect of X on Y should be larger than direct
    const totalXY = result.totalEffects.get("X")?.get("Y");
    const directXY = result.directEffects.get("X")?.get("Y");
    expect(totalXY).toBeDefined();
    expect(directXY).toBeDefined();
    // Total = direct + indirect (through M)
    expect(totalXY!).toBeGreaterThan(directXY! - 0.01);
  });

  it("computes R² for endogenous variables", () => {
    const result = pathAnalysis(
      { X: xData, M: mData, Y: yData },
      [
        { from: "X", to: "M" },
        { from: "X", to: "Y" },
        { from: "M", to: "Y" },
      ],
    );

    const r2M = result.rSquared.get("M");
    const r2Y = result.rSquared.get("Y");
    expect(r2M).toBeDefined();
    expect(r2M!).toBeGreaterThan(0.5);
    expect(r2Y).toBeDefined();
    expect(r2Y!).toBeGreaterThan(0.5);
  });

  it("stores coefficients with path names", () => {
    const result = pathAnalysis(
      { X: xData, M: mData, Y: yData },
      [{ from: "X", to: "M" }, { from: "M", to: "Y" }],
    );

    expect(result.coefficients.has("X -> M")).toBe(true);
    expect(result.coefficients.has("M -> Y")).toBe(true);
  });

  it("supports fixed coefficients", () => {
    const result = pathAnalysis(
      { X: xData, M: mData, Y: yData },
      [{ from: "X", to: "M", coefficient: 1.0 }],
    );

    const xToM = result.directEffects.get("X")?.get("M");
    expect(xToM).toBeCloseTo(1.0, 8);
  });
});

// ── Confirmatory Factor Analysis ──────────────────────────────────────────

describe("cfa", () => {
  // Generate 2-factor data:
  //   Factor 1 → ind1, ind2, ind3
  //   Factor 2 → ind4, ind5, ind6
  const n = 200;
  const data: Record<string, number[]> = {
    ind1: [], ind2: [], ind3: [],
    ind4: [], ind5: [], ind6: [],
  };
  for (let i = 0; i < n; i++) {
    const f1 = (i - 100) / 50;
    const f2 = ((i * 7) % 200 - 100) / 50;
    data.ind1.push(0.8 * f1 + (i % 3 - 1) * 0.2);
    data.ind2.push(0.7 * f1 + (i % 5 - 2) * 0.2);
    data.ind3.push(0.6 * f1 + (i % 7 - 3) * 0.2);
    data.ind4.push(0.9 * f2 + (i % 4 - 1.5) * 0.2);
    data.ind5.push(0.7 * f2 + (i % 6 - 2.5) * 0.2);
    data.ind6.push(0.5 * f2 + (i % 3 - 1) * 0.3);
  }

  const loadingSpec = [
    { factor: "F1", indicator: "ind1" },
    { factor: "F1", indicator: "ind2" },
    { factor: "F1", indicator: "ind3" },
    { factor: "F2", indicator: "ind4" },
    { factor: "F2", indicator: "ind5" },
    { factor: "F2", indicator: "ind6" },
  ];

  it("returns loadings for each factor", () => {
    const result = cfa(data, loadingSpec);
    expect(result.loadings.has("F1")).toBe(true);
    expect(result.loadings.has("F2")).toBe(true);
    expect(result.loadings.get("F1")!.size).toBe(3);
    expect(result.loadings.get("F2")!.size).toBe(3);
  });

  it("loadings are non-zero", () => {
    const result = cfa(data, loadingSpec);
    for (const [, factorLoadings] of result.loadings) {
      for (const [, loading] of factorLoadings) {
        expect(Math.abs(loading)).toBeGreaterThan(0.1);
      }
    }
  });

  it("communalities are in [0, 1]", () => {
    const result = cfa(data, loadingSpec);
    for (const [, h2] of result.communalities) {
      expect(h2).toBeGreaterThanOrEqual(0);
      expect(h2).toBeLessThanOrEqual(1);
    }
  });

  it("uniquenesses = 1 - communalities", () => {
    const result = cfa(data, loadingSpec);
    for (const [ind, h2] of result.communalities) {
      expect(result.uniquenesses.get(ind)!).toBeCloseTo(1 - h2, 8);
    }
  });

  it("returns fit indices", () => {
    const result = cfa(data, loadingSpec);
    const { fit } = result;
    expect(fit.chiSquared).toBeGreaterThanOrEqual(0);
    expect(fit.df).toBeGreaterThan(0);
    expect(fit.rmsea).toBeGreaterThanOrEqual(0);
    expect(fit.cfi).toBeGreaterThanOrEqual(0);
    expect(fit.cfi).toBeLessThanOrEqual(1);
    expect(fit.srmr).toBeGreaterThanOrEqual(0);
    expect(Number.isFinite(fit.aic)).toBe(true);
    expect(Number.isFinite(fit.bic)).toBe(true);
  });

  it("SRMR is small for well-fitting model", () => {
    const result = cfa(data, loadingSpec);
    expect(result.fit.srmr).toBeLessThan(0.5);
  });

  it("throws for missing indicator", () => {
    expect(() =>
      cfa({}, [{ factor: "F", indicator: "missing" }]),
    ).toThrow("not found");
  });
});

// ── Fit Indices ───────────────────────────────────────────────────────────

describe("computeFitIndices", () => {
  it("perfect fit gives SRMR = 0", () => {
    const R = [[1, 0.5], [0.5, 1]];
    const implied = [[1, 0.5], [0.5, 1]]; // exact match
    const fit = computeFitIndices(R, implied, 100, 2, 1, 1);
    expect(fit.srmr).toBeCloseTo(0, 8);
  });

  it("chi-squared is non-negative", () => {
    const R = [[1, 0.3], [0.3, 1]];
    const implied = [[1, 0.5], [0.5, 1]];
    const fit = computeFitIndices(R, implied, 100, 2, 1, 1);
    expect(fit.chiSquared).toBeGreaterThanOrEqual(0);
  });

  it("p-value is in [0, 1]", () => {
    const R = [[1, 0.3], [0.3, 1]];
    const implied = [[1, 0.3], [0.3, 1]];
    const fit = computeFitIndices(R, implied, 100, 2, 1, 1);
    expect(fit.pValue).toBeGreaterThanOrEqual(0);
    expect(fit.pValue).toBeLessThanOrEqual(1);
  });
});
