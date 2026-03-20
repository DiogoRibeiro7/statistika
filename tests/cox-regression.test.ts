import { coxRegression, CoxObservation } from "../src/cox-regression";

describe("Cox Proportional Hazards", () => {
  // Simple dataset: higher covariate value => faster event
  const observations: CoxObservation[] = [
    { time: 1, event: true, covariates: [2.0] },
    { time: 2, event: true, covariates: [1.8] },
    { time: 3, event: false, covariates: [1.5] },
    { time: 4, event: true, covariates: [1.2] },
    { time: 5, event: true, covariates: [0.8] },
    { time: 6, event: false, covariates: [0.5] },
    { time: 7, event: true, covariates: [0.3] },
    { time: 8, event: true, covariates: [0.1] },
    { time: 9, event: false, covariates: [0.0] },
    { time: 10, event: true, covariates: [-0.2] },
  ];

  it("returns the correct structure", () => {
    const result = coxRegression(observations);
    expect(result.coefficients.length).toBe(1);
    expect(result.standardErrors.length).toBe(1);
    expect(result.hazardRatios.length).toBe(1);
    expect(result.zScores.length).toBe(1);
    expect(result.pValues.length).toBe(1);
    expect(result.hazardRatioCIs.length).toBe(1);
    expect(typeof result.logLikelihood).toBe("number");
    expect(typeof result.concordance).toBe("number");
    expect(result.baselineHazard.length).toBeGreaterThan(0);
  });

  it("positive coefficient means higher risk", () => {
    const result = coxRegression(observations);
    // Higher covariate values lead to earlier events
    expect(result.coefficients[0]).toBeGreaterThan(0);
    expect(result.hazardRatios[0]).toBeGreaterThan(1);
  });

  it("hazard ratio = exp(coefficient)", () => {
    const result = coxRegression(observations);
    expect(result.hazardRatios[0]).toBeCloseTo(
      Math.exp(result.coefficients[0]),
      8,
    );
  });

  it("hazard ratio CI contains the point estimate", () => {
    const result = coxRegression(observations);
    expect(result.hazardRatioCIs[0][0]).toBeLessThan(result.hazardRatios[0]);
    expect(result.hazardRatioCIs[0][1]).toBeGreaterThan(result.hazardRatios[0]);
  });

  it("concordance index is between 0 and 1", () => {
    const result = coxRegression(observations);
    expect(result.concordance).toBeGreaterThanOrEqual(0);
    expect(result.concordance).toBeLessThanOrEqual(1);
  });

  it("concordance is > 0.5 for a meaningful model", () => {
    const result = coxRegression(observations);
    expect(result.concordance).toBeGreaterThan(0.5);
  });

  it("predictHazardRatio works", () => {
    const result = coxRegression(observations);
    const hr = result.predictHazardRatio([1.0]);
    expect(hr).toBeGreaterThan(0);
    expect(hr).toBeCloseTo(Math.exp(result.coefficients[0] * 1.0), 8);
  });

  it("baseline hazard is non-decreasing", () => {
    const result = coxRegression(observations);
    for (let i = 1; i < result.baselineHazard.length; i++) {
      expect(result.baselineHazard[i].hazard).toBeGreaterThanOrEqual(
        result.baselineHazard[i - 1].hazard,
      );
    }
  });

  it("p-values are between 0 and 1", () => {
    const result = coxRegression(observations);
    for (const pv of result.pValues) {
      expect(pv).toBeGreaterThanOrEqual(0);
      expect(pv).toBeLessThanOrEqual(1);
    }
  });

  it("works with multiple covariates", () => {
    const multiObs: CoxObservation[] = [
      { time: 1, event: true, covariates: [2.0, 1] },
      { time: 2, event: true, covariates: [1.5, 0] },
      { time: 3, event: true, covariates: [1.0, 1] },
      { time: 4, event: false, covariates: [0.5, 0] },
      { time: 5, event: true, covariates: [0.2, 1] },
      { time: 6, event: true, covariates: [0.0, 0] },
      { time: 7, event: false, covariates: [-0.3, 1] },
      { time: 8, event: true, covariates: [-0.5, 0] },
    ];
    const result = coxRegression(multiObs);
    expect(result.coefficients.length).toBe(2);
    expect(result.hazardRatios.length).toBe(2);
  });

  it("throws on too few observations", () => {
    expect(() =>
      coxRegression([{ time: 1, event: true, covariates: [1] }]),
    ).toThrow();
  });

  it("throws on empty covariates", () => {
    expect(() =>
      coxRegression([
        { time: 1, event: true, covariates: [] },
        { time: 2, event: true, covariates: [] },
      ]),
    ).toThrow();
  });

  it("throws on mismatched covariate lengths", () => {
    expect(() =>
      coxRegression([
        { time: 1, event: true, covariates: [1, 2] },
        { time: 2, event: true, covariates: [1] },
      ]),
    ).toThrow();
  });
});
