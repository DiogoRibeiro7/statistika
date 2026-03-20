import {
  kaplanMeier,
  nelsonAalen,
  logRankTest,
  SurvivalObservation,
} from "../src/survival";

// ====================================================================
//  KAPLAN-MEIER
// ====================================================================

describe("kaplanMeier", () => {
  it("computes correct survival for simple uncensored data", () => {
    // 5 subjects, all die at different times
    const obs: SurvivalObservation[] = [
      { time: 1, event: true },
      { time: 2, event: true },
      { time: 3, event: true },
      { time: 4, event: true },
      { time: 5, event: true },
    ];
    const result = kaplanMeier(obs);

    expect(result.n).toBe(5);
    expect(result.nEvents).toBe(5);
    expect(result.nCensored).toBe(0);

    // S(1) = 4/5, S(2) = 3/5, S(3) = 2/5, S(4) = 1/5, S(5) = 0
    expect(result.curve[0].survival).toBeCloseTo(0.8);
    expect(result.curve[1].survival).toBeCloseTo(0.6);
    expect(result.curve[2].survival).toBeCloseTo(0.4);
    expect(result.curve[3].survival).toBeCloseTo(0.2);
    expect(result.curve[4].survival).toBeCloseTo(0.0);
  });

  it("handles right censoring correctly", () => {
    const obs: SurvivalObservation[] = [
      { time: 1, event: true },
      { time: 2, event: false }, // censored
      { time: 3, event: true },
      { time: 4, event: false }, // censored
      { time: 5, event: true },
    ];
    const result = kaplanMeier(obs);

    expect(result.nEvents).toBe(3);
    expect(result.nCensored).toBe(2);

    // At t=1: n=5, d=1, S = 4/5 = 0.8
    expect(result.curve[0].survival).toBeCloseTo(0.8);
    expect(result.curve[0].nRisk).toBe(5);

    // At t=3: n=3 (removed t=1 event and t=2 censor), d=1, S = 0.8 * 2/3 ≈ 0.533
    expect(result.curve[1].survival).toBeCloseTo(0.8 * (2 / 3), 2);
    expect(result.curve[1].nRisk).toBe(3);
  });

  it("computes median survival", () => {
    const obs: SurvivalObservation[] = [
      { time: 1, event: true },
      { time: 2, event: true },
      { time: 3, event: true },
      { time: 4, event: true },
    ];
    const result = kaplanMeier(obs);

    // S(1)=0.75, S(2)=0.5, S(3)=0.25 => median at t=2
    expect(result.medianSurvival).toBe(2);
  });

  it("returns NaN for median when not reached", () => {
    const obs: SurvivalObservation[] = [
      { time: 1, event: true },
      { time: 2, event: false },
      { time: 3, event: false },
      { time: 4, event: false },
    ];
    const result = kaplanMeier(obs);
    expect(isNaN(result.medianSurvival)).toBe(true);
  });

  it("survivalAt function interpolates correctly", () => {
    const obs: SurvivalObservation[] = [
      { time: 2, event: true },
      { time: 4, event: true },
      { time: 6, event: true },
      { time: 8, event: true },
    ];
    const result = kaplanMeier(obs);

    expect(result.survivalAt(0)).toBe(1); // before any event
    expect(result.survivalAt(1)).toBe(1); // before first event
    expect(result.survivalAt(2)).toBeCloseTo(0.75); // at first event
    expect(result.survivalAt(3)).toBeCloseTo(0.75); // between events
    expect(result.survivalAt(4)).toBeCloseTo(0.5);
    expect(result.survivalAt(100)).toBeCloseTo(0); // way after last event
  });

  it("handles simultaneous events", () => {
    const obs: SurvivalObservation[] = [
      { time: 1, event: true },
      { time: 1, event: true },
      { time: 2, event: true },
      { time: 2, event: true },
    ];
    const result = kaplanMeier(obs);

    // At t=1: n=4, d=2, S = 2/4 = 0.5
    expect(result.curve[0].survival).toBeCloseTo(0.5);
    expect(result.curve[0].nEvents).toBe(2);

    // At t=2: n=2, d=2, S = 0
    expect(result.curve[1].survival).toBeCloseTo(0);
  });

  it("provides confidence intervals", () => {
    const obs: SurvivalObservation[] = Array.from({ length: 20 }, (_, i) => ({
      time: i + 1,
      event: true,
    }));
    const result = kaplanMeier(obs, 0.95);

    for (const point of result.curve) {
      if (point.survival > 0 && point.survival < 1) {
        expect(point.lower).toBeLessThanOrEqual(point.survival);
        expect(point.upper).toBeGreaterThanOrEqual(point.survival);
        expect(point.standardError).toBeGreaterThanOrEqual(0);
      }
    }
  });

  it("SE is zero when n=1 at event time", () => {
    const obs: SurvivalObservation[] = [{ time: 1, event: true }];
    const result = kaplanMeier(obs);

    expect(result.curve[0].survival).toBe(0);
  });

  it("handles all censored data", () => {
    const obs: SurvivalObservation[] = [
      { time: 1, event: false },
      { time: 2, event: false },
      { time: 3, event: false },
    ];
    const result = kaplanMeier(obs);

    expect(result.nEvents).toBe(0);
    expect(result.curve.length).toBe(0);
    expect(result.survivalAt(10)).toBe(1);
  });

  it("throws on empty input", () => {
    expect(() => kaplanMeier([])).toThrow();
  });
});

// ====================================================================
//  NELSON-AALEN
// ====================================================================

describe("nelsonAalen", () => {
  it("computes cumulative hazard for simple data", () => {
    const obs: SurvivalObservation[] = [
      { time: 1, event: true },
      { time: 2, event: true },
      { time: 3, event: true },
      { time: 4, event: true },
    ];
    const result = nelsonAalen(obs);

    expect(result.n).toBe(4);
    expect(result.nEvents).toBe(4);

    // H(1) = 1/4, H(2) = 1/4 + 1/3, H(3) = 1/4 + 1/3 + 1/2
    expect(result.curve[0].hazard).toBeCloseTo(1 / 4);
    expect(result.curve[1].hazard).toBeCloseTo(1 / 4 + 1 / 3);
    expect(result.curve[2].hazard).toBeCloseTo(1 / 4 + 1 / 3 + 1 / 2);
  });

  it("hazard is monotonically increasing", () => {
    const obs: SurvivalObservation[] = [
      { time: 1, event: true },
      { time: 2, event: false },
      { time: 3, event: true },
      { time: 4, event: true },
      { time: 5, event: false },
      { time: 6, event: true },
    ];
    const result = nelsonAalen(obs);

    for (let i = 1; i < result.curve.length; i++) {
      expect(result.curve[i].hazard).toBeGreaterThan(
        result.curve[i - 1].hazard,
      );
    }
  });

  it("hazardAt function works", () => {
    const obs: SurvivalObservation[] = [
      { time: 2, event: true },
      { time: 4, event: true },
      { time: 6, event: true },
    ];
    const result = nelsonAalen(obs);

    expect(result.hazardAt(0)).toBe(0);
    expect(result.hazardAt(1)).toBe(0);
    expect(result.hazardAt(2)).toBeCloseTo(1 / 3);
    expect(result.hazardAt(3)).toBeCloseTo(1 / 3); // between events
    expect(result.hazardAt(4)).toBeCloseTo(1 / 3 + 1 / 2);
  });

  it("provides standard errors", () => {
    const obs: SurvivalObservation[] = Array.from({ length: 10 }, (_, i) => ({
      time: i + 1,
      event: true,
    }));
    const result = nelsonAalen(obs);

    for (const point of result.curve) {
      expect(point.standardError).toBeGreaterThan(0);
    }
  });

  it("throws on empty input", () => {
    expect(() => nelsonAalen([])).toThrow();
  });
});

// ====================================================================
//  LOG-RANK TEST
// ====================================================================

describe("logRankTest", () => {
  it("rejects when survival curves clearly differ", () => {
    // Group 1: early deaths
    const group1: SurvivalObservation[] = [
      { time: 1, event: true },
      { time: 2, event: true },
      { time: 3, event: true },
      { time: 4, event: true },
      { time: 5, event: true },
      { time: 1, event: true },
      { time: 2, event: true },
      { time: 3, event: true },
    ];
    // Group 2: late deaths
    const group2: SurvivalObservation[] = [
      { time: 10, event: true },
      { time: 11, event: true },
      { time: 12, event: true },
      { time: 13, event: true },
      { time: 14, event: true },
      { time: 10, event: true },
      { time: 11, event: true },
      { time: 12, event: true },
    ];

    const result = logRankTest(group1, group2);

    expect(result.rejected).toBe(true);
    expect(result.pValue).toBeLessThan(0.05);
    expect(result.degreesOfFreedom).toBe(1);
    expect(result.chiSquared).toBeGreaterThan(3.84);
  });

  it("does not reject when survival curves are similar", () => {
    const group1: SurvivalObservation[] = [
      { time: 1, event: true },
      { time: 3, event: true },
      { time: 5, event: true },
      { time: 7, event: true },
    ];
    const group2: SurvivalObservation[] = [
      { time: 2, event: true },
      { time: 4, event: true },
      { time: 6, event: true },
      { time: 8, event: true },
    ];

    const result = logRankTest(group1, group2);

    expect(result.rejected).toBe(false);
    expect(result.pValue).toBeGreaterThan(0.05);
  });

  it("handles censored observations", () => {
    const group1: SurvivalObservation[] = [
      { time: 1, event: true },
      { time: 3, event: false },
      { time: 5, event: true },
      { time: 7, event: false },
      { time: 9, event: true },
    ];
    const group2: SurvivalObservation[] = [
      { time: 20, event: true },
      { time: 25, event: false },
      { time: 30, event: true },
      { time: 35, event: false },
      { time: 40, event: true },
    ];

    const result = logRankTest(group1, group2);
    expect(result.chiSquared).toBeGreaterThanOrEqual(0);
    expect(result.pValue).toBeGreaterThanOrEqual(0);
    expect(result.pValue).toBeLessThanOrEqual(1);
  });

  it("identical groups yield p > 0.5", () => {
    const data: SurvivalObservation[] = [
      { time: 1, event: true },
      { time: 2, event: true },
      { time: 3, event: true },
      { time: 4, event: true },
      { time: 5, event: true },
    ];

    const result = logRankTest(data, data);
    expect(result.pValue).toBeGreaterThan(0.5);
  });

  it("throws on empty groups", () => {
    expect(() => logRankTest([], [{ time: 1, event: true }])).toThrow();
    expect(() => logRankTest([{ time: 1, event: true }], [])).toThrow();
  });
});

// ====================================================================
//  INTEGRATION: KM + Log-rank
// ====================================================================

describe("integration", () => {
  it("KM curves match log-rank conclusion", () => {
    // Two clearly different groups
    const treatment: SurvivalObservation[] = Array.from(
      { length: 15 },
      (_, i) => ({
        time: 20 + i,
        event: true,
      }),
    );
    const control: SurvivalObservation[] = Array.from(
      { length: 15 },
      (_, i) => ({
        time: 5 + i,
        event: true,
      }),
    );

    const kmTreatment = kaplanMeier(treatment);
    const kmControl = kaplanMeier(control);

    // Treatment should have better survival at t=15
    expect(kmTreatment.survivalAt(15)).toBeGreaterThan(
      kmControl.survivalAt(15),
    );

    // Log-rank should detect difference
    const lr = logRankTest(treatment, control);
    expect(lr.rejected).toBe(true);
  });

  it("Nelson-Aalen and KM are consistent", () => {
    const obs: SurvivalObservation[] = Array.from({ length: 10 }, (_, i) => ({
      time: i + 1,
      event: true,
    }));

    const km = kaplanMeier(obs);
    const na = nelsonAalen(obs);

    // For each event time, S_KM ≈ exp(-H_NA)
    for (let i = 0; i < km.curve.length && i < na.curve.length; i++) {
      const sKM = km.curve[i].survival;
      const sNA = Math.exp(-na.curve[i].hazard);

      // They should be close, especially for larger samples
      expect(Math.abs(sKM - sNA)).toBeLessThan(0.15);
    }
  });
});
