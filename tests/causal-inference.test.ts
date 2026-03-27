import {
  propensityScore,
  ipw,
  propensityMatching,
  differenceInDifferences,
  twoSLS,
  rdd,
  fuzzyRDD,
} from "../src/causal-inference";

// ── Propensity score ──────────────────────────────────────────────────────

describe("propensityScore", () => {
  it("produces scores in (0, 1)", () => {
    const X = [[1], [2], [3], [4], [5], [6], [7], [8]];
    const treatment = [0, 0, 0, 0, 1, 1, 1, 1];

    const result = propensityScore(X, treatment);
    for (const s of result.scores) {
      expect(s).toBeGreaterThanOrEqual(0);
      expect(s).toBeLessThanOrEqual(1);
    }
  });

  it("treated units have higher scores", () => {
    const X = [[1], [2], [3], [4], [5], [6], [7], [8]];
    const treatment = [0, 0, 0, 0, 1, 1, 1, 1];

    const result = propensityScore(X, treatment);
    const treatedMean = result.scores.filter((_, i) => treatment[i] === 1)
      .reduce((a, b) => a + b, 0) / 4;
    const controlMean = result.scores.filter((_, i) => treatment[i] === 0)
      .reduce((a, b) => a + b, 0) / 4;

    expect(treatedMean).toBeGreaterThan(controlMean);
  });

  it("returns coefficients and iteration count", () => {
    const X = [[1], [2], [3], [4], [5], [6]];
    const treatment = [0, 0, 0, 1, 1, 1];

    const result = propensityScore(X, treatment);
    expect(result.coefficients.length).toBe(2); // intercept + 1 feature
    expect(result.iterations).toBeGreaterThan(0);
  });

  it("throws for mismatched lengths", () => {
    expect(() => propensityScore([[1]], [0, 1])).toThrow("same length");
  });
});

// ── IPW ───────────────────────────────────────────────────────────────────

describe("ipw", () => {
  it("estimates ATE from known treatment effect", () => {
    // treatment effect = 10
    const y = [10, 12, 11, 13, 20, 22, 21, 23];
    const treatment = [0, 0, 0, 0, 1, 1, 1, 1];
    const scores = [0.3, 0.3, 0.3, 0.3, 0.7, 0.7, 0.7, 0.7];

    const result = ipw(y, treatment, scores);
    expect(result.ate).toBeGreaterThan(5);
    expect(result.att).toBeGreaterThan(5);
  });

  it("clips extreme scores", () => {
    const y = [5, 15];
    const treatment = [0, 1];
    const scores = [0.001, 0.999];

    // Should not throw due to division by zero
    const result = ipw(y, treatment, scores);
    expect(Number.isFinite(result.ate)).toBe(true);
  });

  it("throws for mismatched lengths", () => {
    expect(() => ipw([1], [0, 1], [0.5])).toThrow("same length");
  });
});

// ── Propensity score matching ─────────────────────────────────────────────

describe("propensityMatching", () => {
  it("matches treated to nearest control", () => {
    const y = [10, 12, 14, 20, 22];
    const treatment = [0, 0, 0, 1, 1];
    const scores = [0.2, 0.5, 0.8, 0.75, 0.45];

    const result = propensityMatching(y, treatment, scores);
    expect(result.matches.length).toBe(2); // 2 treated
    expect(result.att).toBeCloseTo(result.treatedMean - result.controlMean, 10);
  });

  it("all treated units get a match", () => {
    const y = [1, 2, 3, 4, 5, 6];
    const treatment = [0, 0, 0, 1, 1, 1];
    const scores = [0.1, 0.3, 0.5, 0.4, 0.6, 0.8];

    const result = propensityMatching(y, treatment, scores);
    expect(result.matches.length).toBe(3);
  });

  it("throws when no treated or control", () => {
    expect(() =>
      propensityMatching([1, 2], [1, 1], [0.5, 0.5]),
    ).toThrow("control");
  });
});

// ── Difference-in-Differences ─────────────────────────────────────────────

describe("differenceInDifferences", () => {
  it("recovers known treatment effect", () => {
    // Control: 10 -> 12 (natural growth = 2)
    // Treated: 10 -> 15 (natural growth + treatment = 5)
    // DiD = (15-10) - (12-10) = 3
    const y = [10, 10, 12, 12, 10, 10, 15, 15];
    const treatment = [0, 0, 0, 0, 1, 1, 1, 1];
    const post = [0, 0, 1, 1, 0, 0, 1, 1];

    const result = differenceInDifferences(y, treatment, post);
    expect(result.estimate).toBeCloseTo(3, 8);
    expect(result.treatPre).toBeCloseTo(10, 8);
    expect(result.treatPost).toBeCloseTo(15, 8);
    expect(result.controlPre).toBeCloseTo(10, 8);
    expect(result.controlPost).toBeCloseTo(12, 8);
  });

  it("returns zero when no treatment effect", () => {
    const y = [10, 10, 15, 15, 10, 10, 15, 15];
    const treatment = [0, 0, 0, 0, 1, 1, 1, 1];
    const post = [0, 0, 1, 1, 0, 0, 1, 1];

    const result = differenceInDifferences(y, treatment, post);
    expect(result.estimate).toBeCloseTo(0, 8);
  });

  it("computes standard error and t-statistic", () => {
    const y = [10, 11, 12, 13, 10, 11, 15, 16];
    const treatment = [0, 0, 0, 0, 1, 1, 1, 1];
    const post = [0, 0, 1, 1, 0, 0, 1, 1];

    const result = differenceInDifferences(y, treatment, post);
    expect(result.standardError).toBeGreaterThan(0);
    expect(Number.isFinite(result.tStatistic)).toBe(true);
  });

  it("throws when a group is empty", () => {
    expect(() =>
      differenceInDifferences([1, 2], [1, 1], [0, 1]),
    ).toThrow("expected at least 1 observation");
  });

  it("throws for mismatched lengths", () => {
    expect(() =>
      differenceInDifferences([1], [0, 1], [0]),
    ).toThrow("same length");
  });
});

// ── Two-Stage Least Squares ───────────────────────────────────────────────

describe("twoSLS", () => {
  it("recovers causal effect with valid instrument", () => {
    // True model: y = 2*x + noise, but x is endogenous
    // Instrument z is correlated with x but not with noise
    const n = 100;
    const z: number[][] = [];
    const x: number[] = [];
    const y: number[] = [];

    for (let i = 0; i < n; i++) {
      const zi = (i - 50) / 10;
      z.push([zi]);
      // x = 3*z + small variation (strong first stage)
      const xi = 3 * zi + (i % 3 - 1) * 0.1;
      x.push(xi);
      // y = 2*x + systematic pattern (not correlated with z beyond x)
      y.push(2 * xi + (i % 5 - 2) * 0.1);
    }

    const result = twoSLS(y, x, z);
    expect(result.slopes[0]).toBeCloseTo(2, 0);
    expect(result.firstStageF).toBeGreaterThan(10); // strong instrument
  });

  it("predict works correctly", () => {
    const z = [[1], [2], [3], [4], [5], [6]];
    const x = [2, 4, 6, 8, 10, 12]; // x = 2*z
    const y = [3, 5, 7, 9, 11, 13]; // y = x + 1

    const result = twoSLS(y, x, z);
    const predicted = result.predict(10);
    expect(predicted).toBeCloseTo(11, 0);
  });

  it("throws when too few instruments", () => {
    expect(() =>
      twoSLS([1, 2, 3], [[1, 2], [3, 4], [5, 6]], [[1], [2], [3]]),
    ).toThrow("at least as many instruments");
  });

  it("throws for mismatched lengths", () => {
    expect(() =>
      twoSLS([1, 2], [3, 4, 5], [[1], [2]]),
    ).toThrow("same number");
  });
});

// ── Regression Discontinuity Design ───────────────────────────────────────

describe("rdd (sharp)", () => {
  it("detects known jump at cutoff", () => {
    const n = 100;
    const y: number[] = [];
    const running: number[] = [];

    for (let i = 0; i < n; i++) {
      const x = (i - 50) / 10;
      running.push(x);
      // y = 2*x + 10*(x >= 0) — jump of 10 at cutoff 0
      y.push(2 * x + (x >= 0 ? 10 : 0));
    }

    const result = rdd(y, running, 0);
    expect(result.estimate).toBeCloseTo(10, 0);
    expect(result.nBelow).toBeGreaterThan(0);
    expect(result.nAbove).toBeGreaterThan(0);
  });

  it("respects bandwidth option", () => {
    const n = 100;
    const y: number[] = [];
    const running: number[] = [];

    for (let i = 0; i < n; i++) {
      const x = (i - 50) / 10;
      running.push(x);
      y.push(x + (x >= 0 ? 5 : 0));
    }

    const wide = rdd(y, running, 0);
    const narrow = rdd(y, running, 0, { bandwidth: 2 });

    expect(narrow.nBelow + narrow.nAbove).toBeLessThan(
      wide.nBelow + wide.nAbove,
    );
  });

  it("returns slopes for each side", () => {
    const running = [-3, -2, -1, 1, 2, 3];
    const y = [-6, -4, -2, 12, 14, 16]; // slope=2 below, slope=2 above, jump=10

    const result = rdd(y, running, 0);
    expect(result.slopeBelow).toBeCloseTo(2, 4);
    expect(result.slopeAbove).toBeCloseTo(2, 4);
    expect(result.estimate).toBeCloseTo(10, 4);
  });

  it("computes standard error", () => {
    const running = [-3, -2, -1, 1, 2, 3];
    const y = [-6, -4, -2, 12, 14, 16];

    const result = rdd(y, running, 0);
    expect(result.standardError).toBeGreaterThanOrEqual(0);
  });

  it("throws when not enough observations on one side", () => {
    expect(() =>
      rdd([1, 2, 3], [1, 2, 3], 0),
    ).toThrow("at least 2");
  });
});

describe("fuzzyRDD", () => {
  it("estimates treatment effect with imperfect compliance", () => {
    const n = 60;
    const y: number[] = [];
    const treatment: number[] = [];
    const running: number[] = [];

    for (let i = 0; i < n; i++) {
      const x = (i - 30) / 10;
      running.push(x);
      // Treatment probability jumps from 0 to ~0.8 at cutoff
      const treated = x >= 0 ? (i % 5 !== 0 ? 1 : 0) : 0;
      treatment.push(treated);
      // y = 5 * treatment + x (no noise)
      y.push(5 * treated + x);
    }

    const result = fuzzyRDD(y, treatment, running, 0);
    // Fuzzy RDD estimates the LATE (local average treatment effect)
    // The estimate should be positive and in the right ballpark
    expect(result.estimate).toBeGreaterThan(0);
    expect(result.firstStageF).toBeGreaterThan(0);
    expect(result.nUsed).toBe(n);
  });

  it("throws for insufficient observations", () => {
    expect(() =>
      fuzzyRDD([1, 2], [0, 1], [0, 1], 0.5),
    ).toThrow("at least 4");
  });
});

// ── Integration ───────────────────────────────────────────────────────────

describe("integration: propensity score pipeline", () => {
  it("full pipeline: estimate -> match -> compare", () => {
    // Create data where treatment effect = 5
    const X = [
      [1], [2], [3], [4], [5],
      [3], [4], [5], [6], [7],
    ];
    const treatment = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
    const y = [10, 12, 14, 16, 18, 19, 21, 23, 25, 27];
    // Without treatment: y ≈ 2*x + 8
    // Treatment adds ~5 to outcome

    // Step 1: Estimate propensity scores
    const ps = propensityScore(X, treatment);
    expect(ps.scores.length).toBe(10);

    // Step 2: Match
    const match = propensityMatching(y, treatment, ps.scores);
    expect(match.matches.length).toBe(5);

    // Step 3: IPW
    const ipwResult = ipw(y, treatment, ps.scores);
    expect(Number.isFinite(ipwResult.ate)).toBe(true);
    expect(Number.isFinite(ipwResult.att)).toBe(true);
  });
});
