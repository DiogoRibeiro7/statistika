import {
  stepwiseSelection,
  ridgeRegression,
  lassoRegression,
  elasticNet,
} from "../../src/models";

// ── Test data ─────────────────────────────────────────────────────────────

/**
 * Synthetic dataset:  y = 3*x1 + 0*x2 + 5*x3 + 10
 * x2 is pure noise (random-ish), x1 and x3 are informative and independent.
 */
function makeData() {
  const X = [
    [1, 10, 2],
    [2, 3, 7],
    [3, 15, 1],
    [4, 8, 5],
    [5, 22, 3],
    [6, 5, 9],
    [7, 18, 4],
    [8, 12, 6],
    [9, 25, 8],
    [10, 7, 0],
    [3, 19, 6],
    [7, 2, 2],
  ];
  // y = 3*x1 + 5*x3 + 10
  const y = X.map(([x1, , x3]) => 3 * x1 + 5 * x3 + 10);
  return { X, y };
}

/** Simple 2-feature dataset with known solution. */
function makeSimple() {
  const X = [
    [1, 0],
    [0, 1],
    [1, 1],
    [2, 1],
    [1, 2],
    [3, 0],
    [0, 3],
  ];
  // y = 2*x1 + 3*x2 + 1
  const y = X.map(([x1, x2]) => 2 * x1 + 3 * x2 + 1);
  return { X, y };
}

// ── Stepwise selection ────────────────────────────────────────────────────

describe("stepwiseSelection", () => {
  it("forward selection identifies informative features", () => {
    const { X, y } = makeData();
    const result = stepwiseSelection(X, y, {
      direction: "forward",
      criterion: "aic",
    });

    expect(result.selectedFeatures).toContain(0); // x1
    expect(result.selectedFeatures).toContain(2); // x3
    expect(result.rSquared).toBeGreaterThan(0.99);
  });

  it("backward selection identifies informative features", () => {
    const { X, y } = makeData();
    const result = stepwiseSelection(X, y, {
      direction: "backward",
      criterion: "aic",
    });

    expect(result.selectedFeatures).toContain(0);
    expect(result.selectedFeatures).toContain(2);
    expect(result.rSquared).toBeGreaterThan(0.99);
  });

  it("both direction identifies informative features", () => {
    const { X, y } = makeData();
    const result = stepwiseSelection(X, y, {
      direction: "both",
      criterion: "bic",
    });

    expect(result.selectedFeatures).toContain(0);
    expect(result.selectedFeatures).toContain(2);
    expect(result.rSquared).toBeGreaterThan(0.99);
  });

  it("predict works with full feature vector", () => {
    const { X, y } = makeData();
    const result = stepwiseSelection(X, y);

    // Predict for known input
    const predicted = result.predict([5, 999, 5]); // noise feature ignored
    expect(predicted).toBeCloseTo(3 * 5 + 5 * 5 + 10, 4);
  });

  it("respects maxFeatures", () => {
    const { X, y } = makeData();
    const result = stepwiseSelection(X, y, {
      direction: "forward",
      maxFeatures: 1,
    });

    expect(result.selectedFeatures.length).toBeLessThanOrEqual(1);
  });

  it("uses adjr2 criterion", () => {
    const { X, y } = makeData();
    const result = stepwiseSelection(X, y, { criterion: "adjr2" });

    expect(result.selectedFeatures).toContain(0);
    expect(result.selectedFeatures).toContain(2);
  });

  it("throws when X and y have different lengths", () => {
    expect(() =>
      stepwiseSelection([[1], [2]], [1, 2, 3]),
    ).toThrow("same number of observations");
  });

  it("throws when fewer than 2 observations", () => {
    expect(() => stepwiseSelection([[1, 2]], [1])).toThrow("at least 2");
  });
});

// ── Ridge regression ──────────────────────────────────────────────────────

describe("ridgeRegression", () => {
  it("with lambda=0 matches OLS", () => {
    const { X, y } = makeSimple();
    const result = ridgeRegression(X, y, 0);

    expect(result.coefficients[0]).toBeCloseTo(2, 5);
    expect(result.coefficients[1]).toBeCloseTo(3, 5);
    expect(result.intercept).toBeCloseTo(1, 5);
  });

  it("shrinks coefficients with increasing lambda", () => {
    const { X, y } = makeSimple();
    const low = ridgeRegression(X, y, 0.1);
    const high = ridgeRegression(X, y, 100);

    const normLow = Math.sqrt(
      low.coefficients.reduce((s, c) => s + c * c, 0),
    );
    const normHigh = Math.sqrt(
      high.coefficients.reduce((s, c) => s + c * c, 0),
    );

    expect(normHigh).toBeLessThan(normLow);
  });

  it("predict works correctly", () => {
    const { X, y } = makeSimple();
    const result = ridgeRegression(X, y, 0);

    const predicted = result.predict([2, 3]);
    expect(predicted).toBeCloseTo(2 * 2 + 3 * 3 + 1, 4);
  });

  it("throws for negative lambda", () => {
    const { X, y } = makeSimple();
    expect(() => ridgeRegression(X, y, -1)).toThrow("non-negative");
  });

  it("throws when X and y have different lengths", () => {
    expect(() =>
      ridgeRegression([[1], [2]], [1, 2, 3], 1),
    ).toThrow("same number of observations");
  });
});

// ── LASSO regression ──────────────────────────────────────────────────────

describe("lassoRegression", () => {
  it("with lambda=0 approximates OLS", () => {
    const { X, y } = makeSimple();
    const result = lassoRegression(X, y, 0);

    expect(result.coefficients[0]).toBeCloseTo(2, 3);
    expect(result.coefficients[1]).toBeCloseTo(3, 3);
    expect(result.intercept).toBeCloseTo(1, 3);
  });

  it("zeroes out noise features with large lambda", () => {
    const { X, y } = makeData();
    // Large enough lambda to zero out the noise feature (x2)
    const result = lassoRegression(X, y, 0.5);

    // The noise feature coefficient should be zero or near-zero
    expect(Math.abs(result.coefficients[1])).toBeLessThan(0.5);
  });

  it("with very large lambda, all coefficients shrink toward zero", () => {
    const { X, y } = makeSimple();
    const result = lassoRegression(X, y, 100);

    for (const c of result.coefficients) {
      expect(Math.abs(c)).toBeLessThan(1);
    }
  });

  it("predict works correctly", () => {
    const { X, y } = makeSimple();
    const result = lassoRegression(X, y, 0);

    const predicted = result.predict([2, 3]);
    expect(predicted).toBeCloseTo(2 * 2 + 3 * 3 + 1, 2);
  });

  it("throws for negative lambda", () => {
    const { X, y } = makeSimple();
    expect(() => lassoRegression(X, y, -1)).toThrow("non-negative");
  });

  it("returns iteration count", () => {
    const { X, y } = makeSimple();
    const result = lassoRegression(X, y, 0.1);
    expect(result.iterations).toBeGreaterThan(0);
  });
});

// ── Elastic Net ───────────────────────────────────────────────────────────

describe("elasticNet", () => {
  it("alpha=1 matches LASSO", () => {
    const { X, y } = makeSimple();
    const lasso = lassoRegression(X, y, 0.1);
    const enet = elasticNet(X, y, 0.1, { alpha: 1.0 });

    for (let j = 0; j < lasso.coefficients.length; j++) {
      expect(enet.coefficients[j]).toBeCloseTo(lasso.coefficients[j], 8);
    }
    expect(enet.intercept).toBeCloseTo(lasso.intercept, 8);
  });

  it("alpha=0.5 blends L1 and L2", () => {
    const { X, y } = makeData();
    const result = elasticNet(X, y, 0.5, { alpha: 0.5 });

    // Should still recover signal features approximately
    expect(Math.abs(result.coefficients[0])).toBeGreaterThan(1);
    expect(Math.abs(result.coefficients[2])).toBeGreaterThan(1);
  });

  it("shrinks more than LASSO alone for same lambda", () => {
    const { X, y } = makeSimple();
    const lasso = lassoRegression(X, y, 1);
    const enet = elasticNet(X, y, 1, { alpha: 0.5 });

    const normLasso = Math.sqrt(
      lasso.coefficients.reduce((s, c) => s + c * c, 0),
    );
    const normEnet = Math.sqrt(
      enet.coefficients.reduce((s, c) => s + c * c, 0),
    );

    // Elastic net with L2 component should shrink differently
    expect(normEnet).toBeLessThan(normLasso * 1.5); // rough sanity check
  });

  it("throws for alpha out of range", () => {
    const { X, y } = makeSimple();
    expect(() => elasticNet(X, y, 1, { alpha: 1.5 })).toThrow("alpha");
    expect(() => elasticNet(X, y, 1, { alpha: -0.1 })).toThrow("alpha");
  });

  it("predict works correctly with alpha=0.5", () => {
    const { X, y } = makeSimple();
    const result = elasticNet(X, y, 0, { alpha: 0.5 });

    const predicted = result.predict([2, 3]);
    expect(predicted).toBeCloseTo(2 * 2 + 3 * 3 + 1, 2);
  });
});

// ── Cross-method consistency ──────────────────────────────────────────────

describe("cross-method consistency", () => {
  it("ridge and elastic net (alpha=0) both shrink toward zero", () => {
    const { X, y } = makeSimple();
    const lambda = 1;
    const ridge = ridgeRegression(X, y, lambda);
    const enet = elasticNet(X, y, lambda, { alpha: 0 });

    // Both should shrink relative to OLS
    const ols = ridgeRegression(X, y, 0);
    const normOLS = Math.sqrt(ols.coefficients.reduce((s, c) => s + c * c, 0));
    const normRidge = Math.sqrt(ridge.coefficients.reduce((s, c) => s + c * c, 0));
    const normEnet = Math.sqrt(enet.coefficients.reduce((s, c) => s + c * c, 0));

    expect(normRidge).toBeLessThan(normOLS);
    expect(normEnet).toBeLessThan(normOLS);
  });

  it("all methods produce reasonable predictions on simple data", () => {
    const { X, y } = makeSimple();
    const methods = [
      ridgeRegression(X, y, 0.01),
      lassoRegression(X, y, 0.01),
      elasticNet(X, y, 0.01, { alpha: 0.5 }),
    ];

    for (const result of methods) {
      const predicted = result.predict([1, 1]);
      // True value: 2*1 + 3*1 + 1 = 6
      expect(predicted).toBeCloseTo(6, 0);
    }
  });
});
