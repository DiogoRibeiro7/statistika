import {
  spatialDistanceMatrix,
  distanceBandWeights,
  knnWeights,
  moranI,
  gearyC,
  empiricalVariogram,
  fitVariogramModel,
  ordinaryKriging,
} from "../src/spatial";

// ── Distance matrix ───────────────────────────────────────────────────────

describe("spatialDistanceMatrix", () => {
  it("computes pairwise Euclidean distances", () => {
    const points = [[0, 0], [3, 4], [0, 1]];
    const D = spatialDistanceMatrix(points);
    expect(D[0][1]).toBeCloseTo(5, 10);
    expect(D[0][2]).toBeCloseTo(1, 10);
    expect(D[1][2]).toBeCloseTo(Math.sqrt(9 + 9), 10);
    expect(D[0][0]).toBe(0);
    // Symmetric
    expect(D[0][1]).toBe(D[1][0]);
  });
});

// ── Spatial weights ───────────────────────────────────────────────────────

describe("distanceBandWeights", () => {
  it("connects points within threshold", () => {
    const points = [[0, 0], [1, 0], [10, 0]];
    const W = distanceBandWeights(points, 2);
    expect(W[0][1]).toBe(1);
    expect(W[1][0]).toBe(1);
    expect(W[0][2]).toBe(0); // too far
    expect(W[0][0]).toBe(0); // no self-loop
  });
});

describe("knnWeights", () => {
  it("connects each point to k nearest neighbours", () => {
    const points = [[0, 0], [1, 0], [2, 0], [10, 0]];
    const W = knnWeights(points, 2);
    // Node 0's 2 nearest: 1, 2
    expect(W[0][1]).toBe(1);
    expect(W[0][2]).toBe(1);
    expect(W[0][3]).toBe(0);
  });

  it("throws if k >= n", () => {
    expect(() => knnWeights([[0], [1]], 2)).toThrow("less than");
  });
});

// ── Moran's I ─────────────────────────────────────────────────────────────

describe("moranI", () => {
  it("detects positive spatial autocorrelation", () => {
    // Clustered values: similar values near each other
    const points = [[0, 0], [1, 0], [2, 0], [10, 0], [11, 0], [12, 0]];
    const values = [1, 1.1, 0.9, 10, 10.1, 9.9]; // two clusters
    const W = distanceBandWeights(points, 2);
    const result = moranI(values, W);

    expect(result.I).toBeGreaterThan(0); // positive autocorrelation
  });

  it("I ≈ expected under no autocorrelation", () => {
    // Regular grid, unrelated values
    const values = [1, 5, 2, 8, 3, 7];
    const W = [
      [0, 1, 0, 0, 0, 0],
      [1, 0, 1, 0, 0, 0],
      [0, 1, 0, 1, 0, 0],
      [0, 0, 1, 0, 1, 0],
      [0, 0, 0, 1, 0, 1],
      [0, 0, 0, 0, 1, 0],
    ];
    const result = moranI(values, W);

    // Should be close to expected I = -1/(n-1) = -0.2
    expect(Math.abs(result.I - result.expectedI)).toBeLessThan(1);
    expect(result.expectedI).toBeCloseTo(-0.2, 8);
  });

  it("returns z-score and p-value", () => {
    const values = [1, 2, 3, 4, 5];
    const W = [
      [0, 1, 0, 0, 0],
      [1, 0, 1, 0, 0],
      [0, 1, 0, 1, 0],
      [0, 0, 1, 0, 1],
      [0, 0, 0, 1, 0],
    ];
    const result = moranI(values, W);
    expect(Number.isFinite(result.zScore)).toBe(true);
    expect(result.pValue).toBeGreaterThanOrEqual(0);
    expect(result.pValue).toBeLessThanOrEqual(1);
  });

  it("throws for insufficient data", () => {
    expect(() => moranI([1, 2], [[0, 1], [1, 0]])).toThrow("at least 3");
  });
});

// ── Geary's C ─────────────────────────────────────────────────────────────

describe("gearyC", () => {
  it("C < 1 for positive spatial autocorrelation", () => {
    const values = [1, 1.1, 0.9, 10, 10.1, 9.9];
    const points = [[0, 0], [1, 0], [2, 0], [10, 0], [11, 0], [12, 0]];
    const W = distanceBandWeights(points, 2);
    const result = gearyC(values, W);

    expect(result.C).toBeLessThan(1); // positive autocorrelation
  });

  it("C ≈ 1 under no autocorrelation", () => {
    const values = [1, 5, 2, 8, 3, 7];
    const W = [
      [0, 1, 0, 0, 0, 0],
      [1, 0, 1, 0, 0, 0],
      [0, 1, 0, 1, 0, 0],
      [0, 0, 1, 0, 1, 0],
      [0, 0, 0, 1, 0, 1],
      [0, 0, 0, 0, 1, 0],
    ];
    const result = gearyC(values, W);

    expect(result.expectedC).toBe(1);
    expect(result.C).toBeGreaterThan(0);
  });
});

// ── Variogram ─────────────────────────────────────────────────────────────

describe("empiricalVariogram", () => {
  it("returns binned semivariance values", () => {
    const points: number[][] = [];
    const values: number[] = [];
    for (let i = 0; i < 20; i++) {
      for (let j = 0; j < 20; j++) {
        points.push([i, j]);
        values.push(i + j + (i * j) * 0.01); // spatially structured
      }
    }

    const bins = empiricalVariogram(points, values, 10);
    expect(bins.length).toBeGreaterThan(0);
    expect(bins.length).toBeLessThanOrEqual(10);

    for (const bin of bins) {
      expect(bin.distance).toBeGreaterThan(0);
      expect(bin.semivariance).toBeGreaterThanOrEqual(0);
      expect(bin.count).toBeGreaterThan(0);
    }

    // Semivariance should generally increase with distance
    if (bins.length >= 3) {
      expect(bins[bins.length - 1].semivariance).toBeGreaterThan(bins[0].semivariance);
    }
  });
});

describe("fitVariogramModel", () => {
  it("fits spherical model to increasing semivariance", () => {
    const bins = [
      { distance: 1, semivariance: 0.5, count: 100 },
      { distance: 2, semivariance: 1.5, count: 80 },
      { distance: 3, semivariance: 2.8, count: 60 },
      { distance: 4, semivariance: 3.5, count: 40 },
      { distance: 5, semivariance: 3.8, count: 30 },
      { distance: 6, semivariance: 4.0, count: 20 },
    ];

    const model = fitVariogramModel(bins, "spherical");
    expect(model.type).toBe("spherical");
    expect(model.nugget).toBeGreaterThanOrEqual(0);
    expect(model.sill).toBeGreaterThan(0);
    expect(model.range).toBeGreaterThan(0);
    expect(model.evaluate(0)).toBe(0);
    expect(model.evaluate(100)).toBeCloseTo(model.nugget + model.sill, 8); // at/beyond range
  });

  it("fits exponential model", () => {
    const bins = [
      { distance: 1, semivariance: 1, count: 50 },
      { distance: 2, semivariance: 2, count: 40 },
      { distance: 5, semivariance: 4, count: 20 },
    ];
    const model = fitVariogramModel(bins, "exponential");
    expect(model.type).toBe("exponential");
    expect(model.evaluate(0)).toBe(0);
  });
});

// ── Kriging ───────────────────────────────────────────────────────────────

describe("ordinaryKriging", () => {
  it("predicts known values exactly at observed points", () => {
    const points = [[0, 0], [1, 0], [0, 1], [1, 1]];
    const values = [10, 20, 30, 40];

    const model = fitVariogramModel([
      { distance: 0.5, semivariance: 5, count: 10 },
      { distance: 1.0, semivariance: 15, count: 10 },
      { distance: 1.5, semivariance: 25, count: 5 },
    ], "spherical");

    const result = ordinaryKriging(points, values, [[0, 0]], model);
    // At an observed point, the prediction should be close to the observed value
    expect(result.predictions[0]).toBeCloseTo(10, 0);
    expect(result.variances[0]).toBeLessThan(5);
  });

  it("returns uncertainty that increases with distance", () => {
    const points = [[0, 0], [1, 0], [0, 1]];
    const values = [10, 20, 15];

    const model = fitVariogramModel([
      { distance: 0.5, semivariance: 5, count: 10 },
      { distance: 1.0, semivariance: 15, count: 10 },
      { distance: 2.0, semivariance: 25, count: 5 },
    ], "exponential");

    const result = ordinaryKriging(points, values, [[0.5, 0.5], [5, 5]], model);
    expect(result.predictions.length).toBe(2);
    expect(result.variances.length).toBe(2);
    // Farther point should have higher variance
    expect(result.variances[1]).toBeGreaterThan(result.variances[0]);
  });

  it("throws for insufficient observations", () => {
    expect(() =>
      ordinaryKriging([[0]], [1], [[1]], {
        type: "spherical", nugget: 0, sill: 1, range: 1,
        evaluate: () => 0,
      }),
    ).toThrow("at least 2");
  });
});
