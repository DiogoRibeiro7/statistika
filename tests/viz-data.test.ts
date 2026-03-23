import {
  histogramBins,
  qqPlot,
  normalQQPlot,
  boxPlotStats,
  kde,
  ecdf,
  scatterMatrixData,
} from "../src/viz-data";
import { Normal } from "../src/distributions/continuous/normal";

// ── Histogram ─────────────────────────────────────────────────────────────

describe("histogramBins", () => {
  it("bins data correctly", () => {
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const bins = histogramBins(data, { bins: 5 });
    expect(bins.length).toBe(5);

    // Total count should equal n
    const totalCount = bins.reduce((s, b) => s + b.count, 0);
    expect(totalCount).toBe(10);
  });

  it("frequencies sum to 1", () => {
    const data = [1, 2, 3, 4, 5];
    const bins = histogramBins(data, { bins: 3 });
    const totalFreq = bins.reduce((s, b) => s + b.frequency, 0);
    expect(totalFreq).toBeCloseTo(1, 8);
  });

  it("density integrates to ~1", () => {
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const bins = histogramBins(data, { bins: 5 });
    let integral = 0;
    for (const bin of bins) integral += bin.density * (bin.hi - bin.lo);
    expect(integral).toBeCloseTo(1, 8);
  });

  it("respects custom range", () => {
    const data = [5, 6, 7, 8];
    const bins = histogramBins(data, { bins: 4, range: [0, 10] });
    expect(bins[0].lo).toBe(0);
    expect(bins[bins.length - 1].hi).toBe(10);
  });

  it("uses Sturges rule by default", () => {
    const data = Array.from({ length: 100 }, (_, i) => i);
    const bins = histogramBins(data);
    const expectedBins = Math.ceil(Math.log2(100) + 1);
    expect(bins.length).toBe(expectedBins);
  });

  it("returns empty for empty data", () => {
    expect(histogramBins([])).toEqual([]);
  });

  it("handles single value", () => {
    const bins = histogramBins([5]);
    expect(bins.length).toBe(1);
    expect(bins[0].count).toBe(1);
  });
});

// ── Q-Q Plot ──────────────────────────────────────────────────────────────

describe("qqPlot", () => {
  it("generates correct number of points", () => {
    const data = [1, 2, 3, 4, 5];
    const dist = new Normal(0, 1);
    const points = qqPlot(data, dist);
    expect(points.length).toBe(5);
  });

  it("sample quantiles are sorted", () => {
    const data = [5, 3, 1, 4, 2];
    const dist = new Normal(0, 1);
    const points = qqPlot(data, dist);
    for (let i = 1; i < points.length; i++) {
      expect(points[i].sample).toBeGreaterThanOrEqual(points[i - 1].sample);
    }
  });

  it("theoretical quantiles are sorted", () => {
    const data = [1, 2, 3, 4, 5];
    const dist = new Normal(0, 1);
    const points = qqPlot(data, dist);
    for (let i = 1; i < points.length; i++) {
      expect(points[i].theoretical).toBeGreaterThan(points[i - 1].theoretical);
    }
  });
});

describe("normalQQPlot", () => {
  it("generates points against standard normal", () => {
    const data = [-1.5, -0.5, 0, 0.5, 1.5];
    const points = normalQQPlot(data);
    expect(points.length).toBe(5);

    // For roughly normal data, theoretical ≈ sample
    for (const p of points) {
      expect(Math.abs(p.theoretical - p.sample)).toBeLessThan(2);
    }
  });
});

// ── Box Plot ──────────────────────────────────────────────────────────────

describe("boxPlotStats", () => {
  it("computes five-number summary", () => {
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const stats = boxPlotStats(data);

    expect(stats.min).toBe(1);
    expect(stats.max).toBe(10);
    expect(stats.median).toBeCloseTo(5.5, 8);
    expect(stats.q1).toBeCloseTo(3.25, 8);
    expect(stats.q3).toBeCloseTo(7.75, 8);
    expect(stats.iqr).toBeCloseTo(4.5, 8);
  });

  it("detects outliers", () => {
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50]; // 50 is outlier
    const stats = boxPlotStats(data);
    expect(stats.upperOutliers).toContain(50);
    expect(stats.max).toBeLessThan(50); // max excludes outliers
  });

  it("no outliers in tight data", () => {
    const data = [4, 5, 5, 6, 6, 6, 7, 7, 8];
    const stats = boxPlotStats(data);
    expect(stats.lowerOutliers.length).toBe(0);
    expect(stats.upperOutliers.length).toBe(0);
  });

  it("computes mean", () => {
    const data = [2, 4, 6, 8];
    const stats = boxPlotStats(data);
    expect(stats.mean).toBe(5);
  });

  it("fences are correct", () => {
    const data = [1, 2, 3, 4, 5];
    const stats = boxPlotStats(data);
    expect(stats.lowerFence).toBe(stats.q1 - 1.5 * stats.iqr);
    expect(stats.upperFence).toBe(stats.q3 + 1.5 * stats.iqr);
  });

  it("throws for empty data", () => {
    expect(() => boxPlotStats([])).toThrow("at least 1");
  });
});

// ── KDE ───────────────────────────────────────────────────────────────────

describe("kde", () => {
  it("returns requested number of points", () => {
    const data = [1, 2, 3, 4, 5];
    const result = kde(data, { nPoints: 50 });
    expect(result.length).toBe(50);
  });

  it("density is non-negative", () => {
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const result = kde(data);
    for (const p of result) {
      expect(p.density).toBeGreaterThanOrEqual(0);
    }
  });

  it("integrates to approximately 1", () => {
    const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const result = kde(data, { nPoints: 200 });
    let integral = 0;
    for (let i = 1; i < result.length; i++) {
      integral += (result[i].density + result[i - 1].density) / 2 * (result[i].x - result[i - 1].x);
    }
    expect(integral).toBeCloseTo(1, 1);
  });

  it("custom bandwidth affects smoothness", () => {
    const data = [1, 2, 3, 10, 11, 12];
    const narrow = kde(data, { bandwidth: 0.5, nPoints: 50 });
    const wide = kde(data, { bandwidth: 3, nPoints: 50 });

    // Wide bandwidth should have lower peak density
    const maxNarrow = Math.max(...narrow.map((p) => p.density));
    const maxWide = Math.max(...wide.map((p) => p.density));
    expect(maxWide).toBeLessThan(maxNarrow);
  });

  it("returns empty for empty data", () => {
    expect(kde([])).toEqual([]);
  });
});

// ── ECDF ──────────────────────────────────────────────────────────────────

describe("ecdf", () => {
  it("returns sorted points", () => {
    const data = [5, 3, 1, 4, 2];
    const points = ecdf(data);
    for (let i = 1; i < points.length; i++) {
      expect(points[i].x).toBeGreaterThanOrEqual(points[i - 1].x);
    }
  });

  it("probabilities go from 1/n to 1", () => {
    const data = [1, 2, 3, 4, 5];
    const points = ecdf(data);
    expect(points[0].probability).toBeCloseTo(0.2, 10);
    expect(points[4].probability).toBeCloseTo(1.0, 10);
  });

  it("returns n points", () => {
    const data = [10, 20, 30];
    expect(ecdf(data).length).toBe(3);
  });
});

// ── Scatter matrix ────────────────────────────────────────────────────────

describe("scatterMatrixData", () => {
  it("generates pairwise combinations", () => {
    const vars = { x: [1, 2, 3], y: [4, 5, 6], z: [7, 8, 9] };
    const pairs = scatterMatrixData(vars);
    // 3 variables → 3 × 2 = 6 off-diagonal pairs
    expect(pairs.length).toBe(6);
  });

  it("each pair has correct data", () => {
    const vars = { a: [1, 2], b: [3, 4] };
    const pairs = scatterMatrixData(vars);
    const ab = pairs.find((p) => p.xName === "a" && p.yName === "b");
    expect(ab).toBeDefined();
    expect(ab!.x).toEqual([1, 2]);
    expect(ab!.y).toEqual([3, 4]);
  });

  it("returns empty for single variable", () => {
    const pairs = scatterMatrixData({ x: [1, 2, 3] });
    expect(pairs.length).toBe(0);
  });
});
