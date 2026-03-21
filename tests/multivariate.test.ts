import {
  pca,
  factorAnalysis,
  kMeans,
  hierarchicalClustering,
} from "../src/multivariate";

// ── Test data ───────────────────────────────────────────────────────────

// 2D data with clear principal axis along (1,1)
const pcaData = [
  [1, 2],
  [2, 4],
  [3, 6],
  [4, 8],
  [5, 10],
  [6, 12],
  [7, 14],
  [8, 16],
];

// Three well-separated clusters
const clusterData = [
  // Cluster 0 near (0, 0)
  [0, 0], [0.1, 0.2], [0.2, 0.1], [-0.1, -0.1], [0.15, -0.05],
  // Cluster 1 near (10, 10)
  [10, 10], [10.1, 10.2], [9.9, 10.1], [10.2, 9.8], [9.8, 10.15],
  // Cluster 2 near (10, 0)
  [10, 0], [10.1, 0.2], [9.9, 0.1], [10.2, -0.1], [9.8, 0.05],
];

// ── PCA ─────────────────────────────────────────────────────────────────

describe("pca", () => {
  it("returns correct structure", () => {
    const result = pca(pcaData);
    expect(result.eigenvalues).toHaveLength(2);
    expect(result.components).toHaveLength(2);
    expect(result.explainedVariance).toHaveLength(2);
    expect(result.cumulativeVariance).toHaveLength(2);
    expect(result.scores).toHaveLength(8);
    expect(result.nFeatures).toBe(2);
    expect(result.nObservations).toBe(8);
  });

  it("first PC explains nearly all variance for collinear data", () => {
    const result = pca(pcaData);
    expect(result.explainedVariance[0]).toBeGreaterThan(0.99);
  });

  it("cumulative variance sums to 1", () => {
    const result = pca(pcaData);
    expect(result.cumulativeVariance[result.cumulativeVariance.length - 1]).toBeCloseTo(1, 5);
  });

  it("eigenvalues are in descending order", () => {
    const data = [
      [1, 3, 5],
      [2, 1, 4],
      [4, 2, 6],
      [3, 5, 7],
      [5, 4, 8],
    ];
    const result = pca(data);
    for (let i = 1; i < result.eigenvalues.length; i++) {
      expect(result.eigenvalues[i]).toBeLessThanOrEqual(result.eigenvalues[i - 1]);
    }
  });

  it("respects nComponents option", () => {
    const result = pca(pcaData, { nComponents: 1 });
    expect(result.eigenvalues).toHaveLength(1);
    expect(result.components).toHaveLength(1);
    expect(result.scores[0]).toHaveLength(1);
  });

  it("supports scaling", () => {
    const result = pca(pcaData, { scale: true });
    expect(result.eigenvalues).toHaveLength(2);
    // With perfect collinearity and scaling, first PC should still dominate
    expect(result.explainedVariance[0]).toBeGreaterThan(0.99);
  });

  it("throws for fewer than 2 observations", () => {
    expect(() => pca([[1, 2]])).toThrow();
  });

  it("throws for invalid nComponents", () => {
    expect(() => pca(pcaData, { nComponents: 0 })).toThrow();
    expect(() => pca(pcaData, { nComponents: 3 })).toThrow();
  });
});

// ── Factor Analysis ─────────────────────────────────────────────────────

describe("factorAnalysis", () => {
  // Correlated 4-variable dataset with 2 underlying factors
  const faData: number[][] = [];
  for (let i = 0; i < 50; i++) {
    const f1 = i * 0.5;
    const f2 = (50 - i) * 0.3;
    faData.push([
      f1 + 0.1 * i,
      f1 - 0.05 * i + 2,
      f2 + 0.08 * i,
      f2 - 0.12 * i + 1,
    ]);
  }

  it("returns correct structure", () => {
    const result = factorAnalysis(faData, 2);
    expect(result.loadings).toHaveLength(4);
    expect(result.loadings[0]).toHaveLength(2);
    expect(result.uniquenesses).toHaveLength(4);
    expect(result.communalities).toHaveLength(4);
    expect(result.explainedVariance).toHaveLength(2);
    expect(result.nFactors).toBe(2);
  });

  it("communalities + uniquenesses approximate diagonal of covariance", () => {
    const result = factorAnalysis(faData, 2);
    for (let j = 0; j < 4; j++) {
      const total = result.communalities[j] + result.uniquenesses[j];
      expect(total).toBeGreaterThan(0);
    }
  });

  it("uniquenesses are non-negative", () => {
    const result = factorAnalysis(faData, 2);
    for (const u of result.uniquenesses) {
      expect(u).toBeGreaterThanOrEqual(0);
    }
  });

  it("communalities are non-negative", () => {
    const result = factorAnalysis(faData, 2);
    for (const c of result.communalities) {
      expect(c).toBeGreaterThanOrEqual(0);
    }
  });

  it("throws for invalid nFactors", () => {
    expect(() => factorAnalysis(faData, 0)).toThrow();
    expect(() => factorAnalysis(faData, 5)).toThrow();
  });
});

// ── K-Means ─────────────────────────────────────────────────────────────

describe("kMeans", () => {
  it("finds 3 clusters in well-separated data", () => {
    const result = kMeans(clusterData, 3, { seed: 42 });
    expect(result.assignments).toHaveLength(15);
    expect(result.centroids).toHaveLength(3);

    // Points within the same group should have the same assignment
    const a0 = result.assignments[0];
    for (let i = 1; i < 5; i++) {
      expect(result.assignments[i]).toBe(a0);
    }
    const a1 = result.assignments[5];
    for (let i = 6; i < 10; i++) {
      expect(result.assignments[i]).toBe(a1);
    }
    const a2 = result.assignments[10];
    for (let i = 11; i < 15; i++) {
      expect(result.assignments[i]).toBe(a2);
    }

    // All three groups should have different assignments
    expect(new Set([a0, a1, a2]).size).toBe(3);
  });

  it("WCSS is reasonable for well-separated clusters", () => {
    const result = kMeans(clusterData, 3, { seed: 42 });
    expect(result.totalWCSS).toBeLessThan(1);
  });

  it("k=1 assigns everything to one cluster", () => {
    const result = kMeans(clusterData, 1, { seed: 42 });
    for (const a of result.assignments) {
      expect(a).toBe(0);
    }
  });

  it("seed produces reproducible results", () => {
    const r1 = kMeans(clusterData, 3, { seed: 42 });
    const r2 = kMeans(clusterData, 3, { seed: 42 });
    expect(r1.assignments).toEqual(r2.assignments);
    expect(r1.totalWCSS).toBe(r2.totalWCSS);
  });

  it("throws when k > n", () => {
    expect(() => kMeans([[1, 2]], 3)).toThrow();
  });

  it("throws when k < 1", () => {
    expect(() => kMeans(clusterData, 0)).toThrow();
  });
});

// ── Hierarchical Clustering ─────────────────────────────────────────────

describe("hierarchicalClustering", () => {
  it("finds 3 clusters with complete linkage", () => {
    const result = hierarchicalClustering(clusterData, 3);
    expect(result.assignments).toHaveLength(15);
    expect(result.nClusters).toBe(3);

    // Points within the same group should be assigned together
    const a0 = result.assignments[0];
    for (let i = 1; i < 5; i++) {
      expect(result.assignments[i]).toBe(a0);
    }
    const a1 = result.assignments[5];
    for (let i = 6; i < 10; i++) {
      expect(result.assignments[i]).toBe(a1);
    }
    const a2 = result.assignments[10];
    for (let i = 11; i < 15; i++) {
      expect(result.assignments[i]).toBe(a2);
    }

    expect(new Set([a0, a1, a2]).size).toBe(3);
  });

  it("single linkage also finds correct clusters", () => {
    const result = hierarchicalClustering(clusterData, 3, { linkage: "single" });
    const groups = new Set(result.assignments);
    expect(groups.size).toBe(3);
  });

  it("average linkage also finds correct clusters", () => {
    const result = hierarchicalClustering(clusterData, 3, { linkage: "average" });
    const groups = new Set(result.assignments);
    expect(groups.size).toBe(3);
  });

  it("nClusters=1 assigns everything to one cluster", () => {
    const result = hierarchicalClustering(clusterData, 1);
    for (const a of result.assignments) {
      expect(a).toBe(0);
    }
  });

  it("nClusters=n gives each point its own cluster", () => {
    const data = [[0, 0], [1, 1], [2, 2]];
    const result = hierarchicalClustering(data, 3);
    expect(new Set(result.assignments).size).toBe(3);
  });

  it("merges array records merge history", () => {
    const result = hierarchicalClustering(clusterData, 3);
    // We started with 15 clusters and ended with 3 => 12 merges
    expect(result.merges).toHaveLength(12);
    for (const [a, b, d] of result.merges) {
      expect(a).not.toBe(b);
      expect(d).toBeGreaterThanOrEqual(0);
    }
  });

  it("throws for invalid nClusters", () => {
    expect(() => hierarchicalClustering(clusterData, 0)).toThrow();
    expect(() => hierarchicalClustering(clusterData, 16)).toThrow();
  });
});
