import {
  sobolIndices,
  morrisMethod,
  fastMethod,
  correlationScreening,
  scatterPlotData,
} from "../src/sensitivity";

// Simple linear model: y = 3*x1 + x2 (x1 is more important)
const linearModel = (x: number[]) => 3 * x[0] + x[1];

// Non-linear model: y = x1^2 + x1*x2
const nonlinearModel = (x: number[]) => x[0] * x[0] + x[0] * x[1];

describe("sobolIndices", () => {
  it("identifies the most important variable in a linear model", () => {
    const result = sobolIndices(linearModel, 2, {
      nSamples: 500,
      seed: 42,
      bounds: [[0, 1], [0, 1]],
    });
    expect(result.firstOrder).toHaveLength(2);
    expect(result.totalOrder).toHaveLength(2);
    // x1 (weight 3) should have higher sensitivity than x2 (weight 1)
    expect(result.firstOrder[0]).toBeGreaterThan(result.firstOrder[1]);
  });

  it("first-order indices are between 0 and 1", () => {
    const result = sobolIndices(linearModel, 2, {
      nSamples: 500,
      seed: 42,
      bounds: [[0, 1], [0, 1]],
    });
    for (const s of result.firstOrder) {
      expect(s).toBeGreaterThan(-0.2); // allow small Monte Carlo error
      expect(s).toBeLessThan(1.2);
    }
  });

  it("total order is at least as large as first order", () => {
    const result = sobolIndices(nonlinearModel, 2, {
      nSamples: 500,
      seed: 42,
      bounds: [[0, 1], [0, 1]],
    });
    for (let i = 0; i < 2; i++) {
      expect(result.totalOrder[i]).toBeGreaterThanOrEqual(result.firstOrder[i] - 0.1);
    }
  });

  it("throws on invalid dimensions", () => {
    expect(() => sobolIndices(linearModel, 0, { seed: 1 })).toThrow();
  });
});

describe("morrisMethod", () => {
  it("identifies important variables via muStar", () => {
    const result = morrisMethod(linearModel, 2, {
      nTrajectories: 10,
      seed: 42,
      bounds: [[0, 1], [0, 1]],
    });
    expect(result.muStar).toHaveLength(2);
    expect(result.sigma).toHaveLength(2);
    expect(result.mu).toHaveLength(2);
    expect(result.rankings).toHaveLength(2);
    // x1 should be more important (higher muStar)
    expect(result.muStar[0]).toBeGreaterThan(result.muStar[1]);
  });

  it("muStar values are non-negative", () => {
    const result = morrisMethod(linearModel, 2, {
      nTrajectories: 10,
      seed: 42,
    });
    for (const m of result.muStar) {
      expect(m).toBeGreaterThanOrEqual(0);
    }
  });

  it("rankings are valid permutations", () => {
    const result = morrisMethod(linearModel, 2, {
      nTrajectories: 10,
      seed: 42,
    });
    const sorted = [...result.rankings].sort((a, b) => a - b);
    expect(sorted).toEqual([1, 2]);
  });
});

describe("fastMethod", () => {
  it("computes first-order sensitivity indices", () => {
    const result = fastMethod(linearModel, 2, {
      nSamples: 500,
      seed: 42,
      bounds: [[0, 1], [0, 1]],
    });
    expect(result.firstOrder).toHaveLength(2);
    expect(result.nSamples).toBeGreaterThan(0);
  });

  it("indices are non-negative", () => {
    const result = fastMethod(linearModel, 2, {
      nSamples: 500,
      seed: 42,
    });
    for (const s of result.firstOrder) {
      expect(s).toBeGreaterThanOrEqual(-0.1); // allow small numerical error
    }
  });
});

describe("correlationScreening", () => {
  it("computes correlations for sampled inputs and outputs", () => {
    // Generate sample data
    const nSamp = 100;
    const inputs: number[][] = [];
    const outputs: number[] = [];
    for (let i = 0; i < nSamp; i++) {
      const xi = [Math.sin(i * 0.17) * 0.5 + 0.5, Math.sin(i * 0.31) * 0.5 + 0.5];
      inputs.push(xi);
      outputs.push(linearModel(xi));
    }
    const result = correlationScreening(inputs, outputs);
    expect(result.pearson).toHaveLength(2);
    expect(result.spearman).toHaveLength(2);
    expect(result.partialCorrelation).toHaveLength(2);
    expect(result.rankings).toHaveLength(2);
  });

  it("correlations are between -1 and 1", () => {
    const nSamp = 100;
    const inputs: number[][] = [];
    const outputs: number[] = [];
    for (let i = 0; i < nSamp; i++) {
      const xi = [Math.sin(i * 0.17) * 0.5 + 0.5, Math.sin(i * 0.31) * 0.5 + 0.5];
      inputs.push(xi);
      outputs.push(linearModel(xi));
    }
    const result = correlationScreening(inputs, outputs);
    for (const r of result.pearson) {
      expect(r).toBeGreaterThanOrEqual(-1);
      expect(r).toBeLessThanOrEqual(1);
    }
  });
});

describe("scatterPlotData", () => {
  it("returns scatter data for each variable", () => {
    const nSamp = 50;
    const inputs: number[][] = [];
    const outputs: number[] = [];
    for (let i = 0; i < nSamp; i++) {
      const xi = [i / nSamp, Math.sin(i)];
      inputs.push(xi);
      outputs.push(linearModel(xi));
    }
    const result = scatterPlotData(inputs, outputs);
    expect(result.plots).toHaveLength(2);
    expect(result.plots[0].x).toHaveLength(nSamp);
    expect(result.plots[0].y).toHaveLength(nSamp);
    expect(result.plots[0].variable).toBe(0);
    expect(result.plots[1].variable).toBe(1);
  });
});
