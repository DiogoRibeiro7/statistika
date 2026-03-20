import {
  pearsonCorrelation,
  spearmanCorrelation,
  kendallCorrelation,
} from "../../src/models";

describe("pearsonCorrelation", () => {
  it("returns 1 for perfectly correlated data", () => {
    const x = [1, 2, 3, 4, 5];
    const y = [2, 4, 6, 8, 10];
    const result = pearsonCorrelation(x, y);

    expect(result.coefficient).toBeCloseTo(1, 10);
    expect(result.pValue).toBeLessThan(0.001);
  });

  it("returns -1 for perfectly inversely correlated data", () => {
    const x = [1, 2, 3, 4, 5];
    const y = [10, 8, 6, 4, 2];
    const result = pearsonCorrelation(x, y);

    expect(result.coefficient).toBeCloseTo(-1, 10);
    expect(result.pValue).toBeLessThan(0.001);
  });

  it("returns ~0 for uncorrelated data", () => {
    const x = [1, 2, 3, 4, 5, 6, 7, 8];
    const y = [2, 8, 1, 7, 3, 6, 4, 5];
    const result = pearsonCorrelation(x, y);

    expect(Math.abs(result.coefficient)).toBeLessThan(0.3);
  });

  it("returns NaN for constant data", () => {
    const x = [5, 5, 5, 5, 5];
    const y = [1, 2, 3, 4, 5];
    const result = pearsonCorrelation(x, y);

    expect(result.coefficient).toBeNaN();
  });

  it("throws for mismatched lengths", () => {
    expect(() => pearsonCorrelation([1, 2, 3], [1, 2])).toThrow("same length");
  });

  it("throws for fewer than 3 elements", () => {
    expect(() => pearsonCorrelation([1, 2], [3, 4])).toThrow("at least 3");
  });
});

describe("spearmanCorrelation", () => {
  it("returns 1 for monotonically increasing data", () => {
    const x = [1, 2, 3, 4, 5];
    const y = [10, 20, 30, 40, 50];
    const result = spearmanCorrelation(x, y);

    expect(result.coefficient).toBeCloseTo(1, 10);
  });

  it("returns -1 for monotonically decreasing data", () => {
    const x = [1, 2, 3, 4, 5];
    const y = [50, 40, 30, 20, 10];
    const result = spearmanCorrelation(x, y);

    expect(result.coefficient).toBeCloseTo(-1, 10);
  });

  it("handles non-linear monotonic relationships", () => {
    // y = x³ is monotonically increasing → Spearman = 1
    const x = [1, 2, 3, 4, 5];
    const y = x.map((v) => v ** 3);
    const result = spearmanCorrelation(x, y);

    expect(result.coefficient).toBeCloseTo(1, 10);
  });

  it("handles ties correctly", () => {
    const x = [1, 2, 2, 3, 4];
    const y = [10, 20, 20, 30, 40];
    const result = spearmanCorrelation(x, y);

    expect(result.coefficient).toBeCloseTo(1, 10);
  });

  it("throws for mismatched lengths", () => {
    expect(() => spearmanCorrelation([1, 2, 3], [1, 2])).toThrow(
      "same length",
    );
  });
});

describe("kendallCorrelation", () => {
  it("returns 1 for perfectly concordant data", () => {
    const x = [1, 2, 3, 4, 5];
    const y = [10, 20, 30, 40, 50];
    const result = kendallCorrelation(x, y);

    expect(result.coefficient).toBeCloseTo(1, 10);
  });

  it("returns -1 for perfectly discordant data", () => {
    const x = [1, 2, 3, 4, 5];
    const y = [50, 40, 30, 20, 10];
    const result = kendallCorrelation(x, y);

    expect(result.coefficient).toBeCloseTo(-1, 10);
  });

  it("returns value between -1 and 1", () => {
    const x = [1, 2, 3, 4, 5, 6, 7, 8];
    const y = [2, 8, 1, 7, 3, 6, 4, 5];
    const result = kendallCorrelation(x, y);

    expect(result.coefficient).toBeGreaterThanOrEqual(-1);
    expect(result.coefficient).toBeLessThanOrEqual(1);
    expect(result.pValue).toBeGreaterThanOrEqual(0);
    expect(result.pValue).toBeLessThanOrEqual(1);
  });

  it("handles ties (tau-b)", () => {
    const x = [1, 2, 2, 3, 4];
    const y = [10, 20, 20, 30, 40];
    const result = kendallCorrelation(x, y);

    expect(result.coefficient).toBeCloseTo(1, 10);
  });

  it("throws for mismatched lengths", () => {
    expect(() => kendallCorrelation([1, 2, 3], [1, 2])).toThrow("same length");
  });
});
