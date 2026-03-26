import { mean, median, variance, stdDev, standardDeviation, skewness, kurtosis, percentile, describe as describeStats } from "../../src/utils";

describe("mean", () => {
  it("computes the arithmetic mean", () => {
    expect(mean([1, 2, 3, 4, 5])).toBe(3);
  });

  it("throws on empty dataset", () => {
    expect(() => mean([])).toThrow();
  });
});

describe("median", () => {
  it("returns middle value for odd-length arrays", () => {
    expect(median([3, 1, 2])).toBe(2);
  });

  it("returns average of two middle values for even-length arrays", () => {
    expect(median([1, 2, 3, 4])).toBe(2.5);
  });
});

describe("variance", () => {
  it("computes sample variance", () => {
    expect(variance([2, 4, 4, 4, 5, 5, 7, 9])).toBeCloseTo(4.571, 2);
  });

  it("computes population variance", () => {
    expect(variance([2, 4, 4, 4, 5, 5, 7, 9], false)).toBe(4);
  });
});

describe("stdDev", () => {
  it("computes sample standard deviation", () => {
    expect(stdDev([2, 4, 4, 4, 5, 5, 7, 9])).toBeCloseTo(2.138, 2);
  });
});

describe("skewness", () => {
  it("returns 0 for a symmetric dataset", () => {
    expect(skewness([1, 2, 3, 4, 5])).toBeCloseTo(0, 5);
  });

  it("returns positive skewness for right-skewed data", () => {
    expect(skewness([1, 1, 1, 1, 1, 1, 10])).toBeGreaterThan(0);
  });

  it("returns negative skewness for left-skewed data", () => {
    expect(skewness([1, 10, 10, 10, 10, 10, 10])).toBeLessThan(0);
  });

  it("throws for fewer than 3 elements", () => {
    expect(() => skewness([1, 2])).toThrow();
  });

  it("returns 0 when all values are equal", () => {
    expect(skewness([5, 5, 5, 5])).toBe(0);
  });
});

describe("kurtosis", () => {
  it("returns near-zero excess kurtosis for normal-like data", () => {
    // Large uniform-ish dataset: excess kurtosis of uniform is -1.2
    const data = Array.from({ length: 1000 }, (_, i) => i);
    expect(kurtosis(data)).toBeCloseTo(-1.2, 1);
  });

  it("returns positive excess kurtosis for heavy-tailed data", () => {
    // Data with outliers
    const data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 100];
    expect(kurtosis(data)).toBeGreaterThan(0);
  });

  it("throws for fewer than 4 elements", () => {
    expect(() => kurtosis([1, 2, 3])).toThrow();
  });

  it("returns 0 when all values are equal", () => {
    expect(kurtosis([5, 5, 5, 5])).toBe(0);
  });
});

describe("percentile", () => {
  it("computes the 50th percentile (median)", () => {
    expect(percentile([1, 2, 3, 4, 5], 50)).toBe(3);
  });

  it("computes the 25th percentile", () => {
    expect(percentile([1, 2, 3, 4, 5], 25)).toBe(2);
  });

  it("computes the 75th percentile", () => {
    expect(percentile([1, 2, 3, 4, 5], 75)).toBe(4);
  });

  it("returns min for 0th percentile", () => {
    expect(percentile([5, 3, 1, 4, 2], 0)).toBe(1);
  });

  it("returns max for 100th percentile", () => {
    expect(percentile([5, 3, 1, 4, 2], 100)).toBe(5);
  });

  it("interpolates between values", () => {
    expect(percentile([10, 20, 30, 40], 30)).toBeCloseTo(19, 0);
  });

  it("throws on empty dataset", () => {
    expect(() => percentile([], 50)).toThrow();
  });

  it("throws for out-of-range percentile", () => {
    expect(() => percentile([1, 2], -1)).toThrow();
    expect(() => percentile([1, 2], 101)).toThrow();
  });
});

describe("describeStats", () => {
  it("returns a full descriptive stats summary", () => {
    const stats = describeStats([1, 2, 3, 4, 5]);
    expect(stats.count).toBe(5);
    expect(stats.mean).toBe(3);
    expect(stats.median).toBe(3);
    expect(stats.min).toBe(1);
    expect(stats.max).toBe(5);
  });
});

describe("API aliases", () => {
  it("standardDeviation is an alias for stdDev", () => {
    expect(standardDeviation).toBe(stdDev);
  });

  it("standardDeviation produces identical results to stdDev", () => {
    const data = [2, 4, 4, 4, 5, 5, 7, 9];
    expect(standardDeviation(data)).toBe(stdDev(data));
    expect(standardDeviation(data, false)).toBe(stdDev(data, false));
  });
});
