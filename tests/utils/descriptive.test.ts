import { mean, median, variance, stdDev, describe as describeStats } from "../../src/utils";

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
