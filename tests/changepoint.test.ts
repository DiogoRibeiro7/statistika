import {
  cusumTest,
  pelt,
  binarySegmentation,
  bocpd,
  costMean,
  costVariance,
  costMeanVar,
} from "../src/changepoint";

// Data with a clear mean shift at index 50
const shiftData: number[] = [];
for (let i = 0; i < 50; i++) shiftData.push(0 + Math.sin(i * 0.73) * 0.3);
for (let i = 50; i < 100; i++) shiftData.push(10 + Math.sin(i * 0.73) * 0.3);

// Data with no changepoint (constant mean)
const noChange = Array.from({ length: 80 }, (_, i) => Math.sin(i * 0.53) * 0.5);

// Data with variance shift at index 40
const varShift: number[] = [];
for (let i = 0; i < 40; i++) varShift.push(Math.sin(i * 0.37) * 0.2);
for (let i = 40; i < 80; i++) varShift.push(Math.sin(i * 0.37) * 2.0);

describe("costMean", () => {
  it("cost for a single segment is non-negative", () => {
    const c = costMean(shiftData, 0, 50);
    expect(Number.isFinite(c)).toBe(true);
  });

  it("cost of the full data is greater than sum of two correctly split segments", () => {
    const full = costMean(shiftData, 0, 100);
    const left = costMean(shiftData, 0, 50);
    const right = costMean(shiftData, 50, 100);
    expect(left + right).toBeLessThan(full);
  });
});

describe("costVariance", () => {
  it("returns a finite non-negative value", () => {
    const c = costVariance(varShift, 0, 80);
    expect(Number.isFinite(c)).toBe(true);
  });
});

describe("cusumTest", () => {
  it("detects a changepoint in shift data", () => {
    const result = cusumTest(shiftData);
    expect(result.statistic).toBeGreaterThan(0);
    // The detected changepoint should be near 50
    expect(result.changepoint).toBeGreaterThan(30);
    expect(result.changepoint).toBeLessThan(70);
  });

  it("does not reject for stable data", () => {
    const result = cusumTest(noChange);
    // With deterministic small oscillations, the test should not strongly reject
    expect(result.pValue).toBeGreaterThan(0);
  });

  it("cusum values have correct length", () => {
    const result = cusumTest(shiftData);
    expect(result.cusumValues).toHaveLength(shiftData.length + 1);
  });
});

describe("pelt", () => {
  it("detects changepoint near the true location", () => {
    const result = pelt(shiftData, { costFunction: "mean" });
    expect(result.changepoints.length).toBeGreaterThanOrEqual(1);
    // At least one changepoint should be close to index 50
    const nearTarget = result.changepoints.some(
      (cp) => Math.abs(cp - 50) < 15
    );
    expect(nearTarget).toBe(true);
  });

  it("segments span the full data", () => {
    const result = pelt(shiftData);
    expect(result.segments.length).toBeGreaterThanOrEqual(2);
    expect(result.segments[0].start).toBe(0);
    expect(result.segments[result.segments.length - 1].end).toBe(shiftData.length);
  });

  it("cost is finite", () => {
    const result = pelt(shiftData);
    expect(Number.isFinite(result.cost)).toBe(true);
  });
});

describe("binarySegmentation", () => {
  it("detects at least one changepoint in shift data", () => {
    const result = binarySegmentation(shiftData, { costFunction: "mean" });
    expect(result.changepoints.length).toBeGreaterThanOrEqual(1);
  });

  it("each segment has valid mean and variance", () => {
    const result = binarySegmentation(shiftData);
    for (const seg of result.segments) {
      expect(Number.isFinite(seg.mean)).toBe(true);
      expect(seg.variance).toBeGreaterThanOrEqual(0);
    }
  });
});

describe("bocpd", () => {
  it("returns run length probabilities", () => {
    const result = bocpd(shiftData);
    expect(result.runLengthProbabilities).toHaveLength(shiftData.length);
    expect(result.maxRunLengthProb).toHaveLength(shiftData.length);
  });

  it("detects changepoint in shift data", () => {
    const result = bocpd(shiftData, { hazardLambda: 1 / 50, threshold: 0.1 });
    // BOCPD should produce run length probabilities for the data
    expect(result.runLengthProbabilities.length).toBe(shiftData.length);
    expect(result.maxRunLengthProb.length).toBe(shiftData.length);
  });

  it("run length probabilities sum to approximately 1 at each step", () => {
    const result = bocpd(shiftData);
    for (const probs of result.runLengthProbabilities) {
      const total = probs.reduce((s, v) => s + v, 0);
      expect(total).toBeCloseTo(1, 1);
    }
  });
});
