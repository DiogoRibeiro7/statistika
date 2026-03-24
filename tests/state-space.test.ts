import {
  kalmanFilter,
  kalmanSmoother,
  stateSpaceEM,
  localLevelModel,
  localLinearTrendModel,
  stateSpacePredict,
} from "../src/state-space";
import type { StateSpaceModel } from "../src/state-space";

// Simple local level data: random walk + noise
const observations: number[] = [];
let state = 0;
for (let t = 0; t < 100; t++) {
  state += Math.sin(t * 0.53) * 0.3;
  observations.push(state + Math.sin(t * 1.17) * 0.5);
}

// Define a simple local level SSM
const localLevelSSM: StateSpaceModel = {
  F: [[1]],
  H: [[1]],
  Q: [[0.1]],
  R: [[0.5]],
  x0: [0],
  P0: [[1]],
};

describe("kalmanFilter", () => {
  it("returns filtered states with correct dimensions", () => {
    const obsMatrix = observations.map((o) => [o]);
    const result = kalmanFilter(localLevelSSM, obsMatrix);
    expect(result.states).toHaveLength(100);
    expect(result.states[0]).toHaveLength(1);
    expect(result.covariances).toHaveLength(100);
    expect(result.predictions).toHaveLength(100);
  });

  it("log-likelihood is finite", () => {
    const obsMatrix = observations.map((o) => [o]);
    const result = kalmanFilter(localLevelSSM, obsMatrix);
    expect(Number.isFinite(result.logLikelihood)).toBe(true);
  });

  it("innovations have correct length", () => {
    const obsMatrix = observations.map((o) => [o]);
    const result = kalmanFilter(localLevelSSM, obsMatrix);
    expect(result.innovations).toHaveLength(100);
    expect(result.innovationCovariances).toHaveLength(100);
  });

  it("filtered covariances are positive", () => {
    const obsMatrix = observations.map((o) => [o]);
    const result = kalmanFilter(localLevelSSM, obsMatrix);
    for (const P of result.covariances) {
      expect(P[0][0]).toBeGreaterThan(0);
    }
  });
});

describe("kalmanSmoother", () => {
  it("returns smoothed states", () => {
    const obsMatrix = observations.map((o) => [o]);
    const result = kalmanSmoother(localLevelSSM, obsMatrix);
    expect(result.smoothedStates).toHaveLength(100);
    expect(result.smoothedCovariances).toHaveLength(100);
  });

  it("smoothed covariances are no larger than filtered covariances", () => {
    const obsMatrix = observations.map((o) => [o]);
    const result = kalmanSmoother(localLevelSSM, obsMatrix);
    for (let t = 0; t < 100; t++) {
      expect(result.smoothedCovariances[t][0][0]).toBeLessThanOrEqual(
        result.covariances[t][0][0] + 1e-10
      );
    }
  });
});

describe("localLevelModel", () => {
  it("fits a local level model and returns filter and smoother results", () => {
    const result = localLevelModel(observations);
    expect(result).toBeDefined();
    expect(result.filterResult).toBeDefined();
    expect(result.smootherResult).toBeDefined();
    expect(result.filterResult.states).toHaveLength(100);
  });
});

describe("localLinearTrendModel", () => {
  it("fits a local linear trend model", () => {
    const trending = Array.from({ length: 80 }, (_, i) => i * 0.5 + Math.sin(i * 0.31) * 2);
    const result = localLinearTrendModel(trending);
    expect(result).toBeDefined();
    expect(result.filterResult.states).toHaveLength(80);
    // State has 2 components: level and slope
    expect(result.filterResult.states[0]).toHaveLength(2);
  });
});

describe("stateSpacePredict", () => {
  it("forecasts ahead with correct dimensions", () => {
    const obsMatrix = observations.map((o) => [o]);
    const filterResult = kalmanFilter(localLevelSSM, obsMatrix);
    const pred = stateSpacePredict(localLevelSSM, filterResult, 10);
    expect(pred.stateForecast).toHaveLength(10);
    expect(pred.observationForecast).toHaveLength(10);
    expect(pred.lowerBound).toHaveLength(10);
    expect(pred.upperBound).toHaveLength(10);
  });

  it("prediction uncertainty grows with horizon", () => {
    const obsMatrix = observations.map((o) => [o]);
    const filterResult = kalmanFilter(localLevelSSM, obsMatrix);
    const pred = stateSpacePredict(localLevelSSM, filterResult, 10);
    const width1 = pred.upperBound[0][0] - pred.lowerBound[0][0];
    const width10 = pred.upperBound[9][0] - pred.lowerBound[9][0];
    expect(width10).toBeGreaterThan(width1);
  });
});
