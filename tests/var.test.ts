import {
  varFit,
  varLagSelect,
  grangerCausality,
  impulseResponse,
  varianceDecomposition,
  varForecast,
  johansenTest,
} from "../src/var";

// Synthetic VAR(1) data
const data: number[][] = [];
let x = [1, 2];
for (let t = 0; t < 100; t++) {
  x = [
    0.5 * x[0] + 0.1 * x[1] + (Math.sin(t * 0.73) * 0.5),
    0.2 * x[0] + 0.3 * x[1] + (Math.sin(t * 1.21 + 1) * 0.5),
  ];
  data.push([...x]);
}

describe("varFit", () => {
  it("fits a VAR(1) model and returns valid structure", () => {
    const result = varFit(data, 1);
    expect(result.p).toBe(1);
    expect(result.k).toBe(2);
    expect(result.coefficients).toHaveLength(1);
    expect(result.coefficients[0]).toHaveLength(2);
    expect(result.coefficients[0][0]).toHaveLength(2);
    expect(result.intercept).toHaveLength(2);
    expect(result.nobs).toBe(99);
  });

  it("residuals have correct dimensions", () => {
    const result = varFit(data, 1);
    expect(result.residuals).toHaveLength(99);
    expect(result.residuals[0]).toHaveLength(2);
  });

  it("AIC and BIC are finite", () => {
    const result = varFit(data, 1);
    expect(Number.isFinite(result.aic)).toBe(true);
    expect(Number.isFinite(result.bic)).toBe(true);
  });

  it("sigma is a valid 2x2 covariance matrix", () => {
    const result = varFit(data, 1);
    expect(result.sigma).toHaveLength(2);
    expect(result.sigma[0]).toHaveLength(2);
    // Diagonal elements should be positive
    expect(result.sigma[0][0]).toBeGreaterThan(0);
    expect(result.sigma[1][1]).toBeGreaterThan(0);
  });
});

describe("varLagSelect", () => {
  it("selects optimal lag order", () => {
    const result = varLagSelect(data, 3);
    expect(result.aic).toHaveLength(3);
    expect(result.bic).toHaveLength(3);
    expect(result.hq).toHaveLength(3);
    expect(result.aicLag).toBeGreaterThanOrEqual(1);
    expect(result.aicLag).toBeLessThanOrEqual(3);
    expect(result.bicLag).toBeGreaterThanOrEqual(1);
    expect(result.bicLag).toBeLessThanOrEqual(3);
  });

  it("AIC and BIC lags are positive integers", () => {
    const result = varLagSelect(data, 2);
    expect(Number.isInteger(result.aicLag)).toBe(true);
    expect(Number.isInteger(result.bicLag)).toBe(true);
    expect(result.aicLag).toBeGreaterThanOrEqual(1);
    expect(result.bicLag).toBeGreaterThanOrEqual(1);
  });
});

describe("grangerCausality", () => {
  it("returns valid test statistics", () => {
    const fit = varFit(data, 1);
    const result = grangerCausality(data, 1, 0, 1);
    expect(result.fStatistic).toBeGreaterThanOrEqual(0);
    expect(result.pValue).toBeGreaterThanOrEqual(0);
    expect(result.pValue).toBeLessThanOrEqual(1);
    expect(result.cause).toBe(0);
    expect(result.effect).toBe(1);
  });

  it("degrees of freedom are positive", () => {
    const result = grangerCausality(data, 2, 1, 0);
    expect(result.df[0]).toBeGreaterThan(0);
    expect(result.df[1]).toBeGreaterThan(0);
  });
});

describe("impulseResponse", () => {
  it("returns IRF with correct horizon", () => {
    const fit = varFit(data, 1);
    const irf = impulseResponse(fit, 10);
    expect(irf.horizon).toBe(10);
    expect(irf.irf).toHaveLength(11); // horizon + 1 (includes h=0)
    expect(irf.irf[0]).toHaveLength(2);
    expect(irf.irf[0][0]).toHaveLength(2);
  });

  it("initial shock is identity-like", () => {
    const fit = varFit(data, 1);
    const irf = impulseResponse(fit, 5);
    // At h=0, shock to variable j should have non-zero effect on variable j
    expect(irf.irf[0][0][0]).not.toBe(0);
  });
});

describe("varForecast", () => {
  it("produces forecasts with correct dimensions", () => {
    const fit = varFit(data, 1);
    const fc = varForecast(fit, 5);
    expect(fc.forecast).toHaveLength(5);
    expect(fc.forecast[0]).toHaveLength(2);
  });

  it("forecast values are finite numbers", () => {
    const fit = varFit(data, 1);
    const fc = varForecast(fit, 10);
    for (const row of fc.forecast) {
      for (const v of row) {
        expect(Number.isFinite(v)).toBe(true);
      }
    }
  });
});
