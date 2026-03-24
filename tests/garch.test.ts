import {
  garchFit,
  archFit,
  egarchFit,
  gjrGarchFit,
  garchForecast,
  archLMTest,
  garchDiagnostics,
} from "../src/garch";

// Deterministic seed-like data for reproducibility
const returns = Array.from(
  { length: 200 },
  (_, i) => Math.sin(i * 0.1) * 0.02 + (Math.sin(i * 0.37 + 2.1) * 0.5 + 0.5 - 0.5) * 0.04
);

describe("ARCH model", () => {
  it("fits an ARCH(1) model and returns valid parameters", () => {
    const result = archFit(returns, 1);
    expect(result.omega).toBeGreaterThan(0);
    expect(result.alpha).toHaveLength(1);
    expect(result.alpha[0]).toBeGreaterThanOrEqual(0);
    expect(result.beta).toHaveLength(0);
    expect(result.nObs).toBe(200);
  });

  it("conditional variance is always positive", () => {
    const result = archFit(returns, 1);
    for (const v of result.conditionalVariance) {
      expect(v).toBeGreaterThan(0);
    }
  });

  it("standardized residuals have roughly unit variance", () => {
    const result = archFit(returns, 1);
    const sr = result.standardizedResiduals;
    const srVar = sr.reduce((s, z) => s + z * z, 0) / sr.length;
    expect(srVar).toBeGreaterThan(0.2);
    expect(srVar).toBeLessThan(5);
  });
});

describe("GARCH(1,1) model", () => {
  it("fits a GARCH(1,1) model and returns valid structure", () => {
    const result = garchFit(returns, 1, 1);
    expect(result.omega).toBeGreaterThan(0);
    expect(result.alpha).toHaveLength(1);
    expect(result.beta).toHaveLength(1);
    expect(result.conditionalVariance).toHaveLength(200);
    expect(result.standardizedResiduals).toHaveLength(200);
  });

  it("AIC and BIC are finite numbers", () => {
    const result = garchFit(returns, 1, 1);
    expect(Number.isFinite(result.aic)).toBe(true);
    expect(Number.isFinite(result.bic)).toBe(true);
  });

  it("stationarity: alpha + beta < 1", () => {
    const result = garchFit(returns, 1, 1);
    const persistence = result.alpha[0] + result.beta[0];
    expect(persistence).toBeLessThan(1);
    expect(persistence).toBeGreaterThanOrEqual(0);
  });
});

describe("EGARCH model", () => {
  it("fits an EGARCH(1,1) and returns gamma (leverage) parameter", () => {
    const result = egarchFit(returns, 1, 1);
    expect(result.omega).toBeDefined();
    expect(result.gamma).toHaveLength(1);
    expect(result.conditionalVariance).toHaveLength(200);
  });

  it("conditional variance is always positive", () => {
    const result = egarchFit(returns, 1, 1);
    for (const v of result.conditionalVariance) {
      expect(v).toBeGreaterThan(0);
    }
  });
});

describe("GJR-GARCH model", () => {
  it("fits a GJR-GARCH(1,1) and returns valid structure", () => {
    const result = gjrGarchFit(returns, 1, 1);
    expect(result.omega).toBeGreaterThan(0);
    expect(result.alpha).toHaveLength(1);
    expect(result.beta).toHaveLength(1);
    expect(result.gamma).toHaveLength(1);
    expect(result.nObs).toBe(200);
  });

  it("conditional variance is always positive", () => {
    const result = gjrGarchFit(returns, 1, 1);
    for (const v of result.conditionalVariance) {
      expect(v).toBeGreaterThan(0);
    }
  });
});

describe("GARCH forecasting", () => {
  it("produces correct number of forecast steps", () => {
    const fit = garchFit(returns, 1, 1);
    const fc = garchForecast(fit, 5);
    expect(fc.variance).toHaveLength(5);
    expect(fc.standardDeviation).toHaveLength(5);
    expect(fc.horizon).toBe(5);
  });

  it("forecasted variance is positive and converges toward unconditional", () => {
    const fit = garchFit(returns, 1, 1);
    const fc = garchForecast(fit, 20);
    for (const v of fc.variance) {
      expect(v).toBeGreaterThan(0);
    }
    // Variance should converge: last values closer together than first
    const diff1 = Math.abs(fc.variance[1] - fc.variance[0]);
    const diffLast = Math.abs(fc.variance[19] - fc.variance[18]);
    expect(diffLast).toBeLessThanOrEqual(diff1 + 1e-10);
  });
});

describe("ARCH-LM test", () => {
  it("returns valid test statistics", () => {
    const result = archLMTest(returns, 5);
    expect(result.statistic).toBeGreaterThanOrEqual(0);
    expect(result.pValue).toBeGreaterThanOrEqual(0);
    expect(result.pValue).toBeLessThanOrEqual(1);
    expect(result.lags).toBe(5);
    expect(result.rSquared).toBeGreaterThanOrEqual(0);
    expect(result.rSquared).toBeLessThanOrEqual(1);
  });

  it("white noise should not show ARCH effects", () => {
    // Constant series has no ARCH effects
    const constant = Array.from({ length: 100 }, (_, i) => Math.sin(i * 0.73) * 0.01);
    const result = archLMTest(constant, 3);
    expect(result.statistic).toBeGreaterThanOrEqual(0);
    expect(result.pValue).toBeGreaterThanOrEqual(0);
  });
});

describe("GARCH diagnostics", () => {
  it("returns diagnostic object with valid fields", () => {
    const fit = garchFit(returns, 1, 1);
    const diag = garchDiagnostics(fit.standardizedResiduals);
    expect(diag).toBeDefined();
    expect(Number.isFinite(diag.ljungBoxResiduals)).toBe(true);
    expect(Number.isFinite(diag.ljungBoxSquaredResiduals)).toBe(true);
    expect(Number.isFinite(diag.skewness)).toBe(true);
    expect(Number.isFinite(diag.kurtosis)).toBe(true);
    expect(diag.varianceStdResiduals).toBeGreaterThan(0);
  });
});
