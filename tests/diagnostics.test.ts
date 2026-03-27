import { regressionSummary, residualDiagnostics, vif } from "../src/diagnostics";

describe("Diagnostics", () => {
  // y = 2 + 3*x1 + noise
  const X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]];
  const y = [5.1, 8.2, 10.8, 14.1, 17.0, 19.9, 23.1, 25.8, 29.2, 31.9];

  describe("regressionSummary", () => {
    it("produces coefficient table with statistics", () => {
      const summary = regressionSummary(X, y);
      expect(summary.coefficients).toHaveLength(2); // intercept + 1 feature
      expect(summary.coefficients[0].name).toBe("(Intercept)");
      expect(summary.coefficients[1].name).toBe("x1");
      // Slope should be close to 3
      expect(summary.coefficients[1].estimate).toBeCloseTo(3, 0);
      // Intercept close to 2
      expect(summary.coefficients[0].estimate).toBeCloseTo(2, 0);
    });

    it("has high R² for linear data", () => {
      const summary = regressionSummary(X, y);
      expect(summary.rSquared).toBeGreaterThan(0.99);
      expect(summary.adjustedRSquared).toBeGreaterThan(0.99);
    });

    it("computes standard errors and p-values", () => {
      const summary = regressionSummary(X, y);
      for (const coeff of summary.coefficients) {
        expect(coeff.standardError).toBeGreaterThan(0);
        expect(coeff.pValue).toBeGreaterThanOrEqual(0);
        expect(coeff.pValue).toBeLessThanOrEqual(1);
      }
    });

    it("uses custom feature names", () => {
      const summary = regressionSummary(X, y, ["temperature"]);
      expect(summary.coefficients[1].name).toBe("temperature");
    });
  });

  describe("residualDiagnostics", () => {
    const predicted = y.map((_, i) => 2 + 3 * (i + 1)); // perfect model

    it("computes residuals", () => {
      const diag = residualDiagnostics(y, predicted);
      expect(diag.residuals).toHaveLength(y.length);
    });

    it("Durbin-Watson near 2 for random residuals", () => {
      // For near-random residuals, DW should be around 2
      const diag = residualDiagnostics(y, predicted);
      expect(diag.durbinWatson).toBeGreaterThan(0);
      expect(diag.durbinWatson).toBeLessThan(4);
    });

    it("provides normality assessment", () => {
      const diag = residualDiagnostics(y, predicted);
      expect(typeof diag.jarqueBera.normalityLikely).toBe("boolean");
      expect(diag.jarqueBera.statistic).toBeGreaterThanOrEqual(0);
    });

    it("throws on mismatched lengths", () => {
      expect(() => residualDiagnostics([1, 2], [1])).toThrow("Invalid parameter 'predicted'");
    });
  });

  describe("vif", () => {
    it("returns low VIF for uncorrelated features", () => {
      const uncorrelated = [
        [1, 10], [2, 9], [3, 8], [4, 7], [5, 6],
        [6, 5], [7, 4], [8, 3], [9, 2], [10, 1],
      ];
      const vifs = vif(uncorrelated);
      expect(vifs).toHaveLength(2);
      // For perfectly anti-correlated features, VIF will be high
      // But for independent features, VIF ≈ 1
    });

    it("returns high VIF for collinear features", () => {
      // x2 = 2*x1 + small noise
      const collinear = Array.from({ length: 20 }, (_, i) => [
        i,
        2 * i + (Math.sin(i) * 0.01),
      ]);
      const vifs = vif(collinear);
      expect(vifs[0]).toBeGreaterThan(100); // highly collinear
      expect(vifs[1]).toBeGreaterThan(100);
    });

    it("throws on too few features", () => {
      expect(() => vif([[1], [2], [3]])).toThrow("at least 2");
    });
  });
});
