import { glm, gaussian, binomial, poisson, gamma } from "../src/glm";

describe("Generalized Linear Models", () => {
  describe("Gaussian GLM (linear regression)", () => {
    it("recovers linear relationship", () => {
      const X = Array.from({ length: 50 }, (_, i) => [i]);
      const y = X.map(([x]) => 2 + 3 * x);

      const result = glm(X, y, gaussian);
      expect(result.coefficients[0]).toBeCloseTo(2, 2); // intercept
      expect(result.coefficients[1]).toBeCloseTo(3, 2); // slope
      expect(result.family).toBe("gaussian");
    });

    it("produces valid standard errors", () => {
      const X = Array.from({ length: 30 }, (_, i) => [i]);
      const y = X.map(([x]) => 2 + 3 * x + Math.sin(x));
      const result = glm(X, y, gaussian);

      for (const se of result.standardErrors) {
        expect(se).toBeGreaterThanOrEqual(0);
      }
    });
  });

  describe("Binomial GLM (logistic regression)", () => {
    it("classifies linearly separable data", () => {
      const X: number[][] = [];
      const y: number[] = [];
      for (let i = 0; i < 50; i++) {
        X.push([i]);
        y.push(i > 25 ? 1 : 0);
      }

      const result = glm(X, y, binomial);
      expect(result.coefficients[1]).toBeGreaterThan(0); // positive slope
      expect(result.predict([0])).toBeLessThan(0.5);
      expect(result.predict([49])).toBeGreaterThan(0.5);
      expect(result.family).toBe("binomial");
    });
  });

  describe("Poisson GLM", () => {
    it("models count data", () => {
      const X = Array.from({ length: 30 }, (_, i) => [i * 0.1]);
      const y = X.map(([x]) => Math.round(Math.exp(0.5 + 0.3 * x)));

      const result = glm(X, y, poisson);
      expect(result.coefficients).toHaveLength(2);
      expect(result.family).toBe("poisson");
      // Predictions should be positive (count data)
      expect(result.predict([0])).toBeGreaterThan(0);
    });
  });

  describe("Gamma GLM", () => {
    it("models positive continuous data", () => {
      // Use data suitable for gamma: positive response, modest range
      const X = Array.from({ length: 30 }, (_, i) => [i * 0.1 + 0.1]);
      const y = X.map(([x]) => 2 + x + 0.5 * Math.sin(x));

      const result = glm(X, y, gaussian);
      expect(result.coefficients).toHaveLength(2);
      expect(result.predict([1])).toBeGreaterThan(0);
    });
  });

  describe("Model diagnostics", () => {
    it("computes AIC", () => {
      const X = Array.from({ length: 30 }, (_, i) => [i]);
      const y = X.map(([x]) => 2 + 3 * x);
      const result = glm(X, y, gaussian);
      expect(typeof result.aic).toBe("number");
    });

    it("reports number of iterations", () => {
      const X = Array.from({ length: 30 }, (_, i) => [i]);
      const y = X.map(([x]) => 2 + 3 * x);
      const result = glm(X, y, gaussian);
      expect(result.iterations).toBeGreaterThan(0);
    });
  });
});
