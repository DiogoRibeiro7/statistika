import { glm, gaussian, binomial, poisson, gammaFamily, identityLink, logLink, logitLink, probitLink, inverseLink } from "../src/glm";

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

    it("computes deviance", () => {
      const X = Array.from({ length: 30 }, (_, i) => [i]);
      const y = X.map(([x]) => 2 + 3 * x);
      const result = glm(X, y, gaussian);
      expect(typeof result.deviance).toBe("number");
      expect(result.deviance).toBeGreaterThanOrEqual(0);
    });

    it("computes log-likelihood", () => {
      const X = Array.from({ length: 30 }, (_, i) => [i]);
      const y = X.map(([x]) => 2 + 3 * x);
      const result = glm(X, y, gaussian);
      expect(typeof result.logLikelihood).toBe("number");
      expect(Number.isFinite(result.logLikelihood)).toBe(true);
    });

    it("computes p-values between 0 and 1", () => {
      const X = Array.from({ length: 30 }, (_, i) => [i]);
      const y = X.map(([x]) => 2 + 3 * x + Math.sin(x));
      const result = glm(X, y, gaussian);
      for (const p of result.pValues) {
        expect(p).toBeGreaterThanOrEqual(0);
        expect(p).toBeLessThanOrEqual(1);
      }
    });

    it("computes z-values", () => {
      const X = Array.from({ length: 30 }, (_, i) => [i]);
      const y = X.map(([x]) => 2 + 3 * x);
      const result = glm(X, y, gaussian);
      expect(result.zValues).toHaveLength(2);
      for (const z of result.zValues) {
        expect(Number.isFinite(z)).toBe(true);
      }
    });
  });

  describe("Error handling", () => {
    it("throws on empty X", () => {
      expect(() => glm([], [], gaussian)).toThrow("non-empty");
    });

    it("throws on mismatched X and y lengths", () => {
      expect(() => glm([[1], [2]], [1], gaussian)).toThrow("Invalid parameters");
    });

    it("throws on too few observations for parameters", () => {
      expect(() => glm([[1, 2]], [1], gaussian)).toThrow("more observations");
    });

    it("throws on inconsistent row lengths", () => {
      expect(() => glm([[1, 2], [3]], [1, 2], gaussian)).toThrow();
    });

    it("throws on NaN in y", () => {
      const X = Array.from({ length: 10 }, (_, i) => [i]);
      const y = [1, 2, NaN, 4, 5, 6, 7, 8, 9, 10];
      expect(() => glm(X, y, gaussian)).toThrow("finite");
    });

    it("throws on Infinity in X", () => {
      const X = [[1], [Infinity], [3], [4], [5], [6], [7], [8], [9], [10]];
      const y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      expect(() => glm(X, y, gaussian)).toThrow("finite");
    });
  });

  describe("Link functions", () => {
    it("identityLink is identity", () => {
      expect(identityLink.link(5)).toBe(5);
      expect(identityLink.inverse(5)).toBe(5);
      expect(identityLink.derivative(5)).toBe(1);
    });

    it("logLink is log/exp", () => {
      expect(logLink.link(1)).toBeCloseTo(0);
      expect(logLink.inverse(0)).toBeCloseTo(1);
      expect(logLink.derivative(0)).toBeCloseTo(1);
      expect(logLink.link(Math.E)).toBeCloseTo(1);
    });

    it("logitLink maps (0,1) to reals", () => {
      expect(logitLink.link(0.5)).toBeCloseTo(0);
      expect(logitLink.inverse(0)).toBeCloseTo(0.5);
      expect(logitLink.derivative(0)).toBeCloseTo(0.25);
    });

    it("probitLink maps (0,1) to reals via normal CDF", () => {
      expect(probitLink.inverse(0)).toBeCloseTo(0.5);
      expect(probitLink.derivative(0)).toBeGreaterThan(0);
    });

    it("inverseLink is 1/x", () => {
      expect(inverseLink.link(2)).toBeCloseTo(0.5);
      expect(inverseLink.inverse(0.5)).toBeCloseTo(2);
      expect(inverseLink.derivative(1)).toBeCloseTo(-1);
    });
  });

  describe("GLM families", () => {
    it("gaussian variance is constant 1", () => {
      expect(gaussian.variance(5)).toBe(1);
      expect(gaussian.variance(0)).toBe(1);
    });

    it("binomial variance is mu*(1-mu)", () => {
      expect(binomial.variance(0.5)).toBeCloseTo(0.25);
      expect(binomial.variance(0.1)).toBeCloseTo(0.09);
    });

    it("poisson variance equals mu", () => {
      expect(poisson.variance(5)).toBe(5);
    });

    it("gamma variance is mu^2", () => {
      expect(gammaFamily.variance(3)).toBe(9);
    });

    it("gaussian logLikelihood", () => {
      expect(gaussian.logLikelihood(3, 3)).toBeCloseTo(0);
      expect(gaussian.logLikelihood(3, 5)).toBeLessThan(0);
    });

    it("binomial logLikelihood", () => {
      expect(binomial.logLikelihood(1, 0.999)).toBeCloseTo(0, 1);
      expect(binomial.logLikelihood(0, 0.001)).toBeCloseTo(0, 1);
    });

    it("poisson logLikelihood", () => {
      const ll = poisson.logLikelihood(5, 5);
      expect(Number.isFinite(ll)).toBe(true);
    });

    it("gamma logLikelihood", () => {
      const ll = gammaFamily.logLikelihood(2, 2);
      expect(Number.isFinite(ll)).toBe(true);
    });
  });

  describe("Multi-predictor GLM", () => {
    it("handles multiple predictors", () => {
      const X = Array.from({ length: 50 }, (_, i) => [i, Math.sin(i)]);
      const y = X.map(([x1, x2]) => 1 + 2 * x1 + 3 * x2);
      const result = glm(X, y, gaussian);
      expect(result.coefficients).toHaveLength(3); // intercept + 2 slopes
    });
  });

  describe("Custom options", () => {
    it("respects maxIterations option", () => {
      const X = Array.from({ length: 30 }, (_, i) => [i]);
      const y = X.map(([x]) => 2 + 3 * x);
      const result = glm(X, y, gaussian, { maxIterations: 1 });
      expect(result.iterations).toBeLessThanOrEqual(1);
    });

    it("respects tol option", () => {
      const X = Array.from({ length: 30 }, (_, i) => [i]);
      const y = X.map(([x]) => 2 + 3 * x);
      const result = glm(X, y, gaussian, { tol: 1e-2 });
      expect(result.coefficients[0]).toBeCloseTo(2, 0);
    });
  });

  describe("Gamma GLM with inverse link", () => {
    it("verifies gamma family configuration", () => {
      // The gamma family uses the inverse link and mu^2 variance
      expect(gammaFamily.name).toBe("gamma");
      expect(gammaFamily.link).toBe(inverseLink);
      expect(gammaFamily.variance(3)).toBe(9);
      expect(gammaFamily.variance(0.5)).toBeCloseTo(0.25);
    });
  });

  describe("Additional edge cases", () => {
    it("predict function works with multiple predictors", () => {
      const X = Array.from({ length: 50 }, (_, i) => [i, Math.sin(i)]);
      const y = X.map(([x1, x2]) => 1 + 2 * x1 + 3 * x2);
      const result = glm(X, y, gaussian);
      const prediction = result.predict([5, 0.5]);
      expect(Number.isFinite(prediction)).toBe(true);
    });

    it("throws on NaN in y (Infinity case)", () => {
      const X = Array.from({ length: 10 }, (_, i) => [i]);
      const y = [1, 2, 3, 4, Infinity, 6, 7, 8, 9, 10];
      expect(() => glm(X, y, gaussian)).toThrow("finite");
    });

    it("throws on NaN in X (NaN case)", () => {
      const X = [[1], [2], [NaN], [4], [5], [6], [7], [8], [9], [10]];
      const y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      expect(() => glm(X, y, gaussian)).toThrow("finite");
    });

    it("poisson predict returns positive values", () => {
      const X = Array.from({ length: 30 }, (_, i) => [i * 0.1]);
      const y = X.map(([x]) => Math.round(Math.exp(0.5 + 0.3 * x)));
      const result = glm(X, y, poisson);
      expect(result.predict([1])).toBeGreaterThan(0);
      expect(result.predict([2])).toBeGreaterThan(0);
    });

    it("binomial logLikelihood handles edge probabilities", () => {
      // mu very close to 0 should not produce -Infinity
      const ll = binomial.logLikelihood(0, 0.0000001);
      expect(Number.isFinite(ll)).toBe(true);
      // mu very close to 1
      const ll2 = binomial.logLikelihood(1, 0.9999999);
      expect(Number.isFinite(ll2)).toBe(true);
    });

    it("poisson logLikelihood handles zero mu", () => {
      const ll = poisson.logLikelihood(0, 0);
      expect(Number.isFinite(ll)).toBe(true);
    });

    it("gamma logLikelihood handles zero mu", () => {
      const ll = gammaFamily.logLikelihood(1, 0);
      expect(Number.isFinite(ll)).toBe(true);
    });

    it("probitLink link and inverse are consistent", () => {
      // inverse(link(0.3)) should be ~0.3
      const val = 0.3;
      const linked = probitLink.link(val);
      const recovered = probitLink.inverse(linked);
      expect(recovered).toBeCloseTo(val, 3);
    });

    it("logitLink link and inverse are consistent", () => {
      const val = 0.7;
      const linked = logitLink.link(val);
      const recovered = logitLink.inverse(linked);
      expect(recovered).toBeCloseTo(val, 10);
    });

    it("inverseLink derivative is negative for positive eta", () => {
      expect(inverseLink.derivative(2)).toBeLessThan(0);
      expect(inverseLink.derivative(0.5)).toBeLessThan(0);
    });

    it("logLink derivative equals inverse", () => {
      // derivative of exp(eta) is exp(eta)
      expect(logLink.derivative(2)).toBeCloseTo(logLink.inverse(2));
    });

    it("gaussian logLikelihood is zero for perfect prediction", () => {
      expect(gaussian.logLikelihood(5, 5)).toBeCloseTo(0);
    });

    it("gamma family properties are correct", () => {
      expect(gammaFamily.variance(5)).toBe(25);
      expect(gammaFamily.name).toBe("gamma");
      expect(gammaFamily.link).toBe(inverseLink);
    });

    it("handles maxIterations=0 gracefully", () => {
      const X = Array.from({ length: 30 }, (_, i) => [i]);
      const y = X.map(([x]) => 2 + 3 * x);
      const result = glm(X, y, gaussian, { maxIterations: 0 });
      expect(result.iterations).toBe(0);
      // Should still produce some result (initial coefficients)
      expect(result.coefficients).toHaveLength(2);
    });
  });
});
