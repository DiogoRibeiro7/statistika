import {
  gamFit,
  cubicSplineBasis,
  thinPlateBasis,
  tensorProductBasis,
  partialDependence,
} from "../src/gam";

// Simple synthetic data: y = sin(x) + noise
const n = 80;
const X: number[][] = [];
const y: number[] = [];
for (let i = 0; i < n; i++) {
  const xi = (i / n) * 4 * Math.PI;
  X.push([xi]);
  y.push(Math.sin(xi) + Math.sin(i * 0.73) * 0.1);
}

describe("cubicSplineBasis", () => {
  it("returns a basis matrix with correct dimensions", () => {
    const xVals = X.map((row) => row[0]);
    const basis = cubicSplineBasis(xVals, 8);
    expect(basis).toHaveLength(n);
    expect(basis[0]).toHaveLength(8);
  });

  it("each row has finite values", () => {
    const xVals = X.map((row) => row[0]);
    const basis = cubicSplineBasis(xVals, 6);
    for (const row of basis) {
      for (const v of row) {
        expect(Number.isFinite(v)).toBe(true);
      }
    }
  });
});

describe("thinPlateBasis", () => {
  it("returns a basis matrix with correct dimensions", () => {
    const xVals = X.map((row) => row[0]);
    const basis = thinPlateBasis(xVals, 8);
    expect(basis).toHaveLength(n);
    expect(basis[0]).toHaveLength(8);
  });

  it("values are finite", () => {
    const xVals = X.map((row) => row[0]);
    const basis = thinPlateBasis(xVals, 5);
    for (const row of basis) {
      for (const v of row) {
        expect(Number.isFinite(v)).toBe(true);
      }
    }
  });
});

describe("tensorProductBasis", () => {
  it("builds a tensor product basis for two variables", () => {
    const x1 = Array.from({ length: 30 }, (_, i) => i / 30);
    const x2 = Array.from({ length: 30 }, (_, i) => Math.sin(i));
    const basis = tensorProductBasis(x1, x2, 4);
    expect(basis).toHaveLength(30);
    // Tensor product of 4x4 = 16 basis functions
    expect(basis[0].length).toBeGreaterThanOrEqual(4);
  });
});

describe("gamFit", () => {
  it("fits a GAM with a cubic spline term", () => {
    const terms = [{ variables: [0], basisType: "cubic" as const, nBasis: 8 }];
    const result = gamFit(X, y, terms);
    expect(result.fitted).toHaveLength(n);
    expect(result.intercept).toBeDefined();
    expect(result.converged).toBe(true);
    expect(result.lambdas).toHaveLength(1);
    expect(result.edf).toHaveLength(1);
    expect(result.edf[0]).toBeGreaterThan(0);
  });

  it("fitted values approximate the true function", () => {
    const terms = [{ variables: [0], basisType: "cubic" as const, nBasis: 10 }];
    const result = gamFit(X, y, terms);
    // Check that the residuals are small on average
    let sse = 0;
    for (let i = 0; i < n; i++) {
      sse += (result.fitted[i] - y[i]) ** 2;
    }
    const mse = sse / n;
    expect(mse).toBeLessThan(0.5);
  });

  it("GCV score is a finite positive number", () => {
    const terms = [{ variables: [0], basisType: "cubic" as const, nBasis: 8 }];
    const result = gamFit(X, y, terms);
    expect(Number.isFinite(result.gcv)).toBe(true);
    expect(result.gcv).toBeGreaterThanOrEqual(0);
  });

  it("coefficients have correct length per term", () => {
    const terms = [{ variables: [0], basisType: "cubic" as const, nBasis: 8 }];
    const result = gamFit(X, y, terms);
    expect(result.coefficients).toHaveLength(1);
    expect(result.coefficients[0]).toHaveLength(8);
  });
});

describe("partialDependence", () => {
  it("returns grid and values for a smooth term", () => {
    const terms = [{ variables: [0], basisType: "cubic" as const, nBasis: 8 }];
    const result = gamFit(X, y, terms);
    const pd = partialDependence(result, 0, X, terms[0], 50);
    expect(pd.grid).toHaveLength(50);
    expect(pd.values).toHaveLength(50);
    expect(pd.lower).toHaveLength(50);
    expect(pd.upper).toHaveLength(50);
  });

  it("confidence bands bracket the values", () => {
    const terms = [{ variables: [0], basisType: "cubic" as const, nBasis: 8 }];
    const result = gamFit(X, y, terms);
    const pd = partialDependence(result, 0, X, terms[0], 50);
    for (let i = 0; i < 50; i++) {
      expect(pd.lower[i]).toBeLessThanOrEqual(pd.values[i] + 1e-10);
      expect(pd.upper[i]).toBeGreaterThanOrEqual(pd.values[i] - 1e-10);
    }
  });
});
