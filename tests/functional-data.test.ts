import {
  polynomialBasis,
  fourierBasis,
  bsplineBasis,
  smoothBasisExpansion,
  functionalPCA,
  functionalMean,
  l2InnerProduct,
  l2Norm,
} from "../src/functional-data";

// ── Basis Systems ─────────────────────────────────────────────────────────

describe("polynomialBasis", () => {
  it("degree 2 has 3 basis functions", () => {
    const basis = polynomialBasis(2);
    expect(basis.nBasis).toBe(3);
    expect(basis.type).toBe("polynomial");
  });

  it("evaluates to [1, t, t²] on [0,1]", () => {
    const basis = polynomialBasis(2);
    const vals = basis.evaluate(0.5);
    expect(vals[0]).toBeCloseTo(1, 10);
    expect(vals[1]).toBeCloseTo(0.5, 10);
    expect(vals[2]).toBeCloseTo(0.25, 10);
  });
});

describe("fourierBasis", () => {
  it("nBasis=5 gives 1 constant + 2 pairs", () => {
    const basis = fourierBasis(5);
    expect(basis.nBasis).toBe(5);
    expect(basis.type).toBe("fourier");
  });

  it("constant basis is 1 everywhere", () => {
    const basis = fourierBasis(3);
    expect(basis.evaluate(0.5)[0]).toBeCloseTo(1, 10);
    expect(basis.evaluate(0)[0]).toBeCloseTo(1, 10);
  });

  it("sin and cos are periodic", () => {
    const basis = fourierBasis(3);
    const v0 = basis.evaluate(0);
    const v1 = basis.evaluate(1);
    // sin(2π·0) = sin(2π·1) = 0
    expect(v0[1]).toBeCloseTo(v1[1], 8);
  });
});

describe("bsplineBasis", () => {
  it("creates correct number of basis functions", () => {
    const basis = bsplineBasis(6);
    expect(basis.nBasis).toBe(6);
    expect(basis.type).toBe("bspline");
  });

  it("basis functions sum to ~1 (partition of unity)", () => {
    const basis = bsplineBasis(6);
    for (const t of [0.1, 0.3, 0.5, 0.7, 0.9]) {
      const vals = basis.evaluate(t);
      const sum = vals.reduce((a, b) => a + b, 0);
      expect(sum).toBeCloseTo(1, 2);
    }
  });

  it("throws for nBasis < 4", () => {
    expect(() => bsplineBasis(3)).toThrow("at least 4");
  });
});

// ── Basis Expansion ───────────────────────────────────────────────────────

describe("smoothBasisExpansion", () => {
  it("fits a polynomial to polynomial data", () => {
    const t = [0, 0.25, 0.5, 0.75, 1];
    const y = t.map((ti) => 3 * ti * ti + 2 * ti + 1); // 3t² + 2t + 1
    const basis = polynomialBasis(2);

    const fo = smoothBasisExpansion(t, y, basis);
    expect(fo.evaluate(0)).toBeCloseTo(1, 4);
    expect(fo.evaluate(0.5)).toBeCloseTo(3 * 0.25 + 2 * 0.5 + 1, 4);
    expect(fo.evaluate(1)).toBeCloseTo(6, 4);
  });

  it("smoothing penalty shrinks coefficients", () => {
    const t = Array.from({ length: 20 }, (_, i) => i / 19);
    const y = t.map((ti) => Math.sin(2 * Math.PI * ti) + (ti > 0.5 ? 0.3 : -0.3));
    const basis = fourierBasis(9);

    const noSmooth = smoothBasisExpansion(t, y, basis, 0);
    const smooth = smoothBasisExpansion(t, y, basis, 10);

    // Smoothed version has smaller coefficients
    const normNo = noSmooth.coefficients.reduce((a, b) => a + b * b, 0);
    const normSm = smooth.coefficients.reduce((a, b) => a + b * b, 0);
    expect(normSm).toBeLessThan(normNo);
  });

  it("throws for mismatched lengths", () => {
    expect(() =>
      smoothBasisExpansion([0, 1], [0], polynomialBasis(1)),
    ).toThrow("same length");
  });
});

// ── Functional PCA ────────────────────────────────────────────────────────

describe("functionalPCA", () => {
  // Generate curves: y_i(t) = a_i + b_i * t + noise
  const curves: { t: number[]; y: number[] }[] = [];
  const tGrid = Array.from({ length: 20 }, (_, i) => i / 19);
  for (let i = 0; i < 30; i++) {
    const a = (i - 15) / 10;
    const b = ((i * 3) % 30 - 15) / 10;
    curves.push({
      t: tGrid,
      y: tGrid.map((t) => a + b * t),
    });
  }

  it("returns eigenvalues in decreasing order", () => {
    const result = functionalPCA(curves, polynomialBasis(3), 3);
    for (let i = 1; i < result.eigenvalues.length; i++) {
      expect(result.eigenvalues[i]).toBeLessThanOrEqual(result.eigenvalues[i - 1] + 1e-8);
    }
  });

  it("variance explained sums to ~1", () => {
    const result = functionalPCA(curves, polynomialBasis(3));
    const total = result.cumulativeVariance[result.cumulativeVariance.length - 1];
    expect(total).toBeCloseTo(1, 1);
  });

  it("scores have correct dimensions", () => {
    const result = functionalPCA(curves, polynomialBasis(3), 2);
    expect(result.scores.length).toBe(30); // n curves
    expect(result.scores[0].length).toBe(2); // 2 components
  });

  it("returns mean coefficients", () => {
    const result = functionalPCA(curves, polynomialBasis(3));
    expect(result.meanCoefficients.length).toBe(4); // degree 3 + 1
  });

  it("throws for fewer than 2 curves", () => {
    expect(() =>
      functionalPCA([curves[0]], polynomialBasis(2)),
    ).toThrow("at least 2");
  });
});

// ── Functional Mean ───────────────────────────────────────────────────────

describe("functionalMean", () => {
  it("averages curves pointwise", () => {
    const curves = [
      { t: [0, 0.5, 1], y: [2, 4, 6] },
      { t: [0, 0.5, 1], y: [4, 6, 8] },
    ];
    const result = functionalMean(curves, [0, 0.5, 1]);
    expect(result[0]).toBeCloseTo(3, 8);
    expect(result[1]).toBeCloseTo(5, 8);
    expect(result[2]).toBeCloseTo(7, 8);
  });
});

// ── L² Inner Product and Norm ─────────────────────────────────────────────

describe("l2InnerProduct and l2Norm", () => {
  it("norm of constant function = sqrt(b-a) * c", () => {
    const basis = polynomialBasis(0); // constant
    const f = smoothBasisExpansion([0, 1], [3, 3], basis);
    // ∫₀¹ 3² dt = 9, norm = 3
    expect(l2Norm(f)).toBeCloseTo(3, 1);
  });

  it("inner product of orthogonal functions ≈ 0", () => {
    const basis = fourierBasis(3);
    // sin and cos are orthogonal on [0,1]
    const f = { basis, coefficients: [0, 1, 0], evaluate: (t: number) => Math.sin(2 * Math.PI * t) };
    const g = { basis, coefficients: [0, 0, 1], evaluate: (t: number) => Math.cos(2 * Math.PI * t) };

    expect(Math.abs(l2InnerProduct(f, g))).toBeLessThan(0.1);
  });
});
