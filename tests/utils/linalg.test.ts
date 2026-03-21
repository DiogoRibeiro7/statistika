import {
  solveLinearSystem,
  invertMatrix,
  transpose,
  matMul,
  symmetricEigen,
  normalCdf,
  normalQuantile,
  createRng,
  randomSample,
  hasNativeLinalg,
} from "../../src/utils/linalg";

// ── hasNativeLinalg ─────────────────────────────────────────────────────

describe("hasNativeLinalg", () => {
  it("is a boolean", () => {
    expect(typeof hasNativeLinalg).toBe("boolean");
  });
});

// ── solveLinearSystem ───────────────────────────────────────────────────

describe("solveLinearSystem", () => {
  it("solves a 2x2 system", () => {
    // 2x + y = 5, x + 3y = 7  =>  x = 1.6, y = 1.8
    const x = solveLinearSystem(
      [[2, 1], [1, 3]],
      [5, 7],
    );
    expect(x[0]).toBeCloseTo(1.6, 10);
    expect(x[1]).toBeCloseTo(1.8, 10);
  });

  it("solves a 3x3 system", () => {
    // x + y + z = 6, 2x + y - z = 1, x - y + 2z = 5  =>  x = 1, y = 2, z = 3
    const x = solveLinearSystem(
      [[1, 1, 1], [2, 1, -1], [1, -1, 2]],
      [6, 1, 5],
    );
    expect(x[0]).toBeCloseTo(1, 10);
    expect(x[1]).toBeCloseTo(2, 10);
    expect(x[2]).toBeCloseTo(3, 10);
  });

  it("solves the identity system", () => {
    const x = solveLinearSystem(
      [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
      [4, 5, 6],
    );
    expect(x).toEqual([4, 5, 6]);
  });

  it("throws on a singular matrix", () => {
    expect(() => solveLinearSystem(
      [[1, 2], [2, 4]],
      [3, 6],
    )).toThrow();
  });

  it("does not modify the input arrays", () => {
    const A = [[2, 1], [1, 3]];
    const b = [5, 7];
    const Acopy = A.map(row => [...row]);
    const bcopy = [...b];
    solveLinearSystem(A, b);
    expect(A).toEqual(Acopy);
    expect(b).toEqual(bcopy);
  });
});

// ── invertMatrix ────────────────────────────────────────────────────────

describe("invertMatrix", () => {
  it("inverts a 2x2 matrix", () => {
    // [[1,2],[3,4]]^-1 = [[-2,1],[1.5,-0.5]]
    const inv = invertMatrix([[1, 2], [3, 4]]);
    expect(inv).not.toBeNull();
    expect(inv![0][0]).toBeCloseTo(-2, 10);
    expect(inv![0][1]).toBeCloseTo(1, 10);
    expect(inv![1][0]).toBeCloseTo(1.5, 10);
    expect(inv![1][1]).toBeCloseTo(-0.5, 10);
  });

  it("inverts the identity matrix to itself", () => {
    const I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    const inv = invertMatrix(I);
    expect(inv).not.toBeNull();
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        expect(inv![i][j]).toBeCloseTo(i === j ? 1 : 0, 10);
      }
    }
  });

  it("A * A^-1 = I for a 3x3 matrix", () => {
    const A = [[2, 1, 0], [1, 3, 1], [0, 1, 2]];
    const inv = invertMatrix(A);
    expect(inv).not.toBeNull();
    const product = matMul(A, inv!);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        expect(product[i][j]).toBeCloseTo(i === j ? 1 : 0, 8);
      }
    }
  });

  it("returns null for a singular matrix", () => {
    const result = invertMatrix([[1, 2], [2, 4]]);
    expect(result).toBeNull();
  });
});

// ── transpose ───────────────────────────────────────────────────────────

describe("transpose", () => {
  it("transposes a square matrix", () => {
    expect(transpose([[1, 2], [3, 4]])).toEqual([[1, 3], [2, 4]]);
  });

  it("transposes a non-square matrix", () => {
    expect(transpose([[1, 2, 3], [4, 5, 6]])).toEqual([
      [1, 4],
      [2, 5],
      [3, 6],
    ]);
  });

  it("double transpose returns the original", () => {
    const A = [[1, 2, 3], [4, 5, 6]];
    expect(transpose(transpose(A))).toEqual(A);
  });

  it("transposes a 1x1 matrix", () => {
    expect(transpose([[42]])).toEqual([[42]]);
  });
});

// ── matMul ──────────────────────────────────────────────────────────────

describe("matMul", () => {
  it("multiplies two 2x2 matrices", () => {
    // [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
    const C = matMul([[1, 2], [3, 4]], [[5, 6], [7, 8]]);
    expect(C).toEqual([[19, 22], [43, 50]]);
  });

  it("multiplies non-square matrices (2x3 * 3x2)", () => {
    const C = matMul(
      [[1, 2, 3], [4, 5, 6]],
      [[7, 8], [9, 10], [11, 12]],
    );
    expect(C).toEqual([[58, 64], [139, 154]]);
  });

  it("multiplies non-square matrices (3x2 * 2x3)", () => {
    const C = matMul(
      [[1, 2], [3, 4], [5, 6]],
      [[7, 8, 9], [10, 11, 12]],
    );
    expect(C).toEqual([
      [27, 30, 33],
      [61, 68, 75],
      [95, 106, 117],
    ]);
  });

  it("multiplying by identity returns the original", () => {
    const A = [[1, 2], [3, 4]];
    const I = [[1, 0], [0, 1]];
    expect(matMul(A, I)).toEqual(A);
    expect(matMul(I, A)).toEqual(A);
  });

  it("multiplies 1x1 matrices", () => {
    expect(matMul([[3]], [[7]])).toEqual([[21]]);
  });

  it("multiplies a row vector by a column vector", () => {
    // [1,2,3] * [4;5;6] = [32]
    expect(matMul([[1, 2, 3]], [[4], [5], [6]])).toEqual([[32]]);
  });
});

// ── symmetricEigen ──────────────────────────────────────────────────────

describe("symmetricEigen", () => {
  it("decomposes a 2x2 symmetric matrix", () => {
    // [[2,1],[1,2]] has eigenvalues 3, 1
    const { eigenvalues, eigenvectors } = symmetricEigen([[2, 1], [1, 2]]);
    expect(eigenvalues[0]).toBeCloseTo(3, 8);
    expect(eigenvalues[1]).toBeCloseTo(1, 8);

    // Eigenvectors should be orthonormal
    const dot = eigenvectors[0][0] * eigenvectors[0][1] +
                eigenvectors[1][0] * eigenvectors[1][1];
    expect(dot).toBeCloseTo(0, 8);
  });

  it("returns eigenvalues in descending order", () => {
    const A = [[5, 2, 0], [2, 3, 1], [0, 1, 4]];
    const { eigenvalues } = symmetricEigen(A);
    for (let i = 0; i < eigenvalues.length - 1; i++) {
      expect(eigenvalues[i]).toBeGreaterThanOrEqual(eigenvalues[i + 1] - 1e-10);
    }
  });

  it("eigenvalues of the identity are all 1", () => {
    const { eigenvalues } = symmetricEigen([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
    eigenvalues.forEach(v => expect(v).toBeCloseTo(1, 8));
  });

  it("A * v = lambda * v for each eigenpair", () => {
    const A = [[4, 1, 0], [1, 3, 1], [0, 1, 2]];
    const { eigenvalues, eigenvectors } = symmetricEigen(A);
    const n = A.length;

    for (let k = 0; k < n; k++) {
      const v = eigenvectors.map(row => row[k]); // k-th eigenvector
      const Av = A.map(row => row.reduce((s, aij, j) => s + aij * v[j], 0));
      const lambdaV = v.map(vi => vi * eigenvalues[k]);
      for (let i = 0; i < n; i++) {
        expect(Av[i]).toBeCloseTo(lambdaV[i], 8);
      }
    }
  });

  it("decomposes a diagonal matrix", () => {
    const { eigenvalues } = symmetricEigen([[5, 0], [0, 2]]);
    expect(eigenvalues[0]).toBeCloseTo(5, 8);
    expect(eigenvalues[1]).toBeCloseTo(2, 8);
  });
});

// ── normalCdf ───────────────────────────────────────────────────────────

describe("normalCdf", () => {
  it("Phi(0) = 0.5", () => {
    expect(normalCdf(0)).toBeCloseTo(0.5, 10);
  });

  it("Phi(1.96) ≈ 0.975", () => {
    expect(normalCdf(1.96)).toBeCloseTo(0.975, 3);
  });

  it("Phi(-1.96) ≈ 0.025", () => {
    expect(normalCdf(-1.96)).toBeCloseTo(0.025, 3);
  });

  it("is monotonically increasing", () => {
    const xs = [-3, -2, -1, 0, 1, 2, 3];
    const cdfs = xs.map(normalCdf);
    for (let i = 1; i < cdfs.length; i++) {
      expect(cdfs[i]).toBeGreaterThan(cdfs[i - 1]);
    }
  });

  it("approaches 0 and 1 at extremes", () => {
    expect(normalCdf(-6)).toBeLessThan(1e-6);
    expect(normalCdf(6)).toBeGreaterThan(1 - 1e-6);
  });

  it("Phi(x) + Phi(-x) = 1 (symmetry)", () => {
    for (const x of [0.5, 1, 2, 3]) {
      expect(normalCdf(x) + normalCdf(-x)).toBeCloseTo(1, 10);
    }
  });
});

// ── normalQuantile ──────────────────────────────────────────────────────

describe("normalQuantile", () => {
  it("Q(0.5) = 0", () => {
    expect(normalQuantile(0.5)).toBe(0);
  });

  it("Q(0.975) ≈ 1.96", () => {
    expect(normalQuantile(0.975)).toBeCloseTo(1.96, 2);
  });

  it("Q(0.025) ≈ -1.96", () => {
    expect(normalQuantile(0.025)).toBeCloseTo(-1.96, 2);
  });

  it("Q(0) = -Infinity, Q(1) = +Infinity", () => {
    expect(normalQuantile(0)).toBe(-Infinity);
    expect(normalQuantile(1)).toBe(Infinity);
  });

  it("is the inverse of normalCdf", () => {
    for (const p of [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]) {
      expect(normalCdf(normalQuantile(p))).toBeCloseTo(p, 4);
    }
  });

  it("is monotonically increasing", () => {
    const ps = [0.1, 0.25, 0.5, 0.75, 0.9];
    const qs = ps.map(normalQuantile);
    for (let i = 1; i < qs.length; i++) {
      expect(qs[i]).toBeGreaterThan(qs[i - 1]);
    }
  });
});

// ── createRng ───────────────────────────────────────────────────────────

describe("createRng", () => {
  it("produces values in [0, 1)", () => {
    const rng = createRng(42);
    for (let i = 0; i < 1000; i++) {
      const v = rng();
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });

  it("same seed produces same sequence", () => {
    const rng1 = createRng(123);
    const rng2 = createRng(123);
    for (let i = 0; i < 100; i++) {
      expect(rng1()).toBe(rng2());
    }
  });

  it("different seeds produce different sequences", () => {
    const rng1 = createRng(1);
    const rng2 = createRng(2);
    const vals1 = Array.from({ length: 10 }, () => rng1());
    const vals2 = Array.from({ length: 10 }, () => rng2());
    expect(vals1).not.toEqual(vals2);
  });
});

// ── randomSample ────────────────────────────────────────────────────────

describe("randomSample", () => {
  it("returns k unique indices from [0, n)", () => {
    const rng = createRng(42);
    const sample = randomSample(10, 5, rng);
    expect(sample).toHaveLength(5);
    expect(new Set(sample).size).toBe(5);
    sample.forEach(idx => {
      expect(idx).toBeGreaterThanOrEqual(0);
      expect(idx).toBeLessThan(10);
    });
  });

  it("returns all indices when k = n", () => {
    const rng = createRng(42);
    const sample = randomSample(5, 5, rng);
    expect(sample.sort()).toEqual([0, 1, 2, 3, 4]);
  });

  it("returns empty array when k = 0", () => {
    const rng = createRng(42);
    expect(randomSample(10, 0, rng)).toEqual([]);
  });

  it("is deterministic with the same seed", () => {
    const s1 = randomSample(20, 10, createRng(99));
    const s2 = randomSample(20, 10, createRng(99));
    expect(s1).toEqual(s2);
  });
});
