import { Mat } from "../../src/utils/matrix";

// ── Factories ──────────────────────────────────────────────────────────

describe("Mat.from", () => {
  it("creates a matrix from a 2D array", () => {
    const m = Mat.from([[1, 2], [3, 4]]);
    expect(m.rows).toBe(2);
    expect(m.cols).toBe(2);
    expect(m.get(0, 0)).toBe(1);
    expect(m.get(1, 1)).toBe(4);
  });

  it("creates a non-square matrix", () => {
    const m = Mat.from([[1, 2, 3], [4, 5, 6]]);
    expect(m.rows).toBe(2);
    expect(m.cols).toBe(3);
  });

  it("throws on jagged arrays", () => {
    expect(() => Mat.from([[1, 2], [3]])).toThrow("expected 2 columns");
  });

  it("throws on empty array", () => {
    expect(() => Mat.from([])).toThrow();
  });
});

describe("Mat.zeros / ones / identity / diag", () => {
  it("creates a zero matrix", () => {
    const m = Mat.zeros(3, 2);
    expect(m.rows).toBe(3);
    expect(m.cols).toBe(2);
    expect(m.normF()).toBe(0);
  });

  it("creates a ones matrix", () => {
    const m = Mat.ones(2, 3);
    for (let i = 0; i < 2; i++)
      for (let j = 0; j < 3; j++)
        expect(m.get(i, j)).toBe(1);
  });

  it("creates an identity matrix", () => {
    const I = Mat.identity(3);
    expect(I.toArray()).toEqual([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
  });

  it("creates a diagonal matrix", () => {
    const D = Mat.diag([2, 3, 5]);
    expect(D.toArray()).toEqual([[2, 0, 0], [0, 3, 0], [0, 0, 5]]);
  });
});

describe("Mat.fromVector", () => {
  it("creates a column vector", () => {
    const v = Mat.fromVector([1, 2, 3]);
    expect(v.rows).toBe(3);
    expect(v.cols).toBe(1);
    expect(v.toVector()).toEqual([1, 2, 3]);
  });
});

// ── Element access ─────────────────────────────────────────────────────

describe("element access", () => {
  const m = Mat.from([[1, 2, 3], [4, 5, 6]]);

  it("row() extracts a row", () => {
    expect(m.row(0)).toEqual([1, 2, 3]);
    expect(m.row(1)).toEqual([4, 5, 6]);
  });

  it("col() extracts a column", () => {
    expect(m.col(0)).toEqual([1, 4]);
    expect(m.col(2)).toEqual([3, 6]);
  });

  it("diagonal() extracts the diagonal", () => {
    expect(m.diagonal()).toEqual([1, 5]);
  });

  it("set() modifies an element", () => {
    const c = m.clone();
    c.set(0, 0, 99);
    expect(c.get(0, 0)).toBe(99);
    // Original unchanged
    expect(m.get(0, 0)).toBe(1);
  });

  it("toArray() round-trips", () => {
    const data = [[1, 2], [3, 4], [5, 6]];
    expect(Mat.from(data).toArray()).toEqual(data);
  });
});

// ── Predicates ─────────────────────────────────────────────────────────

describe("predicates", () => {
  it("isSquare", () => {
    expect(Mat.identity(3).isSquare()).toBe(true);
    expect(Mat.zeros(2, 3).isSquare()).toBe(false);
  });

  it("sameSize", () => {
    expect(Mat.zeros(2, 3).sameSize(Mat.ones(2, 3))).toBe(true);
    expect(Mat.zeros(2, 3).sameSize(Mat.ones(3, 2))).toBe(false);
  });

  it("equals with tolerance", () => {
    const a = Mat.from([[1, 2], [3, 4]]);
    const b = Mat.from([[1 + 1e-12, 2], [3, 4 - 1e-12]]);
    expect(a.equals(b)).toBe(true);
    expect(a.equals(b, 1e-13)).toBe(false);
  });
});

// ── Arithmetic ─────────────────────────────────────────────────────────

describe("arithmetic", () => {
  const A = Mat.from([[1, 2], [3, 4]]);
  const B = Mat.from([[5, 6], [7, 8]]);

  it("add", () => {
    expect(A.add(B).toArray()).toEqual([[6, 8], [10, 12]]);
  });

  it("subtract", () => {
    expect(B.subtract(A).toArray()).toEqual([[4, 4], [4, 4]]);
  });

  it("scale", () => {
    expect(A.scale(2).toArray()).toEqual([[2, 4], [6, 8]]);
  });

  it("negate", () => {
    expect(A.negate().toArray()).toEqual([[-1, -2], [-3, -4]]);
  });

  it("multiply (2x2 * 2x2)", () => {
    // [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
    expect(A.multiply(B).toArray()).toEqual([[19, 22], [43, 50]]);
  });

  it("multiply (2x3 * 3x2)", () => {
    const X = Mat.from([[1, 2, 3], [4, 5, 6]]);
    const Y = Mat.from([[7, 8], [9, 10], [11, 12]]);
    expect(X.multiply(Y).toArray()).toEqual([[58, 64], [139, 154]]);
  });

  it("multiply by identity returns original", () => {
    const I = Mat.identity(2);
    expect(A.multiply(I).equals(A)).toBe(true);
    expect(I.multiply(A).equals(A)).toBe(true);
  });

  it("throws on dimension mismatch", () => {
    expect(() => A.add(Mat.zeros(3, 3))).toThrow("Invalid parameter 'other'");
    expect(() => A.multiply(Mat.zeros(3, 2))).toThrow("Invalid parameter 'other'");
  });

  it("transpose", () => {
    const X = Mat.from([[1, 2, 3], [4, 5, 6]]);
    const T = X.transpose();
    expect(T.rows).toBe(3);
    expect(T.cols).toBe(2);
    expect(T.toArray()).toEqual([[1, 4], [2, 5], [3, 6]]);
  });

  it("double transpose returns original", () => {
    expect(A.transpose().transpose().equals(A)).toBe(true);
  });

  it("trace", () => {
    expect(A.trace()).toBe(5); // 1 + 4
  });

  it("normF (Frobenius)", () => {
    const I = Mat.identity(3);
    expect(I.normF()).toBeCloseTo(Math.sqrt(3), 10);
  });
});

// ── Submatrix ──────────────────────────────────────────────────────────

describe("submatrix", () => {
  it("extracts a submatrix", () => {
    const m = Mat.from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    const sub = m.submatrix(0, 2, 1, 3);
    expect(sub.toArray()).toEqual([[2, 3], [5, 6]]);
  });
});

// ── LU Decomposition ──────────────────────────────────────────────────

describe("LU decomposition", () => {
  it("decomposes a 3x3 matrix and PA = LU", () => {
    const A = Mat.from([[2, 1, 1], [4, 3, 3], [8, 7, 9]]);
    const { L, U, P } = A.lu();

    // PA = LU
    const PA = P.multiply(A);
    const LU = L.multiply(U);
    expect(PA.equals(LU, 1e-10)).toBe(true);
  });

  it("L is lower triangular with unit diagonal", () => {
    const A = Mat.from([[4, 3], [6, 3]]);
    const { L } = A.lu();
    // Diagonal should be 1
    expect(L.get(0, 0)).toBeCloseTo(1, 10);
    expect(L.get(1, 1)).toBeCloseTo(1, 10);
    // Upper triangle should be 0
    expect(L.get(0, 1)).toBeCloseTo(0, 10);
  });

  it("U is upper triangular", () => {
    const A = Mat.from([[4, 3], [6, 3]]);
    const { U } = A.lu();
    expect(U.get(1, 0)).toBeCloseTo(0, 10);
  });

  it("throws on singular matrix", () => {
    expect(() => Mat.from([[1, 2], [2, 4]]).lu()).toThrow("singular");
  });

  it("decomposes a 4x4 matrix", () => {
    const A = Mat.from([
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [2, 1, 4, 3],
      [7, 8, 2, 1],
    ]);
    const { L, U, P } = A.lu();
    const PA = P.multiply(A);
    const LU = L.multiply(U);
    expect(PA.equals(LU, 1e-10)).toBe(true);
  });
});

// ── QR Decomposition ──────────────────────────────────────────────────

describe("QR decomposition", () => {
  it("decomposes a square matrix: A = Q*R", () => {
    const A = Mat.from([[1, 2], [3, 4]]);
    const { Q, R } = A.qr();
    const QR = Q.multiply(R);
    expect(QR.equals(A, 1e-10)).toBe(true);
  });

  it("Q has orthonormal columns", () => {
    const A = Mat.from([[1, 2, 0], [0, 1, 1], [1, 0, 1]]);
    const { Q } = A.qr();
    const QtQ = Q.transpose().multiply(Q);
    expect(QtQ.equals(Mat.identity(3), 1e-10)).toBe(true);
  });

  it("R is upper triangular", () => {
    const A = Mat.from([[1, 2, 3], [4, 5, 6], [7, 8, 10]]);
    const { R } = A.qr();
    for (let i = 1; i < R.rows; i++) {
      for (let j = 0; j < i; j++) {
        expect(Math.abs(R.get(i, j))).toBeLessThan(1e-10);
      }
    }
  });

  it("works for tall matrices (4x2)", () => {
    const A = Mat.from([[1, 2], [3, 4], [5, 6], [7, 8]]);
    const { Q, R } = A.qr();
    const QR = Q.multiply(R);
    expect(QR.equals(A, 1e-10)).toBe(true);
    // Q^T Q = I (thin Q)
    const QtQ = Q.transpose().multiply(Q);
    expect(QtQ.equals(Mat.identity(2), 1e-10)).toBe(true);
  });
});

// ── Cholesky Decomposition ────────────────────────────────────────────

describe("Cholesky decomposition", () => {
  it("decomposes a 2x2 SPD matrix: A = L*L^T", () => {
    const A = Mat.from([[4, 2], [2, 3]]);
    const L = A.cholesky();
    const LLt = L.multiply(L.transpose());
    expect(LLt.equals(A, 1e-10)).toBe(true);
  });

  it("L is lower triangular", () => {
    const A = Mat.from([[4, 2], [2, 3]]);
    const L = A.cholesky();
    expect(L.get(0, 1)).toBeCloseTo(0, 10);
  });

  it("decomposes a 3x3 SPD matrix", () => {
    const A = Mat.from([[25, 15, -5], [15, 18, 0], [-5, 0, 11]]);
    const L = A.cholesky();
    const LLt = L.multiply(L.transpose());
    expect(LLt.equals(A, 1e-10)).toBe(true);
  });

  it("throws on non-positive-definite matrix", () => {
    expect(() => Mat.from([[1, 2], [2, 1]]).cholesky()).toThrow(
      "positive definite",
    );
  });

  it("works for identity matrix", () => {
    const L = Mat.identity(3).cholesky();
    expect(L.equals(Mat.identity(3), 1e-10)).toBe(true);
  });
});

// ── SVD ────────────────────────────────────────────────────────────────

describe("SVD", () => {
  it("decomposes a 2x2 matrix: A ≈ U*diag(S)*V^T", () => {
    const A = Mat.from([[3, 2], [2, 3]]);
    const { U, S, V } = A.svd();

    // Reconstruct
    const Smat = Mat.diag(S);
    const recon = U.multiply(Smat).multiply(V.transpose());
    expect(recon.equals(A, 1e-10)).toBe(true);
  });

  it("singular values are in descending order", () => {
    const A = Mat.from([[1, 2, 3], [4, 5, 6], [7, 8, 10]]);
    const { S } = A.svd();
    for (let i = 0; i < S.length - 1; i++) {
      expect(S[i]).toBeGreaterThanOrEqual(S[i + 1] - 1e-10);
    }
  });

  it("singular values are non-negative", () => {
    const A = Mat.from([[1, -2], [-3, 4], [5, -6]]);
    const { S } = A.svd();
    for (const s of S) {
      expect(s).toBeGreaterThanOrEqual(-1e-10);
    }
  });

  it("U has orthonormal columns", () => {
    const A = Mat.from([[1, 2], [3, 4], [5, 6]]);
    const { U } = A.svd();
    const UtU = U.transpose().multiply(U);
    expect(UtU.equals(Mat.identity(U.cols), 1e-10)).toBe(true);
  });

  it("V has orthonormal columns", () => {
    const A = Mat.from([[1, 2], [3, 4], [5, 6]]);
    const { V } = A.svd();
    const VtV = V.transpose().multiply(V);
    expect(VtV.equals(Mat.identity(V.cols), 1e-10)).toBe(true);
  });

  it("reconstructs a tall matrix (3x2)", () => {
    const A = Mat.from([[1, 2], [3, 4], [5, 6]]);
    const { U, S, V } = A.svd();
    const Smat = Mat.diag(S);
    const recon = U.multiply(Smat).multiply(V.transpose());
    expect(recon.equals(A, 1e-10)).toBe(true);
  });

  it("reconstructs a wide matrix (2x3)", () => {
    const A = Mat.from([[1, 2, 3], [4, 5, 6]]);
    const { U, S, V } = A.svd();
    const Smat = Mat.diag(S);
    const recon = U.multiply(Smat).multiply(V.transpose());
    expect(recon.equals(A, 1e-10)).toBe(true);
  });

  it("handles a diagonal matrix", () => {
    const A = Mat.diag([5, 3, 1]);
    const { S } = A.svd();
    expect(S[0]).toBeCloseTo(5, 8);
    expect(S[1]).toBeCloseTo(3, 8);
    expect(S[2]).toBeCloseTo(1, 8);
  });

  it("handles a 1x1 matrix", () => {
    const A = Mat.from([[7]]);
    const { U, S, V } = A.svd();
    expect(S[0]).toBeCloseTo(7, 10);
    const recon = U.multiply(Mat.diag(S)).multiply(V.transpose());
    expect(recon.equals(A, 1e-10)).toBe(true);
  });
});

// ── Solve ──────────────────────────────────────────────────────────────

describe("solve", () => {
  it("solves a 2x2 system", () => {
    const A = Mat.from([[2, 1], [1, 3]]);
    const x = A.solve([5, 7]);
    expect(x[0]).toBeCloseTo(1.6, 10);
    expect(x[1]).toBeCloseTo(1.8, 10);
  });

  it("solves a 3x3 system", () => {
    const A = Mat.from([[1, 1, 1], [2, 1, -1], [1, -1, 2]]);
    const x = A.solve([6, 1, 5]);
    expect(x[0]).toBeCloseTo(1, 10);
    expect(x[1]).toBeCloseTo(2, 10);
    expect(x[2]).toBeCloseTo(3, 10);
  });

  it("solves the identity system", () => {
    const x = Mat.identity(3).solve([4, 5, 6]);
    expect(x[0]).toBeCloseTo(4, 10);
    expect(x[1]).toBeCloseTo(5, 10);
    expect(x[2]).toBeCloseTo(6, 10);
  });

  it("accepts a column vector Mat as RHS", () => {
    const A = Mat.from([[2, 1], [1, 3]]);
    const b = Mat.fromVector([5, 7]);
    const x = A.solve(b);
    expect(x[0]).toBeCloseTo(1.6, 10);
    expect(x[1]).toBeCloseTo(1.8, 10);
  });

  it("throws on singular matrix", () => {
    expect(() => Mat.from([[1, 2], [2, 4]]).solve([3, 6])).toThrow();
  });

  it("throws on dimension mismatch", () => {
    expect(() => Mat.from([[1, 2], [3, 4]]).solve([1])).toThrow("Invalid parameter 'b'");
  });
});

// ── Least Squares ──────────────────────────────────────────────────────

describe("leastSquares", () => {
  it("solves an exact square system", () => {
    const A = Mat.from([[1, 1], [0, 1]]);
    const x = A.leastSquares([3, 2]);
    expect(x[0]).toBeCloseTo(1, 8);
    expect(x[1]).toBeCloseTo(2, 8);
  });

  it("solves an overdetermined system (linear fit y = 2x + 1)", () => {
    // Points: (0,1), (1,3), (2,5) => y = 2x + 1
    // A = [[1,0],[1,1],[1,2]], b = [1,3,5]
    const A = Mat.from([[1, 0], [1, 1], [1, 2]]);
    const x = A.leastSquares([1, 3, 5]);
    expect(x[0]).toBeCloseTo(1, 8); // intercept
    expect(x[1]).toBeCloseTo(2, 8); // slope
  });

  it("solves a noisy overdetermined system", () => {
    // y ≈ 1 + 2x with noise
    const A = Mat.from([[1, 0], [1, 1], [1, 2], [1, 3]]);
    const b = [1.1, 2.9, 5.2, 6.8];
    const x = A.leastSquares(b);
    expect(x[0]).toBeCloseTo(1, 0); // intercept ~ 1
    expect(x[1]).toBeCloseTo(2, 0); // slope ~ 2
  });
});

// ── Determinant ────────────────────────────────────────────────────────

describe("det", () => {
  it("det of identity is 1", () => {
    expect(Mat.identity(3).det()).toBeCloseTo(1, 10);
  });

  it("det of 2x2 matrix", () => {
    // [[1,2],[3,4]] => det = -2
    expect(Mat.from([[1, 2], [3, 4]]).det()).toBeCloseTo(-2, 10);
  });

  it("det of 3x3 matrix", () => {
    // [[1,2,3],[0,1,4],[5,6,0]] => det = 1
    expect(Mat.from([[1, 2, 3], [0, 1, 4], [5, 6, 0]]).det()).toBeCloseTo(1, 8);
  });

  it("det of diagonal matrix is product of diagonal", () => {
    expect(Mat.diag([2, 3, 5]).det()).toBeCloseTo(30, 10);
  });
});

// ── Inverse ────────────────────────────────────────────────────────────

describe("inverse", () => {
  it("A * A^-1 = I", () => {
    const A = Mat.from([[2, 1], [1, 3]]);
    const Ainv = A.inverse();
    expect(Ainv).not.toBeNull();
    const product = A.multiply(Ainv!);
    expect(product.equals(Mat.identity(2), 1e-10)).toBe(true);
  });

  it("inverse of identity is identity", () => {
    const inv = Mat.identity(3).inverse();
    expect(inv).not.toBeNull();
    expect(inv!.equals(Mat.identity(3), 1e-10)).toBe(true);
  });

  it("returns null for singular matrix", () => {
    expect(Mat.from([[1, 2], [2, 4]]).inverse()).toBeNull();
  });
});

// ── Rank ───────────────────────────────────────────────────────────────

describe("rank", () => {
  it("full rank for identity", () => {
    expect(Mat.identity(3).rank()).toBe(3);
  });

  it("rank 1 for rank-deficient matrix", () => {
    expect(Mat.from([[1, 2], [2, 4]]).rank()).toBe(1);
  });

  it("rank of tall matrix", () => {
    expect(Mat.from([[1, 0], [0, 1], [1, 1]]).rank()).toBe(2);
  });
});

// ── Condition number ───────────────────────────────────────────────────

describe("cond", () => {
  it("condition number of identity is 1", () => {
    expect(Mat.identity(3).cond()).toBeCloseTo(1, 8);
  });

  it("condition number of ill-conditioned matrix is large", () => {
    const A = Mat.from([[1, 1], [1, 1.0001]]);
    expect(A.cond()).toBeGreaterThan(1000);
  });
});

// ── Pseudoinverse ──────────────────────────────────────────────────────

describe("pinv", () => {
  it("pinv of invertible matrix equals inverse", () => {
    const A = Mat.from([[1, 2], [3, 4]]);
    const pinvA = A.pinv();
    const invA = A.inverse();
    expect(invA).not.toBeNull();
    expect(pinvA.equals(invA!, 1e-8)).toBe(true);
  });

  it("A * pinv(A) * A = A for a tall matrix", () => {
    const A = Mat.from([[1, 2], [3, 4], [5, 6]]);
    const Ap = A.pinv();
    const result = A.multiply(Ap).multiply(A);
    expect(result.equals(A, 1e-8)).toBe(true);
  });

  it("A * pinv(A) * A = A for a rank-deficient matrix", () => {
    const A = Mat.from([[1, 2], [2, 4]]);
    const Ap = A.pinv();
    const result = A.multiply(Ap).multiply(A);
    expect(result.equals(A, 1e-8)).toBe(true);
  });
});
