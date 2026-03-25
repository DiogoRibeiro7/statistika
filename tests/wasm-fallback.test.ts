import { tsFallback, getAccelerated, getWasm, loadWasm } from "../src/wasm/index";
import { gammaLn as mathGammaLn } from "../src/utils/math";

// ---------------------------------------------------------------------------
// 1. tsFallback module — pure TS implementation of WasmModule
// ---------------------------------------------------------------------------

describe("tsFallback", () => {
  describe("gammaLn", () => {
    it("gammaLn(5) ≈ ln(24) ≈ 3.178", () => {
      expect(tsFallback.gammaLn(5)).toBeCloseTo(Math.log(24), 5);
    });

    it("gammaLn(1) = 0", () => {
      expect(tsFallback.gammaLn(1)).toBeCloseTo(0, 10);
    });

    it("gammaLn(0.5) ≈ 0.5724", () => {
      expect(tsFallback.gammaLn(0.5)).toBeCloseTo(0.5723649429247001, 5);
    });
  });

  describe("gamma", () => {
    it("gamma(5) = 24", () => {
      expect(tsFallback.gamma(5)).toBeCloseTo(24, 5);
    });

    it("gamma(1) = 1", () => {
      expect(tsFallback.gamma(1)).toBeCloseTo(1, 10);
    });
  });

  describe("erf", () => {
    it("erf(0) = 0", () => {
      expect(tsFallback.erf(0)).toBeCloseTo(0, 8);
    });

    it("erf(1) ≈ 0.8427", () => {
      expect(tsFallback.erf(1)).toBeCloseTo(0.8427, 3);
    });

    it("erf(-1) ≈ -0.8427", () => {
      expect(tsFallback.erf(-1)).toBeCloseTo(-0.8427, 3);
    });
  });

  describe("erfc", () => {
    it("erfc(0) = 1", () => {
      expect(tsFallback.erfc(0)).toBeCloseTo(1, 8);
    });

    it("erfc(1) ≈ 0.1573", () => {
      expect(tsFallback.erfc(1)).toBeCloseTo(0.1573, 3);
    });
  });

  describe("betaFn", () => {
    it("betaFn(2, 3) ≈ 1/12", () => {
      expect(tsFallback.betaFn(2, 3)).toBeCloseTo(1 / 12, 5);
    });

    it("betaFn(1, 1) = 1", () => {
      expect(tsFallback.betaFn(1, 1)).toBeCloseTo(1, 5);
    });
  });

  describe("matMul", () => {
    it("2x2 * 2x2 identity", () => {
      const A = new Float64Array([1, 2, 3, 4]);
      const I = new Float64Array([1, 0, 0, 1]);
      const C = tsFallback.matMul(A, I, 2, 2, 2);
      expect(Array.from(C)).toEqual([1, 2, 3, 4]);
    });

    it("2x3 * 3x2", () => {
      // A = [[1,2,3],[4,5,6]]  B = [[7,8],[9,10],[11,12]]
      // C = [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
      //   = [[58, 64], [139, 154]]
      const A = new Float64Array([1, 2, 3, 4, 5, 6]);
      const B = new Float64Array([7, 8, 9, 10, 11, 12]);
      const C = tsFallback.matMul(A, B, 2, 3, 2);
      expect(Array.from(C)).toEqual([58, 64, 139, 154]);
    });
  });

  describe("solve", () => {
    it("2x2 system Ax = b", () => {
      // A = [[2, 1], [5, 3]], b = [4, 7]
      // x = [5, -6]  since 2*5 + 1*(-6) = 4,  5*5 + 3*(-6) = 7
      const A = new Float64Array([2, 1, 5, 3]);
      const b = new Float64Array([4, 7]);
      const x = tsFallback.solve(A, b, 2);
      expect(x[0]).toBeCloseTo(5, 10);
      expect(x[1]).toBeCloseTo(-6, 10);
    });
  });

  describe("cholesky", () => {
    it("2x2 positive definite matrix", () => {
      // A = [[4, 2], [2, 5]]
      // L = [[2, 0], [1, 2]]  since L*L^T = [[4,2],[2,5]]
      const A = new Float64Array([4, 2, 2, 5]);
      const L = tsFallback.cholesky(A, 2);
      expect(L[0]).toBeCloseTo(2, 10);
      expect(L[1]).toBeCloseTo(0, 10);
      expect(L[2]).toBeCloseTo(1, 10);
      expect(L[3]).toBeCloseTo(2, 10);
    });
  });
});

// ---------------------------------------------------------------------------
// 2. getAccelerated() returns a usable module
// ---------------------------------------------------------------------------

describe("getAccelerated", () => {
  it("returns a usable module (tsFallback when WASM is unavailable)", () => {
    const mod = getAccelerated();
    expect(mod).toBeDefined();
    expect(typeof mod.gammaLn).toBe("function");
    expect(typeof mod.gamma).toBe("function");
    expect(typeof mod.erf).toBe("function");
    expect(typeof mod.erfc).toBe("function");
    expect(typeof mod.betaFn).toBe("function");
    expect(typeof mod.matMul).toBe("function");
    expect(typeof mod.solve).toBe("function");
    expect(typeof mod.cholesky).toBe("function");

    // Should produce correct results
    expect(mod.gamma(5)).toBeCloseTo(24, 5);
  });
});

// ---------------------------------------------------------------------------
// 3. getWasm() returns null when WASM hasn't been loaded
// ---------------------------------------------------------------------------

describe("getWasm", () => {
  it("returns null when WASM has not been loaded", () => {
    expect(getWasm()).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// 4. loadWasm() returns null gracefully when .wasm file doesn't exist
// ---------------------------------------------------------------------------

describe("loadWasm", () => {
  it("returns null gracefully when the .wasm file does not exist", async () => {
    const result = await loadWasm();
    expect(result).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// 5. Numerical consistency with main math utilities
// ---------------------------------------------------------------------------

describe("Numerical consistency: tsFallback vs utils/math", () => {
  const testValues = [0.5, 1, 1.5, 2, 3, 5, 10];

  it("gammaLn matches the library's main gammaLn for various inputs", () => {
    for (const x of testValues) {
      expect(tsFallback.gammaLn(x)).toBeCloseTo(mathGammaLn(x), 8);
    }
  });
});
