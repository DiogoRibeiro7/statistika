import {
  fromArray2D,
  toArray2D,
  matMulTyped,
  transposeTyped,
  dotTyped,
  matVecMulTyped,
  addTyped,
  scaleTyped,
  eyeTyped,
  frobeniusNormTyped,
} from "../src/utils/typed-matrix";

describe("typed-matrix utilities", () => {
  describe("fromArray2D / toArray2D", () => {
    it("round-trips a 2D array", () => {
      const data = [
        [1, 2, 3],
        [4, 5, 6],
      ];
      const { data: flat, rows, cols } = fromArray2D(data);
      expect(rows).toBe(2);
      expect(cols).toBe(3);
      expect(Array.from(flat)).toEqual([1, 2, 3, 4, 5, 6]);
      expect(toArray2D(flat, rows, cols)).toEqual(data);
    });

    it("handles empty input", () => {
      const { data, rows, cols } = fromArray2D([]);
      expect(rows).toBe(0);
      expect(cols).toBe(0);
      expect(data.length).toBe(0);
    });
  });

  describe("matMulTyped", () => {
    it("multiplies 2x3 by 3x2", () => {
      const a = new Float64Array([1, 2, 3, 4, 5, 6]);
      const b = new Float64Array([7, 8, 9, 10, 11, 12]);
      const result = matMulTyped(a, b, 2, 3, 2);
      // [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
      // [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
      expect(Array.from(result)).toEqual([58, 64, 139, 154]);
    });

    it("multiplies identity by matrix", () => {
      const eye = eyeTyped(2);
      const a = new Float64Array([3, 4, 5, 6]);
      const result = matMulTyped(eye, a, 2, 2, 2);
      expect(Array.from(result)).toEqual([3, 4, 5, 6]);
    });
  });

  describe("transposeTyped", () => {
    it("transposes a 2x3 matrix", () => {
      const a = new Float64Array([1, 2, 3, 4, 5, 6]);
      const t = transposeTyped(a, 2, 3);
      expect(Array.from(t)).toEqual([1, 4, 2, 5, 3, 6]);
    });

    it("transposes a 1x4 into 4x1", () => {
      const a = new Float64Array([1, 2, 3, 4]);
      const t = transposeTyped(a, 1, 4);
      expect(Array.from(t)).toEqual([1, 2, 3, 4]);
    });
  });

  describe("dotTyped", () => {
    it("computes dot product", () => {
      const a = new Float64Array([1, 2, 3]);
      const b = new Float64Array([4, 5, 6]);
      expect(dotTyped(a, b)).toBe(32); // 4+10+18
    });

    it("returns 0 for orthogonal vectors", () => {
      const a = new Float64Array([1, 0]);
      const b = new Float64Array([0, 1]);
      expect(dotTyped(a, b)).toBe(0);
    });
  });

  describe("matVecMulTyped", () => {
    it("multiplies 2x3 matrix by 3-vector", () => {
      const a = new Float64Array([1, 2, 3, 4, 5, 6]);
      const x = new Float64Array([1, 1, 1]);
      const result = matVecMulTyped(a, x, 2, 3);
      expect(Array.from(result)).toEqual([6, 15]);
    });
  });

  describe("addTyped", () => {
    it("adds two arrays element-wise", () => {
      const a = new Float64Array([1, 2, 3]);
      const b = new Float64Array([4, 5, 6]);
      expect(Array.from(addTyped(a, b))).toEqual([5, 7, 9]);
    });
  });

  describe("scaleTyped", () => {
    it("scales by a scalar", () => {
      const a = new Float64Array([1, 2, 3]);
      expect(Array.from(scaleTyped(a, 3))).toEqual([3, 6, 9]);
    });

    it("scales by zero", () => {
      const a = new Float64Array([1, 2, 3]);
      expect(Array.from(scaleTyped(a, 0))).toEqual([0, 0, 0]);
    });
  });

  describe("eyeTyped", () => {
    it("creates a 3x3 identity", () => {
      const eye = eyeTyped(3);
      expect(Array.from(eye)).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    });

    it("creates a 1x1 identity", () => {
      expect(Array.from(eyeTyped(1))).toEqual([1]);
    });
  });

  describe("frobeniusNormTyped", () => {
    it("computes Frobenius norm", () => {
      // [[1,2],[3,4]] -> sqrt(1+4+9+16) = sqrt(30)
      const a = new Float64Array([1, 2, 3, 4]);
      expect(frobeniusNormTyped(a)).toBeCloseTo(Math.sqrt(30), 10);
    });

    it("returns 0 for zero matrix", () => {
      const a = new Float64Array([0, 0, 0, 0]);
      expect(frobeniusNormTyped(a)).toBe(0);
    });
  });
});
