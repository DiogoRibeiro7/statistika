import {
  sma,
  ema,
  wma,
  loess,
  cubicSpline,
} from "../src/smoothing";

describe("Smoothing & Interpolation", () => {
  describe("sma", () => {
    it("computes SMA correctly", () => {
      const result = sma([1, 2, 3, 4, 5], 3);
      expect(result).toHaveLength(3);
      expect(result[0]).toBeCloseTo(2); // (1+2+3)/3
      expect(result[1]).toBeCloseTo(3); // (2+3+4)/3
      expect(result[2]).toBeCloseTo(4); // (3+4+5)/3
    });

    it("returns original data for window=1", () => {
      const data = [1, 2, 3];
      expect(sma(data, 1)).toEqual(data);
    });

    it("throws on invalid window", () => {
      expect(() => sma([1, 2], 3)).toThrow("between 1");
      expect(() => sma([1, 2], 0)).toThrow("between 1");
    });
  });

  describe("ema", () => {
    it("starts with first value", () => {
      const result = ema([10, 20, 30], 0.5);
      expect(result[0]).toBe(10);
    });

    it("smooths data", () => {
      const data = [1, 10, 1, 10, 1, 10];
      const result = ema(data, 0.3);
      // EMA should be smoother than raw data
      const rawRange = Math.max(...data) - Math.min(...data);
      const emaRange = Math.max(...result) - Math.min(...result);
      expect(emaRange).toBeLessThan(rawRange);
    });

    it("alpha=1 returns original data", () => {
      const data = [1, 2, 3];
      expect(ema(data, 1)).toEqual(data);
    });

    it("throws on invalid alpha", () => {
      expect(() => ema([1], 0)).toThrow("between 0");
      expect(() => ema([1], 1.5)).toThrow("between 0");
    });
  });

  describe("wma", () => {
    it("computes WMA with linearly increasing weights", () => {
      const result = wma([1, 2, 3, 4, 5], 3);
      expect(result).toHaveLength(3);
      // WMA(1,2,3) with weights 1,2,3: (1*1 + 2*2 + 3*3)/(1+2+3) = 14/6
      expect(result[0]).toBeCloseTo(14 / 6);
    });
  });

  describe("loess", () => {
    it("smooths noisy linear data", () => {
      const x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      const y = [2.1, 3.9, 6.2, 7.8, 10.1, 12.0, 13.8, 16.2, 17.9, 20.1];
      const smoothed = loess(x, y, 0.5);

      expect(smoothed).toHaveLength(10);
      // Smoothed values should be close to 2*x trend
      for (let i = 0; i < 10; i++) {
        expect(smoothed[i]).toBeCloseTo(2 * x[i], 0);
      }
    });

    it("throws on mismatched lengths", () => {
      expect(() => loess([1, 2], [1])).toThrow("same length");
    });

    it("throws on too few points", () => {
      expect(() => loess([1, 2], [1, 2])).toThrow("at least 3");
    });
  });

  describe("cubicSpline", () => {
    it("passes through all data points", () => {
      const xs = [0, 1, 2, 3, 4];
      const ys = [0, 1, 0, 1, 0];
      const spline = cubicSpline(xs, ys);

      for (let i = 0; i < xs.length; i++) {
        expect(spline(xs[i])).toBeCloseTo(ys[i], 10);
      }
    });

    it("interpolates between knots", () => {
      const xs = [0, 1, 2, 3];
      const ys = [0, 1, 4, 9]; // roughly x^2
      const spline = cubicSpline(xs, ys);

      expect(spline(0.5)).toBeGreaterThan(0);
      expect(spline(0.5)).toBeLessThan(1);
      expect(spline(1.5)).toBeGreaterThan(1);
      expect(spline(1.5)).toBeLessThan(4);
    });

    it("handles linear data", () => {
      const xs = [0, 1, 2, 3, 4];
      const ys = [0, 2, 4, 6, 8];
      const spline = cubicSpline(xs, ys);

      expect(spline(0.5)).toBeCloseTo(1);
      expect(spline(2.5)).toBeCloseTo(5);
    });

    it("throws on unsorted xs", () => {
      expect(() => cubicSpline([2, 1, 3], [0, 0, 0])).toThrow("strictly increasing");
    });

    it("throws on too few points", () => {
      expect(() => cubicSpline([1, 2], [0, 0])).toThrow("at least 3");
    });

    it("throws on mismatched xs and ys lengths", () => {
      expect(() => cubicSpline([1, 2, 3], [0, 0])).toThrow("same length");
    });

    it("throws on NaN in xs", () => {
      expect(() => cubicSpline([1, NaN, 3], [0, 0, 0])).toThrow("NaN or Infinity");
    });

    it("throws on Infinity in ys", () => {
      expect(() => cubicSpline([1, 2, 3], [0, Infinity, 0])).toThrow("NaN or Infinity");
    });

    it("extrapolates below first knot", () => {
      const xs = [0, 1, 2, 3];
      const ys = [0, 1, 4, 9];
      const spline = cubicSpline(xs, ys);
      const val = spline(-1);
      expect(Number.isFinite(val)).toBe(true);
    });

    it("extrapolates above last knot", () => {
      const xs = [0, 1, 2, 3];
      const ys = [0, 1, 4, 9];
      const spline = cubicSpline(xs, ys);
      const val = spline(5);
      expect(Number.isFinite(val)).toBe(true);
    });

    it("throws on duplicate xs", () => {
      expect(() => cubicSpline([1, 1, 2], [0, 0, 0])).toThrow("strictly increasing");
    });
  });

  describe("sma edge cases", () => {
    it("throws on empty dataset", () => {
      expect(() => sma([], 1)).toThrow("must not be empty");
    });

    it("throws on non-integer window", () => {
      expect(() => sma([1, 2, 3], 1.5)).toThrow("must be an integer");
    });

    it("throws on NaN in data", () => {
      expect(() => sma([1, NaN, 3], 2)).toThrow("NaN or Infinity");
    });

    it("throws on Infinity in data", () => {
      expect(() => sma([1, Infinity, 3], 2)).toThrow("NaN or Infinity");
    });

    it("window equal to data length returns single value (the mean)", () => {
      const data = [1, 2, 3, 4, 5];
      const result = sma(data, 5);
      expect(result).toHaveLength(1);
      expect(result[0]).toBeCloseTo(3);
    });
  });

  describe("ema edge cases", () => {
    it("throws on empty dataset", () => {
      expect(() => ema([], 0.5)).toThrow("must not be empty");
    });

    it("throws on negative alpha", () => {
      expect(() => ema([1], -0.5)).toThrow("between 0");
    });

    it("throws on NaN in data", () => {
      expect(() => ema([1, NaN], 0.5)).toThrow("NaN or Infinity");
    });

    it("single element returns that element", () => {
      expect(ema([42], 0.5)).toEqual([42]);
    });

    it("small alpha creates more smoothing", () => {
      const data = [0, 10, 0, 10, 0];
      const smoothLow = ema(data, 0.1);
      const smoothHigh = ema(data, 0.9);
      // Low alpha should have less variation
      const rangeLow = Math.max(...smoothLow) - Math.min(...smoothLow);
      const rangeHigh = Math.max(...smoothHigh) - Math.min(...smoothHigh);
      expect(rangeLow).toBeLessThan(rangeHigh);
    });
  });

  describe("wma edge cases", () => {
    it("throws on empty dataset", () => {
      expect(() => wma([], 1)).toThrow("must not be empty");
    });

    it("throws on invalid window", () => {
      expect(() => wma([1, 2], 3)).toThrow("between 1");
      expect(() => wma([1, 2], 0)).toThrow("between 1");
    });

    it("throws on NaN in data", () => {
      expect(() => wma([1, NaN, 3], 2)).toThrow("NaN or Infinity");
    });

    it("window=1 returns original data", () => {
      expect(wma([1, 2, 3], 1)).toEqual([1, 2, 3]);
    });
  });

  describe("loess edge cases", () => {
    it("throws on invalid span", () => {
      expect(() => loess([1, 2, 3], [1, 2, 3], 0)).toThrow("between 0");
      expect(() => loess([1, 2, 3], [1, 2, 3], 1.5)).toThrow("between 0");
    });

    it("throws on NaN in x", () => {
      expect(() => loess([1, NaN, 3], [1, 2, 3], 0.5)).toThrow("NaN or Infinity");
    });

    it("throws on NaN in y", () => {
      expect(() => loess([1, 2, 3], [1, NaN, 3], 0.5)).toThrow("NaN or Infinity");
    });

    it("throws on all identical x values", () => {
      expect(() => loess([5, 5, 5], [1, 2, 3], 0.5)).toThrow("identical");
    });

    it("span=1 uses all data for each local fit", () => {
      const x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      const y = x.map(v => 2 * v);
      const result = loess(x, y, 1.0);
      expect(result).toHaveLength(10);
      for (let i = 0; i < 10; i++) {
        expect(result[i]).toBeCloseTo(2 * x[i], 0);
      }
    });
  });
});
