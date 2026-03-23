import {
  clr, clrInverse,
  alr, alrInverse,
  ilr, ilrInverse,
  closure,
  perturbation,
  powering,
  aitchisonInnerProduct,
  aitchisonDistance,
  aitchisonNorm,
  compositionalCentre,
  variationMatrix,
} from "../src/compositional";

// ── CLR ───────────────────────────────────────────────────────────────────

describe("clr / clrInverse", () => {
  it("CLR of equal composition is zero vector", () => {
    const result = clr([0.25, 0.25, 0.25, 0.25]);
    for (const v of result) expect(v).toBeCloseTo(0, 10);
  });

  it("CLR sums to zero", () => {
    const result = clr([0.1, 0.3, 0.6]);
    const sum = result.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(0, 10);
  });

  it("roundtrips clr → clrInverse", () => {
    const x = [0.2, 0.3, 0.5];
    const recovered = clrInverse(clr(x));
    for (let i = 0; i < x.length; i++) {
      expect(recovered[i]).toBeCloseTo(x[i], 8);
    }
  });
});

// ── ALR ───────────────────────────────────────────────────────────────────

describe("alr / alrInverse", () => {
  it("ALR has D-1 components", () => {
    const result = alr([0.2, 0.3, 0.5]);
    expect(result.length).toBe(2);
  });

  it("ALR of [a, a, a] gives zeros", () => {
    const result = alr([0.33, 0.33, 0.33]);
    for (const v of result) expect(Math.abs(v)).toBeLessThan(0.02);
  });

  it("roundtrips alr → alrInverse", () => {
    const x = [0.1, 0.4, 0.5];
    const recovered = alrInverse(alr(x));
    for (let i = 0; i < x.length; i++) {
      expect(recovered[i]).toBeCloseTo(x[i], 8);
    }
  });

  it("custom reference component works", () => {
    const x = [0.2, 0.3, 0.5];
    const result = alr(x, 0); // reference = first component
    expect(result.length).toBe(2);
    const recovered = alrInverse(result, 0);
    for (let i = 0; i < x.length; i++) {
      expect(recovered[i]).toBeCloseTo(x[i], 8);
    }
  });
});

// ── ILR ───────────────────────────────────────────────────────────────────

describe("ilr / ilrInverse", () => {
  it("ILR has D-1 components", () => {
    const result = ilr([0.2, 0.3, 0.5]);
    expect(result.length).toBe(2);
  });

  it("roundtrips ilr → ilrInverse", () => {
    const x = [0.15, 0.35, 0.5];
    const recovered = ilrInverse(ilr(x));
    for (let i = 0; i < x.length; i++) {
      expect(recovered[i]).toBeCloseTo(x[i], 6);
    }
  });

  it("ILR of equal composition is zero vector", () => {
    const result = ilr([0.25, 0.25, 0.25, 0.25]);
    for (const v of result) expect(Math.abs(v)).toBeLessThan(1e-8);
  });
});

// ── Closure ───────────────────────────────────────────────────────────────

describe("closure", () => {
  it("projects to simplex (sums to 1)", () => {
    const result = closure([2, 3, 5]);
    expect(result.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
    expect(result[0]).toBeCloseTo(0.2, 10);
  });

  it("custom total works", () => {
    const result = closure([1, 1, 1], 100);
    expect(result.reduce((a, b) => a + b, 0)).toBeCloseTo(100, 10);
  });

  it("throws for zero vector", () => {
    expect(() => closure([0, 0, 0])).toThrow("zero vector");
  });
});

// ── Aitchison Geometry ────────────────────────────────────────────────────

describe("perturbation", () => {
  it("perturbation with [1,1,1] is identity (up to closure)", () => {
    const x = [0.2, 0.3, 0.5];
    const result = perturbation(x, [1, 1, 1]);
    for (let i = 0; i < x.length; i++) {
      expect(result[i]).toBeCloseTo(x[i], 8);
    }
  });

  it("result sums to 1", () => {
    const result = perturbation([0.3, 0.3, 0.4], [2, 1, 1]);
    expect(result.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
  });
});

describe("powering", () => {
  it("powering by 1 is identity", () => {
    const x = [0.2, 0.3, 0.5];
    const result = powering(x, 1);
    for (let i = 0; i < x.length; i++) {
      expect(result[i]).toBeCloseTo(x[i], 8);
    }
  });

  it("powering by 0 gives equal parts", () => {
    const result = powering([0.1, 0.2, 0.7], 0);
    // x^0 = 1 for all, so closure gives 1/3 each
    for (const v of result) expect(v).toBeCloseTo(1 / 3, 8);
  });
});

describe("aitchisonDistance", () => {
  it("distance to self is 0", () => {
    expect(aitchisonDistance([0.3, 0.3, 0.4], [0.3, 0.3, 0.4])).toBeCloseTo(0, 10);
  });

  it("distance is symmetric", () => {
    const x = [0.2, 0.3, 0.5];
    const y = [0.4, 0.4, 0.2];
    expect(aitchisonDistance(x, y)).toBeCloseTo(aitchisonDistance(y, x), 10);
  });

  it("distance is positive for different compositions", () => {
    expect(aitchisonDistance([0.1, 0.9], [0.9, 0.1])).toBeGreaterThan(0);
  });
});

describe("aitchisonNorm", () => {
  it("norm of equal composition is 0", () => {
    expect(aitchisonNorm([0.5, 0.5])).toBeCloseTo(0, 8);
    expect(aitchisonNorm([0.25, 0.25, 0.25, 0.25])).toBeCloseTo(0, 8);
  });

  it("norm is non-negative", () => {
    expect(aitchisonNorm([0.1, 0.2, 0.7])).toBeGreaterThanOrEqual(0);
  });
});

// ── Centre and Variation ──────────────────────────────────────────────────

describe("compositionalCentre", () => {
  it("centre of identical compositions is itself", () => {
    const comps = [[0.2, 0.3, 0.5], [0.2, 0.3, 0.5], [0.2, 0.3, 0.5]];
    const centre = compositionalCentre(comps);
    for (let i = 0; i < 3; i++) {
      expect(centre[i]).toBeCloseTo(comps[0][i], 6);
    }
  });

  it("centre sums to 1", () => {
    const comps = [[0.1, 0.2, 0.7], [0.3, 0.3, 0.4], [0.5, 0.1, 0.4]];
    const centre = compositionalCentre(comps);
    expect(centre.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 8);
  });
});

describe("variationMatrix", () => {
  it("diagonal is zero", () => {
    const comps = [[0.2, 0.3, 0.5], [0.3, 0.2, 0.5], [0.1, 0.4, 0.5]];
    const T = variationMatrix(comps);
    for (let i = 0; i < 3; i++) expect(T[i][i]).toBe(0);
  });

  it("is symmetric", () => {
    const comps = [[0.2, 0.3, 0.5], [0.3, 0.2, 0.5], [0.1, 0.4, 0.5]];
    const T = variationMatrix(comps);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        expect(T[i][j]).toBeCloseTo(T[j][i], 10);
      }
    }
  });

  it("values are non-negative", () => {
    const comps = [[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]];
    const T = variationMatrix(comps);
    for (const row of T) {
      for (const v of row) expect(v).toBeGreaterThanOrEqual(0);
    }
  });
});
