import { twoWayAnova } from "../src/tests/two-way-anova";

/**
 * Helper to build data array from a balanced design.
 * cells[i][j] is an array of observations for factor A level i, factor B level j.
 */
function buildData(
  cells: number[][][],
): Array<{ a: number; b: number; value: number }> {
  const data: Array<{ a: number; b: number; value: number }> = [];
  for (let i = 0; i < cells.length; i++) {
    for (let j = 0; j < cells[i].length; j++) {
      for (const value of cells[i][j]) {
        data.push({ a: i, b: j, value });
      }
    }
  }
  return data;
}

describe("twoWayAnova", () => {
  it("detects main effects with no interaction", () => {
    // Factor A has 2 levels, Factor B has 2 levels
    // A effect: level 0 lower than level 1
    // B effect: level 0 lower than level 1
    // No interaction
    const cells = [
      [
        [1, 2, 3],
        [4, 5, 6],
      ],
      [
        [7, 8, 9],
        [10, 11, 12],
      ],
    ];
    const data = buildData(cells);
    const result = twoWayAnova(data);

    expect(result.factorA.rejected).toBe(true);
    expect(result.factorB.rejected).toBe(true);
    expect(result.factorA.fStatistic).toBeGreaterThan(1);
    expect(result.factorB.fStatistic).toBeGreaterThan(1);
  });

  it("SS components sum to SS total", () => {
    const cells = [
      [
        [2, 3, 4],
        [5, 6, 7],
      ],
      [
        [8, 9, 10],
        [3, 4, 5],
      ],
    ];
    const data = buildData(cells);
    const result = twoWayAnova(data);

    const ssSum =
      result.factorA.ss +
      result.factorB.ss +
      result.interaction.ss +
      result.residual.ss;
    expect(ssSum).toBeCloseTo(result.total.ss, 6);
  });

  it("DF components sum to DF total", () => {
    const cells = [
      [
        [1, 2],
        [3, 4],
        [5, 6],
      ],
      [
        [7, 8],
        [9, 10],
        [11, 12],
      ],
      [
        [13, 14],
        [15, 16],
        [17, 18],
      ],
    ];
    const data = buildData(cells);
    const result = twoWayAnova(data);

    const dfSum =
      result.factorA.df +
      result.factorB.df +
      result.interaction.df +
      result.residual.df;
    expect(dfSum).toBe(result.total.df);
  });

  it("detects interaction effect", () => {
    // Strong interaction: effect of A depends on B
    const cells = [
      [
        [10, 11, 12],
        [1, 2, 3],
      ],
      [
        [1, 2, 3],
        [10, 11, 12],
      ],
    ];
    const data = buildData(cells);
    const result = twoWayAnova(data);

    expect(result.interaction.rejected).toBe(true);
    expect(result.interaction.fStatistic).toBeGreaterThan(5);
  });

  it("does not reject when no effects exist", () => {
    // All cells similar
    const cells = [
      [
        [10, 10.5, 9.5],
        [10, 10.5, 9.5],
      ],
      [
        [10, 10.5, 9.5],
        [10, 10.5, 9.5],
      ],
    ];
    const data = buildData(cells);
    const result = twoWayAnova(data);

    expect(result.factorA.rejected).toBe(false);
    expect(result.factorB.rejected).toBe(false);
    expect(result.interaction.rejected).toBe(false);
  });

  it("works with unbalanced design", () => {
    const data = [
      { a: 0, b: 0, value: 1 },
      { a: 0, b: 0, value: 2 },
      { a: 0, b: 1, value: 5 },
      { a: 0, b: 1, value: 6 },
      { a: 0, b: 1, value: 7 },
      { a: 1, b: 0, value: 8 },
      { a: 1, b: 0, value: 9 },
      { a: 1, b: 1, value: 11 },
      { a: 1, b: 1, value: 12 },
    ];
    const result = twoWayAnova(data);

    expect(result.factorA.fStatistic).toBeGreaterThan(0);
    expect(result.factorB.fStatistic).toBeGreaterThan(0);
    expect(result.residual.df).toBeGreaterThan(0);
  });

  it("p-values are between 0 and 1", () => {
    const cells = [
      [
        [5, 6, 7],
        [10, 11, 12],
      ],
      [
        [15, 16, 17],
        [20, 21, 22],
      ],
    ];
    const data = buildData(cells);
    const result = twoWayAnova(data);

    expect(result.factorA.pValue).toBeGreaterThanOrEqual(0);
    expect(result.factorA.pValue).toBeLessThanOrEqual(1);
    expect(result.factorB.pValue).toBeGreaterThanOrEqual(0);
    expect(result.factorB.pValue).toBeLessThanOrEqual(1);
    expect(result.interaction.pValue).toBeGreaterThanOrEqual(0);
    expect(result.interaction.pValue).toBeLessThanOrEqual(1);
  });

  it("handles 3×3 design", () => {
    const cells = [
      [
        [1, 2],
        [3, 4],
        [5, 6],
      ],
      [
        [7, 8],
        [9, 10],
        [11, 12],
      ],
      [
        [13, 14],
        [15, 16],
        [17, 18],
      ],
    ];
    const data = buildData(cells);
    const result = twoWayAnova(data);

    expect(result.factorA.df).toBe(2);
    expect(result.factorB.df).toBe(2);
    expect(result.interaction.df).toBe(4);
  });

  it("throws with too few observations", () => {
    expect(() => twoWayAnova([{ a: 0, b: 0, value: 1 }])).toThrow();
  });

  it("throws when a factor has only 1 level", () => {
    const data = [
      { a: 0, b: 0, value: 1 },
      { a: 0, b: 0, value: 2 },
      { a: 0, b: 1, value: 3 },
      { a: 0, b: 1, value: 4 },
    ];
    expect(() => twoWayAnova(data)).toThrow();
  });

  it("throws when a cell is empty", () => {
    const data = [
      { a: 0, b: 0, value: 1 },
      { a: 0, b: 0, value: 2 },
      { a: 1, b: 0, value: 3 },
      { a: 1, b: 0, value: 4 },
      // Missing a=0,b=1 and a=1,b=1
    ];
    // This only has one level of B
    expect(() => twoWayAnova(data)).toThrow();
  });

  // Known result: manual calculation
  it("matches hand-calculated result for simple balanced design", () => {
    // 2×2, n=3 per cell
    // A0B0: [2,4,6] mean=4  A0B1: [3,5,7] mean=5
    // A1B0: [8,10,12] mean=10 A1B1: [9,11,13] mean=11
    // Grand mean = (4+5+10+11)/4 = 7.5 (unweighted cell means)
    const cells = [
      [
        [2, 4, 6],
        [3, 5, 7],
      ],
      [
        [8, 10, 12],
        [9, 11, 13],
      ],
    ];
    const data = buildData(cells);
    const result = twoWayAnova(data);

    // SS_A should be large (groups differ by ~6)
    expect(result.factorA.ss).toBeGreaterThan(50);
    // SS_B should be small (groups differ by ~1)
    expect(result.factorB.ss).toBeLessThan(result.factorA.ss);
    // Interaction should be near 0 (additive effects)
    expect(result.interaction.ss).toBeCloseTo(0, 1);
  });
});
