import {
  raschModel,
  twoPlModel,
  threePlModel,
  gradedResponseModel,
  itemInformation,
  testInformation,
  itemCharacteristicCurve,
  estimateAbility,
} from "../src/irt";

describe("raschModel", () => {
  it("probability is 0.5 when theta equals difficulty", () => {
    const result = raschModel(1, { b: 1 });
    expect(result.probability).toBeCloseTo(0.5, 5);
    expect(result.theta).toBe(1);
  });

  it("probability increases with higher theta", () => {
    const low = raschModel(-2, { b: 0 });
    const high = raschModel(2, { b: 0 });
    expect(high.probability).toBeGreaterThan(low.probability);
  });

  it("probability is between 0 and 1", () => {
    for (const theta of [-5, -2, 0, 2, 5]) {
      const result = raschModel(theta, { b: 0 });
      expect(result.probability).toBeGreaterThanOrEqual(0);
      expect(result.probability).toBeLessThanOrEqual(1);
    }
  });
});

describe("twoPlModel", () => {
  it("higher discrimination produces steeper curve", () => {
    const lowDisc = twoPlModel(1, { a: 0.5, b: 0 });
    const highDisc = twoPlModel(1, { a: 2, b: 0 });
    // Both should be above 0.5 since theta > b
    expect(highDisc.probability).toBeGreaterThan(lowDisc.probability);
  });

  it("at theta = b, probability is 0.5", () => {
    const result = twoPlModel(2, { a: 1.5, b: 2 });
    expect(result.probability).toBeCloseTo(0.5, 5);
  });

  it("defaults to a=1 when not specified", () => {
    const withA = twoPlModel(1, { a: 1, b: 0 });
    const withoutA = twoPlModel(1, { b: 0 });
    expect(withA.probability).toBeCloseTo(withoutA.probability, 8);
  });
});

describe("threePlModel", () => {
  it("lower asymptote equals guessing parameter", () => {
    const result = threePlModel(-100, { a: 1, b: 0, c: 0.25 });
    expect(result.probability).toBeCloseTo(0.25, 2);
  });

  it("upper asymptote approaches 1", () => {
    const result = threePlModel(100, { a: 1, b: 0, c: 0.25 });
    expect(result.probability).toBeCloseTo(1, 2);
  });

  it("reduces to 2PL when c=0", () => {
    const twopl = twoPlModel(1, { a: 1.5, b: 0 });
    const threepl = threePlModel(1, { a: 1.5, b: 0, c: 0 });
    expect(threepl.probability).toBeCloseTo(twopl.probability, 8);
  });
});

describe("gradedResponseModel", () => {
  it("category probabilities sum to 1", () => {
    const result = gradedResponseModel(0, { a: 1, b: [-1, 0, 1] });
    const total = result.categoryProbabilities.reduce((s, p) => s + p, 0);
    expect(total).toBeCloseTo(1, 5);
  });

  it("number of categories equals b.length + 1", () => {
    const result = gradedResponseModel(0, { a: 1, b: [-1, 1] });
    expect(result.categoryProbabilities).toHaveLength(3);
  });

  it("all probabilities are non-negative", () => {
    const result = gradedResponseModel(0, { a: 1.5, b: [-2, -0.5, 0.5, 2] });
    for (const p of result.categoryProbabilities) {
      expect(p).toBeGreaterThanOrEqual(0);
    }
  });

  it("high theta favors higher categories", () => {
    const low = gradedResponseModel(-3, { a: 1, b: [-1, 0, 1] });
    const high = gradedResponseModel(3, { a: 1, b: [-1, 0, 1] });
    // High theta: last category should have higher prob
    expect(high.categoryProbabilities[3]).toBeGreaterThan(low.categoryProbabilities[3]);
  });
});

describe("itemInformation", () => {
  it("peaks at the difficulty parameter for 1PL", () => {
    const atB = itemInformation(0, { b: 0 });
    const away = itemInformation(3, { b: 0 });
    expect(atB.information).toBeGreaterThan(away.information);
  });

  it("higher discrimination yields more information", () => {
    const lowA = itemInformation(0, { a: 0.5, b: 0 });
    const highA = itemInformation(0, { a: 2, b: 0 });
    expect(highA.information).toBeGreaterThan(lowA.information);
  });

  it("information is non-negative", () => {
    for (const theta of [-3, -1, 0, 1, 3]) {
      const info = itemInformation(theta, { a: 1, b: 0, c: 0.2 });
      expect(info.information).toBeGreaterThanOrEqual(0);
    }
  });
});

describe("testInformation", () => {
  it("equals sum of item information", () => {
    const items = [{ a: 1, b: -1 }, { a: 1.5, b: 0 }, { a: 0.8, b: 1 }];
    const testInfo = testInformation(0, items);
    let sum = 0;
    for (const item of items) {
      sum += itemInformation(0, item).information;
    }
    expect(testInfo.information).toBeCloseTo(sum, 8);
  });
});

describe("itemCharacteristicCurve", () => {
  it("returns the correct number of points", () => {
    const curve = itemCharacteristicCurve({ a: 1, b: 0 });
    expect(curve).toHaveLength(81);
    expect(curve[0].theta).toBeCloseTo(-4, 1);
    expect(curve[80].theta).toBeCloseTo(4, 1);
  });

  it("probabilities are monotonically increasing", () => {
    const curve = itemCharacteristicCurve({ a: 1, b: 0 });
    for (let i = 1; i < curve.length; i++) {
      expect(curve[i].probability).toBeGreaterThanOrEqual(curve[i - 1].probability - 1e-10);
    }
  });
});

describe("estimateAbility", () => {
  it("estimates ability from a set of item responses", () => {
    const items = [
      { a: 1, b: -2 },
      { a: 1, b: -1 },
      { a: 1, b: 0 },
      { a: 1, b: 1 },
      { a: 1, b: 2 },
    ];
    // All correct: high ability
    const highTheta = estimateAbility([1, 1, 1, 1, 1], items);
    // All incorrect: low ability
    const lowTheta = estimateAbility([0, 0, 0, 0, 0], items);
    expect(highTheta).toBeGreaterThan(lowTheta);
  });

  it("mixed responses give intermediate ability", () => {
    const items = [
      { a: 1, b: -2 },
      { a: 1, b: -1 },
      { a: 1, b: 0 },
      { a: 1, b: 1 },
      { a: 1, b: 2 },
    ];
    const theta = estimateAbility([1, 1, 1, 0, 0], items);
    expect(theta).toBeGreaterThan(-2);
    expect(theta).toBeLessThan(2);
  });

  it("throws on mismatched lengths", () => {
    expect(() => estimateAbility([1, 0], [{ b: 0 }])).toThrow();
  });
});
