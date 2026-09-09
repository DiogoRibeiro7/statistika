import {
  betaSampleBatch,
  gammaSampleBatch,
  hasNativeSampling,
  normalSampleBatch,
  uniformSampleBatch,
} from "../src/utils/native-sampling";

describe("native sampling bridge", () => {
  it("produces deterministic batch samples with a fixed seed", () => {
    const seed = 12345;

    const u1 = uniformSampleBatch(10, 0, 1, seed);
    const u2 = uniformSampleBatch(10, 0, 1, seed);
    expect(u1).toEqual(u2);

    const n1 = normalSampleBatch(10, 0, 1, seed);
    const n2 = normalSampleBatch(10, 0, 1, seed);
    expect(n1).toEqual(n2);

    const g1 = gammaSampleBatch(10, 2, 1, seed);
    const g2 = gammaSampleBatch(10, 2, 1, seed);
    expect(g1).toEqual(g2);

    const b1 = betaSampleBatch(10, 2, 5, seed);
    const b2 = betaSampleBatch(10, 2, 5, seed);
    expect(b1).toEqual(b2);
  });

  it("returns arrays of the expected length and range", () => {
    const u = uniformSampleBatch(20, -1, 2, 987);
    expect(u).toHaveLength(20);
    expect(u.every((x) => x >= -1 && x < 2)).toBe(true);

    const n = normalSampleBatch(20, 1, 0.5, 987);
    expect(n).toHaveLength(20);

    const g = gammaSampleBatch(20, 3, 1, 987);
    expect(g).toHaveLength(20);
    expect(g.every((x) => x >= 0)).toBe(true);

    const b = betaSampleBatch(20, 2, 3, 987);
    expect(b).toHaveLength(20);
    expect(b.every((x) => x >= 0 && x <= 1)).toBe(true);
  });

  it("exposes native sampling support when the addon is loaded", () => {
    expect(typeof hasNativeSampling).toBe("boolean");
  });
});
