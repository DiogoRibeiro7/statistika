import {
  LRUCache,
  setCacheEnabled,
  isCacheEnabled,
  clearAllCaches,
  getAllCacheStats,
  cachedUnary,
  cachedBinary,
} from "../../src/utils/lru-cache";
import { gammaLn, erf, erfc, betaFn } from "../../src/utils/math";

describe("LRUCache", () => {
  it("stores and retrieves values", () => {
    const cache = new LRUCache<number>(4);
    cache.set("a", 1);
    cache.set("b", 2);
    expect(cache.get("a")).toBe(1);
    expect(cache.get("b")).toBe(2);
  });

  it("returns undefined for missing keys", () => {
    const cache = new LRUCache<number>(4);
    expect(cache.get("missing")).toBeUndefined();
  });

  it("evicts least recently used entry when full", () => {
    const cache = new LRUCache<number>(3);
    cache.set("a", 1);
    cache.set("b", 2);
    cache.set("c", 3);
    // "a" is the LRU entry
    cache.set("d", 4);
    expect(cache.get("a")).toBeUndefined();
    expect(cache.get("b")).toBe(2);
    expect(cache.get("d")).toBe(4);
  });

  it("accessing a key refreshes its position", () => {
    const cache = new LRUCache<number>(3);
    cache.set("a", 1);
    cache.set("b", 2);
    cache.set("c", 3);
    // Access "a" to make it recently used
    cache.get("a");
    // Now "b" is LRU
    cache.set("d", 4);
    expect(cache.get("a")).toBe(1);
    expect(cache.get("b")).toBeUndefined();
  });

  it("updates value for existing key", () => {
    const cache = new LRUCache<number>(3);
    cache.set("a", 1);
    cache.set("a", 99);
    expect(cache.get("a")).toBe(99);
    expect(cache.stats().size).toBe(1);
  });

  it("tracks hit/miss statistics", () => {
    const cache = new LRUCache<number>(4);
    cache.set("a", 1);
    cache.get("a"); // hit
    cache.get("a"); // hit
    cache.get("b"); // miss
    const stats = cache.stats();
    expect(stats.hits).toBe(2);
    expect(stats.misses).toBe(1);
    expect(stats.size).toBe(1);
    expect(stats.maxSize).toBe(4);
  });

  it("clear resets entries and counters", () => {
    const cache = new LRUCache<number>(4);
    cache.set("a", 1);
    cache.get("a");
    cache.get("b");
    cache.clear();
    const stats = cache.stats();
    expect(stats.hits).toBe(0);
    expect(stats.misses).toBe(0);
    expect(stats.size).toBe(0);
  });
});

describe("cachedUnary", () => {
  it("returns correct results and caches them", () => {
    let callCount = 0;
    const fn = cachedUnary((x: number) => {
      callCount++;
      return x * x;
    }, 8);

    expect(fn(3)).toBe(9);
    expect(fn(3)).toBe(9);
    expect(fn(4)).toBe(16);
    expect(callCount).toBe(2); // 3 and 4, not 3 twice
  });
});

describe("cachedBinary", () => {
  it("returns correct results and caches them", () => {
    let callCount = 0;
    const fn = cachedBinary((a: number, b: number) => {
      callCount++;
      return a + b;
    }, 8);

    expect(fn(1, 2)).toBe(3);
    expect(fn(1, 2)).toBe(3);
    expect(fn(2, 3)).toBe(5);
    expect(callCount).toBe(2);
  });
});

describe("global cache control", () => {
  afterEach(() => {
    setCacheEnabled(true);
    clearAllCaches();
  });

  it("setCacheEnabled / isCacheEnabled", () => {
    expect(isCacheEnabled()).toBe(true);
    setCacheEnabled(false);
    expect(isCacheEnabled()).toBe(false);
    setCacheEnabled(true);
    expect(isCacheEnabled()).toBe(true);
  });

  it("disabled cache still produces correct results", () => {
    setCacheEnabled(false);
    expect(gammaLn(5)).toBeCloseTo(Math.log(24), 8);
    expect(erf(0)).toBeCloseTo(0, 8);
    expect(erfc(0)).toBeCloseTo(1, 8);
    expect(betaFn(1, 1)).toBeCloseTo(1, 8);
  });

  it("getAllCacheStats returns aggregated stats", () => {
    // Exercise two functions that are intentionally memoized.
    gammaLn(5);
    gammaLn(5); // cache hit
    betaFn(2, 3); // second cache miss
    const stats = getAllCacheStats();
    expect(stats.hits).toBeGreaterThanOrEqual(1);
    expect(stats.misses).toBeGreaterThanOrEqual(2);
  });

  it("clearAllCaches resets stats", () => {
    gammaLn(5);
    clearAllCaches();
    const stats = getAllCacheStats();
    expect(stats.hits).toBe(0);
    expect(stats.misses).toBe(0);
    expect(stats.size).toBe(0);
  });
});

describe("special functions correctness", () => {
  afterEach(() => {
    setCacheEnabled(true);
    clearAllCaches();
  });

  it("gammaLn returns same value on repeated calls", () => {
    const first = gammaLn(5.5);
    const second = gammaLn(5.5);
    expect(first).toBe(second);
  });

  it("erf returns same value on repeated calls", () => {
    const first = erf(1.5);
    const second = erf(1.5);
    expect(first).toBe(second);
  });

  it("erfc returns same value on repeated calls", () => {
    const first = erfc(0.7);
    const second = erfc(0.7);
    expect(first).toBe(second);
  });

  it("betaFn returns same value on repeated calls", () => {
    const first = betaFn(2, 3);
    const second = betaFn(2, 3);
    expect(first).toBe(second);
  });
});
