/**
 * Lightweight LRU (Least Recently Used) cache for memoizing expensive
 * pure-function evaluations such as gammaLn, etc.
 */

export interface CacheStats {
  hits: number;
  misses: number;
  size: number;
  maxSize: number;
}

export class LRUCache<V> {
  private map = new Map<string, V>();
  private hits = 0;
  private misses = 0;

  constructor(private readonly maxSize: number) {}

  get(key: string): V | undefined {
    const value = this.map.get(key);
    if (value !== undefined) {
      this.hits++;
      // Move to end (most recently used)
      this.map.delete(key);
      this.map.set(key, value);
      return value;
    }
    this.misses++;
    return undefined;
  }

  set(key: string, value: V): void {
    if (this.map.has(key)) {
      this.map.delete(key);
    } else if (this.map.size >= this.maxSize) {
      // Evict least recently used (first entry)
      const first = this.map.keys().next().value!;
      this.map.delete(first);
    }
    this.map.set(key, value);
  }

  clear(): void {
    this.map.clear();
    this.hits = 0;
    this.misses = 0;
  }

  stats(): CacheStats {
    return {
      hits: this.hits,
      misses: this.misses,
      size: this.map.size,
      maxSize: this.maxSize,
    };
  }
}

// ---------------------------------------------------------------------------
// Global cache configuration
// ---------------------------------------------------------------------------

let cacheEnabled = true;
const DEFAULT_CACHE_SIZE = 1024;

const caches: LRUCache<number>[] = [];

/**
 * Enable or disable all special-function caches globally.
 * When disabled, cached functions evaluate directly without lookup.
 */
export function setCacheEnabled(enabled: boolean): void {
  cacheEnabled = enabled;
}

/** Returns whether special-function caching is currently enabled. */
export function isCacheEnabled(): boolean {
  return cacheEnabled;
}

/** Clear all special-function caches and reset hit/miss counters. */
export function clearAllCaches(): void {
  for (const cache of caches) {
    cache.clear();
  }
}

/** Return aggregate stats across all special-function caches. */
export function getAllCacheStats(): CacheStats {
  let hits = 0;
  let misses = 0;
  let size = 0;
  let maxSize = 0;
  for (const cache of caches) {
    const s = cache.stats();
    hits += s.hits;
    misses += s.misses;
    size += s.size;
    maxSize += s.maxSize;
  }
  return { hits, misses, size, maxSize };
}

/**
 * Creates a cached wrapper around a single-argument numeric function.
 */
export function cachedUnary(
  fn: (x: number) => number,
  size = DEFAULT_CACHE_SIZE,
): (x: number) => number {
  const cache = new LRUCache<number>(size);
  caches.push(cache);
  return (x: number): number => {
    if (!cacheEnabled) return fn(x);
    const key = "" + x;
    const cached = cache.get(key);
    if (cached !== undefined) return cached;
    const result = fn(x);
    cache.set(key, result);
    return result;
  };
}

/**
 * Creates a cached wrapper around a two-argument numeric function.
 */
export function cachedBinary(
  fn: (a: number, b: number) => number,
  size = DEFAULT_CACHE_SIZE,
): (a: number, b: number) => number {
  const cache = new LRUCache<number>(size);
  caches.push(cache);
  return (a: number, b: number): number => {
    if (!cacheEnabled) return fn(a, b);
    const key = a + "," + b;
    const cached = cache.get(key);
    if (cached !== undefined) return cached;
    const result = fn(a, b);
    cache.set(key, result);
    return result;
  };
}
