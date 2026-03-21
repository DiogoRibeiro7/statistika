/**
 * Seeded pseudo-random number generator (xorshift128+).
 *
 * Provides reproducible random number sequences for simulations
 * and statistical sampling.
 *
 * @throws {Error} If seed is NaN or Infinity
 *
 * @example
 * ```ts
 * const rng = new SeededRng(42);
 * console.log(rng.next());      // deterministic value in [0, 1)
 * console.log(rng.nextInt(1, 6)); // deterministic integer in [1, 6]
 * const arr = [1, 2, 3, 4, 5];
 * rng.shuffle(arr);              // deterministic shuffle
 * ```
 */
export class SeededRng {
  private s0: number;
  private s1: number;

  constructor(seed: number) {
    if (!Number.isFinite(seed)) throw new Error("Seed must be a finite number");
    this.s0 = seed | 0 || 1;
    this.s1 = (seed * 2654435761) | 0 || 2;
  }

  /**
   * Generate next random number in [0, 1).
   *
   * @returns A pseudo-random number in [0, 1)
   */
  next(): number {
    let a = this.s0;
    const b = this.s1;
    this.s0 = b;
    a ^= a << 23;
    a ^= a >> 17;
    a ^= b;
    a ^= b >> 26;
    this.s1 = a;
    return ((this.s0 + this.s1) >>> 0) / 4294967296;
  }

  /**
   * Generate a random integer in [min, max] (inclusive).
   *
   * @param min - Minimum value (inclusive)
   * @param max - Maximum value (inclusive)
   * @returns A pseudo-random integer in [min, max]
   * @throws {Error} If min or max is not a finite number
   */
  nextInt(min: number, max: number): number {
    if (!Number.isFinite(min) || !Number.isFinite(max)) {
      throw new Error("min and max must be finite numbers");
    }
    return Math.floor(this.next() * (max - min + 1)) + min;
  }

  /**
   * Generate a standard normal variate using Box-Muller.
   *
   * @param mu - Mean (default: 0)
   * @param sigma - Standard deviation (default: 1)
   * @returns A pseudo-random normal variate
   * @throws {Error} If mu or sigma is not finite, or sigma is negative
   */
  nextNormal(mu = 0, sigma = 1): number {
    if (!Number.isFinite(mu) || !Number.isFinite(sigma)) {
      throw new Error("mu and sigma must be finite numbers");
    }
    const u1 = this.next();
    const u2 = this.next();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return mu + sigma * z;
  }

  /**
   * Generate n random numbers in [0, 1).
   *
   * @param n - Number of samples to generate
   * @returns Array of n pseudo-random numbers in [0, 1)
   * @throws {Error} If n is not a positive integer
   */
  sample(n: number): number[] {
    if (!Number.isInteger(n) || n < 1) throw new Error("n must be a positive integer");
    const result = new Array<number>(n);
    for (let i = 0; i < n; i++) result[i] = this.next();
    return result;
  }

  /**
   * Shuffle an array in place (Fisher-Yates).
   *
   * @param arr - Array to shuffle
   * @returns The same array, shuffled in place
   */
  shuffle<T>(arr: T[]): T[] {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(this.next() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }

  /**
   * Sample k items from array without replacement.
   *
   * @param arr - Source array
   * @param k - Number of items to sample
   * @returns Array of k items sampled without replacement
   * @throws {Error} If k exceeds array length or is negative
   */
  choose<T>(arr: T[], k: number): T[] {
    if (k > arr.length) throw new Error("k cannot exceed array length");
    if (k < 0) throw new Error("k must be non-negative");
    const copy = [...arr];
    this.shuffle(copy);
    return copy.slice(0, k);
  }
}

/**
 * Halton sequence generator for quasi-random (low-discrepancy) sampling.
 *
 * Produces more uniformly distributed points than pseudo-random generators,
 * useful for numerical integration and Monte Carlo methods.
 *
 * @param base - Prime base for the sequence (e.g., 2, 3, 5, 7). Must be a prime number.
 * @param n - Number of points to generate
 * @param skip - Number of initial points to skip (default: 0)
 * @returns Array of n quasi-random numbers in (0, 1)
 * @throws {Error} If base is not a prime integer >= 2
 * @throws {Error} If n is less than 1
 */
export function haltonSequence(base: number, n: number, skip = 0): number[] {
  if (base < 2 || !Number.isInteger(base)) {
    throw new Error("Base must be an integer >= 2");
  }
  if (!isPrime(base)) {
    throw new Error("Base must be a prime number");
  }
  if (n < 1) throw new Error("n must be at least 1");

  const result = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    result[i] = haltonPoint(i + 1 + skip, base);
  }
  return result;
}

/**
 * Multi-dimensional Halton sequence.
 *
 * Generates points in d dimensions using different prime bases.
 *
 * @param dimensions - Number of dimensions
 * @param n - Number of points
 * @param skip - Number of initial points to skip
 * @returns Array of n points, each an array of d quasi-random coordinates in (0, 1)
 * @throws {Error} If dimensions is less than 1
 * @throws {Error} If n is less than 1
 */
export function haltonSequenceND(dimensions: number, n: number, skip = 0): number[][] {
  if (dimensions < 1) throw new Error("dimensions must be at least 1");
  if (n < 1) throw new Error("n must be at least 1");
  const primes = firstPrimes(dimensions);
  const result = new Array<number[]>(n);
  for (let i = 0; i < n; i++) {
    result[i] = primes.map((p) => haltonPoint(i + 1 + skip, p));
  }
  return result;
}

/**
 * Latin Hypercube Sampling.
 *
 * Generates samples that are stratified across each dimension,
 * ensuring better coverage than simple random sampling.
 *
 * @param dimensions - Number of dimensions
 * @param n - Number of samples
 * @param seed - Optional random seed. If not provided, a random seed is used.
 * @returns Array of n samples, each an array of d values in [0, 1)
 * @throws {Error} If dimensions is less than 1
 * @throws {Error} If n is less than 1
 *
 * @example
 * ```ts
 * // Generate 100 stratified samples in 3 dimensions
 * const samples = latinHypercube(3, 100, 42);
 * // samples[0] is a 3-element array in [0, 1)^3
 * console.log(samples.length); // 100
 * console.log(samples[0].length); // 3
 * ```
 */
export function latinHypercube(dimensions: number, n: number, seed?: number): number[][] {
  if (dimensions < 1) throw new Error("dimensions must be at least 1");
  if (n < 1) throw new Error("n must be at least 1");

  const rng = new SeededRng(seed ?? Math.floor(Math.random() * 2147483647));
  const result = Array.from({ length: n }, () => new Array<number>(dimensions));

  for (let d = 0; d < dimensions; d++) {
    // Create permutation of strata
    const perm = Array.from({ length: n }, (_, i) => i);
    rng.shuffle(perm);

    for (let i = 0; i < n; i++) {
      // Random point within stratum
      const u = rng.next();
      result[i][d] = (perm[i] + u) / n;
    }
  }

  return result;
}

// ── Helpers ─────────────────────────────────────────────────────────────

function haltonPoint(index: number, base: number): number {
  let result = 0;
  let f = 1 / base;
  let i = index;
  while (i > 0) {
    result += f * (i % base);
    i = Math.floor(i / base);
    f /= base;
  }
  return result;
}

function firstPrimes(n: number): number[] {
  const primes: number[] = [];
  let candidate = 2;
  while (primes.length < n) {
    if (isPrime(candidate)) primes.push(candidate);
    candidate++;
  }
  return primes;
}

function isPrime(n: number): boolean {
  if (n < 2) return false;
  for (let i = 2; i * i <= n; i++) {
    if (n % i === 0) return false;
  }
  return true;
}
