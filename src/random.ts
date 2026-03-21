/**
 * Seeded pseudo-random number generator (xorshift128+).
 *
 * Provides reproducible random number sequences for simulations
 * and statistical sampling.
 */
export class SeededRng {
  private s0: number;
  private s1: number;

  constructor(seed: number) {
    this.s0 = seed | 0 || 1;
    this.s1 = (seed * 2654435761) | 0 || 2;
  }

  /** Generate next random number in [0, 1). */
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

  /** Generate a random integer in [min, max] (inclusive). */
  nextInt(min: number, max: number): number {
    return Math.floor(this.next() * (max - min + 1)) + min;
  }

  /** Generate a standard normal variate using Box-Muller. */
  nextNormal(mu = 0, sigma = 1): number {
    const u1 = this.next();
    const u2 = this.next();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return mu + sigma * z;
  }

  /** Generate n random numbers in [0, 1). */
  sample(n: number): number[] {
    const result = new Array<number>(n);
    for (let i = 0; i < n; i++) result[i] = this.next();
    return result;
  }

  /** Shuffle an array in place (Fisher-Yates). */
  shuffle<T>(arr: T[]): T[] {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(this.next() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }

  /** Sample k items from array without replacement. */
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
 * @param base - Prime base for the sequence (e.g., 2, 3, 5, 7)
 * @param n - Number of points to generate
 * @param skip - Number of initial points to skip (default: 0)
 */
export function haltonSequence(base: number, n: number, skip = 0): number[] {
  if (base < 2 || !Number.isInteger(base)) {
    throw new Error("Base must be an integer >= 2");
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
 */
export function haltonSequenceND(dimensions: number, n: number, skip = 0): number[][] {
  if (dimensions < 1) throw new Error("dimensions must be at least 1");
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
 * @param seed - Optional random seed
 */
export function latinHypercube(dimensions: number, n: number, seed?: number): number[][] {
  if (dimensions < 1) throw new Error("dimensions must be at least 1");
  if (n < 1) throw new Error("n must be at least 1");

  const rng = seed != null ? new SeededRng(seed) : { next: Math.random, shuffle: fisherYates };
  const result = Array.from({ length: n }, () => new Array<number>(dimensions));

  for (let d = 0; d < dimensions; d++) {
    // Create permutation of strata
    const perm = Array.from({ length: n }, (_, i) => i);
    if ('shuffle' in rng && typeof rng.shuffle === 'function') {
      (rng as SeededRng).shuffle(perm);
    } else {
      fisherYates(perm);
    }

    for (let i = 0; i < n; i++) {
      // Random point within stratum
      const u = seed != null ? (rng as SeededRng).next() : Math.random();
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

function fisherYates<T>(arr: T[]): T[] {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}
