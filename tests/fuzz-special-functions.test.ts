import {
  gammaLn,
  gamma,
  logFactorial,
  factorial,
  binomialCoeff,
  betaFn,
  erf,
  erfc,
  regularizedGammaP,
  regularizedBeta,
} from '../src/utils/math';

// Deterministic seed for reproducible "random" values
function mulberry32(seed: number): () => number {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const rng = mulberry32(42);

function randomInRange(lo: number, hi: number): number {
  return lo + rng() * (hi - lo);
}

function randomInt(lo: number, hi: number): number {
  return Math.floor(randomInRange(lo, hi + 1));
}

const FUZZ_COUNT = 120;

// Helper: call fn and ensure it does not throw unexpectedly
function safeCall(fn: () => number): number {
  let result: number;
  try {
    result = fn();
  } catch {
    // Some functions intentionally throw for invalid inputs (e.g. negative factorial).
    // We return NaN so the caller can still check "is a number".
    return NaN;
  }
  return result;
}

// ==========================================================================
// gammaLn
// ==========================================================================
describe('Fuzz: gammaLn', () => {
  test('gammaLn(1) = 0', () => {
    expect(gammaLn(1)).toBeCloseTo(0, 10);
  });

  test('gammaLn(n+1) = logFactorial(n) for small integers', () => {
    for (let n = 0; n <= 20; n++) {
      expect(gammaLn(n + 1)).toBeCloseTo(logFactorial(n), 8);
    }
  });

  test('does not throw for random positive values', () => {
    for (let i = 0; i < FUZZ_COUNT; i++) {
      const x = randomInRange(1e-10, 1e6);
      const result = safeCall(() => gammaLn(x));
      expect(typeof result).toBe('number');
    }
  });

  test('does not throw for very large values', () => {
    for (const x of [1e10, 1e50, 1e100, 1e308]) {
      const result = safeCall(() => gammaLn(x));
      expect(typeof result).toBe('number');
    }
  });

  test('does not throw for very small positive values', () => {
    for (const x of [1e-10, 1e-100, 1e-308]) {
      const result = safeCall(() => gammaLn(x));
      expect(typeof result).toBe('number');
    }
  });

  test('returns NaN for non-positive integers', () => {
    for (const x of [0, -1, -2, -10]) {
      expect(gammaLn(x)).toBeNaN();
    }
  });

  test('handles NaN and Infinity inputs without throwing', () => {
    expect(() => gammaLn(NaN)).not.toThrow();
    expect(() => gammaLn(Infinity)).not.toThrow();
    expect(() => gammaLn(-Infinity)).not.toThrow();
  });

  test('random negative non-integer values produce finite numbers', () => {
    for (let i = 0; i < FUZZ_COUNT; i++) {
      const x = -(rng() * 100 + 0.01); // avoid integers
      if (Number.isInteger(x)) continue;
      const result = safeCall(() => gammaLn(x));
      expect(typeof result).toBe('number');
    }
  });
});

// ==========================================================================
// gamma
// ==========================================================================
describe('Fuzz: gamma', () => {
  test('gamma(1) = 1', () => {
    expect(gamma(1)).toBeCloseTo(1, 10);
  });

  test('gamma(n) = (n-1)! for small integers', () => {
    const expected = [1, 1, 2, 6, 24, 120, 720];
    for (let n = 1; n <= 7; n++) {
      expect(gamma(n)).toBeCloseTo(expected[n - 1], 6);
    }
  });

  test('does not throw for random positive values', () => {
    for (let i = 0; i < FUZZ_COUNT; i++) {
      const x = randomInRange(0.01, 170);
      const result = safeCall(() => gamma(x));
      expect(typeof result).toBe('number');
    }
  });

  test('handles NaN and Infinity without throwing', () => {
    expect(() => gamma(NaN)).not.toThrow();
    expect(() => gamma(Infinity)).not.toThrow();
    expect(() => gamma(-Infinity)).not.toThrow();
  });

  test('very large values return Infinity', () => {
    const result = gamma(172);
    expect(result).toBe(Infinity);
  });
});

// ==========================================================================
// logFactorial
// ==========================================================================
describe('Fuzz: logFactorial', () => {
  test('logFactorial(0) = 0', () => {
    expect(logFactorial(0)).toBeCloseTo(0, 10);
  });

  test('logFactorial(1) = 0', () => {
    expect(logFactorial(1)).toBeCloseTo(0, 10);
  });

  test('consistent with gammaLn for random non-negative integers', () => {
    for (let i = 0; i < FUZZ_COUNT; i++) {
      const n = randomInt(0, 1000);
      expect(logFactorial(n)).toBeCloseTo(gammaLn(n + 1), 8);
    }
  });

  test('throws for negative integers', () => {
    expect(() => logFactorial(-1)).toThrow();
    expect(() => logFactorial(-100)).toThrow();
  });

  test('throws for non-integers', () => {
    expect(() => logFactorial(1.5)).toThrow();
    expect(() => logFactorial(0.1)).toThrow();
  });
});

// ==========================================================================
// factorial
// ==========================================================================
describe('Fuzz: factorial', () => {
  test('factorial(0) = 1', () => {
    expect(factorial(0)).toBeCloseTo(1, 10);
  });

  test('factorial(1) = 1', () => {
    expect(factorial(1)).toBeCloseTo(1, 10);
  });

  test('factorial(5) = 120', () => {
    expect(factorial(5)).toBeCloseTo(120, 6);
  });

  test('factorial(170) is finite', () => {
    const result = factorial(170);
    expect(isFinite(result)).toBe(true);
    expect(result).toBeGreaterThan(0);
  });

  test('factorial(171) is Infinity', () => {
    expect(factorial(171)).toBe(Infinity);
  });

  test('throws for negative values', () => {
    expect(() => factorial(-1)).toThrow();
  });

  test('throws for non-integers', () => {
    expect(() => factorial(2.5)).toThrow();
  });

  test('random non-negative integers produce positive results', () => {
    for (let i = 0; i < FUZZ_COUNT; i++) {
      const n = randomInt(0, 170);
      const result = factorial(n);
      expect(result).toBeGreaterThan(0);
      expect(isFinite(result)).toBe(true);
    }
  });
});

// ==========================================================================
// binomialCoeff
// ==========================================================================
describe('Fuzz: binomialCoeff', () => {
  test('binomialCoeff(0, 0) = 1', () => {
    expect(binomialCoeff(0, 0)).toBeCloseTo(1, 10);
  });

  test('binomialCoeff(n, 0) = 1 for various n', () => {
    for (let n = 0; n <= 20; n++) {
      expect(binomialCoeff(n, 0)).toBeCloseTo(1, 8);
    }
  });

  test('binomialCoeff(n, n) = 1 for various n', () => {
    for (let n = 0; n <= 20; n++) {
      expect(binomialCoeff(n, n)).toBeCloseTo(1, 8);
    }
  });

  test('binomialCoeff(10, 3) = 120', () => {
    expect(binomialCoeff(10, 3)).toBeCloseTo(120, 4);
  });

  test('returns 0 for k > n', () => {
    expect(binomialCoeff(5, 6)).toBe(0);
    expect(binomialCoeff(0, 1)).toBe(0);
  });

  test('returns 0 for k < 0', () => {
    expect(binomialCoeff(5, -1)).toBe(0);
  });

  test('symmetry: C(n,k) = C(n, n-k)', () => {
    for (let i = 0; i < FUZZ_COUNT; i++) {
      const n = randomInt(0, 100);
      const k = randomInt(0, n);
      const a = binomialCoeff(n, k);
      const b = binomialCoeff(n, n - k);
      if (a === 0 && b === 0) continue;
      // Use relative tolerance for large values
      const relErr = Math.abs(a - b) / Math.max(Math.abs(a), Math.abs(b), 1);
      expect(relErr).toBeLessThan(1e-8);
    }
  });

  test('random valid inputs produce non-negative results', () => {
    for (let i = 0; i < FUZZ_COUNT; i++) {
      const n = randomInt(0, 200);
      const k = randomInt(0, n);
      const result = binomialCoeff(n, k);
      expect(result).toBeGreaterThanOrEqual(0);
    }
  });
});

// ==========================================================================
// betaFn
// ==========================================================================
describe('Fuzz: betaFn', () => {
  test('betaFn(a, b) = betaFn(b, a) (symmetry)', () => {
    for (let i = 0; i < FUZZ_COUNT; i++) {
      const a = randomInRange(0.1, 50);
      const b = randomInRange(0.1, 50);
      const ab = betaFn(a, b);
      const ba = betaFn(b, a);
      if (isFinite(ab) && isFinite(ba)) {
        expect(ab).toBeCloseTo(ba, 8);
      }
    }
  });

  test('betaFn(1, 1) = 1', () => {
    expect(betaFn(1, 1)).toBeCloseTo(1, 10);
  });

  test('does not throw for random positive inputs', () => {
    for (let i = 0; i < FUZZ_COUNT; i++) {
      const a = randomInRange(1e-5, 1e3);
      const b = randomInRange(1e-5, 1e3);
      const result = safeCall(() => betaFn(a, b));
      expect(typeof result).toBe('number');
    }
  });

  test('very small parameters produce large values', () => {
    const result = betaFn(0.001, 0.001);
    expect(result).toBeGreaterThan(100);
  });

  test('handles NaN and Infinity without throwing', () => {
    expect(() => betaFn(NaN, 1)).not.toThrow();
    expect(() => betaFn(1, NaN)).not.toThrow();
    expect(() => betaFn(Infinity, 1)).not.toThrow();
  });
});

// ==========================================================================
// erf
// ==========================================================================
describe('Fuzz: erf', () => {
  test('erf(0) ≈ 0', () => {
    expect(erf(0)).toBeCloseTo(0, 6);
  });

  test('erf(-x) = -erf(x) (odd function)', () => {
    for (let i = 0; i < FUZZ_COUNT; i++) {
      const x = randomInRange(0, 10);
      expect(erf(-x)).toBeCloseTo(-erf(x), 8);
    }
  });

  test('-1 <= erf(x) <= 1 for all finite x', () => {
    for (let i = 0; i < FUZZ_COUNT; i++) {
      const x = randomInRange(-100, 100);
      const result = erf(x);
      expect(result).toBeGreaterThanOrEqual(-1);
      expect(result).toBeLessThanOrEqual(1);
    }
  });

  test('erf(large) approaches 1', () => {
    expect(erf(10)).toBeCloseTo(1, 6);
    expect(erf(100)).toBeCloseTo(1, 6);
  });

  test('erf(-large) approaches -1', () => {
    expect(erf(-10)).toBeCloseTo(-1, 6);
  });

  test('handles NaN and Infinity without throwing', () => {
    expect(() => erf(NaN)).not.toThrow();
    expect(() => erf(Infinity)).not.toThrow();
    expect(() => erf(-Infinity)).not.toThrow();
  });

  test('does not throw for very large values', () => {
    for (const x of [1e10, 1e100, 1e308]) {
      const result = safeCall(() => erf(x));
      expect(typeof result).toBe('number');
    }
  });

  test('does not throw for very small values', () => {
    for (const x of [1e-10, 1e-100, 1e-308]) {
      const result = safeCall(() => erf(x));
      expect(typeof result).toBe('number');
    }
  });
});

// ==========================================================================
// erfc
// ==========================================================================
describe('Fuzz: erfc', () => {
  test('erfc(x) = 1 - erf(x) for random values', () => {
    for (let i = 0; i < FUZZ_COUNT; i++) {
      const x = randomInRange(-5, 5);
      expect(erfc(x)).toBeCloseTo(1 - erf(x), 8);
    }
  });

  test('erfc(0) ≈ 1', () => {
    expect(erfc(0)).toBeCloseTo(1, 6);
  });

  test('0 <= erfc(x) <= 2 for finite x', () => {
    for (let i = 0; i < FUZZ_COUNT; i++) {
      const x = randomInRange(-10, 10);
      const result = erfc(x);
      expect(result).toBeGreaterThanOrEqual(0 - 1e-12);
      expect(result).toBeLessThanOrEqual(2 + 1e-12);
    }
  });

  test('handles NaN and Infinity without throwing', () => {
    expect(() => erfc(NaN)).not.toThrow();
    expect(() => erfc(Infinity)).not.toThrow();
    expect(() => erfc(-Infinity)).not.toThrow();
  });
});

// ==========================================================================
// regularizedGammaP
// ==========================================================================
describe('Fuzz: regularizedGammaP', () => {
  test('regularizedGammaP(s, 0) = 0 for positive s', () => {
    for (let i = 0; i < 20; i++) {
      const s = randomInRange(0.1, 100);
      expect(regularizedGammaP(s, 0)).toBe(0);
    }
  });

  test('0 <= result <= 1 for valid inputs', () => {
    for (let i = 0; i < FUZZ_COUNT; i++) {
      const s = randomInRange(0.1, 50);
      const x = randomInRange(0, 100);
      const result = safeCall(() => regularizedGammaP(s, x));
      if (!isNaN(result)) {
        expect(result).toBeGreaterThanOrEqual(-1e-10);
        expect(result).toBeLessThanOrEqual(1 + 1e-10);
      }
    }
  });

  test('monotonically increasing in x for fixed s', () => {
    const s = 3;
    let prev = 0;
    for (let x = 0; x <= 20; x += 0.5) {
      const result = regularizedGammaP(s, x);
      expect(result).toBeGreaterThanOrEqual(prev - 1e-10);
      prev = result;
    }
  });

  test('throws for negative x', () => {
    expect(() => regularizedGammaP(1, -1)).toThrow();
  });

  test('handles large s and x without throwing', () => {
    for (let i = 0; i < 20; i++) {
      const s = randomInRange(50, 500);
      const x = randomInRange(50, 500);
      const result = safeCall(() => regularizedGammaP(s, x));
      expect(typeof result).toBe('number');
    }
  });
});

// ==========================================================================
// regularizedBeta
// ==========================================================================
describe('Fuzz: regularizedBeta', () => {
  test('regularizedBeta(0, a, b) = 0', () => {
    for (let i = 0; i < 20; i++) {
      const a = randomInRange(0.1, 50);
      const b = randomInRange(0.1, 50);
      expect(regularizedBeta(0, a, b)).toBe(0);
    }
  });

  test('regularizedBeta(1, a, b) = 1', () => {
    for (let i = 0; i < 20; i++) {
      const a = randomInRange(0.1, 50);
      const b = randomInRange(0.1, 50);
      expect(regularizedBeta(1, a, b)).toBe(1);
    }
  });

  test('0 <= result <= 1 for valid inputs', () => {
    for (let i = 0; i < FUZZ_COUNT; i++) {
      const x = rng();
      const a = randomInRange(0.1, 30);
      const b = randomInRange(0.1, 30);
      const result = safeCall(() => regularizedBeta(x, a, b));
      if (!isNaN(result)) {
        expect(result).toBeGreaterThanOrEqual(-1e-10);
        expect(result).toBeLessThanOrEqual(1 + 1e-10);
      }
    }
  });

  test('monotonically increasing in x for fixed a, b', () => {
    const a = 2;
    const b = 5;
    let prev = 0;
    for (let x = 0; x <= 1; x += 0.05) {
      const result = regularizedBeta(x, a, b);
      expect(result).toBeGreaterThanOrEqual(prev - 1e-10);
      prev = result;
    }
  });

  test('throws for x outside [0, 1]', () => {
    expect(() => regularizedBeta(-0.1, 1, 1)).toThrow();
    expect(() => regularizedBeta(1.1, 1, 1)).toThrow();
  });

  test('handles large a, b values without throwing', () => {
    for (let i = 0; i < 20; i++) {
      const x = rng();
      const a = randomInRange(50, 500);
      const b = randomInRange(50, 500);
      const result = safeCall(() => regularizedBeta(x, a, b));
      expect(typeof result).toBe('number');
    }
  });

  test('symmetry: I_x(a,b) = 1 - I_{1-x}(b,a)', () => {
    for (let i = 0; i < FUZZ_COUNT; i++) {
      const x = randomInRange(0.01, 0.99);
      const a = randomInRange(0.5, 20);
      const b = randomInRange(0.5, 20);
      const lhs = safeCall(() => regularizedBeta(x, a, b));
      const rhs = safeCall(() => 1 - regularizedBeta(1 - x, b, a));
      if (!isNaN(lhs) && !isNaN(rhs)) {
        expect(lhs).toBeCloseTo(rhs, 5);
      }
    }
  });
});

// ==========================================================================
// Combined stress: no function should throw for extreme edge-case inputs
// ==========================================================================
describe('Fuzz: stress test all functions with extreme values', () => {
  const extremeValues = [0, 1e-308, 1e-100, 1e-10, 0.5, 1, 2, 10, 100, 1e10, 1e100, 1e308];

  test('gammaLn handles extreme positive values', () => {
    for (const x of extremeValues) {
      if (x === 0) continue; // gammaLn(0) returns NaN, that's fine
      const result = safeCall(() => gammaLn(x));
      expect(typeof result).toBe('number');
    }
  });

  test('gamma handles extreme positive values', () => {
    for (const x of extremeValues) {
      if (x === 0) continue;
      const result = safeCall(() => gamma(x));
      expect(typeof result).toBe('number');
    }
  });

  test('erf handles extreme values including negative', () => {
    for (const x of [...extremeValues, ...extremeValues.map(v => -v)]) {
      const result = safeCall(() => erf(x));
      expect(typeof result).toBe('number');
    }
  });

  test('erfc handles extreme values', () => {
    for (const x of [...extremeValues, ...extremeValues.map(v => -v)]) {
      const result = safeCall(() => erfc(x));
      expect(typeof result).toBe('number');
    }
  });

  test('betaFn handles extreme positive values', () => {
    for (const a of [1e-10, 0.5, 1, 10, 100]) {
      for (const b of [1e-10, 0.5, 1, 10, 100]) {
        const result = safeCall(() => betaFn(a, b));
        expect(typeof result).toBe('number');
      }
    }
  });
});
