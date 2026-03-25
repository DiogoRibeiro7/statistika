# Performance Baselines

Benchmark results for the pure-TypeScript special function implementations.
Run with `npm run bench` (100,000 iterations per function).

## Environment

- **Runtime**: Node.js v22
- **Platform**: Linux x86_64
- **Mode**: Pure TypeScript (no native Fortran addon)

## Special Functions (100k iterations)

| Function                        | Time (ms) | Ops/sec      |
| ------------------------------- | --------: | -----------: |
| `gammaLn(5.5)`                  |     22.40 |    4,464,286 |
| `gammaLn(0.1)`                  |     10.52 |    9,505,703 |
| `gamma(5.5)`                    |      8.60 |   11,627,907 |
| `betaFn(2, 5)`                  |     21.12 |    4,734,848 |
| `betaFn(0.5, 0.5)`             |     18.51 |    5,402,485 |
| `erf(1.0)`                     |      1.65 |   60,606,061 |
| `erf(2.5)`                     |      1.76 |   56,818,182 |
| `erfc(1.0)`                    |      1.42 |   70,422,535 |
| `regularizedGammaP(2, 3)`      |      9.73 |   10,277,492 |
| `regularizedGammaP(10, 5)`     |     21.65 |    4,618,938 |
| `regularizedBeta(0.5, 2, 5)`   |     30.71 |    3,256,331 |
| `regularizedBeta(0.3, 0.5, 0.5)` |   33.69 |    2,968,238 |
| `logFactorial(100)`            |      5.09 |   19,646,365 |
| `binomialCoeff(20, 10)`        |     18.52 |    5,399,568 |

## CI Regression Detection Baselines

These values are used by `scripts/compare-benchmarks.js` to detect regressions >10%.
Update by running `BENCH_JSON=1 yarn bench:all` and copying values here.

| Suite | Benchmark | tsTimeMs |
|-------|-----------|----------|
| special-functions | gammaLn(5.5) | 22.40 |
| special-functions | gammaLn(0.1) | 10.52 |
| special-functions | gamma(5.5) | 8.60 |
| special-functions | betaFn(2, 5) | 21.12 |
| special-functions | betaFn(0.5, 0.5) | 18.51 |
| special-functions | erf(1.0) | 1.65 |
| special-functions | erf(2.5) | 1.76 |
| special-functions | erfc(1.0) | 1.42 |
| special-functions | regularizedGammaP(2, 3) | 9.73 |
| special-functions | regularizedGammaP(10, 5) | 21.65 |
| special-functions | regularizedBeta(0.5, 2, 5) | 30.71 |
| special-functions | regularizedBeta(0.3, 0.5, 0.5) | 33.69 |
| special-functions | logFactorial(100) | 5.09 |
| special-functions | binomialCoeff(20, 10) | 18.52 |

## Notes

- The `erf`/`erfc` functions are the fastest since they use a simple polynomial
  approximation (Abramowitz & Stegun 7.1.26).
- `regularizedBeta` is the slowest as it requires iterative continued-fraction
  evaluation internally.
- When the native Fortran addon is compiled (`npm run build`), these functions
  dispatch to Fortran implementations automatically. Run `npm run bench` to
  compare.
- These numbers serve as regression baselines. If a refactor causes a function
  to be >2x slower, investigate.
