# Performance and acceleration

statistika is designed to remain usable without a compiler or native runtime. Acceleration is therefore an implementation detail of individual operations, not a requirement for using the library.

## Backend order

Where an operation supports all three implementations, the preferred order is:

1. native Fortran/LAPACK through the N-API addon
2. WebAssembly
3. pure TypeScript

Not every operation implements every backend. The library falls back automatically when a faster backend is unavailable.

The native addon is treated as available only after a LAPACK-backed linear solve succeeds during initialization. A partial native build without working LAPACK is rejected so that downstream modules do not accidentally select incomplete native routines.

## Current dispatch paths

The table below describes the current implementation. It is intentionally specific because a single package-wide "native enabled" flag would be misleading.

| Area | Operations | Dispatch |
| --- | --- | --- |
| Linear algebra | `solveLinearSystem`, `matMul` | native → WASM → TypeScript |
| Linear algebra | `invertMatrix`, `normalCdf`, `symmetricEigen` | native → TypeScript |
| Linear algebra | `transpose`, `normalQuantile` | TypeScript |
| Special functions | `betaFn`, `erf`, `erfc` | native → WASM → TypeScript |
| Special functions | `binomialCoeff` | native → TypeScript |
| Special functions | `gammaLn`, `gamma`, `logFactorial`, `factorial`, `regularizedGammaP`, `regularizedBeta` | TypeScript |
| Statistics kernels | pairwise distances, KDE, weighted cross-products, Welford batches | native → TypeScript |
| Time series, distributions, Kalman, GARCH | routines exposed by their native bridge | native → TypeScript |

Some modules expose more granular native flags. Prefer those flags when checking whether a particular subsystem is accelerated.

## Inspect the active backend

The main package exports the backend flags used by the public modules:

```typescript
import {
  hasNativeDistributions,
  hasNativeGarch,
  hasNativeKalman,
  hasNativeLinalg,
  hasNativeStats,
  hasNativeTimeSeries,
  isNativeAvailable,
  isWasmAvailable,
} from "@diogoribeiro7/statistika";

const acceleration = {
  specialFunctionsNative: isNativeAvailable(),
  linalgNative: hasNativeLinalg,
  statsNative: hasNativeStats,
  timeSeriesNative: hasNativeTimeSeries,
  distributionsNative: hasNativeDistributions,
  kalmanNative: hasNativeKalman,
  garchNative: hasNativeGarch,
  wasm: isWasmAvailable(),
};

console.table(acceleration);
```

`isNativeAvailable()` reports whether the validated native addon is visible to the special-function module. It does not mean that every special function dispatches to native code. Use the dispatch table above when the backend of a specific operation matters.

## WASM initialization

WASM is optional and is not required for normal use. It must be loaded explicitly:

```typescript
import { loadWasm, isWasmAvailable } from "@diogoribeiro7/statistika";

const wasm = await loadWasm();

console.log({
  loaded: wasm !== null,
  active: isWasmAvailable(),
});
```

The current loader reads the compiled `stats.wasm` binary from the Node.js filesystem. Browser-compatible loading is not yet part of the supported runtime path.

Most call-time dispatchers observe WASM availability after initialization. A small number of scalar hot-path wrappers choose their implementation when their module is initialized. If you specifically need those wrappers to bind to WASM, initialize the WASM subpath before dynamically importing the main API.

## Native build

A normal TypeScript build does not require Fortran:

```bash
yarn build:ts
```

To build the optional native addon:

```bash
yarn build
```

The native path requires `gfortran`, a compatible native build toolchain, and LAPACK. If the addon cannot be built or fails its runtime capability probe, the TypeScript implementation remains available.

## When acceleration helps

Backend overhead matters. Native or WASM execution is most useful for work such as larger matrix operations, repeated linear solves, pairwise distance matrices, KDE evaluation, and batch statistical kernels. Very small matrices and cheap scalar functions may see little benefit because data conversion and boundary-crossing costs can dominate the computation.

For that reason, avoid treating backend availability as a performance guarantee. Measure the actual workload that matters to your application.

## Benchmark locally

The repository includes focused benchmark suites and a combined runner:

```bash
yarn bench
yarn bench:linalg
yarn bench:regression
yarn bench:mcmc
yarn bench:all
```

The regression checks use the baselines stored in `benchmark/BASELINES.md`. Benchmark results depend on CPU, compiler, BLAS/LAPACK implementation, Node.js version, workload size, and whether the native addon was successfully loaded, so repository baselines should be used as regression references rather than universal speedup claims.

For a meaningful comparison on your machine:

1. record the backend flags before benchmarking
2. run the same Node.js version and benchmark inputs
3. compare native/WASM and TypeScript paths on the same hardware
4. report both runtime and problem size
