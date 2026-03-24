# WASM Acceleration Module

WebAssembly-compiled special functions and linear algebra operations for
near-native performance in environments where the Fortran native addon is
not available (browsers, serverless, etc.).

## Fallback Chain

The library uses a three-tier acceleration strategy:

1. **Native Fortran/LAPACK addon** (fastest, requires compilation at install)
2. **WASM module** (near-native, portable, no install-time compilation)
3. **Pure TypeScript** (always available, slowest)

The appropriate backend is selected automatically. Use `loadWasm()` to
attempt WASM initialization, and `getAccelerated()` to get whichever
implementation is available.

## Building the WASM Module

### Prerequisites

Install the [Emscripten SDK](https://emscripten.org/docs/getting_started/):

```bash
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install latest
./emsdk activate latest
source ./emsdk_env.sh
```

### Build

From the project root:

```bash
yarn build:wasm
# or directly:
./scripts/build-wasm.sh
```

This compiles `special-functions.c` and `linalg.c` into `stats.wasm`,
placed in `src/wasm/`.

## Source Files

| File                   | Description                                        |
| ---------------------- | -------------------------------------------------- |
| `special-functions.c`  | gammaLn, gamma, erf, erfc, betaFn                  |
| `linalg.c`             | matMul, solve (LU with pivoting), cholesky          |
| `index.ts`             | WASM loader with lazy initialization and caching   |
| `fallback.ts`          | Pure TypeScript implementations of the same API    |

## Usage

```typescript
import { loadWasm, getWasm, getAccelerated, tsFallback } from './wasm';

// Option 1: Async load (preferred at startup)
const wasm = await loadWasm();
if (wasm) {
  console.log(wasm.gammaLn(5.0));
}

// Option 2: Sync access after loading
const mod = getWasm(); // null if not loaded
if (mod) {
  console.log(mod.erf(1.0));
}

// Option 3: Always get a working module (WASM or TS fallback)
const accel = getAccelerated();
console.log(accel.gammaLn(5.0)); // works regardless of WASM availability

// Option 4: Direct fallback (no WASM attempt)
console.log(tsFallback.gammaLn(5.0));
```

## Matrix Format

All matrix operations use **row-major flat `Float64Array`** storage:

```typescript
// 2x2 matrix [[1, 2], [3, 4]]
const mat = new Float64Array([1, 2, 3, 4]);

// Multiply: C = A * B
const C = accel.matMul(A, B, rows_A, cols_A, cols_B);
```

## Exported WASM Functions

| C Function   | WASM Export  | Description                          |
| ------------ | ------------ | ------------------------------------ |
| `gammaLn`    | `_gammaLn`   | Log-gamma (Lanczos approximation)    |
| `gamma_fn`   | `_gamma_fn`  | Gamma function                       |
| `erf_fn`     | `_erf_fn`    | Error function                       |
| `erfc_fn`    | `_erfc_fn`   | Complementary error function         |
| `betaFn`     | `_betaFn`    | Beta function                        |
| `matMul`     | `_matMul`    | Matrix multiplication                |
| `solve`      | `_solve`     | Linear solve via LU decomposition    |
| `cholesky`   | `_cholesky`  | Cholesky factorization               |
