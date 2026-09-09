# Installation

## Standard Installation

Install statistika using your preferred package manager:

::: code-group

```bash [yarn]
yarn add @diogoribeiro7/statistika
```

```bash [npm]
npm install @diogoribeiro7/statistika
```

```bash [pnpm]
pnpm add @diogoribeiro7/statistika
```

:::

This installs the pure TypeScript version. All functionality works out of the box without any native dependencies.

## Native Fortran Acceleration

For best performance on special math functions (gamma, beta, erf, regularized incomplete functions), you can enable native Fortran acceleration via N-API.

### Requirements

- `gfortran` (GNU Fortran compiler)
- C/C++ build tools (`make`, `gcc`)
- `node-gyp` (included as an optional dependency)

### Setup

On Ubuntu/Debian:

```bash
sudo apt install gfortran build-essential
```

On macOS (with Homebrew):

```bash
brew install gcc
```

On Fedora/RHEL:

```bash
sudo dnf install gcc-gfortran gcc-c++ make
```

### Building Native Extensions

After installing the system dependencies, build the native extensions:

```bash
yarn build
```

This runs both the Fortran compilation and TypeScript compilation. If `gfortran` is not found or the native toolchain is not available, the native build step is skipped and the library falls back to pure TypeScript implementations.

On Windows, `yarn build` will try the native Fortran build when `gfortran` is installed on `PATH`. This usually requires a compatible Windows native addon toolchain such as MSYS2/MinGW or Visual Studio build tools, plus LAPACK available to the compiler. If LAPACK is unavailable, the build still completes with special functions only and TypeScript fallbacks for linear algebra.

### TypeScript-only Build

If you only want to compile TypeScript (skipping native acceleration):

```bash
yarn build:ts
```

### Verifying Native Acceleration

You can check whether native acceleration is active:

```typescript
import { gamma } from 'statistika';

// The function works identically regardless of backend.
// Native acceleration is transparent — same API, faster execution.
console.log(gamma(5)); // 24
```

## Module Systems

statistika ships dual CJS/ESM builds:

- **ESM**: `dist/esm/index.js` (for `import` syntax)
- **CJS**: `dist/cjs/index.js` (for `require` syntax)

The `exports` field in `package.json` ensures the correct format is resolved automatically by Node.js and bundlers.

## Supported Node.js Versions

statistika requires Node.js 18 or later. It is tested against:

- Node.js 18 (LTS)
- Node.js 20 (LTS)
- Node.js 22 (Current)
