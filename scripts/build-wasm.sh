#!/usr/bin/env bash
# --------------------------------------------------------------------------
# Build the WASM module from C source files using Emscripten (emcc).
#
# Prerequisites:
#   - Install Emscripten SDK: https://emscripten.org/docs/getting_started/
#   - Activate: source <emsdk>/emsdk_env.sh
#
# Usage:
#   ./scripts/build-wasm.sh
#
# Output:
#   src/wasm/stats.wasm   — The compiled WebAssembly binary
#   src/wasm/stats.js     — JS glue (optional, for browser use)
# --------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WASM_SRC="$PROJECT_ROOT/src/wasm"
OUTPUT_DIR="$WASM_SRC"

# Check that emcc is available
if ! command -v emcc &> /dev/null; then
  echo "Error: emcc (Emscripten compiler) not found."
  echo "Install Emscripten SDK: https://emscripten.org/docs/getting_started/"
  echo "Then activate it: source <emsdk>/emsdk_env.sh"
  exit 1
fi

echo "Building WASM module..."
echo "  Source: $WASM_SRC"
echo "  Output: $OUTPUT_DIR"

emcc \
  "$WASM_SRC/special-functions.c" \
  "$WASM_SRC/linalg.c" \
  -O3 \
  -s WASM=1 \
  -s EXPORTED_FUNCTIONS='["_gammaLn","_gamma_fn","_erf_fn","_erfc_fn","_betaFn","_matMul","_solve","_cholesky","_malloc","_free"]' \
  -s EXPORTED_RUNTIME_METHODS='["cwrap","ccall"]' \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s MODULARIZE=1 \
  -s EXPORT_NAME="StatsWasm" \
  -s ENVIRONMENT='node' \
  -s STANDALONE_WASM=1 \
  --no-entry \
  -lm \
  -o "$OUTPUT_DIR/stats.wasm"

echo "WASM build complete."
echo "  Binary: $OUTPUT_DIR/stats.wasm"
ls -lh "$OUTPUT_DIR/stats.wasm"
