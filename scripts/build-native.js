#!/usr/bin/env node

/**
 * Attempts to build the native Fortran/C++ addon.
 * Gracefully skips if gfortran or node-gyp are not available.
 * The library falls back to pure TypeScript implementations at runtime.
 */

const { execSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");

function run(cmd, label) {
  try {
    execSync(cmd, { cwd: ROOT, stdio: "pipe" });
    return true;
  } catch {
    console.warn(`⚠  Skipping native build: ${label} failed`);
    return false;
  }
}

function main() {
  // Check if gfortran is available
  try {
    execSync("gfortran --version", { stdio: "pipe" });
  } catch {
    console.warn(
      "⚠  gfortran not found — skipping native addon build.\n" +
        "   The library will use pure TypeScript fallbacks (no performance impact for most use cases).",
    );
    return;
  }

  // Check if source files exist
  const fortranSrc = path.join(ROOT, "native/fortran/special_functions.f90");
  if (!fs.existsSync(fortranSrc)) {
    console.warn("⚠  Fortran source not found — skipping native addon build.");
    return;
  }

  // Compile Fortran object
  const objPath = path.join(ROOT, "native/fortran/special_functions.o");
  if (
    !run(
      `gfortran -c -fPIC -O2 -o ${objPath} ${fortranSrc}`,
      "Fortran compilation",
    )
  ) {
    return;
  }

  // Run node-gyp
  if (!run("npx node-gyp configure build", "node-gyp build")) {
    return;
  }

  console.log("✓  Native Fortran addon built successfully.");
}

main();
