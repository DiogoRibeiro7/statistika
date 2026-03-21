/**
 * Shared native addon loader.
 *
 * Loads the .node binary once, shared by math.ts and linalg.ts.
 * Works in both CJS and ESM contexts, and during development (ts-node/jest).
 */

import { createRequire } from "node:module";
import { join } from "node:path";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let addon: any = null;
try {
  // Determine the project root from __dirname.
  //   - From source (ts-node/jest): __dirname = <root>/src/utils
  //   - From CJS build:             __dirname = <root>/dist/cjs/utils
  //   - From ESM build:             __dirname is undefined
  let projectRoot: string;
  if (typeof __dirname !== "undefined") {
    // Walk up from the utils directory. Works whether we're in src/ or dist/cjs/.
    // We detect which by checking if __dirname contains "dist".
    projectRoot = __dirname.includes("dist")
      ? join(__dirname, "..", "..", "..")   // dist/cjs/utils -> root
      : join(__dirname, "..", "..");        // src/utils -> root
  } else {
    projectRoot = process.cwd();
  }

  const addonPath = join(projectRoot, "build", "Release", "fortran_special.node");
  const require_ = createRequire(join(projectRoot, "package.json"));
  addon = require_(addonPath);
} catch {
  // Native addon not available — pure TypeScript fallbacks will be used.
}

/** The loaded native addon, or null if unavailable. */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const nativeAddon: Record<string, (...args: any[]) => any> | null = addon;
