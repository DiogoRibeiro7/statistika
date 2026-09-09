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
      ? join(__dirname, "..", "..", "..")
      : join(__dirname, "..", "..");
  } else {
    projectRoot = process.cwd();
  }

  const addonPath = join(projectRoot, "build", "Release", "fortran_special.node");
  const require_ = createRequire(join(projectRoot, "package.json"));
  addon = require_(addonPath);

  // A no-LAPACK build can still produce a loadable .node binary because the
  // linalg entry points are backed by stubs. Do not expose that partial addon
  // as a fully capable native backend: downstream modules otherwise select
  // native sampling/statistics paths while LAPACK-dependent symbols remain
  // unavailable. Probe a 1x1 solve; the no-LAPACK stub reports info=-999.
  if (typeof addon?.solve !== "function") {
    addon = null;
  } else {
    const probe = addon.solve([[1]], [1]);
    const x0 = Array.isArray(probe?.x) ? probe.x[0] : NaN;
    if (probe?.info !== 0 || !Number.isFinite(x0) || Math.abs(x0 - 1) > 1e-12) {
      addon = null;
    }
  }
} catch {
  // Native addon not available or not fully functional — pure TypeScript
  // fallbacks will be used.
  addon = null;
}

/** The loaded native addon, or null if unavailable. */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const nativeAddon: Record<string, (...args: any[]) => any> | null = addon;
