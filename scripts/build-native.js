#!/usr/bin/env node

/**
 * Attempts to build the native Fortran/C++ addon.
 * Gracefully skips if gfortran, LAPACK, or node-gyp are not available.
 * The library falls back to pure TypeScript implementations at runtime.
 */

const { execFileSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const PLATFORM = process.platform;
const IS_WINDOWS = PLATFORM === "win32";
const FORTRAN_COMPILE_FLAGS = IS_WINDOWS
  ? ["-c", "-O2", "-Wno-error=line-truncation"]
  : ["-c", "-fPIC", "-O2"];

function run(executable, args, label) {
  try {
    execFileSync(executable, args, { cwd: ROOT, stdio: "pipe", shell: false });
    return true;
  } catch (error) {
    console.warn(`⚠  Skipping native build: ${label} failed`);
    console.warn(`    Command: ${executable} ${args.join(" ")}`);
    if (error.stdout) {
      console.warn(`    stdout: ${error.stdout.toString().trim()}`);
    }
    if (error.stderr) {
      console.warn(`    stderr: ${error.stderr.toString().trim()}`);
    }
    return false;
  }
}

function findGfortranLibrary(name) {
  try {
    const resolved = execFileSync("gfortran", [`-print-file-name=${name}`], {
      stdio: "pipe",
      shell: false,
    })
      .toString()
      .trim();
    if (resolved && resolved !== name && fs.existsSync(resolved)) {
      return resolved;
    }
  } catch {
    // ignore
  }
  return null;
}

function isNodeAddonApiInstalled() {
  try {
    require.resolve("node-addon-api");
    return true;
  } catch {
    return false;
  }
}

// Walk PATH for the directory that holds the LAPACK runtime DLLs.
// We require BOTH liblapack.dll and libblas.dll to live there, so that
// dependent runtime DLLs (libgfortran/libgcc/libquadmath/libwinpthread)
// copied from the same directory match the libgfortran ABI lapack was
// linked against — avoiding mismatched libgfortran-5.dll loads at runtime.
function findLapackRuntimeBinDir() {
  const sep = IS_WINDOWS ? ";" : ":";
  const pathDirs = (process.env.PATH || "").split(sep).filter(Boolean);
  // Common fallbacks if PATH happens not to include the toolchain bin.
  const fallbacks = IS_WINDOWS
    ? [
        "C:\\rtools45\\ucrt64\\bin",
        "C:\\msys64\\ucrt64\\bin",
        "C:\\msys64\\mingw64\\bin",
      ]
    : [];
  for (const dir of [...pathDirs, ...fallbacks]) {
    if (
      fs.existsSync(path.join(dir, "liblapack.dll")) &&
      fs.existsSync(path.join(dir, "libblas.dll"))
    ) {
      return dir;
    }
  }
  return null;
}

// Windows-only: link all Fortran .o files into a single bridge DLL using
// gfortran. This isolates the GCC/MinGW Fortran world from MSVC's link.exe
// (which can't consume -lgfortran or MinGW import libs reliably). The
// addon.cpp, still compiled by MSVC via node-gyp, then links against the
// bridge's MinGW import lib (libfortran_native.dll.a) — for plain C symbols
// this works because both toolchains emit COFF on x64.
function linkFortranBridgeDll(objFiles, hasLapack, outDir) {
  fs.mkdirSync(outDir, { recursive: true });
  const dll = path.join(outDir, "fortran_native.dll");
  const implib = path.join(outDir, "libfortran_native.dll.a");
  const args = [
    "-shared",
    "-o",
    dll,
    `-Wl,--out-implib=${implib}`,
    "-Wl,--export-all-symbols",
    ...objFiles,
  ];
  if (hasLapack) args.push("-llapack", "-lblas");
  return run("gfortran", args, "Fortran bridge DLL link");
}

function copyLapackRuntimeDlls(sourceDir, destDir) {
  const required = ["liblapack.dll", "libblas.dll"];
  // Optional but expected — copy whatever subset is present, which is
  // enough for typical UCRT64 lapack builds.
  const optional = [
    "libgfortran-5.dll",
    "libgcc_s_seh-1.dll",
    "libquadmath-0.dll",
    "libwinpthread-1.dll",
  ];
  fs.mkdirSync(destDir, { recursive: true });
  for (const name of required) {
    const src = path.join(sourceDir, name);
    if (!fs.existsSync(src)) {
      console.warn(`⚠  Missing required runtime DLL: ${src}`);
      return false;
    }
    fs.copyFileSync(src, path.join(destDir, name));
  }
  for (const name of optional) {
    const src = path.join(sourceDir, name);
    if (fs.existsSync(src)) {
      fs.copyFileSync(src, path.join(destDir, name));
    }
  }
  return true;
}

function hasSystemLapack() {
  const knownPaths = [
    "/usr/lib/liblapack.so",
    "/usr/lib/x86_64-linux-gnu/liblapack.so",
  ];
  if (knownPaths.some((candidate) => fs.existsSync(candidate))) return true;

  try {
    const libraries = execFileSync("ldconfig", ["-p"], {
      stdio: "pipe",
      shell: false,
    }).toString();
    return libraries.includes("liblapack");
  } catch {
    return false;
  }
}

function compileFortran(source, output, label, moduleDir = null) {
  const args = [...FORTRAN_COMPILE_FLAGS];
  if (moduleDir) args.push(`-J${moduleDir}`);
  args.push("-o", output, source);
  return run("gfortran", args, label);
}

function main() {
  // Check if gfortran is available
  try {
    execFileSync("gfortran", ["--version"], { stdio: "pipe", shell: false });
  } catch {
    console.warn(
      "⚠  gfortran not found — skipping native addon build.\n" +
        "   The library will use pure TypeScript fallbacks (no performance impact for most use cases).",
    );
    return;
  }

  // Detect whether LAPACK is available
  let hasLapack = false;
  if (IS_WINDOWS) {
    if (
      findGfortranLibrary("liblapack.a") ||
      findGfortranLibrary("liblapack.dll") ||
      findGfortranLibrary("liblapack.dll.a") ||
      findGfortranLibrary("liblapack.lib")
    ) {
      hasLapack = true;
    } else {
      console.warn(
        "⚠  LAPACK not detected on Windows — linear algebra will use pure TypeScript fallbacks if needed.\n" +
          "   If you have LAPACK installed, ensure gfortran can resolve liblapack (for example via liblapack.dll.a or liblapack.a in your MinGW/MSYS2 toolchain).",
      );
    }
  } else if (hasSystemLapack()) {
    hasLapack = true;
  } else {
    console.warn(
      "⚠  LAPACK not found — linear algebra will use pure TypeScript fallbacks.\n" +
        "   Install liblapack-dev (Debian/Ubuntu) or lapack-devel (RHEL/Fedora) for native acceleration.",
    );
  }

  // Check if source files exist
  const moduleDir = path.join(ROOT, "native/fortran");
  const specialSrc = path.join(moduleDir, "special_functions.f90");
  const linalgSrc = path.join(moduleDir, "linalg.f90");
  if (!fs.existsSync(specialSrc)) {
    console.warn("⚠  Fortran source not found — skipping native addon build.");
    return;
  }

  // Compile special_functions.f90
  const specialObj = path.join(moduleDir, "special_functions.o");
  if (!compileFortran(specialSrc, specialObj, "Fortran special functions compilation", moduleDir)) {
    return;
  }

  // Compile distributions.f90 (depends on special_functions module)
  const distSrc = path.join(moduleDir, "distributions.f90");
  const distObj = path.join(moduleDir, "distributions.o");
  if (fs.existsSync(distSrc)) {
    if (!compileFortran(distSrc, distObj, "Fortran distributions compilation", moduleDir)) {
      try { fs.unlinkSync(distObj); } catch {}
    }
  }

  // Compile statistics.f90
  const statsSrc = path.join(moduleDir, "statistics.f90");
  const statsObj = path.join(moduleDir, "statistics.o");
  if (fs.existsSync(statsSrc)) {
    if (!compileFortran(statsSrc, statsObj, "Fortran statistics compilation")) {
      try { fs.unlinkSync(statsObj); } catch {}
    }
  }

  // Compile time_series.f90
  const tsSrc = path.join(moduleDir, "time_series.f90");
  const tsObj = path.join(moduleDir, "time_series.o");
  if (fs.existsSync(tsSrc)) {
    if (!compileFortran(tsSrc, tsObj, "Fortran time_series compilation")) {
      try { fs.unlinkSync(tsObj); } catch {}
    }
  }

  // Compile kalman.f90
  const kalmanSrc = path.join(moduleDir, "kalman.f90");
  const kalmanObj = path.join(moduleDir, "kalman.o");
  if (fs.existsSync(kalmanSrc)) {
    if (!compileFortran(kalmanSrc, kalmanObj, "Fortran kalman compilation")) {
      try { fs.unlinkSync(kalmanObj); } catch {}
    }
  }

  // Compile optimization.f90
  const optSrc = path.join(moduleDir, "optimization.f90");
  const optObj = path.join(moduleDir, "optimization.o");
  if (fs.existsSync(optSrc)) {
    if (!compileFortran(optSrc, optObj, "Fortran optimization compilation")) {
      try { fs.unlinkSync(optObj); } catch {}
    }
  }

  // Compile sampling.f90
  const samplingSrc = path.join(moduleDir, "sampling.f90");
  const samplingObj = path.join(moduleDir, "sampling.o");
  if (fs.existsSync(samplingSrc)) {
    if (!compileFortran(samplingSrc, samplingObj, "Fortran sampling compilation")) {
      try { fs.unlinkSync(samplingObj); } catch {}
    }
  }

  // Compile linalg.f90 (only if LAPACK is available and source exists)
  if (hasLapack && fs.existsSync(linalgSrc)) {
    const linalgObj = path.join(moduleDir, "linalg.o");
    if (!compileFortran(linalgSrc, linalgObj, "Fortran linalg compilation")) {
      try { fs.unlinkSync(linalgObj); } catch {}
      hasLapack = false;
    }
  } else {
    hasLapack = false;
  }

  // If no LAPACK, create a dummy linalg.o stub so binding.gyp doesn't fail.
  // Statistical routines are provided by statistics.o and must not be
  // duplicated here.
  if (!hasLapack) {
    const linalgObj = path.join(moduleDir, "linalg.o");
    const stubSrc = path.join(moduleDir, "linalg_stub.f90");
    fs.writeFileSync(stubSrc, `
subroutine fortran_mat_mul(a, b, c, m, k, n) bind(C, name="fortran_mat_mul")
  use iso_c_binding
  real(c_double), intent(in) :: a(*), b(*)
  real(c_double), intent(out) :: c(*)
  integer(c_int), intent(in) :: m, k, n
  c(1) = 0.0d0
end subroutine

subroutine fortran_solve(a, b, x, n, info) bind(C, name="fortran_solve")
  use iso_c_binding
  real(c_double), intent(in) :: a(*), b(*)
  real(c_double), intent(out) :: x(*)
  integer(c_int), intent(in) :: n
  integer(c_int), intent(out) :: info
  info = -999
end subroutine

subroutine fortran_invert(a, inv, n, info) bind(C, name="fortran_invert")
  use iso_c_binding
  real(c_double), intent(in) :: a(*)
  real(c_double), intent(out) :: inv(*)
  integer(c_int), intent(in) :: n
  integer(c_int), intent(out) :: info
  info = -999
end subroutine

subroutine fortran_sym_eigen(a, evals, evecs, n, info) bind(C, name="fortran_sym_eigen")
  use iso_c_binding
  real(c_double), intent(in) :: a(*)
  real(c_double), intent(out) :: evals(*), evecs(*)
  integer(c_int), intent(in) :: n
  integer(c_int), intent(out) :: info
  info = -999
end subroutine

subroutine fortran_normal_cdf(x, result) bind(C, name="fortran_normal_cdf")
  use iso_c_binding
  real(c_double), intent(in), value :: x
  real(c_double), intent(out) :: result
  result = 0.0d0
end subroutine

subroutine fortran_lu(a, lu_out, ipiv_out, n, info) bind(C, name="fortran_lu")
  use iso_c_binding
  real(c_double), intent(in) :: a(*)
  real(c_double), intent(out) :: lu_out(*)
  integer(c_int), intent(out) :: ipiv_out(*)
  integer(c_int), intent(in) :: n
  integer(c_int), intent(out) :: info
  info = -999
end subroutine

subroutine fortran_qr(a, q_out, r_out, m, n, info) bind(C, name="fortran_qr")
  use iso_c_binding
  real(c_double), intent(in) :: a(*)
  real(c_double), intent(out) :: q_out(*), r_out(*)
  integer(c_int), intent(in) :: m, n
  integer(c_int), intent(out) :: info
  info = -999
end subroutine

subroutine fortran_cholesky(a, l_out, n, info) bind(C, name="fortran_cholesky")
  use iso_c_binding
  real(c_double), intent(in) :: a(*)
  real(c_double), intent(out) :: l_out(*)
  integer(c_int), intent(in) :: n
  integer(c_int), intent(out) :: info
  info = -999
end subroutine

subroutine fortran_svd(a, u_out, s_out, vt_out, m, n, info) bind(C, name="fortran_svd")
  use iso_c_binding
  real(c_double), intent(in) :: a(*)
  real(c_double), intent(out) :: u_out(*), s_out(*), vt_out(*)
  integer(c_int), intent(in) :: m, n
  integer(c_int), intent(out) :: info
  info = -999
end subroutine
`);
    compileFortran(stubSrc, linalgObj, "Fortran linalg stub compilation");
    try { fs.unlinkSync(stubSrc); } catch {}
  }

  if (!isNodeAddonApiInstalled()) {
    console.warn(
      "⚠  node-addon-api is not installed — native addon build cannot continue.\n" +
        "   Run `npm install` or `yarn install` and retry.",
    );
    return;
  }

  // Run node-gyp with the LAPACK flag set for the generated build.
  if (!run("npx", ["node-gyp", "configure", "--", `-Duse_lapack=${hasLapack ? 1 : 0}`], "node-gyp configure")) {
    return;
  }

  // On Windows, pre-link all Fortran objects into a single bridge DLL so
  // the MSVC-driven node-gyp build only has to link addon.cpp against an
  // import lib. binding.gyp expects the import lib at build/Release/.
  if (IS_WINDOWS) {
    const bridgeObjs = [
      path.join(moduleDir, "special_functions.o"),
      path.join(moduleDir, "distributions.o"),
      path.join(moduleDir, "linalg.o"),
      path.join(moduleDir, "statistics.o"),
      path.join(moduleDir, "time_series.o"),
      path.join(moduleDir, "kalman.o"),
      path.join(moduleDir, "optimization.o"),
      path.join(moduleDir, "sampling.o"),
    ].filter((p) => fs.existsSync(p));
    const releaseDir = path.join(ROOT, "build", "Release");
    if (!linkFortranBridgeDll(bridgeObjs, hasLapack, releaseDir)) {
      return;
    }
  }

  if (!run("npx", ["node-gyp", "build"], "node-gyp build")) {
    return;
  }

  // On Windows with LAPACK, mirror the LAPACK toolchain's runtime DLLs
  // alongside the .node addon so Node's LOAD_WITH_ALTERED_SEARCH_PATH
  // resolves them from the addon's own directory before traversing PATH.
  // This isolates us from any older libgfortran-5.dll on PATH (e.g. a
  // Strawberry Perl gfortran) that may be missing symbols the LAPACK
  // build was linked against.
  if (IS_WINDOWS && hasLapack) {
    const sourceDir = findLapackRuntimeBinDir();
    const destDir = path.join(ROOT, "build", "Release");
    if (!sourceDir) {
      console.warn(
        "⚠  Could not locate liblapack.dll / libblas.dll on PATH — addon may fail to load.\n" +
          "   Add the LAPACK toolchain bin directory (e.g. C:\\rtools45\\ucrt64\\bin) to PATH or copy the DLLs into build/Release/ manually.",
      );
    } else if (!copyLapackRuntimeDlls(sourceDir, destDir)) {
      console.warn(`⚠  Failed to stage LAPACK runtime DLLs from ${sourceDir} into ${destDir}.`);
    } else {
      console.log(`   Staged LAPACK runtime DLLs from ${sourceDir} into build/Release/.`);
    }
  }

  if (hasLapack) {
    console.log("✓  Native Fortran addon built successfully (with LAPACK acceleration).");
  } else {
    console.log("✓  Native Fortran addon built successfully (special functions only, no LAPACK).");
  }
}

main();