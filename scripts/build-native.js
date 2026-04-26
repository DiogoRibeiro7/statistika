#!/usr/bin/env node

/**
 * Attempts to build the native Fortran/C++ addon.
 * Gracefully skips if gfortran, LAPACK, or node-gyp are not available.
 * The library falls back to pure TypeScript implementations at runtime.
 */

const { execSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const PLATFORM = process.platform;
const IS_WINDOWS = PLATFORM === "win32";
const FORTRAN_COMPILE_FLAGS = IS_WINDOWS ? "-c -O2 -Wno-error=line-truncation" : "-c -fPIC -O2";

function run(cmd, label) {
  try {
    execSync(cmd, { cwd: ROOT, stdio: "pipe", shell: true });
    return true;
  } catch (error) {
    console.warn(`⚠  Skipping native build: ${label} failed`);
    console.warn(`    Command: ${cmd}`);
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
    const resolved = execSync(`gfortran -print-file-name=${name}`, {
      stdio: "pipe",
      shell: true,
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
  const libs = hasLapack ? "-llapack -lblas" : "";
  const objs = objFiles.map((f) => `"${f}"`).join(" ");
  // -Wl,--export-all-symbols ensures every bind(C) routine is exported even
  // if MinGW's auto-export heuristic ever changes. The link will fail loudly
  // if -llapack/-lblas can't be resolved, which is the right behavior.
  const cmd = `gfortran -shared -o "${dll}" -Wl,--out-implib="${implib}" -Wl,--export-all-symbols ${objs} ${libs}`;
  return run(cmd, "Fortran bridge DLL link");
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

function main() {
  // Check if gfortran is available
  try {
    execSync("gfortran --version", { stdio: "pipe", shell: true });
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
  } else {
    try {
      execSync("ldconfig -p 2>/dev/null | grep -q liblapack || test -f /usr/lib/liblapack.so || test -f /usr/lib/x86_64-linux-gnu/liblapack.so", {
        stdio: "pipe",
        shell: true,
      });
      hasLapack = true;
    } catch {
      console.warn(
        "⚠  LAPACK not found — linear algebra will use pure TypeScript fallbacks.\n" +
          "   Install liblapack-dev (Debian/Ubuntu) or lapack-devel (RHEL/Fedora) for native acceleration.",
      );
    }
  }

  // Check if source files exist
  const specialSrc = path.join(ROOT, "native/fortran/special_functions.f90");
  const linalgSrc = path.join(ROOT, "native/fortran/linalg.f90");
  if (!fs.existsSync(specialSrc)) {
    console.warn("⚠  Fortran source not found — skipping native addon build.");
    return;
  }

  // Compile special_functions.f90
  const specialObj = path.join(ROOT, "native/fortran/special_functions.o");
  if (
    !run(
      `gfortran ${FORTRAN_COMPILE_FLAGS} -J${path.join(ROOT, "native/fortran")} -o ${specialObj} ${specialSrc}`,
      "Fortran special functions compilation",
    )
  ) {
    return;
  }

  // Compile distributions.f90 (depends on special_functions module)
  const distSrc = path.join(ROOT, "native/fortran/distributions.f90");
  const distObj = path.join(ROOT, "native/fortran/distributions.o");
  if (fs.existsSync(distSrc)) {
    if (
      !run(
        `gfortran ${FORTRAN_COMPILE_FLAGS} -J${path.join(ROOT, "native/fortran")} -o ${distObj} ${distSrc}`,
        "Fortran distributions compilation",
      )
    ) {
      try { fs.unlinkSync(distObj); } catch {}
    }
  }

  // Compile statistics.f90
  const statsSrc = path.join(ROOT, "native/fortran/statistics.f90");
  const statsObj = path.join(ROOT, "native/fortran/statistics.o");
  if (fs.existsSync(statsSrc)) {
    if (
      !run(
        `gfortran ${FORTRAN_COMPILE_FLAGS} -o ${statsObj} ${statsSrc}`,
        "Fortran statistics compilation",
      )
    ) {
      // Create empty stub if compilation fails
      try { fs.unlinkSync(statsObj); } catch {}
    }
  }

  // Compile time_series.f90
  const tsSrc = path.join(ROOT, "native/fortran/time_series.f90");
  const tsObj = path.join(ROOT, "native/fortran/time_series.o");
  if (fs.existsSync(tsSrc)) {
    if (
      !run(
        `gfortran ${FORTRAN_COMPILE_FLAGS} -o ${tsObj} ${tsSrc}`,
        "Fortran time_series compilation",
      )
    ) {
      try { fs.unlinkSync(tsObj); } catch {}
    }
  }

  // Compile kalman.f90
  const kalmanSrc = path.join(ROOT, "native/fortran/kalman.f90");
  const kalmanObj = path.join(ROOT, "native/fortran/kalman.o");
  if (fs.existsSync(kalmanSrc)) {
    if (
      !run(
        `gfortran ${FORTRAN_COMPILE_FLAGS} -o ${kalmanObj} ${kalmanSrc}`,
        "Fortran kalman compilation",
      )
    ) {
      try { fs.unlinkSync(kalmanObj); } catch {}
    }
  }

  // Compile optimization.f90
  const optSrc = path.join(ROOT, "native/fortran/optimization.f90");
  const optObj = path.join(ROOT, "native/fortran/optimization.o");
  if (fs.existsSync(optSrc)) {
    if (
      !run(
        `gfortran ${FORTRAN_COMPILE_FLAGS} -o ${optObj} ${optSrc}`,
        "Fortran optimization compilation",
      )
    ) {
      try { fs.unlinkSync(optObj); } catch {}
    }
  }

  // Compile sampling.f90
  const samplingSrc = path.join(ROOT, "native/fortran/sampling.f90");
  const samplingObj = path.join(ROOT, "native/fortran/sampling.o");
  if (fs.existsSync(samplingSrc)) {
    if (
      !run(
        `gfortran ${FORTRAN_COMPILE_FLAGS} -o ${samplingObj} ${samplingSrc}`,
        "Fortran sampling compilation",
      )
    ) {
      try { fs.unlinkSync(samplingObj); } catch {}
    }
  }

  // Compile linalg.f90 (only if LAPACK is available and source exists)
  if (hasLapack && fs.existsSync(linalgSrc)) {
    const linalgObj = path.join(ROOT, "native/fortran/linalg.o");
    if (
      !run(
        `gfortran ${FORTRAN_COMPILE_FLAGS} -o ${linalgObj} ${linalgSrc}`,
        "Fortran linalg compilation",
      )
    ) {
      // If linalg fails, remove the object so binding.gyp doesn't try to link it
      try { fs.unlinkSync(linalgObj); } catch {}
      hasLapack = false;
    }
  } else {
    hasLapack = false;
  }

  // If no LAPACK, create a dummy linalg.o stub so binding.gyp doesn't fail
  if (!hasLapack) {
    const linalgObj = path.join(ROOT, "native/fortran/linalg.o");
    const stubSrc = path.join(ROOT, "native/fortran/linalg_stub.f90");
    // Write a minimal stub that provides the expected symbols (no-ops)
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

subroutine fortran_pairwise_euclidean(data, dist, n, p) bind(C, name="fortran_pairwise_euclidean")
  use iso_c_binding
  real(c_double), intent(in) :: data(*)
  real(c_double), intent(out) :: dist(*)
  integer(c_int), intent(in) :: n, p
  dist(1) = 0.0d0
end subroutine

subroutine fortran_gaussian_pdf_batch(x, mu, sigma2, result, n) bind(C, name="fortran_gaussian_pdf_batch")
  use iso_c_binding
  real(c_double), intent(in) :: x(*)
  real(c_double), intent(in), value :: mu, sigma2
  real(c_double), intent(out) :: result(*)
  integer(c_int), intent(in) :: n
  result(1) = 0.0d0
end subroutine

subroutine fortran_kde_gaussian(data, eval_points, density, n, m, bandwidth) bind(C, name="fortran_kde_gaussian")
  use iso_c_binding
  real(c_double), intent(in) :: data(*), eval_points(*)
  real(c_double), intent(out) :: density(*)
  integer(c_int), intent(in) :: n, m
  real(c_double), intent(in), value :: bandwidth
  density(1) = 0.0d0
end subroutine

subroutine fortran_weighted_cross_products(X, W, z, XtWX, XtWz, n, cols) bind(C, name="fortran_weighted_cross_products")
  use iso_c_binding
  real(c_double), intent(in) :: X(*), W(*), z(*)
  real(c_double), intent(out) :: XtWX(*), XtWz(*)
  integer(c_int), intent(in) :: n, cols
  XtWX(1) = 0.0d0
  XtWz(1) = 0.0d0
end subroutine

subroutine fortran_welford_batch(values, n, count_in, mean_in, m2_in, min_in, max_in, &
    count_out, mean_out, m2_out, min_out, max_out) bind(C, name="fortran_welford_batch")
  use iso_c_binding
  real(c_double), intent(in) :: values(*)
  integer(c_int), intent(in) :: n
  integer(c_int), intent(in), value :: count_in
  real(c_double), intent(in), value :: mean_in, m2_in, min_in, max_in
  integer(c_int), intent(out) :: count_out
  real(c_double), intent(out) :: mean_out, m2_out, min_out, max_out
  count_out = count_in
  mean_out = mean_in
  m2_out = m2_in
  min_out = min_in
  max_out = max_in
end subroutine
`);
    run(
      `gfortran ${FORTRAN_COMPILE_FLAGS} -o ${linalgObj} ${stubSrc}`,
      "Fortran linalg stub compilation",
    );
    // Clean up stub source
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
  if (!run(`npx node-gyp configure -- -Duse_lapack=${hasLapack ? 1 : 0}`, "node-gyp configure")) {
    return;
  }

  // On Windows, pre-link all Fortran objects into a single bridge DLL so
  // the MSVC-driven node-gyp build only has to link addon.cpp against an
  // import lib. binding.gyp expects the import lib at build/Release/.
  if (IS_WINDOWS) {
    const bridgeObjs = [
      path.join(ROOT, "native/fortran/special_functions.o"),
      path.join(ROOT, "native/fortran/distributions.o"),
      path.join(ROOT, "native/fortran/linalg.o"),
      path.join(ROOT, "native/fortran/statistics.o"),
      path.join(ROOT, "native/fortran/time_series.o"),
      path.join(ROOT, "native/fortran/kalman.o"),
      path.join(ROOT, "native/fortran/optimization.o"),
      path.join(ROOT, "native/fortran/sampling.o"),
    ].filter((p) => fs.existsSync(p));
    const releaseDir = path.join(ROOT, "build", "Release");
    if (!linkFortranBridgeDll(bridgeObjs, hasLapack, releaseDir)) {
      return;
    }
  }

  if (!run("npx node-gyp build", "node-gyp build")) {
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
