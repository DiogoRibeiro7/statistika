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

  // Check if LAPACK is available
  let hasLapack = false;
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
      `gfortran -c -fPIC -O2 -o ${specialObj} ${specialSrc}`,
      "Fortran special functions compilation",
    )
  ) {
    return;
  }

  // Compile statistics.f90
  const statsSrc = path.join(ROOT, "native/fortran/statistics.f90");
  const statsObj = path.join(ROOT, "native/fortran/statistics.o");
  if (fs.existsSync(statsSrc)) {
    if (
      !run(
        `gfortran -c -fPIC -O2 -o ${statsObj} ${statsSrc}`,
        "Fortran statistics compilation",
      )
    ) {
      // Create empty stub if compilation fails
      try { fs.unlinkSync(statsObj); } catch {}
    }
  }

  // Compile linalg.f90 (only if LAPACK is available and source exists)
  if (hasLapack && fs.existsSync(linalgSrc)) {
    const linalgObj = path.join(ROOT, "native/fortran/linalg.o");
    if (
      !run(
        `gfortran -c -fPIC -O2 -o ${linalgObj} ${linalgSrc}`,
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

subroutine fortran_welford_batch(values, n, count_in, mean_in, m2_in, min_in, max_in, count_out, mean_out, m2_out, min_out, max_out) bind(C, name="fortran_welford_batch")
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
      `gfortran -c -fPIC -O2 -o ${linalgObj} ${stubSrc}`,
      "Fortran linalg stub compilation",
    );
    // Clean up stub source
    try { fs.unlinkSync(stubSrc); } catch {}
  }

  // Run node-gyp
  if (!run("npx node-gyp configure build", "node-gyp build")) {
    return;
  }

  if (hasLapack) {
    console.log("✓  Native Fortran addon built successfully (with LAPACK acceleration).");
  } else {
    console.log("✓  Native Fortran addon built successfully (special functions only, no LAPACK).");
  }
}

main();
