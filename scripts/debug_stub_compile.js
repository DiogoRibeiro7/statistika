const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const os = require('os');
const ROOT = path.resolve(__dirname, '..');
const stubPath = path.join(ROOT, 'native', 'fortran', 'linalg_stub.f90');
const outPath = path.join(os.tmpdir(), 'compile_test.o');
const code = `subroutine fortran_mat_mul(a, b, c, m, k, n) bind(C, name="fortran_mat_mul")
  use iso_c_binding
  real(c_double), intent(in) :: a(*), b(*)
  real(c_double), intent(out) :: c(*)
  integer(c_int), intent(in) :: m, k, n
  c(1) = 0.0d0
end subroutine
`;
fs.writeFileSync(stubPath, code);
console.log('stubPath=', stubPath);
console.log('outPath=', outPath);
try {
  execSync(`gfortran -c -O2 -o ${JSON.stringify(outPath)} ${JSON.stringify(stubPath)}`, {
    stdio: 'pipe',
    shell: true,
  });
  console.log('COMPILATION_OK');
} catch (err) {
  console.log('COMPILATION_FAIL');
  if (err.stdout) console.log(err.stdout.toString());
  if (err.stderr) console.log(err.stderr.toString());
  else console.log(err.toString());
}
