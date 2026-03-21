#include <napi.h>
#include <vector>
#include <cstring>

// Fortran function declarations — special_functions.f90
extern "C" {
  void fortran_gamma_ln(double x, double* result);
  void fortran_gamma(double x, double* result);
  void fortran_log_factorial(int n, double* result);
  void fortran_factorial(int n, double* result);
  void fortran_binomial_coeff(int n, int k, double* result);
  void fortran_beta_fn(double a, double b, double* result);
  void fortran_erf(double x, double* result);
  void fortran_erfc(double x, double* result);
  void fortran_regularized_gamma_p(double s, double x, double* result);
  void fortran_regularized_beta(double x, double a, double b, double* result);
}

// Fortran function declarations — linalg.f90 (LAPACK-backed)
extern "C" {
  void fortran_mat_mul(const double* a, const double* b, double* c,
                       const int* m, const int* k, const int* n);
  void fortran_solve(const double* a, const double* b, double* x,
                     const int* n, int* info);
  void fortran_invert(const double* a, double* inv, const int* n, int* info);
  void fortran_sym_eigen(const double* a, double* eigenvalues,
                         double* eigenvectors, const int* n, int* info);
  void fortran_normal_cdf(double x, double* result);
}

// ==========================================================================
// Helpers: JS row-major array-of-arrays <-> Fortran column-major flat array
// ==========================================================================

// Read a JS array-of-arrays (row-major) into a column-major flat vector.
static std::vector<double> jsMatrixToColMajor(Napi::Env env,
                                               Napi::Array jsMatrix,
                                               int rows, int cols) {
  std::vector<double> flat(rows * cols);
  for (int i = 0; i < rows; i++) {
    Napi::Array row = jsMatrix.Get(static_cast<uint32_t>(i)).As<Napi::Array>();
    for (int j = 0; j < cols; j++) {
      // Column-major: flat[j * rows + i]
      flat[j * rows + i] = row.Get(static_cast<uint32_t>(j))
                               .As<Napi::Number>().DoubleValue();
    }
  }
  return flat;
}

// Read a JS flat array into a std::vector<double>.
static std::vector<double> jsArrayToVector(Napi::Env env,
                                            Napi::Array jsArr, int n) {
  std::vector<double> v(n);
  for (int i = 0; i < n; i++) {
    v[i] = jsArr.Get(static_cast<uint32_t>(i)).As<Napi::Number>().DoubleValue();
  }
  return v;
}

// Convert a column-major flat array into a JS array-of-arrays (row-major).
static Napi::Array colMajorToJsMatrix(Napi::Env env,
                                       const double* flat,
                                       int rows, int cols) {
  Napi::Array result = Napi::Array::New(env, rows);
  for (int i = 0; i < rows; i++) {
    Napi::Array row = Napi::Array::New(env, cols);
    for (int j = 0; j < cols; j++) {
      row.Set(static_cast<uint32_t>(j),
              Napi::Number::New(env, flat[j * rows + i]));
    }
    result.Set(static_cast<uint32_t>(i), row);
  }
  return result;
}

// ==========================================================================
// Special function wrappers (unchanged)
// ==========================================================================

Napi::Value GammaLn(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  double x = info[0].As<Napi::Number>().DoubleValue();
  double result;
  fortran_gamma_ln(x, &result);
  return Napi::Number::New(env, result);
}

Napi::Value Gamma(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  double x = info[0].As<Napi::Number>().DoubleValue();
  double result;
  fortran_gamma(x, &result);
  return Napi::Number::New(env, result);
}

Napi::Value LogFactorial(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  int n = info[0].As<Napi::Number>().Int32Value();
  double result;
  fortran_log_factorial(n, &result);
  return Napi::Number::New(env, result);
}

Napi::Value Factorial(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  int n = info[0].As<Napi::Number>().Int32Value();
  double result;
  fortran_factorial(n, &result);
  return Napi::Number::New(env, result);
}

Napi::Value BinomialCoeff(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  int n = info[0].As<Napi::Number>().Int32Value();
  int k = info[1].As<Napi::Number>().Int32Value();
  double result;
  fortran_binomial_coeff(n, k, &result);
  return Napi::Number::New(env, result);
}

Napi::Value BetaFn(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  double a = info[0].As<Napi::Number>().DoubleValue();
  double b = info[1].As<Napi::Number>().DoubleValue();
  double result;
  fortran_beta_fn(a, b, &result);
  return Napi::Number::New(env, result);
}

Napi::Value Erf(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  double x = info[0].As<Napi::Number>().DoubleValue();
  double result;
  fortran_erf(x, &result);
  return Napi::Number::New(env, result);
}

Napi::Value Erfc(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  double x = info[0].As<Napi::Number>().DoubleValue();
  double result;
  fortran_erfc(x, &result);
  return Napi::Number::New(env, result);
}

Napi::Value RegularizedGammaP(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  double s = info[0].As<Napi::Number>().DoubleValue();
  double x = info[1].As<Napi::Number>().DoubleValue();
  double result;
  fortran_regularized_gamma_p(s, x, &result);
  return Napi::Number::New(env, result);
}

Napi::Value RegularizedBeta(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  double x = info[0].As<Napi::Number>().DoubleValue();
  double a = info[1].As<Napi::Number>().DoubleValue();
  double b = info[2].As<Napi::Number>().DoubleValue();
  double result;
  fortran_regularized_beta(x, a, b, &result);
  return Napi::Number::New(env, result);
}

// ==========================================================================
// Linear algebra wrappers (LAPACK-backed)
// ==========================================================================

// matMul(A, B, m, k, n) -> C
// A: m×k array-of-arrays, B: k×n array-of-arrays -> C: m×n array-of-arrays
Napi::Value MatMul(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  Napi::Array jsA = info[0].As<Napi::Array>();
  Napi::Array jsB = info[1].As<Napi::Array>();
  int m = info[2].As<Napi::Number>().Int32Value();
  int k = info[3].As<Napi::Number>().Int32Value();
  int n = info[4].As<Napi::Number>().Int32Value();

  std::vector<double> a = jsMatrixToColMajor(env, jsA, m, k);
  std::vector<double> b = jsMatrixToColMajor(env, jsB, k, n);
  std::vector<double> c(m * n);

  fortran_mat_mul(a.data(), b.data(), c.data(), &m, &k, &n);

  return colMajorToJsMatrix(env, c.data(), m, n);
}

// solve(A, b) -> { x, info }
// A: n×n array-of-arrays, b: flat array of length n
Napi::Value Solve(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  Napi::Array jsA = info[0].As<Napi::Array>();
  Napi::Array jsB = info[1].As<Napi::Array>();
  int n = info[2].As<Napi::Number>().Int32Value();

  std::vector<double> a = jsMatrixToColMajor(env, jsA, n, n);
  std::vector<double> b = jsArrayToVector(env, jsB, n);
  std::vector<double> x(n);
  int lapack_info = 0;

  fortran_solve(a.data(), b.data(), x.data(), &n, &lapack_info);

  Napi::Object result = Napi::Object::New(env);
  Napi::Array jsX = Napi::Array::New(env, n);
  for (int i = 0; i < n; i++) {
    jsX.Set(static_cast<uint32_t>(i), Napi::Number::New(env, x[i]));
  }
  result.Set("x", jsX);
  result.Set("info", Napi::Number::New(env, lapack_info));
  return result;
}

// invert(A, n) -> { inv, info }
// A: n×n array-of-arrays -> inv: n×n array-of-arrays
Napi::Value Invert(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  Napi::Array jsA = info[0].As<Napi::Array>();
  int n = info[1].As<Napi::Number>().Int32Value();

  std::vector<double> a = jsMatrixToColMajor(env, jsA, n, n);
  std::vector<double> inv(n * n);
  int lapack_info = 0;

  fortran_invert(a.data(), inv.data(), &n, &lapack_info);

  Napi::Object result = Napi::Object::New(env);
  result.Set("inv", colMajorToJsMatrix(env, inv.data(), n, n));
  result.Set("info", Napi::Number::New(env, lapack_info));
  return result;
}

// symEigen(A, n) -> { eigenvalues, eigenvectors, info }
// A: n×n symmetric array-of-arrays
// eigenvalues: flat array (ascending), eigenvectors: n×n array-of-arrays
Napi::Value SymEigen(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  Napi::Array jsA = info[0].As<Napi::Array>();
  int n = info[1].As<Napi::Number>().Int32Value();

  std::vector<double> a = jsMatrixToColMajor(env, jsA, n, n);
  std::vector<double> eigenvalues(n);
  std::vector<double> eigenvectors(n * n);
  int lapack_info = 0;

  fortran_sym_eigen(a.data(), eigenvalues.data(), eigenvectors.data(),
                    &n, &lapack_info);

  Napi::Object result = Napi::Object::New(env);

  Napi::Array jsEvals = Napi::Array::New(env, n);
  for (int i = 0; i < n; i++) {
    jsEvals.Set(static_cast<uint32_t>(i),
                Napi::Number::New(env, eigenvalues[i]));
  }
  result.Set("eigenvalues", jsEvals);
  result.Set("eigenvectors", colMajorToJsMatrix(env, eigenvectors.data(), n, n));
  result.Set("info", Napi::Number::New(env, lapack_info));
  return result;
}

// normalCdf(x) -> number
Napi::Value NormalCdf(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  double x = info[0].As<Napi::Number>().DoubleValue();
  double result;
  fortran_normal_cdf(x, &result);
  return Napi::Number::New(env, result);
}

// ==========================================================================
// Module initialization
// ==========================================================================

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  // Special functions
  exports.Set("gammaLn", Napi::Function::New(env, GammaLn));
  exports.Set("gamma", Napi::Function::New(env, Gamma));
  exports.Set("logFactorial", Napi::Function::New(env, LogFactorial));
  exports.Set("factorial", Napi::Function::New(env, Factorial));
  exports.Set("binomialCoeff", Napi::Function::New(env, BinomialCoeff));
  exports.Set("betaFn", Napi::Function::New(env, BetaFn));
  exports.Set("erf", Napi::Function::New(env, Erf));
  exports.Set("erfc", Napi::Function::New(env, Erfc));
  exports.Set("regularizedGammaP", Napi::Function::New(env, RegularizedGammaP));
  exports.Set("regularizedBeta", Napi::Function::New(env, RegularizedBeta));

  // Linear algebra (LAPACK-backed)
  exports.Set("matMul", Napi::Function::New(env, MatMul));
  exports.Set("solve", Napi::Function::New(env, Solve));
  exports.Set("invert", Napi::Function::New(env, Invert));
  exports.Set("symEigen", Napi::Function::New(env, SymEigen));
  exports.Set("normalCdf", Napi::Function::New(env, NormalCdf));

  return exports;
}

NODE_API_MODULE(fortran_special, Init)
