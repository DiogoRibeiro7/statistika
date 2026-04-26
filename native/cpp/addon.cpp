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
  void fortran_lu(const double* a, double* lu_out, int* ipiv_out,
                  const int* n, int* info);
  void fortran_qr(const double* a, double* q_out, double* r_out,
                  const int* m, const int* n, int* info);
  void fortran_cholesky(const double* a, double* l_out, const int* n,
                        int* info);
  void fortran_svd(const double* a, double* u_out, double* s_out,
                   double* vt_out, const int* m, const int* n, int* info);
}

// Fortran function declarations — statistics.f90
extern "C" {
  void fortran_pairwise_euclidean(const double* data, double* dist,
                                  const int* n, const int* p);
  void fortran_gaussian_pdf_batch(const double* x, double mu, double sigma2,
                                   double* result, const int* n);
  void fortran_kde_gaussian(const double* data, const double* eval_points,
                             double* density, const int* n, const int* m,
                             double bandwidth);
  void fortran_weighted_cross_products(const double* X, const double* W,
                                        const double* z, double* XtWX,
                                        double* XtWz, const int* n,
                                        const int* cols);
  void fortran_welford_batch(const double* values, const int* n,
                              int count_in, double mean_in, double m2_in,
                              double min_in, double max_in,
                              int* count_out, double* mean_out,
                              double* m2_out, double* min_out,
                              double* max_out);
  void fortran_elastic_net_cd(const double* X, const double* yc,
                               double* beta, double* residuals,
                               const double* col_norms,
                               double lambda, double alpha,
                               int max_iter, double tol,
                               const int* n, const int* p,
                               int* iters_out);
  void fortran_ridge_solve(const double* X, const double* yc,
                            double* beta_out, double lambda,
                            const int* n, const int* p, int* info);
}

// Fortran function declarations — time_series.f90
extern "C" {
  void fortran_autocovariance(const double* series, const int* n,
                               double mu, double* gamma_out,
                               const int* maxlag);
  void fortran_acf(const double* series, const int* n,
                    double* acf_out, const int* maxlag);
  void fortran_pacf_durbin_levinson(const double* acf_in,
                                     double* pacf_out, const int* maxlag);
  void fortran_exponential_smoothing(const double* series, const int* n,
                                      double alpha, double* smoothed);
  void fortran_holt_winters(const double* series, const int* n,
                             double alpha, double beta,
                             double* level_out, double* trend_out,
                             double* fitted_out);
  void fortran_difference(const double* series, const int* n,
                           const int* d, double* result);
}

// Fortran function declarations — kalman.f90
extern "C" {
  void fortran_kalman_predict(const double* F, double* x, double* P,
                               const double* Q, const int* m);
  void fortran_kalman_update(const double* H, const double* R,
                              const double* y, double* x, double* P,
                              double* innovation_out, double* S_out,
                              double* loglik_out, const int* m,
                              const int* p);
  void fortran_kalman_filter_univariate(const double* F, const double* H,
                                         const double* Q, double R_scalar,
                                         const double* y, const double* x0,
                                         const double* P0,
                                         double* states_out,
                                         double* loglik_out,
                                         const int* T, const int* m);
}

// Fortran function declarations — distributions.f90
extern "C" {
  void fortran_chi2_cdf(double x, double df, double* result);
  void fortran_chi2_pdf(double x, double df, double* result);
  void fortran_t_cdf(double x, double df, double* result);
  void fortran_t_pdf(double x, double df, double* result);
  void fortran_f_cdf(double x, double d1, double d2, double* result);
  void fortran_f_pdf(double x, double d1, double d2, double* result);
  void fortran_normal_cdf_dist(double x, double* result);
  void fortran_normal_pdf(double x, double* result);
  void fortran_gamma_cdf(double x, double shape, double scale, double* result);
  void fortran_beta_cdf(double x, double a, double b, double* result);
  void fortran_chi2_cdf_batch(const double* x, const int* n, double df,
                                double* result);
  void fortran_t_cdf_batch(const double* x, const int* n, double df,
                             double* result);
  void fortran_normal_cdf_batch(const double* x, const int* n,
                                  double* result);
  void fortran_uniform_sample_batch(int seed, double a, double b,
                                    double* result, const int* n);
  void fortran_normal_sample_batch(int seed, double mu, double sigma,
                                    double* result, const int* n);
  void fortran_gamma_sample_batch(int seed, double shape, double rate,
                                   double* result, const int* n);
  void fortran_beta_sample_batch(int seed, double alpha, double beta,
                                  double* result, const int* n);
}

// Fortran function declarations — optimization.f90
extern "C" {
  void fortran_garch11_loglik(const double* eps, const int* T,
                                double omega, double alpha1, double beta1,
                                double* sigma2_out, double* loglik_out);
  void fortran_garch_pq_loglik(const double* eps, const int* T,
                                 double omega, const double* alpha,
                                 const double* beta, double* sigma2_out,
                                 double* loglik_out, const int* p,
                                 const int* q);
  void fortran_gjr_garch11_loglik(const double* eps, const int* T,
                                    double omega, double alpha1,
                                    double beta1, double gamma1,
                                    double* sigma2_out, double* loglik_out);
  void fortran_egarch11_loglik(const double* eps, const int* T,
                                 double omega, double alpha1,
                                 double beta1, double gamma1,
                                 double* sigma2_out, double* loglik_out);
  void fortran_garch11_forecast(double last_eps2, double last_sigma2,
                                  double omega, double alpha1, double beta1,
                                  double* forecast_out, const int* h);
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
// Matrix decomposition wrappers (LAPACK-backed)
// ==========================================================================

// lu(A, n) -> { lu, ipiv, info }
// A: n×n array-of-arrays -> lu: flat col-major n*n, ipiv: int[n]
Napi::Value LU(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  Napi::Array jsA = info[0].As<Napi::Array>();
  int n = info[1].As<Napi::Number>().Int32Value();

  std::vector<double> a = jsMatrixToColMajor(env, jsA, n, n);
  std::vector<double> lu_out(n * n);
  std::vector<int> ipiv(n);
  int lapack_info = 0;

  fortran_lu(a.data(), lu_out.data(), ipiv.data(), &n, &lapack_info);

  Napi::Object result = Napi::Object::New(env);
  result.Set("lu", colMajorToJsMatrix(env, lu_out.data(), n, n));

  Napi::Array jsIpiv = Napi::Array::New(env, n);
  for (int i = 0; i < n; i++) {
    jsIpiv.Set(static_cast<uint32_t>(i), Napi::Number::New(env, ipiv[i]));
  }
  result.Set("ipiv", jsIpiv);
  result.Set("info", Napi::Number::New(env, lapack_info));
  return result;
}

// qr(A, m, n) -> { Q, R, info }
// A: m×n array-of-arrays -> Q: m×n, R: n×n
Napi::Value QR(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  Napi::Array jsA = info[0].As<Napi::Array>();
  int m = info[1].As<Napi::Number>().Int32Value();
  int n = info[2].As<Napi::Number>().Int32Value();

  std::vector<double> a = jsMatrixToColMajor(env, jsA, m, n);
  std::vector<double> q_out(m * n);
  std::vector<double> r_out(n * n);
  int lapack_info = 0;

  fortran_qr(a.data(), q_out.data(), r_out.data(), &m, &n, &lapack_info);

  Napi::Object result = Napi::Object::New(env);
  result.Set("Q", colMajorToJsMatrix(env, q_out.data(), m, n));
  result.Set("R", colMajorToJsMatrix(env, r_out.data(), n, n));
  result.Set("info", Napi::Number::New(env, lapack_info));
  return result;
}

// cholesky(A, n) -> { L, info }
// A: n×n SPD array-of-arrays -> L: n×n lower triangular
Napi::Value Cholesky(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  Napi::Array jsA = info[0].As<Napi::Array>();
  int n = info[1].As<Napi::Number>().Int32Value();

  std::vector<double> a = jsMatrixToColMajor(env, jsA, n, n);
  std::vector<double> l_out(n * n);
  int lapack_info = 0;

  fortran_cholesky(a.data(), l_out.data(), &n, &lapack_info);

  Napi::Object result = Napi::Object::New(env);
  result.Set("L", colMajorToJsMatrix(env, l_out.data(), n, n));
  result.Set("info", Napi::Number::New(env, lapack_info));
  return result;
}

// svd(A, m, n) -> { U, S, Vt, info }
// A: m×n array-of-arrays -> U: m×k, S: k, Vt: k×n where k = min(m,n)
Napi::Value SVD(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  Napi::Array jsA = info[0].As<Napi::Array>();
  int m = info[1].As<Napi::Number>().Int32Value();
  int n = info[2].As<Napi::Number>().Int32Value();
  int k = std::min(m, n);

  std::vector<double> a = jsMatrixToColMajor(env, jsA, m, n);
  std::vector<double> u_out(m * k);
  std::vector<double> s_out(k);
  std::vector<double> vt_out(k * n);
  int lapack_info = 0;

  fortran_svd(a.data(), u_out.data(), s_out.data(), vt_out.data(),
              &m, &n, &lapack_info);

  Napi::Object result = Napi::Object::New(env);
  result.Set("U", colMajorToJsMatrix(env, u_out.data(), m, k));

  Napi::Array jsS = Napi::Array::New(env, k);
  for (int i = 0; i < k; i++) {
    jsS.Set(static_cast<uint32_t>(i), Napi::Number::New(env, s_out[i]));
  }
  result.Set("S", jsS);

  // V^T is k×n, but caller typically wants V (n×k), so return Vt as-is
  // and let the TypeScript layer transpose if needed.
  result.Set("Vt", colMajorToJsMatrix(env, vt_out.data(), k, n));
  result.Set("info", Napi::Number::New(env, lapack_info));
  return result;
}

// ==========================================================================
// Statistics module wrappers (statistics.f90)
// ==========================================================================

// pairwiseEuclidean(data: number[][], n: number, p: number) => number[][]
Napi::Value PairwiseEuclidean(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  int n = info[1].As<Napi::Number>().Int32Value();
  int p = info[2].As<Napi::Number>().Int32Value();

  auto data = jsMatrixToColMajor(env, info[0].As<Napi::Array>(), n, p);
  std::vector<double> dist(n * n);

  fortran_pairwise_euclidean(data.data(), dist.data(), &n, &p);

  return colMajorToJsMatrix(env, dist.data(), n, n);
}

// gaussianPdfBatch(x: number[], mu: number, sigma2: number) => number[]
Napi::Value GaussianPdfBatch(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  Napi::Array xArr = info[0].As<Napi::Array>();
  double mu = info[1].As<Napi::Number>().DoubleValue();
  double sigma2 = info[2].As<Napi::Number>().DoubleValue();
  int n = xArr.Length();

  auto x = jsArrayToVector(env, xArr, n);
  std::vector<double> result(n);

  fortran_gaussian_pdf_batch(x.data(), mu, sigma2, result.data(), &n);

  Napi::Array jsResult = Napi::Array::New(env, n);
  for (int i = 0; i < n; i++) {
    jsResult.Set(static_cast<uint32_t>(i), Napi::Number::New(env, result[i]));
  }
  return jsResult;
}

// kdeGaussian(data: number[], evalPoints: number[], bandwidth: number) => number[]
Napi::Value KdeGaussian(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  Napi::Array dataArr = info[0].As<Napi::Array>();
  Napi::Array evalArr = info[1].As<Napi::Array>();
  double bandwidth = info[2].As<Napi::Number>().DoubleValue();
  int n = dataArr.Length();
  int m = evalArr.Length();

  auto data = jsArrayToVector(env, dataArr, n);
  auto eval_points = jsArrayToVector(env, evalArr, m);
  std::vector<double> density(m);

  fortran_kde_gaussian(data.data(), eval_points.data(), density.data(),
                       &n, &m, bandwidth);

  Napi::Array jsResult = Napi::Array::New(env, m);
  for (int i = 0; i < m; i++) {
    jsResult.Set(static_cast<uint32_t>(i), Napi::Number::New(env, density[i]));
  }
  return jsResult;
}

// weightedCrossProducts(X: number[][], W: number[], z: number[], n, cols)
Napi::Value WeightedCrossProducts(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  int n = info[3].As<Napi::Number>().Int32Value();
  int cols = info[4].As<Napi::Number>().Int32Value();

  auto X = jsMatrixToColMajor(env, info[0].As<Napi::Array>(), n, cols);
  auto W = jsArrayToVector(env, info[1].As<Napi::Array>(), n);
  auto z = jsArrayToVector(env, info[2].As<Napi::Array>(), n);

  std::vector<double> XtWX(cols * cols);
  std::vector<double> XtWz(cols);

  fortran_weighted_cross_products(X.data(), W.data(), z.data(),
                                   XtWX.data(), XtWz.data(), &n, &cols);

  Napi::Object result = Napi::Object::New(env);
  result.Set("XtWX", colMajorToJsMatrix(env, XtWX.data(), cols, cols));

  Napi::Array jsXtWz = Napi::Array::New(env, cols);
  for (int i = 0; i < cols; i++) {
    jsXtWz.Set(static_cast<uint32_t>(i), Napi::Number::New(env, XtWz[i]));
  }
  result.Set("XtWz", jsXtWz);

  return result;
}

// uniformSampleBatch(n: number, a: number, b: number, seed: number) => number[]
Napi::Value UniformSampleBatch(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  int n = info[0].As<Napi::Number>().Int32Value();
  double a = info[1].As<Napi::Number>().DoubleValue();
  double b = info[2].As<Napi::Number>().DoubleValue();
  int seed = info[3].As<Napi::Number>().Int32Value();

  std::vector<double> result(n);
  fortran_uniform_sample_batch(seed, a, b, result.data(), &n);

  Napi::Array jsResult = Napi::Array::New(env, n);
  for (int i = 0; i < n; i++) {
    jsResult.Set(static_cast<uint32_t>(i), Napi::Number::New(env, result[i]));
  }
  return jsResult;
}

// normalSampleBatch(n: number, mu: number, sigma: number, seed: number) => number[]
Napi::Value NormalSampleBatch(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  int n = info[0].As<Napi::Number>().Int32Value();
  double mu = info[1].As<Napi::Number>().DoubleValue();
  double sigma = info[2].As<Napi::Number>().DoubleValue();
  int seed = info[3].As<Napi::Number>().Int32Value();

  std::vector<double> result(n);
  fortran_normal_sample_batch(seed, mu, sigma, result.data(), &n);

  Napi::Array jsResult = Napi::Array::New(env, n);
  for (int i = 0; i < n; i++) {
    jsResult.Set(static_cast<uint32_t>(i), Napi::Number::New(env, result[i]));
  }
  return jsResult;
}

// gammaSampleBatch(n: number, shape: number, rate: number, seed: number) => number[]
Napi::Value GammaSampleBatch(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  int n = info[0].As<Napi::Number>().Int32Value();
  double shape = info[1].As<Napi::Number>().DoubleValue();
  double rate = info[2].As<Napi::Number>().DoubleValue();
  int seed = info[3].As<Napi::Number>().Int32Value();

  std::vector<double> result(n);
  fortran_gamma_sample_batch(seed, shape, rate, result.data(), &n);

  Napi::Array jsResult = Napi::Array::New(env, n);
  for (int i = 0; i < n; i++) {
    jsResult.Set(static_cast<uint32_t>(i), Napi::Number::New(env, result[i]));
  }
  return jsResult;
}

// betaSampleBatch(n: number, alpha: number, beta: number, seed: number) => number[]
Napi::Value BetaSampleBatch(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  int n = info[0].As<Napi::Number>().Int32Value();
  double alpha = info[1].As<Napi::Number>().DoubleValue();
  double beta = info[2].As<Napi::Number>().DoubleValue();
  int seed = info[3].As<Napi::Number>().Int32Value();

  std::vector<double> result(n);
  fortran_beta_sample_batch(seed, alpha, beta, result.data(), &n);

  Napi::Array jsResult = Napi::Array::New(env, n);
  for (int i = 0; i < n; i++) {
    jsResult.Set(static_cast<uint32_t>(i), Napi::Number::New(env, result[i]));
  }
  return jsResult;
}

// welfordBatch(values: number[], count, mean, m2, min, max)
Napi::Value WelfordBatch(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  Napi::Array valArr = info[0].As<Napi::Array>();
  int n = valArr.Length();
  int count_in = info[1].As<Napi::Number>().Int32Value();
  double mean_in = info[2].As<Napi::Number>().DoubleValue();
  double m2_in = info[3].As<Napi::Number>().DoubleValue();
  double min_in = info[4].As<Napi::Number>().DoubleValue();
  double max_in = info[5].As<Napi::Number>().DoubleValue();

  auto values = jsArrayToVector(env, valArr, n);

  int count_out;
  double mean_out, m2_out, min_out, max_out;

  fortran_welford_batch(values.data(), &n, count_in, mean_in, m2_in,
                         min_in, max_in, &count_out, &mean_out, &m2_out,
                         &min_out, &max_out);

  Napi::Object result = Napi::Object::New(env);
  result.Set("count", Napi::Number::New(env, count_out));
  result.Set("mean", Napi::Number::New(env, mean_out));
  result.Set("m2", Napi::Number::New(env, m2_out));
  result.Set("min", Napi::Number::New(env, min_out));
  result.Set("max", Napi::Number::New(env, max_out));
  return result;
}

// ==========================================================================
// Feature selection wrappers (statistics.f90)
// ==========================================================================

// elasticNetCd(X: number[][], yc: number[], beta: number[], residuals: number[],
//              colNorms: number[], lambda, alpha, maxIter, tol)
// => { beta: number[], residuals: number[], iterations: number }
Napi::Value ElasticNetCd(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  int n = info[0].As<Napi::Array>().Length();
  Napi::Array jsX = info[0].As<Napi::Array>();
  Napi::Array jsYc = info[1].As<Napi::Array>();
  Napi::Array jsBeta = info[2].As<Napi::Array>();
  Napi::Array jsResid = info[3].As<Napi::Array>();
  Napi::Array jsColNorms = info[4].As<Napi::Array>();
  double lambda = info[5].As<Napi::Number>().DoubleValue();
  double alpha = info[6].As<Napi::Number>().DoubleValue();
  int maxIter = info[7].As<Napi::Number>().Int32Value();
  double tol = info[8].As<Napi::Number>().DoubleValue();

  int p = jsX.Get(static_cast<uint32_t>(0)).As<Napi::Array>().Length();

  auto X = jsMatrixToColMajor(env, jsX, n, p);
  auto yc = jsArrayToVector(env, jsYc, n);
  auto beta = jsArrayToVector(env, jsBeta, p);
  auto residuals = jsArrayToVector(env, jsResid, n);
  auto colNorms = jsArrayToVector(env, jsColNorms, p);
  int itersOut = 0;

  fortran_elastic_net_cd(X.data(), yc.data(), beta.data(), residuals.data(),
                          colNorms.data(), lambda, alpha, maxIter, tol,
                          &n, &p, &itersOut);

  Napi::Object result = Napi::Object::New(env);

  Napi::Array jsBetaOut = Napi::Array::New(env, p);
  for (int i = 0; i < p; i++) {
    jsBetaOut.Set(static_cast<uint32_t>(i), Napi::Number::New(env, beta[i]));
  }
  result.Set("beta", jsBetaOut);

  Napi::Array jsResidOut = Napi::Array::New(env, n);
  for (int i = 0; i < n; i++) {
    jsResidOut.Set(static_cast<uint32_t>(i), Napi::Number::New(env, residuals[i]));
  }
  result.Set("residuals", jsResidOut);
  result.Set("iterations", Napi::Number::New(env, itersOut));

  return result;
}

// ridgeSolve(X: number[][], yc: number[], lambda: number)
// => { beta: number[], info: number }
Napi::Value RidgeSolve(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  Napi::Array jsX = info[0].As<Napi::Array>();
  Napi::Array jsYc = info[1].As<Napi::Array>();
  double lambda = info[2].As<Napi::Number>().DoubleValue();

  int n = jsX.Length();
  int p = jsX.Get(static_cast<uint32_t>(0)).As<Napi::Array>().Length();

  auto X = jsMatrixToColMajor(env, jsX, n, p);
  auto yc = jsArrayToVector(env, jsYc, n);
  std::vector<double> betaOut(p);
  int lapack_info = 0;

  fortran_ridge_solve(X.data(), yc.data(), betaOut.data(), lambda,
                       &n, &p, &lapack_info);

  Napi::Object result = Napi::Object::New(env);
  Napi::Array jsBeta = Napi::Array::New(env, p);
  for (int i = 0; i < p; i++) {
    jsBeta.Set(static_cast<uint32_t>(i), Napi::Number::New(env, betaOut[i]));
  }
  result.Set("beta", jsBeta);
  result.Set("info", Napi::Number::New(env, lapack_info));

  return result;
}

// ==========================================================================
// Time series wrappers (time_series.f90)
// ==========================================================================

// acf(series: number[], maxLag: number) => number[]
Napi::Value Acf(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  Napi::Array seriesArr = info[0].As<Napi::Array>();
  int maxLag = info[1].As<Napi::Number>().Int32Value();
  int n = seriesArr.Length();

  auto series = jsArrayToVector(env, seriesArr, n);
  std::vector<double> acf_out(maxLag + 1);

  fortran_acf(series.data(), &n, acf_out.data(), &maxLag);

  Napi::Array jsResult = Napi::Array::New(env, maxLag + 1);
  for (int i = 0; i <= maxLag; i++) {
    jsResult.Set(static_cast<uint32_t>(i), Napi::Number::New(env, acf_out[i]));
  }
  return jsResult;
}

// pacfDurbinLevinson(acf: number[], maxLag: number) => number[]
Napi::Value PacfDurbinLevinson(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  Napi::Array acfArr = info[0].As<Napi::Array>();
  int maxLag = info[1].As<Napi::Number>().Int32Value();

  auto acf_in = jsArrayToVector(env, acfArr, maxLag + 1);
  std::vector<double> pacf_out(maxLag + 1);

  fortran_pacf_durbin_levinson(acf_in.data(), pacf_out.data(), &maxLag);

  Napi::Array jsResult = Napi::Array::New(env, maxLag + 1);
  for (int i = 0; i <= maxLag; i++) {
    jsResult.Set(static_cast<uint32_t>(i), Napi::Number::New(env, pacf_out[i]));
  }
  return jsResult;
}

// exponentialSmoothing(series: number[], alpha: number) => number[]
Napi::Value ExponentialSmoothing(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  Napi::Array seriesArr = info[0].As<Napi::Array>();
  double alpha = info[1].As<Napi::Number>().DoubleValue();
  int n = seriesArr.Length();

  auto series = jsArrayToVector(env, seriesArr, n);
  std::vector<double> smoothed(n);

  fortran_exponential_smoothing(series.data(), &n, alpha, smoothed.data());

  Napi::Array jsResult = Napi::Array::New(env, n);
  for (int i = 0; i < n; i++) {
    jsResult.Set(static_cast<uint32_t>(i), Napi::Number::New(env, smoothed[i]));
  }
  return jsResult;
}

// holtWinters(series: number[], alpha: number, beta: number)
// => { level: number[], trend: number[], fitted: number[] }
Napi::Value HoltWinters(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  Napi::Array seriesArr = info[0].As<Napi::Array>();
  double alpha = info[1].As<Napi::Number>().DoubleValue();
  double beta = info[2].As<Napi::Number>().DoubleValue();
  int n = seriesArr.Length();

  auto series = jsArrayToVector(env, seriesArr, n);
  std::vector<double> level(n), trend(n), fitted(n);

  fortran_holt_winters(series.data(), &n, alpha, beta,
                        level.data(), trend.data(), fitted.data());

  Napi::Object result = Napi::Object::New(env);
  Napi::Array jsLevel = Napi::Array::New(env, n);
  Napi::Array jsTrend = Napi::Array::New(env, n);
  Napi::Array jsFitted = Napi::Array::New(env, n);
  for (int i = 0; i < n; i++) {
    jsLevel.Set(static_cast<uint32_t>(i), Napi::Number::New(env, level[i]));
    jsTrend.Set(static_cast<uint32_t>(i), Napi::Number::New(env, trend[i]));
    jsFitted.Set(static_cast<uint32_t>(i), Napi::Number::New(env, fitted[i]));
  }
  result.Set("level", jsLevel);
  result.Set("trend", jsTrend);
  result.Set("fitted", jsFitted);
  return result;
}

// difference(series: number[], d: number) => number[]
Napi::Value Difference(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  Napi::Array seriesArr = info[0].As<Napi::Array>();
  int d = info[1].As<Napi::Number>().Int32Value();
  int n = seriesArr.Length();

  auto series = jsArrayToVector(env, seriesArr, n);
  int resultLen = n - d;
  if (resultLen <= 0) return Napi::Array::New(env, 0);

  std::vector<double> result(resultLen);
  fortran_difference(series.data(), &n, &d, result.data());

  Napi::Array jsResult = Napi::Array::New(env, resultLen);
  for (int i = 0; i < resultLen; i++) {
    jsResult.Set(static_cast<uint32_t>(i), Napi::Number::New(env, result[i]));
  }
  return jsResult;
}

// ==========================================================================
// Kalman filter wrappers (kalman.f90)
// ==========================================================================

// kalmanFilterUnivariate(F, H, Q, R, y, x0, P0, T, m)
// => { states: number[][], logLikelihood: number }
Napi::Value KalmanFilterUnivariate(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  Napi::Array jsF = info[0].As<Napi::Array>();
  Napi::Array jsH = info[1].As<Napi::Array>();
  Napi::Array jsQ = info[2].As<Napi::Array>();
  double R_scalar = info[3].As<Napi::Number>().DoubleValue();
  Napi::Array jsY = info[4].As<Napi::Array>();
  Napi::Array jsX0 = info[5].As<Napi::Array>();
  Napi::Array jsP0 = info[6].As<Napi::Array>();
  int T = info[7].As<Napi::Number>().Int32Value();
  int m = info[8].As<Napi::Number>().Int32Value();

  auto F = jsMatrixToColMajor(env, jsF, m, m);
  auto H = jsArrayToVector(env, jsH, m);
  auto Q = jsMatrixToColMajor(env, jsQ, m, m);
  auto y = jsArrayToVector(env, jsY, T);
  auto x0 = jsArrayToVector(env, jsX0, m);
  auto P0 = jsMatrixToColMajor(env, jsP0, m, m);

  std::vector<double> states_out(T * m);
  double loglik_out = 0.0;

  fortran_kalman_filter_univariate(F.data(), H.data(), Q.data(), R_scalar,
                                    y.data(), x0.data(), P0.data(),
                                    states_out.data(), &loglik_out,
                                    &T, &m);

  // Convert states to array-of-arrays (T x m, row-major)
  Napi::Array jsStates = Napi::Array::New(env, T);
  for (int t = 0; t < T; t++) {
    Napi::Array jsRow = Napi::Array::New(env, m);
    for (int i = 0; i < m; i++) {
      jsRow.Set(static_cast<uint32_t>(i),
                Napi::Number::New(env, states_out[t * m + i]));
    }
    jsStates.Set(static_cast<uint32_t>(t), jsRow);
  }

  Napi::Object result = Napi::Object::New(env);
  result.Set("states", jsStates);
  result.Set("logLikelihood", Napi::Number::New(env, loglik_out));
  return result;
}

// ==========================================================================
// Distribution wrappers (distributions.f90)
// ==========================================================================

Napi::Value Chi2Cdf(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  double x = info[0].As<Napi::Number>().DoubleValue();
  double df = info[1].As<Napi::Number>().DoubleValue();
  double result;
  fortran_chi2_cdf(x, df, &result);
  return Napi::Number::New(env, result);
}

Napi::Value Chi2Pdf(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  double x = info[0].As<Napi::Number>().DoubleValue();
  double df = info[1].As<Napi::Number>().DoubleValue();
  double result;
  fortran_chi2_pdf(x, df, &result);
  return Napi::Number::New(env, result);
}

Napi::Value TCdf(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  double x = info[0].As<Napi::Number>().DoubleValue();
  double df = info[1].As<Napi::Number>().DoubleValue();
  double result;
  fortran_t_cdf(x, df, &result);
  return Napi::Number::New(env, result);
}

Napi::Value TPdf(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  double x = info[0].As<Napi::Number>().DoubleValue();
  double df = info[1].As<Napi::Number>().DoubleValue();
  double result;
  fortran_t_pdf(x, df, &result);
  return Napi::Number::New(env, result);
}

Napi::Value FCdf(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  double x = info[0].As<Napi::Number>().DoubleValue();
  double d1 = info[1].As<Napi::Number>().DoubleValue();
  double d2 = info[2].As<Napi::Number>().DoubleValue();
  double result;
  fortran_f_cdf(x, d1, d2, &result);
  return Napi::Number::New(env, result);
}

Napi::Value FPdf(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  double x = info[0].As<Napi::Number>().DoubleValue();
  double d1 = info[1].As<Napi::Number>().DoubleValue();
  double d2 = info[2].As<Napi::Number>().DoubleValue();
  double result;
  fortran_f_pdf(x, d1, d2, &result);
  return Napi::Number::New(env, result);
}

Napi::Value NormalPdf(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  double x = info[0].As<Napi::Number>().DoubleValue();
  double result;
  fortran_normal_pdf(x, &result);
  return Napi::Number::New(env, result);
}

Napi::Value GammaCdf(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  double x = info[0].As<Napi::Number>().DoubleValue();
  double shape = info[1].As<Napi::Number>().DoubleValue();
  double scale = info[2].As<Napi::Number>().DoubleValue();
  double result;
  fortran_gamma_cdf(x, shape, scale, &result);
  return Napi::Number::New(env, result);
}

Napi::Value BetaCdf(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  double x = info[0].As<Napi::Number>().DoubleValue();
  double a = info[1].As<Napi::Number>().DoubleValue();
  double b = info[2].As<Napi::Number>().DoubleValue();
  double result;
  fortran_beta_cdf(x, a, b, &result);
  return Napi::Number::New(env, result);
}

// Batch CDF evaluations
Napi::Value Chi2CdfBatch(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  Napi::Array xArr = info[0].As<Napi::Array>();
  double df = info[1].As<Napi::Number>().DoubleValue();
  int n = xArr.Length();

  auto x = jsArrayToVector(env, xArr, n);
  std::vector<double> result(n);
  fortran_chi2_cdf_batch(x.data(), &n, df, result.data());

  Napi::Array jsResult = Napi::Array::New(env, n);
  for (int i = 0; i < n; i++) {
    jsResult.Set(static_cast<uint32_t>(i), Napi::Number::New(env, result[i]));
  }
  return jsResult;
}

Napi::Value TCdfBatch(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  Napi::Array xArr = info[0].As<Napi::Array>();
  double df = info[1].As<Napi::Number>().DoubleValue();
  int n = xArr.Length();

  auto x = jsArrayToVector(env, xArr, n);
  std::vector<double> result(n);
  fortran_t_cdf_batch(x.data(), &n, df, result.data());

  Napi::Array jsResult = Napi::Array::New(env, n);
  for (int i = 0; i < n; i++) {
    jsResult.Set(static_cast<uint32_t>(i), Napi::Number::New(env, result[i]));
  }
  return jsResult;
}

Napi::Value NormalCdfBatch(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  Napi::Array xArr = info[0].As<Napi::Array>();
  int n = xArr.Length();

  auto x = jsArrayToVector(env, xArr, n);
  std::vector<double> result(n);
  fortran_normal_cdf_batch(x.data(), &n, result.data());

  Napi::Array jsResult = Napi::Array::New(env, n);
  for (int i = 0; i < n; i++) {
    jsResult.Set(static_cast<uint32_t>(i), Napi::Number::New(env, result[i]));
  }
  return jsResult;
}

// ==========================================================================
// GARCH/optimization wrappers (optimization.f90)
// ==========================================================================

// garch11Loglik(eps: number[], omega, alpha1, beta1)
// => { sigma2: number[], logLikelihood: number }
Napi::Value Garch11Loglik(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  Napi::Array epsArr = info[0].As<Napi::Array>();
  double omega = info[1].As<Napi::Number>().DoubleValue();
  double alpha1 = info[2].As<Napi::Number>().DoubleValue();
  double beta1 = info[3].As<Napi::Number>().DoubleValue();
  int T = epsArr.Length();

  auto eps = jsArrayToVector(env, epsArr, T);
  std::vector<double> sigma2(T);
  double loglik = 0.0;

  fortran_garch11_loglik(eps.data(), &T, omega, alpha1, beta1,
                          sigma2.data(), &loglik);

  Napi::Object result = Napi::Object::New(env);
  Napi::Array jsSigma2 = Napi::Array::New(env, T);
  for (int i = 0; i < T; i++) {
    jsSigma2.Set(static_cast<uint32_t>(i), Napi::Number::New(env, sigma2[i]));
  }
  result.Set("sigma2", jsSigma2);
  result.Set("logLikelihood", Napi::Number::New(env, loglik));
  return result;
}

// garchPqLoglik(eps: number[], omega, alpha: number[], beta: number[])
// => { sigma2: number[], logLikelihood: number }
Napi::Value GarchPqLoglik(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  Napi::Array epsArr = info[0].As<Napi::Array>();
  double omega = info[1].As<Napi::Number>().DoubleValue();
  Napi::Array alphaArr = info[2].As<Napi::Array>();
  Napi::Array betaArr = info[3].As<Napi::Array>();
  int T = epsArr.Length();
  int q = alphaArr.Length();
  int p = betaArr.Length();

  auto eps = jsArrayToVector(env, epsArr, T);
  auto alpha = jsArrayToVector(env, alphaArr, q);
  auto beta = jsArrayToVector(env, betaArr, p);
  std::vector<double> sigma2(T);
  double loglik = 0.0;

  fortran_garch_pq_loglik(eps.data(), &T, omega, alpha.data(), beta.data(),
                            sigma2.data(), &loglik, &p, &q);

  Napi::Object result = Napi::Object::New(env);
  Napi::Array jsSigma2 = Napi::Array::New(env, T);
  for (int i = 0; i < T; i++) {
    jsSigma2.Set(static_cast<uint32_t>(i), Napi::Number::New(env, sigma2[i]));
  }
  result.Set("sigma2", jsSigma2);
  result.Set("logLikelihood", Napi::Number::New(env, loglik));
  return result;
}

// gjrGarch11Loglik(eps, omega, alpha1, beta1, gamma1)
Napi::Value GjrGarch11Loglik(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  Napi::Array epsArr = info[0].As<Napi::Array>();
  double omega = info[1].As<Napi::Number>().DoubleValue();
  double alpha1 = info[2].As<Napi::Number>().DoubleValue();
  double beta1 = info[3].As<Napi::Number>().DoubleValue();
  double gamma1 = info[4].As<Napi::Number>().DoubleValue();
  int T = epsArr.Length();

  auto eps = jsArrayToVector(env, epsArr, T);
  std::vector<double> sigma2(T);
  double loglik = 0.0;

  fortran_gjr_garch11_loglik(eps.data(), &T, omega, alpha1, beta1, gamma1,
                               sigma2.data(), &loglik);

  Napi::Object result = Napi::Object::New(env);
  Napi::Array jsSigma2 = Napi::Array::New(env, T);
  for (int i = 0; i < T; i++) {
    jsSigma2.Set(static_cast<uint32_t>(i), Napi::Number::New(env, sigma2[i]));
  }
  result.Set("sigma2", jsSigma2);
  result.Set("logLikelihood", Napi::Number::New(env, loglik));
  return result;
}

// egarch11Loglik(eps, omega, alpha1, beta1, gamma1)
Napi::Value Egarch11Loglik(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  Napi::Array epsArr = info[0].As<Napi::Array>();
  double omega = info[1].As<Napi::Number>().DoubleValue();
  double alpha1 = info[2].As<Napi::Number>().DoubleValue();
  double beta1 = info[3].As<Napi::Number>().DoubleValue();
  double gamma1 = info[4].As<Napi::Number>().DoubleValue();
  int T = epsArr.Length();

  auto eps = jsArrayToVector(env, epsArr, T);
  std::vector<double> sigma2(T);
  double loglik = 0.0;

  fortran_egarch11_loglik(eps.data(), &T, omega, alpha1, beta1, gamma1,
                            sigma2.data(), &loglik);

  Napi::Object result = Napi::Object::New(env);
  Napi::Array jsSigma2 = Napi::Array::New(env, T);
  for (int i = 0; i < T; i++) {
    jsSigma2.Set(static_cast<uint32_t>(i), Napi::Number::New(env, sigma2[i]));
  }
  result.Set("sigma2", jsSigma2);
  result.Set("logLikelihood", Napi::Number::New(env, loglik));
  return result;
}

// garch11Forecast(lastEps2, lastSigma2, omega, alpha1, beta1, h)
Napi::Value Garch11Forecast(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  double lastEps2 = info[0].As<Napi::Number>().DoubleValue();
  double lastSigma2 = info[1].As<Napi::Number>().DoubleValue();
  double omega = info[2].As<Napi::Number>().DoubleValue();
  double alpha1 = info[3].As<Napi::Number>().DoubleValue();
  double beta1 = info[4].As<Napi::Number>().DoubleValue();
  int h = info[5].As<Napi::Number>().Int32Value();

  std::vector<double> forecast(h);
  fortran_garch11_forecast(lastEps2, lastSigma2, omega, alpha1, beta1,
                             forecast.data(), &h);

  Napi::Array jsResult = Napi::Array::New(env, h);
  for (int i = 0; i < h; i++) {
    jsResult.Set(static_cast<uint32_t>(i), Napi::Number::New(env, forecast[i]));
  }
  return jsResult;
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

  // Matrix decompositions (LAPACK-backed)
  exports.Set("lu", Napi::Function::New(env, LU));
  exports.Set("qr", Napi::Function::New(env, QR));
  exports.Set("cholesky", Napi::Function::New(env, Cholesky));
  exports.Set("svd", Napi::Function::New(env, SVD));

  // Statistics (Fortran-accelerated)
  exports.Set("pairwiseEuclidean", Napi::Function::New(env, PairwiseEuclidean));
  exports.Set("gaussianPdfBatch", Napi::Function::New(env, GaussianPdfBatch));
  exports.Set("kdeGaussian", Napi::Function::New(env, KdeGaussian));
  exports.Set("weightedCrossProducts", Napi::Function::New(env, WeightedCrossProducts));
  exports.Set("welfordBatch", Napi::Function::New(env, WelfordBatch));
  exports.Set("uniformSampleBatch", Napi::Function::New(env, UniformSampleBatch));
  exports.Set("normalSampleBatch", Napi::Function::New(env, NormalSampleBatch));
  exports.Set("gammaSampleBatch", Napi::Function::New(env, GammaSampleBatch));
  exports.Set("betaSampleBatch", Napi::Function::New(env, BetaSampleBatch));

  // Feature selection (Fortran-accelerated)
  exports.Set("elasticNetCd", Napi::Function::New(env, ElasticNetCd));
  exports.Set("ridgeSolve", Napi::Function::New(env, RidgeSolve));

  // Time series (Fortran-accelerated)
  exports.Set("acf", Napi::Function::New(env, Acf));
  exports.Set("pacfDurbinLevinson", Napi::Function::New(env, PacfDurbinLevinson));
  exports.Set("exponentialSmoothing", Napi::Function::New(env, ExponentialSmoothing));
  exports.Set("holtWinters", Napi::Function::New(env, HoltWinters));
  exports.Set("difference", Napi::Function::New(env, Difference));

  // Kalman filter (Fortran-accelerated)
  exports.Set("kalmanFilterUnivariate", Napi::Function::New(env, KalmanFilterUnivariate));

  // Distributions (Fortran-accelerated)
  exports.Set("chi2Cdf", Napi::Function::New(env, Chi2Cdf));
  exports.Set("chi2Pdf", Napi::Function::New(env, Chi2Pdf));
  exports.Set("tCdf", Napi::Function::New(env, TCdf));
  exports.Set("tPdf", Napi::Function::New(env, TPdf));
  exports.Set("fCdf", Napi::Function::New(env, FCdf));
  exports.Set("fPdf", Napi::Function::New(env, FPdf));
  exports.Set("normalPdf", Napi::Function::New(env, NormalPdf));
  exports.Set("gammaCdf", Napi::Function::New(env, GammaCdf));
  exports.Set("betaCdf", Napi::Function::New(env, BetaCdf));
  exports.Set("chi2CdfBatch", Napi::Function::New(env, Chi2CdfBatch));
  exports.Set("tCdfBatch", Napi::Function::New(env, TCdfBatch));
  exports.Set("normalCdfBatch", Napi::Function::New(env, NormalCdfBatch));

  // GARCH/optimization (Fortran-accelerated)
  exports.Set("garch11Loglik", Napi::Function::New(env, Garch11Loglik));
  exports.Set("garchPqLoglik", Napi::Function::New(env, GarchPqLoglik));
  exports.Set("gjrGarch11Loglik", Napi::Function::New(env, GjrGarch11Loglik));
  exports.Set("egarch11Loglik", Napi::Function::New(env, Egarch11Loglik));
  exports.Set("garch11Forecast", Napi::Function::New(env, Garch11Forecast));

  return exports;
}

NODE_API_MODULE(fortran_special, Init)
