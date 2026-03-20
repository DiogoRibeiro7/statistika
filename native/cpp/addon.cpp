#include <napi.h>

// Fortran function declarations (ISO_C_BINDING exports)
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

// --- N-API wrappers ---

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

// --- Module initialization ---

Napi::Object Init(Napi::Env env, Napi::Object exports) {
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
  return exports;
}

NODE_API_MODULE(fortran_special, Init)
