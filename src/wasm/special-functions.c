/**
 * Special mathematical functions for WASM compilation.
 *
 * Compiled with Emscripten (emcc) to produce a WebAssembly module.
 * These functions provide near-native performance for hot-path
 * statistical computations.
 *
 * Build: see scripts/build-wasm.sh
 */

#include <math.h>

/* ========================================================================
 * Lanczos approximation coefficients (g = 7, n = 9)
 * Same coefficients used in the TypeScript fallback for consistency.
 * ======================================================================== */

static const double LANCZOS_G = 7.0;
static const int LANCZOS_N = 9;
static const double LANCZOS_COEFF[9] = {
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7
};

/* ========================================================================
 * gammaLn — Log-gamma function using Lanczos approximation
 * ======================================================================== */

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((used))
double gammaLn(double x) {
    if (x <= 0.0 && x == floor(x)) {
        return NAN;
    }

    if (x < 0.5) {
        /* Reflection formula: Gamma(x) * Gamma(1-x) = pi / sin(pi*x) */
        return log(M_PI / sin(M_PI * x)) - gammaLn(1.0 - x);
    }

    x -= 1.0;
    double a = LANCZOS_COEFF[0];
    for (int i = 1; i < LANCZOS_N; i++) {
        a += LANCZOS_COEFF[i] / (x + (double)i);
    }

    double t = x + LANCZOS_G + 0.5;
    return 0.5 * log(2.0 * M_PI) + (x + 0.5) * log(t) - t + log(a);
}

/* ========================================================================
 * gamma — Gamma function
 * ======================================================================== */

__attribute__((used))
double gamma_fn(double x) {
    return exp(gammaLn(x));
}

/* ========================================================================
 * erf — Error function using Abramowitz & Stegun approximation (7.1.26)
 *
 * Maximum error: |epsilon(x)| <= 1.5e-7
 * ======================================================================== */

__attribute__((used))
double erf_fn(double x) {
    double sign = (x >= 0.0) ? 1.0 : -1.0;
    x = fabs(x);

    double t = 1.0 / (1.0 + 0.3275911 * x);
    double poly = t * (0.254829592
        + t * (-0.284496736
        + t * (1.421413741
        + t * (-1.453152027
        + t * 1.061405429))));

    return sign * (1.0 - poly * exp(-x * x));
}

/* ========================================================================
 * erfc — Complementary error function
 * ======================================================================== */

__attribute__((used))
double erfc_fn(double x) {
    return 1.0 - erf_fn(x);
}

/* ========================================================================
 * betaFn — Beta function: B(a,b) = Gamma(a)*Gamma(b) / Gamma(a+b)
 * ======================================================================== */

__attribute__((used))
double betaFn(double a, double b) {
    return exp(gammaLn(a) + gammaLn(b) - gammaLn(a + b));
}

#ifdef __cplusplus
}
#endif
