/**
 * Linear algebra operations for WASM compilation.
 *
 * All matrices are stored in row-major order as flat double arrays.
 * Compiled with Emscripten (emcc) to produce a WebAssembly module.
 *
 * Build: see scripts/build-wasm.sh
 */

#include <math.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * matMul — Matrix multiplication: C = A * B
 *
 * A is m x k, B is k x n, C is m x n.
 * All matrices are row-major flat arrays.
 * ======================================================================== */

__attribute__((used))
void matMul(const double *a, const double *b, double *c,
            int m, int k, int n) {
    /* Zero-initialize C */
    memset(c, 0, (size_t)m * (size_t)n * sizeof(double));

    for (int i = 0; i < m; i++) {
        for (int p = 0; p < k; p++) {
            double aip = a[i * k + p];
            if (aip == 0.0) continue;
            for (int j = 0; j < n; j++) {
                c[i * n + j] += aip * b[p * n + j];
            }
        }
    }
}

/* ========================================================================
 * solve — Solve A * x = b using LU decomposition with partial pivoting.
 *
 * A is n x n (row-major), b is length n.
 * The solution is written into x (length n).
 * Returns 0 on success, -1 if the matrix is singular.
 *
 * NOTE: A is modified in-place (overwritten with the LU factors).
 * ======================================================================== */

__attribute__((used))
int solve(double *a, const double *b, double *x, int n) {
    /* Permutation vector */
    int perm[1024]; /* Supports up to 1024x1024 matrices */
    if (n > 1024) return -2;

    for (int i = 0; i < n; i++) perm[i] = i;

    /* LU factorization with partial pivoting */
    for (int col = 0; col < n; col++) {
        /* Find pivot */
        double maxVal = fabs(a[col * n + col]);
        int maxRow = col;
        for (int row = col + 1; row < n; row++) {
            double val = fabs(a[row * n + col]);
            if (val > maxVal) {
                maxVal = val;
                maxRow = row;
            }
        }

        if (maxVal < 1e-15) return -1; /* Singular */

        /* Swap rows */
        if (maxRow != col) {
            for (int j = 0; j < n; j++) {
                double tmp = a[col * n + j];
                a[col * n + j] = a[maxRow * n + j];
                a[maxRow * n + j] = tmp;
            }
            int tmpP = perm[col];
            perm[col] = perm[maxRow];
            perm[maxRow] = tmpP;
        }

        /* Eliminate below */
        for (int row = col + 1; row < n; row++) {
            double factor = a[row * n + col] / a[col * n + col];
            a[row * n + col] = factor; /* Store L factor */
            for (int j = col + 1; j < n; j++) {
                a[row * n + j] -= factor * a[col * n + j];
            }
        }
    }

    /* Apply permutation to b */
    for (int i = 0; i < n; i++) {
        x[i] = b[perm[i]];
    }

    /* Forward substitution: L * y = Pb */
    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            x[i] -= a[i * n + j] * x[j];
        }
    }

    /* Back substitution: U * x = y */
    for (int i = n - 1; i >= 0; i--) {
        for (int j = i + 1; j < n; j++) {
            x[i] -= a[i * n + j] * x[j];
        }
        x[i] /= a[i * n + i];
    }

    return 0;
}

/* ========================================================================
 * cholesky — Cholesky decomposition: A = L * L^T
 *
 * A is n x n symmetric positive-definite (row-major).
 * The lower-triangular factor L is written into l (n x n, row-major).
 * Returns 0 on success, -1 if not positive-definite.
 * ======================================================================== */

__attribute__((used))
int cholesky(const double *a, double *l, int n) {
    memset(l, 0, (size_t)n * (size_t)n * sizeof(double));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0.0;
            for (int k = 0; k < j; k++) {
                sum += l[i * n + k] * l[j * n + k];
            }

            if (i == j) {
                double diag = a[i * n + i] - sum;
                if (diag <= 0.0) return -1; /* Not positive-definite */
                l[i * n + j] = sqrt(diag);
            } else {
                l[i * n + j] = (a[i * n + j] - sum) / l[j * n + j];
            }
        }
    }

    return 0;
}

#ifdef __cplusplus
}
#endif
