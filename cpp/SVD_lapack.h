#ifndef SVD_LAPACK_H
#define SVD_LAPACK_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h> // For memcpy

// Include LAPACKE header. You need to have LAPACKE and LAPACK/BLAS libraries installed.
// The path might vary depending on your system.
#include <lapacke.h>

// --- Least Squares Solver using SVD from LAPACKE ---

// Solves Ax = b in the least squares sense using SVD via LAPACKE.
// A (n x n), b (n), x (n). x will store the solution.
// A is NOT modified. Uses LAPACK_ROW_MAJOR.
int solve_least_squares_svd_lapacke(int n, const double* A, const double* b, double* x, double svd_tolerance, bool bHeapAlloc=false) {
    // Prepare stack vs heap storage
    int n_ = bHeapAlloc ? 0 : n;
    double A_copy_stack[n*n];
    double U_stack[n*n];
    double sigma_stack[n];
    double VT_stack[n*n]; // holds V^T
    double y_stack[n];
    int superb_sz = (n > 1 ? n-1 : 1);
    double superb_stack[superb_sz];
    double *A_copy, *U, *sigma, *VT, *y;
    double *superb;
    if (bHeapAlloc) {
        A_copy = (double*)malloc(n*n*sizeof(double));
        U      = (double*)malloc(n*n*sizeof(double));
        sigma  = (double*)malloc(n*sizeof(double));
        VT     = (double*)malloc(n*n*sizeof(double));
        y      = (double*)malloc(n*sizeof(double));
        superb = (double*)malloc(superb_sz * sizeof(double));
        if (!A_copy||!U||!sigma||!VT||!y||!superb) {
            fprintf(stderr, "Error: allocation failed in solve_least_squares_svd_lapacke()\n");
            free(A_copy); free(U); free(sigma); free(VT); free(y); free(superb);
            return -2;
        }
    } else {
        A_copy = A_copy_stack;
        U      = U_stack;
        sigma  = sigma_stack;
        VT     = VT_stack;
        y      = y_stack;
        superb = superb_stack;
    }
    // Copy A since LAPACKE overwrites
    memcpy(A_copy, A, n*n*sizeof(double));
    // Compute SVD: A = U * Sigma * VT
    int info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A','A', n, n,
                              A_copy, n, sigma,
                              U, n, VT, n, superb);
    if (info != 0) {
        fprintf(stderr, "Error: dgesvd failed (info=%d)\n", info);
        if (bHeapAlloc){ free(A_copy), free(U), free(sigma), free(VT), free(y), free(superb); }
        return -1;
    }
    // Compute y = U^T * b
    for (int i = 0; i < n; ++i) {
        y[i] = 0.0;
        double *u_col = U + i;
        for (int k = 0; k < n; ++k, u_col += n) {
            y[i] += (*u_col) * b[k];
        }
    }
    // Compute x = V * Sigma_plus * y using VT (V^T)
    for (int i = 0; i < n; ++i) {
        x[i] = 0.0;
        for (int j = 0; j < n; ++j) {
            if (sigma[j] > svd_tolerance) {
                double v_ij = VT[j*n + i]; // V_{i,j} = VT[j][i]
                x[i] += v_ij * (y[j] / sigma[j]);
            }
        }
    }
    if (bHeapAlloc){ free(A_copy), free(U), free(sigma), free(VT), free(y), free(superb); }
    return 0;
}

#endif // SVD_H
