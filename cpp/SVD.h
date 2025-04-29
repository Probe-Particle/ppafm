#ifndef SVD_H
#define SVD_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h> // For memcpy

// --- Optimized Helper functions for matrix/vector operations (Plain C, Pointer Arithmetic) ---

// Calculate dot product of two columns of a matrix using pointer arithmetic.
// Cost: O(n)
static inline double dot_product_cols(const double* matrix, int col1_idx, int col2_idx, int n) {
    double sum = 0.0;
    // Start pointers at the first element of each column
    const double *ptr1 = matrix + col1_idx;
    const double *ptr2 = matrix + col2_idx;

    for (int k = 0; k < n; ++k) {
        sum += (*ptr1) * (*ptr2);
        // Move pointers to the next row (jump by n elements)
        ptr1 += n;
        ptr2 += n;
    }
    return sum;
}

// Calculate squared Euclidean norm of a column using pointer arithmetic.
// Cost: O(n)
static inline double column_norm_sq(const double* matrix, int col_idx, int n) {
    double sum = 0.0;
    // Start pointer at the first element of the column
    const double *ptr = matrix + col_idx;

    for (int k = 0; k < n; ++k) {
        double val = *ptr;
        sum += val * val;
        // Move pointer to the next row
        ptr += n;
    }
    return sum;
}

// Apply a 2x2 rotation [c -s; s c] to columns i and j of a matrix, in-place,
// using pointer arithmetic.
// Cost: O(n)
static inline void apply_col_rotation(double* matrix, int i, int j, double c, double s, int n) {
    // Start pointers at the first element of each column
    double *ptr_i = matrix + i;
    double *ptr_j = matrix + j;

    for (int k = 0; k < n; ++k) {
        double val_i_orig = *ptr_i; // Get original value of matrix[k][i]
        double val_j_orig = *ptr_j; // Get original value of matrix[k][j]

        // Apply rotation (use original values for both updates in this row k)
        *ptr_i = c * val_i_orig - s * val_j_orig; // Update matrix[k][i]
        *ptr_j = s * val_i_orig + c * val_j_orig; // Update matrix[k][j]

        // Move pointers to the next row
        ptr_i += n;
        ptr_j += n;
    }
}


// --- Jacobi Sweep Function ---

// Performs one sweep of the One-Sided Jacobi method on columns i and j.
// Iterates through all column pairs (i, j) with i < j.
// A and V are modified in-place.
// Returns the sum of squared off-diagonal elements related to rotated pairs (can be used for convergence check).
static double jacobi_sweep(int n, double* A, double* V, double svd_tolerance) {
    double off_diag_sq_sum = 0.0;

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {

            // Calculate dot product (p), squared norms (q, r) of columns i and j of A
            double p = dot_product_cols(A, i, j, n); // col_i . col_j
            double q = column_norm_sq(A, i, n);     // ||col_i||^2
            double r = column_norm_sq(A, j, n);     // ||col_j||^2

            // Add squared dot product to sum (measure of non-orthogonality)
            // For a full convergence check, one would sum squares of all off-diagonal
            // elements of A^T * A, which is related to p^2. Here, we just track progress.
            off_diag_sq_sum += p * p;


            // Check if columns are already orthogonal enough (using relative tolerance)
            // Condition abs(p) < tol * sqrt(q*r) checks if columns are nearly orthogonal.
            if (fabs(p) < svd_tolerance * sqrt(q * r)) {
                 continue; // Skip rotation if columns are nearly orthogonal
            }

            // Calculate rotation angle phi such that columns i and j of A become orthogonal
            // tan(2*phi) = 2*p / (r - q)
            double phi = 0.5 * atan2(2.0 * p, r - q);
            double c = cos(phi);
            double s = sin(phi);

            // Apply rotation [c -s; s c] to columns i and j of A (in-place)
            apply_col_rotation(A, i, j, c, s, n);

            // Accumulate rotation into V (V = V * R_ij)
            // Note: We are applying the rotation to the *columns* of V.
            apply_col_rotation(V, i, j, c, s, n);
        }
    }
    return off_diag_sq_sum;
}


// --- SVD Implementation (One-Sided Jacobi - In-Place Optimized) ---

// Performs SVD on matrix A (n x n) using One-Sided Jacobi.
// A is MODIFIED in place.
// Stores results in U (n x n), sigma (n), V (n x n).
// sigma will contain singular values sorted in descending order.
// U and V columns are sorted accordingly.
// U, sigma, V must be pre-allocated.
// num_sweeps: Number of Jacobi sweeps.
// svd_tolerance: Tolerance for considering singular values as zero. Also used in sweep orthogonality check.
// Returns 0 on success, -1 on error.
int svd_jacobi_one_sided_inplace(int n, double* A, double* U, double* sigma, double* V, int num_sweeps, double svd_tolerance) {

    // Initialize V as identity matrix using pointer arithmetic
    for (int i = 0; i < n; ++i) {
        double *row_ptr_V = V + i * n; // Pointer to start of row i in V
        for (int j = 0; j < n; ++j) {
            row_ptr_V[j] = (i == j) ? 1.0 : 0.0; // V[i][j]
        }
    }

    // Perform Jacobi sweeps
    for (int sweep = 0; sweep < num_sweeps; ++sweep) {
        // We could check the return value (off_diag_sq_sum) for convergence
        // and break early if it's below a certain threshold, but for a fixed N=8,
        // a fixed number of sweeps (like 50) is often sufficient and simpler.
        jacobi_sweep(n, A, V, svd_tolerance);
    }

    // After sweeps, A has orthogonal columns.
    // Column norms are singular values. Normalized columns are U.

    // Calculate singular values and U using pointer arithmetic
    for (int i = 0; i < n; ++i) { // Iterate through columns of A
        double norm_sq = 0.0;
        const double *col_ptr_A = A + i; // Pointer to start of column i in A

        // Calculate squared norm of column i in A
        for(int k = 0; k < n; ++k) {
            double val = *col_ptr_A; // A[k][i]
            norm_sq += val * val;
            col_ptr_A += n; // Move to A[k+1][i]
        }

        double s_val = sqrt(norm_sq);
        sigma[i] = s_val;

        // Calculate column i of U (normalized column i of A)
        double *col_ptr_U = U + i; // Pointer to start of column i in U
        col_ptr_A = A + i;         // Reset pointer for column i in A

        if (s_val > svd_tolerance) {
            // Normalize the column from A to get the corresponding column of U
            for (int k = 0; k < n; ++k) {
                *col_ptr_U = (*col_ptr_A) / s_val; // U[k][i] = A[k][i] / s_val
                col_ptr_U += n;
                col_ptr_A += n;
            }
        } else {
            // Singular value is zero. Column of U is in the left null space. Set to zero.
            for (int k = 0; k < n; ++k) {
                *col_ptr_U = 0.0; // U[k][i] = 0.0
                col_ptr_U += n;
            }
        }
    }

    // Optional: Sort singular values and corresponding columns of U and V
    // We'll use a temporary buffer to swap columns efficiently
    double* temp_col = (double*)malloc(n * sizeof(double));
    if (temp_col == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for temp_col in SVD sorting.\n");
        // This is non-fatal for the SVD core, but sorting won't happen.
        // Depending on strictness, you might return an error here.
        // For now, just print error and continue without sorting.
    } else {
        for (int i = 0; i < n - 1; ++i) {
            int max_idx = i;
            for (int j = i + 1; j < n; ++j) {
                if (sigma[j] > sigma[max_idx]) {
                    max_idx = j;
                }
            }
            if (max_idx != i) {
                // Swap sigma values
                double temp_s = sigma[i];
                sigma[i] = sigma[max_idx];
                sigma[max_idx] = temp_s;

                // Swap columns i and max_idx in U using the buffer
                double *col_i_U = U + i;
                double *col_max_U = U + max_idx;
                for(int k=0; k<n; ++k) { temp_col[k] = *col_i_U; col_i_U += n; }
                col_i_U = U + i; // Reset pointer
                for(int k=0; k<n; ++k) { *col_i_U = *col_max_U; col_i_U += n; col_max_U += n; }
                col_max_U = U + max_idx; // Reset pointer
                for(int k=0; k<n; ++k) { *col_max_U = temp_col[k]; col_max_U += n; }


                // Swap columns i and max_idx in V using the buffer
                double *col_i_V = V + i;
                double *col_max_V = V + max_idx;
                 for(int k=0; k<n; ++k) { temp_col[k] = *col_i_V; col_i_V += n; }
                col_i_V = V + i; // Reset pointer
                for(int k=0; k<n; ++k) { *col_i_V = *col_max_V; col_i_V += n; col_max_V += n; }
                col_max_V = V + max_idx; // Reset pointer
                for(int k=0; k<n; ++k) { *col_max_V = temp_col[k]; col_max_V += n; }
            }
        }
        free(temp_col);
    }

    return 0; // Indicate success
}

// --- Least Squares Solver using SVD ---

// Solves Ax = b in the least squares sense using SVD.
// A (n x n), b (n), x (n). x will store the solution.
// svd_tolerance: Tolerance for considering singular values as zero.
int solve_least_squares_svd(int n, const double* A, const double* b, double* x, double svd_tolerance, int num_sweeps, bool bHeapAlloc) {

    int n_= bHeapAlloc ? 0 : n;
    double U_     [n*n];
    double V_     [n*n];
    double sigma_ [n  ];
    double y_     [n  ];
    double  *U, *sigma, *V, *y;
    if(bHeapAlloc) {
        U     = (double*)malloc(n * n * sizeof(double));
        sigma = (double*)malloc(n     * sizeof(double));
        V     = (double*)malloc(n * n * sizeof(double));
        y     = (double*)malloc(n     * sizeof(double)); // Intermediate vector U_T * b
        if (U == NULL || sigma == NULL || V == NULL || y == NULL) {
            fprintf(stderr, "EROOR in solve_least_squares_svd(): Memory allocation failed\n");
            free(U); free(sigma); free(V); free(y);
            return -1; // Indicate failure
        }
    }else{
        U     = U_;
        V     = V_;
        sigma = sigma_;
        y     = y_;
    }

    // 1. Compute SVD of A: copy matrix and run one-sided Jacobi SVD
    double A_copy_[n*n];
    memcpy(A_copy_, A, n * n * sizeof(double));
    int svd_status = svd_jacobi_one_sided_inplace(n, A_copy_, U, sigma, V, num_sweeps, svd_tolerance);
    if (svd_status != 0) {
        fprintf(stderr, "ERROR in solve_least_squares_svd(): svd_jacobi_one_sided_inplace() failed.\n");
        if (bHeapAlloc) { free(U); free(sigma); free(V); free(y); }
        return -1;
    }

    // 2. Compute y = U_T * b
    // y_i = dot_product(column i of U, b)
    for (int i = 0; i < n; ++i) { // y[i] calculation (corresponds to column i of U)
        y[i] = 0.0;
        const double *u_col_ptr = U + i; // Pointer to start of column i in U
        // Dot product of U[:, i] and b[:]
        for (int k = 0; k < n; ++k) { // Iterate down column i of U
            y[i] += (*u_col_ptr) * b[k];
            u_col_ptr += n; // Move to U[k+1][i]
        }
    }

    // 3. Compute x = V * Sigma_plus * y
    // Sigma_plus is the pseudoinverse of Sigma.
    // Sigma_plus is diagonal with elements 1/sigma_i if sigma_i > tolerance, else 0.
    // x = sum_j (V[:, j] * y_j / sigma_j) for sigma_j > tolerance
    // More efficiently: x_i = sum_j (V[i][j] * (Sigma_plus * y)_j)
    for (int i = 0; i < n; ++i) { // x[i] calculation (corresponds to row i of V)
        x[i] = 0.0;
        const double *v_row_ptr = V + i * n; // Pointer to the start of row i in V
        // Iterate across row i of V and corresponding elements of Sigma_plus * y
        for (int j = 0; j < n; ++j) {
            double sigma_plus_y_j = 0.0;
            if (sigma[j] > svd_tolerance) {
                 sigma_plus_y_j = y[j] / sigma[j];
            }
            x[i] += (*v_row_ptr) * sigma_plus_y_j; // V[i][j] * (Sigma_plus * y)[j]
            v_row_ptr++; // Move to the next element in the row V[i][j+1]
        }
    }

    if(bHeapAlloc) {
        free(U);
        free(sigma);
        free(V);
        free(y);
    }

    return 0; // Indicate success
}


#endif // SVD_H
