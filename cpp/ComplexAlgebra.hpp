#ifndef COMPLEX_ALGEBRA_H
#define COMPLEX_ALGEBRA_H

#include <cstdlib>
#include <stdio.h>
#include <cstdio>

#include "Vec2.h"
#include <math.h>

// Complex matrix multiplication C = A*B
// A is n×k matrix, B is k×m matrix, C is n×m matrix
inline void multiply_complex_matrices(int n, int k, int m, Vec2d* A, Vec2d* B, Vec2d* C) {
    //printf("\nMatrix multiplication debug:\n");
    //printf("Matrix A (%dx%d):\n", n, k);
    // for(int i=0; i<n; i++) {
    //     for(int j=0; j<k; j++) {
    //         printf("(%g,%g) ", A[i*k + j].x, A[i*k + j].y);
    //     }
    //     printf("\n");
    // }
    
    //printf("\nMatrix B (%dx%d):\n", k, m);
    // for(int i=0; i<k; i++) {
    //     for(int j=0; j<m; j++) {
    //         printf("(%g,%g) ", B[i*m + j].x, B[i*m + j].y);
    //     }
    //     printf("\n");
    // }
    
    for(int i=0; i<n; i++) {
        for(int j=0; j<m; j++) {
            Vec2d sum = {0.0, 0.0};
            //printf("\nCalculating C[%d,%d]:\n", i, j);
            for(int p=0; p<k; p++) {
                Vec2d prod;
                prod.set_mul_cmplx(A[i*k + p], B[p*m + j]);
                //printf("  A[%d,%d] * B[%d,%d] = (%g,%g) * (%g,%g) = (%g,%g)\n",  i, p, p, j,  A[i*k + p].x, A[i*k + p].y,  B[p*m + j].x, B[p*m + j].y,   prod.x, prod.y);
                sum = sum + prod;
                //printf("  Running sum = (%g,%g)\n", sum.x, sum.y);
            }
            C[i*m + j] = sum;
        }
    }
    
    //printf("\nResult matrix C (%dx%d):\n", n, m);
    for(int i=0; i<n; i++) {
        for(int j=0; j<m; j++) {
            printf("(%g,%g) ", C[i*m + j].x, C[i*m + j].y);
        }
        printf("\n");
    }
}

// Wrapper for square matrix multiplication (for backward compatibility)
inline void multiply_complex_matrices(int n, Vec2d* A, Vec2d* B, Vec2d* C) {
    multiply_complex_matrices(n, n, n, A, B, C);
}

// Find pivot using partial pivoting (rows only)
inline void find_pivot_partial(int n, int i, Vec2d* A, int stride, int& pivot_row, int& pivot_col) {
    pivot_row = i;
    pivot_col = i;
    //double max_val = complex_magnitude(A[i*stride + i]);
    double max_val = A[i*stride + i].norm();
    
    for(int r = i; r < n; r++) {
        //double val = complex_magnitude(A[r*stride + i]);
        double val = A[r*stride + i].norm();
        if(val > max_val) {
            max_val = val;
            pivot_row = r;
        }
    }
}

// Swap rows in augmented matrix
inline void swap_rows(int n, int row1, int row2, Vec2d* A, int stride) {
    for(int j = 0; j < stride; j++) {
        Vec2d temp = A[row1*stride + j];
        A[row1*stride + j] = A[row2*stride + j];
        A[row2*stride + j] = temp;
    }
}

// Swap columns in left part and corresponding rows in right part
inline void swap_columns_and_rows(int n, int col1, int col2, Vec2d* A, int stride) {
    // Swap columns in left half (first n columns)
    for(int i = 0; i < n; i++) {
        Vec2d temp = A[i*stride + col1];
        A[i*stride + col1] = A[i*stride + col2];
        A[i*stride + col2] = temp;
    }
}

// Perform Gauss-Jordan elimination on augmented matrix [A|B]
// A is n×n matrix, B is n×m matrix (augmented part)
// Result will be [I|A^(-1)B]
inline void gauss_jordan_eliminate(int n, int m, Vec2d* aug) {
    int stride = n + m;
    
    for(int i = 0; i < n; i++) {
        // Find pivot
        int pivot_row, pivot_col;
        find_pivot_partial(n, i, aug, stride, pivot_row, pivot_col);
        
        // Swap rows if needed
        if(pivot_row != i) {
            swap_rows(n, i, pivot_row, aug, stride);
        }
                
        // Scale pivot row
        Vec2d pivot = aug[i*stride + i];
        if(pivot.norm() < 1e-10) {
            printf("Error: Near-zero pivot encountered (norm = %g)\n", pivot.norm());
            // Set all remaining elements to zero to indicate singular matrix
            for(int r = i; r < n; r++) {
                for(int c = 0; c < stride; c++) {
                    aug[r*stride + c] = Vec2d{0.0, 0.0};
                }
            }
            return;  // Exit early
        }
        
        // Compute 1/pivot
        double denom = pivot.x*pivot.x + pivot.y*pivot.y;
        Vec2d pivot_inv = {pivot.x/denom, -pivot.y/denom};
        
        for(int j = 0; j < stride; j++) {
            Vec2d prod;
            prod.set_mul_cmplx(aug[i*stride + j], pivot_inv);
            aug[i*stride + j] = prod;
        }
        
        // Eliminate column
        for(int j = 0; j < n; j++) {
            if(j != i) {
                Vec2d factor = aug[j*stride + i];
                for(int k = 0; k < stride; k++) {
                    Vec2d prod;
                    prod.set_mul_cmplx(factor, aug[i*stride + k]);
                    aug[j*stride + k] = aug[j*stride + k] - prod;
                }
            }
        }
    }
}

// Invert matrix using Gauss-Jordan elimination
// workspace should be pre-allocated with size 2*n*n for augmented matrix [A|I]
inline void invert_complex_matrix(int n, Vec2d* A, Vec2d* Ainv, Vec2d* workspace) {
    Vec2d* aug = workspace;  // Use pre-allocated workspace for [A|I]
    
    // Initialize augmented matrix [A|I]
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            aug[i*(2*n) + j] = A[i*n + j];
            aug[i*(2*n) + (j+n)] = {(i == j) ? 1.0 : 0.0, 0.0};
        }
    }
    
    // Perform Gauss-Jordan elimination
    gauss_jordan_eliminate(n, n, aug);
    
    // For partial pivoting, just copy directly
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            Ainv[i*n + j] = aug[i*(2*n) + (j+n)];
        }
    }
}

// Solve system of linear equations AX = B
// A is n×n matrix, B is n×m matrix (m systems to solve simultaneously)
// X will contain the solution
inline void solve_complex_system(int n, int m, Vec2d* A, Vec2d* B, Vec2d* X, Vec2d* workspace) {
    Vec2d* aug = workspace;  // Use pre-allocated workspace for [A|B]

    // Initialize augmented matrix [A|B]
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            aug[i*(n+m) + j] = A[i*n + j];
        }
        for(int j = 0; j < m; j++) {
            aug[i*(n+m) + (j+n)] = B[i*m + j];
        }
    }
    
    // Perform Gauss-Jordan elimination
    gauss_jordan_eliminate(n, m, aug);
    
    // For partial pivoting, just copy directly
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < m; j++) {
            X[i*m + j] = aug[i*(n+m) + (j+n)];
        }
    }
}

void save_matrix_to_file(const char* filename, const char* title, Vec2d* matrix, int rows, int cols) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        printf("Error: Could not open file %s for writing\n", filename);
        return;
    }
    fprintf(f, "%s\n", title);
    fprintf(f, "Dimensions: %d %d\n", rows, cols);
    fprintf(f, "Format: (real,imag)\n");
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            Vec2d val = matrix[i * cols + j];
            fprintf(f, "(%e,%e) ", val.x, val.y);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

#endif // COMPLEX_ALGEBRA_H
