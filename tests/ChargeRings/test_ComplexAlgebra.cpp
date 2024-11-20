#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "ComplexAlgebra.hpp"

void print_matrix(int rows, int cols, Vec2d* A, const char* name=nullptr) {
    if(name) printf("%s:\n", name);
    for(int i=0; i<rows; i++) {
        for(int j=0; j<cols; j++) {
            printf("(%6.3f,%6.3f) ", A[i*cols + j].x, A[i*cols + j].y);
        }
        printf("\n");
    }
    printf("\n");
}

// Test matrix inversion
void test_random_matrix(int n, double perturbation) {
    printf("\nTesting matrix inversion (n=%d, perturbation=%.3f ):\n",  n, perturbation );
    
    // Allocate matrices
    Vec2d* A         = new Vec2d[n*n];
    Vec2d* Ainv      = new Vec2d[n*n];
    Vec2d* I         = new Vec2d[n*n];
    Vec2d* workspace = new Vec2d[2*n*n];  // For augmented matrix [A|I]
    
    // Initialize A = I + a*R (well-conditioned matrix)
    for(int i=0; i<n; i++) {
        for(int j=0; j<n; j++) {
            double re = (i == j) ? 1.0 : 0.0;
            double im = 0.0;
            if(i == j) {
                re += perturbation * (2.0 * rand()/RAND_MAX - 1.0);
                im += perturbation * (2.0 * rand()/RAND_MAX - 1.0);
            } else {
                re = perturbation * (2.0 * rand()/RAND_MAX - 1.0);
                im = perturbation * (2.0 * rand()/RAND_MAX - 1.0);
            }
            A[i*n + j] = {re, im};
        }
    }
    
    print_matrix(n, n, A, "Original matrix A");
    
    // Invert matrix
    invert_complex_matrix(n, A, Ainv, workspace);
    print_matrix(n, n, Ainv, "Inverse matrix A^(-1)");
    
    // Verify by multiplication
    multiply_complex_matrices(n, n, n, A, Ainv, I);  // A(n×n) * Ainv(n×n) = I(n×n)
    print_matrix(n, n, I, "Verification A*A^(-1) (should be identity)");
    
    // Calculate maximum deviation from identity
    double max_error = 0.0;
    for(int i=0; i<n; i++) {
        for(int j=0; j<n; j++) {
            double expected = (i == j) ? 1.0 : 0.0;
            // Calculate absolute difference for both real and imaginary parts
            double real_diff = fabs(I[i*n + j].x - expected);
            double imag_diff = fabs(I[i*n + j].y);
            double error = sqrt(real_diff*real_diff + imag_diff*imag_diff);
            if(error > max_error) max_error = error;
        }
    }
    printf("Maximum deviation from identity: %.2e\n\n", max_error);
    
    // Clean up
    delete[] A;
    delete[] Ainv;
    delete[] I;
    delete[] workspace;
}

// Test solving system of equations
void test_solve_system(int n, int m, double perturbation) {
    printf("\nTesting system solver (n=%d, m=%d, perturbation=%.3f):\n", n, m, perturbation);
    
    // Allocate matrices
    Vec2d* A = new Vec2d[n*n];     // System matrix
    Vec2d* X = new Vec2d[n*m];     // Solution matrix
    Vec2d* B = new Vec2d[n*m];     // Right-hand side
    Vec2d* Bcheck = new Vec2d[n*m]; // For verification
    Vec2d* workspace = new Vec2d[n*(n+m)];  // For augmented matrix [A|B]
    
    // Initialize A = I + a*R (well-conditioned matrix)
    for(int i=0; i<n; i++) {
        for(int j=0; j<n; j++) {
            double re = (i == j) ? 1.0 : 0.0;
            double im = 0.0;
            if(i == j) {
                re += perturbation * (2.0 * rand()/RAND_MAX - 1.0);
                im += perturbation * (2.0 * rand()/RAND_MAX - 1.0);
            } else {
                re = perturbation * (2.0 * rand()/RAND_MAX - 1.0);
                im = perturbation * (2.0 * rand()/RAND_MAX - 1.0);
            }
            A[i*n + j] = {re, im};
        }
    }
    
    // Initialize X with some known values
    for(int i=0; i<n; i++) {
        for(int j=0; j<m; j++) {
            X[i*m + j] = {(double)(i+1), (double)(j+1)};
        }
    }
    
    // Compute B = A*X
    multiply_complex_matrices(n, n, m, A, X, B);
    
    print_matrix(n, n, A, "System matrix A");
    print_matrix(n, m, X, "Original X");
    print_matrix(n, m, B, "Right-hand side B = A*X");
    
    // Solve system AX = B
    Vec2d* X_solved = new Vec2d[n*m];
    solve_complex_system(n, m, A, B, X_solved, workspace);
    print_matrix(n, m, X_solved, "Solved X");
    
    // Verify solution by computing A*X_solved
    multiply_complex_matrices(n, n, m, A, X_solved, Bcheck);  // A(n×n) * X(n×m) = Bcheck(n×m)
    print_matrix(n, m, Bcheck, "Verification B' = A*X_solved");
    
    // Calculate maximum deviation from original B
    double max_error = 0.0;
    for(int i=0; i<n; i++) {
        for(int j=0; j<m; j++) {
            // Calculate absolute difference for both real and imaginary parts
            double real_diff = fabs(Bcheck[i*m + j].x - B[i*m + j].x);
            double imag_diff = fabs(Bcheck[i*m + j].y - B[i*m + j].y);
            double error = sqrt(real_diff*real_diff + imag_diff*imag_diff);
            if(error > max_error) max_error = error;
        }
    }
    printf("Maximum deviation from original B: %.2e\n\n", max_error);
    
    // Clean up all allocated memory
    delete[] A;
    delete[] X;
    delete[] B;
    delete[] Bcheck;
    delete[] X_solved;
    delete[] workspace;
}

int main() {
    srand(time(NULL));

    bool b_do_partial_pivot = false;
    bool b_do_system_solver = true;
    bool b_do_inverse       = false;
    bool b_do_3x3           = true;
    bool b_do_5x5           = true;
    
    // Test matrix inversion
    if(b_do_inverse) {
        if(b_do_3x3) test_random_matrix(3, 0.1);  // Same matrix with partial pivot
        if(b_do_5x5) test_random_matrix(5, 0.1);  // Same matrix with partial pivot
    }

    // Test system solver
    if(b_do_system_solver) {
        if(b_do_3x3) test_solve_system(3, 2, 0.1);  // Same with partial pivot
        if(b_do_5x5) test_solve_system(5, 3, 0.1); // Same with partial pivot
    }

    // Clean up any remaining workspace memory
    return 0;
}
