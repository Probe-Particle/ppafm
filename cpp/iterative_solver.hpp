#ifndef ITERATIVE_SOLVER_HPP
#define ITERATIVE_SOLVER_HPP

#include <cstdio>
#include <cmath>

static void solve_Jacobi(const double* A, const double* b, double* x, int n,  double tol = 1e-10, int max_iter = 1000) {
    // Initial guess
    //std::fill(x, x + n, 0.0);
    //x[0] = 1.0;  // First element is 1.0 as per Pauli equation requirements
    
    double* x_new = new double[n];
    
    for(int iter = 0; iter < max_iter; iter++) {
        // Save current x for convergence check
        std::copy(x, x + n, x_new);
        
        // Jacobi iteration
        //double sumA
        for(int i = 0; i < n; i++) {  
            double sum = b[i];
            for(int j = 0; j < n; j++) {
                if(j != i) {
                    sum -= A[i*n + j] * x[j];
                }
            }
            x_new[i] = sum / A[i*n + i];
        }
        
        // Enforce first row normalization condition
        //x_new[0] = 1.0 - std::accumulate(x_new + 1, x_new + n, 0.0);
        
        // Check convergence
        double diff = 0.0;
        for(int i = 0; i < n; i++) {
            double d = x_new[i] - x[i];
            diff += d*d;
        }
        
        std::copy(x_new, x_new + n, x);
        
        if(diff < tol*tol) { 
            delete[] x_new;
            return;
        }else{
            printf("Iteration %d: diff = %f\n", iter, sqrt(diff) );
        }
    }
    delete[] x_new;
}

#endif // ITERATIVE_SOLVER_HPP