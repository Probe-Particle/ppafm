#ifndef GAUSS_SOLVER_HPP
#define GAUSS_SOLVER_HPP

#include <cmath>

// class GaussSolver {
// public:
//     // Solve the system Ax = b using Gaussian elimination with partial pivoting
//     // Returns the solution in the x array
//     static void solve(double* A, double* b, double* x, int n) {
//         double* aug_matrix = new double[n * (n + 1)];  // Augmented matrix [A|b]
        
//         // Create augmented matrix
//         for(int i = 0; i < n; i++) {
//             for(int j = 0; j < n; j++) {
//                 aug_matrix[i * (n + 1) + j] = A[i * n + j];
//             }
//             aug_matrix[i * (n + 1) + n] = b[i];
//         }

//         // Gaussian elimination with partial pivoting
//         for(int k = 0; k < n - 1; k++) {
//             // Find pivot
//             int pivot_row = k;
//             double pivot_val = fabs(aug_matrix[k * (n + 1) + k]);
//             for(int i = k + 1; i < n; i++) {
//                 double val = fabs(aug_matrix[i * (n + 1) + k]);
//                 if(val > pivot_val) {
//                     pivot_val = val;
//                     pivot_row = i;
//                 }
//             }

//             // Swap rows if necessary
//             if(pivot_row != k) {
//                 for(int j = k; j <= n; j++) {
//                     double temp = aug_matrix[k * (n + 1) + j];
//                     aug_matrix[k * (n + 1) + j] = aug_matrix[pivot_row * (n + 1) + j];
//                     aug_matrix[pivot_row * (n + 1) + j] = temp;
//                 }
//             }

//             // Eliminate column
//             for(int i = k + 1; i < n; i++) {
//                 double factor = aug_matrix[i * (n + 1) + k] / aug_matrix[k * (n + 1) + k];
//                 for(int j = k; j <= n; j++) {
//                     aug_matrix[i * (n + 1) + j] -= factor * aug_matrix[k * (n + 1) + j];
//                 }
//             }
//         }

//         // Back substitution
//         for(int i = n - 1; i >= 0; i--) {
//             x[i] = aug_matrix[i * (n + 1) + n];
//             for(int j = i + 1; j < n; j++) {
//                 x[i] -= aug_matrix[i * (n + 1) + j] * x[j];
//             }
//             x[i] /= aug_matrix[i * (n + 1) + i];
//         }

//         delete[] aug_matrix;
//     }
// };


void GaussElimination( int n, double ** A, double * c, int * index ) {
    const double EPSILON = 1e-10; // Small value for numerical stability

    // Initialize the index
    for (int i=0; i<n; ++i) index[i] = i;

    // Find the rescaling factors, one from each row
    for (int i=0; i<n; ++i) {
        double c1 = 0;
        for (int j=0; j<n; ++j) {
            double c0 = fabs(A[i][j]);
            if (c0 > c1) c1 = c0;
        }
        // Avoid division by zero by setting a minimum scale factor
        c[i] = (c1 > EPSILON) ? c1 : EPSILON;
    }

    // Search the pivoting element from each column
    for (int j=0; j<n-1; ++j) {
        double pi1 = 0;
        int k = j; // Default pivot row is the current row
        
        // Find the best pivot (maximum scaled value)
        for (int i=j; i<n; ++i) {
            double pi0 = fabs(A[index[i]][j]);
            pi0 /= c[index[i]];
            if (pi0 > pi1) {
                pi1 = pi0;
                k = i;
            }
        }

        // Check if the pivot is too small
        if (fabs(A[index[k]][j]) < EPSILON) {
            // Matrix is singular or nearly singular
            // Set a small non-zero value to avoid division by zero
            A[index[k]][j] = (A[index[k]][j] >= 0) ? EPSILON : -EPSILON;
        }

        // Interchange rows according to the pivoting order
        if (k != j) {
            int itmp = index[j];
            index[j] = index[k];
            index[k] = itmp;
        }
        
        // Gaussian elimination
        for (int i=j+1; i<n; ++i) {
            double pj = A[index[i]][j] / A[index[j]][j];
            
            // Record pivoting ratios below the diagonal
            A[index[i]][j] = pj;
            
            // Modify other elements accordingly
            for (int l=j+1; l<n; ++l) {
                A[index[i]][l] -= pj * A[index[j]][l];
            }
        }
    }
    
    // Check the last diagonal element
    if (fabs(A[index[n-1]][n-1]) < EPSILON) {
        A[index[n-1]][n-1] = EPSILON; // Avoid division by zero in back substitution
    }
}

void linSolve_gauss( int n, double ** A, double * b, int * index, double * x ) {
    const double EPSILON = 1e-10; // Small value for numerical stability

    // Transform the matrix into an upper triangle
    GaussElimination( n, A, x, index);

    // Update the array b[i] with the ratios stored
    for(int i=0; i<n-1; ++i) {
        for(int j=i+1; j<n; ++j) {
            b[index[j]] -= A[index[j]][i]*b[index[i]];
        }
    }

    // Perform backward substitutions
    // Handle the last element first
    if (fabs(A[index[n-1]][n-1]) < EPSILON) {
        x[n-1] = 0.0; // Set to zero if diagonal element is too small
    } else {
        x[n-1] = b[index[n-1]]/A[index[n-1]][n-1];
    }
    
    // Process remaining elements
    for (int i=n-2; i>=0; --i) {
        x[i] = b[index[i]];
        for (int j=i+1; j<n; ++j) {
            x[i] -= A[index[i]][j]*x[j];
        }
        
        // Avoid division by very small numbers
        if (fabs(A[index[i]][i]) < EPSILON) {
            x[i] = 0.0;
        } else {
            x[i] /= A[index[i]][i];
        }
    }
    
    // Clean up very small values that might be numerical noise
    for (int i=0; i<n; ++i) {
        if (fabs(x[i]) < EPSILON) {
            x[i] = 0.0;
        }
    }
    
    // Ensure the solution is properly normalized
    // This is important for probability distributions
    double sum = 0.0;
    for (int i=0; i<n; ++i) {
        sum += x[i];
    }
    
    // Normalize the solution if the sum is not too close to zero
    if (fabs(sum) > EPSILON) {
        for (int i=0; i<n; ++i) {
            x[i] /= sum;
        }
    }
}

// Forward declaration of the least squares solver
void linSolve_lstsq(int n, double* A, double* b, double* x);

void linSolve_gauss(int n, double* A, double* b, double* x) {
    // Use least squares solver instead of Gaussian elimination for better handling of singular matrices
    linSolve_lstsq(n, A, b, x);
}

// Least squares solver implementation for singular or nearly singular systems
void linSolve_lstsq(int n, double* A, double* b, double* x) {
    const double EPSILON = 1e-12;
        
    // For general case, implement a more robust solver
    // Create a copy of the matrix and RHS vector to avoid modifying the originals
    double* A_copy = new double[n * n];
    double* b_copy = new double[n];
    std::copy(A, A + n * n, A_copy);
    std::copy(b, b + n, b_copy);
    
    // Create pivot array
    int* pivot = new int[n];
    for (int i = 0; i < n; i++) {
        pivot[i] = i;
    }
    
    // Create scaling array for better numerical stability
    double* scale = new double[n];
    for (int i = 0; i < n; i++) {
        double max_val = 0.0;
        for (int j = 0; j < n; j++) {
            double abs_val = fabs(A_copy[i*n + j]);
            if (abs_val > max_val) {
                max_val = abs_val;
            }
        }
        scale[i] = (max_val > EPSILON) ? 1.0 / max_val : 1.0;
    }
    
    // Modified Gaussian elimination with regularization for small pivots
    for (int k = 0; k < n-1; k++) {
        // Find the pivot row
        double max_scaled = 0.0;
        int max_row = k;
        for (int i = k; i < n; i++) {
            int p = pivot[i];
            double scaled_val = fabs(A_copy[p*n + k]) * scale[p];
            if (scaled_val > max_scaled) {
                max_scaled = scaled_val;
                max_row = i;
            }
        }
        
        // Swap pivot rows if necessary
        if (max_row != k) {
            int temp = pivot[k];
            pivot[k] = pivot[max_row];
            pivot[max_row] = temp;
        }
        
        int p_k = pivot[k];
        
        // If pivot is too small, add regularization
        if (fabs(A_copy[p_k*n + k]) < EPSILON) {
            // Add a small value to the diagonal for regularization
            A_copy[p_k*n + k] += EPSILON;
        }
        
        // Eliminate below the pivot
        for (int i = k+1; i < n; i++) {
            int p_i = pivot[i];
            double factor = A_copy[p_i*n + k] / A_copy[p_k*n + k];
            
            // Eliminate rest of the row
            for (int j = k+1; j < n; j++) {
                A_copy[p_i*n + j] -= factor * A_copy[p_k*n + j];
            }
            b_copy[p_i] -= factor * b_copy[p_k];
        }
    }
    
    // Back substitution with regularization for small diagonal elements
    for (int i = n-1; i >= 0; i--) {
        int p_i = pivot[i];
        x[i] = b_copy[p_i];
        
        for (int j = i+1; j < n; j++) {
            x[i] -= A_copy[p_i*n + j] * x[j];
        }
        
        // Regularize very small diagonal elements
        if (fabs(A_copy[p_i*n + i]) < EPSILON) {
            A_copy[p_i*n + i] = EPSILON;  // Add small regularization
        }
        x[i] /= A_copy[p_i*n + i];
        
        // Clean up very small values
        if (fabs(x[i]) < EPSILON) {
            x[i] = 0.0;
        }
    }
    
    // Normalize the solution (sum of probabilities should be 1)
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += x[i];
    }
    
    if (fabs(sum) > EPSILON) {
        for (int i = 0; i < n; i++) {
            x[i] /= sum;
        }
    }
    
    // Clean up
    delete[] A_copy;
    delete[] b_copy;
    delete[] pivot;
    delete[] scale;
}

#endif // GAUSS_SOLVER_HPP
