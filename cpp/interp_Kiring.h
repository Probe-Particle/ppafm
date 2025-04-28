#ifndef  interp_Kriging_h
#define  interp_Kriging_h

#include <stdio.h>
#include <cmath> // For NAN, isnan, isinf
#include <new>   // For std::nothrow

#include "Vec2.h"
#include "gauss_solver.hpp" // Include the Gaussian solver header

// --- Kriging Functions ---

// Compactly supported C2 Wendland function used as a covariance model C(r)
// C(0) = 1, C(R_basis) = 0
double compact_c2_covariance(double r, double Rcut) {
    if (r < 0) r = -r;
    if (r >= Rcut) return 0.0; // Kernel is 0 beyond R_basis
    double t  = r / Rcut;
    double t1 = 1.0 - t;
    double t2 = t1 * t1;
    double t4 = t2 * t2;
    double out = t4 * (4.0 * t + 1.0); // Returns value in [0, 1]
    printf( "compact_c2_covariance() r: % 16.8f t: % 16.8 t1: % 16.8f t2: % 16.8f t4: % 16.8f out: % 16.8f \n", r, t, t1, t2, t4, out );
    return out; // Returns value in [0, 1]
}

// Variogram gamma(r) = C(0) - C(r)    Assuming C(0) = 1.0 (sill) for the normalized covariance kernel
double compact_c2_variogram(double r, double Rcut) {
    return 1.0 - compact_c2_covariance(r, Rcut);
}

// --- Ordinary Kriging Interpolation ---

// Forward declaration for the linear solver from gauss_solver.hpp
// Assuming void linSolve_gauss(int n, double* A, double* b, double* x);
// void linSolve_gauss(int n, double* A, double* b, double* x); // Already included via header


void interpolate_local_kriging(
    const Vec2d* data_points, const double* data_vals, int ndata,
    const Vec2d* gps, double* out_vals, int ngps,
    double Rcut, int nNeighMax, int* out_neighs, double* out_weights
) {
    printf("interpolate_local_kriging() ndata: %i ngps: %i nNeighMax: %i Rcut: %g \n", ndata, ngps, nNeighMax, Rcut);

    // CORRECTED: Stack allocation sizes must be based on the maximum possible system size
    // Kriging system size is nng + 1
    // Max nng is nNeighMax. Max system size is nNeighMax + 1.
    int max_system_size = nNeighMax + 1;

    // Allocate A for (nNeighMax+1) x (nNeighMax+1)
    double A[max_system_size * max_system_size];
    // Allocate b (RHS) for nNeighMax + 1
    double b[max_system_size];
    // Allocate w (solution: lambdas + mu) for nNeighMax + 1
    double w[max_system_size];

    // neighs array size is just nNeighMax, used by find_neighbors
    int neighs[nNeighMax];


    for (int k = 0; k < ngps; ++k) {
        Vec2d p = gps[k];
        // Assuming find_neighbors puts up to nNeighMax indices into neighs and returns the count
        int nng = find_neighbors(ndata, data_points, p, Rcut, neighs, nNeighMax);

        // --- Handle edge cases based on neighbor count ---

        if (nng == 0) {
            // No neighbors found. Cannot interpolate.
            out_vals[k] = NAN; // Use NAN from math.h
            if (out_neighs) { for(int i=0; i<nNeighMax; ++i) out_neighs[k*nNeighMax+i] = -1; }
            // out_weights could store 0s, but the system wasn't solved, perhaps fill with NAN too? Or specific indicator?
            // Let's match your original idea and fill with 0s if out_weights is provided
            if (out_weights) { for(int i=0; i<nNeighMax; ++i) out_weights[k*nNeighMax+i] = 0.0; }
            // printf("Kriging k: %3i nng:   0 -> NAN\n", k); // Debug print
            continue; // Move to the next query point
        }

        // Special case: Exactly one neighbor. Kriging simplifies to returning the neighbor's value.
        if (nng == 1) {
            int ip = neighs[0];
            out_vals[k] = data_vals[ip];
            if (out_neighs) {
                 out_neighs[k*nNeighMax+0] = ip;
                 for(int i=1; i<nNeighMax; ++i) out_neighs[k*nNeighMax+i] = -1;
            }
            if (out_weights) {
                out_weights[k*nNeighMax+0] = 1.0; // lambda for the single neighbor is 1
                // If out_weights stores lambdas + mu, you'd store 1.0 at index 0 and something else (0.0?) at index 1 for mu
                // But let's stick to storing just lambdas up to nNeighMax for simplicity matching your output array size idea
                for(int i=1; i<nNeighMax; ++i) out_weights[k*nNeighMax+i] = 0.0;
            }
            // printf("Kriging k: %3i nng:   1 -> val=%.4f\n", k, out_vals[k]); // Debug print
            continue; // Move to the next query point
        }


        // --- General case: nng > 1 neighbors ---
        // System size is (nng + 1) x (nng + 1)
        int nsz = nng + 1; // This is the actual size for the system matrix and vectors A, b, w

        // 2. Set up Ordinary Kriging system: [ Gamma  1 ] [ lambda ] = [ gamma_p ]
        //                                    [ 1^T    0 ] [ mu     ]   [ 1       ]

        // Fill the Gamma matrix (top-left nng x nng block of A)
        for (int j=0; j<nng; ++j) {
            int jp = neighs[j];
            for (int i = 0; i < nng; ++i) {
                int ip = neighs[i];
                double r = (data_points[jp] - data_points[ip]).norm();
                // A is stored row-major with 'nsz' columns: A[row * nsz + col]
                A[j * nsz + i] = compact_c2_variogram(r, Rcut);
            }
        }

        // Fill the last column (vector of ones for the sum of weights constraint)
        for (int j = 0; j < nng; ++j) { A[j * nsz + nng] = 1.0; }
        // Fill the last row    (vector of ones and zero for the constraint)
        for (int i = 0; i < nng; ++i) { A[nng * nsz + i] = 1.0; }
        A[nng * nsz + nng] = 0.0; // Bottom-right element

        // Fill the right-hand side vector 'b' (gamma_p and 1)
        for (int j = 0; j < nng; ++j) {
            int jp = neighs[j];
            double r = (p - data_points[jp]).norm();
            b[j] = compact_c2_variogram(r, Rcut);
            printf( "Kriging k: %3i j: %3i  r: %16.8f b: %16.8f Rcut %g \n", k, j, r, b[j], Rcut );
        }
        b[nng] = 1.0; // Last element for the constraint sum(lambda_i) = 1

        // 3. Solve the system A * x = b for weights 'x' (lambda and mu) using linSolve_gauss
        // Assuming linSolve_gauss(n, A, b_in, x_out) solves A*x_out = b_in
        //bool solve_ok = 
        linSolve_gauss(nsz, A, b, w); // Solution [lambda_0...lambda_{nng-1}, mu] goes into 'w'

        // if (!solve_ok) {
        //     printf( "Kriging: linsolve failed for query point %d (nng=%d, nsz=%d). Matrix potentially singular.\n", k, nng, nsz);
        //     out_vals[k] = NAN; // Indicate failure
        //     if (out_neighs) { for(int i=0; i<nNeighMax; ++i) out_neighs[k*nNeighMax+i] = -1; }
        //     if (out_weights) { for(int i=0; i<nNeighMax; ++i) out_weights[k*nNeighMax+i] = 0.0; }
        //     continue; // Move to the next query point
        // }

        // Solution is in w: w[0]...w[nng-1] are lambda weights, w[nng] is mu

        // Optional: Save weights and neighbor indices
        if (out_neighs ){ for(int i=0;i<nNeighMax;++i){ out_neighs [k*nNeighMax+i] = i<nng ? neighs[i] : -1   ; } }
        if (out_weights ){
             // Store lambdas (first nng elements of w)
            for(int i=0;i<nng;++i){ out_weights[k*nNeighMax+i] = w[i]; }
             // Pad with 0s up to nNeighMax
            for(int i=nng;i<nNeighMax;++i){ out_weights[k*nNeighMax+i] = 0.0; }
            // Note: mu (w[nng]) is NOT stored in this nNeighMax-sized out_weights array
        }

        printf( "interpolate_local_kriging() k: %3i nng: %3i  bs: %16.8f %16.8f %16.8f %16.8f ws: %16.8f %16.8f %16.8f %16.8f \n", k, nng,  b[0], b[1], b[2], b[3],  w[0], w[1], w[2], w[3] );

        // Corrected Debug Print: Print actual weights and mu from 'w'
        //printf( "Kriging k: %3i nng: %3i  lambdas (first %d): ", k, nng, nng > 4 ? 4 : nng );
        //for(int i=0; i< (nng > 4 ? 4 : nng); ++i) printf("%16.8f ", w[i]);
        //if (nng > 4) printf("...");
        //printf(" mu: %16.8f\n", w[nng]);


        // 4. Compute the interpolated value at p using the lambda weights
        double val = 0.0;
        for (int i  = 0; i < nng; ++i) {
            int ip  = neighs[i];
            val    += w[i] * data_vals[ip]; // w[i] holds lambda_i
        }
        out_vals[k] = val;
    }

    // No need to free stack arrays
}


/*
void interpolate_local_kriging( const Vec2d* data_points, const double* data_vals, int ndata, const Vec2d* gps, double* out_vals, int ngps, double Rcut, int nNeighMax=8, int* out_neighs=0, double* out_weights=0 ) {
    //printf("interpolate_local_kriging() Rcut = %f\n", Rcut);
    printf("interpolate_local_rbf() ndata: %i ngps: %i nNeighMax: %i Rcut: %g \n", ndata, ngps, nNeighMax, Rcut);

    int nsz = nng + 1;

    int neighs[nsz];
    double A[nsz * nsz];
    double b[nsz];
    double w[nsz];

    for (int k = 0; k < ngps; ++k) {
        Vec2d p = gps[k];
        int nng = find_neighbors( ndata, data_points, p, Rcut, neighs, nNeighMax );


        // 2. Set up Ordinary Kriging system: [ Gamma  1 ] [ lambda ] = [ gamma_p ]
        //                                    [ 1^T    0 ] [ mu     ]   [ 1       ]
        // System size is (nng + 1) x (nng + 1)
    
        if (nng == 0) {
            // No neighbors found. Cannot interpolate.
            out_vals[k] = NAN; // Use NAN from math.h
            if (out_neighs) { for(int i=0; i<nNeighMax; ++i) out_neighs[k*nNeighMax+i] = -1; }
            // out_weights could store 0s, but the system wasn't solved, perhaps fill with NAN too? Or specific indicator?
            // Let's match your original idea and fill with 0s if out_weights is provided
            if (out_weights) { for(int i=0; i<nNeighMax; ++i) out_weights[k*nNeighMax+i] = 0.0; }
            // printf("Kriging k: %3i nng:   0 -> NAN\n", k); // Debug print
            continue; // Move to the next query point
        } else if (nng == 1) {
            int ip = neighs[0];
            out_vals[k] = data_vals[ip];
            if (out_neighs) {
                 out_neighs[k*nNeighMax+0] = ip;
                 for(int i=1; i<nNeighMax; ++i) out_neighs[k*nNeighMax+i] = -1;
            }
            if (out_weights) {
                out_weights[k*nNeighMax+0] = 1.0; // lambda for the single neighbor is 1
                // If out_weights stores lambdas + mu, you'd store 1.0 at index 0 and something else (0.0?) at index 1 for mu
                // But let's stick to storing just lambdas up to nNeighMax for simplicity matching your output array size idea
                for(int i=1; i<nNeighMax; ++i) out_weights[k*nNeighMax+i] = 0.0;
            }
            // printf("Kriging k: %3i nng:   1 -> val=%.4f\n", k, out_vals[k]); // Debug print
            continue; // Move to the next query point
        }


        // Fill the Gamma matrix (top-left nng x nng block of A)
        for (int j=0; j<nng; ++j) {
            int jp = neighs[j];
            for (int i = 0; i < nng; ++i) {
                int ip = neighs[i];
                double r = (data_points[jp] - data_points[ip]).norm();
                // A is stored row-major: A[row * num_cols + col] = A[j * nsz + i]
                A[j * nsz + i] = compact_c2_variogram(r, Rcut);
            }
        }

        for (int j = 0; j < nng; ++j) {  A[j * nsz + nng] = 1.0; }   // Fill the last column (vector of ones for the sum of weights constraint)        
        for (int i = 0; i < nng; ++i) {  A[nng * nsz + i] = 1.0; }   // Fill the last row    (vector of ones and zero for the constraint)
        A[nng * nsz + nng] = 0.0; // Bottom-right element

        // Fill the right-hand side vector 'b' (gamma_p and 1)
        for (int j = 0; j < nng; ++j) {
            int jp = neighs[j];
            double r = (p - data_points[jp]).norm();
            b[j] = compact_c2_variogram(r, Rcut);
        }
        b[nng] = 1.0; // Last element for the constraint sum(lambda_i) = 1

        // 3. Solve the system A * x = b for weights 'x' (lambda and mu) using linSolve_gauss The solution vector 'x' will contain [lambda_0, ..., lambda_{n-1}, mu]
        linSolve_gauss(nsz, A, b, w);

        printf( "interpolate_local_kriging() k: %3i nng: %3i  bs: %16.8f %16.8f %16.8f %16.8f ws: %16.8f %16.8f %16.8f %16.8f \n", k, nng,  b[0], b[1], b[2], b[3],  w[0], w[1], w[2], w[3] );

        if (out_neighs  ){ for(int i=0;i<nNeighMax;++i){ out_neighs [k*nNeighMax+i] = i<nng ? neighs[i] : -1   ; } }
        if (out_weights ){ for(int i=0;i<nNeighMax;++i){ out_weights[k*nNeighMax+i] = i<nng ? w[i]      :  0.0 ; } }
        // 4. Compute the interpolated value at p using the lambda weights interpolated_value = sum( lambda[i] * data_vals[neighbor_i] ) for i in neighbors
        double val = 0.0;
        for (int i  = 0; i < nng; ++i) {
            int ip  = neighs[i];
            val    += w[i] * data_vals[ip]; // x[i] holds lambda_i
        }
        out_vals[k] = val;
    }

}

*/

#endif
