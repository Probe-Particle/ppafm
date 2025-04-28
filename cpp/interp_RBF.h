#ifndef  interp_RBF_h
#define  interp_RBF_h

#include <stdio.h>
#include <cmath> // For NAN, isnan, isinf
#include <new>   // For std::nothrow

#include "Vec2.h"
#include "gauss_solver.hpp" // Include the Gaussian solver header

// Forward declaration for the linear solver function used within this file
// Assuming linSolve_gauss is declared in gauss_solver.hpp as:
// void linSolve_gauss(int n, double* A, double* b, double* x);
// If it has a different signature (e.g., returns bool), adjust accordingly.

double wendland_c2(double r, double Rcut) {
    if (r < 0) r = -r;
    if (r >= Rcut) return 0.0;
    double t = r / Rcut;
    double t1 = 1.0 - t;
    double t2 = t1 * t1;
    double t4 = t2 * t2;
    //printf( "wendland_c2() r: % 16.8f t: % 16.8f t1: % 16.8f t2: % 16.8f t4: % 16.8f \n", r, t, t1, t2, t4 );
    return t4 * (4.0 * t + 1.0);
}

double invRcube(double r, double Rcut ) {
    double damp=1e-8;
    double t = r / Rcut;
    if(t>1) return 0;
    double inv = 1/(t+damp);
    double fc = 1-r;
    return fc*fc*inv;
}

double exp4(double r, double Rcut ) {
    double t = 1-r/Rcut;
    if(t<0) return 0;
    t=t*t;
    return t*t;
}

double exp8(double r, double Rcut ) {
    double t = 1-r/Rcut;
    if(t<0) return 0;
    t=t*t;
    t=t*t;
    return t*t;
}



template<typename Func>
void interpolate_rbf_shepard( Func func, const Vec2d* data_points, const double* data_vals, int ndata, const Vec2d* gps, double* out_vals, int ngps, double Rcut, int nNeighMax=8, int* out_neighs=0, double* out_weights=0, double w_reg=1e-8 ){
    printf("interpolate_rbf_shepard() ndata: %i ngps: %i nNeighMax: %i Rcut: %g \n", ndata, ngps, nNeighMax, Rcut);

    int neighs[nNeighMax];
    double ws [nNeighMax];
    double R2cut = Rcut*Rcut;

    for (int k = 0; k < ngps; ++k) {
        Vec2d p = gps[k];
        if (out_neighs ){ for (int i=0;i<nNeighMax;++i) { neighs[i]=-1;  } }
        if (out_weights){ for (int i=0;i<nNeighMax;++i) { ws[i]    =0.0; } }
        //int ioff = k*nNeighMax;
        //ws[0] = k/30;
        //ws[1] = k%30;
        //ws[2] = p.x;
        //ws[3] = p.y;

        double wf_sum = 0.0;
        double w_sum  = w_reg; 

        int nng = 0;
        for (int j = 0; j < ndata; ++j) {
            Vec2d  d  = p - data_points[j];
            double r2  = d.norm2();
            
            if(r2>R2cut) continue;

            double r  = sqrt(r2);
            //double wj = wendland_c2(r, Rcut); 
            double wj = func(r, Rcut); 

            //double wj = 1 - r/Rcut;

            //printf( " k: %3i j: %3i  r: %16.8f wj: %16.8f \n", k, j, r, wj );

            double vj = data_vals[j]; 
            double wfj = wj * vj;
            
            wf_sum += wfj;
            w_sum  += wj;

            neighs[nng] = j;
            ws    [nng] = wj;
            nng++;
        }

        double denom = 1.0/w_sum;
        int ioff = k*nNeighMax;
        double val = 0.0; 
        if( nng>0 ){ 
            //printf( "interpolate_rbf_shepard() k: %3i nng: %3i  wf_sum: %16.8f w_sum: %16.8f val: %16.8f \n", k, nng, wf_sum, w_sum, wf_sum*denom ); 
            val = wf_sum * denom;
        };
        if( out_vals ){ out_vals[k] = wf_sum * denom; }

        if (out_neighs ) { for (int i=0;i<nNeighMax;++i){ out_neighs [ioff+i] = neighs[i]; } }
        if (out_weights) { for (int i=0;i<nNeighMax;++i){ out_weights[ioff+i] = ws[i];     } }
    }
}


void interpolate_local_rbf( const Vec2d* data_points, const double* data_vals, int ndata, const Vec2d* gps, double* out_vals, int ngps, double Rcut, int nNeighMax=8, int* out_neighs=0, double* out_weights=0 ){
    printf("interpolate_local_rbf() ndata: %i ngps: %i nNeighMax: %i Rcut: %g \n", ndata, ngps, nNeighMax, Rcut);

    int neighs[nNeighMax];
    double A[nNeighMax * nNeighMax];
    double b[nNeighMax];
    double w[nNeighMax];

    for (int k = 0; k < ngps; ++k) {
        Vec2d p = gps[k];
        int nng = find_neighbors( ndata, data_points, p, Rcut, neighs, nNeighMax );

        printf( "interpolate_local_rbf() k: %3i nng: %3i \n", k, nng );

        if (nng == 0) {
            if( out_vals    ){ out_vals[k] = NAN; }
            if( out_neighs  ){ for (int i = 0; i < nNeighMax; ++i) { out_neighs [k * nNeighMax + i] = -1;  } }
            if( out_weights ){ for (int i = 0; i < nNeighMax; ++i) { out_weights[k * nNeighMax + i] = 0.0; } }
            continue;
        }

        // 2. Set up local RBF system: A * w = b
        // A_ji = phi(||p_j - p_i||) where p_j, p_i are neighbor data points
        // b_j = data_vals[neighs[j]]
        for (int j = 0; j < nng; ++j) {
            int jp = neighs[j];
            b[j] = data_vals[jp]; // Right-hand side vector
            Vec2d pj = data_points[jp];
            for (int i = 0; i<nng; ++i) {
                int ip = neighs[i];   // Use Vec2d subtraction and norm method
                double r = ( pj - data_points[ip]).norm();  // Matrix A is stored row-major (or column-major depending on solver expectation, assume row-major)
                A[j * nng + i] = wendland_c2(r, Rcut);      // A[row * num_cols + col] = A[j * nng + i]
            }
        }

        // 3. Solve the system A * w = b for weights 'w' using linSolve_gauss The linSolve_gauss function from gauss_solver.hpp is expected to solve Ax=b and put the solution 'x' into the last argument. Here, 'w' is our solution vector.
        linSolve_gauss(nng, A, b, w);

        if (out_neighs) {
            for (int i = 0; i < nNeighMax; ++i) {
                out_neighs[k * nNeighMax + i] = i<nng ? neighs[i] : -1;
            }
        }

        if (out_weights) {
            for (int i = 0; i < nNeighMax; ++i) {
                out_weights[k * nNeighMax + i] = i<nng ? w[i] : 0.0;
            }
        }

        if( out_vals ){
            // 4. Compute the interpolated value at p interpolated_value = sum( w[i] * phi(||p - p_i||) ) for i in neighbors
            double val = 0.0;
            for (int i = 0; i < nng; ++i) {
                int ip = neighs[i];
                double r = (p - data_points[ip]).norm(); 
                val += w[i] * wendland_c2(r, Rcut);
            }
            out_vals[k] = val;
        }
        
    }

}


#endif
