

#include <cstdio>
#include <cmath>

#include "Vec2.h"
#include "interp_RBF.h"
#include "interp_Kiring.h"



extern "C"{

void sample_kernel_func( int kind,  int n, double* xs, double* vals, double Rcut=1.0 ){
    //printf( "sample_kernel_func() kind: %i n: %i Rcut: %g \n", kind, n, Rcut );
    switch ( kind ){
        case 1 : for (int i=0; i<n; ++i) { vals[i] = wendland_c2  (xs[i], Rcut ); } break;
        case 2 : for (int i=0; i<n; ++i) { vals[i] = invRcube     (xs[i], Rcut ); } break;
        default: printf( "ERROR:  No such kind" ); exit(0);
    }
}

int interpolate_2d( int mode, double* data_points, double* data_vals, int ndata, double* gps, double* out_vals, int ngps, double Rcut, int nNeighMax=8, int* out_neighs=0, double* out_weights=0 ){
    // turn of stdout buffering
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    switch ( mode ){
        case 1 : interpolate_rbf_shepard  ( wendland_c2, (Vec2d*)data_points, data_vals, ndata, (Vec2d*)gps, out_vals, ngps, Rcut, nNeighMax, out_neighs, out_weights ); break;
        case 2 : interpolate_rbf_shepard  ( invRcube   , (Vec2d*)data_points, data_vals, ndata, (Vec2d*)gps, out_vals, ngps, Rcut, nNeighMax, out_neighs, out_weights ); break;
        case 3 : interpolate_rbf_shepard  ( exp4       , (Vec2d*)data_points, data_vals, ndata, (Vec2d*)gps, out_vals, ngps, Rcut, nNeighMax, out_neighs, out_weights ); break;
        case 4 : interpolate_rbf_shepard  ( exp8       , (Vec2d*)data_points, data_vals, ndata, (Vec2d*)gps, out_vals, ngps, Rcut, nNeighMax, out_neighs, out_weights ); break;
        case 5 : interpolate_local_rbf    ( (Vec2d*)data_points, data_vals, ndata, (Vec2d*)gps, out_vals, ngps, Rcut, nNeighMax, out_neighs, out_weights ); break;
        case 6 : interpolate_local_kriging( (Vec2d*)data_points, data_vals, ndata, (Vec2d*)gps, out_vals, ngps, Rcut, nNeighMax, out_neighs, out_weights ); break;
        default: printf( "ERROR:  No such mode %i\n", mode ); exit(0);
    }
    return 0;
}

}

