
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "Vec2.h"
#include "Vec3.h"
//#include "Mat3.h"

#include "VecN.h"
#include "CG.h"

//CG cg;

double* work = 0;

Vec2i   ns_2d = (Vec2i){0,0};

int     nkernel      = 0;
double* kernel_coefs = 0;


inline double Bspline(double x){
    double absx = fabs(x);
    if(absx>2) return 0;
    double xx = x*x;
    if(absx>1) return   -absx*(xx+12) + 6*xx + 8;
    return             3*absx* xx     - 6*xx + 4;
}

//inline void set    ( int n,                  double* y, double c ){ for(int i=0; i<n; i++){ y[i]=c; } }
//inline void add_mul( int n, const double* x, double* y, double c ){ for(int i=0; i<n; i++){ y[i]+=x[i]*c; } }

inline void conv1D( const int m, const int n, const double* coefs, const double* x, double* y ){
    const double* cs = coefs + m;
    //const double* cs = coefs + m;
    //printf( "coefs : "); for(int j=-m; j<=m; j++){printf( "%g ", cs[j] );}    printf( "\n" );
    //printf( "ys : ");
    for(int i=0; i<n; i++){
        const double* xi = x+i+m;
              double  yi = 0;
        for(int j=-m; j<=m; j++){
            yi += xi[j]*cs[j];
        }
        //printf( " %g", yi );
        y[i] = yi;
    }
    //printf( "\n" );
    
}

//   Mult  y = A * x
void conv2D_tensorProd( const int ord, const Vec2i& ns, const double* coefs, const double* x, double* y ){
    //printf( "Y %g %g \n", y[10], y[1000] );
    //const int mx=ns.x-ord;
    const int my=ns.y-ord;
    const int ordsym = ord*2 + 1; //printf( "ord %i ordsym %i ns %i,%i \n", ord, ordsym, ns.x, ns.y );
    if(work==0){ 
        work = new double[ns.x]; 
        //printf( "allocate work %i \n", ns.x ); 
    }
    for(int iy=ord; iy<my; iy++){
        const double* xi = x + (iy-ord)*ns.x;
              double* yi = y + (iy    )*ns.x;
        //printf( "DEBUG .1 \n" );
        VecN::set( ns.x, 0.0, work );
        //printf( "DEBUG .2 \n" );
        //printf("work0: "); for(int i=0; i<ns.x; i++){ printf("%g ",work[i]); }; printf("\n");
        for(int ky=0; ky<ordsym; ky++ ){
            //add_mul( ns.x, xi, work, coefs[ky] );
            VecN::mul( ns.x, coefs[ky], xi, work );
            //printf("work: "); for(int i=0; i<ns.x; i++){ printf("%g ",work[i]); }; printf("\n");
            xi+=ns.x;
        }
        //printf( "DEBUG .3 \n" );
        //exit(0);
        conv1D( ord, ns.x-ord*2, coefs, work, yi+ord );
        
        //conv1D( ord, ns.x-ord*2, coefs, xi+ord, yi+ord );
    }
    //delete [] work;
}

void dotFunc_conv2D_tensorProd( int n,const double * x, double * Ax ){
    //printf( "dotFunc_conv2D_tensorProd n %i \n", n );
    for(int i=0; i<n; i++){ Ax[i]=0; }
    conv2D_tensorProd( nkernel, ns_2d, kernel_coefs, x, Ax );
};



extern "C"{

void convolve1D(int m,int n,double* coefs, double* x, double* y ){ 
    conv1D( m, n-m*2, coefs, x, y+m ); 
}

void convolve2D_tensorProduct( int ord, int nx, int ny, double* coefs, double* x, double* y ){ 
    conv2D_tensorProd( ord, (Vec2i){nx,ny}, coefs, x, y ); 
    delete [] work; work=0; 
}

void fit_tensorProd( int ord, int nx, int ny, double* kernel_coefs_, double* BYref, double* Ycoefs, int maxIters, double maxErr ){
    /*
    if( bYref ){
        double * Yref = ;
        for(int i=0; i<  i++)
        dotFunc_conv2D_tensorProd( int n, double * x, double * Ax );
        delete [] Yref;
    }
    */
    nkernel = ord;
    ns_2d   = (Vec2i){nx,ny};
    kernel_coefs = kernel_coefs_;
    printf( " ns_2d %i,%i \n", ns_2d.x, ns_2d.y );
    CG cg( nx*ny, BYref, Ycoefs );
    cg.dotFunc = dotFunc_conv2D_tensorProd;
    //cg.step_CG();
    printf( " to CG ... \n" );
    cg.solve_CG( maxIters, maxErr, true );
    //cg.solve_GD( maxIters, maxErr, 100.0,  true );
    delete [] work; work=0;
}

}
