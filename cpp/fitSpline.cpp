
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "Vec2.h"
#include "Vec3.h"
//#include "Mat3.h"

#include "VecN.h"
#include "CG.h"

//CG cg;

double* work   = 0;
double* work2D = 0;

Vec2i   ns_2d = (Vec2i){0,0};

int     nkernel      = 0;
double* kernel_coefs = 0;

int nConvPerCG = 1;


inline double Bspline(double x){
    double absx = fabs(x);
    if(absx>2) return 0;
    double xx = x*x;
    if(absx>1) return   -absx*(xx+12) + 6*xx + 8;
    return             3*absx* xx     - 6*xx + 4;
}

//inline void set    ( int n,                  double* y, double c ){ for(int i=0; i<n; i++){ y[i]=c; } }
//inline void add_mul( int n, const double* x, double* y, double c ){ for(int i=0; i<n; i++){ y[i]+=x[i]*c; } }

//inline double dot_range(int i0,int i1, double* a, double* b ){ double y=0; for(int j=i0; j<=i1; j++){ y+=a[j]*b[j]; }; return y; }

inline int rbc_left (int ijm      ){ return abs(ijm); };
inline int rbc_right(int ijm,int n){ return (ijm>=n) ? 2*n-ijm-2 : ijm; }; 
inline int rbc(int ijm,int n){ ijm=abs(ijm); return (ijm>=n) ? 2*n-ijm-2 : ijm; }; 

inline void conv1D( int m, int n, const double* coefs, const double* x, double* y ){
    //const double* cs = coefs + m;
    //const double* cs = coefs + m;
    //printf( "coefs : "); for(int j=-m; j<=m; j++){printf( "%g ", cs[j] );}    printf( "\n" );
    //printf( "ys : ");
    int mtot=m+m+1;
    for(int i=0; i<m; i++){
        double yi=0;
        for(int j=0; j<mtot; j++){
            //int ix = abs(i+j-m);
            //printf( " i,j,m %i,%i,%i -> %i   %g %g  \n", i,j,m, ix, x[ix], coefs[j] );
            yi += x[rbc_left(i+j-m)]*coefs[j];
        };
        y[i]=yi;
        //int m1 = mtot-i;
        //y[i] = VecN::dot_back(m1,coefs   ,x+m1-1);
        //     + VecN::dot     (i ,coefs+m1,x     );
    }
    int n_=n-m;
    for(int i=m; i<n_; i++){ y[i]=VecN::dot(mtot,x+i-m,coefs); }
    const double* xend = x + n;
    for(int i=n_; i<n; i++){
        //int m1 = n-i;
        //int m2 = mtot-m1;
        //y[i] = VecN::dot     (m1,coefs   ,xend-m1-1);
        //     + VecN::dot_back(m2,coefs+m1,xend-1   );
        double yi=0;
        for(int j=0; j<mtot; j++){
            //int ix = i+j-m;
            //int dix = n-ix;
            //ix = (dix<=0) ? n+dix-2 : ix;
            //printf( " i,j,m %i,%i,%i  | %i -> %i   %g %g  \n", i,j,m,  dix, ix, x[ix], coefs[j] );
            yi += x[rbc_right(i+j-m,n)]*coefs[j];
        };
        y[i]=yi;
    }
    //printf( "\n" );
    
}

//   Mult  y = A * x
void conv2D_tensorProd( const int ord, const Vec2i& ns, const double* coefs, const double* x, double* y ){
    //printf( "Y %g %g \n", y[10], y[1000] );
    //const int mx=ns.x-ord;
    //const int my=ns.y-ord;
    const int ordsym = ord*2 + 1; //printf( "ord %i ordsym %i ns %i,%i \n", ord, ordsym, ns.x, ns.y );
    if(work==0){ 
        work = new double[ns.x]; 
        //printf( "allocate work %i \n", ns.x ); 
    }
    for(int iy=0; iy<ns.y; iy++){
        //const double* xi = x + (iy-ord-ord)*ns.x;
        double* yi = y + iy*ns.x;
        //printf( "DEBUG .1 \n" );
        VecN::set( ns.x, 0.0, work );
        //printf( "DEBUG .2 \n" );
        //printf("work0: "); for(int i=0; i<ns.x; i++){ printf("%g ",work[i]); }; printf("\n");
        for(int ky=0; ky<ordsym; ky++ ){
            //add_mul( ns.x, xi, work, coefs[ky] );
            int jx = rbc(iy-ord+ky, ns.x );
            //printf( "i,k,m %i,%i,%i -> %i  %g \n", iy, ky, ord, jx, coefs[ky] );
            VecN::add_mul( ns.x, coefs[ky], x+jx*ns.x, work );
            //printf("work: "); for(int i=0; i<ns.x; i++){ printf("%g ",work[i]); }; printf("\n");
            //xi+=ns.x;
        }
        //printf( "DEBUG .3 \n" );
        //exit(0);
        //conv1D( ord, ns.x-ord*2, coefs, work, yi+ord );
        conv1D( ord, ns.x, coefs, work, yi );
        //for(int ix=0; ix<ns.x; ix++){ yi[ix]=work[ix]; }
        
        //conv1D( ord, ns.x-ord*2, coefs, xi+ord, yi+ord );
    }
    //delete [] work;
}

void dotFunc_conv2D_tensorProd( int n,const double * x, double * Ax ){
    //printf( "dotFunc_conv2D_tensorProd n %i \n", n );
    //for(int i=0; i<n; i++){ Ax[i]=0; }
    if(nConvPerCG==1){
        conv2D_tensorProd( nkernel, ns_2d, kernel_coefs, x, Ax );
    }else{
        double* out1=work2D;
        double* out2=Ax;
        conv2D_tensorProd( nkernel, ns_2d, kernel_coefs, x, out1 );
        for(int itr=1;itr<nConvPerCG;itr++){
            double* tmp=out1; out1=out2; out2=tmp;
            conv2D_tensorProd( nkernel, ns_2d, kernel_coefs, out2, out1 );
        }
        if( nConvPerCG & 1 ) for(int i=0; i<n; i++){ Ax[i]=out2[i]; }
    }
};

// DEBUG
CG cg_glob;

extern "C"{

void convolve1D(int m,int n,double* coefs, double* x, double* y ){ 
    //conv1D( m, n-m*2, coefs, x, y+m ); 
    conv1D( m, n, coefs, x, y ); 
}

void convolve2D_tensorProduct( int ord, int nx, int ny, double* coefs, double* x, double* y ){ 
    conv2D_tensorProd( ord, (Vec2i){nx,ny}, coefs, x, y ); 
    delete [] work; work=0; 
}

void solveCG( int n, double* A, double* b, double* x, int maxIters, double maxErr ){
    CG cg( n, x, b, A );
    cg.solve_CG( maxIters, maxErr, true );
}

void fit_tensorProd( int ord, int nx, int ny, double* kernel_coefs_, double* BYref, double* Ycoefs, int maxIters, double maxErr ){
    nkernel = ord;
    ns_2d   = (Vec2i){nx,ny};
    kernel_coefs = kernel_coefs_;
    printf( " ns_2d %i,%i \n", ns_2d.x, ns_2d.y );
    CG cg( nx*ny, Ycoefs, BYref );
    cg.dotFunc = dotFunc_conv2D_tensorProd;
    //cg.step_CG();
    printf( " to CG ... \n" );
    cg.solve_CG( maxIters, maxErr, true );
    //cg.solve_GD( maxIters*5, maxErr, 100.0,  true );
    delete [] work; work=0;
}

void setup_fit_tensorProd( int ord, int nx, int ny, double* kernel_coefs_, double* BYref, double* Ycoefs, int nConvPerCG_ ){
    if(nConvPerCG_>0) work2D = new double[nx*ny];
    nConvPerCG = nConvPerCG_;
    nkernel = ord;
    ns_2d   = (Vec2i){nx,ny};
    kernel_coefs = kernel_coefs_;
    printf( " ns_2d %i,%i \n", ns_2d.x, ns_2d.y );
    cg_glob.setLinearProblem( nx*ny, Ycoefs, BYref );
    cg_glob.dotFunc = dotFunc_conv2D_tensorProd;
    //delete [] work; work=0;
}

void step_fit_tensorProd( ){
    double err2 = cg_glob.step_CG();
    printf( "CG[%i] err %g \n", cg_glob.istep, sqrt(err2) );
}


}
