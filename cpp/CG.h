
#ifndef CG_h
#define CG_h

#include "macroUtils.h"
#include "VecN.h"


typedef  void (*DotFunc)( int n, const double * x, double * Ax ); 

class CG{ public:

    int n;
    int    istep = 0;
    double *  r  = 0;
    double *  r2 = 0;
    double *  p  = 0;
    double *  Ap = 0;
    double   rho = 0;
    double alpha = 0;

    // to solve
    double* x = 0;
    double* b = 0;
    double* M = 0;

    DotFunc dotFunc=0;

    void realloc(int n_){
        if(n_!=n){
            n = n_;
            _realloc(r,n);
            _realloc(r2,n);
            _realloc(p,n);
            _realloc(Ap,n);
        }
    };

    void dealloc(){
        _dealloc( r );
        _dealloc( r2);
        _dealloc( p );
        _dealloc( Ap);
    };

    void setLinearProblem(int n_, double* x_, double* b_, double* M_ = 0 ){
        realloc(n_);
        x=x_; b=b_; M=M_;
    }

    CG() = default;
    CG( int n_, double* x_, double* b_, double* M_ = 0 ){
        setLinearProblem(n_,x_,b_, M_ );
    };
    ~CG(){ dealloc(); }

    double step_GD(double dt){
        dotFunc  ( n, x, r );
        VecN::sub( n, b, r, r );
        //VecN::add( n, b, r, r );
        VecN::fma( n, x, r, dt, x );
        istep++;
        return VecN::dot(n, r,r);
    }

    double step_CG(){
        // see https://en.wikipedia.org/wiki/Conjugate_gradient_method
        //printf( "step_CG %i \n", istep );
        if(istep==0){
            //printf( " istep == 0 \n" );
            dotFunc  ( n, x, r );     //printf( "DEBUG 1 \n" );
            //printf("r   "); VecN::print_vector(n, r);
            VecN::sub( n, b, r, r );  //printf( "DEBUG 2 \n" ); // r = b - A*x
            //printf("r_  "); VecN::print_vector(n, r);
            VecN::set( n, r, p );     //printf( "DEBUG 3 \n" ); // p = r
            rho = VecN::dot(n, r,r);  //printf( "DEBUG 4 \n" );
            alpha = 0;
            //printf( "rho %f alpha %f \n", rho, alpha );
        }else{
            double rho2 = VecN::dot(n, r2,r2);
            double beta = rho2 / rho;
            VecN::fma( n, r2, p, beta, p );
            rho = rho2;
            double * tmp = r; r = r2; r2 = tmp;
        }
        //printf( "to dotFunc \n" );
        // NOTE : BCQ can be done if (A.T()*A) is applied instead of A in dotFunc
        //printf("p  "); VecN::print_vector(n, p);
        dotFunc( n, p, Ap);
        //printf("Ap "); VecN::print_vector(n, Ap);
        alpha = rho / VecN::dot(n, p, Ap);    // a  = <r|r>/<p|A|p>
        //printf( "rho %f alpha %f \n", rho, alpha );
        VecN::fma( n, x, p ,  alpha,   x  );   // x  = x - a*p
        VecN::fma( n, r, Ap, -alpha,   r2 );  // r2 = r - a*A|p>
        double err2 = VecN::dot(n, r2,r2);
        istep++;
        return err2;
        //printf( " iter: %i  err2: %f |  alpha %f \n", i, err2,     alpha );
        //printf( " iter: %i  err2: %f \n", i, err2 );
        //if (err2 < maxErr2 ) break;
    }

    void solve_CG( int maxIters, double maxErr, bool bPrint = false ){
        istep = 0;
        double maxErr2 = maxErr*maxErr;
        for ( int i =0; i<maxIters; i++) {
            double err2 = step_CG();
            if(bPrint) printf( "CG[%i] err %g \n", istep, sqrt(err2) );
            if ( err2 < maxErr2 ) break;
        }
    }

    void solve_GD( int maxIters, double maxErr, double dt, bool bPrint = false ){
        istep = 0;
        double maxErr2 = maxErr*maxErr;
        for ( int i =0; i<maxIters; i++) {
            double err2 = step_GD( dt );
            if(bPrint) printf( "CG[%04i] err %g \n", istep, sqrt(err2) );
            if ( err2 < maxErr2 ) break;
        }
    }

};


#endif