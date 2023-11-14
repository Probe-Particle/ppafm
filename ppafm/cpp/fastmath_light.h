
#ifndef  fastmath_light_h
#define  fastmath_light_h

#define _USE_MATH_DEFINES

#include <math.h>

#ifndef M_PI
#define M_PI 3.141592653589793238463
#endif

#define GOLDEN_RATIO  1.61803398875
#define DEG2RAD       0.0174533
#define RAD2DEG      57.2958

constexpr double M_TWO_PI = M_PI * 2;

//#define sq(a) a*a
template <class TYPE> inline TYPE sq   (TYPE a){ return a*a; }
template <class TYPE> inline TYPE clip(TYPE x, TYPE xmin, TYPE xmax ){ if( x<xmin ) return xmin; if( x>xmax ) return xmax; return x; }

typedef int (*Func1i)( int  );
typedef int (*Func2i)( int, int );

typedef float  (*Func1f)( float  );
typedef double (*Func1d)( double );

typedef float  (*Func2f)( float,  float  );
typedef double (*Func2d)( double, double );

typedef float  (*Func3f)( float,  float,  float  );
typedef double (*Func3d)( double, double, double );

typedef void   (*Func1d2)( double,                 double&, double& );
typedef void   (*Func2d2)( double, double,         double&, double& );
typedef void   (*Func3d2)( double, double, double, double&, double& );

typedef void   (*Func1d3)( double,                 double&, double&, double& );
typedef void   (*Func2d3)( double, double,         double&, double&, double& );
typedef void   (*Func3d3)( double, double, double, double&, double&, double& );

inline double x2grid( double x, double xstep, double invXstep, int& ix ){
    double x_=x*invXstep;
    ix=(int)x_;
    //printf( " %f %f %i \n", x, x_, ix );
    return x - ix*xstep;
}

inline double dangle(double da){
    if      (da> M_PI){ return da - 2*M_PI; }
    else if (da<-M_PI){ return da + 2*M_PI; }
    return da;
}


inline double clamp( double x, double xmin, double xmax ){
    if(x<xmin){ return xmin; }else{ if(x>xmax) return xmax; };
    return x;
}

inline double clamp_abs( double x, double xmax ){
    if( x>0 ){ if(x>xmax) return xmax; }else{ if(x<-xmax) return -xmax; };
    return x;
}

inline int fastFloor( float f ){ int i=(int)f; if(f<0)i--; return i; }

// ========= random ===========

const  float INV_RAND_MAX = 1.0f/RAND_MAX;
inline float randf(){ return INV_RAND_MAX*rand(); }
inline float randf( float min, float max ){ return randf()*( max - min ) + min; }

/*
// from http://burtleburtle.net/bob/hash/integer.html
inline uint32_t hash_Wang( uint32_t a){
    a = (a ^ 61) ^ (a >> 16);
    a = a + (a << 3);
    a = a ^ (a >> 4);
    a = a * 0x27d4eb2d;
    a = a ^ (a >> 15);
    return a;
}
inline double fhash_Wang( uint32_t h ){
    return (hash_Wang( h )&(0xffff))/((double)(0xffff));
}
*/

// there are some examples of hash functions
// https://en.wikipedia.org/wiki/Linear_congruential_generator
// https://en.wikipedia.org/wiki/Xorshift
// https://gist.github.com/badboy/6267743


inline bool quadratic_roots( double a, double b, double c,  double& x1, double& x2 ){
    double D     = b*b - 4*a*c;
    if (D < 0) return false;
    double sqrtD = sqrt( D );
    double ia    = -0.5/a;
    if( ia>0 ){
        x1       = ( b - sqrtD )*ia;
        x2       = ( b + sqrtD )*ia;
    }else{
        x1       = ( b + sqrtD )*ia;
        x2       = ( b - sqrtD )*ia;
    }
    //printf( "  a,b,c, %f %f %f  x1,x2 %f %f \n", a,b,c, x1, x2 );
    return true;
}


#endif
