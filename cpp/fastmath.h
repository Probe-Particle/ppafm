
#ifndef  fastmath_h
#define  fastmath_h

#include <math.h>
#include <cstdlib>
#include <stdio.h>
#include <stdint.h>

#include "macroUtils.h"
#include "integerOps.h"
#include "fastmath_light.h"
#include "gonioApprox.h"

// https://en.wikipedia.org/wiki/Error_function#Approximation_with_elementary_functions

inline double erf_4_plus(double x){
    double p = 1 + x*( 0.278393 + x*( 0.230389 + x*(0.000972 + x*0.078108 )));
    p=p*p; p=p*p;
    return 1 - 1/p;
}
inline double erf_4(double x){ if(x>0){ return erf_4_plus(x); }else{ return -erf_4_plus(-x); } }


inline double erf_6_plus(double x){
    double p = 1 + x*( 0.0705230784 + x*( 0.0422820123 + x*( 0.0092705272 + x*( 0.0001520143 + x*( 0.0002765672 + x*0.0000430638 )))));
    p=p*p; p=p*p; p=p*p; p=p*p;
    return 1 - 1/p;
}
inline double erf_6(double x){ if(x>0){ return erf_6_plus(x); }else{ return -erf_6_plus(-x); } }

template <typename T>
inline T fastExp(T x, size_t n ){
    T e = 1 + x/(1<<n);
    for(int i=0; i<n; i++) e*=e;
    return e;
}

template <typename T>
inline T fastExp_n4(T x){
    T e = 1 + x*0.0625;
    e*=e; e*=e; e*=e; e*=e;
    return e;
}

template <typename T>
inline T fastExp_n4m(T x){
    T e = 1 + x*0.0625;
    if(e<0)e=0; // smooth landing at zero - cut of divergent part
    e*=e; e*=e; e*=e; e*=e;
    return e;
}

template <typename T>
inline T fastExp_n8(T x){
    T e = 1 + x*0.00390625;
    e*=e; e*=e; e*=e; e*=e;
    e*=e; e*=e; e*=e; e*=e;
    return e;
}

inline double pow3(double x) { return x*x*x; }
inline double pow4(double x) { x*=x;            return x*x;     }
inline double pow5(double x) { double x2=x*x;   return x2*x2*x; }
inline double pow6(double x) { x*=x;            return x*x*x;   }
inline double pow7(double x) { double x3=x*x*x; return x3*x3*x; }
inline double pow8(double x) { x*=x; x*=x;      return x*x;     }

inline double powN(double x, uint8_t n) {
    uint8_t mask=1;
    double xi     = x;
    double result = 1.0;
    while(mask<n){
        if(mask&n){ result*=xi; }
        xi*=xi;
        mask<<=1;
    }
    return result;
}

// from http://martin.ankerl.com/2012/01/25/optimized-approximative-pow-in-c-and-cpp/
inline double fastPow(double a, double b) {
  union {
    double d;
    int x[2];
  } u = { a };
  u.x[1] = (int)(b * (u.x[1] - 1072632447) + 1072632447);
  u.x[0] = 0;
  return u.d;
}

// from http://martin.ankerl.com/2012/01/25/optimized-approximative-pow-in-c-and-cpp/
// should be much more precise with large b
inline double fastPrecisePow(double a, double b) {
  // calculate approximation with fraction of the exponent
  int e = (int) b;
  union {
    double d;
    int x[2];
  } u = { a };
  u.x[1] = (int)((b - e) * (u.x[1] - 1072632447) + 1072632447);
  u.x[0] = 0;

  // exponentiation by squaring with the exponent's integer part
  // double r = u.d makes everything much slower, not sure why
  double r = 1.0;
  while (e) {
    if (e & 1) {
      r *= a;
    }
    a *= a;
    e >>= 1;
  }

  return r * u.d;
}

// sqrt(1+dx) taylor(dx=0)
// http://m.wolframalpha.com/input/?i=sqrt%281%2Bx%29+taylor+x%3D0
template <typename T> inline T sqrt1_taylor1( T dx ){ return 1 + dx*  0.5; }
template <typename T> inline T sqrt1_taylor2( T dx ){ return 1 + dx*( 0.5 + dx*  -0.125 ); }
template <typename T> inline T sqrt1_taylor3( T dx ){ return 1 + dx*( 0.5 + dx*( -0.125 + dx*  0.0625 ) ); }
template <typename T> inline T sqrt1_taylor4( T dx ){ return 1 + dx*( 0.5 + dx*( -0.125 + dx*( 0.0625 + dx *  -0.0390625 ) ) ); }
template <typename T> inline T sqrt1_taylor5( T dx ){ return 1 + dx*( 0.5 + dx*( -0.125 + dx*( 0.0625 + dx *( -0.0390625 + dx * 0.02734375 ) ) ) ); }

// 1/sqrt(1+dx) taylor(dx=0)
// http://m.wolframalpha.com/input/?i=1%2Fsqrt%281%2Bx%29+taylor+x%3D0
template <typename T> inline T invSqrt1_taylor1( T dx ){ return 1 + dx*  -0.5; }
template <typename T> inline T invSqrt1_taylor2( T dx ){ return 1 + dx*( -0.5 + dx*  0.375 ); }
template <typename T> inline T invSqrt1_taylor3( T dx ){ return 1 + dx*( -0.5 + dx*( 0.375 + dx*  -0.3125 ) ); }
template <typename T> inline T invSqrt1_taylor4( T dx ){ return 1 + dx*( -0.5 + dx*( 0.375 + dx*( -0.3125 + dx * 0.2734375 ) ) ); }
template <typename T> inline T invSqrt1_taylor5( T dx ){ return 1 + dx*( -0.5 + dx*( 0.375 + dx*( -0.3125 + dx *(0.2734375 + dx * -0.24609375 ) ) ) ); }

/*
template <class FLOAT,class INT> INT fastFloor( FLOAT x ){
    if( x > 0 ){
        INT ix = static_cast <INT>(x);
        //dx = x - ix;
        return ix;
    }else{
        INT ix = static_cast <INT>(-x);
        return 1-ix;
    }
};
*/




/*
template <class TYPE>
inline clamp( TYPE x, TYPE xmin, TYPE xmax ){
	if( x<xmin ) return xmin;
	if( x>xmax ) return xmax;
	return x;
}
*/



// ========= Treshold functions ( Sigmoide, hevyside etc. ) ===========

template <class TYPE>
inline TYPE trashold_step( TYPE x, TYPE x1 ){
	if   (x<x1){ return 0.0; }
	else       { return 1.0; }
}

template <class TYPE>
inline TYPE trashold_lin( TYPE x, TYPE x1, TYPE x2 ){
	if      (x<x1){ return 0.0; }
	else if (x>x2){ return 1.0; }
	else    {       return (x-x1)/(x2-x1); };
}

template <class TYPE>
inline TYPE trashold_cub( TYPE x, TYPE x1, TYPE x2 ){
	if      (x<x1){ return 0.0; }
	else if (x>x2){ return 1.0; }
	else    {  double a =(x-x1)/(x2-x1); return a*a*( 3 - 2*a );  };
}



#endif
