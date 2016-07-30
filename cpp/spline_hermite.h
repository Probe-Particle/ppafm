
#ifndef  spline_hermite_h
#define  spline_hermite_h

#include <math.h>
#include <cstdlib>
#include <stdio.h>

//#include "Vec3.cpp"

//========================
//   Hermite splines
//========================
// https://en.wikipedia.org/wiki/Cubic_Hermite_spline
//
//       x3   x2   x   1
//  ----------------------
//  y0   2   -3        1
//  y1  -2   +3
// dy0   1   -2    1
// dy1   1   -1


// ============ optimized

namespace Spline_Hermite{

const static double C[4][4] = {
{ 1, 0, 0, 0 },
{ 0, 1, 0, 0 },
{ 1, 1, 1, 1 },
{ 0, 1, 2, 3 }
};

const static double B[4][4] = {
{  1,  0,  0,  0 },
{  0,  1,  0,  0 },
{ -3, -2,  3, -1 },
{  2,  1, -2,  1 }
};



template <class TYPE>
inline TYPE val( TYPE x,    TYPE y0, TYPE y1, TYPE dy0, TYPE dy1 ){
	TYPE y01 = y0-y1;
	return      y0
		+x*(           dy0
		+x*( -3*y01 -2*dy0 - dy1
		+x*(  2*y01 +  dy0 + dy1 )));
}

template <class TYPE>
inline TYPE dval( TYPE x,    TYPE y0, TYPE y1, TYPE dy0, TYPE dy1 ){
	TYPE y01 = y0-y1;
	return                 dy0
		+x*( 2*( -3*y01 -2*dy0 - dy1 )
		+x*  3*(  2*y01 +  dy0 + dy1 ));
}

template <class TYPE>
inline TYPE ddval( TYPE x, TYPE y0, TYPE y1, TYPE dy0, TYPE dy1 ){
	TYPE y01 = y0-y1;
	return 2*( -3*y01 -2*dy0 - dy1 )
		+x*6*(  2*y01 +  dy0 + dy1 );
}

template <class TYPE>
inline void basis( TYPE x, TYPE& c0, TYPE& c1, TYPE& d0, TYPE& d1 ){
	TYPE x2   = x*x;
	TYPE K    =  x2*(x - 1);
	c0        =  2*K - x2 + 1;   //    2*x3 - 3*x2 + 1
	c1        = -2*K + x2    ;   //   -2*x3 + 3*x2
	d0        =    K - x2 + x;   //      x3 - 2*x2 + x
	d1        =    K         ;   //      x3 -   x2
}

template <class TYPE>
inline void dbasis( TYPE x, TYPE& c0, TYPE& c1, TYPE& d0, TYPE& d1 ){
	TYPE K    =  3*x*(x - 1);
	c0        =  2*K        ;   //    6*x2 - 6*x
	c1        = -2*K        ;   //   -6*x2 + 6*x
	d0        =    K - x + 1;   //    3*x2 - 4*x + 1
	d1        =    K + x    ;   //    3*x2 - 2*x
}

template <class TYPE>
inline void ddbasis( TYPE x, TYPE& c0, TYPE& c1, TYPE& d0, TYPE& d1 ){
//               x3     x2    x  1
	TYPE x6   =  6*x;
	c0        =  x6 + x6 -  6;   //    12*x - 6
	c1        =   6 - x6 - x6;   //   -12*x + 6
	d0        =  x6 -  4;        //     6*x - 4
	d1        =  x6 -  2;        //     6*x - 2
}

/*
template <class TYPE>
inline void curve_point( TYPE u, const Vec3TYPE<TYPE>& p0, const Vec3TYPE<TYPE>& p1, const Vec3TYPE<TYPE>& t0, const Vec3TYPE<TYPE>& t1,	Vec3TYPE<TYPE>& p ){
	TYPE c0,c1,d0,d1;
	basis<TYPE>( u,  c0, c1, d0, d1 );
	p.set_mul( p0, c0 ); p.add_mul( p1, c1 ); p.add_mul( t0, d0 ); p.add_mul( t1, d1 );
}

template <class TYPE>
inline void curve_tangent( TYPE u, const Vec3TYPE<TYPE>& p0, const Vec3TYPE<TYPE>& p1, const Vec3TYPE<TYPE>& t0, const Vec3TYPE<TYPE>& t1,	Vec3TYPE<TYPE>& t ){
	TYPE c0,c1,d0,d1;
	dbasis<TYPE>( u,  c0, c1, d0, d1 );
	t.set_mul( p0, c0 ); t.add_mul( p1, c1 ); t.add_mul( t0, d0 ); t.add_mul( t1, d1 );
}

template <class TYPE>
inline void curve_accel( TYPE u, const Vec3TYPE<TYPE>& p0, const Vec3TYPE<TYPE>& p1, const Vec3TYPE<TYPE>& t0, const Vec3TYPE<TYPE>& t1,	Vec3TYPE<TYPE>& a ){
	TYPE c0,c1,d0,d1;
	ddbasis<TYPE>( u,  c0, c1, d0, d1 );
	a.set_mul( p0, c0 ); a.add_mul( p1, c1 ); a.add_mul( t0, d0 ); a.add_mul( t1, d1 );
}
**/

template <class TYPE>
inline TYPE val2D( TYPE x, TYPE y,
	TYPE f00, TYPE f01, TYPE f02, TYPE f03,
	TYPE f10, TYPE f11, TYPE f12, TYPE f13,
	TYPE f20, TYPE f21, TYPE f22, TYPE f23,
	TYPE f30, TYPE f31, TYPE f32, TYPE f33
){
	TYPE f0 = val<TYPE>( x, f01, f02, 0.5*(f02-f00), 0.5*(f03-f01) );
	TYPE f1 = val<TYPE>( x, f11, f12, 0.5*(f12-f10), 0.5*(f13-f11) );
	TYPE f2 = val<TYPE>( x, f21, f22, 0.5*(f22-f20), 0.5*(f23-f21) );
	TYPE f3 = val<TYPE>( x, f31, f32, 0.5*(f32-f30), 0.5*(f33-f31) );
	return val<TYPE>( y, f1, f2, 0.5*(f2-f0), 0.5*(f3-f1) );
}

template <class TYPE>
inline TYPE val( TYPE x, TYPE * fs ){
	TYPE f0 = *fs; fs++;
	TYPE f1 = *fs; fs++;
	TYPE f2 = *fs; fs++;
	TYPE f3 = *fs;
	return val<TYPE>( x, f1, f2, (f2-f0)*0.5, (f3-f1)*0.5 );
}

template <class TYPE>
inline TYPE val2D( TYPE x, TYPE y, TYPE * f0s, TYPE * f1s, TYPE * f2s, TYPE * f3s ){
	TYPE f0 = val<TYPE>( x, f0s );
	TYPE f1 = val<TYPE>( x, f1s );
	TYPE f2 = val<TYPE>( x, f2s );
	TYPE f3 = val<TYPE>( x, f3s );
	return val<TYPE>( y, f1, f2, 0.5*(f2-f0), 0.5*(f3-f1) );
}

template <class TYPE>
inline int find_index( int i, int di, TYPE x, TYPE * xs ){
	while (di>1){
		di = di>>1;
		int i_=i+di;
		if( xs[i_]<x ){
			i=i_;
		}
	}
	return i;
}

template <class TYPE>
inline TYPE find_val( TYPE x, int n, TYPE * xs, TYPE * ydys ){
	int i    = find_index( 0, n, x, xs );
	double u = (x-xs[i])/(xs[i+1]-xs[i]);
	i=i<<1;
	return val( u, ydys[i], ydys[i+2], ydys[i+1], ydys[i+3] );
}

template <class TYPE>
inline TYPE find_vals( int n, TYPE * xs, TYPE * ydys, int m, TYPE * xs_, TYPE * ys_ ){
	int i=0;
	for( int j=0; j<m; j++ ){
		TYPE x = xs_[j]; 
		if( x>xs[i] ){
			int i = find_index( 0, n, x, xs );
		}
		ys_[j] = val( x, ydys[i], ydys[i+2], ydys[i+1], ydys[i+3] );
	}
}

};

#endif



