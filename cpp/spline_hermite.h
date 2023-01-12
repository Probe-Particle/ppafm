
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



template <class T>
inline T val( T x,    T y0, T y1, T dy0, T dy1 ){
	T y01 = y0-y1;
	return      y0
		+x*(           dy0
		+x*( -3*y01 -2*dy0 - dy1
		+x*(  2*y01 +  dy0 + dy1 )));
}

template <class T>
inline T dval( T x,    T y0, T y1, T dy0, T dy1 ){
	T y01 = y0-y1;
	return                 dy0
		+x*( 2*( -3*y01 -2*dy0 - dy1 )
		+x*  3*(  2*y01 +  dy0 + dy1 ));
}

template <class T>
inline T ddval( T x, T y0, T y1, T dy0, T dy1 ){
	T y01 = y0-y1;
	return 2*( -3*y01 -2*dy0 - dy1 )
		+x*6*(  2*y01 +  dy0 + dy1 );
}

template <class T>
inline void basis( T x, T& c0, T& c1, T& d0, T& d1 ){
	T x2   = x*x;
	T K    =  x2*(x - 1);
	c0        =  2*K - x2 + 1;   //    2*x3 - 3*x2 + 1
	c1        = -2*K + x2    ;   //   -2*x3 + 3*x2
	d0        =    K - x2 + x;   //      x3 - 2*x2 + x
	d1        =    K         ;   //      x3 -   x2
}

template <class T>
inline void dbasis( T x, T& c0, T& c1, T& d0, T& d1 ){
	T K    =  3*x*(x - 1);
	c0        =  2*K        ;   //    6*x2 - 6*x
	c1        = -2*K        ;   //   -6*x2 + 6*x
	d0        =    K - x + 1;   //    3*x2 - 4*x + 1
	d1        =    K + x    ;   //    3*x2 - 2*x
}

template <class T>
inline void ddbasis( T x, T& c0, T& c1, T& d0, T& d1 ){
//               x3     x2    x  1
	T x6   =  6*x;
	c0        =  x6 + x6 -  6;   //    12*x - 6
	c1        =   6 - x6 - x6;   //   -12*x + 6
	d0        =  x6 -  4;        //     6*x - 4
	d1        =  x6 -  2;        //     6*x - 2
}

/*
template <class T>
inline void curve_point( T u, const Vec3T<T>& p0, const Vec3T<T>& p1, const Vec3T<T>& t0, const Vec3T<T>& t1,	Vec3T<T>& p ){
	T c0,c1,d0,d1;
	basis<T>( u,  c0, c1, d0, d1 );
	p.set_mul( p0, c0 ); p.add_mul( p1, c1 ); p.add_mul( t0, d0 ); p.add_mul( t1, d1 );
}

template <class T>
inline void curve_tangent( T u, const Vec3T<T>& p0, const Vec3T<T>& p1, const Vec3T<T>& t0, const Vec3T<T>& t1,	Vec3T<T>& t ){
	T c0,c1,d0,d1;
	dbasis<T>( u,  c0, c1, d0, d1 );
	t.set_mul( p0, c0 ); t.add_mul( p1, c1 ); t.add_mul( t0, d0 ); t.add_mul( t1, d1 );
}

template <class T>
inline void curve_accel( T u, const Vec3T<T>& p0, const Vec3T<T>& p1, const Vec3T<T>& t0, const Vec3T<T>& t1,	Vec3T<T>& a ){
	T c0,c1,d0,d1;
	ddbasis<T>( u,  c0, c1, d0, d1 );
	a.set_mul( p0, c0 ); a.add_mul( p1, c1 ); a.add_mul( t0, d0 ); a.add_mul( t1, d1 );
}
**/

template <class T>
inline T val2D( T x, T y,
	T f00, T f01, T f02, T f03,
	T f10, T f11, T f12, T f13,
	T f20, T f21, T f22, T f23,
	T f30, T f31, T f32, T f33
){
	T f0 = val<T>( x, f01, f02, 0.5*(f02-f00), 0.5*(f03-f01) );
	T f1 = val<T>( x, f11, f12, 0.5*(f12-f10), 0.5*(f13-f11) );
	T f2 = val<T>( x, f21, f22, 0.5*(f22-f20), 0.5*(f23-f21) );
	T f3 = val<T>( x, f31, f32, 0.5*(f32-f30), 0.5*(f33-f31) );
	return val<T>( y, f1, f2, 0.5*(f2-f0), 0.5*(f3-f1) );
}

template <class T>
inline T val( T x, T * fs ){
	T f0 = *fs; fs++;
	T f1 = *fs; fs++;
	T f2 = *fs; fs++;
	T f3 = *fs;
	return val<T>( x, f1, f2, (f2-f0)*0.5, (f3-f1)*0.5 );
}

template <class T>
inline T val_at( T x, T * fs ){
	int i   = (int)x;
	return val<T>( x-i, fs+i );
}

template <class T>
inline T val2D( T x, T y, T * f0s, T * f1s, T * f2s, T * f3s ){
	T f0 = val<T>( x, f0s );
	T f1 = val<T>( x, f1s );
	T f2 = val<T>( x, f2s );
	T f3 = val<T>( x, f3s );
	return val<T>( y, f1, f2, 0.5*(f2-f0), 0.5*(f3-f1) );
}

template <class T>
inline int find_index( int i, int di, T x, T * xs ){
	while (di>1){
		di = di>>1;
		int i_=i+di;
		if( xs[i_]<x ){
			i=i_;
		}
	}
	return i;
}

template <class T>
inline T find_val( T x, int n, T * xs, T * ydys ){
	int i    = find_index( 0, n, x, xs );
	double u = (x-xs[i])/(xs[i+1]-xs[i]);
	i=i<<1;
	return val( u, ydys[i], ydys[i+2], ydys[i+1], ydys[i+3] );
}

template <class T>
inline T find_vals( int n, T * xs, T * ydys, int m, T * xs_, T * ys_ ){
	int i=0;
	for( int j=0; j<m; j++ ){
		T x = xs_[j];
		if( x>xs[i] ){
			int i = find_index( 0, n, x, xs );
		}
		ys_[j] = val( x, ydys[i], ydys[i+2], ydys[i+1], ydys[i+3] );
	}
}

};

#endif
