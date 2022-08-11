
#ifndef  SMat3_h
#define  SMat3_h

#include <math.h>
#include <cstdlib>
#include <cstdio>
#include <stdint.h>

//#include "fastmath.h"
#include "fastmath_light.h"
#include "Vec3.h"

//template <class T, class VEC, class MAT>
//template <class T, class VEC>
template <class T>
class SMat3{
	using VEC = Vec3T<T>;
	using MAT = SMat3<T>;
	public:
	union{
		struct{
			T xx,yy,zz;
			T yz,xz,xy;
		};
		struct{	VEC diag,offd; };
		T array[6];
		VEC  vecs[2];
	};

	inline void setOne(     ){ xx=yy=zz=1; xy=xz=yz=0; };
	inline void set   ( T f ){ xx=yy=zz=f; xy=xz=yz=0; };

	inline MAT operator* ( T f   ) const { MAT m; m.diag.set_mul(diag,f); m.offd.set_mul(offd,f); return m; };

    inline void mul ( T f           ){ diag.mul(f); offd.mul(f);  };

    inline void add_mul( MAT m, T f ){ diag.add_mul(m.diag,f); offd.add_mul(m.offd,f); }
   
// ====== dot product with vector

	inline VEC dot( const VEC&  v ) const {
		VEC vout;
		vout.x = xx*v.x + xy*v.y + xz*v.z;
		vout.y = xy*v.x + yy*v.y + yz*v.z;
		vout.z = xz*v.x + yz*v.y + zz*v.z;
		return vout;
	}

	inline void dot_to( const VEC&  v, VEC&  vout ) const {
        T vx=v.x,vy=v.y,vz=v.z; // to make it safe use inplace
		vout.x = xx*vx + xy*vy + xz*vz;
		vout.y = xy*vx + yy*vy + yz*vz;
		vout.z = xz*vx + yz*vy + zz*vz;
	};

};

/*
class Mat3i : public Mat3T< int   , Vec3i, Mat3i >{};
class Mat3f : public Mat3T< float , Vec3f, Mat3f >{};
class MAT : public Mat3T< T, VEC, MAT >{};
*/

using SMat3i = SMat3< int   >;
using SMat3f = SMat3< float >;
using SMat3d = SMat3< double>;

static constexpr SMat3d SMat3dIdentity = (SMat3d){ 1.0,1.0,1.0, 0.0,0.0,0.0 };
static constexpr SMat3d SMat3dZero     = (SMat3d){ 0.0,0.0,0.0, 0.0,0.0,0.0 };

static constexpr SMat3f SMat3fIdentity = (SMat3f){1.0f,1.0f,1.0f, 0.0f,0.0f,0.0f };
static constexpr SMat3f SMat3fZero     = (SMat3f){0.0f,0.0f,0.0f, 0.0f,0.0f,0.0f };

inline void convert( const SMat3f& from, SMat3d& to ){ convert( from.diag, to.diag ); convert( from.offd, to.offd ); };
inline void convert( const SMat3d& from, SMat3f& to ){ convert( from.diag, to.diag ); convert( from.offd, to.offd ); };

inline SMat3f toFloat( const SMat3d& from){ SMat3f to; convert( from.diag, to.diag ); convert( from.offd, to.offd ); return to; }

#endif

