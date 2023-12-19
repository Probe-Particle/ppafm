
#ifndef  Mat3_h
#define  Mat3_h

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
class Mat3T{
	using VEC = Vec3T<T>;
	using MAT = Mat3T<T>;
	public:
	union{
		struct{
			T xx,xy,xz;
			T yx,yy,yz;
			T zx,zy,zz;
		};
		struct{
			T ax,ay,az;
			T bx,by,bz;
			T cx,cy,cz;
		};
		struct{	VEC a,b,c; };
		T array[9];
		VEC  vecs [3];
	};


// ====== initialization

	inline explicit operator Mat3T<double>()const{ return (Mat3T<double>){ (double)xx,(double)xy,(double)xz, (double)yx,(double)yy,(double)yz, (double)zx,(double)zy,(double)zz }; }
	inline explicit operator Mat3T<float >()const{ return (Mat3T<float >){ (float)xx,(float)xy,(float)xz,    (float)yx,(float)yy,(float)yz,    (float)zx,(float)zy,(float)zz }; }
	inline explicit operator Mat3T<int >()  const{ return (Mat3T<int   >){ (int)xx,(int)xy,(int)xz,          (int)yx,(int)yy,(int)yz,          (int)zx,(int)zy,(int)zz }; }

	//inline Mat3T<double> toDouble()const{ return (Mat3T<double>){ (double)xx,(double)xy,(double)xz, (double)yx,(double)yy,(double)yz, (double)zx,(double)zy,(double)zz }; }
	//inline Mat3T<float > toFloat ()const{ return (Mat3T<float >){ (float)xx,(float)xy,(float)xz,    (float)yx,(float)yy,(float)yz,    (float)zx,(float)zy,(float)zz }; }
	//inline Mat3T<int >   toInt   ()const{ return (Mat3T<int   >){ (int)xx,(int)xy,(int)xz,          (int)yx,(int)yy,(int)yz,          (int)zx,(int)zy,(int)zz }; }

	inline void setOne(        ){ xx=yy=zz=1; xy=xz=yx=yz=zx=zy=0; };
	inline void set   ( T f ){ xx=yy=zz=f; xy=xz=yx=yz=zx=zy=0; };

	inline void set  ( const VEC& va, const VEC& vb, const VEC& vc ){ a.set(va); b.set(vb); c.set(vc); }
	inline void set  ( const MAT& M ){
		xx=M.xx; xy=M.xy; xz=M.xz;
		yx=M.yx; yy=M.yy; yz=M.yz;
		zx=M.zx; zy=M.zy; zz=M.zz;
	};

	inline void set_outer  ( const VEC& a, const VEC& b ){
		xx=a.x*b.x; xy=a.x*b.y; xz=a.x*b.z;
		yx=a.y*b.x; yy=a.y*b.y; yz=a.y*b.z;
		zx=a.z*b.x; zy=a.z*b.y; zz=a.z*b.z;
	};

	inline void diag_add( T f ){ xx+=f; yy+=f; zz+=f; };

	inline VEC getColx(){ VEC out; out.x = xx; out.y = yx; out.z = zx; return out; };
    inline VEC getColy(){ VEC out; out.x = xy; out.y = yy; out.z = zy; return out; };
    inline VEC getColz(){ VEC out; out.x = xz; out.y = yz; out.z = zz; return out; };

	inline void  colx_to( VEC& out){ out.x = xx; out.y = yx; out.z = zx; };
    inline void  coly_to( VEC& out){ out.x = xy; out.y = yy; out.z = zy; };
    inline void  colz_to( VEC& out){ out.x = xz; out.y = yz; out.z = zz; };

	inline void  setColx( const VEC v ){ xx = v.x; yx = v.y; zx = v.z; };
	inline void  setColy( const VEC v ){ xy = v.x; yy = v.y; zy = v.z; };
	inline void  setColz( const VEC v ){ xz = v.x; yz = v.y; zz = v.z; };

	// Don't need this, because we use union: use representation a,b,c
	//inline VEC getRowx(){ VEC out; out.x = xx; out.y = xy; out.z = xz; return out; };
	//inline VEC getRowy(){ VEC out; out.x = yx; out.y = yy; out.z = yz; return out; };
	//inline VEC getRowz(){ VEC out; out.x = zx; out.y = zy; out.z = zz; return out; };
	//inline void rowx_to( VEC& out ){ out.x = xx; out.y = xy; out.z = xz; };
	//inline void rowy_to( VEC& out ){ out.x = yx; out.y = yy; out.z = yz; };
	//inline void rowz_to( VEC& out ){ out.x = zx; out.y = zy; out.z = zz; };
	//inline void  setRowx( const VEC& v ){ xx = v.x; xy = v.y; xz = v.z; };
	//inline void  setRowy( const VEC& v ){ yx = v.x; yy = v.y; yz = v.z; };
	//inline void  setRowz( const VEC& v ){ zx = v.x; zy = v.y; zz = v.z; };

// ====== transpose

	inline void makeT(){
		T t1=yx; yx=xy; xy=t1;
		T t2=zx; zx=xz; xz=t2;
		T t3=zy; zy=yz; yz=t3;
	};

	inline void setT  ( const MAT& M ){
		xx=M.xx; xy=M.yx; xz=M.zx;
		yx=M.xy; yy=M.yy; yz=M.zy;
		zx=M.xz; zy=M.yz; zz=M.zz;
	};

	inline void setT  ( const VEC& va, const VEC& vb, const VEC& vc ){
		a.set( va.x, vb.x, vc.x );
		b.set( va.y, vb.y, vc.y );
		c.set( va.z, vb.z, vc.z );
	};

	inline MAT operator* ( T f   ) const { MAT m; m.a.set_mul(a,f); m.b.set_mul(b,f); m.c.set_mul(c,f); return m; };

    inline void mul ( T f        ){ a.mul(f);    b.mul(f);    c.mul(f);    };
    inline void mul ( const VEC& va ){ a.mul(va.a); b.mul(va.b); c.mul(va.c); };

    inline void div ( const VEC& va ){ a.mul(1/va.a); b.mul(1/va.b); c.mul(1/va.c); };

    inline void mulT ( const VEC& va ){
		ax*=va.x; ay*=va.y; az*=va.z;
		bx*=va.x; by*=va.y; bz*=va.z;
		cx*=va.x; cy*=va.y; cz*=va.z;
	};

    inline void divT ( const VEC& va ){
        T fx=1/va.x,fy=1/va.y,fz=1/va.z;
		ax*=fx; ay*=fy; az*=fz;
		bx*=fx; by*=fy; bz*=fz;
		cx*=fx; cy*=fy; cz*=fz;
	};


// ====== dot product with vector

	inline VEC dot( const VEC&  v ) const {
		VEC vout;
		vout.x = xx*v.x + xy*v.y + xz*v.z;
		vout.y = yx*v.x + yy*v.y + yz*v.z;
		vout.z = zx*v.x + zy*v.y + zz*v.z;
		return vout;
	}

    inline VEC dotT( const VEC&  v ) const {
		VEC vout;
		vout.x = xx*v.x + yx*v.y + zx*v.z;
		vout.y = xy*v.x + yy*v.y + zy*v.z;
		vout.z = xz*v.x + yz*v.y + zz*v.z;
		return vout;
	}

	inline void dot_to( const VEC&  v, VEC&  vout ) const {
        T vx=v.x,vy=v.y,vz=v.z; // to make it safe use inplace
		vout.x = xx*vx + xy*vy + xz*vz;
		vout.y = yx*vx + yy*vy + yz*vz;
		vout.z = zx*vx + zy*vy + zz*vz;
	};

	inline void dot_to_T( const VEC&  v, VEC&  vout ) const {
        T vx=v.x,vy=v.y,vz=v.z;
		vout.x = xx*vx + yx*vy + zx*vz;
		vout.y = xy*vx + yy*vy + zy*vz;
		vout.z = xz*vx + yz*vy + zz*vz;
	};

    inline bool tryOrthoNormalize( double errMax, int ia, int ib, int ic ){
        VEC& a = vecs[ia];
        VEC& b = vecs[ib];
        VEC& c = vecs[ic];
        bool res = false;
        res |= a.tryNormalize    ( errMax );
        res |= b.tryOrthogonalize( errMax, a );
        res |= b.tryNormalize    ( errMax );
        res |= c.tryOrthogonalize( errMax, a );
        res |= c.tryOrthogonalize( errMax, b );
        res |= c.tryNormalize    ( errMax );
        return res;
	};


    inline void orthogonalize( int ia, int ib, int ic ){
        VEC& a = vecs[ia];
        VEC& b = vecs[ib];
        VEC& c = vecs[ic];
        a.normalize ();
        b.makeOrthoU(a);
        b.normalize ();
        c.makeOrthoU(a);
        c.makeOrthoU(b);
        c.normalize();
	};

	inline void orthogonalize_taylor3( int ia, int ib, int ic ){
        VEC& a = vecs[ia];
        VEC& b = vecs[ib];
        VEC& c = vecs[ic];
        a.normalize_taylor3();
        b.makeOrthoU(a);
        b.normalize_taylor3();
        c.makeOrthoU(a);
        c.makeOrthoU(b);
        c.normalize_taylor3();
	};


// ====== matrix multiplication

	inline void set_mmul( const MAT& A, const MAT& B ){
		xx = A.xx*B.xx + A.xy*B.yx + A.xz*B.zx;
		xy = A.xx*B.xy + A.xy*B.yy + A.xz*B.zy;
		xz = A.xx*B.xz + A.xy*B.yz + A.xz*B.zz;
		yx = A.yx*B.xx + A.yy*B.yx + A.yz*B.zx;
		yy = A.yx*B.xy + A.yy*B.yy + A.yz*B.zy;
		yz = A.yx*B.xz + A.yy*B.yz + A.yz*B.zz;
		zx = A.zx*B.xx + A.zy*B.yx + A.zz*B.zx;
		zy = A.zx*B.xy + A.zy*B.yy + A.zz*B.zy;
		zz = A.zx*B.xz + A.zy*B.yz + A.zz*B.zz;
	};

	inline void set_mmul_NT( const MAT& A, const MAT& B ){
		xx = A.xx*B.xx + A.xy*B.xy + A.xz*B.xz;
		xy = A.xx*B.yx + A.xy*B.yy + A.xz*B.yz;
		xz = A.xx*B.zx + A.xy*B.zy + A.xz*B.zz;
		yx = A.yx*B.xx + A.yy*B.xy + A.yz*B.xz;
		yy = A.yx*B.yx + A.yy*B.yy + A.yz*B.yz;
		yz = A.yx*B.zx + A.yy*B.zy + A.yz*B.zz;
		zx = A.zx*B.xx + A.zy*B.xy + A.zz*B.xz;
		zy = A.zx*B.yx + A.zy*B.yy + A.zz*B.yz;
		zz = A.zx*B.zx + A.zy*B.zy + A.zz*B.zz;
	};

	inline void set_mmul_TN( const MAT& A, const MAT& B ){
		xx = A.xx*B.xx + A.yx*B.yx + A.zx*B.zx;
		xy = A.xx*B.xy + A.yx*B.yy + A.zx*B.zy;
		xz = A.xx*B.xz + A.yx*B.yz + A.zx*B.zz;
		yx = A.xy*B.xx + A.yy*B.yx + A.zy*B.zx;
		yy = A.xy*B.xy + A.yy*B.yy + A.zy*B.zy;
		yz = A.xy*B.xz + A.yy*B.yz + A.zy*B.zz;
		zx = A.xz*B.xx + A.yz*B.yx + A.zz*B.zx;
		zy = A.xz*B.xy + A.yz*B.yy + A.zz*B.zy;
		zz = A.xz*B.xz + A.yz*B.yz + A.zz*B.zz;
	};

	inline void set_mmul_TT( const MAT& A, const MAT& B ){
		xx = A.xx*B.xx + A.yx*B.xy + A.zx*B.xz;
		xy = A.xx*B.yx + A.yx*B.yy + A.zx*B.yz;
		xz = A.xx*B.zx + A.yx*B.zy + A.zx*B.zz;
		yx = A.xy*B.xx + A.yy*B.xy + A.zy*B.xz;
		yy = A.xy*B.yx + A.yy*B.yy + A.zy*B.yz;
		yz = A.xy*B.zx + A.yy*B.zy + A.zy*B.zz;
		zx = A.xz*B.xx + A.yz*B.xy + A.zz*B.xz;
		zy = A.xz*B.yx + A.yz*B.yy + A.zz*B.yz;
		zz = A.xz*B.zx + A.yz*B.zy + A.zz*B.zz;
	};

// ====== matrix solver

   inline T determinant() {
        T fCoxx = yy * zz - yz * zy;
        T fCoyx = yz * zx - yx * zz;
        T fCozx = yx * zy - yy * zx;
        T fDet = xx * fCoxx + xy * fCoyx + xz * fCozx;
        return fDet;
    };

	inline void invert_to( MAT& Mout ) {
        T idet = 1/determinant(); // we dont check det|M|=0
        Mout.xx = ( yy * zz - yz * zy ) * idet;
        Mout.xy = ( xz * zy - xy * zz ) * idet;
        Mout.xz = ( xy * yz - xz * yy ) * idet;
        Mout.yx = ( yz * zx - yx * zz ) * idet;
        Mout.yy = ( xx * zz - xz * zx ) * idet;
        Mout.yz = ( xz * yx - xx * yz ) * idet;
        Mout.zx = ( yx * zy - yy * zx ) * idet;
        Mout.zy = ( xy * zx - xx * zy ) * idet;
        Mout.zz = ( xx * yy - xy * yx ) * idet;
    };

    inline void invert_T_to( MAT& Mout ) {
        T idet = 1/determinant(); // we dont check det|M|=0
        Mout.xx = ( yy * zz - yz * zy ) * idet;
        Mout.yx = ( xz * zy - xy * zz ) * idet;
        Mout.zx = ( xy * yz - xz * yy ) * idet;
        Mout.xy = ( yz * zx - yx * zz ) * idet;
        Mout.yy = ( xx * zz - xz * zx ) * idet;
        Mout.zy = ( xz * yx - xx * yz ) * idet;
        Mout.xz = ( yx * zy - yy * zx ) * idet;
        Mout.yz = ( xy * zx - xx * zy ) * idet;
        Mout.zz = ( xx * yy - xy * yx ) * idet;
    };

    inline void adjoint_to( MAT& Mout ) {
        Mout.xx = yy * zz - yz * zy;
        Mout.xy = xz * zy - xy * zz;
        Mout.xz = xy * yz - xz * yy;
        Mout.yx = yz * zx - yx * zz;
        Mout.yy = xx * zz - xz * zx;
        Mout.yz = xz * yx - xx * yz;
        Mout.zx = yx * zy - yy * zx;
        Mout.zy = xy * zx - xx * zy;
        Mout.zz = xx * yy - xy * yx;
    };

// ======= Rotation

	inline void rotate( T angle, VEC axis  ){
		//VEC uaxis;
		//uaxis.set( axis * axis.norm() );
		axis.normalize();
		T ca   = cos(angle);
		T sa   = sin(angle);
 		rotate_csa( ca, sa, axis );
	};

	inline void rotate_csa( T ca, T sa, const VEC& uaxis ){
		a.rotate_csa( ca, sa, uaxis );
		b.rotate_csa( ca, sa, uaxis );
		c.rotate_csa( ca, sa, uaxis );
		//a.set(1);
		//b.set(2);
		//c.set(3);
	};

	inline void drotate_omega6( const VEC& w ){
        // consider not-normalized vector omega
        T ca,sa;
        sincosR2_taylor(w.norm2(), sa, ca );
        a.drotate_omega_csa(w,ca,sa);
        b.drotate_omega_csa(w,ca,sa);
        c.drotate_omega_csa(w,ca,sa);
	};

	void dRotateToward( int pivot, const MAT& rot0, T dPhi ){
        int i3 = pivot*3;
        VEC& piv  = *(VEC*)(     array+i3);
        VEC& piv0 = *(VEC*)(rot0.array+i3);
        VEC ax; ax.set_cross(piv,piv0);
        T sa = ax.norm();
        if( sa > dPhi ){
            ax.mul(1.0/sa);
            Vec2d csa; csa.fromAngle( dPhi );
            rotate_csa( csa.x, csa.y, ax );
        }else{
            set(rot0);
        }
    }

	// ==== generation

	inline void fromDirUp( const VEC& dir, const VEC& up ){
		// make orthonormal rotation matrix c=dir; b=(up-<b|c>c)/|b|; a=(c x b)/|a|;
		c.set(dir);
		//c.normalize(); // we assume dir is already normalized
		b.set(up);
		b.add_mul( c, -b.dot(c) );   //
		b.normalize();
		a.set_cross(b,c);
		//a.normalize(); // we don't need this since b,c are orthonormal
	};

    inline void fromSideUp( const VEC& side, const VEC&  up ){
		// make orthonormal rotation matrix c=dir; b=(up-<b|c>c)/|b|; a=(c x b)/|a|;
		a.set(side);
		//c.normalize(); // we assume dir is already normalized
		b.set(up);
		b.add_mul( a, -b.dot(a) );   //
		b.normalize();
		c.set_cross(b,a);
		//a.normalize(); // we don't need this since b,c are orthonormal
	};

	inline void fromCrossSafe( const Vec3d& v1, const Vec3d& v2 ){
        b.set_cross( v1, v2 );
        a.set_sub(v2,v1); a.normalize();
        double r2b = b.norm2();
        if( r2b<1e-15 ){
            a.getSomeOrtho(b,c);
        }else{
            b.mul( 1/sqrt(r2b) );
            c.set_cross(b,a);
        }
	}

	inline void fromEuler( T phi, T theta, T psi ){
        // http://mathworld.wolfram.com/EulerAngles.html
        T ca=1,sa=0, cb=1,sb=0, cc=1,sc=0;
        //if(phi*phi    >1e-16){ ca=cos(phi);   sa=sin(phi); }
        //if(theta*theta>1e-16){ cb=cos(theta); sb=sin(theta); }
        //if(psi*psi    >1e-16){ cc=cos(psi);   sc=sin(psi); }
        ca=cos(phi);   sa=sin(phi);
        cb=cos(theta); sb=sin(theta);
        cc=cos(psi);   sc=sin(psi);
        /*
        xx =  cc*ca-cb*sa*sc;
		xy =  cc*sa+cb*ca*sc;
		xz =  sc*sb;
		yx = -sc*ca-cb*sa*cc;
		yy = -sc*sa+cb*ca*cc;
		yz =  cc*sb;
		zx =  sb*sa;
		zy = -sb*ca;
		zz =  cb;
		*/

        xx =  cc*ca-cb*sa*sc;
		xy =  cc*sa+cb*ca*sc;
		xz =  sc*sb;
		zx = -sc*ca-cb*sa*cc;
		zy = -sc*sa+cb*ca*cc;
		zz =  cc*sb;
		yx =  sb*sa;
		yy = -sb*ca;
		yz =  cb;
	};

	// http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
    // http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
    // http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.53.1357&rep=rep1&type=pdf
     //  RAND _ ROTATION   Author: Jim Arvo, 1991
     //  This routine maps three values (x[0], x[1], x[2]) in the range [0,1]
     //  into a 3x3 rotation matrix, M.  Uniformly distributed random variables
     //  x0, x1, and x2 create uniformly distributed random rotation matrices.
     //  To create small uniformly distributed "perturbations", supply
     //  samples in the following ranges
     //      x[0] in [ 0, d ]
     //      x[1] in [ 0, 1 ]
     //      x[2] in [ 0, d ]
     // where 0 < d < 1 controls the size of the perturbation.  Any of the
     // random variables may be stratified (or "jittered") for a slightly more
     // even distribution.
     //=========================================================================
	inline void fromRand( const VEC& vrand  ){
        T theta = vrand.x * M_TWO_PI; // Rotation about the pole (Z).
        T phi   = vrand.y * M_TWO_PI; // For direction of pole deflection.
        T z     = vrand.z * 2.0;      // For magnitude of pole deflection.
        // Compute a vector V used for distributing points over the sphere
        // via the reflection I - V Transpose(V).  This formulation of V
        // will guarantee that if x[1] and x[2] are uniformly distributed,
        // the reflected points will be uniform on the sphere.  Note that V
        // has length sqrt(2) to eliminate the 2 in the Householder matrix.
        T r  = sqrt( z );
        T Vx = sin ( phi ) * r;
        T Vy = cos ( phi ) * r;
        T Vz = sqrt( 2.0 - z );
        // Compute the row vector S = Transpose(V) * R, where R is a simple
        // rotation by theta about the z-axis.  No need to compute Sz since
        // it's just Vz.
        T st = sin( theta );
        T ct = cos( theta );
        T Sx = Vx * ct - Vy * st;
        T Sy = Vx * st + Vy * ct;
        // Construct the rotation matrix  ( V Transpose(V) - I ) R, which
        // is equivalent to V S - R.
        xx = Vx * Sx - ct;   xy = Vx * Sy - st;   xz = Vx * Vz;
        yx = Vy * Sx + st;   yy = Vy * Sy - ct;   yz = Vy * Vz;
        zx = Vz * Sx;        zy = Vz * Sy;        zz = 1.0 - z;   // This equals Vz * Vz - 1.0
	}

    // took from here
    // Smith, Oliver K. (April 1961), "Eigenvalues of a symmetric 3 × 3 matrix.", Communications of the ACM 4 (4): 168
    // http://www.geometrictools.com/Documentation/EigenSymmetric3x3.pdf
    // https://www.geometrictools.com/GTEngine/Include/Mathematics/GteSymmetricEigensolver3x3.h
	inline void eigenvals( VEC& evs ) const {
		const T inv3  = 0.33333333333;
        const T root3 = 1.73205080757;
		T amax = array[0];
		for(int i=1; i<9; i++){ double a=array[i]; if(a>amax)amax=a; }
		T c0 = xx*yy*zz + 2*xy*xz*yz -  xx*yz*yz   - yy*xz*xz   -  zz*xy*xy;
		T c1 = xx*yy - xy*xy + xx*zz - xz*xz + yy*zz - yz*yz;
		T c2 = xx + yy + zz; // >0
		T amax2 = amax*amax; c2/=amax; c1/=amax2; c0/=(amax2*amax);
		T c2Div3 = c2*inv3;  // >0
		T aDiv3  = (c1 - c2*c2Div3)*inv3;
		if (aDiv3 > 0.0) aDiv3 = 0.0;
		T mbDiv2 = 0.5*( c0 + c2Div3*(2.0*c2Div3*c2Div3 - c1) );
		T q = mbDiv2*mbDiv2 + aDiv3*aDiv3*aDiv3;
		if (q > 0.0) q = 0.0;
		T magnitude = sqrt(-aDiv3); // > 0
		T angle = atan2( sqrt(-q), mbDiv2 ) * inv3;
		T cs    = cos(angle);
		T sn    = sin(angle);
		evs.a = amax*( c2Div3 + 2.0*magnitude*cs );
		evs.b = amax*( c2Div3 - magnitude*(cs + root3*sn) );
		evs.c = amax*( c2Div3 - magnitude*(cs - root3*sn) );
        // NOTE: sorting: evs.a >= evs.b >= evs.c
	}

	inline void eigenvec( T eval, VEC& evec ) const{
		VEC row0;  row0.set( ax - eval, ay, az );
		VEC row1;  row1.set( bx, by - eval, bz );
		VEC row2;  row2.set( cx, cy,  cz- eval );
		VEC r0xr1; r0xr1.set_cross(row0, row1);
		VEC r0xr2; r0xr2.set_cross(row0, row2);
		VEC r1xr2; r1xr2.set_cross(row1, row2);
		T d0 = r0xr1.dot( r0xr1);
		T d1 = r0xr2.dot( r0xr2);
		T d2 = r1xr2.dot( r1xr2);
		T dmax = d0; int imax = 0;
		if (d1 > dmax) { dmax = d1; imax = 1; }
		if (d2 > dmax) { imax = 2;            }
		if      (imax == 0) { evec.set_mul( r0xr1, 1/sqrt(d0) ); }
		else if (imax == 1) { evec.set_mul( r0xr2, 1/sqrt(d1) ); }
		else                { evec.set_mul( r1xr2, 1/sqrt(d2) ); }
	}

	void print() const {
        printf( " %f %f %f \n", ax, ay, az );
        printf( " %f %f %f \n", bx, by, bz );
        printf( " %f %f %f \n", cx, cy, cz );
    }

    void printOrtho() const { printf( " %f %f %f   %e %e %e \n", a.norm2(),b.norm2(),c.norm2(),   a.dot(b),a.dot(c),b.dot(c) ); }
    void printOrthoErr() const { printf( " %e %e %e   %e %e %e \n", a.norm()-1,b.norm()-1,c.norm()-1,   a.dot(b),a.dot(c),b.dot(c) ); }

    void transformVectors( int n, Vec3T<T>* v0s, Vec3T<T>* vs )const{
        for( int j=0; j<n; j++ ){
            Vec3T<T> v;
            //mrot.dot_to_T( h0s[j], h );
            dot_to( v0s[j], v );
            vs[j] = v;
            //ps[j].set_add_mul( pos, p_, r0 );
        }
    }

    void transformPoints0( int n, Vec3T<T>* v0s, Vec3T<T>* ps, const Vec3T<T>& toPos )const{
        for( int j=0; j<n; j++ ){
            Vec3T<T> v;
            //mrot.dot_to_T( apos0[j], v );
            dot_to( v0s[j], v );
            ps[j].set_add( v, toPos );
            //printf( "frag2atoms[%i]  (%g,%g,%g) (%g,%g,%g) \n", j,  apos0[j].x, apos0[j].y, apos0[j].z,   apos[j].x, apos[j].y, apos[j].z  );
            //printf( "%i %i  (%g,%g,%g) (%g,%g,%g) \n", ifrag, j,  m_apos[j].x, m_apos[j].y, m_apos[j].z,   Tp.x, Tp.y, Tp.z  );
        }
    }

    void transformPoints( int n, Vec3T<T>* p0s, Vec3T<T>* ps, const Vec3T<T>& pos0 )const{
        for( int j=0; j<n; j++ ){
            Vec3T<T> v0,v;
            v0.set_sub( p0s[j], pos0 );
            dot_to( v0, v );
            ps[j].set_add( pos0, v );
            //printf( "frag2atoms[%i]  (%g,%g,%g) (%g,%g,%g) \n", j,  apos0[j].x, apos0[j].y, apos0[j].z,   apos[j].x, apos[j].y, apos[j].z  );
            //printf( "%i %i  (%g,%g,%g) (%g,%g,%g) \n", ifrag, j,  m_apos[j].x, m_apos[j].y, m_apos[j].z,   Tp.x, Tp.y, Tp.z  );
        }
    }

};





/*
class Mat3i : public Mat3T< int   , Vec3i, Mat3i >{};
class Mat3f : public Mat3T< float , Vec3f, Mat3f >{};
class MAT : public Mat3T< T, VEC, MAT >{};
*/

using Mat3i = Mat3T< int   >;
using Mat3f = Mat3T< float >;
using Mat3d = Mat3T< double>;

static constexpr Mat3d Mat3dIdentity = Mat3d {1.0,0.0,0.0, 0.0,1.0,0.0,  0.0,0.0,1.0};
static constexpr Mat3d Mat3dZero     = Mat3d {0.0,0.0,0.0, 0.0,0.0,0.0,  0.0,0.0,0.0};

static constexpr Mat3f Mat3fIdentity = Mat3f {1.0f,0.0f,0.0f, 0.0f,1.0f,0.0f,  0.0f,0.0f,1.0f};
static constexpr Mat3f Mat3fZero     = Mat3f {0.0f,0.0f,0.0f, 0.0f,0.0f,0.0f,  0.0f,0.0f,0.0f};

inline void convert( const Mat3f& from, Mat3d& to ){ convert( from.a, to.a ); convert( from.b, to.b ); convert( from.c, to.c ); };
inline void convert( const Mat3d& from, Mat3f& to ){ convert( from.a, to.a ); convert( from.b, to.b ); convert( from.c, to.c ); };

inline Mat3f toFloat( const Mat3d& from){ Mat3f to; convert( from.a, to.a ); convert( from.b, to.b ); convert( from.c, to.c ); return to; }

#endif
