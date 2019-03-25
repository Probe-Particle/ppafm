
#ifndef  Vec3_h
#define  Vec3_h

#include <math.h>
#include <cstdlib>
#include <stdio.h>

#include "fastmath.h"
#include "Vec2.h"

//template <class TYPE,class VEC>
template <class TYPE>
class Vec3TYPE{
	using VEC  = Vec3TYPE<TYPE>;
	using VEC2 = Vec2TYPE<TYPE>;
	public:
	union{
		struct{ TYPE x,y,z; };
		struct{ TYPE a,b,c; };
		TYPE array[3];
	};

	// Constructors would prevent us from making Unions etc. so don't do it
	// https://stackoverflow.com/questions/4178175/what-are-aggregates-and-pods-and-how-why-are-they-special
	// but here it seems to work  https://www.youtube.com/watch?v=14Cyfz_tE20&index=10&list=PLlrATfBNZ98fqE45g3jZA_hLGUrD4bo6_
	//constexpr Vec3TYPE() = default;
	//constexpr Vec3TYPE(TYPE x_, TYPE y_, TYPE z_ ): x(x_),y(y_),z(z_){};

	// ===== methods

	// Automatic conversion (works) but would be problematic
	//inline operator Vec3TYPE<float >()const{ return (Vec3TYPE<float >){(float)x,(float)y,(float)z}; }
	//inline operator Vec3TYPE<double>()const{ return (Vec3TYPE<double>){(double)x,(double)y,(double)z}; }
	//inline operator Vec3TYPE<int   >()const{ return (Vec3TYPE<int   >){(int)x,(int)y,(int)z}; }

	// Explicit conversion
    inline explicit operator Vec3TYPE<float >()const{ return (Vec3TYPE<float >){(float)x,(float)y,(float)z}; }
	inline explicit operator Vec3TYPE<double>()const{ return (Vec3TYPE<double>){(double)x,(double)y,(double)z}; }
	inline explicit operator Vec3TYPE<int   >()const{ return (Vec3TYPE<int   >){(int)x,(int)y,(int)z}; }

	//inline operator (const char*)()const{ return (; }

	//inline Vec3TYPE<double> toDouble()const{ return (Vec3TYPE<double>){ (double)x,(double)y,(double)z}; }
	//inline Vec3TYPE<float > toFloat ()const{ return (Vec3TYPE<float >){ (float)x, (double)y,(double)z}; }
	//inline Vec3TYPE<int >   toInt   ()const{ return (Vec3TYPE<int   >){ (int)x,      (int)y,   (int)z}; }

	// swizzles
	inline VEC2 xy() const { return {x,y}; };
	inline VEC2 xz() const { return {x,z}; };
	inline VEC2 yz() const { return {x,y}; };
    inline VEC2 yx() const { return {x,y}; };
	inline VEC2 zx() const { return {x,z}; };
	inline VEC2 zy() const { return {x,y}; };
    inline VEC xzy() const { return {x,z,y}; };
	inline VEC yxz() const { return {y,x,z}; };
	inline VEC yzx() const { return {y,z,x}; };
	inline VEC zxy() const { return {z,x,y}; };
	inline VEC zyx() const { return {z,y,x}; };

	inline VEC& set( TYPE f                    ) { x=f;   y=f;   z=f;   return *this; };
    inline VEC& set( TYPE fx, TYPE fy, TYPE fz ) { x=fx;  y=fy;  z=fz;  return *this; };
    inline VEC& set( const VEC& v              ) { x=v.x; y=v.y; z=v.z; return *this; };
	inline VEC& set( TYPE* arr                 ) { x=arr[0]; y=arr[1]; z=arr[2]; return *this; };

    inline void get( TYPE& fx, TYPE& fy, TYPE& fz ) { fx=x;  fy=y;  fz=z;           return *this; };
	inline void get( TYPE* arr                    ) { arr[0]=x; arr[1]=y; arr[2]=z; return *this; };

    inline VEC& add( TYPE f ) { x+=f; y+=f; z+=f; return *this;};
    inline VEC& mul( TYPE f ) { x*=f; y*=f; z*=f; return *this;};

    inline VEC& add( const VEC&  v ) { x+=v.x; y+=v.y; z+=v.z; return *this;};
    inline VEC& sub( const VEC&  v ) { x-=v.x; y-=v.y; z-=v.z; return *this;};
    inline VEC& mul( const VEC&  v ) { x*=v.x; y*=v.y; z*=v.z; return *this;};
    inline VEC& div( const VEC&  v ) { x/=v.x; y/=v.y; z/=v.z; return *this;};

    inline VEC& set_inv( const VEC&  v ) { x=1/v.x; y=1/v.y; z=1/v.z; return *this; };
    inline VEC  get_inv()                { VEC o; o.x=1/x; o.y=1/y; o.z=1/z; return o; };

    inline VEC& add( TYPE fx, TYPE fy, TYPE fz ) { x+=fx; y+=fy; z+=fz; return *this;};
    inline VEC& sub( TYPE fx, TYPE fy, TYPE fz ) { x-=fx; y-=fy; z-=fz; return *this;};
    inline VEC& mul( TYPE fx, TYPE fy, TYPE fz ) { x*=fx; y*=fy; z*=fz; return *this;};
    inline VEC& div( TYPE fx, TYPE fy, TYPE fz ) { x/=fx; y/=fy; z/=fz; return *this;};

	inline VEC& set_add( const VEC& a, TYPE f ){ x=a.x+f; y=a.y+f; z=a.z+f; return *this;};
	inline VEC& set_mul( const VEC& a, TYPE f ){ x=a.x*f; y=a.y*f; z=a.z*f; return *this;};
	inline VEC& set_mul( const VEC& a, const VEC& b, TYPE f ){ x=a.x*b.x*f; y=a.y*b.y*f; z=a.z*b.z*f; return *this; };

	inline VEC& set_add( const VEC& a, const VEC& b ){ x=a.x+b.x; y=a.y+b.y; z=a.z+b.z; return *this; };
	inline VEC& set_sub( const VEC& a, const VEC& b ){ x=a.x-b.x; y=a.y-b.y; z=a.z-b.z; return *this; };
	inline VEC& set_mul( const VEC& a, const VEC& b ){ x=a.x*b.x; y=a.y*b.y; z=a.z*b.z; return *this; };
	inline VEC& set_div( const VEC& a, const VEC& b ){ x=a.x/b.x; y=a.y/b.y; z=a.z/b.z; return *this; };

	inline VEC& add_mul( const VEC& a, TYPE f                ){ x+=a.x*f;     y+=a.y*f;     z+=a.z*f;   return *this;};
	inline VEC& add_mul( const VEC& a, const VEC& b          ){ x+=a.x*b.x;   y+=a.y*b.y;   z+=a.z*b.z; return *this;};
	inline VEC& sub_mul( const VEC& a, const VEC& b          ){ x-=a.x*b.x;   y-=a.y*b.y;   z-=a.z*b.z; return *this;};
	inline VEC& add_mul( const VEC& a, const VEC& b, TYPE f  ){ x+=a.x*b.x*f; y+=a.y*b.y*f; z+=a.z*b.z*f;   return *this;};


	inline VEC& set_add_mul( const VEC& a, const VEC& b, TYPE f ){ x= a.x + f*b.x;     y= a.y + f*b.y;     z= a.z + f*b.z;  return *this;};


	inline VEC& set_lincomb( TYPE fa, const VEC& a, TYPE fb, const VEC& b ){ x = fa*a.x + fb*b.x;  y = fa*a.y + fb*b.y;  z = fa*a.z + fb*b.z; return *this;};
	inline VEC& add_lincomb( TYPE fa, const VEC& a, TYPE fb, const VEC& b ){ x+= fa*a.x + fb*b.x;  y+= fa*a.y + fb*b.y;  z+= fa*a.z + fb*b.z; return *this;};

	inline VEC& set_lincomb( TYPE fa, TYPE fb, TYPE fc, const VEC& a, const VEC& b, const VEC& c ){ x = fa*a.x + fb*b.x + fc*c.x;  y = fa*a.y + fb*b.y + fc*c.y;  z = fa*a.z + fb*b.z + fc*c.z; return *this;};
	inline VEC& add_lincomb( TYPE fa, TYPE fb, TYPE fc, const VEC& a, const VEC& b, const VEC& c ){ x+= fa*a.x + fb*b.x + fc*c.x;  y+= fa*a.y + fb*b.y + fc*c.y;  z+= fa*a.z + fb*b.z + fc*c.z; return *this;};

    inline VEC& set_lincomb( const VEC& fs, const VEC& a, const VEC& b, const VEC& c ){ x = fs.a*a.x + fs.b*b.x + fs.c*c.x;  y = fs.a*a.y + fs.b*b.y + fs.c*c.y;  z = fs.a*a.z + fs.b*b.z + fs.c*c.z; return *this;};
	inline VEC& add_lincomb( const VEC& fs, const VEC& a, const VEC& b, const VEC& c ){ x+= fs.a*a.x + fs.b*b.x + fs.c*c.x;  y+= fs.a*a.y + fs.b*b.y + fs.c*c.y;  z+= fs.a*a.z + fs.b*b.z + fs.c*c.z; return *this;};


    inline VEC& set_cross( const VEC& a, const VEC& b ){ x =a.y*b.z-a.z*b.y; y =a.z*b.x-a.x*b.z; z =a.x*b.y-a.y*b.x; return *this;};
	inline VEC& add_cross( const VEC& a, const VEC& b ){ x+=a.y*b.z-a.z*b.y; y+=a.z*b.x-a.x*b.z; z+=a.x*b.y-a.y*b.x; return *this;};
	inline VEC& sub_cross( const VEC& a, const VEC& b ){ x-=a.y*b.z-a.z*b.y; y-=a.z*b.x-a.x*b.z; z-=a.x*b.y-a.y*b.x; return *this;};

	TYPE makeOrthoU( const VEC& a ){ TYPE c = dot(a);          add_mul(a, -c); return c; }
	TYPE makeOrtho ( const VEC& a ){ TYPE c = dot(a)/a.norm(); add_mul(a, -c); return c; }

    inline VEC operator+ ( TYPE f   ) const { VEC vo; vo.x=x+f; vo.y=y+f; vo.z=z+f; return vo; };
    inline VEC operator* ( TYPE f   ) const { VEC vo; vo.x=x*f; vo.y=y*f; vo.z=z*f; return vo; };

    inline VEC operator+ ( const VEC& vi ) const { VEC vo; vo.x=x+vi.x; vo.y=y+vi.y; vo.z=z+vi.z; return vo;  };
    inline VEC operator- ( const VEC& vi ) const { VEC vo; vo.x=x-vi.x; vo.y=y-vi.y; vo.z=z-vi.z; return vo; };
    inline VEC operator* ( const VEC& vi ) const { VEC vo; vo.x=x*vi.x; vo.y=y*vi.y; vo.z=z*vi.z; return vo; };
    inline VEC operator/ ( const VEC& vi ) const { VEC vo; vo.x=x/vi.x; vo.y=y/vi.y; vo.z=z/vi.z; return vo; };

	inline TYPE dot  ( const VEC& a ) const { return x*a.x + y*a.y + z*a.z;  };
	inline TYPE norm2(              ) const { return x*x + y*y + z*z;        };
	inline TYPE norm ( ) const { return  sqrt( x*x + y*y + z*z ); };
    inline TYPE normalize() {
		TYPE norm  = sqrt( x*x + y*y + z*z );
		TYPE inVnorm = 1.0d/norm;
		x *= inVnorm;    y *= inVnorm;    z *= inVnorm;
		return norm;
    }
    inline VEC normalized() {
        VEC v; v.set(*this);
        v.normalize();
        return v;
    }

    inline bool tryNormalize(double errMax){
        double r2 = norm2();
        if( fabs(r2-1.0)>errMax ){
            mul( 1/sqrt(r2) );
            return true;
        }
        return false;
    }

    inline bool tryOrthogonalize( double errMax, const VEC& u ){
        double c = dot(u);
        if( fabs(c)>errMax ){
            add_mul( u, -c );
            return true;
        }
        return false;
    }

    inline VEC getOrtho( VEC& up ) const {
        up.makeOrthoU(*this); up.normalize();
        VEC out; out.set_cross(*this,up);
        return out;
	}


	inline VEC& normalize_taylor3(){
        // sqrt(1+x) ~= 1 + 0.5*x - 0.125*x*x
        // sqrt(r2) = sqrt((r2-1)+1) ~= 1 + 0.5*(r2-1)
        // 1/sqrt(1+x) ~= 1 - 0.5*x + (3/8)*x^2 - (5/16)*x^3 + (35/128)*x^4 - (63/256)*x^5
        TYPE dr2    = x*x+y*y+z*z-1;
        TYPE renorm = 1 + dr2*( -0.5d + dr2*( 0.375d + dr2*-0.3125d ) );
        x*=renorm;
        y*=renorm;
        z*=renorm;
        return *this;
	}


	inline void getSomeOrtho( VEC& v1, VEC& v2 ) const {
        TYPE xx = x*x;
        TYPE yy = y*y;
		if(xx<yy){
//			x : y*vz - z*vy;
//			y : z*vx - x*vz;
//			z : x*vy - y*vx;
//			x : y*0 - z*0 ;
//			y : z*1 - x*0 ;
//			z : x*0 - y*1 ;
//			float vx = 0; float vy = z; float vz =-y;
			v1.x =  -yy -z*z;
			v1.y =  x*y;
			v1.z =  x*z;
		}else{
//			x : y*0 - z*1;
//			y : z*0 - x*0;
//			z : x*1 - y*0;
//			float vx = -z; float vy = 0; float vz = x;
			v1.x =  y*x;
			v1.y =  -z*z -xx;
			v1.z =  y*z;
		}
		v2.x = y*v1.z - z*v1.y;
		v2.y = z*v1.x - x*v1.z;
		v2.z = x*v1.y - y*v1.x;
	}

    inline VEC& drotate_omega(const VEC& w){
        TYPE dx =y*w.z-z*w.y;
        TYPE dy =z*w.x-x*w.z;
        TYPE dz =x*w.y-y*w.x;
        //x+=dx; y+=dy; z+=dz;
        x-=dx; y-=dy; z-=dz;
        return *this;
    }

    inline VEC& drotate_omega2(const VEC& w){
        TYPE dx  = y*w.z- z*w.y;
        TYPE dy  = z*w.x- x*w.z;
        TYPE dz  = x*w.y- y*w.x;
        TYPE ddx =dy*w.z-dz*w.y;
        TYPE ddy =dz*w.x-dx*w.z;
        TYPE ddz =dx*w.y-dy*w.x;
        //x+=dx - ddx*0.5;
        //y+=dy - ddy*0.5;
        //z+=dz - ddz*0.5;
        x-=dx - ddx*0.5;
        y-=dy - ddy*0.5;
        z-=dz - ddz*0.5;
        return *this;
    }

    inline VEC& drotate_omega_csa(const VEC& w, TYPE ca, TYPE sa){
        TYPE dx  =  y*w.z -  z*w.y;
        TYPE dy  =  z*w.x -  x*w.z;
        TYPE dz  =  x*w.y -  y*w.x;
        TYPE ddx = dy*w.z - dz*w.y;
        TYPE ddy = dz*w.x - dx*w.z;
        TYPE ddz = dx*w.y - dy*w.x;
        //x+=dx*sa + ddx*ca;
        //y+=dy*sa + ddy*ca;
        //z+=dz*sa + ddz*ca;
        x-=dx*sa + ddx*ca;
        y-=dy*sa + ddy*ca;
        z-=dz*sa + ddz*ca;
        return *this;
    }

    inline VEC& drotate_omega6(const VEC& w){
        /*
        constexpr TYPE c2 = -1.0d/2;
        constexpr TYPE c3 = -1.0d/6;
        constexpr TYPE c4 =  1.0d/24;
        constexpr TYPE c5 =  1.0d/120;
        constexpr TYPE c6 = -1.0d/720;
        TYPE r2  = w.x*w.x + w.y*w.y + w.z*w.z;
        TYPE sa  =   1 + r2*( c3 + c5*r2 );
        TYPE ca  =  c2 + r2*( c4 + c6*r2 );
        */
        TYPE ca,sa;
        sincosR2_taylor(w.norm2(), sa, ca );
        drotate_omega_csa(w,ca,sa);
    }

	// Rodrigues rotation formula: v' = cosa*v + sina*(uaxis X v) + (1-cosa)*(uaxis . v)*uaxis
	inline VEC& rotate( TYPE angle, const VEC& axis  ){
		VEC uaxis;
		uaxis.set( axis * axis.norm() );
		TYPE ca   = cos(angle);
		TYPE sa   = sin(angle);
 		rotate_csa( ca, sa, uaxis );
 		return *this;
	};

	inline VEC& rotate_csa( TYPE ca, TYPE sa, const VEC& uaxis ){
		TYPE cu = (1-ca)*dot(uaxis);
		TYPE utx  = uaxis.y*z - uaxis.z*y;
		TYPE uty  = uaxis.z*x - uaxis.x*z;
		TYPE utz  = uaxis.x*y - uaxis.y*x;
		TYPE x_ = ca*x + sa*utx + cu*uaxis.x;
		TYPE y_ = ca*y + sa*uty + cu*uaxis.y;
		       z  = ca*z + sa*utz + cu*uaxis.z;
		x = x_; y = y_;
		return *this;
	};

	inline VEC& rotateTo( const VEC& rot0, double coef ){
        //rot.add_mul( rot0, coef ); rot.normalize();
        VEC ax; ax.set_cross( *this, rot0 );
        double sa2 = ax.norm2();
        if( sa2 < coef*coef ){
            ax.mul( 1/sqrt(sa2) ); // this is safe if coef is large enough
            double ca = sqrt( 1-coef*coef );
            rotate_csa( ca, coef, ax );
        }else{
            set(rot0);
        }
        return *this;
    }

    inline void getInPlaneRotation( const VEC& rot0, const VEC& xhat, const VEC& yhat, double& ca, double& sa ){
        double x0 = rot0.dot(xhat);
        double y0 = rot0.dot(yhat);
        double x_ = dot(xhat);
        double y_ = dot(yhat);
        // http://mathworld.wolfram.com/ComplexDivision.html
        double renorm = 1.0/sqrt( (x0*x0 + y0*y0)*(x_*x_ + y_*y_) );
        ca = ( x0*x_ + y0*y_ ) * renorm;
        sa = ( y0*x_ - x0*y_ ) * renorm;
    }

	inline TYPE along_hat( const VEC& hat, const VEC& p ){ VEC ap; ap.set( p.x-x, p.y-y ); return hat.dot( ap ); }
	inline TYPE along    ( const VEC& b,   const VEC& p ){
		VEC ab,ap;
		ab.set( b.x - x, b.y - y, b.z - z );
		ap.set( p.x - x, p.y - y, b.z - z );
		return ab.dot(ap) / ab.norm(ab);
	}

    inline bool isLower  ( const VEC& vmax ) const { return (x<vmax.x)&&(y<vmax.y)&&(x<vmax.z); }
    inline bool isGreater( const VEC& vmin ) const { return (x>vmin.x)&&(y>vmin.y)&&(x>vmin.z); }
    inline bool isBetween( const VEC& vmin, const VEC& vmax ) const { return (x>vmin.x)&&(x<vmax.x)&&(y>vmin.y)&&(y<vmax.y)&&(z>vmin.z)&&(z<vmax.z); }

    inline VEC& setIfLower  (const VEC& a){ if(a.x<x)x=a.x;if(a.y<y)y=a.y;if(a.z<z)z=a.z; return *this; }
    inline VEC& setIfGreater(const VEC& a){ if(a.x>x)x=a.x;if(a.y>y)y=a.y;if(a.z>z)z=a.z; return *this; }
    //inline VEC min(VEC a){ return {fmin(x,a.x),fmin(y,a.y),fmin(z,a.z)}; };
    //inline VEC max(VEC a){ return {fmax(x,a.x),fmax(y,a.y),fmax(z,a.z)}; };
    //inline VEC set_min(VEC a,VEC b){ return {fmin(x,a.x),fmin(y,a.y),fmin(z,a.z)}; };
    //inline VEC set_max(VEC a,VEC b){ return {fmax(x,a.x),fmax(y,a.y),fmax(z,a.z)}; };

    inline TYPE dist2( const VEC& a ) const { VEC d; d.set( x-a.x, y-a.y, z-a.z ); return d.norm2(); }
    inline TYPE dist ( const VEC& a ) const { VEC d; d.set( x-a.x, y-a.y, z-a.z ); return d.norm (); }

    inline TYPE totprod(){ return x*y*z; };

    TYPE angleInPlane( const VEC& a, const VEC& b ){
        TYPE x = dot(a);
        TYPE y = dot(b);
        return atan2( y, x );
    }

    inline VEC& setHomogenousSphericalSample( TYPE u, TYPE v ){
        TYPE  r = sqrt(1-u*u);
        TYPE  c = cos(v);
        TYPE  s = sin(v);
        //printf( "%f %f  %f %f %f \n", u,v,  r, c, s );
        x = r*c;
        y = r*s;
        z = u;
        return *this;
    }

    inline VEC& fromRandomSphereSample(){
        setHomogenousSphericalSample( (randf()*2)-1, randf()*2*M_PI );
        return *this;
    }

    inline VEC& fromRandomCube( double d ){ x=randf(-d,d); y=randf(-d,d); z=randf(-d,d);  return *this; }
    inline VEC& fromRandomBox( const VEC& vmin, const VEC& vmax ){ x=randf(vmin.x,vmax.x); y=randf(vmin.y,vmax.y); z=randf(vmin.z,vmax.z);  return *this; }

    inline VEC& fromLinearSolution( const VEC& va, const VEC& vb, const VEC& vc, const VEC& p ){
        // https://en.wikipedia.org/wiki/Cramer%27s_rule
        // 30 multiplications
        TYPE Dax = vb.y*vc.z - vb.z*vc.y;
        TYPE Day = vb.x*vc.z - vb.z*vc.x;
        TYPE Daz = vb.x*vc.y - vb.y*vc.x;
        TYPE idet = 1/( va.x*Dax - va.y*Day + va.z*Daz );
        x =  idet*( p.x*Dax - p.y*Day + p.z*Daz );
        y = -idet*( p.x*(va.y*vc.z - va.z*vc.y) - p.y*(va.x*vc.z - va.z*vc.x) + p.z*(va.x*vc.y - va.y*vc.x) );
        z =  idet*( p.x*(va.y*vb.z - va.z*vb.y) - p.y*(va.x*vb.z - va.z*vb.x) + p.z*(va.x*vb.y - va.y*vb.x) );
        return *this;
    }

    static inline VEC average( int n, VEC* vs ){
        VEC v;
        v.set(0.0);
        for(int i=0; i<n; i++){ v.add(vs[i]); }
        v.mul( 1/(TYPE)n);
        return v;
    }

    static inline void average( int n, int* selection, VEC* vs ){
        VEC v;
        v.set(0.0);
        for(int i=0; i<n; i++){ v.add(vs[selection[i]]); }
        v.mul( 1/(TYPE)n );
        return v;
    }

};

template<typename VEC> inline VEC cross( VEC a, VEC b ){ return (VEC){ a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x }; }
template<typename VEC> inline VEC add  ( VEC a, VEC b ){ return (VEC){ a.x+b.x, a.z+b.z, a.z+b.z }; }

using Vec3i = Vec3TYPE<int>;
using Vec3f = Vec3TYPE<float>;
using Vec3d = Vec3TYPE<double>;

static constexpr Vec3d Vec3dZero = (Vec3d){0.0d,0.0d,0.0d};
static constexpr Vec3d Vec3dOne  = (Vec3d){1.0d,1.0d,1.0d};
static constexpr Vec3d Vec3dX    = (Vec3d){1.0d,0.0d,0.0d};
static constexpr Vec3d Vec3dY    = (Vec3d){0.0d,1.0d,0.0d};
static constexpr Vec3d Vec3dZ    = (Vec3d){0.0d,0.0d,1.0d};

static constexpr Vec3f Vec3fZero = (Vec3f){0.0f,0.0f,0.0f};
static constexpr Vec3f Vec3fOne  = (Vec3f){1.0f,1.0f,1.0f};
static constexpr Vec3f Vec3fX    = (Vec3f){1.0f,0.0f,0.0f};
static constexpr Vec3f Vec3fY    = (Vec3f){0.0f,1.0f,0.0f};
static constexpr Vec3f Vec3fZ    = (Vec3f){0.0f,0.0f,1.0f};

static constexpr Vec3i Vec3iZero = (Vec3i){0,0,0};
static constexpr Vec3i Vec3iOne  = (Vec3i){1,1,1};
static constexpr Vec3i Vec3iX    = (Vec3i){1,0,0};
static constexpr Vec3i Vec3iY    = (Vec3i){0,1,0};
static constexpr Vec3i Vec3iZ    = (Vec3i){0,0,1};

inline uint64_t scalar_id  ( const Vec3i& v){ return ( v.x | (((uint64_t)v.y)<<16) | (((uint64_t)v.z)<<32) ); }
inline Vec3i    from_id    ( uint64_t id   ){
    Vec3i vi;
    vi.x=( id & 0xFFFF ); id>>16;
    vi.y=( id & 0xFFFF ); id>>16;
    vi.z=( id & 0xFFFF );
    return vi;
}

inline void convert( const Vec3f& from, Vec3d& to ){ to.x=from.x;        to.y=from.y;        to.z=from.z; };
inline void convert( const Vec3d& from, Vec3f& to ){ to.x=(float)from.x; to.y=(float)from.y; to.z=(float)from.z; };

inline Vec3f toFloat( const Vec3d& from){ return (Vec3f){(float)from.x,(float)from.y,(float)from.z}; }

//inline void print(Vec3d p){printf("(%.16g,%.16g,%.16g)", p.x,p.y,p.z);};
//inline void print(Vec3f p){printf("(%.8g,%.8g,%.8g)", p.x,p.y,p.z);};
//inline void print(Vec3d p){printf("(%lg,%lg,%lg)", p.x,p.y,p.z);};
//inline void print(Vec3f p){printf("(%g,%g,%g)", p.x,p.y,p.z);};
//inline void print(Vec3i p){printf("(%i,%i,%i)", p.x,p.y,p.z);};

inline int print( const Vec3f&  v){ return printf( "%g %g %g", v.x, v.y, v.z ); };
inline int print( const Vec3d&  v){ return printf( "%g %g %g", v.x, v.y, v.z ); };
inline int print( const Vec3i&  v){ return printf( "%i %i %i", v.x, v.y, v.z ); };

#endif



