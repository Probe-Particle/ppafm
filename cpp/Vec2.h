
#ifndef  Vec2_h
#define  Vec2_h

#include "fastmath.h"

//template <class TYPE,class VEC>
template <class TYPE>
class Vec2TYPE{
	using VEC = Vec2TYPE<TYPE>;
	public:
	union{
		struct{ TYPE x,y; };
		struct{ TYPE a,b; };
		TYPE array[2];
	};

	// ===== methods

	inline void order       (){ if(a>b){ SWAP(a,b,TYPE); }; }
	inline void orderReverse(){ if(a<b){ SWAP(a,b,TYPE); }; }

    inline explicit operator Vec2TYPE<float >()const{ return (Vec2TYPE<float >){(float)x,(float)y}; }
	inline explicit operator Vec2TYPE<double>()const{ return (Vec2TYPE<double>){(double)x,(double)y}; }
	inline explicit operator Vec2TYPE<int   >()const{ return (Vec2TYPE<int   >){(int)x,(int)y}; }


	inline void set( TYPE f            ) { x=f;   y=f;   };
    inline void set( TYPE fx, TYPE fy  ) { x=fx;  y=fy;  };
    inline void set( const VEC& v      ) { x=v.x; y=v.y; };
	inline void set( TYPE* arr         ) { x=arr[0]; y=arr[1]; };

	//inline Vec2TYPE(){};
	//inline Vec2TYPE( TYPE f            ){ x=f;   y=f; };
	//inline Vec2TYPE( TYPE fx, TYPE fy  ){ x=fx;  y=fy; };
	//inline Vec2TYPE( const VEC& v      ){ x=v.x; y=v.y; };
	//inline Vec2TYPE( TYPE* arr         ){ x=arr[0]; y=arr[1]; };

    inline void get( TYPE& fx, TYPE& fy ) { fx=x;  fy=y;        };
	inline void get( TYPE* arr                    ) { arr[0]=x; arr[1]=y; };

    inline void add( TYPE f ) { x+=f; y+=f; };
    inline void mul( TYPE f ) { x*=f; y*=f; };

    inline void add( const VEC&  v ) { x+=v.x; y+=v.y; };
    inline void sub( const VEC&  v ) { x-=v.x; y-=v.y; };
    inline void mul( const VEC&  v ) { x*=v.x; y*=v.y; };
    inline void div( const VEC&  v ) { x/=v.x; y/=v.y; };

    inline void add( TYPE fx, TYPE fy) { x+=fx; y+=fy; };
    inline void sub( TYPE fx, TYPE fy) { x-=fx; y-=fy; };
    inline void mul( TYPE fx, TYPE fy ) { x*=fx; y*=fy; };
    inline void div( TYPE fx, TYPE fy ) { x/=fx; y/=fy; };

    inline void set_inv( const VEC&  v ) { x=1/v.x; y=1/v.y; };

	inline void set_add( const VEC& a, TYPE f ){ x=a.x+f; y=a.y+f; };
	inline void set_mul( const VEC& a, TYPE f ){ x=a.x*f; y=a.y*f; };
	inline void set_mul( const VEC& a, const VEC& b, TYPE f ){ x=a.x*b.x*f; y=a.y*b.y*f; };

	inline void set_add( const VEC& a, const VEC& b ){ x=a.x+b.x; y=a.y+b.y; };
	inline void set_sub( const VEC& a, const VEC& b ){ x=a.x-b.x; y=a.y-b.y; };
	inline void set_mul( const VEC& a, const VEC& b ){ x=a.x*b.x; y=a.y*b.y; };
	inline void set_div( const VEC& a, const VEC& b ){ x=a.x/b.x; y=a.y/b.y; };

	inline void add_mul( const VEC& a, TYPE f                ){ x+=a.x*f;     y+=a.y*f;     };
	inline void add_mul( const VEC& a, const VEC& b          ){ x+=a.x*b.x;   y+=a.y*b.y;   };
	inline void sub_mul( const VEC& a, const VEC& b          ){ x-=a.x*b.x;   y-=a.y*b.y;   };
	inline void add_mul( const VEC& a, const VEC& b, TYPE f  ){ x+=a.x*b.x*f; y+=a.y*b.y*f; };

	inline void set_add_mul( const VEC& a, const VEC& b, TYPE f ){ x= a.x + f*b.x;     y= a.y + f*b.y; };

	inline void set_lincomb( TYPE fa, const VEC& a, TYPE fb, const VEC& b ){ x = fa*a.x + fb*b.x;  y = fa*a.y + fb*b.y; };
	inline void add_lincomb( TYPE fa, const VEC& a, TYPE fb, const VEC& b ){ x+= fa*a.x + fb*b.x;  y+= fa*a.y + fb*b.y; };

	inline void set_lincomb( TYPE fa, TYPE fb, TYPE fc, const VEC& a, const VEC& b, const VEC& c ){ x = fa*a.x + fb*b.x + fc*c.x;  y = fa*a.y + fb*b.y + fc*c.y; };
	inline void add_lincomb( TYPE fa, TYPE fb, TYPE fc, const VEC& a, const VEC& b, const VEC& c ){ x+= fa*a.x + fb*b.x + fc*c.x;  y+= fa*a.y + fb*b.y + fc*c.y; };


    inline VEC operator+ ( TYPE f   ) const { VEC vo; vo.x=x+f; vo.y=y+f; return vo; };
    inline VEC operator* ( TYPE f   ) const { VEC vo; vo.x=x*f; vo.y=y*f; return vo; };

    inline VEC operator+ ( const VEC& vi ) const { VEC vo; vo.x=x+vi.x; vo.y=y+vi.y; return vo; };
    inline VEC operator- ( const VEC& vi ) const { VEC vo; vo.x=x-vi.x; vo.y=y-vi.y; return vo; };
    inline VEC operator* ( const VEC& vi ) const { VEC vo; vo.x=x*vi.x; vo.y=y*vi.y; return vo; };
    inline VEC operator/ ( const VEC& vi ) const { VEC vo; vo.x=x/vi.x; vo.y=y/vi.y; return vo; };

	inline TYPE dot      ( const VEC& a ) const { return x*a.x + y*a.y; };
	inline TYPE dot_perp ( const VEC& a ) const { return y*a.x - x*a.y; };
	inline TYPE norm2(              ) const { return x*x + y*y;     };

	inline TYPE norm ( ) const { return  sqrt( x*x + y*y ); };
    inline TYPE normalize() {
		TYPE norm  = sqrt( x*x + y*y );
		TYPE inVnorm = 1.0d/norm;
		x *= inVnorm;    y *= inVnorm;
		return norm;
    };

    inline TYPE dist2( const VEC& a) const { VEC d; TYPE dx = x-a.x; TYPE dy = y-a.y; return dx*dx + dy*dy; }
    inline TYPE dist ( const VEC& a) const { return sqrt( dist2(a) ); }

    inline double makePerpUni( const VEC& a ) { double cdot=x*a.x+y*a.y; x-=a.x*cdot; y-=a.y*cdot; return cdot; }

	inline void set_perp( const VEC& a )     { x=-a.y; y=a.x; }
	inline TYPE cross ( const VEC& a ) const { return x*a.y - y*a.x; };

	bool isBetweenRotations( const VEC& a, const VEC& b ){ return (cross(a)<0)&&(cross(b)>0);  }

	inline void     mul_cmplx (               const VEC& b ){                            TYPE x_ =    x*b.x -   y*b.y;         y =    y*b.x +   x*b.y;       x=x_;  }
	inline void     udiv_cmplx(               const VEC& b ){                            TYPE x_ =    x*b.x +   y*b.y;         y =    y*b.x -   x*b.y;       x=x_;  }
	inline void pre_mul_cmplx ( const VEC& a               ){                            TYPE x_ =  a.x*  x - a.y*  y;         y =  a.y*  x + a.x*  y;       x=x_;  }
	inline void set_mul_cmplx ( const VEC& a, const VEC& b ){                            TYPE x_ =  a.x*b.x - a.y*b.y;         y =  a.y*b.x + a.x*b.y;       x=x_;  }
	inline void set_udiv_cmplx( const VEC& a, const VEC& b ){                            TYPE x_ =  a.x*b.x + a.y*b.y;         y =  a.y*b.x - a.x*b.y;       x=x_;  }
	inline void set_div_cmplx ( const VEC& a, const VEC& b ){ TYPE ir2 = 1/b.norm2();  TYPE x_ = (a.x*b.x + a.y*b.y)*ir2;    y = (a.y*b.x - a.x*b.y)*ir2;    x=x_;  }

	inline void fromAngle        ( TYPE phi ){	x = cos( phi );	y = sin( phi );	    }
	inline void fromAngle_taylor2( TYPE phi ){	sincos_taylor2<TYPE>( phi, y, x );	}
	inline void fromCos          ( TYPE ca  ){  x=ca; y=sqrt(1-ca*ca); }
	inline void fromSin          ( TYPE sa  ){  y=sa; x=sqrt(1-sa*sa); }

	inline void rotate( TYPE phi ){
		TYPE bx = cos( phi );   		  TYPE by = sin( phi );
		TYPE x_ =    x*bx -   y*by;         y =    y*bx +   x*by;       x=x_;
	}

	inline void rotate_taylor2( TYPE phi ){
		TYPE bx,by;  sincos_taylor2<TYPE>( phi, by, bx );
		TYPE x_ =    x*bx -   y*by;         y =    y*bx +   x*by;       x=x_;
	}

	inline TYPE along_hat( const VEC& hat, const VEC& p ){ VEC ap; ap.set( p.x-x, p.y-y ); return hat.dot( ap ); }
	inline TYPE along    ( const VEC& b,   const VEC& p ){
		VEC ab,ap;
		ab.set( b.x - x, b.y - y );
		ap.set( p.x - x, p.y - y );
		return ab.dot(ap) / ab.norm(ab);
	}

	inline TYPE angle( const VEC& a ){
		TYPE d = cdot ( a );
		TYPE c = cross( a );
		return atan2( d, c );
	}

	inline TYPE totprod(){ return x*y; }

	inline void fromLinearSolution( const VEC& va, const VEC& vb, const VEC& rhs ){
        TYPE invD = 1/va .cross(vb);
        TYPE Da   =  rhs.cross(vb);
        TYPE Db   = -rhs.cross(va);
        x = Da*invD;
        y = Db*invD;
	};

	// === static functions:

	static inline TYPE dot      (VEC& a, VEC& b){ return a.x*b.y - a.y*b.x; };
	static inline TYPE cross    (VEC& a, VEC& b){ return a.x*b.x + a.y*b.y ; };
	static inline VEC  mul_cmplx(VEC& a, VEC& b){ return (VEC){ a.x*b.x-a.y*b.y, a.y*b.x+a.x*b.y };  }

};

using Vec2i = Vec2TYPE<int>;
using Vec2f = Vec2TYPE<float>;
using Vec2d = Vec2TYPE<double>;

static constexpr Vec2d Vec2dZero = (Vec2d){0.0d,0.0d};
static constexpr Vec2d Vec2dOnes = (Vec2d){1.0d,1.0d};
static constexpr Vec2d Vec2dX    = (Vec2d){1.0d,0.0d};
static constexpr Vec2d Vec2dY    = (Vec2d){0.0d,1.0d};

static constexpr Vec2f Vec2fZero = (Vec2f){0.0f,0.0f};
static constexpr Vec2f Vec2fOnes = (Vec2f){1.0f,1.0f};
static constexpr Vec2f Vec2fX    = (Vec2f){1.0f,0.0f};
static constexpr Vec2f Vec2fY    = (Vec2f){0.0f,1.0f};

static constexpr Vec2i Vec2iZero = (Vec2i){0,0};
static constexpr Vec2i Vec2iOnes = (Vec2i){1,1};
static constexpr Vec2i Vec2iX    = (Vec2i){1,0};
static constexpr Vec2i Vec2iY    = (Vec2i){0,1};

inline uint64_t scalar_id  ( const Vec2i& v){ return (((uint64_t)v.a)<<32)|v.b; }
inline uint64_t symetric_id( const Vec2i& v){ if( v.a>v.b ){ return (((uint64_t)v.b)<<32)|v.a; }else{ return (((uint64_t)v.a)<<32)|v.b; }}

inline void convert( const Vec2f& from, Vec2d& to ){ to.x=from.x;        to.y=from.y;        };
inline void convert( const Vec2d& from, Vec2f& to ){ to.x=(float)from.x; to.y=(float)from.y; };

inline Vec2f toFloat( const Vec2d& from){ return (Vec2f){(float)from.x,(float)from.y}; }

inline int print( const Vec2f&  v){ return printf( "%lg %g", v.x, v.y ); };
inline int print( const Vec2d&  v){ return printf( "%lg %g", v.x, v.y ); };
inline int print( const Vec2i&  v){ return printf( "%li %i", v.x, v.y ); };

#endif


