


template <class TYPE, class CHILD> 
class Vec3TYPE{
	public:
	union{
		struct{ TYPE x,y,z; };
		struct{ TYPE a,b,c; };
		TYPE array[3];
	};

	// ===== Constructors
	//inline CHILD(){};
	//inline CHILD( TYPE  f                       ) { x=f;   y=f;   z=f;   };
    //inline CHILD( TYPE fx, TYPE fy, TYPE fz ) { x=fx;  y=fy;  z=fz;  };
    //inline CHILD( const CHILD& v                  ) { x=v.x; y=v.y; z=v.z; };

	// ===== methods
	inline void set( TYPE f                        ) { x=f;   y=f;   z=f;   };
    inline void set( TYPE fx, TYPE fy, TYPE fz ) { x=fx;  y=fy;  z=fz;  };
    inline void set( const CHILD& v                  ) { x=v.x; y=v.y; z=v.z; };
	inline void set( TYPE* arr                     ) {  x = arr[0]; y = arr[1]; z = arr[2]; };

    inline void add( TYPE f ) { x+=f; y+=f; z+=f; };
    inline void mul( TYPE f ) { x*=f; y*=f; z*=f; };

    inline void add( const CHILD&  v ) { x+=v.x; y+=v.y; z+=v.z; };
    inline void sub( const CHILD&  v ) { x-=v.x; y-=v.y; z-=v.z; };
    inline void mul( const CHILD&  v ) { x*=v.x; y*=v.y; z*=v.z; };
    inline void div( const CHILD&  v ) { x/=v.x; y/=v.y; z/=v.z; };

	inline void set_add( const CHILD& a, TYPE f ){ x=a.x+f; y=a.y+f; z=a.z+f; };
	inline void set_mul( const CHILD& a, TYPE f ){ x=a.x*f; y=a.y*f; z=a.z*f; };
	inline void set_mul( const CHILD& a, const CHILD& b, TYPE f ){ x=a.x*b.x*f; y=a.y*b.y*f; z=a.z*b.z*f; };

	inline void set_add( const CHILD& a, const CHILD& b ){ x=a.x+b.x; y=a.y+b.y; z=a.z+b.z; };
	inline void set_sub( const CHILD& a, const CHILD& b ){ x=a.x-b.x; y=a.y-b.y; z=a.z-b.z; };
	inline void set_mul( const CHILD& a, const CHILD& b ){ x=a.x*b.x; y=a.y*b.y; z=a.z*b.z; };
	inline void set_div( const CHILD& a, const CHILD& b ){ x=a.x/b.x; y=a.y/b.y; z=a.z/b.z; };

	inline void add_mul( const CHILD& a, TYPE f                  ){ x+=a.x*f;     y+=a.y*f;     z+=a.z*f;   };
	inline void add_mul( const CHILD& a, const CHILD& b          ){ x+=a.x*b.x;   y+=a.y*b.y;   z+=a.z*b.z; };
	inline void sub_mul( const CHILD& a, const CHILD& b          ){ x-=a.x*b.x;   y-=a.y*b.y;   z-=a.z*b.z; };
	inline void add_mul( const CHILD& a, const CHILD& b, TYPE f  ){ x+=a.x*b.x*f; y+=a.y*b.y*f; z+=a.z*b.z*f;   };

    inline void  set_cross( const CHILD& a, const CHILD& b ){ x =a.y*b.z-a.z*b.y; y =a.z*b.x-a.x*b.z; z =a.x*b.y-a.y*b.x; };
	inline void  add_cross( const CHILD& a, const CHILD& b ){ x+=a.y*b.z-a.z*b.y; y+=a.z*b.x-a.x*b.z; z+=a.x*b.y-a.y*b.x; };

	// ===== Opearators

	// do not use assignement operator because it is not obious what is ment
	// also there is conflict with Mat3d: member ‘CHILD Mat3d::<anonymous union>::<anonymous struct>::a’ with copy assignment operator not allowed in anonymous aggregate
	//inline CHILD& operator =( const TYPE f ) { x=f; y=f; z=f;       return *this; }; 
	//inline CHILD& operator =( const CHILD& v ) { x=v.x; y=v.y; z=v.z; return *this; }; 

    //inline CHILD& operator+=( TYPE f ) { x+=f; y+=f; z+=f; return *this; };
    //inline CHILD& operator*=( TYPE f ) { x*=f; y*=f; z*=f; return *this; };

    //inline CHILD& operator+=( const CHILD&  v ) { x+=v.x; y+=v.y; z+=v.z; return *this; };
    //inline CHILD& operator-=( const CHILD&  v ) { x-=v.x; y-=v.y; z-=v.z; return *this; };
    //inline CHILD& operator*=( const CHILD&  v ) { x*=v.x; y*=v.y; z*=v.z; return *this; };
    //inline CHILD& operator/=( const CHILD&  v ) { x/=v.x; y/=v.y; z/=v.z; return *this; };

	// This creates a new vectors? is it good?
    inline CHILD operator+ ( TYPE f   ) const { CHILD vo; vo.x=x+f; vo.y=y+f; vo.z=z+f; return vo; };
    inline CHILD operator* ( TYPE f   ) const { CHILD vo; vo.x=x*f; vo.y=y*f; vo.z=z*f; return vo; };

    inline CHILD operator+ ( const CHILD& vi ) const { CHILD vo; vo.x=x+vi.x; vo.y=y+vi.y; vo.z=z+vi.z; return vo; };
    inline CHILD operator- ( const CHILD& vi ) const { CHILD vo; vo.x=x-vi.x; vo.y=y-vi.y; vo.z=z-vi.z; return vo; };
    inline CHILD operator* ( const CHILD& vi ) const { CHILD vo; vo.x=x*vi.x; vo.y=y*vi.y; vo.z=z*vi.z; return vo; };
    inline CHILD operator/ ( const CHILD& vi ) const { CHILD vo; vo.x=x/vi.x; vo.y=y/vi.y; vo.z=z/vi.z; return vo; };

	inline TYPE dot  ( const CHILD& a ) const { return x*a.x + y*a.y + z*a.z;  };
	inline TYPE norm2(                   ) const { return x*x + y*y + z*z; };
};


class Vec3i : public Vec3TYPE<int,Vec3i>{};
class Vec3d : public Vec3TYPE<double,Vec3d>{
	public:
	inline double norm ( ) const { return  sqrt( x*x + y*y + z*z ); };
    inline double normalize() {
		double norm  = sqrt( x*x + y*y + z*z );
		double inVnorm = 1.0d/norm;
		x *= inVnorm;    y *= inVnorm;    z *= inVnorm;
		return norm;
    };
};

