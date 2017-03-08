
#ifndef Vec3_h 
#define Vec3_h 

template <class TYPE, class VEC> 
class Vec3TYPE{
	public:
	union{
		struct{ TYPE x,y,z; };
		struct{ TYPE a,b,c; };
		TYPE array[3];
	};

	// ===== Constructors
	//inline VEC(){};
	//inline VEC( TYPE  f                       ) { x=f;   y=f;   z=f;   };
    //inline VEC( TYPE fx, TYPE fy, TYPE fz ) { x=fx;  y=fy;  z=fz;  };
    //inline VEC( const VEC& v                  ) { x=v.x; y=v.y; z=v.z; };

	// ===== methods
	inline void set( TYPE f                    ) { x=f;   y=f;   z=f;   };
    inline void set( TYPE fx, TYPE fy, TYPE fz ) { x=fx;  y=fy;  z=fz;  };
    inline void set( const VEC& v            ) { x=v.x; y=v.y; z=v.z; };
	inline void set( TYPE* arr                 ) {  x = arr[0]; y = arr[1]; z = arr[2]; };

    inline void add( TYPE f ) { x+=f; y+=f; z+=f; };
    inline void mul( TYPE f ) { x*=f; y*=f; z*=f; };

    inline void add( const VEC&  v ) { x+=v.x; y+=v.y; z+=v.z; };
    inline void sub( const VEC&  v ) { x-=v.x; y-=v.y; z-=v.z; };
    inline void mul( const VEC&  v ) { x*=v.x; y*=v.y; z*=v.z; };
    inline void div( const VEC&  v ) { x/=v.x; y/=v.y; z/=v.z; };

	inline void set_add( const VEC& a, TYPE f ){ x=a.x+f; y=a.y+f; z=a.z+f; };
	inline void set_mul( const VEC& a, TYPE f ){ x=a.x*f; y=a.y*f; z=a.z*f; };
	inline void set_mul( const VEC& a, const VEC& b, TYPE f ){ x=a.x*b.x*f; y=a.y*b.y*f; z=a.z*b.z*f; };

	inline void set_add( const VEC& a, const VEC& b ){ x=a.x+b.x; y=a.y+b.y; z=a.z+b.z; };
	inline void set_sub( const VEC& a, const VEC& b ){ x=a.x-b.x; y=a.y-b.y; z=a.z-b.z; };
	inline void set_mul( const VEC& a, const VEC& b ){ x=a.x*b.x; y=a.y*b.y; z=a.z*b.z; };
	inline void set_div( const VEC& a, const VEC& b ){ x=a.x/b.x; y=a.y/b.y; z=a.z/b.z; };

	inline void add_mul( const VEC& a, TYPE f                  ){ x+=a.x*f;     y+=a.y*f;     z+=a.z*f;   };
	inline void add_mul( const VEC& a, const VEC& b          ){ x+=a.x*b.x;   y+=a.y*b.y;   z+=a.z*b.z; };
	inline void sub_mul( const VEC& a, const VEC& b          ){ x-=a.x*b.x;   y-=a.y*b.y;   z-=a.z*b.z; };
	inline void add_mul( const VEC& a, const VEC& b, TYPE f  ){ x+=a.x*b.x*f; y+=a.y*b.y*f; z+=a.z*b.z*f;   };

	inline void set_add_mul( const VEC& a, const VEC& b, TYPE f ){ x= a.x + f*b.x;     y= a.y + f*b.y;     z= a.z + f*b.z;  };

	inline void set_lincomb( TYPE fa, const VEC& a, TYPE fb, const VEC& b ){ x = fa*a.x + fb*b.x;  y = fa*a.y + fb*b.y;  z = fa*a.z + fb*b.z; };
	inline void add_lincomb( TYPE fa, const VEC& a, TYPE fb, const VEC& b ){ x+= fa*a.x + fb*b.x;  y+= fa*a.y + fb*b.y;  z+= fa*a.z + fb*b.z; };

	inline void set_lincomb( TYPE fa, TYPE fb, TYPE fc, const VEC& a, const VEC& b, const VEC& c ){ x = fa*a.x + fb*b.x + fc*c.x;  y = fa*a.y + fb*b.y + fc*c.y;  z = fa*a.z + fb*b.z + fc*c.z; };
	inline void add_lincomb( TYPE fa, TYPE fb, TYPE fc, const VEC& a, const VEC& b, const VEC& c ){ x+= fa*a.x + fb*b.x + fc*c.x;  y+= fa*a.y + fb*b.y + fc*c.y;  z+= fa*a.z + fb*b.z + fc*c.z; };

    inline void  set_cross( const VEC& a, const VEC& b ){ x =a.y*b.z-a.z*b.y; y =a.z*b.x-a.x*b.z; z =a.x*b.y-a.y*b.x; };
	inline void  add_cross( const VEC& a, const VEC& b ){ x+=a.y*b.z-a.z*b.y; y+=a.z*b.x-a.x*b.z; z+=a.x*b.y-a.y*b.x; };

	// ===== Opearators

	// do not use assignement operator because it is not obious what is ment
	// also there is conflict with Mat3d: member ‘VEC Mat3d::<anonymous union>::<anonymous struct>::a’ with copy assignment operator not allowed in anonymous aggregate
	//inline VEC& operator =( const TYPE f ) { x=f; y=f; z=f;       return *this; }; 
	//inline VEC& operator =( const VEC& v ) { x=v.x; y=v.y; z=v.z; return *this; }; 

    //inline VEC& operator+=( TYPE f ) { x+=f; y+=f; z+=f; return *this; };
    //inline VEC& operator*=( TYPE f ) { x*=f; y*=f; z*=f; return *this; };

    //inline VEC& operator+=( const VEC&  v ) { x+=v.x; y+=v.y; z+=v.z; return *this; };
    //inline VEC& operator-=( const VEC&  v ) { x-=v.x; y-=v.y; z-=v.z; return *this; };
    //inline VEC& operator*=( const VEC&  v ) { x*=v.x; y*=v.y; z*=v.z; return *this; };
    //inline VEC& operator/=( const VEC&  v ) { x/=v.x; y/=v.y; z/=v.z; return *this; };

	// This creates a new vectors? is it good?
    inline VEC operator+ ( TYPE f   ) const { VEC vo; vo.x=x+f; vo.y=y+f; vo.z=z+f; return vo; };
    inline VEC operator* ( TYPE f   ) const { VEC vo; vo.x=x*f; vo.y=y*f; vo.z=z*f; return vo; };

    inline VEC operator+ ( const VEC& vi ) const { VEC vo; vo.x=x+vi.x; vo.y=y+vi.y; vo.z=z+vi.z; return vo; };
    inline VEC operator- ( const VEC& vi ) const { VEC vo; vo.x=x-vi.x; vo.y=y-vi.y; vo.z=z-vi.z; return vo; };
    inline VEC operator* ( const VEC& vi ) const { VEC vo; vo.x=x*vi.x; vo.y=y*vi.y; vo.z=z*vi.z; return vo; };
    inline VEC operator/ ( const VEC& vi ) const { VEC vo; vo.x=x/vi.x; vo.y=y/vi.y; vo.z=z/vi.z; return vo; };

	inline TYPE dot  ( const VEC& a ) const { return x*a.x + y*a.y + z*a.z;  };
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

#endif

