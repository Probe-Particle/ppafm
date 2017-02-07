
#ifndef Mat3_h 
#define Mat3_h 

//template <class TYPE, class VEC> 
//template <class TYPE> 

template <class TYPE, class VEC, class MAT> 
class Mat3TYPE{
//	using VEC = Vec3TYPE<TYPE>; 
//	using MAT = Mat3TYPE<TYPE>; 
	public:
	union{
		struct{
			TYPE xx,xy,xz;
			TYPE yx,yy,yz;
			TYPE zx,zy,zz;
		};
		struct{
			TYPE ax,ay,az;
			TYPE bx,by,bz;
			TYPE cx,cy,cz;
		};
		struct{	VEC a,b,c; };
		TYPE array[9];
	};


// ====== initialization

	inline void setOne(          ){ xx=yy=zz=1; xy=xz=yx=yz=zx=zy=0; };
	inline void set   ( TYPE f ){ xx=yy=zz=f; xy=xz=yx=yz=zx=zy=0; };

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
	};; 

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

	inline void T(){
		TYPE t1=yx; yx=xy; xy=t1;  
		TYPE t2=zx; zx=xz; xz=t2;
		TYPE t3=zy; zy=yz; yz=t3;
	};

	inline void setT  ( const MAT& M ){
		xx=M.xx; xy=M.xy; xz=M.xz;
		yx=M.yx; yy=M.yy; yz=M.yz;
		zx=M.zx; zy=M.zy; zz=M.zz;
	};

	inline void setT  ( const VEC& va, const VEC& vb, const VEC& vc ){
		a.set( va.x, vb.x, vc.x ); 
		b.set( va.y, vb.y, vc.y ); 
		c.set( va.z, vb.z, vc.z );
	};

// ====== dot product with vector

	inline VEC dot( const VEC&  v ) const {
		VEC vout;
		vout.x = xx*v.x + xy*v.y + xz*v.z;  
		vout.y = yx*v.x + yy*v.y + yz*v.z;  
		vout.z = zx*v.x + zy*v.y + zz*v.z;  
		return vout;
	}

	inline void dot_to( const VEC&  v, VEC&  vout ) const {
		vout.x = xx*v.x + xy*v.y + xz*v.z;  
		vout.y = yx*v.x + yy*v.y + yz*v.z;  
		vout.z = zx*v.x + zy*v.y + zz*v.z;  
	};

	inline void dot_to_T( const VEC&  v, VEC&  vout ) const {
		vout.x = xx*v.x + yx*v.y + zx*v.z;  
		vout.y = xy*v.x + yy*v.y + zy*v.z;  
		vout.z = xz*v.x + yz*v.y + zz*v.z;  
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

   inline TYPE determinant() {
        TYPE fCoxx = yy * zz - yz * zy;
        TYPE fCoyx = yz * zx - yx * zz;
        TYPE fCozx = yx * zy - yy * zx;
        TYPE fDet = xx * fCoxx + xy * fCoyx + xz * fCozx;
        return fDet;
    };

	inline void invert_to( MAT& Mout ) {
        TYPE idet = 1/determinant(); // we dont check det|M|=0
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
        TYPE idet = 1/determinant(); // we dont check det|M|=0
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

	inline void fromDirUp( const VEC&  dir, const VEC&  up ){
		// make orthonormal rotation matrix c=dir; b=(up-<b|c>c)/|b|; a=(c x b)/|a|;
		c.set(dir);
		//c.normalize(); // we assume dir is already normalized
		b.set(up);
		b.add_mul( c, -b.dot(c) );   // 
		b.normalize();
		a.set_cross(b,c);
		a.normalize();
	}

};

class Mat3i : public Mat3TYPE< int     , Vec3i, Mat3i >{};
//class Mat3f : public Mat3TYPE< float , Vec3f, Mat3f >{};
class Mat3d : public Mat3TYPE< double , Vec3d, Mat3d >{};

/*
using Mat3i = Mat3TYPE< int   >;
using Mat3f = Mat3TYPE< float >;
using Mat3d = Mat3TYPE< double>;
*/

#endif

