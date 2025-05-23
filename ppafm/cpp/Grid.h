
#ifndef Grid_h
#define Grid_h

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <omp.h>
#include "Vec3.h"
#include "Mat3.h"
//#include <string.h>

// ================= MACROS

#define fast_floor_offset  1000
#define fast_floor( x )    ( ( (int)( x + fast_floor_offset ) ) - fast_floor_offset )
#define i3D( ix, iy, iz )  ( (ix*ny + iy)*nz + iz  )

// ================= CONSTANTS

// Force-Field namespace
class GridShape {
	public:
	Vec3d   pos0;
	Mat3d   cell;       // lattice vector
	Mat3d   dCell;      // basis vector of each voxel ( lattice vectors divided by number of points )
	Mat3d   diCell;     // inversion of voxel basis vector
	Vec3i   n;          // number of pixels along each basis vector

	inline void setCell( const Mat3d& cell_ ){
		//n.set( n_ );
		cell.set( cell_ );
		dCell.a.set_mul( cell.a, 1.0/n.a );
		dCell.b.set_mul( cell.b, 1.0/n.b );
		dCell.c.set_mul( cell.c, 1.0/n.c );
		dCell.invert_T_to( diCell );
	};

	//inline void set( int * n_, double * cell_ ){ set( *(Vec3d*) n_, *(Mat3d*)cell_ ); };

	inline void grid2cartesian( const Vec3d& gpos, Vec3d& cpos ){
		cpos.set_mul( dCell.a, gpos.x );
		cpos.add_mul( dCell.b, gpos.y );
		cpos.add_mul( dCell.c, gpos.z );
	};

	inline void cartesian2grid( const Vec3d& cpos, Vec3d& gpos ){
		gpos.a = cpos.dot( diCell.a );
		gpos.b = cpos.dot( diCell.b );
		gpos.c = cpos.dot( diCell.c );
	};

	void printCell(){
	    printf( " n      %i %i %i \n", n.x,        n.y,       n.z        );
	    printf( " a      %f %f %f \n", cell.a.x,   cell.a.y,   cell.a.z  );
	    printf( " b      %f %f %f \n", cell.b.x,   cell.b.y,   cell.b.z  );
	    printf( " c      %f %f %f \n", cell.c.x,   cell.c.y,   cell.c.z  );
	    printf( " da     %f %f %f \n", dCell.a.x,  dCell.a.y,  dCell.a.z  );
	    printf( " db     %f %f %f \n", dCell.b.x,  dCell.b.y,  dCell.b.z  );
	    printf( " dc     %f %f %f \n", dCell.c.x,  dCell.c.y,  dCell.c.z  );
	    printf( " inv_da %f %f %f \n", diCell.a.x, diCell.a.y, diCell.a.z );
	    printf( " inv_db %f %f %f \n", diCell.b.x, diCell.b.y, diCell.b.z );
	    printf( " inv_dc %f %f %f \n", diCell.c.x, diCell.c.y, diCell.c.z );
    }

};

// interpolation of vector force-field Vec3d[ix,iy,iz] in periodic boundary condition
inline double interpolate3DWrap( double * grid, const Vec3i& n, const Vec3d& r ){
  //#pragma omp simd
  //{
  //int xoff = n.x<<3; int imx = r.x +xoff;	double tx = r.x - imx +xoff;	double mx = 1 - tx;		int itx = (imx+1)%n.x;  imx=imx%n.x;
  //int yoff = n.y<<3; int imy = r.y +yoff;	double ty = r.y - imy +yoff;	double my = 1 - ty;		int ity = (imy+1)%n.y;  imy=imy%n.y;
  //int zoff = n.z<<3; int imz = r.z +zoff;	double tz = r.z - imz +zoff;	double mz = 1 - tz;		int itz = (imz+1)%n.z;  imz=imz%n.z;
  int nx = n.x; int ny = n.y; int nz = n.z;

  int imx, imy, imz, itx, ity, itz;
  double mx, my, mz, tx, ty, tz;
  if(r.x >= 0) {imx = r.x; tx = r.x - imx; mx = 1 - tx; imx = imx % nx;}
  else {itx = r.x; mx = itx - r.x; tx = 1 - mx; imx = itx % nx + n.x-1;}
  itx = (imx+1) % nx;
  if(r.y >= 0) {imy = r.y; ty = r.y - imy; my = 1 - ty; imy = imy % ny;}
  else {ity = r.y; my = ity - r.y; ty = 1 - my; imy = ity % ny + n.y-1;}
  ity = (imy+1) % ny;
  if(r.z >= 0) {imz = r.z; tz = r.z - imz; mz = 1 - tz; imz = imz % nz;}
  else {itz = r.z; mz = itz - r.z; tz = 1 - mz; imz = itz % nz + n.z-1;}
  itz = (imz+1) % nz;

  double out = mz * (
		     my * ( ( mx * grid[ i3D( imx, imy, imz ) ] ) +  ( tx * grid[ i3D( itx, imy, imz ) ] ) ) +
		     ty * ( ( mx * grid[ i3D( imx, ity, imz ) ] ) +  ( tx * grid[ i3D( itx, ity, imz ) ] ) ) )
    + tz * (
	    my * ( ( mx * grid[ i3D( imx, imy, itz ) ] ) +  ( tx * grid[ i3D( itx, imy, itz ) ] ) ) +
	    ty * ( ( mx * grid[ i3D( imx, ity, itz ) ] ) +  ( tx * grid[ i3D( itx, ity, itz ) ] ) ) );
  //}
  return out;
}

// interpolation of vector force-field Vec3d[ix,iy,iz] in periodic boundary condition
inline Vec3d interpolate3DvecWrap( Vec3d * grid, const Vec3i& n, const Vec3d& r ){
  //#pragma omp simd
  //{
  //int xoff = n.x<<3; int imx = r.x +xoff;	double tx = r.x - imx +xoff;	double mx = 1 - tx;		int itx = (imx+1)%n.x;  imx=imx%n.x;
  //int yoff = n.y<<3; int imy = r.y +yoff;	double ty = r.y - imy +yoff;	double my = 1 - ty;		int ity = (imy+1)%n.y;  imy=imy%n.y;
  //int zoff = n.z<<3; int imz = r.z +zoff;	double tz = r.z - imz +zoff;	double mz = 1 - tz;		int itz = (imz+1)%n.z;  imz=imz%n.z;

  int nx = n.x; int ny = n.y; int nz = n.z;
  int imx, imy, imz, itx, ity, itz;
  double mx, my, mz, tx, ty, tz;
  if(r.x >= 0) {imx = r.x; tx = r.x - imx; mx = 1 - tx; imx = imx % nx;}
  else {itx = r.x; mx = itx - r.x; tx = 1 - mx; imx = itx % nx + n.x-1;}
  itx = (imx+1) % nx;
  if(r.y >= 0) {imy = r.y; ty = r.y - imy; my = 1 - ty; imy = imy % ny;}
  else {ity = r.y; my = ity - r.y; ty = 1 - my; imy = ity % ny + n.y-1;}
  ity = (imy+1) % ny;
  if(r.z >= 0) {imz = r.z; tz = r.z - imz; mz = 1 - tz; imz = imz % nz;}
  else {itz = r.z; mz = itz - r.z; tz = 1 - mz; imz = itz % nz + n.z-1;}
  itz = (imz+1) % nz;

  double mymx = my*mx; double mytx = my*tx; double tymx = ty*mx; double tytx = ty*tx;
  Vec3d out;
  out.set_mul( grid[ i3D( imx, imy, imz ) ], mz*mymx );   out.add_mul( grid[ i3D( itx, imy, imz ) ], mz*mytx );
  out.add_mul( grid[ i3D( imx, ity, imz ) ], mz*tymx );   out.add_mul( grid[ i3D( itx, ity, imz ) ], mz*tytx );
  out.add_mul( grid[ i3D( imx, ity, itz ) ], tz*tymx );   out.add_mul( grid[ i3D( itx, ity, itz ) ], tz*tytx );
  out.add_mul( grid[ i3D( imx, imy, itz ) ], tz*mymx );   out.add_mul( grid[ i3D( itx, imy, itz ) ], tz*mytx );
  //printf( "DEBUG interpolate3DvecWrap gp(%g,%g,%g) igp(%i,%i,%i)/(%i,%i,%i) %i->%g out(%g,%g,%g) \n", r.x, r.y, r.z, imx, imy, imz, n.x,n.y,n.z, i3D( imx, imy, imz ), grid[ i3D( imx, imy, imz ) ], out.x,out.y,out.z );
  //}
  return out;
}

// iterate over field
template< void FUNC( int ibuff, const Vec3d& pos_, void * args ) >
void iterateGrid3D( const Vec3d& pos0, const Vec3i& n, const Mat3d& dCell, void * args ){
	int nx  = n.x; 	int ny  = n.y; 	int nz  = n.z;
	printf( "iterateGrid3D nx,y,z (%i,%i,%i)\n", nx, ny, nz);
	Vec3d pos;  pos.set( pos0 );
	//printf(" iterateGrid3D : args %i \n", args );
	int ibuff = 0;
	for ( int ia=0; ia<nx; ia++ ){
	  std::cout << "ia " << ia;
	  std::cout.flush();
	  std::cout << '\r';
	  for ( int ib=0; ib<ny; ib++ ){
	    for ( int ic=0; ic<nz; ic++ ){
	      //ibuff = i3D( ia, ib, ic );
	      //pos = pos0 + dCell.c*ic + dCell.b*ib + dCell.a*ia;
	      FUNC( ++ibuff, pos, args );
	      //printf("(%i,%i,%i)(%3.3f,%3.3f,%3.3f)\n",ia,ib,ic,pos.x,pos.y,pos.z);
	      pos.add( dCell.c );
	    }
	    pos.add_mul( dCell.c, -nz );
	    pos.add( dCell.b );
	  }
	  //exit(0);
	  pos.add_mul( dCell.b, -ny );
	  pos.add( dCell.a );
	}
    printf ("\n");
}

template< void FUNC( int ibuff, const Vec3d& pos_, void * args ) >
void iterateGrid3D_omp( const Vec3d& pos0, const Vec3i& n, const Mat3d& dCell, void * args ){
    int ntot = n.x*n.y*n.z;
    int ncpu = omp_get_num_threads(); printf( "iterateGrid3D_omp nx,y,z (%i,%i,%i) ntot %i ncpu %i \n",  n.x,n.y,n.z, ntot, ncpu );
    int ndone=0;
    #pragma omp parallel for collapse(3) shared(pos0,n,dCell,args,ndone)
    for ( int ia=0; ia<n.x; ia++ ){
        for ( int ib=0; ib<n.y; ib++ ){
            for ( int ic=0; ic<n.z; ic++ ){
                Vec3d pos = pos0 + dCell.a*ia + dCell.b*ib + dCell.c*ic;
                int ibuff = (ia*n.y + ib)*n.z + ic;
                //ndone[ omp_get_thread_num() ]++;
                if( omp_get_thread_num()==0 ){
                    ndone++;
                    if( ndone%10000==0 ){
                        int ncpu=omp_get_num_threads();
                        printf( "\r %2.2f %% DONE (ncpu=%i)", 100.0*ndone*ncpu / ntot, ncpu );
                        fflush(stdout);
                    }
                }
                //if( ibuff%100000==0 ){ printf( "cpu[%i/%i] progress  %2.2f )\n",  omp_get_thread_num(), omp_get_num_threads(),  100.0*ndone / ntot ); }
                FUNC( ibuff, pos, args );
            }
        }
    }
    printf( "\n" );
    //printf( "DONE ndone %i ntot %i \n", ndone, ntot );
}


#endif
