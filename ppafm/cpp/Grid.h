
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

#define i3D( ix, iy, iz )  ( (iz*n.y + iy)*n.x + ix  )

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
		printf( " n      %i %i %i \n", n.x,        n.y,        n.z        );
		printf( " a      %f %f %f \n", cell.a.x,   cell.a.y,   cell.a.z   );
		printf( " b      %f %f %f \n", cell.b.x,   cell.b.y,   cell.b.z   );
		printf( " c      %f %f %f \n", cell.c.x,   cell.c.y,   cell.c.z   );
		printf( " da     %f %f %f \n", dCell.a.x,  dCell.a.y,  dCell.a.z  );
		printf( " db     %f %f %f \n", dCell.b.x,  dCell.b.y,  dCell.b.z  );
		printf( " dc     %f %f %f \n", dCell.c.x,  dCell.c.y,  dCell.c.z  );
		printf( " inv_da %f %f %f \n", diCell.a.x, diCell.a.y, diCell.a.z );
		printf( " inv_db %f %f %f \n", diCell.b.x, diCell.b.y, diCell.b.z );
		printf( " inv_dc %f %f %f \n", diCell.c.x, diCell.c.y, diCell.c.z );
	}
};

#define SET_GRID_IND(a) \
  if (r.a >= 0) {im.a = r.a; t.a = r.a - im.a; m.a = 1 - t.a; im.a %= n.a;} \
  else {it.a = r.a; m.a = it.a - r.a; t.a = 1 - m.a; im.a = it.a % n.a + n.a - 1;} \
  it.a = (im.a + 1) % n.a

// interpolation of vector force-field Vec3d[ix,iy,iz] in periodic boundary condition
inline double interpolate3DWrap( double * grid, const Vec3i& n, const Vec3d& r ){
  //#pragma omp simd
  //{
  struct {double x, y, z;} m, t;
  struct {int x, y, z;} im, it;
  SET_GRID_IND(x);
  SET_GRID_IND(y);
  SET_GRID_IND(z);

  double out =
    m.z * (
	  m.y * ( ( m.x * grid[ i3D( im.x, im.y, im.z ) ] ) + ( t.x * grid[ i3D( it.x, im.y, im.z ) ] ) ) +
	  t.y * ( ( m.x * grid[ i3D( im.x, it.y, im.z ) ] ) + ( t.x * grid[ i3D( it.x, it.y, im.z ) ] ) ) ) +
    t.z * (
	  m.y * ( ( m.x * grid[ i3D( im.x, im.y, it.z ) ] ) + ( t.x * grid[ i3D( it.x, im.y, it.z ) ] ) ) +
	  t.y * ( ( m.x * grid[ i3D( im.x, it.y, it.z ) ] ) + ( t.x * grid[ i3D( it.x, it.y, it.z ) ] ) ) );
  //}
  return out;
}

// interpolation of vector force-field Vec3d[ix,iy,iz] in periodic boundary condition
inline Vec3d interpolate3DvecWrap( Vec3d * grid, const Vec3i& n, const Vec3d& r ){
  //#pragma omp simd
  ////{
  struct {double x, y, z;} m, t;
  struct {int x, y, z;} im, it;
  SET_GRID_IND(x);
  SET_GRID_IND(y);
  SET_GRID_IND(z);

  double mymx = m.y*m.x; double mytx = m.y*t.x; double tymx = t.y*m.x; double tytx = t.y*t.x;
  Vec3d out;
  out.set_mul( grid[ i3D( im.x, im.y, im.z ) ], m.z*mymx );  out.add_mul( grid[ i3D( it.x, im.y, im.z ) ], m.z*mytx );
  out.add_mul( grid[ i3D( im.x, it.y, im.z ) ], m.z*tymx );  out.add_mul( grid[ i3D( it.x, it.y, im.z ) ], m.z*tytx );
  out.add_mul( grid[ i3D( im.x, it.y, it.z ) ], t.z*tymx );  out.add_mul( grid[ i3D( it.x, it.y, it.z ) ], t.z*tytx );
  out.add_mul( grid[ i3D( im.x, im.y, it.z ) ], t.z*mymx );  out.add_mul( grid[ i3D( it.x, im.y, it.z ) ], t.z*mytx );
  //printf( "DEBUG interpolate3DvecWrap gp(%g,%g,%g) igp(%i,%i,%i)/(%i,%i,%i) %i->%g out(%g,%g,%g) \n", r.x, r.y, r.z, im.x, im.y, im.z, n.x, n.y, n.z, i3D( im.x, im.y, im.z ), grid[ i3D( im.x, im.y, im.z ) ], out.x,out.y,out.z );
  //}
  return out;
}

// iterate over field
template< void FUNC( int ibuff, const Vec3d& pos_, void * args ) >
void iterateGrid3D( const Vec3d& pos0, const Vec3i& n, const Mat3d& dCell, void * args ){
	int nx  = n.x; int ny  = n.y; int nz  = n.z;
	printf( "iterateGrid3D nx,y,z (%i,%i,%i)\n", nx, ny, nz );
	Vec3d pos; pos.set( pos0 );
	//printf( " iterateGrid3D : args %i \n", args );
	for ( int ic=0; ic<nz; ic++ ){
		std::cout << "ic " << ic;
		std::cout.flush();
		std::cout << '\r';
		for ( int ib=0; ib<ny; ib++ ){
			for ( int ia=0; ia<nx; ia++ ){
				int ibuff = i3D( ia, ib, ic );
				//FUNC( ibuff, {ia, ib, ic}, pos );
				//pos = pos0 + dCell.c*ic + dCell.b*ib + dCell.a*ia;
				FUNC( ibuff, pos, args );
				//printf( "(%i,%i,%i)(%3.3f,%3.3f,%3.3f)\n", ia, ib, ic, pos.x, pos.y, pos.z );
				pos.add( dCell.a );
			}
			pos.add_mul( dCell.a, -nx );
			pos.add( dCell.b );
		}
	pos.add_mul( dCell.b, -ny );
	pos.add( dCell.c );
	}
	printf ( "\n" );
}

template< void FUNC( int ibuff, const Vec3d& pos_, void * args ) >
void iterateGrid3D_omp( const Vec3d& pos0, const Vec3i& n, const Mat3d& dCell, void * args ){
	int ntot = n.x * n.y * n.z;
	int ncpu = omp_get_num_threads(); printf( "iterateGrid3D_omp nx,y,z (%i,%i,%i) ntot %i ncpu %i \n", n.x, n.y, n.z, ntot, ncpu );
	int ndone=0;
	#pragma omp parallel for collapse(3) shared(pos0,n,dCell,args,ndone)
	for ( int ic=0; ic<n.z; ic++ ){
		for ( int ib=0; ib<n.y; ib++ ){
			for ( int ia=0; ia<n.x; ia++ ){
				Vec3d pos = pos0 + dCell.c*ic + dCell.b*ib + dCell.a*ia;
				//int ibuff = i3D( ia, ib, ic );
				int ibuff = ic*(n.x*n.y) + ib*n.x + ia;
				//ndone[ omp_get_thread_num() ]++;
				if( omp_get_thread_num()==0 ){
					ndone++;
					if( ndone%10000==0 ){
						int ncpu=omp_get_num_threads();
						printf( "\r %2.2f %% DONE (ncpu=%i)", 100.0*ndone*ncpu / ntot, ncpu );
						fflush(stdout);
					}
				}
				//if( ibuff%100000==0 ){ printf( "cpu[%i/%i] progress  %2.2f )\n", omp_get_thread_num(), omp_get_num_threads(), 100.0*ndone / ntot ); }
				FUNC( ibuff, pos, args );
			}
		}
	}
	printf( "\n" );
	//printf( "DONE ndone %i ntot %i \n", ndone, ntot );
}

#endif
