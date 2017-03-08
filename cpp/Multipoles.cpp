
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "Vec3.h"
#include "Mat3.h"
//#include <string.h>

#include "Grid.h"

// ================= MACROS

//#define i3D( ix, iy, iz )  ( iz*nxy + iy*nx + ix  ) 

// ================= CONSTANTS

// ================= GLOBAL VARIABLES

GridShape gridShape;
double *  gridV;

/*
namespace GRID {		    
	double * grid;      // pointer to data array (3D)
	Mat3d   cell;
	Mat3d   dCell;      // basis vector of each voxel ( lattice vectors divided by number of points )
	Mat3d   diCell;     // inversion of voxel basis vector
	Vec3i   n;          // number of pixels along each basis vector
}
*/

const static int images [27][3] = {
	{ 0, 0, 0}, {-1, 0, 0}, {+1, 0, 0},
	{ 0,-1, 0}, {-1,-1, 0}, {+1,-1, 0},
	{ 0,+1, 0}, {-1,+1, 0}, {+1,+1, 0},
	{ 0, 0,-1}, {-1, 0,-1}, {+1, 0,-1},
	{ 0,-1,-1}, {-1,-1,-1}, {+1,-1,-1},
	{ 0,+1,-1}, {-1,+1,-1}, {+1,+1,-1},
	{ 0, 0,+1}, {-1, 0,+1}, {+1, 0,+1},
	{ 0,-1,+1}, {-1,-1,+1}, {+1,-1,+1},
	{ 0,+1,+1}, {-1,+1,+1}, {+1,+1,+1}
};

// =====================================================
// ==========   Export these functions ( to Python )
// =====================================================

extern "C"{

// set pointer to force field array ( the array is usually allocated in python, we can flexibely switch betweeen different precomputed forcefields )
void setGrid_Pointer( double * data ){
	gridV = data;
}

// set forcefield grid dimension "n"
void setGridN( int * n ){
	//gridShape.n.set( *(Vec3i*)n );
	gridShape.n.set( n[2], n[1], n[0] );
	printf( " nxyz  %i %i %i \n", gridShape.n.x, gridShape.n.y, gridShape.n.z );
}

// set forcefield grid lattice vectors "cell"
void setGridCell( double * cell ){
	gridShape.setCell( *(Mat3d*)cell );
    gridShape.printCell();
}

/*
inline atom_sphere( rProbe  const Vec3d& pos,    bool &withinRmax, bool withinRmin  ){
	Vec3d dr;
	dr.set_sub( atom_pos[iatom], rProbe ); 
	double r2   = dr.norm2();
	double rmin = atom_Rmin[iatom];
	double rmax = atom_Rmax[iatom];
	if( r2<(rmax*rmax) ){ 
		if( atom_mask[iatom] ){ withinRmax = true; }  
		if( r2<(rmin*rmin)   ){ withinRmin = true; }
	}
}
*/

//  sampleGridArroundAtoms
//  	takes from a 3D grid values at grid points which are in area between $atom_Rmin[i] to $atom_Rmaxs[i] distance from any i-th atom at position $atom_posp[i]
//		if $canStore == true it will save values to $sampled_val and postions to $sampled_pos else it only returns number of gridpoints fullfilling the conditions
int sampleGridArroundAtoms( 
	int natoms, Vec3d * atom_pos_, double * atom_Rmin, double * atom_Rmax, bool * atom_mask, 
	double * sampled_val, Vec3d * sampled_pos_, bool canStore, bool pbc, bool show_where
){
	Vec3d * atom_pos    = (Vec3d*) atom_pos_;
	Vec3d * sampled_pos = (Vec3d*) sampled_pos_;
	int nx  = gridShape.n.x;
	int ny  = gridShape.n.y;
	int nz  = gridShape.n.z;
	int nxy = ny * nx; // used in macro i3D( ia, ib, ic )
	Vec3d rProbe;  rProbe.set( 0.0, 0.0, 0.0 ); // we may shift here
	int points_found = 0;
	int nimg = 1; if (pbc) nimg = 27; // PBC ?
	for ( int ia=0; ia<nx; ia++ ){ 
		//printf( " ia %i \n", ia );
		rProbe.add( gridShape.dCell.a );  
		for ( int ib=0; ib<ny; ib++ ){ 
			rProbe.add( gridShape.dCell.b );
			for ( int ic=0; ic<nz; ic++ ){
				rProbe.add( gridShape.dCell.c );
				bool withinRmin = false; 
				bool withinRmax = false; 
				for ( int iimg=0; iimg<nimg; iimg++ ){
					Vec3d cell_shift;
					cell_shift.set_lincomb( images[iimg][0], images[iimg][1], images[iimg][2], gridShape.cell.a, gridShape.cell.b, gridShape.cell.c );
					for ( int iatom=0; iatom<natoms; iatom++ ){
						Vec3d dr;
						dr.set_sub( rProbe, atom_pos[iatom] );
						dr.sub( cell_shift ); 
						double r2   = dr.norm2();
						double rmin = atom_Rmin[iatom];
						double rmax = atom_Rmax[iatom];
						if( r2<(rmax*rmax) ){ 
							if( atom_mask[iatom] ){ withinRmax = true; }  
							if( r2<(rmin*rmin)   ){ withinRmin = true; break; }
						}
					}
				}
				if( withinRmax && (!withinRmin) ){
					if( canStore ){
						sampled_val[points_found] = gridV[ i3D( ia, ib, ic ) ];
						if( show_where ) gridV[ i3D( ia, ib, ic ) ] = +100.0d;
						sampled_pos[points_found].set( rProbe );
					}
					points_found++;
				}
			} 
			rProbe.add_mul( gridShape.dCell.c, -nz );
		} 
		rProbe.add_mul( gridShape.dCell.b, -ny );
	}
	return points_found;
}

}



