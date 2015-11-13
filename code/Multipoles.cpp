
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "Vec3.cpp"
#include "Mat3.cpp"
#include <string.h>

// ================= MACROS

#define i3D( ix, iy, iz )  ( iz*nxy + iy*nx + ix  ) 

// ================= CONSTANTS

// Force-Field namespace 
namespace GRID {		    
	double * grid;      // pointer to data array (3D)
	Mat3d   dCell;      // basis vector of each voxel ( lattice vectors divided by number of points )
	Mat3d   diCell;     // inversion of voxel basis vector
	Vec3i   n;          // number of pixels along each basis vector
}

// =====================================================
// ==========   Export these functions ( to Python )
// =====================================================

extern "C"{

// set pointer to force field array ( the array is usually allocated in python, we can flexibely switch betweeen different precomputed forcefields )
void setGrid_Pointer( double * grid ){
	GRID::grid = grid;
}

// set parameters of forcefield like dimension "n", and lattice vectors "cell"
void setGrid( int * n, double * grid, double * cell ){
	GRID::grid = grid;
	GRID::n.set(n);
	GRID::dCell.a.set( cell[0], cell[1], cell[2] );   GRID::dCell.a.mul( 1.0d/GRID::n.a );
	GRID::dCell.b.set( cell[3], cell[4], cell[5] );   GRID::dCell.b.mul( 1.0d/GRID::n.b );
	GRID::dCell.c.set( cell[6], cell[7], cell[8] );   GRID::dCell.c.mul( 1.0d/GRID::n.c );
	GRID::dCell.invert_T_to( GRID::diCell );	
	printf( " nxyz  %i %i %i \n", GRID::n.x, GRID::n.y, GRID::n.z );
	printf( " a     %f %f %f \n", GRID::dCell.a.x,GRID::dCell.a.y,GRID::dCell.a.z );
	printf( " b     %f %f %f \n", GRID::dCell.b.x,GRID::dCell.b.y,GRID::dCell.b.z );
	printf( " c     %f %f %f \n", GRID::dCell.c.x,GRID::dCell.c.y,GRID::dCell.c.z );
	printf( " inv_a %f %f %f \n", GRID::diCell.a.x,GRID::diCell.a.y,GRID::diCell.a.z );
	printf( " inv_b %f %f %f \n", GRID::diCell.b.x,GRID::diCell.b.y,GRID::diCell.b.z );
	printf( " inv_c %f %f %f \n", GRID::diCell.c.x,GRID::diCell.c.y,GRID::diCell.c.z );
}

//  sampleGridArroundAtoms
//  	takes from a 3D grid values at grid points which are in area between $atom_Rmin[i] to $atom_Rmaxs[i] distance from any i-th atom at position $atom_posp[i]
//		if $canStore == true it will save values to $sampled_val and postions to $sampled_pos else it only returns number of gridpoints fullfilling the conditions
int sampleGridArroundAtoms( 
	int natoms, Vec3d * atom_pos_, double * atom_Rmin, double * atom_Rmax, bool * atom_mask, 
	double * sampled_val, Vec3d * sampled_pos_, bool canStore
){
	Vec3d * atom_pos    = (Vec3d*) atom_pos_;
	Vec3d * sampled_pos = (Vec3d*) sampled_pos_;
	int nx  = GRID::n.x;
	int ny  = GRID::n.y;
	int nz  = GRID::n.z;
	int nxy = ny * nx; // used in macro i3D( ia, ib, ic )
	Vec3d rProbe;  rProbe.set( 0.0, 0.0, 0.0 ); // we may shift here
	int points_found = 0;
	for ( int ia=0; ia<nx; ia++ ){ 
		//printf( " ia %i \n", ia );
		rProbe.add( GRID::dCell.a );  
		for ( int ib=0; ib<ny; ib++ ){ 
			rProbe.add( GRID::dCell.b );
			for ( int ic=0; ic<nz; ic++ ){
				rProbe.add( GRID::dCell.c );
				bool withinRmin = false; 
				bool withinRmax = false; 
				for ( int iatom=0; iatom<natoms; iatom++ ){
					Vec3d dr;
					dr.set_sub( atom_pos[iatom], rProbe ); 
					double r2   = dr.norm2();
					double rmin = atom_Rmin[iatom];
					double rmax = atom_Rmax[iatom];
					if( r2<(rmax*rmax) ){ 
						if( atom_mask[iatom] ){ withinRmax = true; }  
						if( r2<(rmin*rmin)   ){ withinRmin = true; break; }
					}
				}
				if( withinRmax && (!withinRmin) ){
					if( canStore ){
						sampled_val[points_found] = GRID::grid[ i3D( ia, ib, ic ) ];
						sampled_pos[points_found].set( rProbe );
					}
					points_found++;
				}
			} 
			rProbe.add_mul( GRID::dCell.c, -nz );
		} 
		rProbe.add_mul( GRID::dCell.b, -ny );
	}
	return points_found;
}

}












