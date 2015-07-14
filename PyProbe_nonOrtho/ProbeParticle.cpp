
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "Vec3.cpp"
#include "Mat3.cpp"

// ================= MACROS

#define fast_floor( x )    ( ((int)(x+1000))-1000 ) 
#define i3D( ix, iy, iz )  ( iz*nxy + iy*nx + ix  ) 

// ================= CONSTANTS

// ================= GLOBAL VARIABLES


// Force-Field namespace 
namespace FF {		    
	Vec3d * grid;       // pointer to data array (3D)
	Mat3d   dCell;      // basis vector of each voxel ( lattice vectors divided by number of points )
	Mat3d   diCell;     // inversion of voxel basis vector
	Vec3i   n;          // number of pixels along each basis vector
}

// Tip namespace 
namespace TIP{
	Vec3d   rPP0;       // equilibirum bending position
	Vec3d   kSpring;    // bending stiffness ( z component usually zero )
	double  lRadial;    // radial PP-tip distance
	double  kRadial;    // radial PP-tip stiffness

	void makeConsistent(){ // place rPP0 on the sphere to be consistent with radial spring
		if( fabs(kRadial) > 1e-8 ){  
			rPP0.z = -sqrt( lRadial*lRadial - rPP0.x*rPP0.x - rPP0.y*rPP0.y );  
			printf(" rPP0 %f %f %f \n", rPP0.x, rPP0.y, rPP0.z );
		}
	}
}

// relaxation namespace 
namespace RELAX{
	// parameters
	int    maxIters  = 1000;          // maximum iterations steps for each pixel
	double convF2    = 1.0e-8;        // square of convergence criterium ( convergence achieved when |F|^2 < convF2 )
	double dt        = 0.5;           // time step [ abritrary units ]
	double damping   = 0.1;           // velocity damping ( like friction )  v_(i+1) = v_i * ( 1- damping )

	// relaxation step for simple damped-leap-frog molecular dynamics ( just for testing, less efficinet than FIRE )
	inline void  move( const Vec3d& f, Vec3d& r, Vec3d& v ){
		v.mul( 1 - damping );
		v.add_mul( f, dt );
		r.add_mul( v, dt );	
	}
}

// Fast Inertial Realxation Engine namespace 
namespace FIRE{
// "Fast Inertial Realxation Engine" according to  
// Bitzek, E., Koskinen, P., Gähler, F., Moseler, M. & Gumbsch, P. Structural relaxation made simple. Phys. Rev. Lett. 97, 170201 (2006).
// Eidel, B., Stukowski, A. & Schröder, J. Energy-Minimization in Atomic-to-Continuum Scale-Bridging Methods. Pamm 11, 509–510 (2011).
// http://users.jyu.fi/~pekkosk/resources/pdf/FIRE.pdf

	// parameters
	double finc    = 1.1;             // factor by which time step is increased if going downhill
	double fdec    = 0.5;             // factor by which timestep is decreased if going uphill 
	double falpha  = 0.99;            // rate of decrease of damping when going downhill 
	double dtmax   = RELAX::dt;       // maximal timestep
	double acoef0  = RELAX::damping;  // default damping

	// variables
	double dt      = dtmax;           // time-step ( variable
	double acoef   = acoef0;          // damping  ( variable

	inline void setup(){
		dtmax   = RELAX::dt; 
		acoef0  = RELAX::damping;
		dt      = dtmax;
		acoef   = acoef0;
	}

	// relaxation step using FIRE algorithm
	inline void move( const Vec3d& f, Vec3d& r, Vec3d& v ){
		double ff = f.norm2();
		double vv = v.norm2();
		double vf = f.dot(v);
		if( vf < 0 ){ // if velocity along direction of force
			v.set( 0.0d );
			dt    = dt * fdec;
		  	acoef = acoef0;
		}else{       // if velocity against direction of force
			double cf  =     acoef * sqrt(vv/ff);
			double cv  = 1 - acoef;
			v.mul    ( cv );
			v.add_mul( f, cf );	// v = cV * v  + cF * F
			dt     = fmin( dt * finc, dtmax );
			acoef  = acoef * falpha;
		}
		// normal leap-frog times step
		v.add_mul( f , dt );
		r.add_mul( v , dt );	
	}
}

// ========== Classical force field


// radial spring constrain - force length of vector |dR| to be l0
inline Vec3d forceRSpring( const Vec3d& dR, double k, double l0 ){
	double l = sqrt( dR.norm2() );
	Vec3d f; f.set_mul( dR, k*( l - l0 )/l );
	return f;
}

// Lenard-Jones force between two atoms a,b separated by vector dR = Ra - Rb
inline Vec3d forceLJ( const Vec3d& dR, double c6, double c12 ){
	double ir2  = 1.0d/ dR.norm2( ); 
	double ir6  = ir2*ir2*ir2;
	double ir12 = ir6*ir6;
	return dR * ( ( 6*ir6*c6 -12*ir12*c12 ) * ir2  );
}

// coulomb force between two atoms a,b separated by vector dR = R1 - R2, with constant kqq should be set to kqq = - k_coulomb * Qa * Qb 
inline Vec3d forceCoulomb( const Vec3d& dR, double kqq ){
	//const double kcoulomb   = 14.3996448915; 
	double ir2  = 1.0d/ dR.norm2( ); 
	double ir   = sqrt( ir2 );
	return dR * kqq * ir * ir2;
}

// Lenard-Jones force between Probe-Particle (rProbe) and n other atoms
inline Vec3d getAtomsForceLJ( const Vec3d& rProbe, int n, Vec3d * Rs, double * C6, double * C12 ){
	Vec3d f; f.set(0.0d);
	for(int i=0; i<n; i++){
		f.add( forceLJ( Rs[i] - rProbe, C6[i], C12[i] ) );
	}
	return f;
}

// Coulomb force between Probe-Particle (rProbe) and n other atoms
inline Vec3d getAtomsForceCoulomb( const Vec3d& rProbe, int n, Vec3d * Rs, double * kQQs ){
	Vec3d f; f.set(0.0d);
	for(int i=0; i<n; i++){	f.add( forceCoulomb( Rs[i] - rProbe, kQQs[i] ) );	}
	//for(int i=0; i<n; i++){	f.add( kQQs[i]  );	}
	return f;
}




// ========== Interpolations

// interpolation of vector force-field Vec3d[ix,iy,iz] in periodic boundary condition
inline Vec3d interpolate3DvecWrap( Vec3d * grid, const Vec3i& n, const Vec3d& r ){
	int xoff = n.x<<3; int imx = r.x +xoff;	double tx = r.x - imx +xoff;	double mx = 1 - tx;		int itx = (imx+1)%n.x;  imx=imx%n.x;
	int yoff = n.y<<3; int imy = r.y +yoff;	double ty = r.y - imy +yoff;	double my = 1 - ty;		int ity = (imy+1)%n.y;  imy=imy%n.y;
	int zoff = n.z<<3; int imz = r.z +zoff;	double tz = r.z - imz +zoff;	double mz = 1 - tz;		int itz = (imz+1)%n.z;  imz=imz%n.z;
	int nxy = n.x * n.y; int nx = n.x;
	//printf( " %f %f %f   %i %i %i \n", r.x, r.y, r.z, imx, imy, imz );
	double mymx = my*mx; double mytx = my*tx; double tymx = ty*mx; double tytx = ty*tx;
	Vec3d out;
	out.set_mul( grid[ i3D( imx, imy, imz ) ], mz*mymx );   out.add_mul( grid[ i3D( itx, imy, imz ) ], mz*mytx );
	out.add_mul( grid[ i3D( imx, ity, imz ) ], mz*tymx );   out.add_mul( grid[ i3D( itx, ity, imz ) ], mz*tytx );    
	out.add_mul( grid[ i3D( imx, ity, itz ) ], tz*tymx );   out.add_mul( grid[ i3D( itx, ity, itz ) ], tz*tytx );
	out.add_mul( grid[ i3D( imx, imy, itz ) ], tz*mymx );   out.add_mul( grid[ i3D( itx, imy, itz ) ], tz*mytx );
	return out;
}

// relax probe particle position "r" given on particular position of tip (rTip) and initial position "r" 
inline int relaxProbe( int relaxAlg, const Vec3d& rTip, Vec3d& r ){
	Vec3d v; v.set( 0.0d );
	int iter;
	//printf( " alg %i r  %f %f %f  rTip  %f %f %f \n", relaxAlg, r.x,r.y,r.z,  rTip.x, rTip.y, rTip.z );

	for( iter=0; iter<RELAX::maxIters; iter++ ){
		Vec3d rGrid,f,drTip; 
		//rGrid.set_mul(r, FF::invStep );                                                     // transform position from cartesian world coordinates to coordinates along which Force-Field data are sampled (     orthogonal cell )
		rGrid.set( r.dot( FF::diCell.a ), r.dot( FF::diCell.b ), r.dot( FF::diCell.c ) );     // transform position from cartesian world coordinates to coordinates along which Force-Field data are sampled ( non-orthogonal cell )
		drTip.set_sub( r, rTip );                                                             // vector between Probe-particle and tip apex
		f.set    ( interpolate3DvecWrap( FF::grid, FF::n, rGrid ) );                          // force from surface, interpolated from Force-Field data array
		f.add    ( forceRSpring( drTip, TIP::kRadial, TIP::lRadial ) );                       // force from tip - radial component 
		drTip.sub( TIP::rPP0 );
		f.add_mul( drTip, TIP::kSpring );      // spring force                                // force from tip - lateral bending force 
		if( relaxAlg == 1 ){                                                                  // move by either damped-leap-frog ( 0 ) or by FIRE ( 1 )
			FIRE::move( f, r, v );
		}else{
			RELAX::move( f, r, v );
		}			
		//printf( "     %i r  %f %f %f  f  %f %f %f \n", iter, r.x,r.y,r.z,  f.x,f.y,f.z );
		if( f.norm2() < RELAX::convF2 ) break;                                                // check force convergence
	}
	return iter;
}


// =====================================================
// ==========   Export these functions ( to Python )
// ========================================================

extern "C"{

// set basic relaxation parameters
void setRelax( int maxIters, double convF2, double dt, double damping ){
	RELAX::maxIters  = maxIters ;
	RELAX::convF2    = convF2;
	RELAX::dt        = dt;
	RELAX::damping   = damping;
	FIRE ::setup();
}

// set FIRE relaxation parameters
void setFIRE( double finc, double fdec, double falpha ){
	FIRE::finc    = finc; 
	FIRE::fdec    = fdec;
	FIRE::falpha  = falpha;
}

// set pointer to force field array ( the array is usually allocated in python, we can flexibely switch betweeen different precomputed forcefields )
void setFF_Pointer( double * grid ){
	FF::grid = (Vec3d *)grid;
}

// set parameters of forcefield like dimension "n", and lattice vectors "cell"
void setFF( int * n, double * grid, double * cell ){
	FF::grid = (Vec3d *)grid;
	FF::n.set(n);
	FF::dCell.a.set( cell[0], cell[1], cell[2] );   FF::dCell.a.mul( 1.0d/FF::n.a );
	FF::dCell.b.set( cell[3], cell[4], cell[5] );   FF::dCell.b.mul( 1.0d/FF::n.b );
	FF::dCell.c.set( cell[6], cell[7], cell[8] );   FF::dCell.c.mul( 1.0d/FF::n.c );
	FF::dCell.invert_T_to( FF::diCell );	

	printf( " nxyz  %i %i %i \n", FF::n.x, FF::n.y, FF::n.z );
	printf( " a     %f %f %f \n", FF::dCell.a.x,FF::dCell.a.y,FF::dCell.a.z );
	printf( " b     %f %f %f \n", FF::dCell.b.x,FF::dCell.b.y,FF::dCell.b.z );
	printf( " c     %f %f %f \n", FF::dCell.c.x,FF::dCell.c.y,FF::dCell.c.z );
	printf( " inv_a %f %f %f \n", FF::diCell.a.x,FF::diCell.a.y,FF::diCell.a.z );
	printf( " inv_b %f %f %f \n", FF::diCell.b.x,FF::diCell.b.y,FF::diCell.b.z );
	printf( " inv_c %f %f %f \n", FF::diCell.c.x,FF::diCell.c.y,FF::diCell.c.z );
}

// set parameters of the tip like stiffness and equlibirum position in radial and lateral direction
void setTip( double lRad, double kRad, double * rPP0, double * kSpring ){  
	TIP::lRadial=lRad; 
	TIP::kRadial=kRad;  
	TIP::rPP0.set(rPP0);   
	TIP::kSpring.set(kSpring); 
	TIP::makeConsistent();  // rPP0 to be consistent with  lRadial
}

// sample Lenard-Jones Force-field on 3D mesh over provided set of atoms with positions Rs_[i] with given C6 and C12 parameters; 
// results are sampled according to grid parameters defined in "namespace FF" and stored in array to which points by "double * FF::grid"
void getLenardJonesFF( int natom, double * Rs_, double * C6, double * C12 ){
	Vec3d * Rs = (Vec3d*) Rs_;
	int nx  = FF::n.x;
	int ny  = FF::n.y;
	int nz  = FF::n.z;
	int nxy = ny * nx;
	Vec3d rProbe;  rProbe.set( 0.0, 0.0, 0.0 ); // we may shift here
	for ( int ia=0; ia<nx; ia++ ){ 
		printf( " ia %i \n", ia );
		rProbe.add( FF::dCell.a );  
		for ( int ib=0; ib<ny; ib++ ){ 
			rProbe.add( FF::dCell.b );
			for ( int ic=0; ic<nz; ic++ ){
				rProbe.add( FF::dCell.c );
				FF::grid[ i3D( ia, ib, ic ) ].add( getAtomsForceLJ( rProbe, natom, Rs, C6, C12 ) );
				//printf(  " %i %i %i     %f %f %f  \n", ia, ib, ic,     rProbe.x, rProbe.y, rProbe.z  );
				//FF[ i3D( ix, iy, iz ) ].set( rProbe );
			} 
			rProbe.add_mul( FF::dCell.c, -nz );
		} 
		rProbe.add_mul( FF::dCell.b, -ny );
	}
}

// sample Coulomb Force-field on 3D mesh over provided set of atoms with positions Rs_[i] with constant kQQs  =  - k_coulomb * Q_ProbeParticle * Q[i] 
// results are sampled according to grid parameters defined in "namespace FF" and stored in array to which points by "double * FF::grid"
void getCoulombFF( int natom, double * Rs_, double * kQQs ){
	Vec3d * Rs = (Vec3d*) Rs_;
	int nx  = FF::n.x;
	int ny  = FF::n.y;
	int nz  = FF::n.z;
	int nxy = ny * nx;
	Vec3d rProbe;  rProbe.set( 0.0, 0.0, 0.0 ); // we may shift here
	//for ( int i=0; i<natom; i++ ){ 		printf( " atom %i   q=  %f \n", i, kQQs[i] );	}
	for ( int ia=0; ia<nx; ia++ ){ 
		printf( " ia %i \n", ia );
		rProbe.add( FF::dCell.a );  
		for ( int ib=0; ib<ny; ib++ ){ 
			rProbe.add( FF::dCell.b );
			for ( int ic=0; ic<nz; ic++ ){
				rProbe.add( FF::dCell.c );
				FF::grid[ i3D( ia, ib, ic ) ].add( getAtomsForceCoulomb( rProbe, natom, Rs, kQQs ) );
				//printf(  " %i %i %i     %f %f %f  \n", ia, ib, ic,     rProbe.x, rProbe.y, rProbe.z  );
				//FF[ i3D( ix, iy, iz ) ].set( rProbe );
			} 
			rProbe.add_mul( FF::dCell.c, -nz );
		} 
		rProbe.add_mul( FF::dCell.b, -ny );
	}
}

// relax one stroke of tip positions ( stored in 1D array "rTips_" ) using precomputed 3D force-field on grid
// returns position of probe-particle after relaxation in 1D array "rs_" and force between surface probe particle in this relaxed position in 1D array "fs_"
// for efficiency, starting position of ProbeParticle in new point (next postion of Tip) is derived from relaxed postion of ProbeParticle from previous point
// there are several strategies how to do it which are choosen by parameter probeStart 
int relaxTipStroke ( int probeStart, int relaxAlg, int nstep, double * rTips_, double * rs_, double * fs_ ){
	Vec3d * rTips = (Vec3d*) rTips_;
	Vec3d * rs    = (Vec3d*) rs_;
	Vec3d * fs    = (Vec3d*) fs_;
	int itrmin=RELAX::maxIters+1,itrmax=0,itrsum=0;
	Vec3d rTip,rProbe;
	rTip  .set    ( rTips[0]      );
	rProbe.set_add( rTip, TIP::rPP0 );
	//printf( " rTip0: %f %f %f  rProbe0: %f %f %f \n", rTip.x, rTip.y, rTip.z, rProbe.x, rProbe.y, rProbe.z  );
	for( int i=0; i<nstep; i++ ){ // for each postion of tip
		// set starting postion of ProbeParticle
		if       ( probeStart == -1 ) {	 // rProbe stay from previous step
			rTip  .set    ( rTips[i]     );
		}else if ( probeStart == 0 ){   // rProbe reset to tip equilibrium
			rTip  .set    ( rTips[i]      );
			rProbe.set_add( rTip, TIP::rPP0 );
		}else if ( probeStart == 1 ){   // rProbe shifted by the same vector as tip
			Vec3d drp; 
			drp   .set_sub( rProbe, rTip );
			rTip  .set    ( rTips[i]     );
			rProbe.set_add( rTip, drp    );
		} 
		// relax Probe Particle postion
		int itr = relaxProbe( relaxAlg, rTip, rProbe );
		if( itr>RELAX::maxIters ){
			printf( " not converged in %i iterations \n", RELAX::maxIters );
			printf( "exiting \n" );	break;
		}
		//printf( " %i  %i    %f %f %f   %f %f %f \n", i, itr, rTip.x, rTip.y, rTip.z, rProbe.x, rProbe.y, rProbe.z  );
		// compute force in relaxed position
		Vec3d rGrid; 
		rGrid.set( rProbe.dot( FF::diCell.a ), rProbe.dot( FF::diCell.b ), rProbe.dot( FF::diCell.c ) ); 
		rs[i].set( rProbe                               );
		fs[i].set( interpolate3DvecWrap( FF::grid, FF::n, rGrid ) );
		// count some statistics about number of iterations required; just for testing
		itrsum += itr;
		//itrmin  = ( itr < itrmin ) ? itr : itrmin;
		//itrmax  = ( itr > itrmax ) ? itr : itrmax;
	}
	//printf( " itr min, max, average %i %i %f \n", itrmin, itrmax, itrsum/(double)nstep );
	return itrsum;
}

void initProbeParticle(void){

}

}









