
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "Vec3.h"
#include "Mat3.h"
#include "spline_hermite.h"
//#include <string.h>

#include "Grid.h"

// ================= MACROS

// ================= CONSTANTS

const double kcoulomb   = 14.3996448915; 

// ================= GLOBAL VARIABLES

GridShape gridShape;

Vec3d   * gridF = NULL;       // pointer to data    ( 3D vector array [nx,ny,nz,3] )
double  * gridE = NULL;       // pointer to data    ( 3D scalar array [nx,ny,nz]   )

int      natoms;
int      nCoefPerAtom;
Vec3d  * Ratoms; 
double * C6s;
double * C12s;
double * kQQs;

// Tip namespace 
namespace TIP{
	Vec3d   rPP0;       // equilibirum bending position
	Vec3d   kSpring;    // bending stiffness ( z component usually zero )
	double  lRadial;    // radial PP-tip distance
	double  kRadial;    // radial PP-tip stiffness

	// tip forcefiled spline
	int      rff_n    = 0;
	double * rff_xs   = NULL; 
	double * rff_ydys = NULL;

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

// radial spring constrain 
Vec3d forceRSpline( const Vec3d& dR, int n, double * xs, double * ydys ){
	double x     =  sqrt( dR.norm2() );
	int    i     = Spline_Hermite::find_index<double>( i, n-i, x, xs );
	double x0    = xs[i  ];
	double x1    = xs[i+1];
	double dx    = x1-x0;
	double denom = 1/dx;
	double u     = (x-x0)*denom;
	i=i<<1;
	double fr =  Spline_Hermite::val<double>( u, ydys[i], ydys[i+2], ydys[i+1]*dx, ydys[i+3]*dx );
	Vec3d f; f.set_mul( dR, fr/x );
	return f;
}

inline Vec3d forceSpringRotated( const Vec3d& dR, const Vec3d& Fw, const Vec3d& Up, const Vec3d& R0, const Vec3d& K ){
	// dR - vector between actual PPpos and anchor point (in global coords)
	// Fw - forward diraction of anchor coordinate system (previous bond direction; e.g. Tip->C for C->O) (in global coords)
	// Up - Up vector --,,-- ; e.g. x axis (1,0,0), defines rotation of your tip (in global coords)
    // R0 - equlibirum position of PP (in local coords)
    // K  - stiffness (ka,kb,kc) along local coords
	// return force (in global coords)
	Mat3d rot; Vec3d dR_,f_,f;
	rot.fromDirUp( Fw*(1/Fw.norm()), Up );  // build orthonormal rotation matrix
	rot.dot_to  ( dR, dR_   );              // transform dR to rotated coordinate system
	f_ .set_mul ( dR_-R0, K );              // spring force (in rotated system)
    // here you can easily put also other forces - e.g. Torsion etc. 
	rot.dot_to_T( dR_, f );                 // transform force back to world system
	return f;
}

// Lenard-Jones force between two atoms a,b separated by vector dR = Ra - Rb
inline double evalLJ( const Vec3d& dR, double c6, double c12, Vec3d& fout ){
	double ir2  = 1.0d/ dR.norm2( ); 
	double ir6  = ir2*ir2*ir2;
	double E6   = c6  * ir6;
	double E12  = c12 * ir6*ir6;
	//return dR * ( ( 6*ir6*c6 -12*ir12*c12 ) * ir2  );
	fout.set_mul( dR , ( 6*E6 -12*E12 ) * ir2 );
	return E12 - E6;
}

// coulomb force between two atoms a,b separated by vector dR = R1 - R2, with constant kqq should be set to kqq = - k_coulomb * Qa * Qb 
inline double evalCoulomb( const Vec3d& dR, double kqq, Vec3d& fout ){
	//const double kcoulomb   = 14.3996448915; 
	double ir2  = 1.0d/ dR.norm2( ); 
	double E    = sqrt( ir2 ) * kqq;
	//return dR * ir * ir2;
	fout.set_mul( dR , E * ir2 );
	return E;
}

// Lenard-Jones force between Probe-Particle (rProbe) and n other atoms
inline double evalAtomsForceLJ( const Vec3d& rProbe, int n, Vec3d * Rs, double * C6, double * C12, Vec3d& fout ){
	double E=0;
	Vec3d fsum; fsum.set(0.0d);
	for(int i=0; i<n; i++){		
		Vec3d f;
		E += evalLJ( Rs[i] - rProbe, C6[i], C12[i], f );
		fsum.add(f);	
	}
	fout.set( fsum );
	return E;
}

// Coulomb force between Probe-Particle (rProbe) and n other atoms
inline double evalAtomsForceCoulomb( const Vec3d& rProbe, int n, Vec3d * Rs, double * kQQs, Vec3d& fout ){
	double E=0;
	Vec3d fsum; fsum.set(0.0d);
	for(int i=0; i<n; i++){	
		Vec3d f;
		E += evalCoulomb( Rs[i] - rProbe, kQQs[i], f );	
		fsum.add(f);
	}
	fout.set( fsum );
	return E;
}

// ========= eval force templates

inline double force_LJ( const Vec3d& dR, Vec3d& fout, double * coefs  ){
	double ir2  = 1.0d/ dR.norm2( ); 
	double ir6  = ir2*ir2*ir2;
	double E6   = coefs[0] * ir6;
	double E12  = coefs[1] * ir6*ir6;
	fout.set_mul( dR , ( 6*E6 -12*E12 ) * ir2 );
	return E12 - E6;
}

inline double force_Coulomb_s( const Vec3d& dR, Vec3d& fout, double * coefs ){
    //printf(" force_Coulomb_s  dR (%g,%g,%g)  kQQ %i \n", dR.x, dR.y, dR.z, coefs );
    //printf(" force_Coulomb_s  dR (%g,%g,%g)  kQQ %g \n", dR.x, dR.y, dR.z, coefs[0] );
	double ir2  = 1.0d/ dR.norm2( ); 
	double E    = sqrt( ir2 ) * coefs[0];
	fout.set_mul( dR , E * ir2 );
	return E;
}
inline double force_Coulomb_pz( const Vec3d& dR, Vec3d& fout, double * coefs ){
	double ir2  = 1.0d/ dR.norm2( ); 
	double E    = sqrt( ir2 ) * coefs[0];
	fout.set_mul( dR , E * ir2 );
	return E;
}
inline double force_Coulomb_dz2( const Vec3d& dR, Vec3d& fout, double * coefs ){
	double ir2  = 1.0d/ dR.norm2( ); 
	double E    = sqrt( ir2 ) * coefs[0];
	fout.set_mul( dR , E * ir2 );
	return E;
}

/*
// coefs is array of coefficient for each atom; nc is number of coefs for each atom
template<double force_func(const Vec3d& dR, Vec3d& fout, double * coefs)>
inline double evalAtomsForce( const Vec3d& rProbe, Vec3d& fout, int natoms, Vec3d * Rs, int nc, double * coefs ){
	double E=0;
	Vec3d fsum; fsum.set(0.0d);
	for(int i=0; i<n; i++){	
		Vec3d f;
		E += force_func( Rs[i] - rProbe, f, coefs );	
		fsum.add(f);
		coefs+=nc;
	}
	fout.set( fsum );
	return E;
}
*/

// coefs is array of coefficient for each atom; nc is number of coefs for each atom
template<double force_func(const Vec3d& dR, Vec3d& fout, double * coefs)>
inline void evalCell( int ibuff, const Vec3d& rProbe, void * args ){
	double * coefs = (double*)args; 
	//printf(" evalCell : args %i \n", args );
	//printf(" natoms %i nCoefPerAtom %i \n", natoms, nCoefPerAtom );
	double E=0;
	Vec3d f; f.set(0.0d);
	for(int i=0; i<natoms; i++){	
		Vec3d fi;
		E += force_func( Ratoms[i] - rProbe, fi, coefs );	
		f.add(fi);
		coefs+=nCoefPerAtom;
	}
	if(gridF) gridF[ibuff].add(f); 
	if(gridE) gridE[ibuff] += E;
}

// ========== Interpolations

// relax probe particle position "r" given on particular position of tip (rTip) and initial position "r" 
int relaxProbe( int relaxAlg, const Vec3d& rTip, Vec3d& r ){
	Vec3d v; v.set( 0.0d );
	int iter;
	//printf( " alg %i r  %f %f %f  rTip  %f %f %f \n", relaxAlg, r.x,r.y,r.z,  rTip.x, rTip.y, rTip.z );

	for( iter=0; iter<RELAX::maxIters; iter++ ){
		Vec3d rGrid,f,drTip; 
		//rGrid.set_mul(r, gridShape::invStep );                                                     // transform position from cartesian world coordinates to coordinates along which Force-Field data are sampled (     orthogonal cell )
		rGrid.set( r.dot( gridShape.diCell.a ), r.dot( gridShape.diCell.b ), r.dot( gridShape.diCell.c ) );     // transform position from cartesian world coordinates to coordinates along which Force-Field data are sampled ( non-orthogonal cell )
		drTip.set_sub( r, rTip );                                                             // vector between Probe-particle and tip apex
		f.set    ( interpolate3DvecWrap( gridF, gridShape.n, rGrid ) );                          // force from surface, interpolated from Force-Field data array
		if( TIP::rff_xs ){
			f.add( forceRSpline( drTip, TIP::rff_n, TIP::rff_xs, TIP::rff_ydys ) );			  // force from tip - radial component spline	
		}else{		
			f.add( forceRSpring( drTip, TIP::kRadial, TIP::lRadial ) );                       // force from tip - radial component harmonic		
		}		
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
void setFF_Fpointer( double * gridF_ ){
	gridF = (Vec3d *)gridF_;
}

// set pointer to force field array ( the array is usually allocated in python, we can flexibely switch betweeen different precomputed forcefields )
void setFF_Epointer( double * gridE_ ){
	gridE = gridE_;
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

// set parameters of the tip like stiffness and equlibirum position in radial and lateral direction
void setTip( double lRad, double kRad, double * rPP0, double * kSpring ){  
	TIP::lRadial=lRad; 
	TIP::kRadial=kRad;  
	TIP::rPP0.set(rPP0);   
	TIP::kSpring.set(kSpring); 
	TIP::makeConsistent();  // rPP0 to be consistent with  lRadial
}

// set parameters of the tip like stiffness and equlibirum position in radial and lateral direction
void setTipSpline( int n, double * xs, double * ydys ){  
	TIP::rff_n    = n;
	TIP::rff_xs   = xs;  
	TIP::rff_ydys = ydys;   
}

// sample Lenard-Jones Force-field on 3D mesh over provided set of atoms with positions Rs_[i] with given C6 and C12 parameters; 
// results are sampled according to grid parameters defined in "namespace FF" and stored in array to which points by "double * FF::grid"
inline void evalCell_LJ( int ibuff, const Vec3d& rProbe, void * args ){
	Vec3d f; double E;
	E = evalAtomsForceLJ( rProbe, natoms, Ratoms, C6s, C12s,  f );
	if( gridF ) gridF[ ibuff ]   .add( f );
	if( gridE ) gridE[ ibuff ] +=      E  ;
	//gridF[ ibuff ]   .set( rProbe );
	//FF::grid[ ibuff ].add( getAtomsForceLJ( rProbe, natom, Rs, C6, C12 ) );
	//printf(  " %i %i %i     %f %f %f  \n", ia, ib, ic,     rProbe.x, rProbe.y, rProbe.z  );
	//FF[ ibuff ].set( rProbe );
}
void getLenardJonesFF( int natoms_, double * Ratoms_, double * C6s_, double * C12s_ ){
    natoms=natoms_; Ratoms=(Vec3d*)Ratoms_; C6s=C6s_; C12s=C12s_;
    Vec3d r0; r0.set(0.0,0.0,0.0);
    interateGrid3D<evalCell_LJ>( r0, gridShape.n, gridShape.dCell, NULL );
}

// sample Coulomb Force-field on 3D mesh over provided set of atoms with positions Rs_[i] with constant kQQs  =  - k_coulomb * Q_ProbeParticle * Q[i] 
// results are sampled according to grid parameters defined in "namespace FF" and stored in array to which points by "double * FF::grid"
void getCoulombFF( int natoms_, double * Ratoms_, double * kQQs_, int kind ){
    natoms=natoms_; 
    Ratoms=(Vec3d*)Ratoms_;
    nCoefPerAtom = 1;
    //printf(" kind %i natoms %i nCoefPerAtom %i \n", kind, natoms, nCoefPerAtom );
    Vec3d r0; r0.set(0.0,0.0,0.0);
    switch(kind){
        //case 0: interateGrid3D < evalCell < foo  > >( r0, gridShape.n, gridShape.dCell, kQQs_ );
        case 0: interateGrid3D < evalCell < force_Coulomb_s   > >( r0, gridShape.n, gridShape.dCell, kQQs_ );
        case 1: interateGrid3D < evalCell < force_Coulomb_pz  > >( r0, gridShape.n, gridShape.dCell, kQQs_ );
        case 2: interateGrid3D < evalCell < force_Coulomb_dz2 > >( r0, gridShape.n, gridShape.dCell, kQQs_ );
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
		rGrid.set( rProbe.dot( gridShape.diCell.a ), rProbe.dot( gridShape.diCell.b ), rProbe.dot( gridShape.diCell.c ) ); 
		rs[i].set( rProbe                               );
		fs[i].set( interpolate3DvecWrap( gridF, gridShape.n, rGrid ) );
		// count some statistics about number of iterations required; just for testing
		itrsum += itr;
		//itrmin  = ( itr < itrmin ) ? itr : itrmin;
		//itrmax  = ( itr > itrmax ) ? itr : itrmax;
	}
	//printf( " itr min, max, average %i %i %f \n", itrmin, itrmax, itrsum/(double)nstep );
	return itrsum;
}

void subsample_uniform_spline( double x0, double dx, int n, double * ydys, int m, double * xs_, double * ys_ ){
	double denom = 1/dx;
	for( int j=0; j<m; j++ ){
		double x  = xs_[j];
		double u  = (x - x0)*denom;
		int    i  = (int)u;
		       u -= i;  
		int i_=i<<1;
		//printf( " %i %i %f %f (%f,%f) (%f,%f) (%f,%f) \n", j, i, x, u, xs[i], xs[i+1], ydys[i_], ydys[i_+2], ydys[i_+1], ydys[i_+3] );
		ys_[j] = Spline_Hermite::val<double>( u, ydys[i_], ydys[i_+2], ydys[i_+1]*dx, ydys[i_+3]*dx );
	}
}

void subsample_nonuniform_spline( int n, double * xs, double * ydys, int m, double * xs_, double * ys_ ){
	int i=0;
	//double x0=xs[0],x1=xs[1],dx=x1-x0,denom=1/dx;
	double x0,x1=-1e+300,dx,denom;
	for( int j=0; j<m; j++ ){
		double x = xs_[j]; 
		if( x>x1 ){
			i         = Spline_Hermite::find_index<double>( i, n-i, x, xs );
			x0        = xs[i  ];
			x1        = xs[i+1];
			dx = x1-x0;
			denom = 1/dx;
			//printf( " region shift %i %f %f %f %f \n", i, x0, x1, dx, denom );
		}
		double u = (x-x0)*denom;
		int i_=i<<1;
		//printf( " %i %i %f %f (%f,%f) (%f,%f) (%f,%f) \n", j, i, x, u, xs[i], xs[i+1], ydys[i_], ydys[i_+2], ydys[i_+1], ydys[i_+3] );
		ys_[j] = Spline_Hermite::val<double>( u, ydys[i_], ydys[i_+2], ydys[i_+1]*dx, ydys[i_+3]*dx );
	}
}

void test_force( int type, int n, double * r0_, double * dr_, double * R_, double * fs_ ){
	Vec3d r,dr,R;
	r .set( r0_[0], r0_[1], r0_[2] );
	dr.set( dr_[0], dr_[1], dr_[2] );
	R .set( R_ [0], R_ [1], R_ [2] );
	Vec3d * fs = (Vec3d *) fs_;
	for( int i=0; i<n; i++ ){
		//Vec3d drTip.set_sub( r, R );
		Vec3d f;
		switch( type ){
			case 1 : f = forceRSpline( r-R, TIP::rff_n,   TIP::rff_xs, TIP::rff_ydys ); break;
			case 2 : f = forceRSpring( r-R, TIP::kRadial, TIP::lRadial               ); break;  
		}
		fs[i] = f;
		//printf( " %i (%3.3f,%3.3f,%3.3f) (%3.3f,%3.3f,%3.3f) \n", i, r.x, r.y, r.z,  f.x, f.y, f.z );
		r.add(dr);
	}
}

}



