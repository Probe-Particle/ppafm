
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "Vec3.cpp"
#include "Mat3.cpp"
#include "spline_hermite.h"
#include <string.h>
#include <iostream>
#include <fstream>
// ================= MACROS

#define fast_floor( x )    ( ((int)(x+1000))-1000 )
#define i3D( ix, iy, iz )  ( iz*nxy + iy*nx + ix  ) 

// ================= CONSTANTS

// ================= GLOBAL VARIABLES

// Force-Field namespace 
namespace FFC {		    
	Vec3d   * gridF = NULL;     // pointer to data array (3D)
	double  * gridE = NULL;
	Mat3d   dCell;      // basis vector of each voxel ( lattice vectors divided by number of points )
	Mat3d   diCell;     // inversion of voxel basis vector
	Vec3i   n;          // number of pixels along each basis vector
}
namespace FFO {		    
	Vec3d   * gridF = NULL;     // pointer to data array (3D)
	double  * gridE = NULL;
	Mat3d   dCell;      // basis vector of each voxel ( lattice vectors divided by number of points )
	Mat3d   diCell;     // inversion of voxel basis vector
	Vec3i   n;          // number of pixels along each basis vector
}

// Tip namespace 
namespace TIP{
	Vec3d   rC0;         // equilibirum bending position of the Carbon atom
	Vec3d   rO0;         // equilibirum bending position of the Oxygen atom
	Vec3d   upVector;    // 
	Vec3d   posO0local;  // equilibrium bending position of the Oxygen atom in local coordinates
	Vec3d   CkSpring;    // bending stiffness ( z component usually zero )
	Vec3d   OkSpring;    // bending stiffness ( z component usually zero )
	double  TClRadial;   // radial tip-C distance
	double  COlRadial;   // radial C-O distance
	double  TCkRadial;   // radial tip-C stiffness
	double  COkRadial;   // radial C-O stiffness

	// tip forcefiled spline
	int      rff_n    = 0;
	double * rff_xs   = NULL; 
	double * rff_ydys = NULL;

	void makeConsistent(){ // place rPP0 on the sphere to be consistent with radial spring
		if( fabs(TCkRadial) > 1e-8 ){  
			rC0.z = -sqrt( TClRadial*TClRadial - rC0.x*rC0.x - rC0.y*rC0.y );  
			printf(" rC0 %f %f %f \n", rC0.x, rC0.y, rC0.z );
		}
		if( fabs(COkRadial) > 1e-8 ){  
			rO0.z = -sqrt( COlRadial*COlRadial - rO0.x*rO0.x - rO0.y*rO0.y );  
			printf(" rO0 %f %f %f \n", rC0.x, rC0.y, rC0.z );
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

/*
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
*/

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

inline void getCforce( const Vec3d& rTip, const Vec3d& rC, Vec3d& fC ){
	Vec3d rGrid,drTC; 
	//rGrid.set_mul(r, FF::invStep );                                                     // transform position from cartesian world coordinates to coordinates along which Force-Field data are sampled (     orthogonal cell )
	rGrid.set( rC.dot( FFC::diCell.a ), rC.dot( FFC::diCell.b ), rC.dot( FFC::diCell.c ) );     // transform position from cartesian world coordinates to coordinates along which Force-Field data are sampled ( non-orthogonal cell )
	drTC.set_sub( rC, rTip );                                                             // vector between tip apex and Carbon atom
	fC.set    ( interpolate3DvecWrap( FFC::gridF, FFC::n, rGrid ) );                          // force from surface, interpolated from Force-Field data array
	if( TIP::rff_xs ){
		fC.add( forceRSpline( drTC, TIP::rff_n, TIP::rff_xs, TIP::rff_ydys ) );			  // force from tip - radial component spline	
	}else{		
		fC.add( forceRSpring( drTC, TIP::TCkRadial, TIP::TClRadial ) );                       // force from tip - radial component harmonic		
	}		
	drTC.sub( TIP::rC0 );
	fC.add_mul( drTC, TIP::CkSpring );      // spring force                                // force from tip - lateral bending force
}

void force_bend ( const Vec3d& r12, const Vec3d &  r23, const double k, Vec3d& f1, Vec3d& f2, Vec3d& f3)
// spring force calculations
{
//    testing only
//    r12.set(0,0.1,-1.85);
//    r23.set(0,0.3,-1.15);
//	std::cout <<"r12 : "<<r12.x<<" "<<r12.y<<" "<<r12.z<<std::endl;
//	std::cout <<"r23 : "<<r23.x<<" "<<r23.y<<" "<<r23.z<<std::endl;
    
    double len12 = r12.norm();
    double len23 = r23.norm();
    f1.set(r23);
    f1.sub_mul(r12,r12.dot(r23)/len12/len12);
    f3.set_mul(r23,r12.dot(r23)/len23/len23);
    f3.sub(r12);
    f1.mul(k/len12/len23);
    f3.mul(k/len12/len23);
    f2.set_mul(f1,-1.0);
    f2.sub(f3);
//	std::cout << "Lat T force: "<<f1.x<<" "<<" "<<f1.y<<" "<<f1.z<<std::endl;
//	std::cout << "Lat C force: "<<f2.x<<" "<<" "<<f2.y<<" "<<f2.z<<std::endl;
//	std::cout << "Lat O force: "<<f3.x<<" "<<" "<<f3.y<<" "<<f3.z<<std::endl;
//    exit(0);

}

void forceSpringRotated( const Vec3d& Fw, const Vec3d& dR, const  Vec3d& Up,  const Vec3d& R0, const double K, Vec3d& f1, Vec3d& f2, Vec3d& f3 ){
	// dR - vector between actual PPpos and anchor point (in global coords)
	// Fw - forward diraction of anchor coordinate system (previous bond direction; e.g. Tip->C for C->O) (in global coords)
	// Up - Up vector --,,-- ; e.g. x axis (1,0,0), defines rotation of your tip (in global coords)
    // R0 - equlibirum position of PP (in local coords)
    // K  - stiffness (ka,kb,kc) along local coords
	// return force (in global coords)
	Mat3d rot; Vec3d dR_, R0_,f1_,f2_, f3_;
	rot.fromDirUp( Fw, Up );  // build orthonormal rotation matrix

	rot.dot_to  ( dR, dR_   );              // transform dR to rotated coordinate system
    R0_.set(R0);
    R0_.mul(Fw.norm()/R0.norm());
//	std::cout << ":"<<" dR "<<dR.x<<" "<<dR.y<<" "<<dR.z<<" rRnew: "<<dR_.x<<" "<<" "<<dR_.y<<" "<<dR_.z<<std::endl;
//	std::cout << ":"<<" R0 "<<R0.x<<" "<<R0.y<<" "<<R0.z<<std::endl;
	//f_ .set_mul ( dR_-R0, K );              // spring force (in rotated system)
    force_bend (R0_, dR_,K, f1_, f2_, f3_);
    // here you can easily put also other forces - e.g. Torsion etc. 
	rot.dot_to_T( f1_, f1 );                 // transform force back to world system
	rot.dot_to_T( f2_, f2 );                 // transform force back to world system
	rot.dot_to_T( f3_, f3 );                 // transform force back to world system
}




inline void getCOforce( const Vec3d& rTip, Vec3d& rC, Vec3d& fC, Vec3d& rO, Vec3d& fO ){
	Vec3d rCGrid, rOGrid, drTC,drTCnorm,drCO,posO0,drFake; 
    Vec3d latTforce, latCforce, latOforce, latFakeForce;
    
    drFake.set(0,0,1);
    //!!!!!!!! remove after testing
    /*
    rC.set(4.0, 2.1,15.15);
    rO.set(4.0, 2.5,14.0);
    std::cout<<"C K sprint" << TIP::CkSpring.x << std::endl;
    std::cout<<"O K sprint" << TIP::OkSpring.x << std::endl;
    TIP::OkSpring.x=-1.0;
    TIP::CkSpring.x=-1.0;
    std::cout<<"T C rad sprint" << TIP::TCkRadial << std::endl;
    std::cout<<"C O rad sprint" << TIP::COkRadial << std::endl;
    */
    //!!!!!!!!
//    std::cout << "Getting force"<<std::endl;
	//rGrid.set_mul(r, FF::invStep );                                                     // transform position from cartesian world coordinates to coordinates along which Force-Field data are sampled (     orthogonal cell )
	rCGrid.set( rC.dot( FFC::diCell.a ), rC.dot( FFC::diCell.b ), rC.dot( FFC::diCell.c ) );     // transform position from cartesian world coordinates to coordinates along which Force-Field data are sampled ( non-orthogonal cell )
	rOGrid.set( rO.dot( FFC::diCell.a ), rO.dot( FFC::diCell.b ), rO.dot( FFC::diCell.c ) );     // transform position from cartesian world coordinates to coordinates along which Force-Field data are sampled ( non-orthogonal cell )
	drTC.set_sub( rC, rTip );                                                             // vector between tip apex and Carbon atom
	drCO.set_sub( rO, rC );                                                             // vector between Carbon and Oxygen
	fC.set    ( interpolate3DvecWrap( FFC::gridF, FFC::n, rCGrid ) );                          // force from surface, interpolated from Force-Field data array
	fO.set    ( interpolate3DvecWrap( FFO::gridF, FFO::n, rOGrid ) );                          // force from surface, interpolated from Force-Field data array
//	std::cout << "drTC: "<<drTC.x<<" "<<drTC.y<<" "<<drTC.z<<std::endl;
//	std::cout << "drCO: "<<drCO.x<<" "<<drCO.y<<" "<<drCO.z<<std::endl;
	if( TIP::rff_xs ){
		fC.add( forceRSpline( drTC, TIP::rff_n, TIP::rff_xs, TIP::rff_ydys ) );			  // force from tip - radial component spline	
		fO.add( forceRSpline( drCO, TIP::rff_n, TIP::rff_xs, TIP::rff_ydys ) );			  // force from tip - radial component spline	
        std::cerr<<"Spline was not tested with the two-points PP model, Exit!\n"<<std::endl;
        exit (1) ;
		fC.add( forceRSpline( drTC, TIP::rff_n, TIP::rff_xs, TIP::rff_ydys ) );			  // force from tip - radial component spline	
	}else{		
		fC.add( forceRSpring( drTC, TIP::TCkRadial, TIP::TClRadial ) );                       // force from tip - radial component harmonic		
		fO.add( forceRSpring( drCO, TIP::COkRadial, TIP::COlRadial ) );                       // force from tip - radial component harmonic		
		fC.sub( forceRSpring( drCO, TIP::COkRadial, TIP::COlRadial ) );                       // force from tip - radial component harmonic		
	}
    
//	std::cout << "Grid force + rad force:"<<" rC0: "<<fC.x<<" "<<fC.y<<" "<<fC.z<<" rO0: "<<fO.x<<" "<<" "<<fO.y<<" "<<fO.z<<std::endl;
//    std::cout << std::endl;
//    std::cout << "T position:" <<rTip.x<<" "<<rTip.y<<" "<<rTip.z<<std::endl;
//    std::cout << "C position:" <<rC.x<<" "<<rC.y<<" "<<rC.z<<std::endl;
//    std::cout << "C0 position:" <<TIP::rC0.x<<" "<<TIP::rC0.y<<" "<<TIP::rC0.z<<std::endl;

//    std::cout << "O position:" <<rO.x<<" "<<rO.y<<" "<<rO.z<<std::endl;

    
	//drTC.sub( TIP::rC0 );
    forceSpringRotated(drFake, drTC, TIP::upVector , TIP::rC0, TIP::CkSpring.x, latFakeForce, latTforce, latCforce );
	fC.add(latCforce );        // force from tip - lateral bending force
//	std::cout << "Lat C force: "<<latCforce.x<<" "<<" "<<latCforce.y<<" "<<latCforce.z<<std::endl;
    //exit(1);
    forceSpringRotated(drTC, drCO , TIP::upVector , TIP::posO0local, TIP::OkSpring.x, latTforce, latCforce, latOforce );
//	std::cout << "Lat T force: "<<latTforce.x<<" "<<" "<<latTforce.y<<" "<<latTforce.z<<std::endl;
//	std::cout << "Lat C force: "<<latCforce.x<<" "<<" "<<latCforce.y<<" "<<latCforce.z<<std::endl;
//	std::cout << "Lat O force: "<<latOforce.x<<" "<<" "<<latOforce.y<<" "<<latOforce.z<<std::endl;
    //exit(1);
	fC.add(latCforce);
	fO.add(latOforce);
//	std::cout << "Grid force + rad force:+lat force"<<" rC0: "<<fC.x<<" "<<fC.y<<" "<<fC.z<<" rO0: "<<fO.x<<" "<<" "<<fO.y<<" "<<fO.z<<std::endl;
   //exit(1);
	// dR - vector between actual PPpos and anchor point (in global coords)
	// Fw - forward diraction of anchor coordinate system (previous bond direction; e.g. Tip->C for C->O) (in global coords)
	// Up - Up vector --,,-- ; e.g. x axis (1,0,0), defines rotation of your tip (in global coords)
    // R0 - equlibirum position of PP (in local coords)
    // K  - stiffness (ka,kb,kc) along local coords
	// return force (in global coords)
/*
    drTCnorm=drTC;
    drTCnorm.normalize();
    posO0.set_sub(drTCnorm*drCO.norm(),TIP::rO0);         
	drTC.sub( TIP::rC0 );
    std::cout << "drTCx:" << drTC.x<< "drTCy:" << drTC.y<< "drTCz:" << drTC.z<<std::endl;
	fC.add_mul( drTC, TIP::CkSpring );      // spring force                                // force from tip - lateral bending force
	drCO.sub( posO0 );
	fO.add_mul( drCO, TIP::OkSpring );      // spring force                                // force from tip - lateral bending force
	fC.sub_mul( drCO, TIP::OkSpring*(drCO.norm()/drTC.norm()+1.) );         // The lateral contibution to the force acting from the Oxygen to Carbon
    std::cout << "drCO.norm()" << drCO.norm() << std::endl;
    std::cout << "drTC.norm()" << drTC.norm() << std::endl;
	std::cout << "Returning Lateral bend force:"<<" rC0: "<<fC.x<<" "<<fC.y<<" "<<fC.z<<" rO0: "<<fO.x<<" "<<" "<<fO.y<<" "<<fO.z<<std::endl;
*/
}

// relax probe particle position "r" given on particular position of tip (rTip) and initial position "r" 
int relaxProbe( int relaxAlg, const Vec3d& rTip, Vec3d& rC, Vec3d& rO ){
	Vec3d vC, vO; vC.set( 0.0d ), vO.set(0.0d);
	int iter;
	//printf( " alg %i r  %f %f %f  rTip  %f %f %f \n", relaxAlg, r.x,r.y,r.z,  rTip.x, rTip.y, rTip.z );
	for( iter=0; iter<RELAX::maxIters; iter++ ){
		Vec3d fC, fO;  
//        getCforce( rTip, rC, fC, rO);
        getCOforce( rTip, rC, fC, rO, fO );
		if( relaxAlg == 1 ){                                                                  // move by either damped-leap-frog ( 0 ) or by FIRE ( 1 )
			FIRE::move( fC, rC, vC );
			FIRE::move( fO, rO, vO );
		}else{
			RELAX::move( fC, rC, vC );
			RELAX::move( fO, rO, vO );
		}			
	    //std::cout << "force:"<<" rC0: "<<fC.x<<" "<<fC.y<<" "<<fC.z<<" rO0: "<<fO.x<<" "<<" "<<fO.y<<" "<<fO.z<<std::endl;
	    //std::cout << "MOVED:"<<" rTip0: "<<rTip.x<<" "<<rTip.y<<" "<<rTip.z<<" rC0: "<<rC.x<<" "<<rC.y<<" "<<rC.z<<" rO0: "<<rO.x<<" "<<" "<<rO.y<<" "<<rO.z<<std::endl;
		//printf( "     %i r  %f %f %f  f  %f %f %f \n", iter, r.x,r.y,r.z,  f.x,f.y,f.z );
		if( (fC.norm2() < RELAX::convF2 ) && (fO.norm2() < RELAX::convF2 )) break;                                                // check force convergence
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
void setFFC_Fpointer( double * gridF  ){
	FFC::gridF = (Vec3d *)gridF;
}
void setFFO_Fpointer( double * gridF  ){
	FFO::gridF = (Vec3d *)gridF;
}

// set pointer to force field array ( the array is usually allocated in python, we can flexibely switch betweeen different precomputed forcefields )
void setFFC_Epointer( double * gridE ){
	FFC::gridE = gridE;
}
void setFFO_Epointer( double * gridE ){
	FFO::gridE = gridE;
}

// set parameters of forcefield like dimension "n", and lattice vectors "cell"
void setFFC_shape( int * n, double * cell ){
	//FF::grid = (Vec3d *)grid;
	FFC::n.set(n);
	FFC::dCell.a.set( cell[0], cell[1], cell[2] );   FFC::dCell.a.mul( 1.0d/FFC::n.a );
	FFC::dCell.b.set( cell[3], cell[4], cell[5] );   FFC::dCell.b.mul( 1.0d/FFC::n.b );
	FFC::dCell.c.set( cell[6], cell[7], cell[8] );   FFC::dCell.c.mul( 1.0d/FFC::n.c );
	FFC::dCell.invert_T_to( FFC::diCell );	

	printf( "C nxyz  %i %i %i \n", FFC::n.x,       FFC::n.y,       FFC::n.z );
	printf( "C a     %f %f %f \n", FFC::dCell.a.x, FFC::dCell.a.y, FFC::dCell.a.z );
	printf( "C b     %f %f %f \n", FFC::dCell.b.x, FFC::dCell.b.y, FFC::dCell.b.z );
	printf( "C c     %f %f %f \n", FFC::dCell.c.x, FFC::dCell.c.y, FFC::dCell.c.z );
	printf( "C inv_a %f %f %f \n", FFC::diCell.a.x,FFC::diCell.a.y,FFC::diCell.a.z );
	printf( "C inv_b %f %f %f \n", FFC::diCell.b.x,FFC::diCell.b.y,FFC::diCell.b.z );
	printf( "C inv_c %f %f %f \n", FFC::diCell.c.x,FFC::diCell.c.y,FFC::diCell.c.z );
}
void setFFO_shape( int * n, double * cell ){
	//FF::grid = (Vec3d *)grid;
	FFO::n.set(n);
	FFO::dCell.a.set( cell[0], cell[1], cell[2] );   FFO::dCell.a.mul( 1.0d/FFC::n.a );
	FFO::dCell.b.set( cell[3], cell[4], cell[5] );   FFO::dCell.b.mul( 1.0d/FFC::n.b );
	FFO::dCell.c.set( cell[6], cell[7], cell[8] );   FFO::dCell.c.mul( 1.0d/FFC::n.c );
	FFO::dCell.invert_T_to( FFO::diCell );	

	printf( "O nxyz  %i %i %i \n", FFO::n.x,       FFO::n.y,       FFO::n.z );
	printf( "O a     %f %f %f \n", FFO::dCell.a.x, FFO::dCell.a.y, FFO::dCell.a.z );
	printf( "O b     %f %f %f \n", FFO::dCell.b.x, FFO::dCell.b.y, FFO::dCell.b.z );
	printf( "O c     %f %f %f \n", FFO::dCell.c.x, FFO::dCell.c.y, FFO::dCell.c.z );
	printf( "O inv_a %f %f %f \n", FFO::diCell.a.x,FFO::diCell.a.y,FFO::diCell.a.z );
	printf( "O inv_b %f %f %f \n", FFO::diCell.b.x,FFO::diCell.b.y,FFO::diCell.b.z );
	printf( "O inv_c %f %f %f \n", FFO::diCell.c.x,FFO::diCell.c.y,FFO::diCell.c.z );
}


/*
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
*/

// set parameters of the tip like stiffness and equlibirum position in radial and lateral direction
void setTip( double TClRad, double COlRad, double TCkRad, double COkRad, double * rC0, double *rO0, double * CkSpring, double * OkSpring ){  
	TIP::upVector.set(1.0,0.0,0.0);
	TIP::TClRadial=TClRad; 
	TIP::COlRadial=COlRad; 
	TIP::TCkRadial=TCkRad;  
	TIP::COkRadial=COkRad;  
	TIP::rC0.set(rC0);   
	TIP::rO0.set(rO0);   
	TIP::CkSpring.set(CkSpring); 
	TIP::OkSpring.set(OkSpring); 
	TIP::makeConsistent();  // rC0 and rO0 to be consistent with  lRadial
	Mat3d rot;
	rot.fromDirUp(TIP::rC0, TIP::upVector);
	rot.dot_to (TIP::rO0, TIP::posO0local);
}

// set parameters of the tip like stiffness and equlibirum position in radial and lateral direction
void setTipSpline( int n, double * xs, double * ydys ){  
	TIP::rff_n    = n;
	TIP::rff_xs   = xs;  
	TIP::rff_ydys = ydys;   
}

// sample Lenard-Jones Force-field on 3D mesh over provided set of atoms with positions Rs_[i] with given C6 and C12 parameters; 
// results are sampled according to grid parameters defined in "namespace FF" and stored in array to which points by "double * FF::grid"
void getCLenardJonesFF( int natom, double * Rs_, double * C6, double * C12 ){
	Vec3d * Rs = (Vec3d*) Rs_;
	int nx  = FFC::n.x;
	int ny  = FFC::n.y;
	int nz  = FFC::n.z;
	int nxy = ny * nx;
	Vec3d rProbe;  rProbe.set( 0.0, 0.0, 0.0 ); // we may shift here
	for ( int ia=0; ia<nx; ia++ ){ 
	//	printf( " ia %i \n", ia );
        std::cout << "ia " << ia;
        std::cout.flush();
        std::cout << '\r';

		for ( int ib=0; ib<ny; ib++ ){ 
			for ( int ic=0; ic<nz; ic++ ){
				Vec3d f; double E;
				E = evalAtomsForceLJ( rProbe, natom, Rs, C6, C12,   f );
				if( FFC::gridF ) FFC::gridF[ i3D( ia, ib, ic ) ]   .add( f );
				if( FFC::gridE ) FFC::gridE[ i3D( ia, ib, ic ) ] +=      E  ;
				//FF::grid[ i3D( ia, ib, ic ) ].add( getAtomsForceLJ( rProbe, natom, Rs, C6, C12 ) );
				//printf(  " %i %i %i     %f %f %f  \n", ia, ib, ic,     rProbe.x, rProbe.y, rProbe.z  );
				//FF[ i3D( ix, iy, iz ) ].set( rProbe );
				rProbe.add( FFC::dCell.c );
			} 
			rProbe.add_mul( FFC::dCell.c, -nz );
			rProbe.add( FFC::dCell.b );
		} 
		rProbe.add_mul( FFC::dCell.b, -ny );
		rProbe.add( FFC::dCell.a );  
	}
}
void getOLenardJonesFF( int natom, double * Rs_, double * C6, double * C12 ){
	Vec3d * Rs = (Vec3d*) Rs_;
	int nx  = FFO::n.x;
	int ny  = FFO::n.y;
	int nz  = FFO::n.z;
	int nxy = ny * nx;
	Vec3d rProbe;  rProbe.set( 0.0, 0.0, 0.0 ); // we may shift here
	for ( int ia=0; ia<nx; ia++ ){ 
	//	printf( " ia %i \n", ia );
        std::cout << "ia " << ia;
        std::cout.flush();
        std::cout << '\r';

		for ( int ib=0; ib<ny; ib++ ){ 
			for ( int ic=0; ic<nz; ic++ ){
				Vec3d f; double E;
				E = evalAtomsForceLJ( rProbe, natom, Rs, C6, C12,   f );
				if( FFO::gridF ) FFO::gridF[ i3D( ia, ib, ic ) ]   .add( f );
				if( FFO::gridE ) FFO::gridE[ i3D( ia, ib, ic ) ] +=      E  ;
				//FF::grid[ i3D( ia, ib, ic ) ].add( getAtomsForceLJ( rProbe, natom, Rs, C6, C12 ) );
				//printf(  " %i %i %i     %f %f %f  \n", ia, ib, ic,     rProbe.x, rProbe.y, rProbe.z  );
				//FF[ i3D( ix, iy, iz ) ].set( rProbe );
				rProbe.add( FFO::dCell.c );
			} 
			rProbe.add_mul( FFO::dCell.c, -nz );
			rProbe.add( FFO::dCell.b );
		} 
		rProbe.add_mul( FFO::dCell.b, -ny );
		rProbe.add( FFO::dCell.a );  
	}
}

// sample Coulomb Force-field on 3D mesh over provided set of atoms with positions Rs_[i] with constant kQQs  =  - k_coulomb * Q_ProbeParticle * Q[i] 
// results are sampled according to grid parameters defined in "namespace FF" and stored in array to which points by "double * FF::grid"
void getCCoulombFF( int natom, double * Rs_, double * kQQs ){
	Vec3d * Rs = (Vec3d*) Rs_;
	int nx  = FFC::n.x;
	int ny  = FFC::n.y;
	int nz  = FFC::n.z;
	int nxy = ny * nx;
	Vec3d rProbe;  rProbe.set( 0.0, 0.0, 0.0 ); // we may shift here
	//for ( int i=0; i<natom; i++ ){ 		printf( " atom %i   q=  %f \n", i, kQQs[i] );	}
	for ( int ia=0; ia<nx; ia++ ){ 
//		printf( " ia %i \n", ia );  
        std::cout << "ia " << ia;
        std::cout.flush();
        std::cout << '\r';
		for ( int ib=0; ib<ny; ib++ ){
			for ( int ic=0; ic<nz; ic++ ){
				Vec3d f; double E;
				E = evalAtomsForceCoulomb( rProbe, natom, Rs, kQQs, f );
				if( FFC::gridF ) FFC::gridF[ i3D( ia, ib, ic ) ]   .add( f );
				if( FFC::gridE ) FFC::gridE[ i3D( ia, ib, ic ) ] +=      E  ;
				//FF::grid[ i3D( ia, ib, ic ) ].add( getAtomsForceCoulomb( rProbe, natom, Rs, kQQs ) );
				//printf(  " %i %i %i     %f %f %f  \n", ia, ib, ic,     rProbe.x, rProbe.y, rProbe.z  );
				//FF[ i3D( ix, iy, iz ) ].set( rProbe );
				rProbe.add( FFC::dCell.c );
			} 
			rProbe.add_mul( FFC::dCell.c, -nz ); 
			rProbe.add( FFC::dCell.b );
		} 
		rProbe.add_mul( FFC::dCell.b, -ny );
		rProbe.add( FFC::dCell.a );
	}
}
void getOCoulombFF( int natom, double * Rs_, double * kQQs ){
	Vec3d * Rs = (Vec3d*) Rs_;
	int nx  = FFO::n.x;
	int ny  = FFO::n.y;
	int nz  = FFO::n.z;
	int nxy = ny * nx;
	Vec3d rProbe;  rProbe.set( 0.0, 0.0, 0.0 ); // we may shift here
	//for ( int i=0; i<natom; i++ ){ 		printf( " atom %i   q=  %f \n", i, kQQs[i] );	}
	for ( int ia=0; ia<nx; ia++ ){ 
	//	printf( " ia %i \n", ia );  
        std::cout << "ia " << ia;
        std::cout.flush();
        std::cout << '\r';
		for ( int ib=0; ib<ny; ib++ ){
			for ( int ic=0; ic<nz; ic++ ){
				Vec3d f; double E;
				E = evalAtomsForceCoulomb( rProbe, natom, Rs, kQQs, f );
				if( FFO::gridF ) FFO::gridF[ i3D( ia, ib, ic ) ]   .add( f );
				if( FFO::gridE ) FFO::gridE[ i3D( ia, ib, ic ) ] +=      E  ;
				//FF::grid[ i3D( ia, ib, ic ) ].add( getAtomsForceCoulomb( rProbe, natom, Rs, kQQs ) );
				//printf(  " %i %i %i     %f %f %f  \n", ia, ib, ic,     rProbe.x, rProbe.y, rProbe.z  );
				//FF[ i3D( ix, iy, iz ) ].set( rProbe );
				rProbe.add( FFO::dCell.c );
			} 
			rProbe.add_mul( FFO::dCell.c, -nz ); 
			rProbe.add( FFO::dCell.b );
		} 
		rProbe.add_mul( FFO::dCell.b, -ny );
		rProbe.add( FFO::dCell.a );
	}
}

// relax one stroke of tip positions ( stored in 1D array "rTips_" ) using precomputed 3D force-field on grid
// returns position of probe-particle after relaxation in 1D array "rs_" and force between surface probe particle in this relaxed position in 1D array "fs_"
// for efficiency, starting position of ProbeParticle in new point (next postion of Tip) is derived from relaxed postion of ProbeParticle from previous point
// there are several strategies how to do it which are choosen by parameter probeStart 
int relaxTipStroke ( int probeStart, int relaxAlg, int nstep, double * rTips_, double * rC_,  double * rO_, double * fC_ , double *fO_){
	Vec3d * rTips = (Vec3d*) rTips_;
	Vec3d * rCs   = (Vec3d*) rC_;
	Vec3d * fCs   = (Vec3d*) fC_;
	Vec3d * rOs   = (Vec3d*) rO_;
	Vec3d * fOs   = (Vec3d*) fO_;
	int itrmin=RELAX::maxIters+1,itrmax=0,itrsum=0;
	Vec3d rTip,rC,rO;
	rTip  .set    ( rTips[0]      );
	rC.set_add( rTip, TIP::rC0 );
	rO.set_add( rC, TIP::rO0 );
//	printf( " rTip0: %f %f %f  rC0: %f %f %f  rO0: %f %f %f\n", rTip.x, rTip.y, rTip.z, rC.x, rC.y, rC.z, rO.x, rO.y, rO.z  );
	for( int i=0; i<nstep; i++ ){ // for each postion of tip
		// set starting postion of ProbeParticle
		if       ( probeStart == -1 ) {	 // rProbe stay from previous step
//            printf ("Set Probe position -1\n");
			rTip  .set    ( rTips[i]     );
		}else if ( probeStart == 0 ){   // rProbe reset to tip equilibrium
//            printf ("Set Probe position 0\n");
			rTip  .set    ( rTips[i]      );
			rC.set_add( rTip, TIP::rC0 );
			rO.set_add( rC, TIP::rO0 );
		}else if ( probeStart == 1 ){   // rProbe shifted by the same vector as tip
//            printf ("Set Probe position 1\n");
			Vec3d drC,drO; 
			drC   .set_sub( rC, rTip );
			drO   .set_sub( rO, rC );
			rTip  .set    ( rTips[i]     );
			rC.set_add( rTip, drC    );
			rO.set_add( rC, drO    );
		} 
		// relax Probe Particle postion
//        std::cout << "Relax CO" << std::endl;
//	    std::cout << " rTip0: "<<rTip.x<<" "<<rTip.y<<" "<<rTip.z<<" rC0: "<<rC.x<<" "<<rC.y<<" "<<rC.z<<" rO0: "<<rO.x<<" "<<" "<<rO.y<<" "<<rO.z<<std::endl;

		int itr = relaxProbe( relaxAlg, rTip, rC, rO );

        std::ofstream myfile;
        myfile.open("tipCO.xyz",std::ios::app);
        myfile <<3<<std::endl<<std::endl;
        myfile <<"Cu "<<rTip.x<<" "<<rTip.y<<" "<<rTip.z<<std::endl;
        myfile <<"C "<<rC.x<<" "<<rC.y<<" "<<rC.z<<std::endl;
        myfile <<"O "<<rO.x<<" "<<" "<<rO.y<<" "<<rO.z<<std::endl;
        myfile.close();
        //exit(1);

//        std::cout<< "CO relaxed"<<std::endl;
//	    std::cout << " rTip0: "<<rTip.x<<" "<<rTip.y<<" "<<rTip.z<<" rC0: "<<rC.x<<" "<<rC.y<<" "<<rC.z<<" rO0: "<<rO.x<<" "<<" "<<rO.y<<" "<<rO.z<<std::endl;

//        std::cout << "Niter: "<< itr << std::endl;
		if( itr>RELAX::maxIters ){
			printf( " not converged in %i iterations \n", RELAX::maxIters );
			printf( "exiting \n" );	break;
		}
		//printf( " %i  %i    %f %f %f   %f %f %f \n", i, itr, rTip.x, rTip.y, rTip.z, rProbe.x, rProbe.y, rProbe.z  );
		// compute force in relaxed position
//		std::cout<<"Compute force in relaxed position"<< std::endl;
		Vec3d rCGrid,rOGrid; 
		rCGrid.set( rC.dot( FFC::diCell.a ), rC.dot( FFC::diCell.b ), rC.dot( FFC::diCell.c ) ); 
//		std::cout<<"rCGrid.set"<< std::endl;
		rOGrid.set( rO.dot( FFO::diCell.a ), rO.dot( FFO::diCell.b ), rO.dot( FFO::diCell.c ) ); 
//		std::cout<<"rOGrid.set"<< std::endl;
		rCs[i].set( rC                               );
//		std::cout<<"rCs"<< std::endl;
		rOs[i].set( rO                               );
//		std::cout<<"rOs"<< std::endl;
		fCs[i].set( interpolate3DvecWrap( FFC::gridF, FFC::n, rCGrid ) );
//		std::cout<<"fCs"<< std::endl;
		fOs[i].set( interpolate3DvecWrap( FFO::gridF, FFO::n, rOGrid ) );
//		std::cout<<"fOs"<< std::endl;
		// count some statistics about number of iterations required; just for testing
		itrsum += itr;
		//itrmin  = ( itr < itrmin ) ? itr : itrmin;
		//itrmax  = ( itr > itrmax ) ? itr : itrmax;
	}
	//printf( " itr min, max, average %i %i %f \n", itrmin, itrmax, itrsum/(double)nstep );
	return itrsum;
}
/*
void stiffnessMatrix( double ddisp, int which, int n, double * rTips_, double * rPPs_, double * eigenvals_, double * evec1_, double * evec2_, double * evec3_ ){
	Vec3d * rTips     = (Vec3d*) rTips_;
	Vec3d * rPPs      = (Vec3d*) rPPs_;
	Vec3d * eigenvals = (Vec3d*) eigenvals_;
	Vec3d * evec1     = (Vec3d*) evec1_;
	Vec3d * evec2     = (Vec3d*) evec2_;
	Vec3d * evec3     = (Vec3d*) evec3_;
	for(int i=0; i<n; i++){
		Vec3d rTip,rPP,f1,f2;    
		rTip.set( rTips[i] );    
		rPP.set ( rPPs[i]  );
		Mat3d dynmat;     
		//getPPforce( rTip, rPP, f1 );  eigenvals[i] = f1;   // check if we are converged in f=0
		//rPP.x-=ddisp; getPPforce( rTip, rPP, f1 ); rPP.x+=2*ddisp; getPPforce( rTip, rPP, f2 );  rPP.x-=ddisp; evec1[i].set_sub(f2,f1);
		//rPP.y-=ddisp; getPPforce( rTip, rPP, f1 ); rPP.y+=2*ddisp; getPPforce( rTip, rPP, f2 );  rPP.y-=ddisp; evec2[i].set_sub(f2,f1);
		//rPP.z-=ddisp; getPPforce( rTip, rPP, f1 ); rPP.z+=2*ddisp; getPPforce( rTip, rPP, f2 );  rPP.z-=ddisp; evec3[i].set_sub(f2,f1);
		// eval dynamical matrix    D_xy = df_y/dx    = ( f(r0+dx).y - f(r0-dx).y ) / (2*dx)                
		rPP.x-=ddisp; getPPforce( rTip, rPP, f1 ); rPP.x+=2*ddisp; getPPforce( rTip, rPP, f2 );  rPP.x-=ddisp;  dynmat.a.set_sub(f2,f1); dynmat.a.mul(-0.5/ddisp);
		rPP.y-=ddisp; getPPforce( rTip, rPP, f1 ); rPP.y+=2*ddisp; getPPforce( rTip, rPP, f2 );  rPP.y-=ddisp;  dynmat.b.set_sub(f2,f1); dynmat.b.mul(-0.5/ddisp);
		rPP.z-=ddisp; getPPforce( rTip, rPP, f1 ); rPP.z+=2*ddisp; getPPforce( rTip, rPP, f2 );  rPP.z-=ddisp;  dynmat.c.set_sub(f2,f1); dynmat.c.mul(-0.5/ddisp);
		// symmetrize - to make sure that our symmetric matrix solver work properly
		double tmp;
		tmp = 0.5*(dynmat.xy + dynmat.yx); dynmat.xy = tmp; dynmat.yx = tmp;
		tmp = 0.5*(dynmat.yz + dynmat.zy); dynmat.yz = tmp; dynmat.zy = tmp;
		tmp = 0.5*(dynmat.zx + dynmat.xz); dynmat.zx = tmp; dynmat.xz = tmp;
		// solve mat
		Vec3d evals; dynmat.eigenvals( evals );
		// sort eigenvalues
		if( evals.a > evals.b ){ tmp=evals.a; evals.a=evals.b; evals.b=tmp; } 
		if( evals.b > evals.c ){ tmp=evals.b; evals.b=evals.c; evals.c=tmp; }
		if( evals.a > evals.b ){ tmp=evals.a; evals.a=evals.b; evals.b=tmp; } 
		// output eigenvalues and eigenvectors
		eigenvals[i] = evals;
		//if(which>0) dynmat.eigenvec( evals.a, evec1[i] );
		//if(which>1) dynmat.eigenvec( evals.b, evec2[i] );
		//if(which>2) dynmat.eigenvec( evals.c, evec3[i] );
		evec1[i] = dynmat.a;
		evec2[i] = dynmat.b;
		evec3[i] = dynmat.c;
	}
}
*/
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
/*
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
*/
/*
void test_eigen3x3( double * mat, double * evs ){
	Mat3d* pmat  = (Mat3d*)mat;
    Vec3d* es    = (Vec3d*)evs;
	Vec3d* ev1   = (Vec3d*)(evs+3);
	Vec3d* ev2   = (Vec3d*)(evs+6);
	Vec3d* ev3   = (Vec3d*)(evs+9);
	pmat->eigenvals( *es ); 
	pmat->eigenvec( es->a, *ev1 );
	pmat->eigenvec( es->b, *ev2 );
	pmat->eigenvec( es->c, *ev3 );

}
*/
}









