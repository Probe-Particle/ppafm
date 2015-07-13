
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

namespace FF {
	Vec3d * grid;
//	Vec3d   step;
//	Vec3d   invStep;
	Mat3d   dCell;
	Mat3d   diCell;
	Vec3i   n;
}

namespace TIP{
	Vec3d   rPP0;       // equilibirum bending position
	Vec3d   kSpring;    // bending stiffness ( z component usually zero )
	double  lRadial;    // radial PP-tip distance
	double  kRadial;    // radial PP-tip stiffness

	void makeConsistent(){ 
		if( fabs(kRadial) > 1e-8 ){  
			rPP0.z = -sqrt( lRadial*lRadial - rPP0.x*rPP0.x - rPP0.y*rPP0.y );  
			printf(" rPP0 %f %f %f \n", rPP0.x, rPP0.y, rPP0.z );
		}
	}
}

namespace RELAX{
	int    maxIters  = 1000;
	double convF2    = 1.0e-8;
	//double convF2    = 1.0e-6;
	double dt        = 0.5;
	double damping   = 0.1;

	inline void  move( const Vec3d& f, Vec3d& r, Vec3d& v ){
		v.mul( 1 - damping );
		v.add_mul( f, dt );
		r.add_mul( v, dt );	
	}
}

namespace FIRE{
// "Fast Inertial Realxation Engine" according to  
// Bitzek, E., Koskinen, P., Gähler, F., Moseler, M. & Gumbsch, P. Structural relaxation made simple. Phys. Rev. Lett. 97, 170201 (2006).
// Eidel, B., Stukowski, A. & Schröder, J. Energy-Minimization in Atomic-to-Continuum Scale-Bridging Methods. Pamm 11, 509–510 (2011).
// http://users.jyu.fi/~pekkosk/resources/pdf/FIRE.pdf

	double finc    = 1.1; 
	double fdec    = 0.5;
	double falpha  = 0.99;
	double dtmax   = RELAX::dt; 
	double acoef0  = RELAX::damping;

	double dt      = dtmax;
	double acoef   = acoef0;

	inline void setup(){
		dtmax   = RELAX::dt; 
		acoef0  = RELAX::damping;
		dt      = dtmax;
		acoef   = acoef0;
	}

	inline void move( const Vec3d& f, Vec3d& r, Vec3d& v ){
		double ff = f.norm2();
		double vv = v.norm2();
		double vf = f.dot(v);
		if( vf < 0 ){
			v.set( 0.0d );
			dt    = dt * fdec;
		  	acoef = acoef0;
		}else{
			double cf  =     acoef * sqrt(vv/ff);
			double cv  = 1 - acoef;
			v.mul    ( cv );
			v.add_mul( f, cf );	// v = cV * v  + cF * F
			dt     = fmin( dt * finc, dtmax );
			acoef  = acoef * falpha;
		}
		v.add_mul( f , dt );
		r.add_mul( v , dt );	
	}
}

// ========== Classical force field

inline Vec3d forceRSpring( const Vec3d& dR, double k, double l0 ){
	double l = sqrt( dR.norm2() );
	Vec3d f; f.set_mul( dR, k*( l - l0 )/l );
	return f;
}

inline Vec3d forceLJ( const Vec3d& dR, double c6, double c12 ){
	double ir2  = 1.0d/ dR.norm2( ); 
	double ir6  = ir2*ir2*ir2;
	double ir12 = ir6*ir6;
	return dR * ( ( 6*ir6*c6 -12*ir12*c12 ) * ir2  );
}

inline Vec3d forceCoulomb( const Vec3d& dR, double kqq ){
	//const double kcoulomb   = 14.3996448915; 
	double ir2  = 1.0d/ dR.norm2( ); 
	double ir   = sqrt( ir2 );
	return dR * kqq * ir * ir2;
}


inline Vec3d getAtomsForceLJ( const Vec3d& rProbe, int n, Vec3d * Rs, double * C6, double * C12 ){
	Vec3d f; f.set(0.0d);
	for(int i=0; i<n; i++){
		f.add( forceLJ( Rs[i] - rProbe, C6[i], C12[i] ) );
	}
	return f;
}

inline Vec3d getAtomsForceCoulomb( const Vec3d& rProbe, int n, Vec3d * Rs, double * kQQs ){
	Vec3d f; f.set(0.0d);
	for(int i=0; i<n; i++){	f.add( forceCoulomb( Rs[i] - rProbe, kQQs[i] ) );	}
	//for(int i=0; i<n; i++){	f.add( kQQs[i]  );	}
	return f;
}




// ========== Interpolations

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


inline int relaxProbe( int relaxAlg, const Vec3d& rTip, Vec3d& r ){
	Vec3d v; v.set( 0.0d );
	int iter;
	//printf( " alg %i r  %f %f %f  rTip  %f %f %f \n", relaxAlg, r.x,r.y,r.z,  rTip.x, rTip.y, rTip.z );

	for( iter=0; iter<RELAX::maxIters; iter++ ){
		Vec3d rGrid,f,drTip; 
		//rGrid.set_mul(r, FF::invStep );
		rGrid.set( r.dot( FF::diCell.a ), r.dot( FF::diCell.b ), r.dot( FF::diCell.c ) );  // nonOrthogonal cells
		//printf( "   iter  %i rGrid  %f %f %f \n", iter, rGrid.a,rGrid.b, rGrid.c );
		drTip.set_sub( r, rTip );
		f.set    ( interpolate3DvecWrap( FF::grid, FF::n, rGrid ) );
		f.add    ( forceRSpring( drTip, TIP::kRadial, TIP::lRadial ) );
		drTip.sub( TIP::rPP0 );
		f.add_mul( drTip, TIP::kSpring );      // spring force
		if( relaxAlg == 1 ){
			FIRE::move( f, r, v );
		}else{
			RELAX::move( f, r, v );
		}			
		//printf( "     %i r  %f %f %f  f  %f %f %f \n", iter, r.x,r.y,r.z,  f.x,f.y,f.z );
		if( f.norm2() < RELAX::convF2 ) break;
	}
	return iter;
}


// =====================================================
// ==========   Export these functions ( to Python )
// ========================================================

extern "C"{

void setRelax( int maxIters, double convF2, double dt, double damping ){
	RELAX::maxIters  = maxIters ;
	RELAX::convF2    = convF2;
	RELAX::dt        = dt;
	RELAX::damping   = damping;
	FIRE ::setup();
}

void setFIRE( double finc, double fdec, double falpha ){
	FIRE::finc    = finc; 
	FIRE::fdec    = fdec;
	FIRE::falpha  = falpha;
}

void setFF_Pointer( double * grid ){
	FF::grid = (Vec3d *)grid;
}

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

void setTip( double lRad, double kRad, double * rPP0, double * kSpring ){  
	TIP::lRadial=lRad; 
	TIP::kRadial=kRad;  
	TIP::rPP0.set(rPP0);   
	TIP::kSpring.set(kSpring); 
	TIP::makeConsistent();  // rPP0 to be consistent with  lRadial
}

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
					Vec3d f; f.set(0.0);
					for(int iatom=0; i<natom; i++){
						// only this line differs
						f.add( forceLJ( Rs[iatom] - rProbe, C6[iatom], C12[iatom] ) );
					}
					FF::grid[ i3D( ia, ib, ic ) ].add( f );
				} 
				rProbe.add_mul( FF::dCell.c, -nz );
			} 
			rProbe.add_mul( FF::dCell.b, -ny );
		}
	}

	void getCoulombFF( int natom, double * Rs_, double * kQQs ){
		Vec3d * Rs = (Vec3d*) Rs_;
		int nx  = FF::n.x;
		int ny  = FF::n.y;
		int nz  = FF::n.z;
		int nxy = ny * nx;
		Vec3d rProbe;  rProbe.set( 0.0, 0.0, 0.0 ); // we may shift here

		for ( int i=0; i<natom; i++ ){ 
			printf( " atom %i   q=  %f \n", i, kQQs[i] );
		}

		for ( int ia=0; ia<nx; ia++ ){ 
			printf( " ia %i \n", ia );
			rProbe.add( FF::dCell.a );  
			for ( int ib=0; ib<ny; ib++ ){ 
				rProbe.add( FF::dCell.b );
				for ( int ic=0; ic<nz; ic++ ){
					rProbe.add( FF::dCell.c );
					Vec3d f; f.set(0.0);
					for(int iatom=0; i<natom; i++){
						// only this line differs
						f.add( forceCoulomb( Rs[iatom] - rProbe, kQQs[iatom] );
					}
					FF::grid[ i3D( ia, ib, ic ) ].add( f );
				} 
				rProbe.add_mul( FF::dCell.c, -nz );
			} 
			rProbe.add_mul( FF::dCell.b, -ny );
		}
	}


int relaxTipStroke ( int probeStart, int relaxAlg, int nstep, double * rTips_, double * rs_, double * fs_ ){
	Vec3d * rTips = (Vec3d*) rTips_;
	Vec3d * rs    = (Vec3d*) rs_;
	Vec3d * fs    = (Vec3d*) fs_;
	int itrmin=RELAX::maxIters+1,itrmax=0,itrsum=0;
	Vec3d rTip,rProbe;
	rTip  .set    ( rTips[0]      );
	rProbe.set_add( rTip, TIP::rPP0 );
	//printf( " rTip0: %f %f %f  rProbe0: %f %f %f \n", rTip.x, rTip.y, rTip.z, rProbe.x, rProbe.y, rProbe.z  );
	for( int i=0; i<nstep; i++ ){
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
		int itr = relaxProbe( relaxAlg, rTip, rProbe );
		if( itr>RELAX::maxIters ){
			printf( " not converged in %i iterations \n", RELAX::maxIters );
			printf( "exiting \n" );	break;
		}
		//printf( " %i  %i    %f %f %f   %f %f %f \n", i, itr, rTip.x, rTip.y, rTip.z, rProbe.x, rProbe.y, rProbe.z  );
		Vec3d rGrid; 
		rGrid.set( rProbe.dot( FF::diCell.a ), rProbe.dot( FF::diCell.b ), rProbe.dot( FF::diCell.c ) ); 
		rs[i].set( rProbe                               );
		fs[i].set( interpolate3DvecWrap( FF::grid, FF::n, rGrid ) );
		itrsum += itr;
		itrmin  = ( itr < itrmin ) ? itr : itrmin;
		itrmax  = ( itr > itrmax ) ? itr : itrmax;
	}
	//printf( " itr min, max, average %i %i %f \n", itrmin, itrmax, itrsum/(double)nstep );
	return itrsum;
}

void initProbeParticle(void){

}

}









