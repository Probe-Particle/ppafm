
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "Vec3.h"
#include "Mat3.h"
#include "spline_hermite.h"
//#include <string.h>

#include <omp.h>

#include "Grid.h"
#include "Forces.h"

// ================= MACROS

#ifdef _WIN64 // Required for exports for ctypes on Windows
    #define DLLEXPORT __declspec(dllexport)
#else
    #define DLLEXPORT
#endif

// ================= CONSTANTS

const double const_eVA_SI = 16.021766;

#define MAX_REF_CN 5
#define MAX_D3_ELEM 94

// ================= GLOBAL VARIABLES

GridShape gridShape;

Vec3d   * gridF = NULL;       // pointer to data    ( 3D vector array [nx,ny,nz,3] )
double  * gridE = NULL;       // pointer to data    ( 3D scalar array [nx,ny,nz]   )

int      natoms       = 0;
double   Morse_alpha  = 0;
int      nCoefPerAtom = 0;
Vec3d  * Ratoms       = NULL;
//double * C6s;
//double * C12s;
//double * kQQs;


// vdW daping coefficients
double ADamp_Const=180.0,ADamp_R2=0.5, ADamp_R4=0.5, ADamp_invR4=0.03,  ADamp_invR8=0.01;

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
//	double dt        = 0.5;           // time step [ abritrary units ]

    double dt        = 0.1;           // time step [ abritrary units ]
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
    //double dt      = dtmax;           // time-step ( variable
    //double acoef   = acoef0;          // damping  ( variable

    inline void setup(){
        dtmax   = RELAX::dt;
        acoef0  = RELAX::damping;
        //dt      = dtmax;
        //acoef   = acoef0;
    }
    /*
    // relaxation step using FIRE algorithm
    inline void move( const Vec3d& f, Vec3d& r, Vec3d& v ){
        double ff = f.norm2();
        double vv = v.norm2();
        double vf = f.dot(v);
        if( vf < 0 ){ // if velocity along direction of force
            v.set( 0.0 );
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
    */

}


struct FIREstate{

    double dt      = FIRE::dtmax;           // time-step ( variable
    double acoef   = FIRE::acoef0;          // damping  ( variable

    inline void setup(){
        dt      = FIRE::dtmax;
        acoef   = FIRE::acoef0;
    }

    // relaxation step using FIRE algorithm
    inline void move( const Vec3d& f, Vec3d& r, Vec3d& v ){
        double ff = f.norm2();
        double vv = v.norm2();
        double vf = f.dot(v);
        if( vf < 0 ){ // if velocity along direction of force
            v.set( 0.0 );
            dt    = dt * FIRE::fdec;
            acoef = FIRE::acoef0;
        }else{       // if velocity against direction of force
            double cf  =     acoef * sqrt(vv/ff);
            double cv  = 1 - acoef;
            v.mul    (    cv );
            v.add_mul( f, cf );	// v = cV * v  + cF * F
            dt     = fmin( dt * FIRE::finc, FIRE::dtmax );
            acoef  = acoef * FIRE::falpha;
        }
        // normal leap-frog times step
        v.add_mul( f , dt );
        r.add_mul( v , dt );
    }

};

// ========= eval force templates

#define dstep       0.1
#define inv_dstep   10.0
#define inv_ddstep  100.0

inline double addAtom_LJ        ( Vec3d dR, Vec3d& fout, double * coefs ){ return addAtomLJ                     ( dR, fout, coefs[0], coefs[1]                     ); }
inline double addAtom_LJ_RE     ( Vec3d dR, Vec3d& fout, double * coefs ){ return addAtomLJ_RE                  ( dR, fout, coefs[0], coefs[1]                     ); }
inline double addAtom_invR6     ( Vec3d dR, Vec3d& fout, double * coefs ){ return addAtomVdW_noDamp             ( dR, fout, coefs[0]                               ); }
inline double addAtom_VdW       ( Vec3d dR, Vec3d& fout, double * coefs ){ return addAtomVdW_dampConst          ( dR, fout, coefs[0]          , ADamp_Const        ); }
inline double addAtom_VdW_R2    ( Vec3d dR, Vec3d& fout, double * coefs ){ return addAtomVdW_addDamp<R2_func>   ( dR, fout, coefs[0], coefs[1], ADamp_R2           ); }
inline double addAtom_VdW_R4    ( Vec3d dR, Vec3d& fout, double * coefs ){ return addAtomVdW_addDamp<R4_func>   ( dR, fout, coefs[0], coefs[1], ADamp_R4           ); }
inline double addAtom_VdW_invR4 ( Vec3d dR, Vec3d& fout, double * coefs ){ return addAtomVdW_addDamp<invR4_func>( dR, fout, coefs[0], coefs[1], ADamp_invR4        ); }
inline double addAtom_VdW_invR8 ( Vec3d dR, Vec3d& fout, double * coefs ){ return addAtomVdW_addDamp<invR8_func>( dR, fout, coefs[0], coefs[1], ADamp_invR8        ); }
inline double addAtom_DFTD3     ( Vec3d dR, Vec3d& fout, double * coefs ){ return addAtomDFTD3                  ( dR, fout, coefs[0], coefs[1], coefs[2], coefs[3] ); }
inline double addAtom_Morse     ( Vec3d dR, Vec3d& fout, double * coefs ){ return addAtomMorse                  ( dR, fout, coefs[0], coefs[1], Morse_alpha        ); }
inline double addAtom_Coulomb_s ( Vec3d dR, Vec3d& fout, double * coefs ){ return addAtomCoulomb                ( dR, fout, coefs[0]                               ); }
inline double addAtom_Coulomb_pz( Vec3d dR, Vec3d& fout, double * coefs ){
    double kqq=coefs[0], E=0;
    Vec3d f; f.set(0.0);
    dR.z -=   dstep; E += addAtomCoulomb( dR, f, -kqq );
    dR.z += 2*dstep; E += addAtomCoulomb( dR, f, +kqq );
    fout.add_mul(f,inv_dstep);
    return    E*inv_dstep;
}
inline double addAtom_Coulomb_dz2( Vec3d dR, Vec3d& fout, double * coefs ){
    double kqq=coefs[0], E=0;
    Vec3d f; f.set(0.0);
                     E += addAtomCoulomb( dR, f, -2*kqq );
    dR.z -=   dstep; E += addAtomCoulomb( dR, f,    kqq );
    dR.z += 2*dstep; E += addAtomCoulomb( dR, f,    kqq );
    fout.add_mul(f,inv_ddstep);
    return    E*inv_ddstep;
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

inline double addAtom_splineR4( Vec3d dR, Vec3d& fout, double * coefs ){
    // Normalization is    (32./105)pi R^3  see:  https://www.wolframalpha.com/input/?i=4*pi*x%5E2*%281-x%5E2%29%5E2+integrate+from+0+to+1
    //constexpr static const double invSqrt2pi = 1/sqrt(2*M_PI);
    double sigma2 = coefs[1];   sigma2*=sigma2;
    double r2     = dR.norm2();
    if( r2 > sigma2 )return 0;
    r2/=sigma2;
    double rf = 1-r2;
    return coefs[0]*rf*rf;
}



inline double addAtom_Gauss( Vec3d dR, Vec3d& fout, double * coefs ){
    //constexpr static const double invSqrt2pi = 1/sqrt(2*M_PI);
    double amp      =   coefs[0];
    double invSigma = 1/coefs[1];
    double r2    = dR.norm2();
    double E = amp * exp( -0.5*sq(r2*invSigma) ); // * invSqrt2pi * invSigma; // normalization should be done in python - rescale amp
    // fout =  TODO
    return E;
}

inline double addAtom_Slater( Vec3d dR, Vec3d& fout, double * coefs ){
    double amp  =   coefs[0];
    double beta = 1/coefs[1];
    double r    = dR.norm();
    double E = amp * exp( -beta*r );
    // fout =  TODO
    return E;
}



// coefs is array of coefficient for each atom; nc is number of coefs for each atom
template<double addAtom_func(Vec3d dR, Vec3d& fout, double * coefs)>
inline void evalCell( int ibuff, const Vec3d& rProbe, void * args ){
    double * coefs = (double*)args;
    //printf(" evalCell : args %i \n", args );
    //printf(" natoms %i nCoefPerAtom %i \n", natoms, nCoefPerAtom );
    double E=0;
    Vec3d f; f.set(0.0);
    for(int i=0; i<natoms; i++){
        //if( ibuff==0 ) printf(" atom[%i] (%g,%g,%g) | %g \n", i, Ratoms[i].x, Ratoms[i].y, Ratoms[i].z, coefs[0] );
        E     += addAtom_func( rProbe-Ratoms[i], f, coefs );
        coefs += nCoefPerAtom;
    }
    //printf( "evalCell[%i] %i (%g,%g,%g) %g\n", ibuff, natoms, rProbe.x, rProbe.y, rProbe.z, E ); exit(0);
    if(gridF) gridF[ibuff].add(f);
    if(gridE) gridE[ibuff] += E;
    //exit(0);
}

// ========== Interpolations

inline void getPPforce( const Vec3d& rTip, const Vec3d& r, Vec3d& f ){
    Vec3d rGrid,drTip;
    rGrid.set( r.dot( gridShape.diCell.a ), r.dot( gridShape.diCell.b ), r.dot( gridShape.diCell.c ) );     // transform position from cartesian world coordinates to coordinates along which Force-Field data are sampled ( non-orthogonal cell )
    drTip.set_sub( r, rTip );                                                             // vector between Probe-particle and tip apex
    f.set    ( interpolate3DvecWrap( gridF, gridShape.n, rGrid ) );                          // force from surface, interpolated from Force-Field data array
    if( TIP::rff_xs ){
        f.add( forceRSpline( drTip, TIP::rff_n, TIP::rff_xs, TIP::rff_ydys ) );			  // force from tip - radial component spline
    }else{
        f.add( forceRSpring( drTip, TIP::kRadial, TIP::lRadial ) );                       // force from tip - radial component harmonic
    }
    drTip.sub( TIP::rPP0 );
    f.add_mul( drTip, TIP::kSpring );      // spring force
}

// relax probe particle position "r" given on particular position of tip (rTip) and initial position "r"
int relaxProbe( int relaxAlg, const Vec3d& rTip, Vec3d& r ){
    Vec3d v; v.set( 0.0 );
    int iter;
    FIREstate fire;
    fire.setup();
    for( iter=0; iter<RELAX::maxIters; iter++ ){
        Vec3d f;  getPPforce( rTip, r, f );
        if( relaxAlg == 1 ){                                                                  // move by either damped-leap-frog ( 0 ) or by FIRE ( 1 )
            //FIRE::move( f, r, v );
            fire.move( f, r, v );
        }else{
            RELAX::move( f, r, v );
        }
        if( f.norm2() < RELAX::convF2 ) break;                                                // check force convergence
    }
    return iter;
}

// =====================================================
// ==========   Export these functions ( to Python )
// ========================================================

extern "C"{

// set basic relaxation parameters
DLLEXPORT void setRelax( int maxIters, double convF2, double dt, double damping ){
    RELAX::maxIters  = maxIters ;
    RELAX::convF2    = convF2;
    RELAX::dt        = dt;
    RELAX::damping   = damping;
    FIRE ::setup();
}

// set FIRE relaxation parameters
DLLEXPORT void setFIRE( double finc, double fdec, double falpha ){
    FIRE::finc    = finc;
    FIRE::fdec    = fdec;
    FIRE::falpha  = falpha;
}

// set pointer to force field array ( the array is usually allocated in python, we can flexibely switch betweeen different precomputed forcefields )
DLLEXPORT void setFF_Fpointer( double * gridF_ ){
    gridF = (Vec3d *)gridF_;
}

// set pointer to force field array ( the array is usually allocated in python, we can flexibely switch betweeen different precomputed forcefields )
DLLEXPORT void setFF_Epointer( double * gridE_ ){
    gridE = gridE_;
}

// set forcefield grid dimension "n"
DLLEXPORT void setGridN( int * n ){
    //gridShape.n.set( *(Vec3i*)n );
    gridShape.n.set( n[2], n[1], n[0] );
    printf( " nxyz  %i %i %i \n", gridShape.n.x, gridShape.n.y, gridShape.n.z );
}

// set forcefield grid lattice vectors "cell"
DLLEXPORT void setGridCell( double * cell ){
    gridShape.setCell( *(Mat3d*)cell );
    gridShape.printCell();
}

// set parameters of the tip like stiffness and equlibirum position in radial and lateral direction
DLLEXPORT void setTip( double lRad, double kRad, double * rPP0, double * kSpring ){
    TIP::lRadial=lRad;
    TIP::kRadial=kRad;
    TIP::rPP0.set(rPP0);
    TIP::kSpring.set(kSpring);
    TIP::makeConsistent();  // rPP0 to be consistent with  lRadial
}

// set parameters of the tip like stiffness and equlibirum position in radial and lateral direction
DLLEXPORT void setTipSpline( int n, double * xs, double * ydys ){
    TIP::rff_n    = n;
    TIP::rff_xs   = xs;
    TIP::rff_ydys = ydys;
}

DLLEXPORT void getInPoints_LJ( int npoints, double * points_, double * FEs, int natoms, double * Ratoms_, double * cLJs ){
    Vec3d * Ratoms=(Vec3d*)Ratoms_; Vec3d * points =(Vec3d*)points_;
    //printf("natoms %i npoints %i \n", natoms, npoints);
    int i4=0;
    //for(int ia=0; ia<natoms; ia++){ printf( " atom %i (%g,%g,%g) %g %g \n", ia,Ratoms[ia].x,Ratoms[ia].y,Ratoms[ia].z, cLJs[ia*2], cLJs[ia*2+1] ); }
    for( int ip=0; ip<npoints; ip++ ){
        double E=0;
        Vec3d f; f.set(0.0);
        Vec3d rProbe = points[ip];
        for(int ia=0; ia<natoms; ia++){ E += addAtomLJ( Ratoms[ia]-rProbe, f, cLJs[ia*2], cLJs[ia*2+1] ); }
        //printf( " point[%i] %i (%g,%g,%g) (%g,%g,%g) %g\n", ip, natoms, rProbe.x,rProbe.y,rProbe.z,  f.x,f.y,f.z, E ); exit(0);
        // point 0 (14.877,9.09954,3) (-0.000131649,8.35068e-05,-2.8037e+13) 7.70439e+11
        FEs[0] = f.x; FEs[1] = f.y; FEs[2] = f.z; FEs[3] = E;
        FEs+=4;
    }
}


DLLEXPORT void computeD3Coeffs(
    const int natoms_, const double *rs, const int *elems, const double *r_cov, const double *r_cut, const double *ref_cn,
    const double *ref_c6, const double *r4r2, const double *k, const double *params, const int elem_pp, double *d3_coeffs
) {

    natoms = natoms_;
    Ratoms = (Vec3d*)rs;

    const double k1  = k[0];
    const double k2  = k[1];
    const double k3  = -k[2];
    const double k12 = -k1 * k2;

    const double s6 = params[0];
    const double s8 = params[1];
    const double a1 = params[2];
    const double a2 = params[3];

    float L[MAX_REF_CN * MAX_REF_CN];

    // Compute reference C6 weights for the probe particle
    int pp_ind = elem_pp - 1;
    double L_pp[MAX_REF_CN];
    int i;
    for (i = 0; i < MAX_REF_CN; i++) {
        double pp_cn = ref_cn[pp_ind * MAX_REF_CN + i];
        if (pp_cn >= 0.0f) { // Values less that 0 zero are invalid and should not be counted.
            L_pp[i] = exp(k3 * pp_cn * pp_cn);
        }
    }
    int max_ref_pp = i;


    for (int ia = 0; ia < natoms; ia++) { // Loop over all atoms

        const Vec3d pos = Ratoms[ia];
        const int elem_ind = elems[ia] - 1;
        const double r_cov_elem = r_cov[elem_ind];

        // Compute the coordination number for this atom
        double cn = 0;
        for (int j = 0; j < natoms; j++) {
            if (j == ia) continue; // No self-interaction for coordination number
            double d = (Ratoms[j] - pos).norm();
            double r = r_cov[elems[j] - 1] + r_cov_elem;
            cn += 1.0f / (1.0f + exp(k12 * r / d + k1));
        }

        // Compute gaussian weights and normalization factor for all of the reference
        // coordination numbers.
        double norm = 0;
        int a, b;
        for (a = 0; a < MAX_REF_CN; a++) {
            double ref_cn_a = ref_cn[elem_ind * MAX_REF_CN + a];
            if (ref_cn_a < 0.0f) break; // Invalid values after this
            double diff_cn_a = ref_cn_a - cn;
            double L_a = exp(k3 * diff_cn_a * diff_cn_a);
            for (b = 0; b < max_ref_pp; b++) {
                double L_ab = L_a * L_pp[b];
                norm += L_ab;
                L[a * MAX_REF_CN + b] = L_ab;
            }
        }
        int max_ref_a = a;

        // If the coordination number is so high that the gaussian weights are all zero,
        // then we put all of the weight on the highest reference coordination number.
        if (norm == 0) {
            int a_ind = (max_ref_a - 1) * MAX_REF_CN;
            for (b = 0; b < max_ref_pp; b++) {
                double L_ab = L_pp[b];
                norm += L_ab;
                L[a_ind + b] = L_ab;
            }
        }

        // Compute C6 coefficient as a linear combination of reference C6 values
        int n_zw = MAX_REF_CN * MAX_REF_CN;
        int n_yzw = MAX_D3_ELEM * n_zw;
        int pair_ind = elem_ind * n_yzw + pp_ind * n_zw;  // ref_c6 shape = (MAX_D3_ELEM, MAX_D3_ELEM, MAX_REF_CN, MAX_REF_CN)
        double c6 = 0;
        for (int a = 0; a < max_ref_a; a++) {
            int a_ind = a * MAX_REF_CN;
            for (int b = 0; b < max_ref_pp; b++) {
                int L_ind = a_ind + b;
                int c6_ind = pair_ind + L_ind;
                c6 += L[L_ind] * ref_c6[c6_ind];
            }
        }
        c6 /= norm;

        // The C8 coefficient is inferred from the C6 coefficient
        double qq = 3 * r4r2[elem_ind] * r4r2[pp_ind];
        double c8 = qq * c6;

        // Compute damping constants
        double R0   = a1 * sqrt(qq) + a2;
        double R0_2 = R0 * R0;
        double R0_6 = R0_2 * R0_2 * R0_2;
        double R0_8 = R0_6 * R0_2;

        // Save coefficients
        d3_coeffs[4 * ia    ] = c6 * s6;
        d3_coeffs[4 * ia + 1] = c8 * s8;
        d3_coeffs[4 * ia + 2] = R0_6;
        d3_coeffs[4 * ia + 3] = R0_8;

    }

}

DLLEXPORT void getLennardJonesFF( int natoms_, double * Ratoms_, double * cLJs ){
    natoms=natoms_; Ratoms=(Vec3d*)Ratoms_; nCoefPerAtom = 2;
    Vec3d r0; r0.set(0.0,0.0,0.0);
    //interateGrid3D < evalCell < addAtom_LJ  > >( r0, gridShape.n, gridShape.dCell, cLJs );
    interateGrid3D_omp < evalCell < addAtom_LJ  > >( r0, gridShape.n, gridShape.dCell, cLJs );
}

DLLEXPORT void getVdWFF( int natoms_, double * Ratoms_, double * cLJs ){
    natoms=natoms_; Ratoms=(Vec3d*)Ratoms_; nCoefPerAtom = 2;
    Vec3d r0; r0.set(0.0,0.0,0.0);
    //interateGrid3D < evalCell < addAtom_VdW  > >( r0, gridShape.n, gridShape.dCell, cLJs );
    interateGrid3D_omp < evalCell < addAtom_VdW  > >( r0, gridShape.n, gridShape.dCell, cLJs );

}

DLLEXPORT void getDFTD3FF(int natoms_, double * Ratoms_, double *d3_coeffs){
    natoms = natoms_; Ratoms = (Vec3d*)Ratoms_; nCoefPerAtom = 4;
    Vec3d r0; r0.set(0.0,0.0,0.0);
    //interateGrid3D < evalCell < addAtom_DFTD3  > >( r0, gridShape.n, gridShape.dCell, d3_coeffs );
    interateGrid3D_omp < evalCell < addAtom_DFTD3  > >( r0, gridShape.n, gridShape.dCell, d3_coeffs );

}

DLLEXPORT void getMorseFF( int natoms_, double * Ratoms_, double * REs, double alpha ){
    natoms=natoms_; Ratoms=(Vec3d*)Ratoms_; nCoefPerAtom = 2; Morse_alpha = alpha;
    Vec3d r0; r0.set(0.0,0.0,0.0);
    //interateGrid3D < evalCell < addAtom_Morse > >( r0, gridShape.n, gridShape.dCell, REs );
    interateGrid3D_omp < evalCell < addAtom_Morse > >( r0, gridShape.n, gridShape.dCell, REs );
}

// sample Coulomb Force-field on 3D mesh over provided set of atoms with positions Rs_[i] with constant kQQs  =  - k_coulomb * Q_ProbeParticle * Q[i]
// results are sampled according to grid parameters defined in "namespace FF" and stored in array to which points by "double * FF::grid"
DLLEXPORT void getCoulombFF( int natoms_, double * Ratoms_, double * kQQs, int kind ){
    natoms=natoms_; Ratoms=(Vec3d*)Ratoms_; nCoefPerAtom = 1;
    Vec3d r0; r0.set(0.0,0.0,0.0);
    //printf(" kind %i \n", kind );
    switch(kind){
        //case 0: interateGrid3D < evalCell < foo  > >( r0, gridShape.n, gridShape.dCell, kQQs_ );
        //case 0: interateGrid3D < evalCell < addAtom_Coulomb_s   > >( r0, gridShape.n, gridShape.dCell, kQQs ); break;
        //case 1: interateGrid3D < evalCell < addAtom_Coulomb_pz  > >( r0, gridShape.n, gridShape.dCell, kQQs ); break;
        //case 2: interateGrid3D < evalCell < addAtom_Coulomb_dz2 > >( r0, gridShape.n, gridShape.dCell, kQQs ); break;

        case 0: interateGrid3D_omp < evalCell < addAtom_Coulomb_s   > >( r0, gridShape.n, gridShape.dCell, kQQs ); break;
        case 1: interateGrid3D_omp < evalCell < addAtom_Coulomb_pz  > >( r0, gridShape.n, gridShape.dCell, kQQs ); break;
        case 2: interateGrid3D_omp < evalCell < addAtom_Coulomb_dz2 > >( r0, gridShape.n, gridShape.dCell, kQQs ); break;
    }
}

DLLEXPORT void evalRadialFF( int n, double* rs, double* coefs, double* Es, double* Fs, int kind, double ADamp_ ){
    //printf( "evalRadialFF kind=%i Adamp=%g \n", kind, ADamp_ );
    for(int i=0; i<n; i++){
        Vec3d dR    = Vec3d{rs[i],0,0};
        Vec3d fout  = Vec3dZero;
        double E = 0;
        switch(kind){
            case -3:                                        E=addAtom_LJ_RE    ( dR, fout, coefs ); break;
            case -2:                                        E=addAtom_LJ       ( dR, fout, coefs ); break;
            case -1:                                        E=addAtom_invR6    ( dR, fout, coefs ); break;
            case  0: if(ADamp_>0){ ADamp_Const = ADamp_; }; E=addAtom_VdW      ( dR, fout, coefs ); break;
            case  1: if(ADamp_>0){ ADamp_R2    = ADamp_; }; E=addAtom_VdW_R2   ( dR, fout, coefs ); break;
            case  2: if(ADamp_>0){ ADamp_R4    = ADamp_; }; E=addAtom_VdW_R4   ( dR, fout, coefs ); break;
            case  3: if(ADamp_>0){ ADamp_invR4 = ADamp_; }; E=addAtom_VdW_invR4( dR, fout, coefs ); break;
            case  4: if(ADamp_>0){ ADamp_invR8 = ADamp_; }; E=addAtom_VdW_invR8( dR, fout, coefs ); break;
        }
        Fs[i]=fout.x;
        Es[i]=E;
    }
}

DLLEXPORT void getVdWFF_RE( int natoms_, double * Ratoms_, double * REs, int kind, double ADamp_=-1.0 ){
    //printf( "DEBUG getVdWFF_RE(kind=%i,ADamp=%g) \n", kind, ADamp_ );
    natoms=natoms_; Ratoms=(Vec3d*)Ratoms_; nCoefPerAtom = 2;
    Vec3d r0; r0.set(0.0,0.0,0.0);
    //if(ADamp>0){ ADamp = ADamp_; }
    switch(kind){
        //case 0: if(ADamp_>0){ ADamp_Const = ADamp_; }; E=addAtom_VdW      ( dR, fout, coefs ); break;
        //case 1: if(ADamp_>0){ ADamp_R2    = ADamp_; } interateGrid3D<evalCell<addAtom_VdW_R2   >>( r0, gridShape.n, gridShape.dCell, REs ); break;
        //case 2: if(ADamp_>0){ ADamp_R4    = ADamp_; } interateGrid3D<evalCell<addAtom_VdW_R4   >>( r0, gridShape.n, gridShape.dCell, REs ); break;
        //case 3: if(ADamp_>0){ ADamp_invR4 = ADamp_; } interateGrid3D<evalCell<addAtom_VdW_invR4>>( r0, gridShape.n, gridShape.dCell, REs ); break;
        //case 4: if(ADamp_>0){ ADamp_invR8 = ADamp_; } interateGrid3D<evalCell<addAtom_VdW_invR8>>( r0, gridShape.n, gridShape.dCell, REs ); break;

        case 1: if(ADamp_>0){ ADamp_R2    = ADamp_; } interateGrid3D_omp<evalCell<addAtom_VdW_R2   >>( r0, gridShape.n, gridShape.dCell, REs ); break;
        case 2: if(ADamp_>0){ ADamp_R4    = ADamp_; } interateGrid3D_omp<evalCell<addAtom_VdW_R4   >>( r0, gridShape.n, gridShape.dCell, REs ); break;
        case 3: if(ADamp_>0){ ADamp_invR4 = ADamp_; } interateGrid3D_omp<evalCell<addAtom_VdW_invR4>>( r0, gridShape.n, gridShape.dCell, REs ); break;
        case 4: if(ADamp_>0){ ADamp_invR8 = ADamp_; } interateGrid3D_omp<evalCell<addAtom_VdW_invR8>>( r0, gridShape.n, gridShape.dCell, REs ); break;

        // case 0: interateGrid3D<evalCell<addAtomVdW_addDamp<R2_func>  >>>( r0, gridShape.n, gridShape.dCell, REs ); break;
        // case 1: interateGrid3D<evalCell<addAtomVdW_addDamp<R4_func>  >>>( r0, gridShape.n, gridShape.dCell, REs ); break;
        // case 2: interateGrid3D<evalCell<addAtomVdW_addDamp<invr4_func>>>( r0, gridShape.n, gridShape.dCell, REs ); break;
        // case 2: interateGrid3D<evalCell<addAtomVdW_addDamp<invr8_func>>>( r0, gridShape.n, gridShape.dCell, REs ); break;
    }
}

DLLEXPORT void getGaussDensity( int natoms_, double * Ratoms_, double * cRAs ){
    natoms=natoms_; Ratoms=(Vec3d*)Ratoms_; nCoefPerAtom = 2;
    Vec3d r0; r0.set(0.0,0.0,0.0);
    Vec3d* gridF_=gridF; gridF=0;
    //interateGrid3D < evalCell < addAtom_Gauss  > >( r0, gridShape.n, gridShape.dCell, cRAs );
    interateGrid3D_omp < evalCell < addAtom_Gauss  > >( r0, gridShape.n, gridShape.dCell, cRAs );
    gridF=gridF_;
}

DLLEXPORT void getSlaterDensity( int natoms_, double * Ratoms_, double * cRAs ){
    natoms=natoms_; Ratoms=(Vec3d*)Ratoms_; nCoefPerAtom = 2;
    Vec3d r0; r0.set(0.0,0.0,0.0);
    Vec3d* gridF_=gridF; gridF=0;
    //interateGrid3D < evalCell < addAtom_Slater > >( r0, gridShape.n, gridShape.dCell, cRAs );
    interateGrid3D_omp < evalCell < addAtom_Slater > >( r0, gridShape.n, gridShape.dCell, cRAs );
    gridF=gridF_;
}

DLLEXPORT void getDensityR4spline( int natoms_, double * Ratoms_, double * cRAs ){
    natoms=natoms_; Ratoms=(Vec3d*)Ratoms_; nCoefPerAtom = 2;
    Vec3d r0; r0.set(0.0,0.0,0.0);
    Vec3d* gridF_=gridF; gridF=0;
    //interateGrid3D < evalCell < addAtom_splineR4 > >( r0, gridShape.n, gridShape.dCell, cRAs );
    interateGrid3D_omp < evalCell < addAtom_splineR4 > >( r0, gridShape.n, gridShape.dCell, cRAs );
    gridF=gridF_;
}


// relax one stroke of tip positions ( stored in 1D array "rTips_" ) using precomputed 3D force-field on grid
// returns position of probe-particle after relaxation in 1D array "rs_" and force between surface probe particle in this relaxed position in 1D array "fs_"
// for efficiency, starting position of ProbeParticle in new point (next postion of Tip) is derived from relaxed postion of ProbeParticle from previous point
// there are several strategies how to do it which are choosen by parameter probeStart
DLLEXPORT int relaxTipStroke( int probeStart, int relaxAlg, int nstep, double * rTips_, double * rs_, double * fs_ ){
    Vec3d * rTips = (Vec3d*) rTips_;
    Vec3d * rs    = (Vec3d*) rs_;
    Vec3d * fs    = (Vec3d*) fs_;
    int itrmin=RELAX::maxIters+1,itrmax=0,itrsum=0;
    Vec3d rTip,rProbe;
    rTip  .set    ( rTips[0]      );
    rProbe.set_add( rTip, TIP::rPP0 );
    //printf( " rTip0: %f %f %f  rProbe0: %f %f %f \n", rTip.x, rTip.y, rTip.z, rProbe.x, rProbe.y, rProbe.z  );
    //gridShape.printCell(); exit(0);
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

// relax one stroke of tip positions ( stored in 1D array "rTips_" ) using precomputed 3D force-field on grid
// returns position of probe-particle after relaxation in 1D array "rs_" and force between surface probe particle in this relaxed position in 1D array "fs_"
// for efficiency, starting position of ProbeParticle in new point (next postion of Tip) is derived from relaxed postion of ProbeParticle from previous point
// there are several strategies how to do it which are choosen by parameter probeStart
DLLEXPORT int relaxTipStrokes_omp( int nx, int ny, int probeStart, int relaxAlg, int nstep, double * rTips_, double * rs_, double * fs_ ){
    printf( "relaxTipStrokes_omp()  nx %i ny %i nstep %i \n", nx, ny, nstep );
    int ndone=0;
    #pragma omp parallel for collapse(2) shared( nx, ny, probeStart, relaxAlg, nstep, rTips_, rs_, fs_, ndone )
    for (int ix=0; ix<nx; ix++){
        for (int iy=0; iy<ny; iy++){
            int ioff = (ix + iy*nx)*nstep;
            relaxTipStroke( probeStart, relaxAlg, nstep, rTips_+ioff*3, rs_+ioff*3, fs_+ioff*3 );
            if( omp_get_thread_num()==0 ){
                ndone++;
                if( ndone%100==0 ){
                    int ncpu=omp_get_num_threads();
                    printf( "\r %2.2f %% DONE (ncpu=%i)", (100.0*ndone*ncpu)/(nx*ny), ncpu );
                    fflush(stdout);
                }
            }
        }
    }
    return 0;
}

DLLEXPORT void stiffnessMatrix( double ddisp, int which, int n, double * rTips_, double * rPPs_, double * eigenvals_, double * evec1_, double * evec2_, double * evec3_ ){
    //printf( "C++ stiffnessMatrix() n=%i \n", n );
    Vec3d * rTips     = (Vec3d*) rTips_;
    Vec3d * rPPs      = (Vec3d*) rPPs_;
    Vec3d * eigenvals = (Vec3d*) eigenvals_;
    Vec3d * evec1     = (Vec3d*) evec1_;
    Vec3d * evec2     = (Vec3d*) evec2_;
    Vec3d * evec3     = (Vec3d*) evec3_;
    //printf( "C++ stiffnessMatrix() gridShape.n(%i,%i,%i) \n", gridShape.n.x, gridShape.n.y, gridShape.n.z  );
    //printf( "C++ stiffnessMatrix() gridF=%li \n", (long)gridF  );
    //Vec3d gf=gridF[0];                                        printf( "gridF[0 ] (%g,%g,%g) \n",gf.x,gf.y,gf.z );
    //gf=gridF[ gridShape.n.x*gridShape.n.y*gridShape.n.z -1 ]; printf( "gridF[-1] (%g,%g,%g) \n",gf.x,gf.y,gf.z);
    //Vec3d pmin,pmax; pmin.set( 1e+300, 1e+300, 1e+300 ); pmax.set( -1e+300, -1e+300, -1e+300 );
    for(int i=0; i<n; i++){
        Vec3d rTip,rPP,f1,f2;
        rTip.set( rTips[i] );
        rPP.set ( rPPs[i]  );
        Mat3d dynmat;
        //pmin.setIfLower(rPP); pmax.setIfGreater(rPP);
        // eval dynamical matrix    D_xy = df_y/dx    = ( f(r0+dx).y - f(r0-dx).y ) / (2*dx)
        rPP.x-=ddisp; getPPforce( rTip, rPP, f1 ); rPP.x+=2*ddisp; getPPforce( rTip, rPP, f2 );  rPP.x-=ddisp; dynmat.a.set_sub(f2,f1); dynmat.a.mul(-0.5/ddisp);
        rPP.y-=ddisp; getPPforce( rTip, rPP, f1 ); rPP.y+=2*ddisp; getPPforce( rTip, rPP, f2 );  rPP.y-=ddisp; dynmat.b.set_sub(f2,f1); dynmat.b.mul(-0.5/ddisp);
        rPP.z-=ddisp; getPPforce( rTip, rPP, f1 ); rPP.z+=2*ddisp; getPPforce( rTip, rPP, f2 );  rPP.z-=ddisp; dynmat.c.set_sub(f2,f1); dynmat.c.mul(-0.5/ddisp);
        // symmetrize - to make sure that our symmetric matrix solver work properly
        double tmp;
        tmp = 0.5*(dynmat.xy + dynmat.yx); dynmat.xy = tmp; dynmat.yx = tmp;
        tmp = 0.5*(dynmat.yz + dynmat.zy); dynmat.yz = tmp; dynmat.zy = tmp;
        tmp = 0.5*(dynmat.zx + dynmat.xz); dynmat.zx = tmp; dynmat.xz = tmp;
        // solve mat
        Vec3d evals; dynmat.eigenvals( evals ); Vec3d temp;
        double eval_check = evals.a * evals.b * evals.c;
        //if( fabs(eval_check) < 1e-16 ){  };
        if( isnan(eval_check) ){  printf( "C++ stiffnessMatrix[%i] is NaN evals (%g,%g,%g) \n", i, evals.a, evals.b, evals.c ); exit(0); }
        // sort eigenvalues
        if( evals.a > evals.b ){ tmp=evals.a; evals.a=evals.b; evals.b=tmp; }
        if( evals.b > evals.c ){ tmp=evals.b; evals.b=evals.c; evals.c=tmp; }
        if( evals.a > evals.b ){ tmp=evals.a; evals.a=evals.b; evals.b=tmp; }
        // output eigenvalues and eigenvectors
        eigenvals[i] = evals;
        if(which>0) dynmat.eigenvec( evals.a, evec1[i] );
        if(which>1) dynmat.eigenvec( evals.b, evec2[i] );
        if(which>2) dynmat.eigenvec( evals.c, evec3[i] );
        //evec1[i] = dynmat.a;
        //evec2[i] = dynmat.b;
        //evec3[i] = dynmat.c;
    }
   // printf( "C++ stiffnessMatrix() DONE! pmin(%g,%g,%g) pmax(%g,%g,%g) \n", pmin.x, pmin.y, pmin.z, pmax.x, pmax.y, pmax.z );
}

DLLEXPORT void subsample_uniform_spline( double x0, double dx, int n, double * ydys, int m, double * xs_, double * ys_ ){
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

DLLEXPORT void subsample_nonuniform_spline( int n, double * xs, double * ydys, int m, double * xs_, double * ys_ ){
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

DLLEXPORT void test_force( int type, int n, double * r0_, double * dr_, double * R_, double * fs_ ){
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

DLLEXPORT void test_eigen3x3( double * mat, double * evs ){
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

} // extern "C"{
