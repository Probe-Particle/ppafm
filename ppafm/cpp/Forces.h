
#ifndef Forces_h
#define Forces_h

#include "fastmath_light.h"
#include "Vec2.h"
#include "Vec3.h"

#define COULOMB_CONST  14.399645f

#define RSAFE   1.0e-4f
#define R2SAFE  1.0e-8f
#define F2MAX   10.0f
#define R2_D3_CUTOFF 400.0f

void sum(int n, Vec3d* ps, Vec3d& psum){ for(int i=0;i<n;i++){ psum.add(ps[i]); } };

void sumTroq(int n, Vec3d* fs, Vec3d* ps, const Vec3d& cog, const Vec3d& fav, Vec3d& torq){
    for(int i=0;i<n;i++){  torq.add_cross(ps[i]-cog,fs[i]-fav);  }
    //for(int i=0;i<n;i++){  torq.add_cross(ps[i],fs[i]);  }
}

void checkForceInvariatns( int n, Vec3d* fs, Vec3d* ps, Vec3d& cog, Vec3d& fsum, Vec3d& torq ){
    cog =Vec3dZero;
    fsum=Vec3dZero;
    torq=Vec3dZero;
    double dw = 1./n;
    sum(n, ps, cog ); cog.mul(dw);
    sum(n, fs, fsum); //cog.mul(dw);
    sumTroq(n, fs, ps, cog, fsum*dw, torq );
}

inline double boxForce1D(double x, double xmin, double xmax, double k){
    double f=0;
    if(k<0) return 0;
    if(x>xmax){ f+=k*(xmax-x); }
    if(x<xmin){ f+=k*(xmin-x); }
    return f;
}

inline void boxForce(const Vec3d p, Vec3d& f,const Vec3d& pmin, const Vec3d& pmax, const Vec3d& k){
    f.x+=boxForce1D( p.x, pmin.x, pmax.x, k.x);
    f.y+=boxForce1D( p.y, pmin.y, pmax.y, k.y);
    f.z+=boxForce1D( p.z, pmin.z, pmax.z, k.z);
}

inline double evalCos2(const Vec3d& hi, const Vec3d& hj, Vec3d& fi, Vec3d& fj, double k, double c0){
    double c    = hi.dot(hj) - c0;
    double dfc  =  k*-2*c;
    fi.add_mul(hj,dfc);
    fj.add_mul(hi,dfc);
    return k*c*c;
}

inline double evalCos2_o(const Vec3d& hi, const Vec3d& hj, Vec3d& fi, Vec3d& fj, double k, double c0){
    double c    = hi.dot(hj) - c0;
    double dfc  =  k*-2*c;
    double dfcc = -c*dfc;
    fi.add_lincomb( dfc,hj, dfcc,hi );
    fj.add_lincomb( dfc,hi, dfcc,hj );
    return k*c*c;
}

inline double evalCosHalf(const Vec3d& hi, const Vec3d& hj, Vec3d& fi, Vec3d& fj, double k, Vec2d cs ){
    Vec3d h; h.set_add( hi, hj );
    double c2 = h.norm2()*0.25;               // cos(a/2) = |ha+hb|
    double s2 = 1-c2;
    double c = sqrt(c2);
    double s = sqrt(s2);
    cs.udiv_cmplx({c,s});
    double E         =  k*( 1 - cs.x );  // just for debug ?
    double fr        = -k*(     cs.y );
    fr /= 2*c*s;  // 1/sin(2a)
    c2 *=-2*fr;
    Vec3d fa,fb;
    fi.set_lincomb( fr,h,  c2,hi );
    fj.set_lincomb( fr,h,  c2,hj );
    return E;
}


// ================= BEGIN:  From ProbeParticle.cpp

// radial spring constrain - force length of vector |dR| to be l0
inline Vec3d forceRSpring( const Vec3d& dR, double k, double l0 ){
    double l = sqrt( dR.norm2() );
    Vec3d f; f.set_mul( dR, k*( l - l0 )/l );
    return f;
}

inline Vec3d forceSpringRotated( const Vec3d& dR, const Vec3d& Fw, const Vec3d& Up, const Vec3d& R0, const Vec3d& K ){
    // dR - vector between actual PPpos and anchor point (in global coords)
    // Fw - forward direction of anchor coordinate system (previous bond direction; e.g. Tip->C for C->O) (in global coords)
    // Up - Up vector --,,-- ; e.g. x axis (1,0,0), defines rotation of your tip (in global coords)
    // R0 - equlibirum position of PP (in local coords)
    // K  - stiffness (ka,kb,kc) along local coords
    // return force (in global coords)
    Mat3d rot; Vec3d dR_,f_,f;
    rot.fromDirUp( Fw*(1/Fw.norm()), Up );  // build orthonormal rotation matrix
    rot.dot_to  ( dR, dR_   );              // transform dR to rotated coordinate system
    f_ .set_mul ( dR_-R0, K );              // spring force (in rotated system)
    // here you can easily put also other forces - e.g. Torsion etc.
    rot.dot_to_T( f_, f );                 // transform force back to world system
    return f;
}

// Lennard-Jones force between two atoms a,b separated by vector dR = Ra - Rb
inline double addAtomLJ_RE( const Vec3d& dR, Vec3d& fout, double R0, double E0 ){
    double ir2  = 1/( dR.norm2( ) + R2SAFE );
    double u2   = R0*R0 *ir2;
    double u6   = u2*u2*u2;
    //printf( "r %g u2 %g R*R %g ir2 %g u6 %g R0 %g E0 %g \n", 1/sqrt(ir2), u2, R0*R0, ir2, u6, R0, E0 );
    double E6   = E0*u6;
    double E12  = E6*u6;
    fout.add_mul( dR , 12*( E12 - E6 ) * ir2 );
    return E12 - 2*E6;
}

// Lennard-Jones force between two atoms a,b separated by vector dR = Ra - Rb
inline double addAtomLJ( const Vec3d& dR, Vec3d& fout, double c6, double c12 ){
    double ir2  = 1.0/ ( dR.norm2( ) + R2SAFE );
    double ir6  = ir2*ir2*ir2;
    double E6   = c6  * ir6;
    double E12  = c12 * ir6*ir6;
    //return dR * ( ( 6*ir6*c6 -12*ir12*c12 ) * ir2  );
    fout.add_mul( dR , ( 12*E12 - 6*E6 ) * ir2 );
    //fout.add_mul( dR , -12*E12 * ir2 );
    //fout.add_mul( dR , 6*E6 * ir2 );
    //printf(" (%g,%g,%g)  (%g,%g)  %g \n", dR.x,dR.y,dR.z, c6, c12,  E12 - E6);
    //printf(" (%g,%g,%g)  %f %f  (%g,%g,%g) \n", dR.x,dR.y,dR.z, c6, c12,  fout.x,fout.y,fout.z);
    return E12 - E6;
}

// Lennard-Jones force between two atoms a,b separated by vector dR = Ra - Rb
inline double addAtomVdW_noDamp( const Vec3d& dR, Vec3d& fout, double c6 ){
    double invR2 = 1/dR.norm2();
    double invR4 = invR2*invR2;
    double E     = -c6 * invR4*invR2;
    fout.add_mul( dR , E*6*invR2 );
    return E;
}

// Lennard-Jones force between two atoms a,b separated by vector dR = Ra - Rb
inline double addAtomVdW_dampConst( const Vec3d& dR, Vec3d& fout, double c6, double ADamp ){
    double r2 = dR.norm2(); r2*=r2; r2*=r2;
    fout.add_mul  ( dR , -6*c6 /( r2 + ADamp*c6 ) );
    return 0;
}

template<double Rfunc(double r2,double &df)>
double addAtomVdW_addDamp( const Vec3d& dR, Vec3d& fout, double R, double E0, double ADamp ){
    double D,dD;
    double r2    = dR.norm2();
    double invR2 = 1./(R*R);
    double u2    = r2*invR2;
    double u4    = u2*u2;
    D  = Rfunc(u2,dD);
    double e  = 1./( u4*u2 + D*ADamp);
    double E  = -2*E0*e;
    double fr = E*e*( 6*u4 + dD*ADamp)*invR2 ;
    fout.add_mul(dR,fr);
    return E;
}

inline double R2_func(double r2,double &df){
    //printf( "R2_func() \n");
    if(r2>1){
        df     =  0;
        return    0;
    }else{
        df     =  -2;
        return    1 - r2;
    }
}

inline double R4_func(double r2,double &df){
    //printf( "R4_func() \n");
    if(r2>1){
        df     =  0;
        return    0;
    }else{
        double e = 1 - r2;
        df     =  -4*e;
        return     e*e;
    }
}

inline double invR4_func(double r2,double &df){
    //printf( "invR4_func() \n");
    double invR2 = 1/(r2 + R2SAFE);
    double invR4 = invR2*invR2;
    df           = -4*invR4*invR2;
    return            invR4;
}

inline double invR8_func(double r2,double &df){
    //printf( "invR8_func() \n");
    double invR2 = 1/(r2 + R2SAFE);
    double invR8 = invR2*invR2; invR8*=invR8;
    df           = -8*invR8*invR2;
    return            invR8;
}

// DFT-D3 force between two atoms
inline double addAtomDFTD3(const Vec3d& dR, Vec3d& fout, double c6, double c8, double R0_6, double R0_8) {

    double r2 = dR.norm2();
    if (r2 > R2_D3_CUTOFF) return 0;
    double r4 = r2 * r2;
    double r6 = r4 * r2;
    double r8 = r6 * r2;

    double d6 = 1.0f / (r6 + R0_6);
    double d8 = 1.0f / (r8 + R0_8);
    double E6 = -c6 * d6;
    double E8 = -c8 * d8;
    double F6 = E6 * d6 * 6 * r4;
    double F8 = E8 * d8 * 8 * r6;

    double E = E6 + E8;
    fout.add_mul(dR, (F6 + F8));

    return E;
}

// Morse force between two atoms a,b separated by vector dR = Ra - Rb
inline double addAtomMorse( const Vec3d& dR, Vec3d& fout, double r0, double eps, double alpha ){
    double r     = sqrt( dR.norm2() + R2SAFE );
    double expar = exp( alpha*(r-r0));
    double E     = eps*( expar*expar - 2*expar );
    double fr    = eps*2*alpha*( expar - expar*expar );
    fout.add_mul( dR, fr/r );
    return E;
}

// coulomb force between two atoms a,b separated by vector dR = R1 - R2, with constant kqq should be set to kqq = - k_coulomb * Qa * Qb
inline double addAtomCoulomb( const Vec3d& dR, Vec3d& fout, double kqq ){
    double ir2   = 1.0/( dR.norm2() + R2SAFE );
    double ir    = sqrt(ir2);
    double E     = ir * kqq;
    fout.add_mul( dR , E * ir2 );
    //printf("(%g,%g,%g) %g %g (%g,%g,%g)", dR.x,dR.y,dR.z, kqq, ir, fout.x,fout.y,fout.z );
    return E;
}

// ================= END: From ProbeParticle.cpp



inline void addAtomicForceLJQ( const Vec3d& dp, Vec3d& f, double r0, double eps, double qq ){
    double ir2  = 1/( dp.norm2() + R2SAFE );
    double ir   = sqrt(ir2);
    double ir2_ = ir2*r0*r0;
    double ir6  = ir2_*ir2_*ir2_;
    double fr   = ( ( ir6 - 1 )*ir6*12*eps + ir*qq*COULOMB_CONST )*ir2;
    f.add_mul( dp, fr );
}

inline void addAtomicForceMorse( const Vec3d& dp, Vec3d& f, double r0, double eps, double beta ){
    const double R2ELEC = 1.0;
    double r     = sqrt( dp.norm2()+R2SAFE );
    double expar = exp ( beta*(r-r0) );
    double fr    = eps*2*beta*( expar - expar*expar );
    f.add_mul( dp, fr/r );
}

inline void addAtomicForceMorseQ( const Vec3d& dp, Vec3d& f, double r0, double eps, double qq, double alpha ){
    const double R2ELEC = 1.0;
    double r     = sqrt( dp.norm2()+R2SAFE );
    double expar = exp( alpha*(r-r0));
    double fr    = eps*2*alpha*( expar*expar - expar ) + COULOMB_CONST*qq/( r*r + R2ELEC );
    f.add_mul( dp, fr/r );
}

inline void addAtomicForceQ( const Vec3d& dp, Vec3d& f, double qq ){
    double ir2  = 1/( dp.norm2() + R2SAFE );
    double ir   = sqrt(ir2);
    double fr   = ( ir*qq*-COULOMB_CONST )*ir2;
    f.add_mul( dp, fr );
}

inline void addAtomicForceLJ( const Vec3d& dp, Vec3d& f, double r0, double eps ){;
    double ir2  = 1/( dp.norm2() + R2SAFE );
    double ir2_ = ir2*r0*r0;
    double ir6  = ir2_*ir2_*ir2_;
    double fr   = ( ( 1 - ir6 )*ir6*12*eps )*ir2;
    f.add_mul( dp, fr );
}

inline void addAtomicForceExp( const Vec3d& dp, Vec3d& f, double r0, double eps, double alpha ){
    double r    = sqrt(dp.norm2() + R2SAFE );
    double E    = eps*exp( alpha*(r-r0) );
    double fr   = alpha*E/r;
    f.add_mul( dp, fr );
}

inline Vec3d REQ2PLQ( const Vec3d& REQ, double alpha ){
    double eps   = REQ.y;
    double expar = exp(-alpha*REQ.x);
    double CP    =    eps*expar*expar;
    double CL    = -2*eps*expar;
    return  Vec3d { CP, CL, REQ.z };
}

inline Vec3d REnergyQ2PLQ( const Vec3d& REQ, double alpha ){
    return REQ2PLQ( {REQ.x, sqrt(REQ.y), REQ.z}, alpha );
}

inline Vec3d getForceSpringPlane( const Vec3d& p, const Vec3d& normal, double c0, double k ){
    double cdot = normal.dot(p) - c0;
    return normal * (cdot * k);
}

inline Vec3d getForceHamakerPlane( const Vec3d& p, const Vec3d& normal, double c0, double e0, double r0 ){
    // https://en.wikipedia.org/wiki/Lennard-Jones_potential
    double cdot = normal.dot(p) - c0;
    double ir   = r0/cdot;
    double ir3  = ir*ir*ir;
    double f    = e0*(ir/r0)*ir3*(ir3-1);
    return normal * f;
}

inline Vec3d getForceSpringRay( const Vec3d& p, const Vec3d& hray, const Vec3d& ray0, double k ){
    Vec3d dp; dp.set_sub( p, ray0 );
    double cdot = hray.dot(dp);
    dp.add_mul(hray,-cdot);
    return dp*k;
}

#endif
