#pragma once

#include "Vec2.h"
#include "Vec3.h"
#include "Mat3.h"
//#include "SMat3.h"
#define SQRT3              1.7320508
#define COULOMB_CONST      14.3996448915     // [eV A]
#define const_Boltzman     8.617333262145e-5 // [eV/K]


/**
 * @brief Computes multipole interaction energy between a point and a charge distribution
 * @param d Vector between interaction points
 * @param order Maximum order of multipole expansion (0=monopole, 1=dipole, 2=quadrupole)
 * @param cs Array of multipole coefficients
 * @return Total interaction energy including all multipole terms up to specified order
 */
double Emultipole( const Vec3d& d, int order, const double * cs ){
    //double r   = dR.norm();
    //double ir  = 1 / r;
    //double ir2 = ir*ir;
    double ir2 = 1/d.norm2();
    double E   = cs[0];
    if( order>0 ){ E += ir2    *( cs[1]*d.x + cs[2]*d.y + cs[3]*d.z ); }
    if( order>1 ){ E += ir2*ir2*((cs[4]*d.x + cs[9]*d.y)*d.x +
                                 (cs[5]*d.y + cs[7]*d.z)*d.y +
                                 (cs[6]*d.z + cs[8]*d.x)*d.z );
                                //printf( "Emultipole() Q %g Pxyz(%g,%g,%g) Qxx,yy,zz(%g,%g,%g) Qxy,xz,yz(%g,%g,%g)  \n", cs[0], cs[1],cs[2],cs[3],    cs[4],cs[9],cs[5],cs[7],cs[6],cs[8] );
    }
    return sqrt(ir2)*E;
}


/**
 * @brief Computes combined electrostatic energy including mirror effect and multipole interaction
 * @param pTip Tip position
 * @param pSite Sample position
 * @param VBias Bias voltage
 * @param Rtip Tip radius
 * @param zV Mirror plane and linear potential plane positions
 * @param order Multipole order (0=monopole, 1=dipole, 2=quadrupole)
 * @param cs Multipole coefficients array
 * @param E0 Optional base energy (default=0)
 * @return Combined electrostatic energy
 */
double evalMultipoleMirror( Vec3d pTip, const Vec3d& pSite, double VBias, double Rtip, Vec2d zV, int order, const double* cs, double E0 = 0, const Mat3d* rotSite = nullptr, bool bMirror = true, bool bRamp = true ) {
    //bMirror = true;
    //bRamp   = true;
    //bMirror = false;
    //bRamp   = false;
    // zV.x = mirror plane, zV.y = offset for linear potential
    double zV0 = zV.x;
    double zVd = zV.y;
    double orig_z = pTip.z;                // original tip z
    double zV1 = orig_z + zVd;             // dynamic plane
    // mirror position
    Vec3d pTipMirror = pTip;
    pTipMirror.z     = 2*zV0 - orig_z;
    // displacement to site
    pTip      .sub(pSite);
    pTipMirror.sub(pSite);
    if(rotSite) { 
        pTip       = rotSite->dot(pTip); 
        pTipMirror = rotSite->dot(pTipMirror); 
    }
    double        E  = Emultipole(pTip,       order, cs);
    if(bMirror) { E -= Emultipole(pTipMirror, order, cs); }
    double VR = VBias * Rtip;
    E*=VR;
    //E = 0.0;
    // linear potential from flat capacitor between zV0 and zV1
    if(bRamp) {
        double ramp = (pSite.z - zV0) / (zV1 - zV0);
        //ramp = _clamp(0.0,1.0,ramp); 
        //if(ramp>1.0)   ramp=0.0;
        if(ramp>1.0)    ramp=1.0;
        if(pSite.z<zV0) ramp=0.0;
        double V_lin = VBias * ramp;
        double E_lin = (cs[0] * V_lin);
        E+=E_lin;
    }
    return E + E0;
}

/**
 * @brief Computes combined electrostatic energy for array of tips and sites
 * @param nTip Number of tips
 * @param pTips Array of tip positions
 * @param pSite Array of sample positions
 * @param E0 Optional base energy (default=0)
 * @param VBias Bias voltage
 * @param Rtip Tip radius
 * @param zV Mirror plane and linear potential plane positions
 * @param order Multipole order (0=monopole, 1=dipole, 2=quadrupole)
 * @param cs Multipole coefficients array
 * @param Eout Pre-allocated output array (nTips x nSites)
 */
void evalSitesTipsMultipoleMirror( int nTip, const Vec3d* pTips, const double* VBias, int nSites, const Vec3d* pSite, const Mat3d* rotSite, double E0, double Rtip, Vec2d zV, int order, const double* cs, double* outEs, bool bMirror = true, bool bRamp = true, bool bSiteScan=false ) {
    //E0 = 0;
    //printf("evalSitesTipsMultipoleMirror() nTip: %7d nSites: %2d E0: %7.3f Rtip: %7.3f VBias[0,-1](%7.3f,%7.3f) zV0: %7.3f pTip.z[0,-1](%7.3f,%7.3f) bMirror: %d bRamp: %d order: %d cs:[ %6.3e, %6.3e, %6.3e, %6.3e, %6.3e, %6.3e, %6.3e, %6.3e, %6.3e, %6.3e ]\n", nTip, nSites, E0, Rtip, VBias[0], VBias[nTip-1], zV.x, pTips[0].z, pTips[nTip-1].z, bMirror, bRamp, order, cs[0], cs[1], cs[2], cs[3], cs[4], cs[5], cs[6], cs[7], cs[8], cs[9] );
    for (int i=0; i<nTip; i++) {
        for (int j=0; j<nSites;j++) {
            const Mat3d* rot = ( rotSite ) ? ( rotSite + j ) : nullptr;
            double E = evalMultipoleMirror(pTips[i], pSite[j], VBias[i], Rtip, zV, order, cs, E0, rot, bMirror, bRamp);
            if(bSiteScan) { outEs[i*nSites+j] = E; }
            else          { outEs[j*nTip  +i] = E; }
            //printf("evalSitesTipsMultipoleMirror() i: %d j: %d outEs: %6.3e VBias: %6.3e pTip( %6.3e %6.3e %6.3e) pSite(%6.3e %6.3e %6.3e) \n", i, j, outEs[i*nSites+j], VBias[i], pTips[i].x, pTips[i].y, pTips[i].z, pSite[j].x, pSite[j].y, pSite[j].z);
        }
    }
    //printf("evalSitesTipsMultipoleMirror() done VR: %6.3e E0: %6.3e VBias: %6.3e Rtip: %6.3e zV0: %6.3e order: %d cs: %6.3e %6.3e %6.3e %6.3e %6.3e \n", VR, E0, VBias, Rtip, zV.x, order, cs[0], cs[1], cs[2], cs[3], cs[4] );
}

/**
 * @brief Computes tunneling rates between tips and sites
 * @param nTips Number of tips
 * @param pTips Array of tip positions
 * @param nSites Number of sites
 * @param pSites Array of site positions
 * @param beta Tunneling decay constant
 * @param Amp Amplitude scaling factor (default=1.0)
 * @param outTs Pre-allocated output array (nTips x nSites)
 */
void evalSitesTipsTunneling( int nTips, const Vec3d* pTips, int nSites, const Vec3d* pSites, double beta, double Amp, double* outTs ){
    for (int i = 0; i < nTips; i++) {
        for (size_t j = 0; j < nSites; j++) {
            Vec3d d = pTips[i] - pSites[j];
            double r = d.norm();
            outTs[i*nSites + j] = Amp * exp(-beta * r);
        }
    }
}
