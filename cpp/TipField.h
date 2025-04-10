#pragma once

#include "Vec3.h"
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
    if( order>0 ) E += ir2    *( cs[1]*d.x + cs[2]*d.y + cs[3]*d.z );
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
 * @param zV0 Mirror plane position
 * @param order Multipole order (0=monopole, 1=dipole, 2=quadrupole)
 * @param cs Multipole coefficients array
 * @param E0 Optional base energy (default=0)
 * @return Combined electrostatic energy
 */
double computeCombinedEnergy(const Vec3d& pTip, const Vec3d& pSite, double VBias, double Rtip, double zV0, int order, const double* cs, double E0 = 0) {
    // Mirror calculation
    Vec3d pTipMirror = pTip;
    pTipMirror.z     = 2*zV0 - pTip.z;
    
    // Distance vectors
    Vec3d dDirect = pTip - pSite;
    Vec3d dMirror = pTipMirror - pSite;
    
    // Compute direct and mirror terms using existing Emultipole function
    double E_direct  = Emultipole(dDirect, order, cs);
    double E_mirror  = Emultipole(dMirror, order, cs);
    
    // Scale by bias voltage and tip radius
    double VR = VBias * Rtip;
    return VR * (E_direct - E_mirror) + E0;
}

/**
 * @brief Computes combined electrostatic energy for array of tips and sites
 * @param nTip Number of tips
 * @param pTips Array of tip positions
 * @param pSite Array of sample positions
 * @param E0 Optional base energy (default=0)
 * @param VBias Bias voltage
 * @param Rtip Tip radius
 * @param zV0 Mirror plane position
 * @param order Multipole order (0=monopole, 1=dipole, 2=quadrupole)
 * @param cs Multipole coefficients array
 * @param Eout Pre-allocated output array (nTips x nSites)
 */
void computeCombinedEnergies( int nTip, const Vec3d* pTips, const Vec3d pSite, double E0, double VBias, double Rtip, double zV0, int order, const double* cs, double* Eout ) {
    double VR        = VBias * Rtip;
    for (int i = 0; i < nTip; i++) {
        Vec3d pTip       = pTips[i];
        Vec3d pTipMirror = pTip;
        pTipMirror.z     = 2*zV0 - pTip.z; // mirror tip position
        double E_direct  = Emultipole( pTip       - pSite , order, cs);
        double E_mirror  = Emultipole( pTipMirror - pSite , order, cs);            
        Eout[i] = VR * (E_direct - E_mirror) + E0;
    }
}
