/*
 * @file ChargeRings.cpp
 * @brief Implementation of molecular charging and STM imaging with electrostatic interactions
 *
 * This module simulates scanning tunneling microscopy (STM) measurements of molecular systems
 * where charging effects and electrostatic interactions play a significant role. The implementation
 * considers a system of molecular sites that can be charged/discharged based on their energy levels,
 * mutual Coulomb interactions, and interaction with an STM tip.
 *
 * The total energy of the system is given by:
 * U_total = sum_i{ E_i Q_i/2 + Q_i Q_tip/(4π ε0 r_i,tip) - μ Q_i } + sum_i,j>i{ Q_i Q_j/(4π ε0 r_ij) }
 *
 * Key Features:
 * - Handles multiple molecular sites with variable charge states
 * - Includes multipole interactions (monopole, dipole, quadrupole)
 * - Supports both gradient descent and Boltzmann statistics for charge optimization
 * - Computes STM signals with temperature effects
 * - Includes site-site and tip-site Coulomb interactions
 *
 * Physical Parameters:
 * - Site energies (E_i): Energy levels of molecular orbitals
 * - Chemical potential (μ): Fermi level controlling charge transfer
 * - Temperature (T): Controls thermal occupation of states
 * - Coulomb coupling: Strength of electrostatic interactions
 * - Tip parameters: Position, charge, and decay constants
 */


#include <stdio.h>

#include "Vec3.h"
#include "Mat3.h"
//#include "SMat3.h"
//#include "CG.h"
#include "LinSolveGauss.cpp"

#define R_SAFE   1e-4

#define SQRT3              1.7320508
#define COULOMB_CONST      14.3996448915     // [eV A]
#define const_Boltzman     8.617333262145e-5 // [eV/K]


/**
 * @struct RingParams
 * @brief Container for system parameters of molecular ring simulation
 *
 * Holds all necessary parameters for simulating molecular charging and STM imaging,
 * including geometry, energy levels, and interaction parameters.
 */
struct RingParams {
    int     nsite=0;       // number of molecular sites (quantum dots)
    Vec3d*  spos=0;        // [nsite,3]    positions of molecular sites 
    Mat3d*  rots=0;        // [nsite][3x3] rotation matrices for multipole orientation 
    double* MultiPoles=0;  // [nsite][10]  multipole moments for each site  (1 + 3 + 6 components - monopole, dipole, quadrupole)
    double* Esites=0;      // [nsite]      original energy levels of molecular orbitals 
    double  E_Fermi=0;     // Fermi level of the substrate
    double  cCoupling=0;   // strength of Coulomb interaction between sites
    double  onSiteCoulomb = 3.0; 
    double  temperature=100.0; // temperature
    //double*  Q_tips;   // [ntips] charge of the STM tip

    // New parameters for predefined configurations
    int     nConf=0;
    double* siteConfs=0;  // Array of size [nConf * nsite]
    //bool    usePredefConfs;

    // RingParams(){
    //     nsite      = 0;
    //     spos       = 0;
    //     rots       = 0;
    //     MultiPoles = 0;
    //     Esites     = 0;
    //     E_Fermi    = 0.0;
    //     cCoupling  = 1.0;
    //     temperature= 100.0;
    //     // Initialize new parameters
    //     nConf      = 0;
    //     siteConfs  = 0;
    //     usePredefConfs = false;
    // }

    void print(){
        printf( "RingParams::print() \n" );
        printf( "nsite      %i \n", nsite      );
        printf( "E_Fermi    %g \n", E_Fermi    );
        printf( "cCoupling  %g \n", cCoupling  );
        printf( "onSiteCoulomb  %g \n", onSiteCoulomb  );
        printf( "temperature %g \n", temperature );
        //if(usePredefConfs){printf( "nConf      %i \n", nConf      );}
    }
};

RingParams params;

int verbosity=0; 

// inline double clamp(double x, double xmin, double xmax){
//     if     ( x<xmin ){ return xmin; }
//     else if( x>xmax ){ return xmax; }
//     return x;
// }

/**
 * @brief Prints a matrix to the console
 * @param ni Number of rows
 * @param nj Number of columns
 * @param A Matrix to print
 * @param format Format string for printing elements
 */
void printmatrix( int ni, int nj, double* A, const char* format="%g " ){

    for(int i=0; i<ni; i++){
        for(int j=0; j<nj; j++){
            printf( format, A[i*nj+j]);
        }
        printf("\n");
    }
}

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
    if( order>1 ) E += ir2*ir2*((cs[4]*d.x + cs[9]*d.y)*d.x +
                                (cs[5]*d.y + cs[7]*d.z)*d.y +
                                (cs[6]*d.z + cs[8]*d.x)*d.z );
    return sqrt(ir2)*E;
}


/**
 * @brief Calculates charging forces on molecular sites
 * @param nsite Number of molecular sites
 * @param Esite Array of site energies
 * @param Coupling Matrix of Coulomb coupling between sites
 * @param E_Fermi Fermi energy (chemical potential)
 * @param Qs Current charge state of sites
 * @param dEdQ Output array for charging forces
 * @return Total electrostatic energy of the system
 */
double getChargingForce( int nsite, const double* Esite, const double* Coupling, double E_Fermi, const double* Qs, double* dEdQ ){
    double f2err=0;
    double E_on  = 0;
    double E_off = 0;
    //printf( "======== getChargingForce() \n" );
    //printf( "getChargingForce() @Esite=%li @Coupling=%li \n", (long)Esite, (long)Coupling  );
    double E=0;
    for(int i=0; i<nsite; i++){
        //double qi = Qs[i];
        //E_on     += qi*fQi;
        double Vcoul = 0;
        if(Coupling){
            //printf( "site[%i] ----- \n", i );
            for(int j=0; j<nsite; j++){
                if(j==i){ continue; }
                double cij = Coupling[i*nsite+j];
                double  qj = Qs[j];
                //fQi += qj*cij;
                Vcoul  += qj*cij;
                //E_off += qi*qj*cij;
                //printf( "site[%i,%i] %g \n", i, j, cij );
            }
        }
        double eps_i = Esite[i];  //   on-site energy (with respct to fermi level)
        double fQi = eps_i + Vcoul - E_Fermi;   //charinging force, if below fermi level it is getting charged, if above it is getting discharged
        { // onsite Coulomb
            double qi  = Qs[i];
            if(qi>1){ fQi += (qi-1)*params.onSiteCoulomb;  }
        }
        //printf( "site[%i] fQi(%g) = eps(%g) + Vcoul(%g) - Ef(%g) \n", i, eps_i, Vcoul, fQi, E_Fermi  );
        //printf( "site[%i] eps %g  Vcoul %g  fQi %g E_Fermi %g \n", i, eps_i, Vcoul, fQi, E_Fermi );
        if(dEdQ){ dEdQ[i] = fQi; }   // NOTE: this total on-site energy (diagonal of the hamiltonian) it is not just energy-shift due to Coulomb interaction
        E += fQi*Qs[i]; 
        f2err += fQi*fQi;
    }
    //printf("E_on %g E_off %g f2err %g\n", E_on, E_off, f2err );
    return E;
}


/**
 * @brief Constructs coupling matrix including site energies and interactions
 * @param nsite Number of molecular sites
 * @param spos Array of site positions
 * @param rot Array of rotation matrices for multipole orientations
 * @param MultiPoles Array of multipole moments for each site
 * @param Esite0 Array of bare site energies
 * @param pT Tip position
 * @param Qt Tip charge
 * @param Esite Output array for site energies including interactions
 * @param Coupling Output matrix for inter-site couplings
 * @param cCoupling Coupling strength parameter
 */
void makeCouplingMatrix( int nsite, const Vec3d* spos, const Mat3d* rot, const double* MultiPoles, const double* Esite0, Vec3d pT, double Qt, double* Esite, double* Coupling, double cCoupling ){
    //printf( "makeCouplingMatrix() cCoupling=%g Qt=%g  @Esite=%li @Coupling=%li  \n", cCoupling, Qt, (long)Esite, (long)Coupling  );
    //printf("========\n");
    for(int i=0; i<nsite; i++){
        const Vec3d pi = spos[i];
        const Vec3d d  = pi - pT;
        double Vitip=0;   

        // Quadrupole
        if(MultiPoles){
            //Vec3d d_ = rot[i].dot(d);
            Vec3d d_ = rot[i].dotT(d);
            //printf("rot[%i]\n", i ); printmatrix(3,3, (double*)&rot[i] );
            Vitip = Emultipole( d_, 3, MultiPoles ) * COULOMB_CONST * Qt;
        }else{
            Vitip = COULOMB_CONST * Qt /d.norm();
        }

        double eps_i = Vitip + Esite0[i];

        //printf( "makeCouplingMatrix().site[%i] eps_i=%10.5e Esite0[i]=%5.2f Vitip=%10.5e  Qt=%5.2f |d|=%8.4f d(%6.3f,%6.3f,%6.3f) \n", i, eps_i, Esite0[i], Vitip,  Qt,  d.norm(), d.x,d.y,d.z );
        Esite[i] = eps_i;

        if(Coupling){
            Coupling[i*nsite+i] = eps_i; // - E_mu;
            //printf( "makeCouplingMatrix[%i] V= %g   r= %g      pi(%g,%g,%g)   pT(%g,%g,%g) \n", i, Vitip, ( pi - pT ).norm(),  pi.x,pi.y,pi.z,   pT.x,pT.y,pT.z   );
            for(int j=i+1; j<nsite; j++){
                //if( i==j ){ continue; }
                Vec3d d  = pi - spos[j];
                double Cij = COULOMB_CONST*cCoupling/( pi - spos[j] ).norm();
                //Cij = 0;
                Coupling[i*nsite+j] = Cij;
                Coupling[j*nsite+i] = Cij;
            }
        }
    }
}


/**
 * @brief Updates charges using gradient descent
 * @param n Number of sites
 * @param x Array of charges (modified in-place)
 * @param f Array of forces
 * @param dt Time step for integration
 * @return Sum of squared forces (convergence measure)
 */
double moveGD(  int n, double *x, double *f, double dt ){
    double  f2=0;
    for(int i=0; i<n; i++){
        double fi = -f[i];    // negative gradient of the energy
        //double xi = x[i];
        //xi += fi*dt;
        //clamp( x, 0.0, 1.0 );
        // x[i] = xi;
        x[i] = clamp( x[i] + fi*dt, 0.0, 1.0 );

        f2 += fi*fi;
    }
    return f2;
}


/**
 * @brief Updates charges using molecular dynamics with damping
 * @param n Number of sites
 * @param x Array of charges (modified in-place)
 * @param f Array of forces
 * @param v Array of velocities
 * @param dt Time step
 * @param damping Damping coefficient
 * @return Sum of squared forces
 */
double moveMD(  int n, double *x, double *f, double *v, double dt, double damping=0.1 ){
    double  f2=0;
    double cdamp=1-damping;
    double vf=0;
    for(int i=0; i<n; i++){  vf += v[i] * f[i]; }
    if( vf<0 ){ cdamp=1.0; }
    for(int i=0; i<n; i++){
        double fi = -f[i];    // negative gradient of the energy
        
        //double xi = x[i];
        //xi += fi*dt;
        //clamp( x, 0.0, 1.0 );
        // x[i] = xi;

        double vi = v[i];
        vi = vi*cdamp + fi*dt;

        double xi = x[i] + vi*dt;
        //if( !clamp2( xi, 0.0, 1.0 ) ){
        if( !clamp2( xi, 0.0, 2.0 ) ){
            f2 += fi*fi;
        };
        
        x[i] = xi;
        
    }
    return f2;
}




/**
 * @brief Optimizes site charges using gradient descent
 * @param nsite Number of sites
 * @param spos Site positions
 * @param rot Rotation matrices
 * @param MultiPoles Multipole moments
 * @param Esite0 Bare site energies
 * @param Qs Initial/final charges
 * @param p_tip Tip position
 * @param Q_tip Tip charge
 * @param E_Fermi Fermi energy
 * @param cCoupling Coupling strength
 * @param niter Maximum iterations
 * @param tol Convergence tolerance
 * @param dt Time step
 * @return Number of iterations performed
 */
int optimizeSiteOccupancy( int nsite, const Vec3d* spos, const Mat3d* rot, const double* MultiPoles, const double* Esite0, double* Qs, Vec3d p_tip, double Q_tip, double E_Fermi, double cCoupling, int niter=1000, double tol=1e-6, double dt=0.1 ){
    //  Site occupancy is 1 if below the Fermi level, 0 otherwise
    //  Fermi level is set by the voltage applied to the STM tip and distance from the site
    // The energy of sites is modified by Coulomb interaction between the sites as well
    //   E_i = (Qi*Qj/|r_i-r_j|) + (Qi*Qtip/|r_i-r_tip|) + E0s[i] + Emu
    //double varOnsite[nsite];
    //double varCoupling[];
    if(verbosity>0) printf( "E_Fermi %g Q_tip %g Esite{%6.3f,%6.3f,%6.3f}  \n", E_Fermi, Q_tip,  Esite0[0],Esite0[1],Esite0[2] );
    double fQs      [ nsite ];
    double vQs      [ nsite ];
    double Esite    [ nsite ];
    double Coupling_[ nsite*nsite ];

    //bool   bCoupling=true;
    double* Coupling=Coupling_;
    if( cCoupling<0 ){ 
        Coupling  = 0;
        //bCoupling = false; 
    }
    makeCouplingMatrix( nsite, spos, rot, MultiPoles, Esite0, p_tip, Q_tip, Esite, Coupling, cCoupling );
    //if(verbosity>1)printmatrix( nsite, nsite, Coupling, "%12.6f" );

    for(int i=0; i<nsite; i++){ vQs[i]=0; } 
    //for(int i=0; i<nsite; i++){ Qs[i] = Coupling[i*nsite+i]; }  // Debug: copy Energy to Qs
    //for(int i=0; i<nsite; i++){ Qs[i] = Esite[i]; }
    //return 0;

    double tol2 = tol*tol;
    int itr=0;
    for(itr=0; itr<niter; itr++){
        //break;
        double E = getChargingForce( nsite, Esite, Coupling, E_Fermi, Qs, fQs );
        //printf("itr=%i Qs{%6.3f,%6.3f,%6.3f} fQs(=eps-Ef){%6.3f,%6.3f,%6.3f} Eps{%6.3f,%6.3f,%6.3f} E_Fermi=%6.3f Q_tip=%6.3f\n", itr, Qs[0],Qs[1],Qs[2],    fQs[0],fQs[1],fQs[2],    Esite[0],Esite[1],Esite[2], E_Fermi, Q_tip );
        //if(verbosity>1) 
        //printf("itr=%i Qs{%6.3f,%6.3f,%6.3f} fQs(=eps-Ef){%6.3f,%6.3f,%6.3f}  E_Fermi=%6.3f Q_tip=%6.3f\n", itr, Qs[0],Qs[1],Qs[2],    fQs[0],fQs[1],fQs[2], E_Fermi, Q_tip );
        //printf("itr=%d, f2err=%g\n", itr, f2err);
        //double f2 = moveGD( nsite, Qs, fQs, dt );
        double f2 = moveMD( nsite, Qs, fQs, vQs, dt, 0.0 );
        //printf("itr=%i f2=%g\n", itr, f2);
        if(f2<tol2){ break; }
    }

    //for(int i=0; i<nsite; i++){ Qs[i] = fQs[i]; } 

    return itr;
}





/**
 * @brief Computes site charges using Boltzmann statistics
 * @param nsite Number of sites
 * @param spos Site positions
 * @param rot Rotation matrices
 * @param MultiPoles Multipole moments
 * @param Esites0 Bare site energies
 * @param Qout Output array for charges
 * @param p_tip Tip position
 * @param Q_tip Tip charge
 * @param E_Fermi Fermi energy
 * @param cCoupling Coupling strength
 * @param T Temperature
 * @param Econf Optional array to store energies of all configurations [size: 2^(2*nsite)]
 * @return Total system energy
 */
double boltzmanSiteOccupancy( int nsite, const Vec3d* spos, const Mat3d* rot, const double* MultiPoles, const double* Esites0, double* Qout, Vec3d p_tip, double Q_tip, double E_Fermi, double cCoupling, double T, double* Econf=0 ){
    if(verbosity>0) printf( "boltzmanSiteOccupancy() E_Fermi %g Q_tip %g Esite{%6.3f,%6.3f,%6.3f}  \n", E_Fermi, Q_tip,  Esites0[0],Esites0[1],Esites0[2] );
    double Qs       [ nsite ];
    double Qav      [ nsite ];
    double Esite    [ nsite ];
    double Coupling_[ nsite*nsite ];
    double* Coupling=Coupling_;
    if( cCoupling<0 ){ Coupling  = 0; }
    makeCouplingMatrix( nsite, spos, rot, MultiPoles, Esites0, p_tip, Q_tip, Esite, Coupling, cCoupling );

    double beta = 1.0/( const_Boltzman * T );

    int nspinorb = nsite*2;
    int nconfs = 1<<nspinorb;
    
    // Arrays to store log probabilities and maximum log probability
    double logPs[nconfs];
    double maxLogP = -1e300;  // Initialize to very negative number
    
    // First pass: compute all log probabilities and find maximum
    for( int ic=0; ic<nconfs; ic++ ){
        for( int j=0; j<nsite; j++){ 
            int jj=j*2; 
            int qi=0;
            if( ic&(1<<(jj  )) ){ qi++; }
            if( ic&(1<<(jj+1)) ){ qi++; }        
            Qs[j] = qi;
        };
        double E = getChargingForce( nsite, Esite, Coupling, E_Fermi, Qs, 0 );
        if(Econf) Econf[ic] = E;  // Store raw energy if array provided
        logPs[ic] = -beta*E;
        maxLogP = fmax(maxLogP, logPs[ic]);
    }
    
    // Second pass: compute normalized probabilities using the log-sum-exp trick
    double sumP = 0;
    for( int i=0; i<nsite; i++ ){ Qav[i]=0; }
    
    for( int ic=0; ic<nconfs; ic++ ){
        for( int j=0; j<nsite; j++){ 
            int jj=j*2; 
            int qi=0;
            if( ic&(1<<(jj  )) ){ qi++; }
            if( ic&(1<<(jj+1)) ){ qi++; }        
            Qs[j] = qi;
        };
        double P = exp(logPs[ic] - maxLogP);  // Subtract maxLogP to avoid overflow
        sumP += P;
        for( int j=0; j<nsite; j++){ Qav[j] += Qs[j]*P; };
    }

    double renorm = 1.0/sumP;
    for( int j=0; j<nsite; j++){ Qout[j] = Qav[j] * renorm; };
    
    return sumP * exp(maxLogP);  // Return the true sum by adding back the maxLogP factor
}

/**
 * @brief Computes site charges using Boltzmann statistics with predefined configurations
 * @param nsite Number of sites
 * @param spos Site positions
 * @param rot Rotation matrices
 * @param MultiPoles Multipole moments
 * @param Esites0 Bare site energies
 * @param Qout Output array for charges
 * @param p_tip Tip position
 * @param Q_tip Tip charge
 * @param E_Fermi Fermi energy
 * @param cCoupling Coupling strength
 * @param T Temperature
 * @param nConf Number of predefined configurations
 * @param siteConfs Array of predefined configurations [size: nConf * nsite]
 * @param Econf Optional array to store energies of configurations [size: nConf]
 * @return Total system energy
 */
double boltzmanSiteOccupancy_new(int nsite, const Vec3d* spos, const Mat3d* rot, const double* MultiPoles, const double* Esites0, double* Qout, Vec3d p_tip, double Q_tip, double E_Fermi, double cCoupling, double T, int nConf, const double* siteConfs, double* Econf=0){
    //if(verbosity>0) printf("boltzmanSiteOccupancy_new() E_Fermi %g Q_tip %g Esite{%6.3f,%6.3f,%6.3f} nConf %i\n", E_Fermi, Q_tip, Esites0[0], Esites0[1], Esites0[2], nConf);
    double Qs[nsite];
    double Qav[nsite];
    double Esite[nsite];
    double Coupling_[nsite*nsite];
    double* Coupling = Coupling_;
    if(cCoupling < 0){ Coupling = 0; }
    
    makeCouplingMatrix(nsite, spos, rot, MultiPoles, Esites0, p_tip, Q_tip, Esite, Coupling, cCoupling);
    
    double beta = 1.0/(const_Boltzman * T);
    
    // Arrays to store log probabilities and maximum log probability
    double logPs[nConf];
    double maxLogP = -1e300;  // Initialize to very negative number
    
    // First pass: compute all log probabilities and find maximum
    for(int ic=0; ic<nConf; ic++){
        for(int j=0; j<nsite; j++){
            Qs[j] = siteConfs[ic*nsite + j];
        }
        double E = getChargingForce(nsite, Esite, Coupling, E_Fermi, Qs, 0);
        if(verbosity>2){
            printf("boltzmanSiteOccupancy_new()[iConf=%3i/%3i] Qs{%6.3f,%6.3f,%6.3f} E=%6.3f\n", ic, nConf, Qs[0], Qs[1], Qs[2], E);
        }
        if(Econf) Econf[ic] = E;  // Store raw energy if array provided
        logPs[ic] = -beta*E;
        maxLogP = fmax(maxLogP, logPs[ic]);
    }
    
    // Second pass: compute normalized probabilities using the log-sum-exp trick
    double sumP = 0;
    for(int ic=0; ic<nConf; ic++){
        logPs[ic] = exp(logPs[ic] - maxLogP);
        sumP += logPs[ic];
    }
    
    // Third pass: compute average charges
    for(int j=0; j<nsite; j++) Qav[j] = 0;
    for(int ic=0; ic<nConf; ic++){
        double p = logPs[ic]/sumP;
        for(int j=0; j<nsite; j++){
            Qav[j] += p * siteConfs[ic*nsite + j];
        }
    }
    
    // Store average charges in output array
    for(int j=0; j<nsite; j++) Qout[j] = Qav[j];
    
    return getChargingForce(nsite, Esite, Coupling, E_Fermi, Qav, 0);
}

/**
 * @brief Computes STM signal from site occupancies
 * @param pos Measurement position
 * @param beta Decay constant
 * @param nsite Number of sites
 * @param spos Site positions
 * @param Amps Site amplitudes
 * @param bInCoh Consider incoherent tunneling
 * @return STM current at specified position
 */
double getSTM_sites( Vec3d pos, double beta, int nsite, const Vec3d* spos, const double* Amps, const bool bInCoh=false ){
    double I=0;
    for(int i=0; i<nsite; i++){
        Vec3d d  = pos - spos[i];
        double r = d.norm();
        double amp = Amps[i]*exp( -beta*r );
        if( bInCoh ){ amp*=amp; }  // sum of independent channels (incoherent)
        I+=amp;
    }
    if( !(bInCoh) ){ I*=I; } // sum of wavefunctions ( choherent )
    return I;
}


/**
 * @brief Computes STM signal with charging effects
 * @param nsite Number of sites
 * @param spos Site positions
 * @param rot Rotation matrices
 * @param MultiPoles Multipole moments
 * @param Esites0 Bare site energies
 * @param Qs Site charges
 * @param p_tip Tip position
 * @param Q_tip Tip charge
 * @param E_Fermi Fermi energy
 * @param cCoupling Coupling strength
 * @param decay Tunneling decay constant
 * @param T Temperature
 * @param bOccupied Consider only occupied states
 * @return STM current
 */
double getSTM( int nsite, const Vec3d* spos, const Mat3d* rot, const double* MultiPoles, const double* Esites0, const double* Qs, Vec3d p_tip, double Q_tip, double E_Fermi, double cCoupling, double decay, double T, bool bOccupied ){
    double Esite[nsite];
    double Coupling[nsite*nsite];
    double dE[nsite];

    //bool bOccupied = false;
    
    // Get energy shifts using existing coupling matrix and charging force calculation
    makeCouplingMatrix(nsite, spos, rot, MultiPoles, Esites0, p_tip, Q_tip, Esite, Coupling, cCoupling);
    getChargingForce(nsite, Esite, Coupling, E_Fermi, Qs, dE);
    
    //double beta_T = 1.0/( const_Boltzman * T );

    // Calculate tunneling current (assuming in-coherent regime - no interference between sites)
    double I = 0.0;
    for(int i=0; i<nsite; i++) { // sum over sites
        Vec3d d = p_tip - spos[i];
        double r = d.norm();
        //if(r < R_SAFE) continue;
        double T = exp(-decay * r);
        if (bOccupied){
            T *= Qs[i]; 
            //T *= ( 1.0 + exp(dE[i]*beta_T );
        }else{
            T *= (1-Qs[i]);  // Fermi function
        }
        I += T;
    }
    return I;
}

/**
 * @brief Solves eigenvalue problem for a given Hamiltonian matrix
 * @param H Input Hamiltonian matrix
 * @param evals Output eigenvalues
 * @param evecs Output eigenvectors
 */
void solveHamiltonian(const Mat3d& H, Vec3d& evals, Vec3d* evecs) {
    H.eigenvals(evals);
    for (int j = 0; j < 3; j++) {
        H.eigenvec(evals.array[j], *(evecs+j) );
    }
}

/**
 * @brief Computes Green's function for transport calculations
 * @param H Hamiltonian matrix
 * @param mu Chemical potential
 * @param Green Output Green's function matrix
 */
void computeGreensFunction( double* H, double mu, double* Green) {
    double H_minus_mu_I[9];  // (H - mu*I)
    for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
            if (j == k) {
                H_minus_mu_I[j * 3 + k] = H[j * 3 + k] - mu;
            } else {
                H_minus_mu_I[j * 3 + k] = H[j * 3 + k];
            }
        }
    }

    double b[3];  // Right-hand side vector for solving linear equations
    int index[3];  // Index array required by linSolve_gauss

    // Solve (H - mu*I) * G = I for each column of the identity matrix
    for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 3; k++) {
            b[k] = (k == j) ? 1.0 : 0.0;
        }

        double x[3];  // Solution vector for the j-th column
        Lingebra::linSolve_gauss(3, (double**)&H_minus_mu_I, b, index, x);

        // Store the solution in the Green's function matrix
        for (int k = 0; k < 3; k++) {
            Green[j * 3 + k] = x[k];
        }
    }
}


extern "C"{

/**
 * @brief Sets verbosity level for debugging output
 * @param verbosity_ New verbosity level
 */
void setVerbosity(int verbosity_){ verbosity=verbosity_; }

/**
 * @brief Initializes ring parameters for simulation
 * @param nsite Number of sites
 * @param spos Site positions
 * @param rots Rotation matrices
 * @param MultiPoles Multipole moments
 * @param Esites Site energies
 * @param E_Fermi Fermi energy
 * @param cCouling Coupling strength
 * @param onSiteCoulomb On-site Coulomb interaction
 * @param temperature System temperature
 */
void initRingParams(int nsite, double* spos, double* rots, double* MultiPoles, double* Esites, double E_Fermi, double cCouling, double onSiteCoulomb, double temperature ) {
    params.nsite       = nsite;
    params.spos        = (Vec3d*)spos;
    params.rots        = (Mat3d*)rots;
    params.MultiPoles  = MultiPoles;
    params.Esites      = Esites;
    params.E_Fermi     = E_Fermi;
    params.cCoupling   = cCouling;
    params.onSiteCoulomb = onSiteCoulomb;
    params.temperature = temperature;
    //printf( "initRingParams() nsite=%i E_Fermi=%g cCoupling=%g temperature=%g \n", nsite, params.E_Fermi, params.cCoupling, params.temperature );
    printf( "initRingParams()"); params.print();
    //params.Q_tip = Q_tip_;
}

/**
 * @brief Legacy function for solving site occupancies
 * @deprecated Use solveSiteOccupancies instead
 */
void solveSiteOccupancies_old( int npos, double* ptips_, double* Qtips, int nsite, double* spos, const double* rot, const double* MultiPoles, const double* Esite, double* Qout, double E_Fermi, double cCoupling, double temperature ){
    printf( "solveSiteOccupancies_old() npos=%i nsite=%i E_Fermi=%g cCoupling=%g temperature=%g \n", npos, nsite, E_Fermi, cCoupling, temperature );
    Vec3d* ptips = (Vec3d*)ptips_;
    //#pragma omp parallel for
    for(int i=0; i<npos; i++){
        double Qs[nsite];
        for(int j=0; j<nsite; j++){ Qs[j]=0; }    
        boltzmanSiteOccupancy( nsite, (Vec3d*)spos, (Mat3d*)rot, MultiPoles, Esite, Qs, ptips[i], Qtips[i], E_Fermi, cCoupling, temperature );  
        for(int j=0; j<nsite; j++){ Qout[i*nsite+j] = Qs[j]; }
    }
}

/**
 * @brief Solves site occupancies for multiple tip positions
 * @param npos Number of tip positions
 * @param ptips_ Array of tip positions
 * @param Qtips Array of tip charges
 * @param Qout Output array for site charges
 * @param Econf Optional array to store energies of all configurations [size: npos * 2^(2*nsite)]
 * @param bUserBasis Use user-defined basis
 */
void solveSiteOccupancies(int npos, double* ptips_, double* Qtips, double* Qout, double* Econf, bool bUserBasis ) {
    printf( "solveSiteOccupancies() npos=%i nsite=%i E_Fermi=%g cCoupling=%g temperature=%g \n", npos, params.nsite, params.E_Fermi, params.cCoupling, params.temperature );
    //printf( "solveSiteOccupancies() npos=%i \n", npos ); params.print();
    Vec3d* ptips = (Vec3d*)ptips_;
    int nconfs = bUserBasis ? params.nConf : 1<<(params.nsite*2);
    #pragma omp parallel for
    for(int i=0; i<npos; i++) {
        if(verbosity>2){ printf( "solveSiteOccupancies()[iPos=%i] pos(%8.4f,%8.4f,%8.4f) q=%8.4f \n", i,  ptips[i].x,ptips[i].y,ptips[i].z,   Qtips[i] ); }
        double* Econf_i = Econf ? Econf + i*nconfs : 0;  // Pointer to current tip's energy array
        double  Qs[params.nsite];
        if( bUserBasis ){ boltzmanSiteOccupancy_new(params.nsite, params.spos, params.rots, params.MultiPoles,   params.Esites, Qs, ptips[i], Qtips[i], params.E_Fermi, params.cCoupling, params.temperature, nconfs, params.siteConfs, Econf_i); }
        else            { boltzmanSiteOccupancy    (params.nsite, params.spos, params.rots, params.MultiPoles,   params.Esites, Qs, ptips[i], Qtips[i], params.E_Fermi, params.cCoupling, params.temperature, Econf_i );
        }
        for(int j=0; j<params.nsite; j++) {  Qout[i*params.nsite+j] = Qs[j];  }
    }
}

// /**
//  * @brief Solves site occupancies for multiple tip positions with predefined configurations
//  * @param npos Number of tip positions
//  * @param ptips_ Array of tip positions
//  * @param Qtips Array of tip charges
//  * @param Qout Output array for site charges
//  * @param Econf Optional array to store energies of all configurations [size: npos * nConf]
//  */
// void solveSiteOccupancies_new(int npos, double* ptips_, double* Qtips, double* Qout, double* Econf, int nConf, const double* siteConfs) {
//     printf( "solveSiteOccupancies_new() npos=%i nsite=%i E_Fermi=%g cCoupling=%g temperature=%g \n", npos, params.nsite, params.E_Fermi, params.cCoupling, params.temperature );
//     //printf( "solveSiteOccupancies() npos=%i \n", npos ); params.print();
//     Vec3d* ptips = (Vec3d*)ptips_;
//     #pragma omp parallel for
//     for(int i=0; i<npos; i++) {
//         //printf( "solveSiteOccupancies[%i|%i,%i] pos(%8.4f,%8.4f,%8.4f) q=%8.4f \n", i, i/nj, i%nj,  ptips[i].x,ptips[i].y,ptips[i].z,   Qtips[i] );
//         double Qs[params.nsite];
//         double* Econf_i = Econf ? Econf + i*nConf : 0;  // Pointer to current tip's energy array
//         boltzmanSiteOccupancy_new(params.nsite, params.spos, params.rots, params.MultiPoles,   params.Esites, Qs, ptips[i], Qtips[i], params.E_Fermi, params.cCoupling, params.temperature, nConf, siteConfs, Econf_i);
//         for(int j=0; j<params.nsite; j++) {  Qout[i*params.nsite+j] = Qs[j];  }
//     }
// }

/**
 * @brief Generates STM map for multiple tip positions
 * @param npos Number of positions
 * @param ptips_ Tip positions
 * @param Qtips Tip charges
 * @param Qsites Site charges
 * @param Iout Output currents
 * @param decay Tunneling decay constant
 * @param bOccupied Consider only occupied states
 */
void STM_map(int npos, double* ptips_, double* Qtips, double* Qsites, double* Iout, double decay, bool bOccupied ){
    Vec3d* ptips = (Vec3d*)ptips_;
    #pragma omp parallel for
    for(int i=0; i<npos; i++) {
        Iout[i] = getSTM(  params.nsite, params.spos, params.rots, params.MultiPoles, params.Esites, Qsites + i*params.nsite, ptips[i], Qtips[i], params.E_Fermi, params.cCoupling, decay, params.temperature, bOccupied );
    }
}

/**
 * @brief Solves Hamiltonians for multiple tip positions
 * @param npos Number of positions
 * @param ptips_ Tip positions
 * @param Qtips Tip charges
 * @param Qsites Site charges
 * @param evals Output eigenvalues
 * @param evecs Output eigenvectors
 * @param Hs Output Hamiltonians
 * @param Gs Output Green's functions
 */
void solveHamiltonians(int npos, double* ptips_, double* Qtips, double* Qsites, double* evals, double* evecs, double* Hs, double* Gs ) {
    printf( "solveHamiltonians() npos=%i nsite=%i E_Fermi=%g cCoupling=%g temperature=%g \n", npos, params.nsite, params.E_Fermi, params.cCoupling, params.temperature );
    Vec3d* ptips = (Vec3d*)ptips_;
    
    int nqd = params.nsite;
    
    double Esite[nqd];
    double dE[nqd];
    Mat3d Hmat;
    for (int i = 0; i < npos; i++) {
        
        //{ Vec3d psite=params.spos[2]; printf( "ptips[%3i] r=%20.10f (%8.4f,%8.4f,%8.4f) psite[2](%8.4f,%8.4f,%8.4f)\n", i, (ptips[i]-psite).norm(), ptips[i].x,ptips[i].y,ptips[i].z, psite.x,psite.y,psite.z ); }
        makeCouplingMatrix( nqd, params.spos, params.rots, params.MultiPoles, params.Esites, ptips[i], Qtips[i], Esite, (double*)&Hmat, params.cCoupling);   // Create Hamiltonian matrix for each tip position

        if(Qsites){
            getChargingForce(nqd, Esite, (double*)&Hmat, params.E_Fermi, Qsites+i*nqd, dE);
            //for(int j=0; j<nqd; j++){ ((double*)&Hmat)[j*nqd+j] += dE[j]; } // NOTE: this is wrong, it assumes that this is just energy-shift of on-site due to Coulomb interaction, which is not true
            for(int j=0; j<nqd; j++){ ((double*)&Hmat)[j*nqd+j] = dE[j]; }   // NOTE: this total on-site energy (diagonal of the hamiltonian) it is not just energy-shift due to Coulomb interaction
        }
        if(Hs){
            ((Mat3d*)Hs)[i] = Hmat;
        }

        solveHamiltonian( Hmat, *(Vec3d*)(evals+i*3), (Vec3d*)(evecs+i*9) );                                                                                        // Solve the Hamiltonian to obtain eigenvalues and eigenvectors
        
        if( Gs != 0 ){ // Compute the Green's function using computeGreensFunction
            double G[ params.nsite * params.nsite ];
            computeGreensFunction( (double*)&Hmat, params.E_Fermi, G );

            // Store the Green's function in the output array
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    Gs[i * 9 + j * 3 + k] = G[j * 3 + k];
                }
            }
        }

    }
}

/**
 * @brief Sets up predefined configurations for Boltzmann statistics
 * @param nConf Number of configurations
 * @param siteConfs Array of predefined configurations [size: nConf * nsite]
 */
void setSiteConfBasis(int nconf, double* siteConfs) {
    if(verbosity>0) printf("setSiteConfBasis() nConf=%i nsite=%i\n", nconf, params.nsite);
    params.nConf = nconf;
    params.siteConfs = siteConfs;
}

};