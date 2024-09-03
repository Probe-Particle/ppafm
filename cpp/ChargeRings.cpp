

/*


- **Total Energy**:

\[
U_{\text{total}} = \sum_{i=1}^3 \left( \frac{E_i Q_i}{2} + \frac{Q_i Q_{\text{tip}}}{4 \pi \epsilon_0 r_{i\text{tip}}} - \mu Q_i \right) + \sum_{i=1}^3 \sum_{j=i+1}^3 \frac{Q_i Q_j}{4 \pi \epsilon_0 r_{ij}}
\]

U = sum_i{  Q_i * ( eps_i - mu + Q_tip/r_{i,tip} + sum_j{ Q_j/r_{ij} } ) }

- **Variational Derivative**:

\[
\frac{\delta U_{\text{total}}}{\delta Q_i} = \frac{E_i}{2} + \frac{Q_{\text{tip}}}{4 \pi \epsilon_0 r_{i\text{tip}}} + \sum_{\substack{j=1 \\ j \neq i}}^3 \frac{Q_j}{4 \pi \epsilon_0 r_{ij}} - \mu
\]

dU/dQ_i =  eps_i - mu + Q_tip/r_{i,tip} + sum_j{ Q_j/r_{ij} }



We can also solve ti as matrix equation

A * qs = b
where:
Aii = mu - K*Q_tip/r_{i,tip}
Aij = K*Q_j/r_{ij}


*/


#include <stdio.h>

#include "Vec3.h"
//#include "Mat3.h"
//#include "SMat3.h"
//#include "CG.h"

#define COULOMB_CONST      14.3996448915 


int verbosity=0; 

void printmatrix( int ni, int nj, double* A, const char* format ){
    for(int i=0; i<ni; i++){
        for(int j=0; j<nj; j++){
            printf( format, A[i*nj+j]);
        }
        printf("\n");
    }
}

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

/*
double OccupancyVarDerivs( int nsite, Vec3d* spos, const double* E0s, double* Qs, double fQs, double* Vtips, double* Coupling, double E_mu ){
    for(int i=0; i<nsite; i++){
        double fQi = Vtips[i] - E_mu;
        for(int j=0; j<nsite; j++){
            if( i==j ){ continue; }
            double r = Coupling[i*nsite+j];
        }
    }
}
*/

double OccupancyVarDerivsMat( int nsite, const double* Coupling, const double* Qs, double* fQs ){
    double f2err=0;
    double E_on  = 0;
    double E_off = 0;
    for(int i=0; i<nsite; i++){
        double fQi = Coupling[i*nsite+i];
        double qi = Qs[i];
        E_on     += qi*fQi;
        for(int j=0; j<nsite; j++){
            double cij = Coupling[i*nsite+j];
            double  qj = Qs[j];
            fQi += qj*cij;
            E_off += qi*qj*cij;
        }
        fQs[i] = -fQi;
        f2err += fQi*fQi;
    }
    //printf("E_on %g E_off %g f2err %g\n", E_on, E_off, f2err );
    return E_on + E_off;
}

void makeCouplingMatrix( int nsite, const Vec3d* spos, const double* Esite, Vec3d p_tip, double Q_tip, double E_mu, double* Coupling, double cCouling ){
    for(int i=0; i<nsite; i++){
        const Vec3d pi = spos[i];
        double Vitip = COULOMB_CONST * Q_tip /( pi - p_tip ).norm();
        Coupling[i*nsite+i] = Vitip + Esite[i] - E_mu;
        for(int j=i+1; j<nsite; j++){
            //if( i==j ){ continue; }
            Vec3d d  = pi - spos[j];
            double Cij = COULOMB_CONST*cCouling/( pi - spos[j] ).norm();
            Coupling[i*nsite+j] = Cij;
            Coupling[j*nsite+i] = Cij;
        }
    }
}


double moveGD(  int n, double *x, double *f, double dt ){
    double  f2=0;
    for(int i=0; i<n; i++){
        double fi = f[i];
        x[i] += fi*dt;
        f2 += fi*fi;
    }
    return f2;
}

int solveSiteOccupancy( int nsite, const Vec3d* spos, const double* Esite, double* Qs, Vec3d p_tip, double Q_tip, double E_mu, double cCouling, int niter=1000, double tol=1e-6, double dt=0.1 ){
    //  Site occupancy is 1 if below the Fermi level, 0 otherwise
    //  Fermi level is set by the voltage applied to the STM tip and distance from the site
    // The energy of sites is modified by Coulomb interaction between the sites as well
    //   E_i = (Qi*Qj/|r_i-r_j|) + (Qi*Qtip/|r_i-r_tip|) + E0s[i] + Emu

    //double varOnsite[nsite];
    //double varCoupling[];

    double Coupling[ nsite*nsite ];
    double fQs[nsite];
    double vQs[nsite];
    makeCouplingMatrix( nsite, spos, Esite, p_tip, Q_tip, E_mu, Coupling, cCouling );
    printmatrix( nsite, nsite, Coupling, "%12.6f" );

    double tol2 = tol*tol;
    int itr=0;
    for(itr=0; itr<niter; itr++){
        //break;
        double E = OccupancyVarDerivsMat( nsite, Coupling, Qs, fQs );
        //printf("itr=%i Qs{%g,%g,%g} fQs{%g,%g,%g}\n", itr, Qs[0],Qs[1],Qs[2],    fQs[0],fQs[1],fQs[2] );
        //printf("itr=%d, f2err=%g\n", itr, f2err);
        double f2 = moveGD( nsite, Qs, fQs, dt );
        if(f2<tol2){ break; }
    }
    return itr;
}


extern "C"{

void solveSiteOccupancies( int npos, double* ptips_, double* Qtips,  int nsite, double* spos, const double* Esite, double* Qout, double E_mu, double cCouling, int niter, double tol, double dt ){
    Vec3d* ptips = (Vec3d*)ptips_;
    double Qs[nsite];
    for(int i=0; i<npos; i++){ Qs[nsite]=0; }
    for(int i=0; i<npos; i++){
        solveSiteOccupancy( nsite, (Vec3d*)spos, Esite, Qs, ptips[i], Qtips[i], E_mu, cCouling, niter, tol, dt );        
        for(int j=0; j<nsite; j++){ Qout[i*nsite+j] = Qs[j]; }
    }
}

};