
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "Vec3.h"

#include "spline_hermite.h"
//#include "Mat3.h"
//#include <string.h>
//#include "Grid.h"
//#include "DynamicOpt.h"

static int iDebug = 0;

int nMaxComps = 9;

int    spline_Ntypes  = 0;
int    spline_Nps     = 0;
double spline_invStep = 0;
double spline_Rcut    = 0;
double * RFuncSplines = 0;

bool bPBC  = false;
Vec3i npbc = (Vec3i){1,1,1};
//GridShape gridShape;
Vec3d gridA;
Vec3d gridB;
Vec3d gridC;



int basis_SplineSPD( const Vec3d& dR, int itype, int ncomp, double * bis ){
    double r   = dR.norm();
    if( r>spline_Rcut ) return 0;
    double Yr = Spline_Hermite::val_at<double>( r*spline_invStep, RFuncSplines + (itype*spline_Nps) );
    bis[0] = Yr;
    if(ncomp<=1) return 1;
    double invr = 1/r;
    Yr*=invr;
    bis[1] += Yr*dR.x;
    bis[2] += Yr*dR.x;
    bis[3] += Yr*dR.x;
    if(ncomp<=4) return 4;
    Yr*=invr;
    bis[4] += Yr*dR.x*dR.y;
    bis[5] += Yr*dR.y*dR.z;
    bis[6] += Yr*dR.z*dR.x;
    double xx = dR.x*dR.x;
    double yy = dR.x*dR.x;
    double zz = dR.x*dR.x;
    bis[7] += xx-yy;
    bis[8] += 2*zz-xx-yy;
    return 9;
}

// coefs is array of coefficient for each atom; nc is number of coefs for each atom
//template<double (Vec3d dR, Vec3d& fout, double * coefs)>
void getBasisProjections_SplineSPD( 
    int nps, 
    int ncenters,
    Vec3d*  ps,
    double* yrefs, 
    Vec3d* centers, 
    int* types,
    int* ncomps,
    double * By,
    double * BB
){

    // TODO:  OpenMP paralelization https://bisqwit.iki.fi/story/howto/openmp/

    int nbas = 0;
    for(int i=0;i<ncenters;i++){ nbas+=ncomps[i]; };

    double*  Bsi = new double[nbas];
    int*    nBsi = new int   [ncenters];

    printf( " nps %i ncenters %i nbas %i \n", nps, ncenters, nbas );

    for(int ip=0; ip<nps; ip++){
        Vec3d pos = ps[ip];

        if( ip%100000==0 ) printf( "point %i (%g,%g,%g) \n", ip, pos.x, pos.y, pos.z );

        // evaluate basis at pos point
        double* bis  = Bsi;
        int*    nbis = nBsi; 
        for(int i=0; i<ncenters; i++){
            //int ncomp = getBasisFunc( pos-centers[i], types[i], bis );
            for(int i=0; i<nMaxComps; i++ ){ bis[i]=0; }

            int ncomp = 0;
            if(bPBC){
                for(int ia=-npbc.a; ia<=npbc.a; ia++){
                    for(int ib=-npbc.b; ib<=npbc.b; ib++){
                        Vec3d shift_ab = gridA*ia + gridB*ib;
                        for(int ic=-npbc.c; ic<=npbc.c; ic++){
                            Vec3d dR = centers[i];
                            dR.add    ( shift_ab  );
                            dR.add_mul( gridA, ic );
                            dR.sub    ( pos       );
                            int ncomp_ = basis_SplineSPD( dR, types[i], ncomps[i], bis );
                            if(ncomp_>ncomp) ncomp=ncomp_;
                        }
                    }
                }
            }else{
                ncomp = basis_SplineSPD( pos-centers[i], types[i], ncomps[i], bis );
            }

            if( ncomp>0 ){
                *nbis = ncomp;
                bis  += ncomp;
                nbis++;
            }
        }

        // accumulate projection    (B^T B)   and  B^T  yref
        double yref = yrefs[ip];
        int ik = 0;
        double* BBij = BB;
        for(int i=0; i<nbas; i++){
            double fi =  Bsi[i]; 
            By[i] += fi*yref;
            for(int j=0; j<nbas; j++){
                *BBij = fi*Bsi[j];
                BBij++;
            }
        }

    } // ip ... points
    delete [] Bsi;
    delete [] nBsi;
}


// =====================================================
// ==========   Export these functions ( to Python )
// =====================================================

extern "C"{

void setPBC( int * npbc_, double * cell ){
    gridA = *  (Vec3d*)cell;
    gridB = *(((Vec3d*)cell)+1);
    gridC = *(((Vec3d*)cell)+2);
    npbc  = *((Vec3i*)npbc_);
    bPBC = true;
}

void setSplines( int ntypes, int npts, double invStep, double Rcut, double* RFuncs  ){
    int    spline_Ntypes  = ntypes;
    int    spline_Nps     = npts;
    double spline_invStep = invStep;
    double spline_Rcut    = Rcut;
    double * RFuncSplines = RFuncs;


}

void getProjections( 
    int nps, int ncenters,
    double*  ps, double* yrefs, 
    double* centers, int* types, int* ncomps,
    double * By,
    double * BB
){
    getBasisProjections_SplineSPD( 
        nps, ncenters, 
        (Vec3d*)ps, yrefs, 
        (Vec3d*)centers, types, ncomps,
        By, BB
    );
}

}





