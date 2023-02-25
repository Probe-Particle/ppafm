
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "Vec3.h"
#include "Mat3.h"

#include "spline_hermite.h"
//#include "Mat3.h"
//#include <string.h>
//#include "Grid.h"
//#include "DynamicOpt.h"

#ifdef _WIN64 // Required for exports for ctypes on Windows
    #define DLLEXPORT __declspec(dllexport)
#else
    #define DLLEXPORT
#endif

static int iDebug = 0;

int nMaxComps = 9;

int    spline_Ntypes  = 0;
int    spline_Nps     = 0;
double spline_invStep = 0;
double spline_Rcut    = 0;
double * RFuncSplines = 0;

bool bPBC  = false;
Vec3i npbc = Vec3i {0,0,0};
//GridShape gridShape;
//Vec3d gridA;
//Vec3d gridB;
//Vec3d gridC;
Mat3d cell;

inline int basis_SplineSPD( const Vec3d& dR, int itype, int ncomp, double * bis ){
    double r   = dR.norm();
    if( r>spline_Rcut ) return 0;
    //printf( " r %g Rcut %g \n", r, spline_Rcut );
    double Yr = Spline_Hermite::val_at<double>( r*spline_invStep, RFuncSplines + (itype*spline_Nps) );
    //printf( " r %g Yr %g Rcut %g \n", r, Yr, spline_Rcut );
    bis[0] = Yr;
    if(ncomp<=1) return 1;
    double invr = 1/r;
    Yr*=invr;
    bis[1] += Yr*dR.x;
    bis[2] += Yr*dR.y;
    bis[3] += Yr*dR.z;
    if(ncomp<=4) return 4;
    Yr*=invr;
    bis[4] += Yr*dR.x*dR.y;
    bis[5] += Yr*dR.y*dR.z;
    bis[6] += Yr*dR.z*dR.x;
    double xx = dR.x*dR.x;
    double yy = dR.y*dR.y;
    double zz = dR.z*dR.z;
    bis[7] += Yr*(xx-yy);
    bis[8] += Yr*(2*zz-xx-yy);
    return 9;
}

inline double eval_SplineSPD( const Vec3d& dR, int itype, int ncomp, const double * bis ){
    double r   = dR.norm();
    if( r>spline_Rcut ) return 0;
    double Yr = Spline_Hermite::val_at<double>( r*spline_invStep, RFuncSplines + (itype*spline_Nps) );
    double Ysum = 0;
    Ysum += bis[0] * Yr;
    if(ncomp<=1) return Ysum;
    double invr = 1/r;
    Yr*=invr;
    Ysum += Yr*(
        + bis[1] * Yr*dR.x
        + bis[2] * Yr*dR.y
        + bis[3] * Yr*dR.z);
    if(ncomp<=4) return Ysum;
    double xx = dR.x*dR.x;
    double yy = dR.y*dR.y;
    double zz = dR.z*dR.z;
    Yr*=invr;
    Ysum += Yr*(
        + bis[4] *dR.x*dR.y
        + bis[5] *dR.y*dR.z
        + bis[6] *dR.z*dR.x
        + bis[7] *( xx-yy )
        + bis[8] *( 2*zz-xx-yy ) );
    return Ysum;
}


void symetrizeMatrix( int n, double * A ){
    for(int i=0; i<n; i++){
        for(int j=0; j<i; j++){
            A[j*n+i] = A[i*n+j];
        }
    }
}

int pointsInBox( const Vec3d& pmin, const Vec3d& pmax, int nin, const Vec3d* ps, int* selection ){
    int nout = 0;
    //printf( "pmin(%g,%g,%g) pmax(%g,%g,%g) \n", pmin.x,pmin.y,pmin.z, pmax.x,pmax.y,pmax.z );
    for(int i=0;i<nin;i++){
        const Vec3d& pi = ps[i];
        //printf( "abc[%i/%i] (%1.3f,%1.3f,%1.3f) \n", i, nin, pi.a, pi.b, pi.c );
        if( pi.isBetween(pmin, pmax) ){
            //printf( "pointsInBox [%i] -> %i \n", nout, i );
            selection[nout]=i;
            nout++;
        }
    }
    return nout;
}

int acumByBB( double yref, int nbas, const double* Bs, double* By, double* BB ){
    // accumulate projection    (B^T B)   and  B^T  yref
    //double yref = yrefs[ip];
    //int ik = 0;
    double* BBij = BB;
    for(int i=0; i<nbas; i++){
        double fi =  Bs[i];
        By[i] += fi*yref;
        for(int j=0; j<nbas; j++){
            *BBij = fi*Bs[j];
            BBij++;
        }
    }
    return nbas*nbas;
}

int acumByBB_sparse( double yref, int nbas, int nsel, const int* i0s, const int* il0s, const int* ns, const double* Bs, double* By, double* BB ){
    // accumulate projection    (B^T B)   and  B^T  yref
    //double yref = yrefs[ip];
    int nop=0;
    int ik = 0;
    double* BBij = BB;
    for(int i=0; i<nsel; i++){
        int ni =  ns[i];
        int i0 = i0s[i];
        //printf( "acumByBB [%i] n %i i0 %i il0 %i yref %g   &By %li \n", i, ni, i0, il0s[i], yref, By );
        const double* Bsi = Bs + il0s[i];
        double* Byi = By + i0;
        for(int ik=0; ik<ni; ik++ ){
            //printf( "   ik %i Bs %g dBy %g \n", ik, Bs[ik], Bs[ik]*yref );
            Byi[ik] += Bs[ik]*yref;
            nop++;
        }
        for(int j=0; j<=i; j++){
            int j0 = i0s[j];
            int nj =  ns[j];
            //printf( "   j[%i] n %i i0 %i il0 %i \n", j, nj, j0, il0s[j] );
            double* BBij = BB + i0*nbas + j0;
            const double* Bsj  = Bs + il0s[j];
            for(int ik=0; ik<ni; ik++ ){
                double fi = Bsi[ik];
                for(int jk=0; jk<nj; jk++ ){
                    //printf( " i(%i,%i) j(%i,%i)   fi %g fj %g fifj %g \n",  i,ik, j,jk,  fi, Bsj[jk],  fi*Bsj[jk]  );
                    BBij[jk] += fi*Bsj[jk];
                    nop++;
                }
                BBij += nbas;
            }
        }
    }
    return nop;
}

void matToFile( FILE* fout, const Mat3d& mat ){
    fprintf( fout, "    %g %g %g\n", mat.a.x, mat.a.y, mat.a.z );
    fprintf( fout, "    %g %g %g\n", mat.b.x, mat.b.y, mat.b.z );
    fprintf( fout, "    %g %g %g\n", mat.c.x, mat.c.y, mat.c.z );
}

void selectedAtomsToFile( FILE* fout, const Vec3d& shift, int n, const Vec3d* ps, const int* elems, const int* selection ){
    for(int ii=0; ii<n; ii++){
        int i;
        int elem=1;
        if(selection){ i=selection[ii]; }else{ i=ii; };
        if(elems){ elem=elems[i]; };
        //printf( "i %i->%i(%i) (%g,%g,%g) \n", ii, i, selection[ii], shift.x,shift.y,shift.z );
        const Vec3d pi = ps[i] + shift;
        fprintf( fout, "%i %5.5f %5.5f %5.5f \n", elem, pi.x,pi.y,pi.z );
    }
}

template<typename Func>
int map_pbc_images( int ncenters, Vec3d* centers, Func func ){

    int*    selection = new int  [ncenters];
    Vec3d*  abcs      = new Vec3d[ncenters];

    // transform atoms to grid cooerdinates
    Mat3d invCell; cell.invert_to(invCell);
    for(int i=0; i<ncenters; i++){
        invCell.dot_to( centers[i], abcs[i] );
    }

    Vec3d margin;
    margin.a = invCell.a.norm() * spline_Rcut;
    margin.b = invCell.b.norm() * spline_Rcut;
    margin.c = invCell.c.norm() * spline_Rcut;

    //printf( " margin (%g,%g,%g) Rcut %g \n", margin.a, margin.b, margin.c, spline_Rcut );

    int ntot = 0;
    for(int ia=-npbc.a; ia<=npbc.a; ia++){
        for(int ib=-npbc.b; ib<=npbc.b; ib++){
            Vec3d shift_ab = cell.a*ia + cell.b*ib;
            for(int ic=-npbc.c; ic<=npbc.c; ic++){
                Vec3d pmin = Vec3d {(double)( -ia),(double)( -ib),(double)( -ic)} - margin;
                Vec3d pmax = Vec3d {(double)(1-ia),(double)(1-ib),(double)(1-ic)} + margin;
                int nfound = pointsInBox( pmin, pmax, ncenters, abcs, selection );
                //printf( " icell(%i,%i,%i) nfound %i pmin(%g,%g,%g) pmax(%g,%g,%g) \n", ia,ib,ic, nfound, pmin.x,pmin.y,pmin.z, pmax.x,pmax.y,pmax.z );
                Vec3d shift = shift_ab + cell.c*ic;
                //selectedAtomsToFile( fout, shift, nfound, centers, NULL, selection );
                func( {ia,ib,ic}, shift, nfound, selection, centers );
                ntot+=nfound;
            }
        }
    }

    delete [] selection;
    delete [] abcs     ;
    return ntot;
}

void saveDebugGeomXsfPBC( int ncenters, Vec3d* centers ){
    FILE* fout = fopen ( "debugGeomPBC.xsf", "w" );
    fprintf( fout, "CRYSTAL\n" );
    fprintf( fout, "PRIMVEC\n" );
    matToFile( fout, cell );
    fprintf( fout, "CONVVEC\n" );
    matToFile( fout, cell );
    fprintf( fout, "PRIMCOORD\n" );
    long int mark_1 = ftell(fout);
    fprintf( fout, "                                \n" );
    int*    selection = new int  [ncenters];
    Vec3d*  abcs      = new Vec3d[ncenters];

    int ntot = map_pbc_images( ncenters, centers,
        [&fout]( const Vec3i iabc, const Vec3d& shift, int n, const int * selection, const Vec3d* centers ){
            selectedAtomsToFile( fout, shift, n, centers, NULL, selection );
    });

    fseek(fout, mark_1, SEEK_SET );
    fprintf( fout, "%i %i", ntot, 1 );
    fclose(fout);
}

// coefs is array of coefficient for each atom; nc is number of coefs for each atom
//template<double (Vec3d dR, Vec3d& fout, double * coefs)>
void getBasisProjections_SplineSPD(
    int nps,
    int ncenters,
    Vec3d*  ps,
    double* Yrefs,
    Vec3d* centers,
    int* types,
    int* ncomps,
    double * By,
    double * BB
){
    // TODO:  OpenMP paralelization https://bisqwit.iki.fi/story/howto/openmp/

    int * offsets  = new int[ncenters];
    int * i0s      = new int[ncenters];
    int * il0s     = new int[ncenters];
    int * nBs      = new int[ncenters];

    int nbas = 0;
    for(int i=0;i<ncenters;i++){ offsets[i]=nbas; nbas+=ncomps[i]; };
    double*  Bs  = new double[nbas];

    int ntot = map_pbc_images( ncenters, centers,
        [&]( const Vec3i iabc, const Vec3d& shift, int nfound, const int * selection, const Vec3d* centers ){
            int nbmax = 0;
            for(int i=0; i<nfound; i++){ nbmax += ncomps[ selection[i] ]; };
            for(int ip=0; ip<nps; ip++){
                //if(ip%100000==0) printf( " iabc(%i,%i,%i)[%i] \n", iabc.a,iabc.b,iabc.c, ip );
                double* bis  = Bs;
                int*    nbis = nBs;
                for(int j=0; j<nbmax; j++ ){ bis[j]=0; }
                Vec3d pi = ps[ip] - shift;
                int nsel = 0;
                for(int ii=0; ii<nfound; ii++){
                    int i          = selection[ii];
                    int ncomp      = ncomps[i];
                    int ncomp_true = basis_SplineSPD( pi-centers[i], types[i], ncomp, bis );
                    if( ncomp_true>0 ){
                        nBs [nsel] = ncomp_true;
                        i0s [nsel] = offsets[i];
                        il0s[nsel] = bis - Bs;
                        bis += ncomp_true;
                        nsel++;
                    }
                } // ii ... atoms/basis functions
                acumByBB_sparse( Yrefs[ip], nbas, nsel, i0s, il0s, nBs, Bs, By, BB );
            } // ip ... grid points
    });
    symetrizeMatrix( nbas, BB );
    delete [] offsets  ;
    delete [] i0s      ;
    delete [] il0s     ;
    delete [] Bs       ;
    delete [] nBs      ;
}

void project_SplineSPD(
    int nps,
    int ncenters,
    Vec3d*  ps,
    double* Youts,
    Vec3d*  centers,
    int* types,
    int* ncomps,
    double* coefs
){
    // TODO:  OpenMP paralelization https://bisqwit.iki.fi/story/howto/openmp/
    int ntot = map_pbc_images( ncenters, centers,
        [&]( const Vec3i iabc, const Vec3d& shift, int nfound, const int * selection, const Vec3d* centers ){
            int nbmax = 0;
            for(int i=0; i<nfound; i++){ nbmax += ncomps[ selection[i] ]; };
            for(int ip=0; ip<nps; ip++){
                //if(ip%100000==0) printf( " iabc(%i,%i,%i)[%i] \n", iabc.a,iabc.b,iabc.c, ip );
                Vec3d pi = ps[ip] - shift;
                double ypi = 0;
                double * coefi = coefs;
                for(int ii=0; ii<nfound; ii++){
                    int i     = selection[ii];
                    int ncomp = ncomps[i];
                    ypi += eval_SplineSPD( pi-centers[i], types[i], ncomp, coefi );
                    coefi += ncomp;
                } // ii ... atoms/basis functions
                Youts[ip] += ypi;
            } // ip ... grid points
    });
}


// =====================================================
// ==========   Export these functions ( to Python )
// =====================================================

extern "C"{

DLLEXPORT void setPBC( int * npbc_, double * cell_ ){
    //gridA = *  (Vec3d*)cell;
    //gridB = *(((Vec3d*)cell)+1);
    //gridC = *(((Vec3d*)cell)+2);
    cell = *(Mat3d*)cell_;
    npbc = *((Vec3i*)npbc_);
    bPBC = true;
}

DLLEXPORT void setSplines( int ntypes, int npts, double invStep, double Rcut, double* RFuncs  ){
    spline_Ntypes  = ntypes;
    spline_Nps     = npts;
    spline_invStep = invStep;
    spline_Rcut    = Rcut;
    RFuncSplines = RFuncs;
}

DLLEXPORT void getProjections(
    int nps, int ncenters,
    double*  ps, double* Yrefs,
    double* centers, int* types, int* ncomps,
    double * By,
    double * BB
){
    getBasisProjections_SplineSPD(
        nps, ncenters,
        (Vec3d*)ps, Yrefs,
        (Vec3d*)centers, types, ncomps,
        By, BB
    );
}

DLLEXPORT void project(
    int nps, int ncenters,
    double*  ps, double* Youts,
    double* centers, int* types, int* ncomps,
    double* coefs
){
    project_SplineSPD(
        nps,
        ncenters,
        (Vec3d*)ps,
        Youts,
        (Vec3d*)centers,
        types,
        ncomps,
        coefs
    );
}

DLLEXPORT void debugGeomPBC_xsf( int ncenters, double* centers ){ saveDebugGeomXsfPBC( ncenters, (Vec3d*)centers ); }

}
