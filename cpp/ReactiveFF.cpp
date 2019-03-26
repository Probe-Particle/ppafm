

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <math.h>

//#include <list>
#include <vector>

//#include <SDL2/SDL.h>
//#include <SDL2/SDL_opengl.h>
//#include "Draw.h"
//#include "Draw3D.h"
//#include "Solids.h"

#include "fastmath.h"
#include "Vec3.h"
#include "Mat3.h"
//#include "VecN.h"

/*
#include "Multipoles.h"
#include "PotentialFlow.h"
#include "grids3D.h"
#include "MultipoleGrid.h"
*/

//#include "AppSDL2OGL_3D.h"
//#include "testUtils.h"
//#include "SDL_utils.h"
//#include "Plot2D.h"

//#include "MMFF.h"

//#include "RARFF.h"
//#include "RARFF2.h"
#include "RARFFarr.h"
#include "QEq.h"

#define R2SAFE  1.0e-8f

// ============ Global Variables

//RARFF2     ff;
RARFF2arr  ff;
QEq       qeq;
//std::list<RigidAtomType>  atomTypes;
std::vector<RigidAtomType*> atomTypes;


// surface
struct {
    Vec3d  h  =  Vec3dZ;
    double K  =  1.0;
    double x0 =  0.0;
} surf;

struct {
    Vec3d p0,p1;
    double K    =  1.0;
    double fmax =  0.0;
} box;

extern "C"{

// ========= Grid initialization

int insertAtomType( int nbond, int ihyb, double rbond0, double aMorse, double bMorse, double c6, double R2vdW, double Epz ){
    RigidAtomType* atyp = new RigidAtomType();
    atomTypes.push_back( atyp );
    //RigidAtomType& atyp = atomTypes.back();
    atyp->nbond  =  nbond;    // number bonds
    atyp->rbond0 =  rbond0;
    atyp->aMorse =  aMorse;
    atyp->bMorse =  bMorse;
    atyp->Epz    =  Epz;
    atyp->c6     =  c6;
    atyp->R2vdW  =  R2vdW;
    printf("insertAtomType %i %i  %g %g %g %g %g ", nbond, ihyb, rbond0, aMorse, bMorse, c6, R2vdW );
    switch(ihyb){
        case 0: atyp->bh0s = (Vec3d*)sp1_hs; printf("sp1\n"); break;
        case 1: atyp->bh0s = (Vec3d*)sp2_hs; printf("sp2\n"); break;
        case 2: atyp->bh0s = (Vec3d*)sp3_hs; printf("sp3\n"); break;
    };
    return atomTypes.size()-1;
}

void ralloc(int natom){
    ff.realloc(natom);
    ff.cleanAux();
}

void clean(){ ff.cleanAux(); }

int*    getTypes (){ return (int*)   ff.types;  }
double* getPoss  (){ return (double*)ff.poss;   }
double* getQrots (){ return (double*)ff.qrots;  }
double* getHbonds(){ return (double*)ff.hbonds; }
double* getEbonds(){ return (double*)ff.ebonds; }
double* getBondCaps(){ return (double*)ff.bondCaps; }

void setTypes( int natoms, int* types ){
    for(int i=0; i<natoms; i++){ ff.types[i]=atomTypes[types[i]]; };
}

/*
void setAtoms( int natoms, int* types, double* apos, double* qrots[i]){
    ff.realloc(natoms);
    for(int i=0; i<natoms; i++){
        if(randf()>0.5){ ff.atoms[i].type=&type1;  }else{ ff.atoms[i].type=&type2; }
        ff.atoms[i].pos  = apos [i];
        ff.atoms[i].qrot = qrots[i];
        ff.atoms[i].cleanAux();
    }
}
*/

void setSurf(double K, double x0, double* h ){
    surf.h  =  *(Vec3d*)h;
    surf.K  =  K;
    surf.x0 =  x0;
}

void setBox(double K, double fmax, double* p0, double* p1 ){
    box.p0  =  *(Vec3d*)p0;
    box.p1  =  *(Vec3d*)p1;
    box.K    =  K;
    box.fmax =  fmax;
}

int passivateBonds( double Ecut ){ return ff.passivateBonds( Ecut ); }

double relaxNsteps( int nsteps, double F2conf, double dt, double damp ){
    double F2=1.0;
    for(int itr=0; itr<nsteps; itr++){
        ff.cleanAtomForce();
        ff.projectBonds();
        ff.interEF();
        if( surf.K < 0.0 ){ ff.applyForceHarmonic1D( surf.h, surf.x0, surf.K); }
        if( box .K < 0.0 ){ ff.applyForceBox       ( box.p0, box.p1, box.K, box.fmax); }
        ff.evalTorques();
        F2 = ff.evalF2pos() + ff.evalF2rot(); // some scaling ?
        //printf( "itr %i F2 %g \n", itr, F2 ); 
        if(F2<F2conf) break;
        ff.moveMDdamp(dt, damp);
        //exit(0);
    }
    return F2;
}


//double* J    = 0;
//double* qs   = 0;
//double* fqs  = 0;
//double* vqs  = 0;
//double* affins = 0;
//double* hards  = 0;

void    setTotalCharge(double q){ qeq.Qtarget = q; }
double* getChargeJ        (){ return (double*)qeq.J;   }
double* getChargeQs       (){ return (double*)qeq.qs;  }
double* getChargeFs       (){ return (double*)qeq.fqs; }
double* getChargeAffinitis(){ return (double*)qeq.affins; }
double* getChargeHardness (){ return (double*)qeq.hards;  }

/*
void setupCharge( int* itypes, double* taffins, double* thards ){
    qeq.realloc( ff.natom );
    for(int i=0;i<ff.natom; i++){
        int ityp  = itypes[i];
        affins[i] = taffins[ityp];
        hards [i] = thards [ityp];
    }
}
*/

void setupChargePos( int natom, double* pos, int* itypes, double* taffins, double* thards ){
    //printf( "to realloc \n"  );
    qeq.realloc( natom );
    //printf( "to realloc Done \n"  );
    for(int i=0;i<natom; i++){
        int ityp  = itypes[i] - 1;
        //printf( "types %i %i %g %g \n", i, ityp, taffins[ityp], thards [ityp] );
        qeq.affins[i] = taffins[ityp];
        qeq.hards [i] = thards [ityp];
    }
    //printf( "to makeCoulombMatrix \n"  );
    makeCoulombMatrix( natom, (Vec3d*)pos, qeq.J );
    for(int i=0; i<natom; i++ ){  
        qeq.qs [i] = 0.0;
        qeq.vqs[i] = 0.0;
        qeq.fqs[i] = 0.0;
    }
    //printf( "makeCoulombMatrix DONE ! \n"  );
}

double relaxCharge( int nsteps, double F2conf, double dt, double damp ){
    //makeCoulombMatrix( ff.natom, ff.poss, qeq.J){
    double F2=1.0;
    for(int itr=0; itr<nsteps; itr++){
        //printf( "to getQvars \n" );
        F2 = qeq.getQvars();
        if(F2<F2conf) break;
        //printf( "F2 %g \n", F2 );
        qeq.moveMDdamp(dt, damp);
    }
    return F2;
}

} // extern "C"{


