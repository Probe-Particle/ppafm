
#include "fastmath.h"
#include "Vec3.h"
#include "Mat3.h"

static int  iDebug = 1;

#include "Forces.h"
#include "FARFF.h"
//#include "NBFF.h"
#include "DynamicOpt.h"

#include <vector>

using namespace FlexibleAtomReactive;

// ========= data

FARFF       ff;        // inter moleculer forcefield
//NBFF        nff;       // non-bonding intermolecular forcefield
DynamicOpt  opt;

std::vector<FlexibleAtomType*> atomTypes;

bool bNonBonded = false;

struct{
    Vec3d pmin=(Vec3d){.0,0,0};
    Vec3d pmax=(Vec3d){.0,0,0};
    Vec3d k   =(Vec3d){-1.,-1,-1};
} box;

struct{
    int itype0=0;
    FlexibleAtomType atype0 = FlexibleAtomType();  
} defaults;

// ========= Export Fucntions to Python

extern "C"{

int insertAtomType( int nbond, int ihyb, double rbond0, double aMorse, double bMorse, double c6, double R2vdW, double Epz ){
    //FlexibleAtomType* atyp = new FlexibleAtomType();
    //atomTypes.push_back( FlexibleAtomType() );
    atomTypes.push_back( new FlexibleAtomType() );
    FlexibleAtomType& atyp = *(atomTypes.back());
    //RigidAtomType& atyp = atomTypes.back();
    //atyp->nbond  =  nbond;    // number bonds
    atyp.conf.a = nbond;
    //atyp->conf.b = nbond;
    //atyp->conf.b = nbond;
    atyp. rbond0 =  0.8;  // Rbond
    atyp. aMorse =  4.0;  // EBond
    atyp. bMorse = -0.7;  // kBond
    atyp. vdW_c6 = -15.0; // c6
    atyp. vdW_R  =  8.0;  // RcutVdW
    atyp. Wee    =  1.0;   // electon-electron width
    atyp. Kee    = 10.0;  // electon-electron strenght
    atyp. Kpp    = 10.0;  // onsite  p-p orthogonalization strenght
    atyp. Kpi    =  2.0;   // offsite p-p alignment         strenght
    return atomTypes.size()-1;
}

int reallocFF(int natom){
    defaults.itype0 = atomTypes.size();
    defaults.atype0  = FlexibleAtomType();
    atomTypes.push_back( &defaults.atype0 );
    return ff.realloc(natom);
}

//int*    getTypes (){ return (int*)   ff.atypes; }
double* getDofs  (){ return (double*)ff.dofs;   }
double* getFDofs (){ return (double*)ff.fdofs;  }
double* getEDofs (){ return (double*)ff.edofs;  }

void setupFF( int natom, int* itypes ){
    for(int i=0; i<natom; i++){ 
        ff.atypes[i]=atomTypes[itypes[i]];
        ff.aconf [i]=ff.atypes[i]->conf; 
    };
    ff.normalizeOrbs();
    opt.bindOrAlloc( ff.nDOF, ff.dofs, 0, ff.fdofs, 0 );
    opt.cleanVel( );
}

double setupOpt( double dt, double damp, double f_limit, double l_limit ){
    opt.initOpt( dt, damp );
    opt.f_limit = f_limit;
    opt.l_limit = l_limit;
}

void setBox(double* pmin, double* pmax, double* k){
    box.pmin=*(Vec3d*)pmin;
    box.pmax=*(Vec3d*)pmax;
    box.k   =*(Vec3d*)k;
}

double relaxNsteps( int nsteps, double Fconv, int ialg ){
    double F2conv = Fconv*Fconv;
    double F2=1.0,E =0;
     printf( "relaxNsteps nsteps %i \n", nsteps );
    for(int itr=0; itr<nsteps; itr++){
        printf( "relaxNsteps itr %i \n", itr );
        ff.cleanForce(); 
        E = ff.eval();   
        //nff.eval();
        //if( surf.K < 0.0 ){ ff.applyForceHarmonic1D( surf.h, surf.x0, surf.K); }
        //if( box .K < 0.0 ){ ff.applyForceBox       ( box.p0, box.p1, box.K, box.fmax); }
        if((box.k.x>0)||(box.k.y>0)||(box.k.z>0)){ 
            for(int i=0; i<ff.natom; i++){
                 boxForce( ff.apos[i], ff.aforce[i], box.pmin, box.pmax, box.k );
            }
        }
        if(iDebug>0){
            Vec3d cog,fsum,torq;
            checkForceInvariatns( ff.natom, ff.aforce, ff.apos, cog, fsum, torq );
            printf( "DEBUG CHECK INVARIANTS  fsum %g torq %g   cog (%g,%g,%g) \n", fsum.norm(), torq.norm(), cog.x, cog.y, cog.z );
        }
        switch(ialg){
            case 0: F2 = opt.move_FIRE();  break;
            case 1: opt.move_GD(opt.dt);   break;
            case 3: opt.move_MD(opt.dt);   break;
        }
        if(iDebug>0){ printf("relaxNsteps[%i] |F| %g(>%g) E %g dt %g(%g..%g) damp %g \n", itr, sqrt(F2), Fconv, E, opt.dt, opt.dt_min, opt.dt_max, opt.damping ); }
        if(F2<F2conv) break;
    }
    return F2;
}

int main(){
    int natom = 2;
    reallocFF(natom);
    ff.apos[0] = (Vec3d){0.0,0.0,0.0};
    ff.apos[0] = (Vec3d){0.0,2.0,0.0};
    int itypes[] = {0,0};
    setupFF( natom, itypes );
    relaxNsteps( 10, 1e-6, 0 );
}

}

