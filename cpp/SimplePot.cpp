
static int iDebug = 0;

#include "macroUtils.h"
#include "SimplePot.h"

SimplePot pot;

double assignWeighs(int n, Vec3d* pos, double* Ws, Vec3d p0, Vec3d K ){
    double wsum = 0;
    for(int i=0; i<n; i++){
        Vec3d d = pos[i] - p0;
        //d.mul(d);
        double E = 0.5*K.dot(d*d);
        double w = exp( -E );
        wsum+=w;
        //printf( "assignWeighs[%i] wsum %g w %g E %g d(%g,%g,%g) \n", i, wsum, w, E, d.x, d.y, d.z );
        Ws[i]=wsum;
    }
    return wsum;
}


extern "C"{

void init( int natom, int neighPerAtom, double* apos, double* Rcovs ){
    pot.natom=natom;
    pot.neighPerAtom=neighPerAtom;
    pot.apos  = (Vec3d*)apos;
    pot.Rcovs = Rcovs;
    _realloc( pot.nneighs, natom );
    _realloc( pot.neighs , natom*neighPerAtom );
    pot.makeNeighbors();
}

void eval( int n, double* Es, double* pos_, double Rcov, double RvdW ){
    Vec3d* pos = (Vec3d*)pos_;
    for(int i=0; i<n; i++){
        Es[i] =  pot.eval( pos[i], Rcov, RvdW );
    }
}

int danglingToArray( double* dangs, double* pmin, double* pmax, double Rcov ){
    return pot.danglingToArray( (Vec3d*)dangs, Rcov, *(Vec3d*)pmin, *(Vec3d*)pmax );
}

void init_random(int seed){ srand(seed); }

int genNewAtom( int np, double* pos_, double* Ws, double* p0_, double* K_ ){
    Vec3d* pos = (Vec3d*)pos_;
    Vec3d p0   = *(Vec3d*)p0_;
    Vec3d K    = *(Vec3d*)K_;
    //printf( "genNewAtom np %i \n", np );
    double wsum = assignWeighs( np, pos, Ws, p0, K );
    //srand(5454);
    double rnd = randf()*wsum;
    for(int i=0; i<np; i++){
        //printf( "genNewAtom[%i] rnd %g w %g \n", i, rnd, Ws[i] );
        if(rnd<Ws[i]){ return i; }
    }
    return -1;
}

double randomOptAtom( int ntry, double* pos_, double* spread_, double Rcov, double RvdW ){
    Vec3d pos    = *(Vec3d*)pos_;
    Vec3d spread = *(Vec3d*)spread_;
    Vec3d pmin=pos-spread;
    Vec3d pmax=pos+spread;
    double Ebest = 1e+300; 
    for( int i=0; i<ntry; i++ ){
        Vec3d p; p.fromRandomBox(pmin,pmax);
        double E = pot.eval( p, Rcov, RvdW );
        if(E<Ebest){
            Ebest=E;
            pos=p;
        }
    }
    return Ebest;
}

}
