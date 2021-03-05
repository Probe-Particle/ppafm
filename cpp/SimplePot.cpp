
static int iDebug = 0;

#include "macroUtils.h"
#include "SimplePot.h"

SimplePot pot;

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

}
