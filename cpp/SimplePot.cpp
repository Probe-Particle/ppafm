
static int iDebug = 0;

#include "macroUtils.h"
#include "SimplePot.h"
#include "GridUtils.cpp"

// Golobals  
namespace grid{
    Vec3i   ns;
    Vec3d   pmin;
    Vec3d   pmax;
    Vec3d   step;
    Vec3d   invStep;
    double* Ws=0;

    double interpolate( Vec3d p, double* V=Ws );
}

SimplePot pot;


double grid::interpolate( Vec3d p, double* V ){
    Vec3d s = (p-pmin)*invStep;
    int ix = (int)s.x;
    int iy = (int)s.y; 
    int iz = (int)s.z; 
    if( (ix<0)||((ix+1)>=ns.x) || (iy<0)||((iy+1)>=ns.y) || (iz<0)||((iz+1)>=ns.z)  ) return 0;
    double dx=s.x-ix; double mx=1-dx;
    double dy=s.y-iy; double my=1-dy;
    double dz=s.z-iz; double mz=1-dz;
    int xdi=ns.z*ns.y;
    int ydi=ns.z;
    int zdi=1;
    int i0 = ix*xdi + iy*ydi + iz*zdi;
    //printf( "interp(%i,%i,%i) i0 %i \n", ix, iy, iz, i0 );
    //return 1.0;
    //return Ws[i0 ];
    return( mx*Ws[i0        ] + dx*Ws[i0+xdi        ] )*my*mz
         +( mx*Ws[i0+ydi    ] + dx*Ws[i0+xdi+ydi    ] )*dy*mz
         +( mx*Ws[i0    +zdi] + dx*Ws[i0+xdi    +zdi] )*my*dz
         +( mx*Ws[i0+ydi+zdi] + dx*Ws[i0+xdi+ydi+zdi] )*dy*dz;
}

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

double assignWeighsGrid( int n, Vec3d* pos, double* Ws, double kT ){
    double wsum = 0;
    //printf( "ns(%i,%i,%i) \n", grid::ns.x,grid::ns.y,grid::ns.z  );
    //printf( "(%g,%g,%g) (%g,%g,%g) \n", grid::pmin.x,grid::pmin.y,grid::pmin.z,  grid::pmax.x,grid::pmax.y,grid::pmax.z  );
    /*
    int ii=0;
    for(int ix=0;ix<grid::ns.x;ix++){ 
    for(int iy=0;iy<grid::ns.y;iy++){
    for(int iz=0;iz<grid::ns.z;iz++){
       printf(  "Ws[%i,%i,%i] = %g \n", ix,iy,iz, grid::Ws[ii] ); ii++;
    }}} exit(0);
    */
    for(int i=0; i<n; i++){
        double E = -grid::interpolate( pos[i] );
        //double w = exp( -E/kT ) - 1.0;
        double w =  -E/kT;
        wsum+=w;
        //if( (-E/kT)>1 ) printf( "assignWeighs[%i] wsum %g w %g E %g d(%g,%g,%g) \n", i, wsum, w, E, pos[i].x, pos[i].y, pos[i].z );
        Ws[i]=wsum;
        //Ws[i]=w;
    }
    return wsum;
}

extern "C"{

void setGridSize( int* ns_, double* pmin_, double* pmax_){
    using namespace grid;
    ns   = *(Vec3i*)ns_;
    pmin = *(Vec3d*)pmin_;
    pmax = *(Vec3d*)pmax_;
    step = (pmax-pmin).div( ns.x,ns.y,ns.z );
    grid::invStep = step.get_inv();
}

void setGridPointer(double* data){
    //printf( "C++ setGridPointer \n" );
    grid::Ws = data; 
}

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

void pickAtomWeighted( int npick, int* ipicks, int nps, double* pos_, double* Ws, double* p0_, double* K_, double kT ){
    //printf( "DEBUG START pickAtomWeighted %i %i \n", npick, nps );
    Vec3d* pos = (Vec3d*)pos_;
    Vec3d p0   = *(Vec3d*)p0_;
    Vec3d K    = *(Vec3d*)K_;
    //printf( "genNewAtom np %i \n", np );
    double wsum;
    if(grid::Ws){ wsum = assignWeighsGrid( nps, pos, Ws, kT );           }
    else        { wsum = assignWeighs    ( nps, pos, Ws, p0, K*(1/kT) ); }
    //srand(5454);
    for(int ip=0; ip<npick; ip++){
        double rnd = randf()*wsum;
        //printf( "ip %i rnd %g \n", ip,  rnd );
        int ipick=-1;
        for(int i=0; i<nps; i++){
            //printf( "genNewAtom[%i] rnd %g w %g \n", i, rnd, Ws[i] );
            if(rnd<Ws[i]){ ipick=i; break; }
        }
        ipicks[ip]=ipick;
    }
    //printf( "DEBUG END pickAtomWeighted \n" );
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
