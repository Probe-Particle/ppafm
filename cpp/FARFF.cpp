
#include "fastmath.h"
#include "Vec3.h"
#include "Mat3.h"

static int  iDebug = 1;

#include "Forces.h"
#include "FARFF.h"
//#include "NBFF.h"
#include "DynamicOpt.h"
#include "Grid.h"

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

class GridFF_AtomBond{ public:
    GridShape gridShape;
    Vec3d    * atomMap = 0;
    Vec3d    * bondMap = 0;

    double lbond             = 0.8;  // bond lever length
    double atomFroceStrenght = 0.5;  // ToDo - this may ne more sophisticated - specific for each atom-type or so
    double orbFroceStrenght  = 0.1;  // ToDo - this may ne more sophisticated - specific for each atom-type or so

    int natom   = 0;
    int norb    = 0;

    Vec3ui8 * aconf  = 0;

    Vec3d  * apos    = 0;      // atomic position // ALIAS
    Vec3d  * aforce  = 0;      // atomic forces   // ALIAS

    Vec3d  * opos    = 0;     // normalized bond unitary vectors  // ALIAS
    Vec3d  * oforce  = 0;

    // ========== Functions 

    void bindDOFs( int natom_, int norb_, Vec3d* apos_, Vec3d* aforce_, Vec3d* opos_, Vec3d* oforce_ ){
        natom=natom_; norb=norb_;
        apos=apos_;   aforce=aforce_;
        opos=opos_;   oforce=oforce_;
    }

    double eval( ){
        double invLbond = 1/lbond;
        //double cbondF   = orbFroceStrenght * invLbond; // lever length
        if(atomMap){
            for(int i=0; i<natom; i++){
                Vec3d gpos;
                Vec3d pa  = apos[i]-gridShape.pos0;
                gridShape.cartesian2grid( pa, gpos );
                Vec3d fg = interpolate3DvecWrap( atomMap, gridShape.n, gpos );
                aforce[i].add_mul( fg , atomFroceStrenght );
                #ifdef DEBUG_GL 
                //printf( "gridFF atom[%i] fg(%g,%g,%g)    gpos(%g,%g,%g)      pa(%g,%g,%g)  \n", i,   fg.x,fg.y,fg.z,   gpos.x,gpos.y,gpos.z,     pa.x, pa.y, pa.z );
                glColor3f(1.0,0.0,0.0); Draw3D::pointCross(apos[i],0.1); Draw3D::vecInPos( fg*100.0, apos[i] );
                #endif 
            }
        }
        
        if(bondMap){
            for(int ia=0; ia<natom; ia++){
                const Vec3d& pa = apos[ia] - gridShape.pos0;
                int ioff = ia*N_BOND_MAX;
                for(int io=0; io<N_BOND_MAX; io++){
                    Vec3d gpos; 
                    int i = ioff+io;
                    Vec3d p = pa + opos[i]*lbond;
                    gridShape.cartesian2grid( p, gpos );
                    Vec3d fg = interpolate3DvecWrap( bondMap, gridShape.n, gpos );

                    fg.mul( orbFroceStrenght );
                    //aforce[i].add( fg );
                    fg.mul( invLbond );
                    oforce[i].add( fg );

                    #ifdef DEBUG_GL 
                    p.add(gridShape.pos0);
                    glColor3f(1.0,1.0,1.0); Draw3D::pointCross(p,0.1); Draw3D::vecInPos( fg*100.0, p );
                    #endif
                }
            }
        }
        
        return 0; // ToDo : some energy ?
    }

};

GridFF_AtomBond gridff;



#ifdef  DEBUG_GL
void debug_draw_GridFF(GridShape gsh, Vec3d * data, bool bCmap, bool bLines ){
    //printf( " debug_draw_GridFF \n" );
    double z0  = 1.5;
    double dz0 = 0.1;
    for(int iy=1;iy<gsh.n.y;iy++){
        if(bCmap){
            glShadeModel(GL_SMOOTH);
            //glEnable( GL_POLYGON_SMOOTH);
            glBegin( GL_TRIANGLE_STRIP );
            for(int ix=0;ix<gsh.n.x;ix++){
                Vec3d p;
                int i = (iy-1)*gsh.n.x + ix;
                glColor3f ( data[i].x+0.5, data[i].y+0.5, 0.5 );
                //p = (gsh.dCell.a*(ix + (gsh.n.x*-0.5))) + (gsh.dCell.b*(iy-1 + (gsh.n.y*-0.5) ));
                p = gsh.dCell.a*ix + gsh.dCell.b*(iy-1) + gsh.pos0;
                glVertex3f(p.x,p.y,p.z+z0+dz0);

                i += gsh.n.x;
                glColor3f ( data[i].x+0.5, data[i].y+0.5, 0.5 );
                p.add(gsh.dCell.b);
                glVertex3f(p.x,p.y,p.z+z0+dz0);
            }
            glEnd();
        }
        if(bLines){
            glBegin(GL_LINES);
            glColor3f(1.0,1.0,1.0);
            double fsc = 0.1;
            for(int ix=0;ix<gsh.n.x;ix++){
                Vec3d p,v;
                p = gsh.dCell.a*ix + gsh.dCell.b*(iy-1) + gsh.pos0;
                Vec3d gpos;
                Vec3d pa  = p-gsh.pos0;
                gsh.cartesian2grid( pa, gpos );
                Vec3d fg = interpolate3DvecWrap( data, gsh.n, gpos );
                p.z+=z0;
                Draw3D::vecInPos( fg*fsc, p );
                Draw3D::pointCross( p, 0.01 );
            }
            glEnd();
        }
    }
}
#endif


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
    ff.guessBonds();
    //ff.normalizeOrbs();
    opt.bindOrAlloc( ff.nDOF, ff.dofs, 0, ff.fdofs, 0 );
    opt.cleanVel( );
    gridff.bindDOFs(ff.natom,ff.norb, ff.apos, ff.aforce, ff.opos, ff.oforce );
}

void setGridShape( int * n, double * cell ){
    //gridff.gridShape.n.set( n[2], n[1], n[0] );
    gridff.gridShape.n.set( n[0], n[1], n[2] );
    printf( " nxyz  %i %i %i \n", gridff.gridShape.n.x, gridff.gridShape.n.y, gridff.gridShape.n.z );
    gridff.gridShape.setCell( *(Mat3d*)cell );
    gridff.gridShape.pos0 = gridff.gridShape.cell.a*-0.5 + gridff.gridShape.cell.b*-0.5 + gridff.gridShape.cell.c*-0.5;
    gridff.gridShape.printCell();
}

void bindGrids( double* atomMap, double*  bondMap ){
    gridff.atomMap=(Vec3d*)atomMap;
    gridff.bondMap=(Vec3d*)bondMap;
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
     //printf( "relaxNsteps nsteps %i \n", nsteps );
    for(int itr=0; itr<nsteps; itr++){
        //printf( "relaxNsteps itr %i \n", itr );
        ff.cleanForce(); 
        gridff.eval();
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
            //printf( "DEBUG CHECK INVARIANTS  fsum %g torq %g   cog (%g,%g,%g) \n", fsum.norm(), torq.norm(), cog.x, cog.y, cog.z );
        }
        
        #ifdef DEBUG_GL
        debug_draw_GridFF(gridff.gridShape, gridff.atomMap, true, false );
        //debug_draw_GridFF(gridff.gridShape, gridff.bondMap, true, false );
        #endif 

        //for(int i=0; i<ff.natom; i++) ff.aforce[i].set(0.);
        
        switch(ialg){
            case 0: F2 = opt.move_FIRE();  break;
            case 1: opt.move_GD(opt.dt);   break;
            case 3: opt.move_MD(opt.dt);   break;
        }
        
        if(iDebug>0){ printf("relaxNsteps[%i] |F| %g(>%g) E %g  <v|f> %g dt %g(%g..%g) damp %g \n", itr, sqrt(F2), Fconv, E, opt.vf, opt.dt, opt.dt_min, opt.dt_max, opt.damping ); }
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

