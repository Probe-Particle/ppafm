
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "Vec3.h"
#include "quaternion.h"
#include "Mat3.h"
//#include <string.h>

//int iDebug = 1;
int iDebug = 0;

#include "Forces.h"
#include "DynamicOpt.h"

const double const_eVA_SI = 16.0217662;

// =====================================================
// ==========   Export these functions ( to Python )
// ========================================================

inline void rotateZBond( const Vec3d& dir, const Vec3d& coefsIn, Vec3d& coefsOut ){
    // to rotate coefficients from molecular coordinate system, to absolute cartesian system
    if( (dir.x*dir.x+dir.y*dir.y) < 1e-6 ){ // direction along z-axis ? 
        coefsOut=coefsIn*dir.z;
        return;
    }
    Mat3d rot;
    rot.fromDirUp( dir, {0,0,1} );
    Vec3d cbak = coefsIn; // this is in case &coefsIn=&coefsOut to prevent messing up
    rot.dot_to_T( cbak, coefsOut );
    //printf( "rotateZBond  (%g,%g,%g)->(%g,%g,%g) | dir (%g,%g,%g)  \n", cbak.x,cbak.y,cbak.z,  coefsOut.x,coefsOut.y,coefsOut.z, dir.x,dir.y,dir.z  );
}

class COCOFF{ public:

    bool bInterSample = false;
    bool bZRot        = true;

    double C6  = 0;
    double C12 = 0;

    double beta = -1.0;

    double lRadial = 4.0;
    double kRadial = 30.0 * const_eVA_SI;
    Vec3d  kSpring = (Vec3d){0.25*const_eVA_SI, 0.25*const_eVA_SI, 0};

    //Vec3d PPanchor;
    //Vec3d PPpos;
    //Vec3d PPvel;
    //Vec3d PPforce;

    int nCOs;
    Vec3d* anchors = 0;
    Vec3d* poss    = 0;
    Vec3d* vels    = 0;
    Vec3d* forces  = 0;

    Quat4d* STMcoefs = 0;

    void setC612(double R0, double E0){
        double r3 = R0*R0*R0;
        double r6 = r3*r3;
        C6        = 2*E0*r6;
        C12       =   E0*r6*r6;
        printf( " R0 %g E0 %g -> C6 %g C12 %g \n", R0, E0, C6, C12 );
    }

    void realloc(int nCOs_){
        nCOs = nCOs_;
        //nCOs++;
        nCOs_++;
        _realloc( anchors,  nCOs_ );
        _realloc( poss,     nCOs_ );
        _realloc( vels,     nCOs_ );
        _realloc( forces,   nCOs_ );
        _realloc( STMcoefs, nCOs_ );
    }

    double evalFF( bool bSprings ){
        double E = 0;
        Vec3d& PPanchor = anchors[nCOs];
        Vec3d& PPpos    = poss   [nCOs];
        Vec3d& PPforce  = forces [nCOs];
        PPforce = Vec3dZero;
        Vec3d dR;
        if(bSprings){
            dR.set_sub( PPanchor, PPpos );
            PPforce.add( forceRSpring( dR, kRadial, lRadial ) );
            PPforce.add_mul     ( dR, kSpring );
        }
        //printf( "PP %i p(%g,%g,%g) anchor(%g,%g,%g)\n", PPpos.x,PPpos.y,PPpos.z, nCOs,  PPanchor.x,PPanchor.y,PPanchor.z );
        for(int i=0; i<nCOs; i++){
            Vec3d f = Vec3dZero;
            //Vec3d& f = COforces[i]; f=Vec3dZero; 
            dR.set_sub( PPpos, poss[i] );
            //printf( "|dR| %g \n", dR.norm() );
            E += addAtomLJ ( dR, f, C6, C12 );
            //printf( "CO[%i] p(%g,%g,%g) dR(%g,%g,%g) f(%g,%g,%g)\n",   i,   poss[i].x,poss[i].y,poss[i].z,   dR.x,dR.y,dR.z,   f.x,f.y,f.z );
            PPforce.sub(f);
            if(bSprings){
                dR.set_sub( anchors[i], poss[i] );
                f.add( forceRSpring( dR, kRadial, lRadial ) );
                f.add_mul          ( dR, kSpring );
            }
            forces[i] = f;
            // ToDo : spring energy is missing
        }
        //printf( "PPforce (%g,%g,%g) \n", PPforce.x, PPforce.y, PPforce.y );
        //ToDo: if( bInterSample )
        return E;
    }



    double evalSTM(){

        const Vec3d&  PPanchor = anchors [nCOs];
        const Vec3d&  PPpos    = poss    [nCOs];
        Vec3d PPdir; PPdir.set_sub( PPpos, PPanchor );
        PPdir.normalize();
        double amp = 0;
        
        Quat4d  PPcoefs;
        if(bZRot){   // rotate orbital from z-direction
            PPcoefs.s=PPcoefs.s;
            rotateZBond( PPdir, STMcoefs[nCOs].p, PPcoefs.p );
        }else{ PPcoefs=STMcoefs[nCOs]; }

        for(int i=0; i<nCOs; i++){
            //const Quat4d& COcoefs  = STMcoefs[i];

            // bond vectror and radial hopping
            Vec3d d; d.set_sub( poss[i], PPpos ); // from PP->Samp
            double r      = d.normalize();
            double radial = exp(beta*r);

            // rotate sp-orbital coeficents to bond-oriented coordinate system 
            Mat3d rot;  rot.fromDirUp( d, PPdir );
            Vec3d pPP,pCO;

            Vec3d COdir; COdir.set_sub( poss[i], anchors[i] );
            COdir.normalize();
            Quat4d  COcoefs;
            if(bZRot){   // rotate orbital from z-direction
                COcoefs.s=STMcoefs[i].s;
                rotateZBond( COdir, STMcoefs[i].p, COcoefs.p );
            }else{ COcoefs=STMcoefs[nCOs]; }

            rot.dot_to( PPcoefs.p,     pPP );
            rot.dot_to( STMcoefs[i].p, pCO );

            //Vec3d COdir; COdir.set_sub( poss[i], anchors[i] );
            //COdir.normalize();
            //double cosa = COdir.dot( PPdir );
            //const Quat4d& COcoefs  = STMcoefs[i];

            //printf( "evalSTM  r %g Yr %g (%g,%g,%g,%g) | (%g,%g,%g) (%g,%g,%g) \n",  r, radial,   PPcoefs.s*STMcoefs[i].s,   pPP.x*pCO.x, pPP.y*pCO.y, pPP.z*pCO.z,   pPP.x,pPP.y,pPP.z,   pCO.x,pCO.y,pCO.z );
            
            double ss = COcoefs.s*PPcoefs.s;
            double zz = pCO.z*pPP.z;
            double sz = COcoefs.s*pPP.z - pCO.z*PPcoefs.s;
            double yy = pCO.y*pPP.y;
            double xx = pCO.x*pPP.x;
            double ang = ss + xx + yy - zz;
            
            //printf( "evalSTM  Y %g | (%g,%g,%g) (%g,%g,%g) \n", ang,  pPP.x,pPP.y,pPP.z,  pCO.x,pCO.y,pCO.z );
            amp +=  radial * ang;
            //amp += ang;
        }
        //printf( " amp %g  \n", amp );
        return amp;
    }

};

COCOFF      ff;
DynamicOpt  opt;

extern "C"{

    void init(int n){
        ff.realloc(n);
        opt.bindOrAlloc( (ff.nCOs+1)*3, (double*)ff.poss, (double*)ff.vels, (double*)ff.forces, 0 );
        printf( " =================== init natom n %i  nCO %i opt.n %i \n", n, ff.nCOs, opt.n );
    }

    //int*    getTypes (){ return (int*)   ff.atypes; }
    double* getPoss   (){ return (double*)ff.poss;    }
    double* getAnchors(){ return (double*)ff.anchors; }
    double* getForces (){ return (double*)ff.forces;  }
    double* getVels   (){ return (double*)ff.vels;    }
    double* getSTMCoefs  (){ return (double*)ff.STMcoefs;  }

    double setFF( double R0, double E0, double lR, double kR, double kxy, double beta ){
        ff.kSpring = (Vec3d){kxy,kxy,0};
        ff.kRadial = kR;
        ff.lRadial = lR;
        ff.beta    = beta;
        ff.setC612( R0, E0 ); 
    }

    double setupOpt( double dt, double damp, double f_limit, double l_limit ){
        opt.initOpt( dt, damp );
        opt.f_limit = f_limit;
        opt.l_limit = l_limit;
    }

    int relaxNsteps( int nsteps, double Fconv, int ialg ){
        double F2 = 1e+8;
        double F2conv=Fconv*Fconv;
        //for(int i=0; i<opt.; i++)
        opt.cleanVel();
        int itr;
        for(itr=0; itr<nsteps; itr++){
            //printf( " relaxNsteps %i %i \n", nsteps, itr );
            opt.cleanForce();
            double E = ff.evalFF( true );
            //printf( "ialng %i ff.n %i opt.n %i \n", ialg, ff.nCOs, opt.n  );
            switch(ialg){
                case 0: F2 = opt.move_FIRE(); break;
                case 1: opt.move_GD(opt.dt);  break;
                case 3: opt.move_MD(opt.dt);  break;
            }
            if(iDebug>0){ printf(":::relaxNsteps[%i] |F| %g(>%g) E %g  <v|f> %g dt %g(%g..%g) damp %g \n", itr, sqrt(F2), Fconv, E, opt.vf, opt.dt, opt.dt_min, opt.dt_max, opt.damping ); }
            if(F2<F2conv){
                //printf( "relaxed in %i iterations \n", itr ); 
                break;
            }
        }
        return itr;
    }

    double getSTM(){ return ff.evalSTM(); }

    void scan( int np, double* tip_pos_, double* PP_pos_, double* tip_forces_, double* STMamp, int nsteps, double Fconv, int ialg ){
        Vec3d* tip_pos    = (Vec3d*)tip_pos_;
        Vec3d* PP_pos     = (Vec3d*)PP_pos_;
        Vec3d* tip_forces = (Vec3d*)tip_forces_;
        for(int ip=0; ip<np; ip++){
            Vec3d oldP = ff.anchors[ff.nCOs];
            ff.anchors[ff.nCOs] = tip_pos[ip];
            ff.poss[ff.nCOs].add(ff.anchors[ff.nCOs]-oldP);
            if(ialg>=0) relaxNsteps( nsteps, Fconv, ialg );
            ff.evalFF(false);
            PP_pos    [ip] = ff.poss  [ff.nCOs];
            tip_forces[ip] = ff.forces[ff.nCOs];
            STMamp[ip] = ff.evalSTM(); 
        }

    }

} // extern "C"{



