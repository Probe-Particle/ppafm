
#include "PolyCycles.h"


double PolyCycleFF::rcyc[9] = 
      {0, 0,   // 0,1
       0.500000000, 0.577350269,               // 2,3
       0.707106781, 0.850650808,               // 4,5
       1.00000000 , 1.15238244 ,               // 6,7
       1.30656296                              // 8
    };
    

PolyCycleFF cff;

extern "C"{

//double* getVpos(){ return (double*)cff.vpos; }
//double* getCpos(){ return (double*)cff.cpos; }

double* getPos(){ return (double*)cff.opt.pos; }

int setup( int ncycles, int* nvs ){
    return cff.realloc(ncycles, nvs);
}

void init( double* angles ){
    cff.initVerts(angles);
}

// void initOpt( double dt_, double damp_ )

double setupOpt( double dt, double damp, double f_limit, double v_limit ){
    cff.opt.initOpt( dt, damp );
    cff.opt.v_limit = v_limit;
    cff.opt.f_limit = f_limit;
}


double relaxNsteps( int kind, int nsteps, double F2conf ){
    double F2=1.0;
    switch(kind){
        case 0: {
            //printf( "kind 0 \n" );
            cff.opt.n=cff.ncycles*2;
            for(int itr=0; itr<nsteps; itr++){
                /*
                //cff.cleanVertForce();
                cff.cleanCycleForce();
                cff.forceCC();
                //cff.forceVV();
                //cff.forceVC();
                //cff.moveVerts (dt,damp);
                cff.moveCycles(dt,damp);
                //if(F2<F2conf) break;
                //ff.moveMDdamp(dt, damp);
                */
                cff.opt.cleanForce();
                cff.forceCC();
                //double f = cff.opt.move_GD_safe(cff.opt.dt);
                //double f = cff.opt.move_MD_safe(cff.opt.dt);
                double f = cff.opt.move_FIRE(); f=sqrt(f);
                //printf( "itr %i dt %g scale_dt %g f %g \n", itr, cff.opt.dt, cff.opt.scale_dt, f );
            }
        }
        break;
        
        case 1: {
            //printf( "kind 1 \n" );
            cff.opt.n=(cff.ncycles+cff.nvert)*2;
            for(int itr=0; itr<nsteps; itr++){
                /*
                cff.cleanVertForce();
                cff.cleanCycleForce();
                cff.forceCC();
                cff.forceVV();
                cff.forceVC();
                cff.moveVerts (dt,damp);
                cff.moveCycles(dt,damp);
                //if(F2<F2conf) break;
                //ff.moveMDdamp(dt, damp);
                */
                
                cff.opt.cleanForce();
                cff.forceCC();
                //cff.forceVV();
                cff.forceVC();
                //double f = cff.opt.move_GD_safe(cff.opt.dt);
                //double f = cff.opt.move_MD_safe(cff.opt.dt);
                double f = cff.opt.move_FIRE(); f=sqrt(f);
                //printf( "itr %i dt %g scale_dt %g f %g \n", itr, cff.opt.dt, cff.opt.scale_dt, f );
            }
        }
        
        break;
    }
    return F2;
}

}
