
#ifndef PolyCycles_h
#define PolyCycles_h

#include "fastmath_light.h"
#include "Vec2.h"
#include "DynamicOpt.h"

/*
TODO: FastSquare Root
F_rSpring      = d*K*(r-r0)/r = d*K*( 1 - r0/r )
F_fastRspring  = d*K*(r-r0)/r = d*K*( 1 -  )
inline fastRspring( double r2 ){}
*/

class PolyCycleFF{ public:
/*
    constexpr static Vec2d drots[8] = {  
       { 0.0,  0.0}, // 0
       {-1.0,  0.0}, // 1
       { 0.0,  1.0}, // 2
       { 0.500000000,  0.866025404}, // 3
       { 0.707106781,  0.707106781}, // 4
       { 0.809016994,  0.587785252}, // 5
       { 0.866025404,  0.500000000}, // 6
       { 0.900968868,  0.433883739}, // 7
       { 0.923879533,  0.382683432}  // 8
    };
*/

    static double rcyc[9];
   // constexpr static const double rcyc[9];
    /*
      =  {0, 0,   // 0,1
       0.500000000, 0.577350269,               // 2,3
       0.707106781, 0.850650808,               // 4,5
       1.00000000 , 1.15238244 ,               // 6,7
       1.30656296                              // 8
    };
    */

    DynamicOpt opt;

    int nvert=0;
    //int*  verts  =0;    // use this later for re-ordering
    Vec2d* vpos  =0;
    Vec2d* vvel  =0;
    Vec2d* vforce=0;
    
    int ncycles  =0;
    //Cyclus* cycles;
    int*   nvs    =0;
    int*   ivs    =0;
    Vec2d* cpos   =0;
    Vec2d* cvel   =0;
    Vec2d* cforce =0;
    
    int voffset = 0;
    
    double Ecc      =0.2;
    double Kvert    =4.0;
    double Kbond    =5.0;
    double Kcenter  =5.0;
    double RvertMax =0.5; 
    
    int realloc(int ncycles_, int* nvs_ ){
        ncycles=ncycles_;
        /*
        _realloc(cpos,  ncycles );
        _realloc(cvel,  ncycles );
        _realloc(cforce,ncycles );
        //_realloc(nvs);
        */
        nvs=nvs_;
        _realloc(ivs, ncycles );
        int iv=0;
        for(int i=0; i<ncycles; i++){
            //printf( "i %i nvi %i \n", i, nvs[i] );
            ivs[i]=iv;
            iv+=nvs[i];
        }
        nvert=iv;
        /*
        _realloc(vpos,  nvert );
        _realloc(vvel,  nvert );
        _realloc(vforce,nvert );
        */
        voffset = ncycles*2;
        opt.realloc( (ncycles+nvert) * 2);
        //opt.initMass(1.0);
        for(int i=0;      i<ncycles; i++){ ((Vec2d*)opt.invMasses)[i].set(1.0/nvs[i]); }
        for(int i=voffset;i<opt.n;   i++){ opt.invMasses[i]=1.0; }
        cpos  =(Vec2d*)opt.pos;
        cvel  =(Vec2d*)opt.vel;
        cforce=(Vec2d*)opt.force;
        vpos  =(Vec2d*)(opt.pos   + voffset);
        vvel  =(Vec2d*)(opt.vel   + voffset);
        vforce=(Vec2d*)(opt.force + voffset);
        //printf("nvert %i %i \n", nvert, iv);
        return nvert;
    }
    
    void initVerts(double* angles=0){
        int iv =0;
        for(int ic=0; ic<ncycles; ic++){
            double phi0;
            if(angles){ phi0=angles[ic];     }
            else      { phi0=randf()*M_PI*2; }
            int   nv     = nvs[ic];
            const Vec2d& pc = cpos[ic];
            double R     = rcyc[nv];
            double dphi  = 2*M_PI/nv;
            Vec2d drot; drot.fromAngle(dphi);
            //rmax=0.5/drot.y; // radius
            //rmin=rmax*drot.x;
            //printf( "n %i (%g,%g) %g %g \n" , n, drot.x, drot.y, rmax, rmin );
            //printf( "ic %i n %i pc(%g,%g) drot(%g,%g) %g \n" , ic, nv, pc.x, pc.y, drot.x, drot.y, R );
            Vec2d rot; rot.fromAngle(phi0);
            for(int i=0; i<nv; i++){
                vpos[iv] = pc + rot*R;
                rot.mul_cmplx(drot);
                vvel[iv].set(0.0);
                iv++;
            }
            cvel[ic].set(0.0);
        }
    }
    
    
    void forceVV(){
        double RvertMax2 = RvertMax*RvertMax;
        double K = Kvert/RvertMax2;
        //double K = Kvert;
        for(int i=1; i<nvert; i++){
            const Vec2d& pi=vpos  [i];
            Vec2d&       fi=vforce[i];
            for(int j=0; j<i; j++){
                Vec2d d=vpos[j]-pi;
                double r2 = d.norm2();
                if(r2<RvertMax2){
                    double fr = K*(RvertMax2-r2);
                    printf( "%i,%i %g %g \n", i, j, sqrt(r2), fr*sqrt(r2) );
                    d.mul( fr );
                    fi       .add(d);
                    vforce[j].sub(d);
                }
            }
        }
    }
    
    void forceCC(){
        for(int i=1; i<ncycles; i++){
            const Vec2d& pi= cpos  [i];
            Vec2d&       fi= cforce[i];
            double Ri      = rcyc[nvs[i]]; 
            for(int j=0; j<i; j++){
                double R = (Ri + rcyc[nvs[j]])*1.2;
                //R=2.0;
                double R2 = R*R;
                Vec2d d=cpos[j]-pi;
                double r2 = d.norm2();
                double ir2  = 1/(r2+0.01);
                //double c2 = R2*2;
                //double c4 = R2*r2;
                //double E  = Ecc*(2-R2*ir2)*R2*ir2;
                double fr = Ecc*4*(1-R2*ir2)*R2*ir2*ir2;
                //printf( "%i,%i   %g - %g =  %g     %g %g : %g %g \n", i, j, pi.x, cpos[j].x, d.x,    R, sqrt(r2), E, fr );
                //printf( "%i,%i %g %g %g \n", i, j, R, sqrt(r2), fr );
                d.mul( fr );
                fi       .add(d);
                cforce[j].sub(d);
                
                if(r2<(2*R2)){
                    forceCpair(i,j);
                }
            }
        }
    }
    
    void forceCpair(int ic, int jc){
        int nvi = nvs[ic];
        int nvj = nvs[jc];
        int i0  = ivs[ic];
        int j0  = ivs[jc];
        
        double RvertMax2 = RvertMax*RvertMax;
        double K = Kvert/RvertMax2;
        //double K = Kvert;
        
        // verts
        /*
        for(int i=0; i<nvi; i++){
            int iv=i0+i;
            const Vec2d& pi=vpos  [iv];
            for(int j=0; j<nvj; j++){
                int jv=j0+j;
                Vec2d d=vpos[jv]-pi;
                double r2 = d.norm2();
                if(r2<RvertMax2){
                    double fr = K*(RvertMax2-r2);
                    printf( "%i,%i %g %g \n", iv, jv, sqrt(r2), fr*sqrt(r2) );
                    d.mul( fr );
                    vforce[iv].add(d);
                    vforce[jv].sub(d);
                }
            }
        }
        */
        
        // edges
        /*
        int oiv=i0+nvi-1;
        for(int i=0; i<nvi; i++){
            int iv =i0+i;
            int ojv=j0+nvj-1;
            Vec2d pi = (vpos[iv]+vpos[oiv])*0.5;
            for(int j=0; j<nvj; j++){
                int jv=j0+j;
                Vec2d d=(vpos[jv]+vpos[ojv])*0.5-pi;
                double r2 = d.norm2();
                if(r2<RvertMax2){
                    double fr = K*(RvertMax2-r2);
                    //printf( "%i,%i %g %g \n", iv, jv, sqrt(r2), fr*sqrt(r2) );
                    d.mul( fr );
                    vforce[ iv].add(d);
                    vforce[oiv].add(d);
                    vforce[ jv].sub(d);
                    vforce[ojv].sub(d);
                }
                ojv=jv;
            }
            oiv=iv;
        }
        */
        
        // Edge+Vert
        int oiv=i0+nvi-1;
        for(int i=0; i<nvi; i++){
            int iv =i0+i;
            int ojv=j0+nvj-1;
            const Vec2d& pi = vpos[iv];
            Vec2d pie = (pi+vpos[oiv])*0.5;
            for(int j=0; j<nvj; j++){
                int jv=j0+j;
                Vec2d d;
                double r2;
                const Vec2d& pj = vpos[jv];
                d=pj-pi;
                r2 = d.norm2();
                if(r2<RvertMax2){
                    double fr = K*sq(RvertMax2-r2);
                    //printf( "%i,%i %g %g \n", iv, jv, sqrt(r2), fr*sqrt(r2) );
                    d.mul( fr );
                    vforce[ iv].add(d);
                    vforce[ jv].sub(d);
                }
                d=(pj+vpos[ojv])*0.5-pie;
                r2 = d.norm2();
                if(r2<RvertMax2){
                    double fr = K*sq(RvertMax2-r2);
                    //printf( "%i,%i %g %g \n", iv, jv, sqrt(r2), fr*sqrt(r2) );
                    d.mul( fr );
                    vforce[ iv].add(d);
                    vforce[oiv].add(d);
                    vforce[ jv].sub(d);
                    vforce[ojv].sub(d);
                }
                ojv=jv;
            }
            oiv=iv;
        }
    }

    void forceVC(){
        //Vec2d* vps=vpos;
        //Vec2d* vfs=vforce;
        int iv =0;
        for(int ic=0; ic<ncycles; ic++){
            int nv=nvs[ic];
            double R = rcyc[nv];
            //int oi    = verts[i-1];
            //Vec2d* op = vps+oi;
            //Vec2d* of = vfs+oi; 
            Vec2d* op = vpos  +iv+nv-1;
            Vec2d* of = vforce+iv+nv-1; 
            const Vec2d& pos   = cpos  [ic];
            Vec2d&      force = cforce[ic];
            //printf( "ic %i nv %i \n", ic, nv );
            //for(int ii=0; ii<nv; ii++){
            for(int ii=0; ii<nv; ii++){
                //int i = verts[ii];
                Vec2d* pi = vpos  +iv;
                Vec2d* fi = vforce+iv;
                Vec2d d;
                // bond force
                {
                    d.set_sub(*pi,*op);
                    double r = d.norm(); // TODO: fast square root approx
                    d.mul( -Kbond*(1-r)/r );
                    of->add(d);
                    fi->sub(d);
                }
                // force to centre
                {
                    d.set_sub(*pi,pos); 
                    double r = d.norm(); // TODO: fast square root approx
                    double fr = Kcenter*(R-r)/r;
                    //printf( "fr(%i,%i) \n",ic,ii, fr );
                    d.mul( fr );
                    fi   ->add(d);
                    force .sub(d);
                }
                op=pi; of=fi;
                iv++;
            }
            //vps+=nv;
            //vfs+=nv;
        }
    }

    void cleanVertForce (){ for(int i=0; i<nvert;   i++){vforce[i].set(0.0); } }
    void cleanCycleForce(){ for(int i=0; i<ncycles; i++){cforce[i].set(0.0); } }
    void cleanVertVel   (){ for(int i=0; i<nvert;   i++){vvel[i]  .set(0.0); } }
    void cleanCycleVel  (){ for(int i=0; i<ncycles; i++){cvel[i]  .set(0.0); } }

    void moveCycles(double dt,double damp){
        for(int i=0; i<ncycles; i++){
            
            Vec2d& vi = cvel[i];
            vi.mul(damp);
            vi.add_mul(cforce[i],dt);
            cpos[i].add_mul( vi, dt);
            
            //cpos[i].add_mul( cforce[i], dt );
        }
    }
    
    void moveVerts(double dt,double damp){
        for(int i=0; i<nvert; i++){
            Vec2d& vi = vvel[i];
            vi.mul(damp);
            vi.add_mul(vforce[i],dt);
            vpos[i].add_mul( vi, dt );
        }
    }
    
    void step(double dt, double damp){
        forceCC();
        forceVV();
        forceVC();
        moveVerts(dt,damp);
        moveCycles(dt,damp);
    }

};

#endif
