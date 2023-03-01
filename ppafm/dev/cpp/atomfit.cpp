
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

#include <unordered_map>
#include <vector>

#include "Vec2.h"

//#include "Grid.h"

Vec2d dpix;
Vec2d invdpix;
Vec2i npix;
Vec2d* gridFF = 0;

int natom = 0;
Vec2d* pos    = 0;
Vec2d* vel    = 0;
Vec2d* force  = 0;

double rmin =  1.0;
double eps  = 0.01;
double C2   = 2*eps*rmin*rmin;
double C4   =   eps*rmin*rmin*rmin*rmin;

//std::unordered_map<int,Vec2d> known;
std::vector<int> known;

struct Particle2d{
    Vec2d pos;
    Vec2d vel;
    Vec2d force;

    void move(double dt, double damp){
        vel.mul(damp);
        vel.add_mul(force,dt);
        pos.add_mul(vel,dt);
    }
};

Particle2d point;


inline Vec2d interpolate2DWrap( Vec2d * grid, const Vec2i& n, const Vec2d& r ){
	int xoff = n.x<<3; int imx = r.x +xoff;	double tx = r.x - imx +xoff;	double mx = 1 - tx;		int itx = (imx+1)%n.x;  imx=imx%n.x;
	int yoff = n.y<<3; int imy = r.y +yoff;	double ty = r.y - imy +yoff;	double my = 1 - ty;		int ity = (imy+1)%n.y;  imy=imy%n.y;

    //printf( "(%g,%g) (%i/%i,%i/%i) (%g,%g) \n", r.x,r.y, imx,n.x, imy,n.y,   mx, my );
	Vec2d out;
	int i = n.x*imy + imx;
	out.set_mul( grid[ i ], my*mx );   out.add_mul( grid[ i+1 ], my*tx ); i+=n.x;
	out.add_mul( grid[ i ], ty*mx );   out.add_mul( grid[ i+1 ], ty*tx );

	return out;
}

void exterForce(){
    for(int i=0; i<natom; i++){
        //printf( "iatom %i (%g,%g) \n", i, pos[i].x,pos[i].y );
        force[i] = interpolate2DWrap( gridFF, npix, pos[i]*invdpix );
        //#printf(100);
       // printf( "iatom %i (%g,%g) (%g,%g) \n", i, pos[i].x,pos[i].y, force[i].x,force[i].y );
    }
};

void interForce(){
    double r2min = rmin*rmin;
    double ir2min = 1/r2min;
    for(int i=0; i<natom; i++){
        Vec2d pi = pos[i];
        for(int j=0; j<i; j++){
            Vec2d d = pos[j] - pi;
            double r2 = d.norm2();

            //--- LJ like
            /*
            double ir2 = 1/(r2+0.1);
            d.mul( ( C2 + C4*ir2 )*ir2*eps );
            */

            if(r2<r2min){
                d.mul( eps*(r2min-r2) );
                force[i].sub(d);
                force[j].add(d);
            }

        }
    }
};

double move(double dt, double damp){
    double F2max = 0;
    for(int i=0; i<natom; i++){

        Vec2d& vi       = vel  [i];
        const Vec2d& fi = force[i];

        vi.mul(damp);
        vi.add_mul(fi,dt);
        vel[i] = vi;
        pos[i].add_mul(vi,dt);

        //pos[i].add_mul(fi,dt);

        F2max = fmax(F2max,fi.norm2());
    }
    return F2max;
}

extern "C"{

void setGridFF(double* npix_, double* dpix_, double* gridFF_ ){
    npix = *(Vec2i*) npix_;
    dpix = *(Vec2d*) dpix_;
    invdpix.set_inv(dpix);
    gridFF = (Vec2d*)gridFF_;
}

void setAtoms(int natom_, double* pos_, double* vel_, double* force_ ){
    natom = natom_;
    pos   = (Vec2d*)pos_;
    vel   = (Vec2d*)vel_;
    force = (Vec2d*)force_;
}

void setParams(double eps_, double rmin_ ){
    eps  = eps_;
    rmin = rmin_;
    C2   = 2*eps*rmin*rmin;
    C4   =   eps*rmin*rmin*rmin*rmin;
}

bool relaxAtoms( int nstep, double dt, double damp, double F2conv ){
    for(int itr=0; itr<nstep;  itr++ ){
        //printf( "itr %i \n", itr );
        exterForce();
        interForce();
        double F2err = move(dt,damp);
        //printf();
        printf( "itr %i Ferr %g natom %i \n", itr, sqrt(F2err), natom );
        if(F2err/F2conv>1) return true;
    }
    return false;
}

bool relaxParticleNstep( int nstep, double dt, double damp ){
    //point.vel
    for(int itr=0; itr<nstep;  itr++ ){
        point.force = interpolate2DWrap( gridFF, npix, point.pos*invdpix );
        point.move(dt,damp);
    }
    return false;
}

int relaxParticlesUnique( int np, Vec2d* poss, int nstep, double dt, double damp, double Fconv ){
    double r2min=rmin*rmin;
    double F2conv = Fconv*Fconv;
    for(int i=0; i<np; i++){
        //printf( "ip %i \n", i );
        point.pos=poss[i];
        //bool isNew=true;
        double F2err = 1e+300;
        //printf( "i %i F2err %g F2conv %g \n",i, F2err, F2conv );
        while(F2err>F2conv){
            //printf( "F2err %g \n", F2err );
            relaxParticleNstep(nstep,dt,damp);
            F2err = point.force.norm2();

            for(int j : known){
                Vec2d d = point.pos-poss[j];
                double r2 = d.norm2();
                if(r2<r2min){ goto GOTO_1; }
            }

            //printf( "ip %i Ferr %g \n", i, F2err );
        }
        known.push_back(i);
        GOTO_1:
        poss[i]=point.pos;
    }

    int i=0;
    for( int j : known ){
        poss[i]=poss[j];
        i++;
    }

    return known.size();
}



int relaxParticlesRepel( int np, Vec2d* poss, int nstep, double dt, double damp, double Fconv ){
    double r2min=rmin*rmin;
    double F2conv = Fconv*Fconv;
    for(int i=0; i<np; i++){
        //printf( "ip %i \n", i );
        point.pos=poss[i];
        //bool isNew=true;
        double F2err = 1e+300;
        //printf( "i %i F2err %g F2conv %g \n",i, F2err, F2conv );
        for(int itr=0; itr<nstep;  itr++ ){
            //point.force = interpolate2DWrap( gridFF, npix, point.pos*invdpix );
            //point.force.set(0.0);
            point.force.set_mul(point.pos,-0.1);
            for(int j : known){
                Vec2d d = point.pos-poss[j];
                double r2 = d.norm2();
                if(r2<r2min){
                    d.mul( eps*(r2min-r2)/r2 );
                    point.force.sub(d);
                }
            }

            point.move(dt,damp);

            F2err = point.force.norm2();
            if(F2err<F2conv) break;
        }
        poss[i]=point.pos;
        known.push_back(i);
    }
    return known.size();
}


} // extern "C"{
