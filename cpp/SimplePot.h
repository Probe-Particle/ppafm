

#ifndef SimplePot_h
#define SimplePot_h

#include <math.h>
#include "Vec3.h"
#include <stdio.h>

#include "fastmath.h"

//#define GOLDEN_RATIO

// https://en.wikipedia.org/wiki/Regular_icosahedron#Cartesian_coordinates
constexpr double IcosA = 0.52573111211; // 1 / |R|
constexpr double IcosB = 0.85065080835; // golden_ratio / |R|
const static int   nIcos = 12;
const static Vec3d Icosahedron_verts[nIcos] = {
    {0.,-IcosA,-IcosB}, // 0
    {0.,-IcosA,+IcosB}, // 1
    {0.,+IcosA,-IcosB}, // 2
    {0.,+IcosA,+IcosB}, // 3
    {-IcosA,-IcosB,0.}, // 4
    {-IcosA,+IcosB,0.}, // 5
    {+IcosA,-IcosB,0.}, // 6
    {+IcosA,+IcosB,0.}, // 7
    {-IcosB,0.,-IcosA}, // 8
    {+IcosB,0.,-IcosA}, // 9
    {-IcosB,0.,+IcosA}, // 10
    {+IcosB,0.,+IcosA}, // 11
};

/*
struct Bond{
    int    i,j;
    Vec3d  h;
    double l;
};
*/

inline void minHarmonic( double s, double& E ){
    double e = s*s-1;
    if(e<E)E=e;
}

inline double evalAngular( double ca, int nb, double invW ){
    double ang = acos(ca); // ToDo: this can be made more efficient
    double E = 0;
    if(nb<2){ minHarmonic( (ang-(M_PI              ))*invW, E ); }  // sp1
    if(nb<3){ minHarmonic( (ang-(M_PI*0.66666666666))*invW, E ); }  // sp2
    if(nb<4){ minHarmonic( (ang-(M_PI*0.60833333333))*invW, E ); }  // sp3
    return E;
}

class SimplePot{ public:
    int natom=0;
    int neighPerAtom=4;
    //std::vector<Bond> bonds;

    Vec3d*  apos=0;
    int*    nepairs=0;
    int*    nneighs=0;
    Vec3d*  neighs=0;
    double* Rcovs=0;  /// [A] covant radius
    double* RvdWs=0; /// [A] van der Waals radius

    double barrierWidth = 1.0; // [A]
    double covWidth     = 0.2; // [A]   width of covalent bond range
    double angWidth     = 0.3; // [rad]
    double covAmp       = 1.0;
    double coreAmp      = 5.0;
    double barrierAmp   = 0.3;

    // ========== Functions

    void makeNeighbors(){
        for(int i=0; i<natom; i++){ nneighs[i]=0; };
        for(int i=0; i<natom; i++){
            Vec3d  pi = apos[i];
            double Ri = Rcovs[i] + covWidth;
            for(int j=i+1; j<natom; j++){
                Vec3d d  = apos[j] - pi;
                double R = Rcovs[j] + Ri;
                double r2 = d.norm2();
                if( r2<(R*R) ){
                    double r = sqrt(r2);
                    d.mul(1/r);
                    //bonds.push_back( (Bond){i,j,d,r} );
                    int& nbi = nneighs[i];
                    int& nbj = nneighs[j];
                    if( (nbi>=neighPerAtom) || (nbj>=neighPerAtom) ){
                        printf( "ERROR in SimplePot::makeNeighbors() :\n Atoms has >%i bonds nbi[%i]=%i nbj[%i]=%i \n", neighPerAtom, i, nbi, j, nbj );
                        exit(0);
                    }
                    neighs[i*neighPerAtom+nbi]=d;
                    neighs[j*neighPerAtom+nbj]=d*-1;
                    nneighs[i]++;
                    nneighs[j]++;
                }
            }
        }
        //printNeighs();
    }

    void printNeighs(){
        for(int i=0; i<natom; i++){
            printf( "nnegihs[%i] %i\n", i, nneighs[i] );
        }
    }

    double eval( Vec3d pos, double Rcov, double RvdW ){
        // Rcov = 0.7;
        // covWidth = 0.4;
        const double invCovWidth = 1/covWidth;
        const double invAngWidth = 1/angWidth;
        double Etot=0;
        for(int i=0; i<natom; i++){
            int nepair=0;
            if(nepairs)nepair=nepairs[i];
            Vec3d  dp = pos  - apos [i];
            double  Rb = Rcov + Rcovs[i];
            double  R  = Rb + barrierWidth;
            double r2  = dp.norm2();
            // bonds
            if( r2<(R*R) ){
                double r  = sqrt(r2);
                // ToDo : barrier energy
                double sb = (r-Rb)*invCovWidth;
                double Eb = (1-sb*sb)*covAmp;
                if( Eb>0 ){
                    int nb = nneighs[i];
                    dp.mul(1/r);
                    double Eang = -1.0;
                    int j0=i*neighPerAtom;
                    for(int j=0; j<nb; j++){
                        double e = evalAngular( dp.dot(neighs[j0+j]), nb+nepair, invAngWidth );
                        if( e>Eang ) Eang = e;
                    }
                    Eb*=Eang;
                }else{
                    Eb=0;
                }
                double Rc = Rb-covWidth;
                double rc = Rc - r;
                if     (rc>0){ Etot += rc*coreAmp + barrierAmp; }
                else         { Etot += (1-Eb)*barrierAmp*(R-r)/(R-Rc); }
                Etot+=Eb;
            }
        }
        return Etot;
    }

    int danglingToArray( Vec3d* dangs, double Rcov, Vec3d pmin, Vec3d pmax, int ncircle=6 ){
        int n=0;
        for(int i=0; i<natom; i++){
            Vec3d  pi   = apos [i];
            double R    = Rcovs[i] + Rcov;
            //Vec3d  margin; margin.set(R);
            //if( ! pi.isBetween( pmin-margin, pmax+margin ) ) continue;
            int nb = nneighs[i];
            int j0=i*neighPerAtom;
            switch(nb){
                case 0: { // sphere - ToDo later
                    for(int i=0; i<nIcos; i++){
                        dangs[n] = pi + Icosahedron_verts[i]*R;
                    }
                } break;
                case 1: {    //
                    Vec3d d = neighs[j0];
                    Vec3d u,s; d.getSomeOrtho(u,s);
                    Vec2d rot=Vec2dX;
                    Vec2d drot; drot.fromAngle( 2*M_PI/ncircle );
                    dangs[n]= pi + d*-R; n++;
                    for(int k=0; k<ncircle; k++){
                        dangs[n] = pi + ( d*0.57714519003 + u*(rot.x*0.81664155516) + s*(rot.y*0.81664155516) )*-R;
                        n++;
                        rot.mul_cmplx(drot);
                    }
                } break;
                case 2:{
                    Vec3d d = (neighs[j0]+neighs[j0+1]); d.normalize();
                    Vec3d u = (neighs[j0]-neighs[j0+1]); u.normalize();
                    Vec3d s; s.set_cross(u,d);
                    dangs[n]= pi+(d*0.57714519003+s*-0.81664155516)*-R; n++; // sp3
                    dangs[n]= pi+(d*0.57714519003+s*+0.81664155516)*-R; n++; // sp3
                    dangs[n]= pi+d*-R; n++; // sp2
                } break;
                case 3:{
                    Vec3d d = (neighs[j0]-neighs[j0+1]); d.normalize();
                    Vec3d u = (neighs[j0]-neighs[j0+2]); u.makeOrthoU(d);  u.normalize();
                    Vec3d s; s.set_cross(u,d);
                    dangs[n]=pi+s* R; n++;
                    dangs[n]=pi+s*-R; n++;
                } break;
            }
        }
        return n;
    }

};

#endif
