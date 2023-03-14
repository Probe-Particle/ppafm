/*
Non-Bonded Force-Field
 should be easily plugged with any molecular dynamics, by either sharing pointer to same data buffer, or by copying data
*/

#ifndef NBFF_h
#define NBFF_h

//#include "Forces.h"

bool checkPairsSorted( int n, Vec2i* pairs ){
    int ia=-1,ja=-1;
    for(int i=0;i<n; i++){
        const Vec2i& b = pairs[i];
        //printf( "pair[%i] %i,%i | %i %i  | %i %i %i ", i, b.i, b.j,   ia,ja ,   b.i>=b.j,  b.i<ia, b.j<=ja );
        if(b.i>=b.j){ return false; }
        if(b.i<ia)  { return false; }
        else if (b.i>ia){ia=b.i; ja=-1; };
        if(b.j<=ja){ return false; }
        ja=b.j;
    }
    return true;
}


inline void combineREQ(const Vec3d& a, const Vec3d& b, Vec3d& out){
    out.a=a.a+b.a; // radius
    out.b=a.b*b.b; // epsilon
    out.c=a.c*b.c; // q*q
}

inline double addAtomicForceLJQ( const Vec3d& dp, Vec3d& f, const Vec3d& REQ ){
    //Vec3f dp; dp.set_sub( p2, p1 );
    const double COULOMB_CONST = 14.3996448915d;  //  [V*A/e] = [ (eV/A) * A^2 /e^2]
    double ir2  = 1/( dp.norm2() + 1e-4 );
    double ir   = sqrt(ir2);
    double ir2_ = ir2*REQ.a*REQ.a;
    double ir6  = ir2_*ir2_*ir2_;
    //double fr   = ( ( 1 - ir6 )*ir6*12*REQ.b + ir*REQ.c*-COULOMB_CONST )*ir2;
    double Eel  = ir*REQ.c*COULOMB_CONST;
    double vdW  = ir6*REQ.b;
    double fr   = ( ( 1 - ir6 )*12*vdW - Eel )*ir2;
    //printf( " (%g,%g,%g) r %g fr %g \n", dp.x,dp.y,dp.z, 1/ir, fr );
    f.add_mul( dp, fr );
    return  ( ir6 - 2 )*vdW + Eel;
}


/*  // ---------- Not yet implemented and tested
// =========== SR repulsion functions

//   good SR repulsion function should have spike in the center ... this is the case of Lorenz
//   (R2-r2)^2
//   (R2-r2)*((R2/r2)-1)^2
//   ((R2/r2)-1)^2
//   (((R2+w2)/(r2+w2))-1)^2

inline double addForceR2( const Vec3d& dp, Vec3d& f, double R2, double K ){
    double r2 = dp.norm2();
    if(r2<R2){
        double q  = R2-r2;
        f.add_mul( dp, 4*K*q );
        return K*q*q;
    }
    return 0;
}

inline double addForceR2inv( const Vec3d& dp, Vec3d& f, double R2, double K, double w2 ){
    //  E =   K*(1-R2/r2)^2
    //  f = 4*K*(1-R2/r2)*(R2/(r2*r2))*x
    //  k = 4*    5R4/r6 - 3R2/r4
    //  k(r=R) 4*(5      - 3 )/r2 = 8/R2
    //  => K_ = K*R2/8
    double r2 = dp.norm2();
    if(r2<R2){
        double R2_ = R2+w2;
        double K_  = R2_*0.125;
        double ir2 = 1/(r2+w2);
        double q   = R2_*ir2-1;
        f.add_mul( dp, K_*q*ir2 );
        return K_*q*q;
    }
    return 0;
}


inline double addForceR2mix( const Vec3d& dp, Vec3d& f, double R2, double K, double w2 ){
    //  E =   K*(R2-r2)(1-R2/r2)
    //  f = 2*K*(1-(R2/r2)^2)*x
    //  k = 2*K*(3*R4/r4-1)
    //  k(r=R) 4*K
    double r2 = dp.norm2();
    if(r2<R2){
        double K_  = K*0.5;
        double R2_ = R2+w2;
        double q   = R2_/(r2+w2);
        f.add_mul( dp, K_*(1-q*q) );
        return K_*q*(R2-r2)*0.5;
    }
    return 0;
}
*/



// ================ THE CLASS    *** NBFF ***


class NBFF{ public:

    //static int iDebug = 0;

// non bonded forcefield
    int n       = 0;
    int nmask   = 0;
    Vec3d* REQs = 0;
    Vec3d* ps   = 0;
    Vec3d* fs   = 0;
    Vec2i* pairMask = 0; // should be ordered ?


/*
    // --------- not yet implemented and tested
    double sr_w      = 0.1;
    double sr_dRmax  = 1.0; // maximum allowed atom movement before neighborlist update
    double sr_K      = 1.0; // collision stiffness
    double sr_Rscale = 0.5; // rescale atoms for short range
    Vec3d* psBack = 0;   // backup positon for neighbor list
    std::vector<Vec2i> neighList; // possibly too slow ?
*/

void realloc(int n_, int nmask_){
    n=n_;
    nmask=nmask_;
    _realloc(REQs,n);
    _realloc(ps  ,n);
    _realloc(fs  ,n);
    _realloc(pairMask ,nmask);
}

void dealloc(){
    _dealloc(REQs);
    _dealloc(ps  );
    _dealloc(fs  );
    _dealloc(pairMask);
}


void unbindAll(){
    int n       = 0;
    int nmask   = 0;
    Vec3d* REQs = 0;
    Vec3d* ps   = 0;
    Vec3d* fs   = 0;
    Vec2i* pairMask = 0;
}

void bindOrRealloc(int n_, int nmask_, Vec3d* ps_, Vec3d* fs_, Vec3d* REQs_, Vec2i* pairMask_ ){
    n=n_;
    nmask=nmask_;
    _bindOrRealloc(n,ps_  ,ps  );
    _bindOrRealloc(n,fs_  ,fs  );
    _bindOrRealloc(n,REQs_,REQs);
    _bindOrRealloc(nmask,pairMask_,pairMask);
}

void setREQs(int i0,int i1, const Vec3d& REQ){
    for(int i=i0;i<i1;i++){ REQs[i]=REQ; }
}

void cleanForce(){
    for(int i=0; i<n; i++){ fs[i].set(0.); }
}

double evalLJQs(){
    const int N=n;
    double E=0;
    for(int i=0; i<N; i++){
        Vec3d fi = Vec3dZero;
        Vec3d pi = ps[i];
        const Vec3d& REQi = REQs[i];
        for(int j=i+1; j<N; j++){    // atom-atom
            Vec3d fij = Vec3dZero;
            Vec3d REQij; combineREQ( REQs[j], REQi, REQij );
            E += addAtomicForceLJQ( ps[j]-pi, fij, REQij );
            fs[j].sub(fij);
            fi   .add(fij);
        }
        fs[i].add(fi);
    }
    return E;
}

double evalLJQ_sortedMask(){
    //printf("evalLJQ_sortedMask n %i nmask %i \n", n, nmask );
    int im=0;
    const int N=n;
    double E=0;
    for(int i=0; i<N; i++){
        Vec3d fi = Vec3dZero;
        Vec3d pi = ps[i];
        const Vec3d& REQi = REQs[i];
        for(int j=i+1; j<N; j++){    // atom-atom
            if( (im<nmask)&&(i==pairMask[im].i)&&(j==pairMask[im].j) ){
                im++; continue;
            }
            Vec3d fij = Vec3dZero;
            Vec3d REQij; combineREQ( REQs[j], REQi, REQij );
            E += addAtomicForceLJQ( ps[j]-pi, fij, REQij );
            //double Eij = addAtomicForceLJQ( ps[j]-pi, fij, REQij ); E+=Eij;
            //Vec3d torq = Vec3dZero; torq.add_cross(fij,pi); torq.add_cross(fij*-1,ps[j]);
            //printf( "[%i,%i] torq %g |fij| %g \n", i,j, torq.norm(), fij.norm() );
            //if(E>100.0){
                //printf( "[%i,%i] Eij %g REQ(%g,%g,%g)\n", i, j, Eij, REQij.x,REQij.y,REQij.z );
                //printf( "%i %i  %g \n", i, j, E );
                //glColor3f(0.0,1.0,0.0); Draw3D::drawLine( ps[j],pi);
                //printf("[%i,%i] |Fij| %g \n", i, j, fij.norm() );
                //printf("[%i,%i] |Fij| %g Eij %g \n", i, j, fij.norm(), Eij );
                //glColor3f(1.0,0.0,0.0); Draw3D::drawVecInPos( fij*-1000,ps[j]);
                //glColor3f(1.0,0.0,0.0); Draw3D::drawVecInPos( fij*1000 ,ps[i]);
            //}

            fs[j].sub(fij);
            fs[i].add(fij);
            fi   .add(fij);

        }
        //fs[i].add(fi);
    }
    //exit(0);
    return E;

}


/*  // ---------- Not yet implemented and tested


// ============= Short Range Force using neighbor list



bool checkListValid(){
    double R2=sq(sr_dRmax);
    const int N=n;
    for(int i=0; i<N; i++){
        Vec3d d; d.set_sub(ps[i],psBack[i]);
        if(d.norm2()>R2){ return false; }
    }
    return true;
}

void makeSRList(){
    //Vec2i* m=pairMask;
    int im=0;
    const int N=n;
    //printf( "N %i \n" );
    neighList.clear();
    for(int i=0; i<N; i++){
        const Vec3d& pi = ps[i];
        psBack[i]=pi;
        double Ri = REQs[i].x;
        const Vec3d& REQi = REQs[i];
        for(int j=i+1; j<N; j++){    // atom-atom
            if( (im<nmask)&&(i==pairMask[im].i)&&(j==pairMask[im].j) ){
                im++; continue;
            }
            Vec3d d   = ps[j]-pi;
            double r2 = d.norm2();
            double Rij = (Ri + REQs[j].x)*sr_Rscale + sr_dRmax;
            if( r2<Rij*Rij ){
                neighList.push_back({i,j});  //  TODO: is this fast enough?
            }
        }
    }
}

double evalSRlist(){
    double E=0;
    double w2 = sq(sr_w);
    for(const Vec2i& ij : neighList ){
        Vec3d fij = Vec3dZero;
        double Rij = (REQs[ij.i].x+REQs[ij.i].y)* sr_Rscale;
        //addForceR2   ( ps[ij.j]-ps[ij.i], fij, Rij*Rij, sr_K     );
        //addForceR2inv( ps[ij.j]-ps[ij.i], fij, Rij*Rij, sr_K, w2 );
        E += addForceR2inv( ps[ij.j]-ps[ij.i], fij, Rij*Rij, sr_K, w2 );
        fs[ij.i].sub(fij);
        fs[ij.j].add(fij);
    }
    return E;
}


*/

};

#endif
