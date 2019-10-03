
/*

Flexible Atom sp-hybridization forcefield
 - each atom has 4 atomic orbitals of 3 types { sigma-bond, election-pair, pi-pond }
 - there are two kinds of interaction "onsite" and "bond"
 - "onsite" interaction
    - sigma and epair try to maximize angle between them ( leadign to tetrahedral configuration )
    - pi orbitals try to orhogonalize
 - "bond" interaction
    - spring constant for bonds
    - pi-pi alignment


TODO:
 - more efficient radial force ( R2spline or R2invR2 )
 - fast pre-relaxation of orbitals with forzen atom positions (does not need to recalculate distance list)
    - orbitals rotate toward closes atoms

*/

#ifndef FARFF_h
#define FARFF_h

#include "fastmath.h"
#include "Vec2.h"
#include "Vec3.h"
#include "Forces.h"

#include "macroUtils.h"

#define N_BOND_MAX 4


#define DEBUG_GL
#ifdef  DEBUG_GL
#include "Draw3D.cpp" // DEBUG
#endif

struct AtomRating{
    int ia; 
    double val;
    inline bool operator<(const AtomRating& rhs){ return val<rhs.val; }
    inline bool operator>(const AtomRating& rhs){ return val>rhs.val; }
};

namespace FlexibleAtomReactive{

constexpr static const Vec2d sp_cs0s[] = {
    {   0.0, 0.0 },           // 0, 0
    {   1.0, 0.0 },           // 1, sp0
    {  -1.0, 0.0 },           // 2, sp1
    {  -0.5, 0.86602540378 }, // 3, sp2
    {-1/3.0, 0.94280904158 }, // 4, sp3
};

struct FlexibleAtomType{

    Vec3ui8 conf = Vec3ui8{3,1,0}; 

    double rbond0 =  0.8;  // Rbond
    double aMorse =  4.0;  // EBond
    double bMorse = -0.7;  // kBond

    double vdW_c6  = -15.0; // c6
    double vdW_R  =  8.0;  // RcutVdW

    double Wee  =  1.0;   // electon-electron width
    double Kee  = 10.0;  // electon-electron strenght
    double Kpp  = 10.0;  // onsite  p-p orthogonalization strenght
    double Kpi  =  1.0;   // offsite p-p alignment         strenght

    void print(){
        printf( "aMorse %g bMorse %g\n", aMorse, bMorse );
        printf( "c6     %g r2vdW  %g\n", vdW_c6, vdW_R  );
    }

};

struct FlexiblePairType{

    double rbond0 =  1.6;    // Rbond
    double aMorse =  4.0;    // EBond
    double bMorse = -1.0*2;  // kBond

    double Kpi    =   0.5;  // kPz
    double vdW_c6 = -15.0; // c6
    double vdW_w6 =  pow6( 2.0 );

    inline void combine(const FlexibleAtomType& a, const FlexibleAtomType& b ){
        aMorse  = a.aMorse * b.aMorse;  // TODO
        bMorse  = a.bMorse + b.bMorse;
        rbond0  = a.rbond0 + b.rbond0;

        Kpi    = a.Kpi * b.Kpi;
        vdW_c6 = -(a.vdW_c6  * b.vdW_c6);
        vdW_w6 = pow3( a.vdW_R * b.vdW_R * 2 ); // replace aritmetic average by geometric ?
    }

    void print(){
        printf( "aMorse %g bMorse %g\n", aMorse, bMorse );
        printf( "c6     %g r2vdW  %g\n", vdW_c6, vdW_w6 );
    }

};

class FARFF{ public:

    bool   substract_LJq = true;
    double Eatoms=0,Epairs=0;

    //FlexibleAtomType atype0;
    //FlexiblePairType ptype0;

    int natom   = 0;
    int nbond   = 0;       // number of all bonding orbitals
    int norb    = 0;
    int nVDOF   = 0;
    int nDOF    = 0;       // natoms + neps + npis

    double * dofs   = 0;       // degrees of freedom
    double * fdofs  = 0;       // degrees of freedom
    double * edofs  = 0;

    Vec3d  * apos    = 0;      // atomic position // ALIAS
    Vec3d  * aforce  = 0;      // atomic forces   // ALIAS
    double * aenergy = 0;

    FlexibleAtomType ** atypes = 0;
    Vec3ui8          *  aconf  = 0;

    Vec3d  * opos     = 0;     // normalized bond unitary vectors  // ALIAS
    Vec3d  * oforce   = 0;
    double * oenergy  = 0;

    int realloc(int natom_){
        natom = natom_;
        norb  = natom*N_BOND_MAX;
        nVDOF = natom + norb;
        nDOF  = nVDOF*3;

        //_realloc( oenergy, norb );

        _realloc( atypes , natom );
        _realloc( aconf  , natom );
        //_realloc( aenergy, natom );

        _realloc( dofs  , nDOF  );
        _realloc( fdofs , nDOF  );
        _realloc( edofs , nVDOF );

        apos    = ((Vec3d*)dofs );
        aforce  = ((Vec3d*)fdofs);
        aenergy = edofs;

        opos     = ((Vec3d*) dofs)+natom;
        oforce   = ((Vec3d*)fdofs)+natom;
        oenergy  =          edofs +natom;

        return nVDOF;
    }

// ======== Force Evaluation

void cleanForce(){
    for(int i=0;i<nDOF; i++){ fdofs  [i]=0; }
    for(int i=0;i<natom;i++){ aenergy[i]=0; }
}

double evalAtom(int ia){
    //printf( "atom[%i] \n", ia );
    const FlexibleAtomType& type = *(atypes[ia]);
    const Vec3ui8& conf = aconf[ia];
    const Vec3d&   pa   = apos [ia];
    const int ih        = ia*N_BOND_MAX;
    Vec3d*  hs  = opos  +ih;
    Vec3d*  fs  = oforce+ih;
    //printf( "evalAtom %i  %g %g %g \n", ia, type.rbond0, type.aMorse, type.bMorse  );
    double E    = 0;
    const double w2ee   = sq(type.Wee);
    int    nsigma = conf.a;
    Vec2d cs0 = sp_cs0s[nsigma];
    // -- repulsion between sigma bonds
    
    for(int i=1; i<nsigma; i++){
        const Vec3d& hi = hs[i];
        Vec3d&       fi = fs[i];
        for(int j=0; j<i; j++){ // electron-electron
            E += evalCos2  (hi,hs[j],fi,fs[j],type.Kee,cs0.x);
        }
    }
    
    // -- orthogonalization with p-orbitals
    for(int i=nsigma; i<N_BOND_MAX; i++){
        const Vec3d& hi = hs[i];
        Vec3d&       fi = fs[i];
        for(int j=0; j<i; j++){
            E += evalCos2   ( hi, hs[j], fi, fs[j], type.Kpp, 0);
        }
    }

    return E;
}

double evalPair( int ia, int ja, const FlexiblePairType& type){
    const Vec3ui8& confi = aconf[ia];
    const Vec3ui8& confj = aconf[ja];
    int nbi = confi.a;
    int nbj = confj.a;

    Vec3d  hij; hij.set_sub( apos[ja], apos[ia] );   // = apos[ja] - apos[ia];

    //printf( " evalPair %i,%i   (%g,%g,%g)  (%g,%g,%g) \n", ia,ja,   apos[ia].x, apos[ia].y, apos[ia].z,  apos[ja].x, apos[ja].y, apos[ja].z  );

    double r2   = hij.norm2() + R2SAFE;
    double r    = sqrt( r2 );
    double invr = 1/r;
    hij.mul(invr);             // = dij*(1/rij);

    double r4     = r2*r2;
    double r6     = r4*r2;
    double invVdW = 1/( r6 + type.vdW_w6 );
    double EvdW   = type.vdW_c6*invVdW;
    double fvdW   = -6*EvdW*invVdW*r4*r;

    // todo: replace this by other short range force ... so we can cutoff bonds
    double expar =    exp( type.bMorse*( r - type.rbond0 ) );
    double Esr   =    type.aMorse*expar*expar;
    double Eb    = -2*type.aMorse*expar;
    double fsr   =  2*type.bMorse*Esr;
    double frb   =    type.bMorse*Eb;

    //printf( " evalPair %i,%i  %g %g %g   %g %g  \n", ia,ja,   expar, Esr, Eb, fsr, frb  );

    double E = Esr + EvdW;

    Vec3d force; force.set_mul( hij, fsr + fvdW );

    //printf( " evalPair %i,%i  f(%g,%g,%g)  \n", ia,ja,  force.x, force.y, force.z  );

    Mat3Sd Jbond;
    Jbond.from_dhat(hij);

    const int ioff= ia*N_BOND_MAX;
    const int joff= ja*N_BOND_MAX;
    Vec3d* his = opos  +ioff;
    Vec3d* hjs = opos  +joff;
    Vec3d* fis = oforce+ioff;
    Vec3d* fjs = oforce+joff;

    double* eis = oenergy+ioff;
    double* ejs = oenergy+joff;

    for(int ib=0; ib<nbi; ib++){

        const Vec3d& hi = his[ib];
              Vec3d& fi = fis[ib];

        double       ci = hij.dot( hi );   // ci = <hi|hij>

        if(ci<=0) continue;

        for(int jb=0; jb<nbj; jb++){

            const Vec3d& hj = hjs[jb];
                  Vec3d& fj = fjs[jb];

            double cj       = hij.dot( hj );  // cj  = <hj|hij>

            if(cj>=0) continue; // avoid symmetric image of bond
            double cc  = ci*cj;

            // ==== Version 1) cos^4
            double cc2 = cc*cc;
            double e   = cc2*cc2;
            double de  = 4*cc2*cc;

            // ==== Version 2) (cos-cos0)^2
            // if(cc<ccut)continue;
            // double ccm = cc-ccut;
            // double e  = invcut2*ccm*ccm;
            // double de = 2*invcut2*ccm;

            // ==== Version 3) Lorenz     1/(1 + cos^2 + w2 )
            // # = w*cc/(1+cc+w) =   w*(1+cc+w-(1+w))/(1+cc+w) = w - w*(1+w)/(1-cc+w)
            // # = w*cc/(1-cc+w) =  -w*(1-cc+w-(1+w))/(1-cc+w) = w + w*(1+w)/(1-cc+w)
            // const double wBond  = 0.05;
            // const double wwBond = wBond*(1+wBond);
            // double invcc   = 1/(1+cc+wBond);
            // double invccww = wwBond*invcc;
            // double e       = (invccww + wBond);
            // double de      = -invccww*invcc;

            double eEb = e*Eb*0.5;
            eis[ib]+=eEb;
            ejs[jb]+=eEb;
            E += eEb+eEb;

            //printf( "force[%i,%i][%i,%i] (%g,%g,%g) (%g,%g,%g) \n", ia,ja, ib,jb,   hi.x,hi.y,hi.z,   hj.x,hj.y,hj.z );
            //printf( "force[%i,%i][%i,%i]  c %g %g  e %g %g   \n", ia,ja, ib,jb, ci, cj, e, eEb  );

            double deEb     =    Eb*de;
            double deEbinvr =  deEb*invr;

            Jbond.mad_ddot( hi, force, deEbinvr*cj );
            Jbond.mad_ddot( hj, force, deEbinvr*ci );
            force.add_mul(hij, e*frb);

            fi.add_mul( hij, -cj*deEb );
            fj.add_mul( hij, -ci*deEb );

            //printf( "force[%i,%i][%i,%i] (%g,%g,%g) (%g,%g,%g) \n", ia,ja, ib,jb, fi.x,fi.y,fi.z, fi.x,fi.y,fi.z );
            //force.add(force_);
        }
    }

    // ---------- Pi-Pi interaction
    if( (nbi==3)&&(nbj==3) ){ // align pz-pz in sp2-sp2 pair
        const Vec3d& hi = his[3];
        const Vec3d& hj = hjs[3];
        //Vec3d&       fi = fis[3];
        //Vec3d&       fj = fjs[3];
        double c = hi.dot(hj);
        double de = -2*c*type.Kpi*Eb;
        //if(fabs(Eb)>0.1){ printf( "pi-pi %g %g %g %g \n", de, c, Eb, type.Kpi ); }
        //if(fabs(Eb)>0.1 && fabs(c)<0.95 ){ printf( "pi-pi %g %g %g %g \n", de, c, Eb, type.Kpi ); }

        Vec3d dfi,dfj;
        dfi.set_lincomb( -c*de, hi,    de, hj );
        dfj.set_lincomb(    de, hi, -c*de, hj );

        #ifdef DEBUG_GL 
        glColor3f(1.0,1.0,0.0);
        Draw3D::vecInPos(  dfi, apos[ia]+hi*0.5 );
        Draw3D::vecInPos(  dfj, apos[ja]+hj*0.5 );
        #endif

        fis[3].add(dfi);
        fjs[3].add(dfj);

        //fi.add_mul(hj,de);
        //fj.add_mul(hi,de);

        //fi.add_mul(hi,-hi.dot(fi));
        //fj.add_mul(hj,-hj.dot(fj));
    }

    //printf( "force[%i,%i] (%g,%g,%g) \n", ia, ja, force.x,force.y,force.z );

    aforce[ia].add(force);
    aforce[ja].sub(force);
    return E;
}

double evalPairs(){
    Epairs = 0;
    for(int i=0; i<natom; i++){
        const FlexibleAtomType& atypei = *(atypes[i]);
        for(int j=i+1; j<natom; j++){
            FlexiblePairType ptype; ptype.combine( atypei, *(atypes[j]) );
            Epairs += evalPair( i, j, ptype );
        }
    }
    return Epairs;
}

inline double evalAtoms(){
    Eatoms=0;
    for(int i=0; i<natom; i++){ Eatoms+=evalAtom(i);  }
    return Eatoms;
}

// ToDo : is this numerically stable? if normal forces are very high ?
void normalizeOrbsFast (){ for(int i=0; i<norb; i++){ opos  [i].normalize_taylor3(); } }
void normalizeOrbs     (){ for(int i=0; i<norb; i++){ opos  [i].normalize();         } }
void removeNormalForces(){ for(int i=0; i<norb; i++){ oforce[i].makeOrthoU(opos[i]); } }

// void transferOrbRecoil(){
//     // ToDo : is this numerically stable? if normal forces are very hi ?
//     for(int i=0; i<norb; i++){
//         int ia=i>>2;
//         Vec3d f = oforce[i]*-1;
//         aforce[ia].add( f );
//     }
// }

double eval(){
    //cleanForce();       
    normalizeOrbsFast();  
    Eatoms=evalAtoms(); // do this first to orthonormalize ?
    //transferOrbRecoil();
    Epairs=evalPairs();
    //transferOrbRecoil();
    removeNormalForces();

    #ifdef  DEBUG_GL
    debug_draw(1.0);
    #endif

    return Eatoms + Epairs;
}

void moveGD(double dt, bool bAtom, bool bOrbital ){
    //for(int i=0; i<nDOF; i++){ dofs[i].add_mul( fdofs[i], dt ); }
    if(bAtom   )for(int i=0; i<natom; i++){ apos[i].add_mul( aforce[i], dt ); }
    if(bOrbital)for(int i=0; i<norb;  i++){ opos[i].add_mul( oforce[i], dt ); }
}

double guessBonds(){
    AtomRating neighs[natom*N_BOND_MAX];
    for(int i=0; i<natom*N_BOND_MAX; i++){ 
        //neighs[i]=-1; neighr2s[i]=+1e+300;
        neighs[i].ia  = -1;
        neighs[i].val = +1e+300;
    }
    for(int ia=0; ia<natom; ia++){
        int ioff = ia*N_BOND_MAX;
        const Vec3d& pi = apos[ia];
        for(int ja=0; ja<ia; ja++){
            int joff = ja*N_BOND_MAX;
            Vec3d d; d.set_sub(apos[ja],pi);
            double r2 = d.norm2();
            insertSorted(N_BOND_MAX,{ja,r2},neighs+ioff);
            insertSorted(N_BOND_MAX,{ia,r2},neighs+joff);
        }
    }
    //Quat4d sps[N_BOND_MAX];
    for(int ia=0; ia<natom; ia++){
        const Vec3d& pi = apos[ia];
        int ioff = ia*N_BOND_MAX;
        int nsigma = aconf[ia].a;

        // for(int j=0;j<nsigma;j++){
        //     int  i= ioff+j;
        //     int ja = neighs[i].ia;
        //     //Vec3d d; d.set_sub(apos[ja],pi);
        //     sps[j].p.set_sub(apos[ja],pi);
        //     sps[j].p.normalize();
        //     sps[j].s = M_SQRT1_2;
        //     for(int k=0; k<j; k++){
        //         double c = sps[j].dot( sps[k] ) * 0.66666666666;
        //         sps[j].add_mul(sps[j],-c);
        //     }
        //     opos[i] = sps[j].p;
        //     printf( "atom[%i] orb[%i] sp(%g,%g,%g|%g) \n", ia, j, sps[j].x, sps[j].y, sps[j].z, sps[j].s );
        // }
       Vec3d fw,up,lf; 
       fw.set_sub(apos[neighs[ioff].ia],pi);
       fw.normalize();
       //if      (nsigma==1){
       //     opos[ioff] = fw;
       if(nsigma==2) {
            opos[ioff+0] = fw;
            opos[ioff+1] = fw*-1.0;
            fw.getSomeOrtho( opos[ioff+2],opos[ioff+3] );
       }else if(nsigma==3){
            up.set_sub(apos[neighs[ioff+1].ia],pi);
            up.makeOrthoU(fw); up.normalize();
            opos[ioff+0] = fw;
            opos[ioff+1] = fw*-0.5 + up*0.86602540378;
            opos[ioff+2] = fw*-0.5 + up*-0.86602540378;
            opos[ioff+3].set_cross( fw,up );
       }else if(nsigma==4){
            up.set_sub(apos[neighs[ioff+1].ia],pi);
            up.makeOrthoU(fw);  up.normalize();
            lf.set_cross(fw,up);
            opos[ioff+0] = fw;
            opos[ioff+1] = fw*-0.33333333333 + up* 0.94280904158;
            opos[ioff+2] = fw*-0.33333333333 + up*-0.47140452079 + lf*0.81649658092;
            opos[ioff+3] = fw*-0.33333333333 + up*-0.47140452079 + lf*0.81649658092;
       }
    }

    //for(int io=0; io<norb; io++){ 
    //    double d = 0.1;
    //    opos[io].add(randf(-d,d),randf(-d,d),randf(-d,d));
    //}
}

#ifdef  DEBUG_GL
void debug_draw(double fscale){
    for(int ia=0;ia<natom;ia++){
        const Vec3d& pa = apos[ia];
        int ioff = ia*N_BOND_MAX;
        int nsigma = aconf[ia].a;
        glColor3f(1,0.5,0); Draw3D::vecInPos( aforce[ia]*fscale, pa );
        for(int io=0;io<N_BOND_MAX; io++){
            const Vec3d& hi = opos  [ioff+io];
            const Vec3d& fi = oforce[ioff+io];
            if(io<nsigma){
                glColor3f(0,0,1); Draw3D::vecInPos(hi,pa);
                glColor3f(1,0,0.5); Draw3D::vecInPos( fi*fscale, pa+hi*0.5 );
            }else{
                glColor3f(0,1  ,0); Draw3D::line(pa+hi*0.5,pa+hi*-0.5);
                glColor3f(1,0.5,0); Draw3D::vecInPos( fi*fscale, pa+hi*0.5 );
            }
        }
    }
}
#endif

}; // class FF


}; // namespace FARFF





#endif
