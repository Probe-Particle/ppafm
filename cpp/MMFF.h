
#ifndef MMFF_h
#define MMFF_h

#include "fastmath_light.h"
#include "Vec2.h"
#include "Vec3.h"
#include "quaternion.h"



// ======================
// ====   MMFF
// ======================

namespace MMFF{

//#include "integerOps.h"

#define SIGN_MASK 2147483648

inline void pairs2triple( const Vec2i& b1, const Vec2i& b2, Vec3i& tri, bool& flip1, bool& flip2 ){
    if     ( b1.y == b2.x ){ tri.set( b1.x, b1.y, b2.y ); flip1=false; flip2=false;  }
    else if( b1.y == b2.y ){ tri.set( b1.x, b1.y, b2.x ); flip1=false; flip2=true;   }
    else if( b1.x == b2.x ){ tri.set( b1.y, b1.x, b2.y ); flip1=true;  flip2=false;  }
    else if( b1.x == b2.y ){ tri.set( b1.y, b1.x, b2.x ); flip1=true;  flip2=true;   }
}

class ForceField{ public:

    //static int iDebug = 0;
    
    int  natoms=0, nbonds=0, nang=0, ntors=0;


    double Ea=0,Eb=0,Et=0;
    // --- Parameters

    Vec2i  * bond2atom = 0;
    double * bond_l0   = 0;  // [A]
    double * bond_k    = 0;  // [eV/A] ?

    Vec2i  * ang2bond  = 0;
    Vec3i  * ang2atom  = 0;
    //Vec2d  * ang_0   = 0; // alpha_0
    Vec2d  * ang_cs0   = 0; // cos(a),sin(a)
    double * ang_k     = 0; // [eV/A^2]

    // ToDo:
    //  We don't care about dihedrals yet
    //  Later we perhaps implement it using additional DOF (pi-orbitals,electron pairs)
    Vec3i  * tors2bond = 0;
    Quat4i * tors2atom = 0;
    //Vec2d  * tors_0    = 0; // [1]
    int    * tors_n    = 0;
    double * tors_k    = 0; // [eV/A^2]

    //Vec3d * hpi = 0;  // NOTE : nstead of pi define dummy atom !!!!


    // --- State variables

    Vec3d  * apos     = 0;   // atomic position
    Vec3d  * aforce   = 0;   // atomic position

    // --- Axuliary variables

    double * lbond  = 0;   // bond lengths
    Vec3d  * hbond  = 0;   // normalized bond unitary vectors


void realloc( int natoms_, int nbonds_, int nang_, int ntors_ ){
    natoms=natoms_; nbonds=nbonds_; nang=nang_; ntors=ntors_;
    //printf( "MMFF::allocate natoms: %i  nbonds: %i  nang: %i ntors: %i \n", natoms, nbonds, nang, ntors );
    _realloc( apos      , natoms );
    _realloc( aforce    , natoms );

    _realloc( lbond     , nbonds );
    _realloc( hbond     , nbonds );

    _realloc( bond2atom , nbonds );
    _realloc( bond_l0   , nbonds );
    _realloc( bond_k    , nbonds );

    _realloc( ang2bond  , nang   );
    _realloc( ang2atom  , nang   );
    _realloc( ang_cs0   , nang   );
    _realloc( ang_k     , nang   );

    _realloc( tors2bond , ntors  );
    _realloc( tors2atom , ntors  );
    _realloc( tors_n    , ntors  );
    _realloc( tors_k    , ntors  );

}

void dealloc(){
    natoms=0; nbonds=0; nang=0; ntors=0;
    _dealloc( apos      );
    _dealloc( aforce    );

    _dealloc( lbond     );
    _dealloc( hbond     );

    _dealloc( bond2atom );
    _dealloc( bond_l0   );
    _dealloc( bond_k    );

    _dealloc( ang2bond  );
    _dealloc( ang2atom  );
    _dealloc( ang_cs0   );
    _dealloc( ang_k     );

    _dealloc( tors2bond );
    _dealloc( tors2atom );
    _dealloc( tors_n    );
    _dealloc( tors_k    );
}

inline void setAngleParam(int i, double a0, double k){
    a0=a0*0.5; // we store half angle
    ang_cs0[i].fromAngle(a0);
    //ang_cs0[i] = (Vec2d){cos(a0),sin(a0)};
    ang_k  [i]=k;
};
inline void setBondParam(int i, double l0, double k){ bond_l0[i] = l0; bond_k[i]  = k; };
inline void setTorsParam(int i, int     n, double k){ tors_n [i] =  n; tors_k[i]  = k; };

inline void readSignedBond(int& i, Vec3d& h){ if(i&SIGN_MASK){ i&=0xFFFF; h = hbond[i]; h.mul(-1.0d); }else{ h = hbond[i]; }; };

void cleanAtomForce(){ for(int i=0; i<natoms; i++){ aforce[i].set(0.0d); } }

// ============== Evaluation


int i_DEBUG = 0;

double eval_bond(int ib){
    //printf( "bond %i\n", ib );
    Vec2i iat = bond2atom[ib];
    Vec3d f; f.set_sub( apos[iat.y], apos[iat.x] );
    double l = f.normalize();
    lbond [ib] = l;
    hbond [ib] = f;
    const double k = bond_k[ib];
    double dl = (l-bond_l0[ib]);
   // printf( "bond[%i] l %g dl %g k %g fr %g \n", ib, l, dl, k,  dl*k*2 );
    f.mul( dl*k*2 );
    aforce[iat.x].add( f );
    aforce[iat.y].sub( f );
    return k*dl*dl;
}

double eval_angle(int ig){

    Vec2i ib = ang2bond[ig];
    Vec3i ia = ang2atom[ig];

    // -- read bond direction ( notice orientation sign A->B vs. A<-B )
    Vec3d ha,hb;
    readSignedBond(ib.a, ha);
    readSignedBond(ib.b, hb);


    // angular force
    Vec3d h; h.set_add( ha, hb );
    //Vec3d h; h.set_sub( hb, ha );
    double c2 = h.norm2()*0.25d;               // cos(a/2) = |ha+hb|
    double s2 = 1-c2;
    //printf( "ang[%i] (%g,%g,%g) (%g,%g,%g) (%g,%g,%g) c2 %g s2 %g \n", ig, ha.x,ha.y,ha.z,  hb.x,hb.y,hb.z,  h.x,h.y,h.z,   c2, s2 );
    double c  = sqrt(c2);
    double s  = sqrt(s2);
    //const Vec2d cs0 = ang_cs0[ig];
    Vec2d cs = ang_cs0[ig];
    const double k   = ang_k  [ig];

    cs.udiv_cmplx({c,s});
    double E         =  k*( 1 - cs.x );  // just for debug ?
    double fr        = -k*(     cs.y );

    //printf( "angle[%i] bs(%i,%i) as(%i,%i,%i) c %g fr %g \n", ig,  ib.i,ib.j,  ia.a,ia.b,ia.c,   c,  fr );

    // project to per leaver
    c2 *=-2;
    double lw     = 2*s*c;       //    |h - 2*c2*a| =  1/(2*s*c) = 1/sin(a)
    double fra    = fr/(lbond[ib.a]*lw);
    double frb    = fr/(lbond[ib.b]*lw);
    Vec3d fa,fb;
    fa.set_lincomb( fra,h,  fra*c2,ha );  //fa = (h - 2*c2*a)*fr / ( la* |h - 2*c2*a| );
    fb.set_lincomb( frb,h,  frb*c2,hb );  //fb = (h - 2*c2*b)*fr / ( lb* |h - 2*c2*b| );

    // to atoms
    aforce[ia.x].add(fa); aforce[ia.y].sub(fa);
    aforce[ia.z].add(fb); aforce[ia.y].sub(fb);

    return E;
}

double eval_torsion(int it){

    Vec3i  ib = tors2bond[it];
    Quat4i ia = tors2atom[it];

    // -- read bond direction ( notice orientation sign A->B vs. A<-B )
    Vec3d ha,hb,hab;

    readSignedBond(ib.x, ha);
    readSignedBond(ib.y, hab);
    readSignedBond(ib.z, hb);

    double ca   = hab.dot(ha);
    double cb   = hab.dot(hb);
    double cab  = ha .dot(hb);
    double sa2  = (1-ca*ca);
    double sb2  = (1-cb*cb);
    double invs = 1/sqrt( sa2*sb2 );
    //double c    = ;  //  c = <  ha - <ha|hab>hab   | hb - <hb|hab>hab    >

    Vec2d cs,csn;
    cs.x = ( cab - ca*cb )*invs;
    cs.y = sqrt(1-cs.x*cs.x); // can we avoid this sqrt ?

    const int n = tors_n[it];
    for(int i=0; i<n-1; i++){
        csn.mul_cmplx(cs);
    }


    // check here : https://www.wolframalpha.com/input/?i=(x+%2B+isqrt(1-x%5E2))%5En+derivative+by+x

    const double k = tors_k[it];
    double E   =  k  *(1-csn.x);
    double dcn =  k*n*   csn.x;
    //double fr  =  k*n*   csn.y;

    //double c   = cos_func(ca,cb,cab);

    //printf( "<fa|fb> %g cT %g cS %g \n", cs.x, cT, cS );

    // derivatives to get forces

    double invs2 = invs*invs;
    dcn *= invs;
    double dcab  = dcn;                           // dc/dcab = dc/d<ha|hb>
    double dca   = (1-cb*cb)*(ca*cab - cb)*dcn;  // dc/dca  = dc/d<ha|hab>
    double dcb   = (1-ca*ca)*(cb*cab - ca)*dcn;  // dc/dca  = dc/d<hb|hab>

    Vec3d fa,fb,fab;

    fa =Vec3dZero;
    fb =Vec3dZero;
    fab=Vec3dZero;

    Mat3Sd J;

    J.from_dhat(ha);    // -- by ha
    J.mad_ddot(hab,fa, dca ); // dca /dha = d<ha|hab>/dha
    J.mad_ddot(hb ,fa, dcab); // dcab/dha = d<ha|hb> /dha

    J.from_dhat(hb);    // -- by hb
    J.mad_ddot(hab,fb, dcb ); // dcb /dhb = d<hb|hab>/dha
    J.mad_ddot(ha ,fb, dcab); // dcab/dhb = d<hb|ha> /dha

    J.from_dhat(hab);         // -- by hab
    J.mad_ddot(ha,fab, dca);  // dca/dhab = d<ha|hab>/dhab
    J.mad_ddot(hb,fab, dcb);  // dcb/dhab = d<hb|hab>/dhab
    // derivative cab = <ha|hb>

    fa .mul( 1/lbond[ib.a] );
    fb .mul( 1/lbond[ib.c] );
    fab.mul( 1/lbond[ib.b] );

    aforce[ia.x].sub(fa);
    aforce[ia.y].add(fa -fab);
    aforce[ia.z].add(fab-fb);
    aforce[ia.w].add(fb);

    return E;
}

double eval_bonds   (){ Eb=0; for(int i=0; i<nbonds; i++){ Eb+= eval_bond(i);    } return Eb; }
double eval_angles  (){ Ea=0; for(int i=0; i<nang;   i++){ Ea+= eval_angle(i);   } return Ea; }
double eval_torsions(){ Et=0; for(int i=0; i<ntors;  i++){ Et+= eval_torsion(i); } return Et; }

double eval(){                
    //cleanAtomForce();   //    move this outside     
    eval_bonds();              
    eval_angles();   
    eval_torsions(); 
    //printf( "Eb %g Ea %g Et %g\n", Eb, Ea, Et );
    return Eb+Ea+Et;
};

// ============== Preparation

void angles_bond2atom(){
    for(int i=0; i<nang; i++){
        Vec2i ib = ang2bond[i];
        Vec2i b1,b2;
        b1 = bond2atom[ib.i];
        b2 = bond2atom[ib.j];
        bool flip1,flip2;
        //ang2atom[i]={0,0,0};
        pairs2triple( b1, b2, ang2atom[i], flip1, flip2 );
        if(!flip1){ ang2bond[i].i|=SIGN_MASK; };
        if( flip2){ ang2bond[i].j|=SIGN_MASK; };
        //printf( "ang[%i] ((%i,%i)(%i,%i))->(%i,%i,%i) (%i,%i)==(%i,%i) \n", i, b1.x,b1.y, b2.x,b2.y, ang2atom[i].x,ang2atom[i].y,ang2atom[i].z,ang2bond[i].x&0xFFFF, ang2bond[i].y&0xFFFF,ang2bond[i].x,        ang2bond[i].y         );
    }
}

void torsions_bond2atom(){
    for(int i=0; i<ntors; i++){
        Vec3i ib = tors2bond[i];
        Vec2i b1,b2,b12;
        b1  = bond2atom[ib.x];
        b12 = bond2atom[ib.y];
        b2  = bond2atom[ib.z];
        bool flip1,flip2,flip12;
        pairs2triple( b1, b12, *(Vec3i*)(        &tors2atom[i]    ), flip1,  flip12 );
        pairs2triple( b12, b2, *(Vec3i*)(((int*)(&tors2atom[i]))+1), flip12, flip2  );
        if( flip1 ){ tors2bond[i].i|=SIGN_MASK; };
        if( flip12){ tors2bond[i].j|=SIGN_MASK; };
        if( flip2 ){ tors2bond[i].j|=SIGN_MASK; };
        //printf( "tors[%i] ((%i,%i)(%i,%i)(%i,%i))->(%i,%i,%i,%i) (%i,%i,%i)==(%i,%i,%i) \n", i, b1.x,b1.y, b12.x,b12.y, b2.x,b2.y,tors2atom[i].x,tors2atom[i].y,tors2atom[i].z,tors2atom[i].w,tors2bond[i].x&0xFFFF, tors2bond[i].y&0xFFFF, tors2bond[i].z&0xFFFF,tors2bond[i].x,        tors2bond[i].y,        tors2bond[i].z         );
    }
}

void printBondParams(){
    for( int i=0; i<nbonds; i++ ){
        printf( "bond[%i] (%i,%i) l0 %g k %g \n", i, bond2atom[i].x+1, bond2atom[i].y+1, bond_l0[i], bond_k[i] );
    }
}

void printAngleParams(){
    for( int i=0; i<nbonds; i++ ){
        printf( "angle[%i] (%i,%i|%i) cs0(%g,%g) k %g \n", i, ang2atom[i].a+1, ang2atom[i].b+1, ang2atom[i].c+1, ang_cs0[i].x, ang_cs0[i].y, ang_k[i] );
    }
}

}; // MMFF 

}; // namespace MMFF

#endif
