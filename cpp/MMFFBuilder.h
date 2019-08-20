#ifndef MMFFBuilder_h
#define MMFFBuilder_h

#include <string>
#include <vector>
#include <unordered_map>

//#include "Molecule.h"
#include "MMFF.h"
#include "Vec2.h"
#include "Vec3.h"
#include "quaternion.h"

//#include "MMFFparams.h"

// =============== Free functions

/*
int write2xyz( FILE* pfile, MMFF * ff, MMFFparams * params ){
    fprintf(pfile, "%i\n", mmff->natoms );
    fprintf(pfile, "#comment \n");
    for(int i=0; i<mmff->natoms; i++){
        int ityp   = mmff->atypes[i];
        Vec3d&  pi = mmff->apos[i];
        //printf( "write2xyz %i %i (%g,%g,%g) %s \n", i, ityp, pi.x,pi.y,pi.z, params->atypes[ityp].name );
        fprintf( pfile, "%s   %15.10f   %15.10f   %15.10f \n", params->atypes[ityp].name, pi.x,pi.y,pi.z );
    };
    return mmff->natoms;
}

int save2xyz( char * fname, MMFF * mmff, MMFFparams * params ){
    FILE* pfile = fopen(fname, "w");
    if( pfile == NULL ) return -1;
    int n = write2xyz(pfile, mmff, params );
    fclose(pfile);
    return n;
}
*/

inline int selectMinHigher(int a0, int n, int* as){
    int amin=0x7FFFFFFF; // 32 int max
    int imin=0;
    for(int i=0;i<n;i++){
        int a=as[i];
        if((a>a0)&&(a<amin)){amin=a;imin=i;};
    }
    return imin;
};


// =============== Structs for Atoms, Bonds etc...


namespace MMFF{

struct Atom{
    constexpr const static Vec3d defaultREQ = (Vec3d){ 1.7d, sqrt(0.0037292524), 0 };
    
    // this breaks {<brace-enclosed initializer list>} in C++11
    //int type  = -1;
    //int frag  = -1;
    //int iconf = -1;
    //Vec3d pos;
    //Vec3d REQ = defaultREQ;   // constexpr Vec3d{1.7,sqrt(0.0037292524),0}

    int type;
    int frag;
    int iconf;
    Vec3d pos;
    Vec3d REQ;   // constexpr Vec3d{1.7,sqrt(0.0037292524),0}
    
    //Atom() = default;
    
    void print()const{ printf( "Atom{ t %i c %i f %i REQ(%g,%g,%g) pos(%g,%g,%g)}", type, iconf, frag, REQ.x, REQ.y, REQ.z, pos.x,pos.y,pos.z ); }

};

#define N_NEIGH_MAX 8
enum class NeighType: int {
    pi    = -1,
    epair = -2,
    H     = -3
};


struct AtomConf{

    int iatom=-1;
    int n     =0;
    int nbond =0;
    int npi   =0; // pi bonds
    int ne    =0; // electron pairs
    int nH    =0; //
    int neighs[N_NEIGH_MAX]; // neighs  - NOTE: bonds not atoms !!!!

    //AtomConf() = default;

    inline bool addNeigh(int ia, int& ninc ){
        if(n>=N_NEIGH_MAX)return false;
        neighs[n]=ia;
        ninc++;
        n++;
        //printf( "bond.addNeigh %i %i\n", n, ninc );
        return true;
    };

    inline bool addBond (int i){ return addNeigh(i,nbond); };
    inline bool addH    (     ){ return addNeigh((int)NeighType::H    ,nH ); };
    inline bool addPi   (     ){ return addNeigh((int)NeighType::pi   ,npi); };
    inline bool addEpair(     ){ return addNeigh((int)NeighType::epair,ne ); };
    inline int  clearNonBond(){ n=nbond; npi=0;ne=0;nH=0; };
    inline bool setNonBond(int npi_,int ne_){ npi=npi_; int ne=ne_; n=nbond+npi+ne+nH; }

    void print()const{ printf( "AtomConf{ ia %i, n %i nb %i np %i ne %i nH %i (%i,%i,%i,%i) }", iatom, n, nbond, npi, ne, nH , neighs[0],neighs[1],neighs[2],neighs[3] ); }
};

struct Bond{
    
    // --- this breaks {<brace-enclosed initializer list>} in C++11
    //int    type  = -1;
    //Vec2i  atoms = (Vec2i){-1,-1};
    //double l0=1,k=0;
    //Bond()=default;
    
    int    type;
    Vec2i  atoms;
    double l0,k;

    inline int getNeighborAtom(int ia)const{
        if     (ia==atoms.i){ return atoms.j; }
        else if(ia==atoms.j){ return atoms.i; }
        return -1;
    }

    void print()const{ printf( " Bond{t %i a(%i,%i) l0 %g k %g}", type, atoms.i, atoms.j, l0, k ); };
};

struct Angle{

    // --- this breaks {<brace-enclosed initializer list>} in C++11
    //int type     = -1;
    //Vec2i  bonds = (Vec2i){-1,-1};
    //double a0    = 0;
    //double k     = 0;
    //Angle()=default;
    
    int type;
    Vec2i  bonds;
    double a0;
    double k;
    
    void print()const{ printf( " Angle{t %i b(%i,%i) a0 %g k %g}", type, bonds.i, bonds.j, a0, k ); }
};


struct Dihedral{

    // --- this breaks {<brace-enclosed initializer list>} in C++11
    //int type     = -1;
    //Vec3i  bonds = (Vec3i){-1,-1,-1};
    //int    n=0;
    //double k=0;
    
    int    type;
    Vec3i  bonds;
    int    n;
    double k;
    
    //Dihedral()=default;

    void print()const{ printf( " Dihedral{t %i b(%i,%i,%i) n %i k %g}", type, bonds.a, bonds.b,bonds.c, n, k ); }
};

struct Molecule{
    Molecule * mol ;
    Vec3i      i0  ;
};

struct Fragment{
    //int imolType;
    int atom0, natom;
    Vec3d  pos;
    Quat4d rot;
    Molecule * mol;
    Vec3d    * pos0s;
};

class Builder{  public:

    //MMFFparams*  params = 0;
    std::unordered_map<std::string,int> molTypeDict;
    std::vector<Molecule*> molTypes;

    std::vector<Atom>       atoms;
    std::vector<Bond>       bonds;
    std::vector<Angle>      angles;
    std::vector<Dihedral>   dihedrals;
    std::vector<Molecule>   mols;
    std::vector<Fragment>   frags;

    std::vector<AtomConf>  confs;
    //std::vector<int>  atom_neighs;

    std::unordered_map<size_t,size_t> fragTypes;
    std::unordered_map<size_t,size_t> mol2molType;

    Bond  defaultBond;
    Angle defaultAngle;

    Atom capAtom; // = (Atom){,,};
    Atom capAtomEpair;
    Atom capAtomPi;
    Bond capBond;
    Vec3d    capUp   = (Vec3d){0.0d,0.0d,1.0d};
    bool bDummyPi    = false;
    bool bDummyEpair = false;


    void clear(){
        atoms.clear(); //printf("DEBUG a.1 \n");
        bonds.clear(); //printf("DEBUG a.2 \n");
        angles.clear();
        mols .clear(); //printf("DEBUG a.3 \n");
        frags.clear(); //printf("DEBUG a.4 \n");
        fragTypes.clear();
    }

    void toForceField( ForceField& ff ){
        //printf( "na %i nb %i nA %i \n", atoms.size(), bonds.size(), dihedrals.size() );
        //mmff->deallocate();
        ff.realloc( atoms.size(), bonds.size(), angles.size(), dihedrals.size() );
        for(int i=0; i<atoms.size(); i++){
            ff.apos [i]  = atoms[i].pos;

            //println(atoms[i]);
            //if( atoms[i].iconf>=0 ) println(confs[atoms[i].iconf]);
        }
        for(int i=0; i<bonds.size(); i++){
            const Bond& b  = bonds[i];
            const Vec2i& ib    = b.atoms;
            ff.bond2atom[i]    = ib;
            //if(params){
            //    params->getBondParams( atoms[ib.x].type, atoms[ib.y].type, bonds[i].type, ff.bond_l0[i], ff.bond_k[i] );
            //}else{
            //    //printf( "no params \n" );
            //    ff.setBondParam(i, b.l0, b.k );
            //}
            //printf( "bond[%i] (%i,%i) %g %g | %g %g\n", i, ff.bond2atom[i].i, ff.bond2atom[i].j, ff.bond_l0[i], ff.bond_k[i], b.l0, b.k );
            //bondTypes[i]       = bonds[i].type;
        }
        for(int i=0; i<angles.size(); i++){
            const Angle& a  = angles[i];
            ff.ang2bond[i] = a.bonds;
            ff.setAngleParam(i, a.a0, a.k );
            //printf( "angle[%i] (%i,%i) (%g,%g) %g\n", i, ff.ang2bond[i].i, ff.ang2bond[i].j, ff.ang_cs0[i].x, ff.ang_cs0[i].y, ff.ang_k[i] );
        }
        for(int i=0; i<dihedrals.size(); i++){
            const Dihedral& d  = dihedrals[i];
            ff.tors2bond[i] = d.bonds;
            ff.setTorsParam( i, d.n, d.k );
            //printf( "dihedrals[%i] (%i,%i,%i) %i %g\n", i, ff.tors2bond[i].a, ff.tors2bond[i].b, ff.tors2bond[i].c, ff.tors_n[i], ff.tors_k[i] );
        }
        ff.angles_bond2atom();
        ff.torsions_bond2atom();
        //exit(0);
    }

    // ============= Add Capping Hydrogens

    const AtomConf* getAtomConf(int ia)const{
        int ic=atoms[ia].iconf;
        if(ic>=0){ return &confs[ic]; }
        return 0;
    }

    int getBondToNeighbor( int ia, int ja )const {
        const AtomConf* conf = getAtomConf(ia);
        if(conf){
            for(int i=0; i<conf->nbond; i++){
                int ib  = conf->neighs[i];
                if(ib<0) continue;
                int jai = bonds[ib].getNeighborAtom(ia);
                if(jai==ja){ return ib; }
            }
        }
        return -1;
    }

    inline int getBondByAtoms(int i, int j)const{
        int ib;
        ib = getBondToNeighbor( i, j ); if( ib>=0 ) return ib;
        ib = getBondToNeighbor( j, i ); if( ib>=0 ) return ib;
        return -1;
    }

    AtomConf* insertAtom(const Atom& atom, bool bConf ){
        atoms.push_back(atom);
        if(bConf){
            int ic = confs.size();
            int ia = atoms.size()-1;
            //printf( "insertAtom ia %i ic %i \n", ia, ic );
            atoms.back().iconf = ic;
            confs.push_back(AtomConf());
            AtomConf& c = confs.back();
            c.iatom = ia;
            return &c;
        }
        return 0;
    }

    void insertBond(const Bond& bond ){
        int ib = bonds.size();
        bonds.push_back(bond);
        int ic = atoms[bond.atoms.i].iconf;
        int jc = atoms[bond.atoms.j].iconf;
        //if(ic>=0){ confs[ic].addBond(bond.atoms.j); }
        //if(jc>=0){ confs[jc].addBond(bond.atoms.i); }
        //printf( "insertBond %i(%i,%i) to c(%i,%i) l0 %g k %g\n", ib, bond.atoms.i,bond.atoms.j, ic, jc, bond.l0, bond.k );
        if(ic>=0){ confs[ic].addBond(ib); }
        if(jc>=0){ confs[jc].addBond(ib); }
    }

    void addCap(int ia,Vec3d& hdir, Atom* atomj, int btype){
        int ja=atoms.size();
        //capAtom;
        Atom atom_tmp;
        if(atomj==0){
            atom_tmp=capAtom;
            atomj=&atom_tmp;
        }
        if(btype<0) btype=capBond.type;
        atomj->pos = atoms[ia].pos + hdir;
        //atoms.push_back( *atomj );
        insertAtom(*atomj,false);
        //bonds.push_back( (Bond){btype,{ia,ja}} );
        capBond.atoms.set(ia,ja);
        insertBond( capBond );
        //int ic = atoms[ia].iconf;
        //confs[ic].addBond(ja);
        //confs[ic].addBond(bonds.size());
        //println(confs[ic]);
    }

    void makeConfGeom(int nb, int npi, Vec3d* hs){
        Mat3d m;
        if(nb==3){ // defined by 3 sigma bonds
            m.b.set_cross( hs[1]-hs[0], hs[2]-hs[0] );
            m.b.mul( -1/m.b.norm() );
            if(npi==0){ // sp3 no-pi
                if( 0 < m.b.dot( hs[0]+hs[1]+hs[2] ) ){ m.b.mul(-1.); }
                hs[3]=m.b;
            }else{
                hs[3]=m.b;
            }
        }else if(nb==2){ // defined by 2 sigma bonds
            m.fromCrossSafe( hs[0], hs[1] );
            if      (npi==0){ // -CH2- like sp3 no-pi
                const double cb = 0.81649658092; // sqrt(2/3)
                const double cc = 0.57735026919;  // sqrt(1/3)
                hs[nb  ] = m.c*cc+m.b*cb;
                hs[nb+1] = m.c*cc-m.b*cb;
            }else if(npi==1){ // =CH- like  sp 1-pi
                hs[nb  ] = m.c;
                hs[nb+1] = m.b;
                //printf("like =CH- H(%g,%g,%g) pi(%g,%g,%g,) \n", hs[nb].x, hs[nb].y, hs[nb].z, hs[nb+1].x, hs[nb+1].y, hs[nb+1].z );
            }else{            // #C- like sp 2-pi
                hs[nb  ] = m.c;
                hs[nb+1] = m.b;
            }
        }else if(nb==1){
            m.c = hs[0]; m.c.normalize();
            m.c.getSomeOrtho(m.b,m.a);
            if      (npi==0){ // -CH3 like sp3 no-pi
                const double ca = 0.81649658092;  // sqrt(2/3)
                const double cb = 0.47140452079;  // sqrt(2/9)
                const double cc =-0.33333333333;  // 1/3
                hs[nb  ] = m.c*cc + m.b*(cb*2) ;
                hs[nb+1] = m.c*cc - m.b* cb    + m.a*ca;
                hs[nb+2] = m.c*cc - m.b* cb    - m.a*ca;
            }else if(npi==1){ // =CH2 like sp2 1-pi
                const double ca = 0.87758256189;  // 1/2
                const double cc =-0.5;            // sqrt(1/8)
                hs[nb  ] = m.c*cc + m.a*ca;
                hs[nb+1] = m.c*cc - m.a*ca;
                hs[nb+2] = m.b;
            }else{            // #CH sp  2-pi
                hs[nb  ] = m.c*-1;
                hs[nb+1] = m.b;
                hs[nb+2] = m.a;
            }
        }
    }

    void makeSPConf(int ia,int npi,int ne){
        //if(nH==0){ // ToDo : check reasonable limits npi, nh
        int ic = atoms[ia].iconf;
        AtomConf& conf = confs[ic];
        conf.clearNonBond();
        int nb = conf.nbond;
        int n  = 4-nb-npi;   // number
        int nH = n-ne;
        //printf("-- "); println(conf);
        //printf( "ia %i nb,npi %i,%i   n,nH,ne %i,%i,%i \n", ia,   nb,npi,  n,nH,ne );
        //Mat3d m;
        Vec3d hs[4];
        for(int i=0;i<nb;i++){
            int ib = conf.neighs[i];
            int ja = bonds[ib].getNeighborAtom(ia);
            hs[i]  = atoms[ja].pos - atoms[ia].pos;
            hs[i].normalize();
        }
        makeConfGeom(conf.nbond, npi, hs);
        bool Hmask[]{1,1,1};
        if(nH!=n) Hmask[rand()%n]=0;
        bool breverse = (nH==2)&&(n==3);
        for(int i=0; i<n; i++){
            if     (Hmask[i])   { addCap(ia,hs[i+nb],&capAtom       ,0); }
            else if(bDummyEpair){ addCap(ia,hs[i+nb],&capAtomEpair  ,0); }
        }
        if(bDummyPi){
            for(int i=0; i<npi; i++){ addCap(ia,hs[i+n+nb],&capAtomPi,0); }
        }
        //printf("-> "); println(conf);
    }

    bool tryMakeSPConf(int ia){
        const AtomConf* conf = getAtomConf(ia);
        if(conf){
            makeSPConf(ia,conf->npi,conf->ne);
            return true;
        }
        return false;
    }

    int makeAllConfsSP(){ int n=0; for(int i=0;i<atoms.size();i++){ if(tryMakeSPConf(i)){n++;} return n; } }

    // ============= Angles

    void addAnglesToBond( int ib, int n, int* neighs, double a0, double k ){
        for(int j=0; j<n; j++){
            angles.push_back( (Angle){-1,  (Vec2i){ neighs[ib], neighs[j]}, a0,k} );
        }
    }

    void addAnglesUpToN( int n, int* neighs, double a0, double k ){
        for(int i=0; i<n; i++){ addAnglesToBond( i, i, neighs, a0, k ); }
    }

    void addAnglesToAtom( int ia, double ksigma, double kpi ){
        AtomConf& conf = confs[ atoms[ia].iconf ];
        int nsigma = conf.nbond;
        if(bDummyPi && (conf.npi>0)){
            nsigma -= conf.npi;
            for(int i=0;i<conf.npi;i++){ addAnglesToBond( i+nsigma, i+nsigma, conf.neighs, M_PI_2, kpi ); }
        }
        //constexpr
        static const double a0s[]{ 0.0d, 0.0d, M_PI, 120*M_PI/180, 109.5*M_PI/180 };
        double a0 = a0s[nsigma];
        //printf( "atom[%i] ns %i a0,ks %g %g   {%g,%g,%g,%g} %g \n", ia, nsigma, a0, ksigma, a0s[0],a0s[1],a0s[2],a0s[3] , a0s[nsigma] );
        addAnglesUpToN( nsigma, conf.neighs, a0, ksigma );
    }

    void autoAngles(double ksigma, double kpi){
        for(int i=0; i<atoms.size(); i++){
            if(atoms[i].iconf>=0){
                addAnglesToAtom( i, ksigma, kpi );
            }
        }
    }

    // =============== Dihedrals

    bool insertDihedralByAtom(const Quat4i& ias, Dihedral& tors ){
        int ib1 = getBondByAtoms(ias.x,ias.y); if(ib1<0) return false;
        int ib2 = getBondByAtoms(ias.y,ias.z); if(ib2<0) return false;
        int ib3 = getBondByAtoms(ias.z,ias.w); if(ib3<0) return false;
        tors.bonds.set(ib1,ib2,ib3);
        dihedrals.push_back(tors);
        return true;
    }

    bool checkBondsSorted()const{
        int ia=-1,ja=-1;
        for(int i=0;i<bonds.size(); i++){
            const Vec2i& b = bonds[i].atoms;
            //printf( "pair[%i] %i,%i | %i %i  | %i %i %i \n", i, b.i, b.j,   ia,ja ,   b.i>=b.j,  b.i<ia, b.j<=ja );
            if(b.i>=b.j){ return false; }
            if(b.i<ia)  { return false; }
            else if (b.i>ia){ia=b.i; ja=-1; };
            if(b.j<=ja){ return false; }
            ja=b.j;
        }
        return true;
    }

    bool sortBonds(){
        //printf( "sortBonds \n" );
        // sort bonds so that
        //   1) (b.i<b.j)
        //   1) if(bk.i) (b.i<b.j)
        //   1) (b.i<b.j)

        //int bsort    = new[bonds.size()];
        Bond * bback = new Bond[bonds.size()];
        int *   invBsort = new int     [bonds.size()];

        int nga[N_NEIGH_MAX];
        int ngb[N_NEIGH_MAX];

        int nb=0;
        for(int ia=0; ia<atoms.size(); ia++ ){
            // assume atoms with conf are first, capping are later
            //const AtomConf* conf = getAtomConf(ia);
            AtomConf* conf = (AtomConf*)getAtomConf(ia); // we need to modify it
            if(!conf){
                if(nb<bonds.size()){
                    printf( " This algorithm assumes all atoms with conf precede atoms without confs in the array \n" );
                    goto _GOTO_failed;
                }
                break;
            }
            int nbconf=conf->nbond;
            int * neighs = conf->neighs;
            for(int i=0;i<nbconf;i++ ){
                int ib=neighs[i];
                if(ib<0){ printf("atom[%i].condf inconsistent nbond=%i neigh[%i]<0 \n", ia, conf->nbond, i ); goto _GOTO_failed; }
                int ja = bonds[ib].getNeighborAtom(ia);
                //if(ja<ia)continue; // this bond was processed before
                nga[i]=ja;
                ngb[i]=ib;
            }
            int ja=-1;
            for(int i=0;i<nbconf;i++ ){      // take bonds on atom in order
                int ipick = selectMinHigher(ja, nbconf, nga );
                ja=nga[ipick];
                neighs[i] = ngb[ipick]; // make conf sorted
                //printf( " atom[%i].neigh[%i] %i \n", ia, i, ja  );
                if(ja<ia)continue;      // this bond was processed before (Hopefully)
                int ib = ngb[ipick];

                //bsort   [nb]=ib;
                bback[nb]   = bonds[ib];
                invBsort[ib]=nb;

                //printf( " bond[%i] -> bond[%i] \n", ib, nb );
                nb++;
            }
        }
        for(int i=0; i<nb;i++){
            bback[i].atoms.order();
            bonds[i]=bback[i];
            //printf( " bond[%i] (%i,%i) \n", i, bback[i].atoms.i, bback[i].atoms.j );
        }
        for(int i=0; i<angles.size();i++){
            Vec2i& bs = angles[i].bonds;
            bs.a = invBsort[bs.a];
            bs.b = invBsort[bs.b];
        }
        for(int i=0; i<dihedrals.size();i++){
            Vec3i& bs = dihedrals[i].bonds;
            bs.a = invBsort[bs.a];
            bs.b = invBsort[bs.b];
            bs.c = invBsort[bs.c];
        }
        delete [] bback;
        delete [] invBsort;
        if(false){
        _GOTO_failed:
            return false;
        }
        return true;
    }









/*

// =================== Curently not used


    void clearMolTypes( bool deep ){
        if(deep){ for(Molecule* mol : molTypes ){ mol->dealloc(); delete mol; } }
        molTypeDict.clear();
        molTypes.clear();
    }


    int loadMolType(const char* fname ){
        Molecule* mol = new Molecule();      //printf( "DEBUG 1.1.1 \n" );
        mol->atypNames = &params->atypNames; //printf( "DEBUG 1.1.2 \n" );
        //printf("mol->atypNames %i %i \n", mol->atypNames, &params->atypNames );
        mol->loadXYZ( fname );             //printf( "DEBUG 1.1.3 \n" );
        if(params) params->assignREs( mol->natoms, mol->atomType, mol->REQs ); //printf( "DEBUG 1.1.4 \n" );
        int ityp = molTypes.size();
        mol2molType[(size_t)mol]=ityp;
        molTypes.push_back(mol);  //printf( "DEBUG 1.1.5 \n" );
        return molTypes.size()-1; //printf( "DEBUG 1.1.6 \n" );
    }


    int registerRigidMolType( int natoms, Vec3d* pos, Vec3d* REQs, int* atomType ){
        Molecule* mol = new Molecule();
        mol->allocate( natoms, 0 );
        for(int i=0; i<mol->natoms; i++){ mol->pos[i]=pos[i]; mol->REQs[i]=REQs[i]; mol->atomType[i]=atomType[i]; }
        int ityp = molTypes.size();
        mol2molType[(size_t)mol]=ityp;
        molTypes.push_back(mol);
        return molTypes.size()-1;
    }

    int loadMolType(const std::string& fname, const std::string& label ){
        //printf( "fname:`%s` label:`%s` \n", fname.c_str(), label.c_str()  );
        int itype = loadMolType( fname.c_str() );
        //printf( "fname:`%s` label:`%s` itype %i \n", fname.c_str(), label.c_str(), itype  );
        molTypeDict[label] = itype;
        return itype;
    };

    int insertMolecule( Molecule * mol, const Vec3d& pos, const Mat3d& rot, bool rigid ){
        int natom0  = atoms.size();
        int nbond0  = bonds.size();
        int nangle0 = angles.size();
        mols.push_back( (MMFFmol){mol, (Vec3i){natom0,nbond0,nangle0} } );

        int natoms0 = atoms.size();
        if( rigid ){
            Quat4d qrot; qrot.fromMatrix(rot);
            int ifrag = frags.size();
            //printf( "insertMolecule mol->natoms %i \n", mol->natoms );
            for(int i=0; i<mol->natoms; i++){
                //Vec3d REQi = (Vec3d){1.0,0.03,mol->}; // TO DO : LJq can be set by type
                //atoms.push_back( (MMFFAtom){mol->atomType[i],mol->pos[i], LJq } );
                Vec3d  REQi = mol->REQs[i];   REQi.y = sqrt(REQi.y); // REQi.z = 0.0;
                Vec3d  p; rot.dot_to(mol->pos[i],p); p.add( pos );
                atoms.push_back( (MMFFAtom){mol->atomType[i], ifrag, -1, p, REQi } );
            }
            frags.push_back( (MMFFfrag){natoms0, atoms.size()-natoms0, pos, qrot, mol}  );
            //size_t mol_id = static_cast<size_t>(mol);
            size_t mol_id = (size_t)(mol);
            auto got = fragTypes.find(mol_id);
            if ( got == fragTypes.end() ) {
                fragTypes[ mol_id ] = frags.size()-1; // WTF ?
            }else{}
            return ifrag;
        }else{
            for(int i=0; i<mol->natoms; i++){
                //Vec3d LJq = (Vec3d){0.0,0.0,0.0}; // TO DO : LJq can be set by type
                //Vec3d LJq = (Vec3d){1.0,0.03,0.0}; // TO DO : LJq can be set by type
                Vec3d  REQi = mol->REQs[i];   REQi.y = sqrt(REQi.y);
                Vec3d p; rot.dot_to(mol->pos[i],p); p.add( pos );
                atoms.push_back( (MMFFAtom){mol->atomType[i], -1, -1, p, REQi } );
            }
            for(int i=0; i<mol->nbonds; i++){
                bonds.push_back( (MMFFBond){mol->bondType[i], mol->bond2atom[i] + ((Vec2i){natom0,natom0}), defaultBond.l0, defaultBond.k } );
            }
            for(int i=0; i<mol->nang; i++){
                angles.push_back( (MMFFAngle){ 1, mol->ang2bond[i] + ((Vec2i){nbond0,nbond0}), defaultAngle.a0, defaultAngle.k } );
            }
            return -1;
        }
    }

    int insertMolecule( int itype, const Vec3d& pos, const Mat3d& rot, bool rigid ){
        return insertMolecule( molTypes[itype], pos, rot, rigid );
    };

    int insertMolecule( const std::string& molName, const Vec3d& pos, const Mat3d& rot, bool rigid ){
        //printf( "insertMolecule molName %s itype %i \n", molName.c_str(), molTypeDict[molName] );
        return insertMolecule( molTypes[ molTypeDict[molName] ], pos, rot, rigid );
    };

    void assignAtomTypes(){
        for(int i=0; i<atoms.size(); i++){
            //mmff->aLJq [i]  = atoms[i].type;
            int ityp = atoms[i].type;
            atoms[i].REQ.x = params->atypes[ityp].RvdW;
            atoms[i].REQ.y = params->atypes[ityp].EvdW;
            atoms[i].REQ.z = 0;
            //atomTypes[i]  = atoms[i].type;
        }
    }



    void toMMFF( MMFF * mmff ){
        //mmff->deallocate();
        mmff->allocate( atoms.size(), bonds.size(), angles.size(), 0 );
        //int * atomTypes = new int[atoms.size()];
        //int * bondTypes = new int[bonds.size()];
        for(int i=0; i<atoms.size(); i++){
            mmff->atypes[i] = atoms[i].type;
            mmff->atom2frag[i] = atoms[i].frag;
            mmff->apos [i]  = atoms[i].pos;
            mmff->aREQ [i]  = atoms[i].REQ;
            //atomTypes[i]  = atoms[i].type;
            //printf( "iatom %i atype %i ifrag %i pos (%g,%g,%g) REQ (%g,%g,%g) \n", i, atoms[i].type, atoms[i].frag, atoms[i].pos.x,atoms[i].pos.y,atoms[i].pos.z, atoms[i].REQ.x,atoms[i].REQ.y,atoms[i].REQ.z );
        }
        for(int i=0; i<bonds.size(); i++){
            mmff->bond2atom[i] = bonds[i].atoms;
            Vec2i ib           = bonds[i].atoms;
            params->getBondParams( atoms[ib.x].type, atoms[ib.y].type, bonds[i].type, mmff->bond_0[i], mmff->bond_k[i] );
            //bondTypes[i]       = bonds[i].type;
        }
        for(int i=0; i<angles.size(); i++){
            mmff->ang2bond[i] = angles[i].bonds;
            mmff->ang_0[i] = {1.0,0.0}; // TODO FIXME
            mmff->ang_k[i] = 0.5;       // TODO FIXME
        }
        if( frags.size()>0 ){
            mmff->allocFragment( frags.size() );
            for(int i=0; i<frags.size(); i++){
                MMFFfrag& fragi = frags[i];
                mmff->frag2a  [i] = fragi.atom0;
                mmff->fragNa  [i] = fragi.natom;
                mmff->fapos0s [i] = fragi.mol->pos;
                double * posi= (mmff->poses + i*8);
                *(Vec3d *)(posi  )= fragi.pos;
                *(Quat4d*)(posi+4)= fragi.rot;
            }
        }
        //params.fillBondParams( mmff->nbonds, mmff->bond2atom, bondTypes, atomTypes, mmff->bond_0, mmff->bond_k );
        //delete [] atomTypes;
        //delete [] bondTypes;
    }

*/


}; // MMFFBuilder


} // namespace MMFF

#endif
