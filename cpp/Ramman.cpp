
// Zhang, Y., Dong, Z., & Aizpurua, J. (2021). Theoretical treatment of single‐molecule scanning Raman picoscopy in strongly inhomogeneous near fields. Journal of Raman Spectroscopy, 52(2), 296–309. https://doi.org/10.1002/jrs.5991
// https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Supplemental_Modules_(Physical_and_Theoretical_Chemistry)/Physical_Properties_of_Matter/Atomic_and_Molecular_Properties/Intermolecular_Forces/Specific_Interactions/Dipole-Dipole_Interactions

// Dipole Dipole Interaction
//   Aij = ( 3*(p_i|r_ij)*(p_j|r_ij) -  (p_i|p_i) )/|r_ij^3|

#include "Vec3.h"
//#include "Mat3.h"
#include "SMat3.h"
//#include "CG.h"

int fieldType = 1;
double  cMonopol = 1;
Vec3d   cDipol   = Vec3dZ;
//SMat3d  cQuadrupol;

int verbosity=0; 

inline void evalField( const Vec3d& p, Vec3d& Efield ){
    // ---- monopole
    double ir2 = 1/( p.norm2() + 1.0e-6 );
    double ir  = sqrt(ir2); 
    double ir3 = ir2*ir;
    Efield.set_mul( p, cMonopol*ir3 ); 
    // --- dipole
    if(fieldType>1){
        // https://physics.stackexchange.com/questions/173101/find-out-gradient-of-electric-potential-at-bf-r-created-by-eletric-dipole-o
        //   E= (1/4πϵ0)  ( 3u(p.u)−p ) /r^3
        Efield.add_mul( cDipol,   ir3 );
        Efield.add_mul( p     , 3.0*(-ir3*ir2)*cDipol.dot(p) );
    }
}

double ProjectPolarizability( const Vec3d& tip_pos, int na, Vec3d * apos, SMat3d* alphas, int idir, int icomp ){
    double Amp=0;
    //printf("C++ ProjectPolarizability() dir,comp %i %i \n",  idir, icomp  );
    for(int ia=0; ia<na; ia++){
        int i3=3*ia;
        Vec3d pos  = apos[ia]-tip_pos;
        double Ampi =  alphas[i3+idir].array[icomp]/pos.norm2();       //   NON-ZERO
        Amp += Ampi;
        //printf( "dir,comp %i,%i atom[%i] pos(%g,%g,%g) Ampi %g Amp %g\n", idir, icomp, ia, pos.x,pos.y,pos.z,  Ampi, Amp );
    }
    return Amp;
}

void printSMat(const SMat3d& A){
    printf("%g %g %g \n", A.xx, A.xy, A.xz );
    printf("%g %g %g \n", A.xy, A.yy, A.yz );
    printf("%g %g %g \n", A.xz, A.yz, A.zz );
}

double RammanAmplitude( const Vec3d& tip_pos, int na, Vec3d * apos, SMat3d* alphas, Vec3d* mode, Vec3d* E_inc=0, Vec3d* E_ind=0 ){
    //   tip_pos ... tip position
    //   apos   ... atom positions
    //   alphas ... atomic polarizability matrix 
    //   mode   ... displacement of atoms along vibration normal mode
    double Amp=0;
    for(int ia=0; ia<na; ia++){
        int i3=3*ia;
        // -- Evaluate electric fiedl at atomic position 
        Vec3d pos  = apos[ia]-tip_pos;
        Vec3d Efield; evalField( pos, Efield );
        if(E_inc){E_inc[ia]=Efield;}
        //printf( "atom[%i] pos(%g,%g,%g) Efield(%g,%g,%g)\n", ia, pos.x,pos.y,pos.z,  Efield.x,Efield.y,Efield.z );
        // -- Get atomic polarizability matrix due to vibration along normal mode 
        Vec3d dadMi = mode[ia];  // displacement of atom i in mode,  Eq.7 of  10.1002/jrs.5991
        SMat3d AA=SMat3dZero;
        //dadMi=(Vec3d){0.,0.,1.};
        AA.add_mul( alphas[i3+0], dadMi.x );  // TODO : is this OK ?.... can I really combine polarizabilities like this ?
        AA.add_mul( alphas[i3+1], dadMi.y );
        AA.add_mul( alphas[i3+2], dadMi.z );

        // -- Polarization of the atom by the field
        Vec3d polarization; 
        AA.dot_to( Efield, polarization );     // Eq.9 of 10.1002/jrs.5991

        if(verbosity>1){
            printf("============ atom[%i]\n", ia);
            if(verbosity>2){
                printf(" ### d_X(Alpha)\n"); printSMat(alphas[i3+0]);
                printf(" ### d_Y(Alpha)\n"); printSMat(alphas[i3+1]);
                printf(" ### d_Z(Alpha)\n"); printSMat(alphas[i3+2]);
            }
            printf(" #### AA(=alpha*dMode): \n"); printSMat(AA);
            printf("d_mode:       %g %g %g \n", dadMi.x,dadMi.y,dadMi.z );
            printf("Efield:       %g %g %g \n", Efield.x,Efield.y,Efield.z );
            printf("polarization: %g %g %g \n", polarization.x,polarization.y,polarization.z );
        }

        if(E_ind){E_ind[ia]=polarization;}
        double Ampi = Efield.dot( polarization );  // ?Should be Like this ? This is like coupling back to cavity;   Eq.13   10.1002/jrs.5991
        //double Ampi += polarization.norm2();           // ?Or rather like this ? This is like far-field radiation    ;   Eq.16   10.1002/jrs.5991

        Amp+=Ampi;
        //printf( "atom[%i] pos(%g,%g,%g) Ampi %g Amp %g P(%g,%g,%g), Efield(%g,%g,%g)\n", ia, pos.x,pos.y,pos.z,  Ampi, Amp, polarization.x,polarization.y,polarization.z, Efield.x,Efield.y,Efield.z );
    }
    return Amp;
}


extern "C"{

void ProjectPolarizability( int npos, double* tpos, double* As, int na, double* apos, double* alphas, int idir, int icomp ){
    //printf("C++ ProjectPolarizability idir, icomp %i %i \n", idir, icomp );
    for(int ip=0; ip<npos; ip++){
        //printf("C++ ProjectPolarizability %i \n", ip);
        As[ip] = ProjectPolarizability( ((Vec3d*)tpos)[ip], na, (Vec3d*)apos, (SMat3d*)alphas, idir, icomp  );
    }
}

void RammanAmplitudes( int npos, double* tpos, double* As, int na, double* apos, double* alphas, double* modes, int imode ){
    for(int ip=0; ip<npos; ip++){
        As[ip] = RammanAmplitude( ((Vec3d*)tpos)[ip], na, (Vec3d*)apos, (SMat3d*)alphas, ((Vec3d*)modes)+(imode*na)  );
    }
}

double RammanDetails( double* tpos, int na, double* apos, double* alphas, double* modes, int imode, double* E_incident, double* E_induced ){
    return RammanAmplitude( *((Vec3d*)tpos), na, (Vec3d*)apos, (SMat3d*)alphas, ((Vec3d*)modes)+(imode*na), (Vec3d*)E_incident, (Vec3d*)E_induced );
}

void EfieldAtPoints( int npos, double* pos, double* Es_ ){
    Vec3d* Es=(Vec3d*)Es_;
    for(int ip=0; ip<npos; ip++){
        Vec3d Efield; evalField( ((Vec3d*)pos)[ip], ((Vec3d*)Es)[ip] );
    }
}

void setEfieldMultipole( int fieldType_, double* coefs ){
    fieldType = fieldType_;
    cMonopol = coefs[0];
    if( fieldType>1 ){
        cDipol = *(Vec3d*)(coefs+1);
    }
    printf( "C++ setEfieldMultipole(): fieldType %i cMonopol %g cDipol (%g,%g,%g) \n", fieldType, cMonopol, cDipol.x,cDipol.y,cDipol.z );
}

void setVerbosity(int verbosity_){ verbosity=verbosity_; }

}