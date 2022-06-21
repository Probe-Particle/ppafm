
// Zhang, Y., Dong, Z., & Aizpurua, J. (2021). Theoretical treatment of single‐molecule scanning Raman picoscopy in strongly inhomogeneous near fields. Journal of Raman Spectroscopy, 52(2), 296–309. https://doi.org/10.1002/jrs.5991
// https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Supplemental_Modules_(Physical_and_Theoretical_Chemistry)/Physical_Properties_of_Matter/Atomic_and_Molecular_Properties/Intermolecular_Forces/Specific_Interactions/Dipole-Dipole_Interactions


// Dipole Dipole Interaction
//   Aij = ( 3*(p_i|r_ij)*(p_j|r_ij) -  (p_i|p_i) )/|r_ij^3|

#include "Vec3.h"
//#include "Mat3.h"
#include "SMat3.h"
//#include "CG.h"

int fieldType = 1;

inline void evalField( const Vec3d& p, Vec3d& Efield ){
    // monopole
    double r2 = p.norm2();
    Efield.set_mul( p, 1/r2 ); 
}

double RammanAmplitude( const Vec3d& tip_pos, int na, Vec3d * apos, SMat3d* alphas, Vec3d* mode ){
    //   tip_pos ... tip position
    //   apos   ... atom positions
    //   alphas ... atomic polarizability matrix 
    //   mode   ... displacement of atoms along vibration normal mode
    double Amp=0;
    for(int ia=0; ia<na; ia++){
            // -- Evaluate electric fiedl at atomic position 
            Vec3d pos  = apos[ia]-tip_pos;
            Vec3d Efield; evalField( pos, Efield );
            //printf( "atom[%i] pos(%g,%g,%g) Efield(%g,%g,%g)\n", ia, pos.x,pos.y,pos.z,  Efield.x,Efield.y,Efield.z );
            // -- Get atomic polarizability matrix due to vibration along normal mode 
            SMat3d AA;
            Vec3d dadMi = mode[ia];  // displacement of atom i in mode,  Eq.7 of  10.1002/jrs.5991
            int i3=3*ia;
            AA.add_mul( alphas[i3+0], dadMi.x );  // TODO : is this OK .... can I really combine polarizabilities like this ?
            AA.add_mul( alphas[i3+1], dadMi.y );
            AA.add_mul( alphas[i3+2], dadMi.z );
            // -- Polarization of the atom by the field
            Vec3d polarization; 
            AA.dot_to( Efield, polarization );     // Eq.9 of 10.1002/jrs.5991
            double Ampi = Efield.dot( polarization  );  // ?Should be Like this ? This is like coupling back to cavity;   Eq.13   10.1002/jrs.5991
            //double Ampi += polarization.norm2();           // ?Or rather like this ? This is like far-field radiation    ;   Eq.16   10.1002/jrs.5991
            //printf( "atom[%i] pos(%g,%g,%g) Ampi %g P(%g,%g,%g), Efield(%g,%g,%g)\n", ia, pos.x,pos.y,pos.z,  Ampi, polarization.x,polarization.y,polarization.z, Efield.x,Efield.y,Efield.z );
            Amp+=Ampi;
    }
    return Amp;
}


extern "C"{

void RammanAmplitudes( int npos, double* tpos, double* As, int na, double* apos, double* alphas, double* modes, int imode ){
    for(int ip=0; ip<npos; ip++){
        As[ip] = RammanAmplitude( ((Vec3d*)tpos)[ip], na, (Vec3d*)apos, (SMat3d*)alphas, ((Vec3d*)modes)+(imode*na)  );
    }
}

}