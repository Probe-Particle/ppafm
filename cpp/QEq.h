
#ifndef QEq_h
#define QEq_h

#include "fastmath.h"
#include "Vec3.h"

// implementation of "ChargeEquilibrationforMolecularDynamicsSimulations" 
// http://www.sklogwiki.org/SklogWiki/index.php/Charge_equilibration_for_molecular_dynamics_simulations
// ChargeEquilibrationforMolecularDynamicsSimulations
// https://pubs.acs.org/doi/pdf/10.1021/j100161a070
// AnthonyK.Rappe, William A.Goddard, Phys.Chem.1991,95,3358-3363

void makeCoulombMatrix(int n, Vec3d* ps, double* J){
    //double Jmax = 0.0;
    for(int i=0; i<n; i++){
        Vec3d pi = ps[i];
        J[i*n+i]=0;
        for(int j=i+1; j<n; j++){
            Vec3d  d = ps[j] - pi;
            double Jij = 14.3996448915/(1+d.norm());
            //double Jij = 14.3996448915/sqrt(1+d.norm2());
            //printf("Jij[%i,%i] %g \n", i,j, Jij);
            //Jmax = fmax(Jmax,Jij);
            J[i*n+j] =Jij;
            J[j*n+i] =Jij;
        }
    }
    //printf("Jmax %g \n", Jmax);
    //exit(0);
}

class QEq{ public:
    int     n    = 0;
    //Vec3d*  ps = 0;
    double* J    = 0;
    double* qs   = 0;
    double* fqs  = 0;
    double* vqs  = 0;

    double* affins = 0;
    double* hards  = 0;

    double Qtarget = 0.0;
    double Qtot    = 0.0;

    void realloc(int n_){
        n=n_;
        _realloc( J, n*n );
        _realloc( qs , n );
        _realloc( fqs, n );
        _realloc( vqs, n );
        _realloc( hards,  n );
        _realloc( affins, n );
    }

    void assignParams(int* itypes, double* taffins, double* thards ){
        for(int i=0;i<n; i++){
            int ityp = itypes[i] - 1;
            affins[i] = taffins[ityp];
            hards [i] = thards [ityp];
        }
    }

    double getQvars(){
        double err2=0;
        //Qtot         = 0.0;
        double fqtot = 0.0;
        for(int i=0; i<n; i++){
            double qi = qs[i];
            //Qtot += qi;
            double fq = affins[i] + hards[i]*qi;
            for(int j=0; j<n; j++){
                fq += J[i*n+j]*qs[j];
            }
            fqtot +=fq; 
            fqs[i]=fq;
            //err2 += fq*fq;
        }
        
        // constrain TOTAL CHARGE
        //fqtot*=(Qtot-Qtarget)/n;
        //double invn = 1/n;
        double dfqtot= fqtot/n;
        for(int i=0; i<n; i++){
            fqs[i] -= dfqtot; 
            err2   += fqs[i]*fqs[i]; 
        };
        //printf( "Qtot %g F2 %g fqtot \n", Qtot, err2, fqtot );
        //for(int i=0; i<n; i++){ qs[i]-=qi; };
        return err2;
    }

    void moveMDdamp(double dt, double damp){
        Qtot = 0.0;
        for(int i=0; i<n; i++){
            vqs[i]  = vqs[i]*damp - fqs[i]*dt;
             qs[i] += vqs[i]*dt;
            Qtot   += qs[i];
        }
        // force Qtarget
        //printf( "Qtot %g \n" );
        double dQ    = (Qtarget-Qtot)/n;
        for(int i=0; i<n; i++){ qs[i] += dQ; }
    }

};


#endif
