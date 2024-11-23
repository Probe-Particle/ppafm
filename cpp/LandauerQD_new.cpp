#include <cmath>
#include <cstring>
#include "Vec2.h"
#include "Vec3.h"
#include "ComplexAlgebra.hpp"


class LandauerQDs {
public:
    // Parameters
    Vec3d* QDpos;
    double K;
    double decay;
    double* Esite;
    double tS;
    double E_sub;
    double E_tip;
    double tA;
    double eta;
    double Gamma_tip;
    double Gamma_sub;
    bool debug;
    int verbosity;
    int n_qds;

    // Complex matrices
    Vec2d* H_sub_QD;    // Coupling to substrate
    //Vec2d* K_matrix;    // Coulomb interaction matrix
    Vec2d* Hqd0;        // Base QD Hamiltonian (without tip effects)

    LandauerQDs(Vec3d* QDpos_, int n_qds_, double* Esite_, double K_=0.01, double decay_=1.0, 
                double tS_=0.01, double E_sub_=0.0, double E_tip_=0.0, double tA_=0.1, 
                double eta_=0.00, double Gamma_tip_=1.0, double Gamma_sub_=1.0, 
                bool debug_=false, int verbosity_=0) {
        // Store parameters
        QDpos = QDpos_;
        n_qds = n_qds_;
        K = K_;
        decay = decay_;
        Esite = Esite_;
        tS = tS_;
        E_sub = E_sub_;
        E_tip = E_tip_;
        tA = tA_;
        eta = eta_;
        Gamma_tip = Gamma_tip_;
        Gamma_sub = Gamma_sub_;
        debug = debug_;
        verbosity = verbosity_;

        // Allocate arrays
        H_sub_QD = new Vec2d[n_qds];
        Hqd0     = new Vec2d[n_qds * n_qds];
        //K_matrix = new Vec2d[n_qds * n_qds];

        // Initialize H_sub_QD (constant coupling)
        for(int i = 0; i < n_qds; i++) {
            H_sub_QD[i] = {tS, 0.0};
        }

        // Initialize K_matrix (Coulomb interaction)
        // for(int i = 0; i < n_qds; i++) {
        //     for(int j = 0; j < n_qds; j++) {
        //         K_matrix[i * n_qds + j] = {(i != j) ? K : 0.0, 0.0};
        //     }
        // }

        // Initialize Hqd0 (base QD Hamiltonian without tip effects)
        for(int i = 0; i < n_qds; i++) {
            double Ei = Esite[i];
            for(int j = 0; j < n_qds; j++) {
                double val;
                if( i == j ){ val = Ei; }else{ val = K; }
                Hqd0[i * n_qds + j] = { val, 0.0};
            }
        }
    }

    ~LandauerQDs() {
        delete[] H_sub_QD;
        delete[] Hqd0;
        //delete[] K_matrix;
    }

    void calculate_greens_function(double E, Vec2d* H, Vec2d* G) {
        printf("calculate_greens_function(): E = %g\n", E);
        int n = n_qds + 2;
        int n2 = n * n;
        
        // Create (E + iη)I - H
        Vec2d A[n2];
        for(int i = 0; i < n2; i++) {
            A[i] = {-H[i].x, -H[i].y};
            if(i % (n + 1) == 0) {  // Diagonal elements
                A[i].x += E;
                A[i].y += eta;
            }
        }

        Vec2d workspace[n2];  // Temporary workspace for inversion
        invert_complex_matrix(n, A, G, workspace);
    }

    double calculate_tip_coupling(const Vec3d& tip_pos, const Vec3d& qd_pos) {
        Vec3d d = tip_pos - qd_pos;
        return tA * exp(-decay * d.norm());
    }

    double calculate_tip_induced_shift(const Vec3d& tip_pos, const Vec3d& qd_pos, double Q_tip) {
        const double COULOMB_CONST = 14.3996;  // eV*Å/e
        Vec3d d = tip_pos - qd_pos;
        return COULOMB_CONST * Q_tip / d.norm();
    }

    // void calculate_tip_effects(const Vec3d& tip_pos, double Q_tip, Vec2d* tip_couplings, Vec2d* Hqd) {
    //     for(int i = 0; i < n_qds; i++) {
    //         double coupling = calculate_tip_coupling_single(tip_pos, QDpos[i]);
    //         tip_couplings[i] = {coupling, 0.0};

    //         if (Q_tip != 0.0) {
    //             double shift = calculate_tip_induced_shift_single(tip_pos, QDpos[i], Q_tip);
    //             Hqd[i * n_qds + i].x += shift;
    //         }
    //     }
    // }

    void makeHqd( Vec3d tip_pos, double Q_tip, Vec2d* Hqd) {
        // Copy base Hamiltonian
        for(int i = 0; i < n_qds * n_qds; i++) {
            Hqd[i] = Hqd0[i];
        }
        double shifts[n_qds];
        // Add tip-induced shifts to diagonal elements if Q_tip is provided
        if(Q_tip != 0.0) {
            //calculate_tip_induced_shifts(tip_pos, Q_tip, shifts);
            for(int i = 0; i < n_qds; i++) {
                double shift = calculate_tip_induced_shift(tip_pos, QDpos[i], Q_tip);
                Hqd[i * n_qds + i].x += shift;
                //Hqd[i * n_qds + i].x += shifts[i];
            }
            
        }
        //delete[] shifts;
    }

    void assemble_full_H(const Vec3d& tip_pos, Vec2d* Hqd, Vec2d* H) {
        int n = n_qds + 2;
        //Vec2d* tip_couplings = new Vec2d[n_qds];
        //Vec2d tip_couplings[n_qds];
        
        //calculate_tip_coupling(tip_pos, tip_couplings);
        
        // Initialize H to zero
        for(int i = 0; i < n * n; i++) {
            H[i] = {0.0, 0.0};
        }
        
        // Fill QD block with broadening
        for(int i = 0; i < n_qds; i++) {
            for(int j = 0; j < n_qds; j++) {
                H[(i + 1) * n + (j + 1)] = Hqd[i * n_qds + j];
                if(i == j) H[(i + 1) * n + (j + 1)].y -= eta;
            }
        }
        
        // Fill substrate part
        H[0] = {E_sub, -Gamma_sub};
        for(int i = 0; i < n_qds; i++) {
            H[i + 1] = H_sub_QD[i];                 // First row
            H[(i + 1) * n] = H_sub_QD[i];          // First column
        }
        
        // Fill tip part
        H[n * n - 1] = {E_tip, -Gamma_tip};
        for(int i = 0; i < n_qds; i++) {
            double t = calculate_tip_coupling(tip_pos, QDpos[i]);
            H[(i + 1) * n + (n - 1)] = Vec2d{ t, 0 };           // Last column
            H[(n - 1) * n + (i + 1)] = Vec2d{ t, 0 }; // Last row (conjugate)
        }
        
        //delete[] tip_couplings;
    }

    void make_full_hamiltonian(const Vec3d& tip_pos, Vec2d* H, double Q_tip=0.0, Vec2d* Hqd_in=nullptr) {
        printf("make_full_hamiltonian()\n" );
        Vec2d Hqd[n_qds * n_qds];
        if(Hqd_in == nullptr) {
            makeHqd(tip_pos, Q_tip, Hqd);
        } else {
            for(int i = 0; i < n_qds * n_qds; i++) {
                Hqd[i] = Hqd_in[i];
            }
        }
        assemble_full_H(tip_pos, Hqd, H);
    }

    double calculate_transmission_from_H(Vec2d* H, double E) {
        printf("calculate_transmission_from_H()\n" );
        int n = n_qds + 2;
        int n2 = n * n;

        // Stack-allocated matrices
        Vec2d G[n2];
        Vec2d Gdag[n2];
        Vec2d Gammat[n2];
        Vec2d Gammas[n2];
        Vec2d Gammat_Gdag[n2];
        Vec2d G_Gammat_Gdag[n2];
        Vec2d Tmat[n2];

        // Calculate Green's function
        calculate_greens_function(E, H, G);

        // Calculate G_dag (conjugate transpose of G)
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                Gdag[i*n + j] = {G[j*n + i].x, -G[j*n + i].y};
            }
        }

        // Initialize coupling matrices to zero
        for(int i = 0; i < n2; i++) {
            Gammat[i] = {0.0, 0.0};
            Gammas[i] = {0.0, 0.0};
        }

        // Set coupling strengths
        Gammas[0   ] = {2.0 * Gamma_sub, 0.0};        // Substrate coupling
        Gammat[n2-1] = {2.0 * Gamma_tip, 0.0};     // Tip coupling

        // Calculate transmission using Caroli formula: T = Tr(Gamma_s @ G @ Gamma_t @ G†)
        multiply_complex_matrices(n, Gammat , Gdag          , Gammat_Gdag    );
        multiply_complex_matrices(n, G      , Gammat_Gdag   , G_Gammat_Gdag  );
        multiply_complex_matrices(n, Gammas , G_Gammat_Gdag , Tmat           );

        // Save matrices for debugging
        if(debug) {

            save_matrix_to_file( 0, "H (LandauerQD.cpp)", H, n, n);

            save_matrix_to_file("cpp_H.txt", "H (LandauerQD.cpp)", H, n, n);
            save_matrix_to_file("cpp_G.txt", "G (LandauerQD.cpp)", G, n, n);
            save_matrix_to_file("cpp_Gdag.txt", "Gdag (LandauerQD.cpp)", Gdag, n, n);
            save_matrix_to_file("cpp_Gamma_t.txt", "Gamma_t (LandauerQD.cpp)", Gammat, n, n);
            save_matrix_to_file("cpp_Gamma_s.txt", "Gamma_s (LandauerQD.cpp)", Gammas, n, n);
            save_matrix_to_file("cpp_Gammat_Gdag.txt", "Gamma_t_G_dag (LandauerQD.cpp)", Gammat_Gdag, n, n);
            save_matrix_to_file("cpp_G_Gammat_Gdag.txt", "G_Gamma_t_G_dag (LandauerQD.cpp)", G_Gammat_Gdag, n, n);
            save_matrix_to_file("cpp_Tmat.txt", "Tmat = Gamma_s @ G @ Gamma_t @ Gdag (LandauerQD.cpp)", Tmat, n, n);
        }

        // Calculate trace
        Vec2d trace = {0.0, 0.0};
        for(int i = 0; i < n; i++) {
            trace = trace + Tmat[i * n + i];
        }

        printf("calculate_transmission_from_H() DONE Tr(Tmat): (real=%g,imag=%g) \n", trace.x, trace.y);
        return trace.x;  // Return real part of trace
    }

    double calculate_transmission(  double E, const Vec3d& tip_pos, double Q_tip=0.0, Vec2d* Hqd=nullptr) {
        printf("calculate_transmission(): E = %g, tip_pos = (%g, %g, %g), Q_tip = %g\n", E, tip_pos.x, tip_pos.y, tip_pos.z, Q_tip);
        int n = n_qds + 2;
        Vec2d H[n * n];
        make_full_hamiltonian(tip_pos, H, Q_tip, Hqd);
        return calculate_transmission_from_H(H, E);
    }

    void scan_1D(const Vec3d* ps_line, int n_points, const double* energies, int n_energies, 
                 double* transmissions, double Q_tip=0.0, Vec2d* H_QDs=nullptr) {
        int n = n_qds + 2;
        Vec2d H[n * n];

        for(int i = 0; i < n_points; i++) {
            if(H_QDs != nullptr) {
                make_full_hamiltonian(ps_line[i], H, 0.0, &H_QDs[i * n_qds * n_qds]);
            } else {
                make_full_hamiltonian(ps_line[i], H, Q_tip);
            }
            for(int j = 0; j < n_energies; j++) {
                transmissions[i * n_energies + j] = calculate_transmission_from_H(H, energies[j]);
            }
        }
    }
};

// C interface
extern "C" {

static LandauerQDs* g_system = nullptr;

void initLandauerQDs(int n_qds, double* QDpos, double* Esite_, double K, double decay, double tS,
                    double E_sub, double E_tip, double tA, double eta, double Gamma_tip, double Gamma_sub,
                    int debug, int verbosity) {
    if (g_system != nullptr) {
        delete g_system;
    }
    g_system = new LandauerQDs( (Vec3d*)QDpos, n_qds, Esite_, K, decay, tS, E_sub, E_tip, tA, eta, Gamma_tip, Gamma_sub, debug, verbosity);
}

void deleteLandauerQDs() {
    if (g_system != nullptr) {
        delete g_system;
        g_system = nullptr;
    }
}

double calculate_transmission(double E, double* tip_pos, double Q_tip, double* Hqd ) {
    if (g_system == nullptr) { printf("ERROR calculate_transmission(): System not initialized\n"); exit(0); }
    return g_system->calculate_transmission( E, *(Vec3d*)tip_pos,  Q_tip, (Vec2d*)Hqd );
    //return g_system->calculate_transmission_from_H(E, (Vec2d*)H );
}

void calculate_transmissions( int nE, double* energies, int npos, double* ptips_, double* Qtips,  double* Hqds_, double* transmissions) {
    if (g_system == nullptr) { printf("ERROR calculate_transmissions(): System not initialized\n"); exit(0);  }
    Vec3d* ptips = (Vec3d*)ptips_;
    Vec2d* Hqds  = (Vec2d*)Hqds_;
    
    for(int i = 0; i < npos; i++) {
        for(int j = 0; j < nE; j++) {
            double transmission = g_system->calculate_transmission( energies[j], ptips[i], Qtips[i], (Vec2d*)Hqds );
            transmissions[i * nE + j] = transmission;
        }
    }
}

} // extern "C"
