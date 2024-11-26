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
        if(debug) {
            write_matrix(0,              "Hqd0 (LandauerQD.cpp)", Hqd0, n_qds, n_qds);
            write_matrix("cpp_Hqd0.txt", "Hqd0 (LandauerQD.cpp)", Hqd0, n_qds, n_qds);
        }
    }

    ~LandauerQDs() {
        delete[] H_sub_QD;
        delete[] Hqd0;
        //delete[] K_matrix;
    }

    void calculate_greens_function(double E, Vec2d* H, Vec2d* G) {
        if(debug) printf("calculate_greens_function(): E = %g\n", E);
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
        const double MIN_DIST = 1e-3;  // Minimum distance to prevent division by zero
        Vec3d d = tip_pos - qd_pos;
        double dist = d.norm();
        if (dist < MIN_DIST){ 
            printf("ERROR in calculate_tip_induced_shift(): dist(%g)<MIN_DIST(%g) tip_pos(%g,%g,%g) qd_pos(%g,%g,%g) \n", 
                   dist, MIN_DIST, tip_pos.x, tip_pos.y, tip_pos.z, qd_pos.x, qd_pos.y, qd_pos.z); 
            exit(0); 
        }
        return COULOMB_CONST * Q_tip / dist;
    }

    void makeHqd(Vec3d tip_pos, double Q_tip, Vec2d* Hqd) {
        // Copy base Hamiltonian and ensure memory is synchronized
        memcpy(Hqd, Hqd0, sizeof(Vec2d) * n_qds * n_qds);
        
        // Add tip-induced shifts to diagonal elements if Q_tip is provided
        if(Q_tip != 0.0) {
            for(int i = 0; i < n_qds; i++) {
                double shift = calculate_tip_induced_shift(tip_pos, QDpos[i], Q_tip);
                Hqd[i * n_qds + i].x += shift;
                if(debug) printf("cpp:makeHqd() shift[%i] = %g\n", i, shift);
            }
        }
        
        if(debug) {
            write_matrix(0, "Hqd (LandauerQD.cpp)", Hqd, n_qds, n_qds);
            write_matrix("cpp_Hqd.txt", "Hqd (LandauerQD.cpp)", Hqd, n_qds, n_qds);
        }
    }

    void assemble_full_H(const Vec3d& tip_pos, Vec2d* Hqd, Vec2d* H) {
        int n = n_qds + 2;
        
        // Initialize H to zero and ensure memory is synchronized
        memset(H, 0, sizeof(Vec2d) * n * n);
        
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
            if(debug) printf("cpp::assemble_full_H(): tip_coupling[%i] = %g\n", i, t);
            H[(i + 1) * n + (n - 1)] = Vec2d{ t, 0 };  // Last column
            H[(n - 1) * n + (i + 1)] = Vec2d{ t, 0 };  // Last row (conjugate)
        }
    }

    void make_full_hamiltonian(const Vec3d& tip_pos, Vec2d* H, double Q_tip=0.0, Vec2d* Hqd_in=nullptr) {
        //printf("make_full_hamiltonian()\n" );
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
        //printf("calculate_transmission_from_H()\n" );
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

            write_matrix( 0, "H (LandauerQD.cpp)", H, n, n);

            write_matrix("cpp_H.txt", "H (LandauerQD.cpp)", H, n, n);
            write_matrix("cpp_G.txt", "G (LandauerQD.cpp)", G, n, n);
            write_matrix("cpp_Gdag.txt", "Gdag (LandauerQD.cpp)", Gdag, n, n);
            write_matrix("cpp_Gamma_t.txt", "Gamma_t (LandauerQD.cpp)", Gammat, n, n);
            write_matrix("cpp_Gamma_s.txt", "Gamma_s (LandauerQD.cpp)", Gammas, n, n);
            write_matrix("cpp_Gammat_Gdag.txt", "Gamma_t_G_dag (LandauerQD.cpp)", Gammat_Gdag, n, n);
            write_matrix("cpp_G_Gammat_Gdag.txt", "G_Gamma_t_G_dag (LandauerQD.cpp)", G_Gammat_Gdag, n, n);
            write_matrix("cpp_Tmat.txt", "Tmat = Gamma_s @ G @ Gamma_t @ Gdag (LandauerQD.cpp)", Tmat, n, n);
        }

        // Calculate trace
        Vec2d trace = {0.0, 0.0};
        for(int i = 0; i < n; i++) {
            trace = trace + Tmat[i * n + i];
        }

        //printf("calculate_transmission_from_H() DONE Tr(Tmat): (real=%g,imag=%g) \n", trace.x, trace.y);
        return trace.x;  // Return real part of trace
    }

    double calculate_transmission(  double E, const Vec3d& tip_pos, double Q_tip=0.0, Vec2d* Hqd=nullptr) {
        //printf("calculate_transmission(): E = %g, tip_pos = (%g, %g, %g), Q_tip = %g\n", E, tip_pos.x, tip_pos.y, tip_pos.z, Q_tip);
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

    void scan_1D(int nE, double* energies, int npos, double* ptips_, double* Qtips, double* Hqds_, double* transmissions) {
        // Convert tip positions array to Vec3d array for easier handling
        Vec3d* tip_positions = (Vec3d*)ptips_;
        
        // Size of full Hamiltonian (QDs + tip + substrate)
        const int n = n_qds + 2;
        const int n2 = n * n;

        int    nqd2 = n_qds * n_qds;
        Vec2d* Hqds = (Vec2d*)Hqds_;
        
        // Temporary arrays for each thread
        #pragma omp parallel
        {
            // Thread-local storage for matrices
            Vec2d H[n2];
            Vec2d G[n2];
            
            // Parallel loop over positions
            #pragma omp for schedule(dynamic)
            for(int i = 0; i < npos; i++) {
                Vec3d tip_pos = tip_positions[i];
                double Q_tip = Qtips ? Qtips[i] : 0.0;
                
                // Get the Hamiltonian
                if(Hqds) {
                    make_full_hamiltonian(tip_pos, H, Q_tip, Hqds + i*nqd2 );
                } else {
                    // Compute QD Hamiltonian with tip effects
                    make_full_hamiltonian(tip_pos, H, Q_tip);
                }
                
                // Calculate transmission for each energy
                for(int j = 0; j < nE; j++) {
                    double E = energies[j];
                    transmissions[i * nE + j] = calculate_transmission_from_H(H, E);
                }
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

double calculate_transmission(double E, double* tip_pos_, double Q_tip, double* Hqd ) {
    if(g_system == nullptr) {
        printf("Error: System not initialized\n");
        return 0.0;
    }
    Vec3d tip_pos = {tip_pos_[0], tip_pos_[1], tip_pos_[2]};
    return g_system->calculate_transmission(E, tip_pos, Q_tip, (Vec2d*)Hqd);
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

void scan_1D(int nE, double* energies, int npos, double* ptips_, double* Qtips, double* Hqds_, double* transmissions) {
    if(g_system) {
        g_system->scan_1D(nE, energies, npos, ptips_, Qtips, Hqds_, transmissions);
    }
}

} // extern "C"
