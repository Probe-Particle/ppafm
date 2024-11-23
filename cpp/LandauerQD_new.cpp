#include <cmath>
#include <cstring>
#include "Vec2.h"
#include "ComplexAlgebra.hpp"

class LandauerQDs {
public:
    // Parameters
    Vec2d* QDpos;
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
    Vec2d* H_sub_QD;
    Vec2d* K_matrix;
    Vec2d* H_QD_no_tip;

    LandauerQDs(Vec2d* QDpos_, int n_qds_, double* Esite_, double K_=0.01, double decay_=1.0, 
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
        K_matrix = new Vec2d[n_qds * n_qds];
        H_QD_no_tip = new Vec2d[(n_qds + 1) * (n_qds + 1)];

        // Initialize H_sub_QD (constant coupling)
        for(int i = 0; i < n_qds; i++) {
            H_sub_QD[i] = {tS, 0.0};
        }

        // Initialize K_matrix (Coulomb interaction)
        for(int i = 0; i < n_qds; i++) {
            for(int j = 0; j < n_qds; j++) {
                K_matrix[i * n_qds + j] = {(i != j) ? K : 0.0, 0.0};
            }
        }

        // Initialize H_QD_no_tip
        // First row/column: substrate coupling
        H_QD_no_tip[0] = {E_sub, 0.0};
        for(int i = 0; i < n_qds; i++) {
            H_QD_no_tip[(i + 1) * (n_qds + 1)] = H_sub_QD[i];  // First column
            H_QD_no_tip[i + 1] = H_sub_QD[i];                  // First row
        }

        // QD block with Coulomb interactions
        for(int i = 0; i < n_qds; i++) {
            for(int j = 0; j < n_qds; j++) {
                double val = (i == j) ? Esite[i] : 0.0;
                H_QD_no_tip[(i + 1) * (n_qds + 1) + (j + 1)] = {val + K_matrix[i * n_qds + j].x, 0.0};
            }
        }
    }

    ~LandauerQDs() {
        delete[] H_sub_QD;
        delete[] K_matrix;
        delete[] H_QD_no_tip;
    }

    void calculate_greens_function(double E, Vec2d* H_QD, Vec2d* G) {
        int n = n_qds + 2;
        int n2 = n * n;
        
        // Create (E + iη)I - H
        Vec2d A[n2];
        for(int i = 0; i < n2; i++) {
            A[i] = {-H_QD[i].x, -H_QD[i].y};
            if(i % (n + 1) == 0) {  // Diagonal elements
                A[i].x += E;
                A[i].y += eta;
            }
        }

        Vec2d workspace[n2];  // Temporary workspace for inversion
        invert_complex_matrix(n, A, G, workspace);
    }

    void calculate_tip_coupling(const Vec2d& tip_pos, Vec2d* tip_couplings) {
        for(int i = 0; i < n_qds; i++) {
            double dx = tip_pos.x - QDpos[i].x;
            double dy = tip_pos.y - QDpos[i].y;
            double dz = tip_pos.z - QDpos[i].z;
            double d = sqrt(dx*dx + dy*dy + dz*dz);
            tip_couplings[i] = {tA * exp(-decay * d), 0.0};
        }
    }

    void calculate_tip_induced_shifts(const Vec2d& tip_pos, double Q_tip, double* shifts) {
        const double COULOMB_CONST = 14.3996;  // eV*Å/e
        for(int i = 0; i < n_qds; i++) {
            double dx = tip_pos.x - QDpos[i].x;
            double dy = tip_pos.y - QDpos[i].y;
            double dz = tip_pos.z - QDpos[i].z;
            double d = sqrt(dx*dx + dy*dy + dz*dz);
            shifts[i] = COULOMB_CONST * Q_tip / d;
        }
    }

    void assemble_full_H(const Vec2d& tip_pos, Vec2d* QD_block, Vec2d* H) {
        int n = n_qds + 2;
        Vec2d* tip_couplings = new Vec2d[n_qds];
        
        // Calculate tip coupling
        calculate_tip_coupling(tip_pos, tip_couplings);
        
        // Initialize H to zero
        for(int i = 0; i < n * n; i++) {
            H[i] = {0.0, 0.0};
        }
        
        // Fill QD block
        for(int i = 0; i < n_qds; i++) {
            for(int j = 0; j < n_qds; j++) {
                H[(i + 1) * n + (j + 1)] = QD_block[i * n_qds + j];
                if(i == j) H[(i + 1) * n + (j + 1)].y -= eta;
            }
        }
        
        // Fill substrate part
        H[0] = {E_sub, -Gamma_sub};
        for(int i = 0; i < n_qds; i++) {
            H[i + 1] = H_sub_QD[i];                 // First row
            H[(i + 1) * n] = H_sub_QD[i];        // First column
        }
        
        // Fill tip part
        H[n * n - 1] = {E_tip, -Gamma_tip};
        for(int i = 0; i < n_qds; i++) {
            H[(i + 1) * n + (n - 1)] = tip_couplings[i];           // Last column
            H[(n - 1) * n + (i + 1)] = {tip_couplings[i].x, -tip_couplings[i].y}; // Last row (conjugate)
        }
        
        delete[] tip_couplings;
    }

    void calculate_coupling_matrices(Vec2d* Gamma_s, Vec2d* Gamma_t) {
        int n = n_qds + 2;
        
        // Initialize to zero
        for(int i = 0; i < n * n; i++) {
            Gamma_s[i] = {0.0, 0.0};
            Gamma_t[i] = {0.0, 0.0};
        }
        
        // Set coupling strengths
        Gamma_s[0] = {2.0 * Gamma_sub, 0.0};               // Substrate coupling
        Gamma_t[n * n - 1] = {2.0 * Gamma_tip, 0.0}; // Tip coupling
    }

    double calculate_transmission(const Vec2d& tip_pos, double energy, Vec2d* H) {
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
        calculate_greens_function(energy, H, G);

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
        Gammas[0] = {2.0 * Gamma_sub, 0.0};        // Substrate coupling
        Gammat[n2-1] = {2.0 * Gamma_tip, 0.0};     // Tip coupling

        // Calculate transmission using Caroli formula: T = Tr(Gamma_s @ G @ Gamma_t @ G†)
        multiply_complex_matrices(n, Gammat, Gdag, Gammat_Gdag);
        multiply_complex_matrices(n, G, Gammat_Gdag, G_Gammat_Gdag);
        multiply_complex_matrices(n, Gammas, G_Gammat_Gdag, Tmat);

        // Save matrices for debugging
        if(debug) {
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

        return trace.x;  // Return real part of trace
    }
};
