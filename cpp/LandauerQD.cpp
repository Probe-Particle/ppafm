#include "LandauerQD.hpp"
#include <stdlib.h>

LandauerQDs::LandauerQDs(int n_qds_, Vec3d* QDpos_, double* Esite_,
                         double K_, double decay_, double tS_,
                         double E_sub_, double E_tip_, double tA_,
                         double eta_, double Gamma_tip_, double Gamma_sub_) 
    : n_qds(n_qds_), QDpos(QDpos_), Esite(Esite_),
      K(K_), decay(decay_), tS(tS_),
      E_sub(E_sub_), E_tip(E_tip_), tA(tA_),
      eta(eta_), Gamma_tip(Gamma_tip_), Gamma_sub(Gamma_sub_)
{
    // Allocate memory for matrices and workspaces
    H_sub_QD = new Vec2d[n_qds];
    K_matrix = new double[n_qds * n_qds];
    H_QD_no_tip = new Vec2d[(n_qds + 1) * (n_qds + 1)];
    workspace = new Vec2d[n_qds * n_qds];
    G_workspace = new Vec2d[(n_qds + 2) * (n_qds + 2)];

    // Initialize substrate-QD coupling
    for(int i = 0; i < n_qds; i++) {
        H_sub_QD[i] = Vec2d{tS, 0.0};
    }

    // Initialize Coulomb interaction matrix
    for(int i = 0; i < n_qds; i++) {
        for(int j = 0; j < n_qds; j++) {
            K_matrix[i * n_qds + j] = (i != j) ? K : 0.0;
        }
    }

    // Initialize H_QD_no_tip
    // First row/column is substrate
    H_QD_no_tip[0] = Vec2d{E_sub, 0.0};
    for(int i = 0; i < n_qds; i++) {
        // Substrate-QD coupling
        H_QD_no_tip[(i + 1) * (n_qds + 1)] = H_sub_QD[i];           // Column
        H_QD_no_tip[i + 1] = Vec2d{H_sub_QD[i].x, -H_sub_QD[i].y};  // Row (conjugate)
        
        // QD diagonal elements
        H_QD_no_tip[(i + 1) * (n_qds + 1) + (i + 1)] = Vec2d{Esite[i], 0.0};
        
        // QD-QD Coulomb interactions
        for(int j = 0; j < n_qds; j++) {
            if(i != j) {
                H_QD_no_tip[(i + 1) * (n_qds + 1) + (j + 1)] = Vec2d{K_matrix[i * n_qds + j], 0.0};
            }
        }
    }
}

LandauerQDs::~LandauerQDs() {
    delete[] H_sub_QD;
    delete[] K_matrix;
    delete[] H_QD_no_tip;
    delete[] workspace;
    delete[] G_workspace;
}

void LandauerQDs::calculate_greens_function(double E, Vec2d* H_QD, Vec2d* G_out) {
    int size = n_qds + 2;  // Include substrate and tip
    
    // G = (E + iη)I - H
    for(int i = 0; i < size * size; i++) {
        G_out[i] = (i % (size + 1) == 0) ? Vec2d{E, eta} - H_QD[i] : Vec2d{0.0, 0.0} - H_QD[i];
    }
    
    // Invert G using ComplexAlgebra routines
    invert_complex_matrix(size, G_out, G_out, workspace);
}

void LandauerQDs::calculate_gamma(Vec2d* coupling_vector, Vec2d* gamma_out, int size) {
    // Γ = 2π * V * V†
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
            Vec2d vi = coupling_vector[i];
            Vec2d vj = coupling_vector[j];
            // Compute conjugate of vj
            vj.y = -vj.y;
            gamma_out[i * size + j].set_mul_cmplx(vi, vj);
            gamma_out[i * size + j] = gamma_out[i * size + j] * (2.0 * M_PI);
        }
    }
}

void LandauerQDs::calculate_tip_coupling(const Vec3d& tip_pos, Vec2d* tip_couplings) {
    for(int i = 0; i < n_qds; i++) {
        double d = (tip_pos - QDpos[i]).norm();
        double t = tA * exp(-decay * d);
        tip_couplings[i] = Vec2d{t, 0.0};
    }
}

void LandauerQDs::calculate_tip_induced_shifts(const Vec3d& tip_pos, double Q_tip, double* shifts) {
    for(int i = 0; i < n_qds; i++) {
        Vec3d d = tip_pos - QDpos[i];
        shifts[i] = COULOMB_CONST * Q_tip / d.norm();
    }
}

void LandauerQDs::make_QD_block(const Vec3d& tip_pos, double Q_tip, Vec2d* H_QD_out) {
    // Copy the pre-computed H_QD_no_tip
    for(int i = 0; i < (n_qds + 1) * (n_qds + 1); i++) {
        H_QD_out[i] = H_QD_no_tip[i];
    }
    
    if(Q_tip != 0.0) {
        // Calculate energy shifts from tip
        double* shifts = new double[n_qds];
        calculate_tip_induced_shifts(tip_pos, Q_tip, shifts);
        
        // Add shifts to diagonal elements
        for(int i = 0; i < n_qds; i++) {
            H_QD_out[(i + 1) * (n_qds + 1) + (i + 1)].x += shifts[i];
        }
        
        delete[] shifts;
    }
}

void LandauerQDs::make_QD_block_with_charge(const Vec3d& tip_pos, double Q_tip, double* Qsites, Vec2d* H_QD_out) {
    // First make the basic QD block with tip-induced shifts
    make_QD_block(tip_pos, Q_tip, H_QD_out);
    
    // Then add the site charges contribution if provided
    if (Qsites != nullptr) {
        for(int i = 0; i < n_qds; i++) {
            for(int j = 0; j < n_qds; j++) {
                if (i != j) {
                    // Add Coulomb interaction from site charges
                    Vec3d d = QDpos[i] - QDpos[j];
                    double dE = COULOMB_CONST * Qsites[j] / d.norm();
                    H_QD_out[(i + 1) * (n_qds + 1) + (i + 1)].x += dE;
                }
            }
        }
    }
}

void LandauerQDs::assemble_full_H(const Vec3d& tip_pos, Vec2d* QD_block, Vec2d* H_out) {
    int full_size = n_qds + 2;  // Include substrate and tip
    
    // Calculate tip couplings
    Vec2d* tip_couplings = new Vec2d[n_qds];
    calculate_tip_coupling(tip_pos, tip_couplings);
    
    // Copy QD block
    for(int i = 0; i < n_qds + 1; i++) {
        for(int j = 0; j < n_qds + 1; j++) {
            H_out[i * full_size + j] = QD_block[i * (n_qds + 1) + j];
        }
    }
    
    // Add tip row/column
    H_out[full_size * (full_size - 1)] = Vec2d{E_tip, 0.0};  // Tip diagonal element
    
    // Add tip-QD couplings
    for(int i = 0; i < n_qds; i++) {
        // Tip-QD coupling
        H_out[(full_size - 1) * full_size + (i + 1)] = tip_couplings[i];  // Column
        H_out[(i + 1) * full_size + (full_size - 1)] = Vec2d{tip_couplings[i].x, -tip_couplings[i].y};  // Row (conjugate)
    }
    
    delete[] tip_couplings;
}

double LandauerQDs::calculate_transmission(double E, const Vec3d& tip_pos, Vec2d* H_QD) {
    int full_size = n_qds + 2;
    
    // Allocate temporary matrices
    Vec2d* H_full = new Vec2d[full_size * full_size];
    Vec2d* G = new Vec2d[full_size * full_size];
    Vec2d* gamma_L = new Vec2d[full_size * full_size];
    Vec2d* gamma_R = new Vec2d[full_size * full_size];
    Vec2d* temp = new Vec2d[full_size * full_size];
    
    // Prepare Hamiltonian
    if (H_QD == nullptr) {
        Vec2d* H_QD_block = new Vec2d[(n_qds + 1) * (n_qds + 1)];
        make_QD_block(tip_pos, 0.0, H_QD_block);
        assemble_full_H(tip_pos, H_QD_block, H_full);
        delete[] H_QD_block;
    } else {
        assemble_full_H(tip_pos, H_QD, H_full);
    }
    
    // Calculate Green's function
    calculate_greens_function(E, H_full, G);
    
    // Prepare coupling vectors for substrate (L) and tip (R)
    Vec2d* V_L = new Vec2d[full_size]();  // Initialize to zero
    Vec2d* V_R = new Vec2d[full_size]();  // Initialize to zero
    
    // Substrate coupling
    for(int i = 0; i < n_qds; i++) {
        V_L[i + 1] = H_sub_QD[i];
    }
    
    // Tip coupling
    Vec2d* tip_couplings = new Vec2d[n_qds];
    calculate_tip_coupling(tip_pos, tip_couplings);
    for(int i = 0; i < n_qds; i++) {
        V_R[i + 1] = tip_couplings[i];
    }
    
    // Calculate gamma matrices
    calculate_gamma(V_L, gamma_L, full_size);
    calculate_gamma(V_R, gamma_R, full_size);
    
    // Calculate transmission: T = Tr(ΓL * G * ΓR * G†)
    multiply_complex_matrices(full_size, gamma_L, G, temp);
    multiply_complex_matrices(full_size, temp, gamma_R, workspace);
    
    // Conjugate G for G†
    for(int i = 0; i < full_size * full_size; i++) {
        G[i].y = -G[i].y;
    }
    
    multiply_complex_matrices(full_size, workspace, G, temp);
    
    // Calculate trace
    double transmission = 0.0;
    for(int i = 0; i < full_size; i++) {
        transmission += temp[i * full_size + i].x;
    }
    
    // Clean up
    delete[] H_full;
    delete[] G;
    delete[] gamma_L;
    delete[] gamma_R;
    delete[] temp;
    delete[] V_L;
    delete[] V_R;
    delete[] tip_couplings;
    
    return transmission;
}

double LandauerQDs::calculate_transmission(double E, const Vec3d& tip_pos) {
    return calculate_transmission(E, tip_pos, nullptr);
}

void LandauerQDs::calculate_occupancies(const Vec3d& tip_pos, double Q_tip, double* occupancies) {
    // Size of the full system (substrate + QDs + tip)
    int full_size = n_qds + 2;
    
    // Allocate temporary matrices
    Vec2d* H_full = new Vec2d[full_size * full_size];
    Vec2d* G = new Vec2d[full_size * full_size];
    
    // Make QD block and assemble full Hamiltonian
    Vec2d* H_QD_block = new Vec2d[(n_qds + 1) * (n_qds + 1)];
    make_QD_block(tip_pos, Q_tip, H_QD_block);
    assemble_full_H(tip_pos, H_QD_block, H_full);
    
    // Calculate Green's function at E = 0 (Fermi energy)
    calculate_greens_function(0.0, H_full, G);
    
    // Calculate occupancies from the imaginary part of the diagonal elements
    for(int i = 0; i < n_qds; i++) {
        occupancies[i] = -G[(i + 1) * full_size + (i + 1)].y / M_PI;
    }
    
    // Clean up
    delete[] H_full;
    delete[] G;
    delete[] H_QD_block;
}

// Global pointer to store the current instance
static LandauerQDs* g_system = nullptr;

extern "C" {

// Initialize the system
void initLandauerQDs(int n_qds, double* QDpos_, double* Esite_,
                     double K, double decay, double tS,
                     double E_sub, double E_tip, double tA,
                     double eta, double Gamma_tip, double Gamma_sub) {
    if (g_system != nullptr) {
        delete g_system;
    }
    Vec3d* QDpos = reinterpret_cast<Vec3d*>(QDpos_);
    g_system = new LandauerQDs(n_qds, QDpos, Esite_, K, decay, tS,
                              E_sub, E_tip, tA, eta, Gamma_tip, Gamma_sub);
}

// Clean up
void deleteLandauerQDs() {
    if (g_system != nullptr) {
        delete g_system;
        g_system = nullptr;
    }
}

// Calculate transmission for multiple tip positions and energies
void calculateTransmissions(int npos, double* ptips_, double* energies, int nE, double* H_QDs, double* transmissions) {
    if (g_system == nullptr) return;
    
    Vec3d* ptips    = reinterpret_cast<Vec3d*>(ptips_);
    Vec2d* H_QDs_complex = reinterpret_cast<Vec2d*>(H_QDs);
    
    // For each position and energy
    for(int i = 0; i < npos; i++) {
        for(int j = 0; j < nE; j++) {
            double E = energies[j];
            if (H_QDs != nullptr) {
                // Use pre-computed Hamiltonian
                int offset = i * (g_system->n_qds + 1) * (g_system->n_qds + 1);
                transmissions[i * nE + j] = g_system->calculate_transmission(E, ptips[i], H_QDs_complex + offset);
            } else {
                // Calculate Hamiltonian on the fly
                transmissions[i * nE + j] = g_system->calculate_transmission(E, ptips[i]);
            }
        }
    }
}

// Solve Hamiltonians for multiple positions
void solveHamiltonians(int npos, double* ptips_, double* Qtips, double* Qsites, 
                      double* evals, double* evecs, double* Hs, double* Gs) {
    if (g_system == nullptr) return;
    
    Vec3d* ptips = reinterpret_cast<Vec3d*>(ptips_);
    int H_size = (g_system->n_qds + 1);
    int H_size2 = H_size * H_size;
    
    for(int i = 0; i < npos; i++) {
        Vec2d* H_QD_block = new Vec2d[H_size2];
        
        // Make Hamiltonian block with charge effects if Qsites is provided
        if (Qsites != nullptr) {
            g_system->make_QD_block_with_charge(ptips[i], Qtips[i], Qsites + i * g_system->n_qds, H_QD_block);
        } else {
            g_system->make_QD_block(ptips[i], Qtips[i], H_QD_block);
        }
        
        // Copy Hamiltonian to output if requested
        if (Hs != nullptr) {
            for(int j = 0; j < H_size2; j++) {
                Hs[i * H_size2 + j] = H_QD_block[j].x;  // Store only real part
            }
        }
        
        delete[] H_QD_block;
    }
}

// Solve site occupancies for multiple positions
void solveSiteOccupancies(int npos, double* ptips_, double* Qtips, double* Qout) {
    if (g_system == nullptr) return;
    
    Vec3d* ptips = reinterpret_cast<Vec3d*>(ptips_);
    
    // For each position
    for(int i = 0; i < npos; i++) {
        g_system->calculate_occupancies(ptips[i], Qtips[i], Qout + i * g_system->n_qds);
    }
}

} // extern "C"
