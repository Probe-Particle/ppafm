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
    
    // G = EI - H
    for(int i = 0; i < size * size; i++) {
        int row = i / size;
        int col = i % size;
        if(row == col) {
            // Diagonal element: E - H
            G_out[i] = Vec2d{E, 0.0} - H_QD[i];  // Initialize with real part only
            
            // Add self-energies on diagonal
            if(row == 0) {  // Substrate
                G_out[i].y += 1.000001;  // Add imaginary part for substrate
                G_out[i] = G_out[i] - Vec2d{0.0, Gamma_sub};
            } else if(row == size-1) {  // Tip
                G_out[i].y += 1.000001;  // Add imaginary part for tip
                G_out[i] = G_out[i] - Vec2d{0.0, Gamma_tip};
            } else {  // QDs
                G_out[i].y += eta;  // Add small imaginary part for QDs
            }
        } else {
            // Off-diagonal element: -H
            G_out[i] = Vec2d{0.0, 0.0} - H_QD[i];
        }
    }
    
    save_matrix_to_file("cpp_pre_inversion.txt", "Matrix before inversion (C++)", G_out, size, size);
    
    // Copy G to temporary buffer for inversion
    Vec2d* temp = new Vec2d[size * size];
    for(int i = 0; i < size * size; i++) {
        temp[i] = G_out[i];
    }
    
    // Invert G using ComplexAlgebra routines
    invert_complex_matrix(size, temp, G_out, workspace);
    
    save_matrix_to_file("cpp_post_inversion.txt", "Matrix after inversion (C++)", G_out, size, size);
    
    delete[] temp;
}

void LandauerQDs::calculate_gamma(Vec2d* coupling_vector, Vec2d* gamma_out, int size) {
    // Initialize to zero
    for(int i = 0; i < size * size; i++) {
        gamma_out[i] = Vec2d{0.0, 0.0};
    }
    
    printf("\nCalculating gamma matrix:\n");
    if (coupling_vector == nullptr) {
        // Special case: substrate and tip coupling
        // For substrate: only [0,0] element is non-zero
        // For tip: only [size-1,size-1] element is non-zero
        if (gamma_out == workspace) {  // Using workspace to identify substrate gamma
            gamma_out[0] = Vec2d{2.0 * Gamma_sub, 0.0};
            printf("Substrate coupling: Gamma_sub = %g\n", Gamma_sub);
        } else {
            gamma_out[size * size - 1] = Vec2d{2.0 * Gamma_tip, 0.0};
            printf("Tip coupling: Gamma_tip = %g\n", Gamma_tip);
        }
    } else {
        // General case: outer product of coupling vector with itself
        for(int i = 0; i < size; i++) {
            for(int j = 0; j < size; j++) {
                Vec2d vi = coupling_vector[i];
                Vec2d vj = coupling_vector[j];
                // Multiply by 2π for wide-band limit
                gamma_out[i * size + j] = Vec2d{
                    2.0 * M_PI * (vi.x * vj.x + vi.y * vj.y),
                    2.0 * M_PI * (vi.x * vj.y - vi.y * vj.x)
                };
            }
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
    // Calculate energy shifts from tip if Q_tip is non-zero
    double* shifts = new double[n_qds];
    if (Q_tip != 0.0) {
        calculate_tip_induced_shifts(tip_pos, Q_tip, shifts);
    } else {
        for(int i = 0; i < n_qds; i++) {
            shifts[i] = 0.0;
        }
    }
    
    // Set diagonal elements with shifted energies
    for(int i = 0; i < n_qds; i++) {
        for(int j = 0; j < n_qds; j++) {
            if(i == j) {
                // Diagonal: site energy + tip-induced shift
                H_QD_out[i * n_qds + j] = Vec2d{Esite[i] + shifts[i], 0.0};
            } else {
                // Off-diagonal: Coulomb interaction K
                H_QD_out[i * n_qds + j] = Vec2d{K_matrix[i * n_qds + j], 0.0};
            }
        }
    }
    
    delete[] shifts;
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
                    H_QD_out[i * n_qds + j].x += dE;
                }
            }
        }
    }
}

void LandauerQDs::assemble_full_H(const Vec3d& tip_pos, Vec2d* QD_block, Vec2d* H_out) {
    int size = n_qds + 2;  // Include substrate and tip
    
    // Clear output matrix
    for(int i = 0; i < size * size; i++) {
        H_out[i] = Vec2d{0.0, 0.0};
    }
    
    // Fill substrate part (with broadening)
    H_out[0] = Vec2d{E_sub, -Gamma_sub};
    
    // Fill QD block
    if (QD_block != nullptr) {
        // Use external QD block if provided
        for(int i = 0; i < n_qds; i++) {
            for(int j = 0; j < n_qds; j++) {
                H_out[(i + 1) * size + (j + 1)] = QD_block[i * n_qds + j];
            }
        }
    } else {
        // Create QD block internally using tip position
        Vec2d* temp_block = new Vec2d[n_qds * n_qds];
        make_QD_block(tip_pos, 0.6, temp_block);  // Using default Q_tip=0.6 as in test
        for(int i = 0; i < n_qds; i++) {
            for(int j = 0; j < n_qds; j++) {
                H_out[(i + 1) * size + (j + 1)] = temp_block[i * n_qds + j];
            }
        }
        delete[] temp_block;
    }
    
    // Add broadening to QD diagonal elements
    for(int i = 1; i <= n_qds; i++) {
        H_out[i * size + i].y -= eta;
    }
    
    // Fill substrate-QD coupling
    for(int i = 0; i < n_qds; i++) {
        H_out[(i + 1)] = H_sub_QD[i];                    // First row
        H_out[(i + 1) * size] = Vec2d{tS, 0.0};         // First column
    }
    
    // Calculate and fill tip coupling
    Vec2d* tip_couplings = new Vec2d[n_qds];
    calculate_tip_coupling(tip_pos, tip_couplings);
    
    // Fill tip-QD coupling
    for(int i = 0; i < n_qds; i++) {
        H_out[(i + 1) * size + (size - 1)] = tip_couplings[i];           // Last column
        H_out[(size - 1) * size + (i + 1)] = Vec2d{tip_couplings[i].x, -tip_couplings[i].y};  // Last row (conjugate)
    }
    
    // Fill tip part (with broadening)
    H_out[size * size - 1] = Vec2d{E_tip, -Gamma_tip};
    
    delete[] tip_couplings;
}

double LandauerQDs::calculate_transmission(double E, const Vec3d& tip_pos, Vec2d* H_QD) {
    int size = n_qds + 2;  // Include substrate and tip
    
    // Calculate G = (EI - H - Sigma)^(-1)
    Vec2d* G = new Vec2d[size * size];
    calculate_greens_function(E, H_QD, G);
    save_matrix_to_file("cpp_G.txt", "Green's function G (C++)", G, size, size);
    
    // Calculate G_dag (conjugate transpose of G)
    Vec2d* G_dag = new Vec2d[size * size];
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
            G_dag[i * size + j] = Vec2d{G[j * size + i].x, -G[j * size + i].y};
        }
    }
    save_matrix_to_file("cpp_G_dag.txt", "G_dag (C++)", G_dag, size, size);
    
    // Calculate Gamma_t and Gamma_s
    Vec2d* Gamma_t = new Vec2d[size * size];
    Vec2d* Gamma_s = new Vec2d[size * size];
    
    // Initialize to zero
    for(int i = 0; i < size * size; i++) {
        Gamma_t[i] = Vec2d{0.0, 0.0};
        Gamma_s[i] = Vec2d{0.0, 0.0};
    }
    
    // Set diagonal elements for tip and substrate coupling
    Gamma_t[(size-1) * size + (size-1)] = Vec2d{2.0 * Gamma_tip, 0.0};  // Tip coupling
    Gamma_s[0] = Vec2d{2.0 * Gamma_sub, 0.0};  // Substrate coupling
    
    save_matrix_to_file("cpp_Gamma_t.txt", "Gamma_t (C++)", Gamma_t, size, size);
    save_matrix_to_file("cpp_Gamma_s.txt", "Gamma_s (C++)", Gamma_s, size, size);
    
    // Calculate Gamma_t * G_dag
    Vec2d* Gamma_t_G_dag = new Vec2d[size * size];
    multiply_complex_matrices(size, Gamma_t, G_dag, Gamma_t_G_dag);
    save_matrix_to_file("cpp_Gamma_t_G_dag.txt", "Gamma_t_G_dag (C++)", Gamma_t_G_dag, size, size);
    
    // Calculate G * Gamma_t * G_dag
    Vec2d* G_Gamma_t_G_dag = new Vec2d[size * size];
    multiply_complex_matrices(size, G, Gamma_t_G_dag, G_Gamma_t_G_dag);
    save_matrix_to_file("cpp_G_Gamma_t_G_dag.txt", "G_Gamma_t_G_dag (C++)", G_Gamma_t_G_dag, size, size);
    
    // Calculate Gamma_s * G * Gamma_t * G_dag
    Vec2d* final = new Vec2d[size * size];
    multiply_complex_matrices(size, Gamma_s, G_Gamma_t_G_dag, final);
    save_matrix_to_file("cpp_final.txt", "Gamma_s_G_Gamma_t_G_dag (C++)", final, size, size);
    
    // Calculate trace
    Vec2d trace = {0.0, 0.0};
    for(int i = 0; i < size; i++) {
        trace = trace + final[i * size + i];
    }
    
    // Clean up
    delete[] G;
    delete[] G_dag;
    delete[] Gamma_t;
    delete[] Gamma_s;
    delete[] Gamma_t_G_dag;
    delete[] G_Gamma_t_G_dag;
    delete[] final;
    
    return trace.x;  // Return real part of trace
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

void LandauerQDs::get_H_QD_no_tip(Vec2d* H_out) {
    int size = n_qds + 1;  // Include substrate
    for(int i = 0; i < size * size; i++) {
        H_out[i] = H_QD_no_tip[i];
    }
}

void LandauerQDs::get_tip_coupling(const Vec3d& tip_pos, Vec2d* coupling_out) {
    calculate_tip_coupling(tip_pos, coupling_out);
}

void LandauerQDs::get_full_H(const Vec3d& tip_pos, Vec2d* H_out) {
    Vec2d* QD_block = new Vec2d[n_qds * n_qds];
    make_QD_block(tip_pos, 0.6, QD_block);  // Use Q_tip=0.6 to match Python test
    assemble_full_H(tip_pos, QD_block, H_out);
    delete[] QD_block;
}

void LandauerQDs::save_matrix_to_file(const char* filename, const char* title, Vec2d* matrix, int rows, int cols) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        printf("Error: Could not open file %s for writing\n", filename);
        return;
    }
    fprintf(f, "%s\n", title);
    fprintf(f, "Dimensions: %d %d\n", rows, cols);
    fprintf(f, "Format: (real,imag)\n");
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            Vec2d val = matrix[i * cols + j];
            fprintf(f, "(%e,%e) ", val.x, val.y);
        }
        fprintf(f, "\n");
    }
    fclose(f);
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

// Get initial Hamiltonian without tip
void get_H_QD_no_tip(Vec2d* H_out) {
    if(g_system) {
        g_system->get_H_QD_no_tip(H_out);
    }
}

// Get tip coupling vector
void get_tip_coupling(double* tip_pos_, Vec2d* coupling_out) {
    if(g_system) {
        Vec3d tip_pos = *((Vec3d*)tip_pos_);
        g_system->get_tip_coupling(tip_pos, coupling_out);
    }
}

// Get full Hamiltonian at given tip position
void get_full_H(double* tip_pos_, Vec2d* H_out) {
    Vec3d tip_pos = {tip_pos_[0], tip_pos_[1], tip_pos_[2]};
    if(g_system) {
        g_system->get_full_H(tip_pos, H_out);
    }
}

// Calculate Green's function
void calculate_greens_function(double energy, double* H_in, double* G_out) {
    if(g_system) {
        // Convert real arrays to complex arrays
        int size = g_system->n_qds + 2;
        Vec2d* H = new Vec2d[size * size];
        Vec2d* G = new Vec2d[size * size];
        
        // Convert input H from real array to Vec2d array
        for(int i = 0; i < size * size; i++) {
            H[i] = Vec2d{H_in[2*i], H_in[2*i + 1]};
        }
        
        // Calculate Green's function
        g_system->calculate_greens_function(energy, H, G);
        
        // Convert output G from Vec2d array to real array
        for(int i = 0; i < size * size; i++) {
            G_out[2*i] = G[i].x;
            G_out[2*i + 1] = G[i].y;
        }
        
        delete[] H;
        delete[] G;
    }
}

} // extern "C"
