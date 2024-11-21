#ifndef LANDAUER_QD_H
#define LANDAUER_QD_H

#include "ComplexAlgebra.hpp"
#include "Vec3.h"
#include <math.h>

#define COULOMB_CONST 14.3996  // eV*Å/e

struct LandauerQDs {
    // Parameters
    int     n_qds;          // number of quantum dots
    Vec3d*  QDpos;          // positions of quantum dots [n_qds][3]
    double  K;              // Coulomb interaction between QDs
    double  decay;          // Decay constant for tip-QD coupling
    double* Esite;          // On-site energies for QDs [n_qds]
    double  tS;             // Coupling strength to substrate
    double  E_sub;          // Substrate energy level
    double  E_tip;          // Tip energy level
    double  tA;             // Tip coupling strength prefactor
    double  eta;            // Infinitesimal broadening parameter
    double  Gamma_tip;      // Broadening of tip state
    double  Gamma_sub;      // Broadening of substrate state

    // Pre-computed matrices
    Vec2d*  H_sub_QD;       // Substrate-QD coupling [n_qds]
    double* K_matrix;       // Inter-QD Coulomb interaction [(n_qds)*(n_qds)]
    Vec2d*  H_QD_no_tip;    // Main Hamiltonian block without tip [(n_qds+1)*(n_qds+1)]

    // Workspace buffers (to avoid repeated allocation)
    Vec2d*  workspace;      // General workspace for matrix operations
    Vec2d*  G_workspace;    // Workspace for Green's function calculation

    // Constructor and destructor
    LandauerQDs(int n_qds_, Vec3d* QDpos_, double* Esite_, 
                double K_=0.01, double decay_=1.0, double tS_=0.01,
                double E_sub_=0.0, double E_tip_=0.0, double tA_=0.1,
                double eta_=0.00, double Gamma_tip_=1.0, double Gamma_sub_=1.0);
    ~LandauerQDs();

    // Matrix logging functionality
    void save_matrix_to_file(const char* filename, const char* title, Vec2d* matrix, int rows, int cols);

    // Core functionality
    void calculate_greens_function(double E, Vec2d* H_QD, Vec2d* G_out);
    void calculate_gamma(Vec2d* coupling_vector, Vec2d* gamma_out, int size);
    void calculate_tip_coupling(const Vec3d& tip_pos, Vec2d* tip_couplings);
    void calculate_tip_induced_shifts(const Vec3d& tip_pos, double Q_tip, double* shifts);
    void make_QD_block(const Vec3d& tip_pos, double Q_tip, Vec2d* H_QD_out);
    void assemble_full_H(const Vec3d& tip_pos, Vec2d* QD_block, Vec2d* H_out);
    
    // Transmission calculation
    double calculate_transmission(double E, const Vec3d& tip_pos);
    double calculate_transmission(double E, const Vec3d& tip_pos, Vec2d* H_QD);
    
    // Occupancy calculation
    void calculate_occupancies(const Vec3d& tip_pos, double Q_tip, double* occupancies);
    void make_QD_block_with_charge(const Vec3d& tip_pos, double Q_tip, double* Qsites, Vec2d* H_QD_out);

    // Functions for testing and comparison
    void get_H_QD_no_tip(Vec2d* H_out);
    void get_tip_coupling(const Vec3d& tip_pos, Vec2d* coupling_out);
    void get_full_H(const Vec3d& tip_pos, Vec2d* H_out);
};

// C interface
extern "C" {
    void initLandauerQDs(int n_qds, double* QDpos_, double* Esite_,
                        double K, double decay, double tS,
                        double E_sub, double E_tip, double tA,
                        double eta, double Gamma_tip, double Gamma_sub);
    void deleteLandauerQDs();
    void calculateTransmissions(int npos, double* ptips_, double* energies, int nE, double* H_QDs, double* transmissions);
    void solveHamiltonians(int npos, double* ptips_, double* Qtips, double* Qsites, 
                          double* evals, double* evecs, double* Hs, double* Gs);
    void solveSiteOccupancies(int npos, double* ptips_, double* Qtips, double* Qout);

    // Functions for testing and comparison
    void get_H_QD_no_tip(Vec2d* H_out);
    void get_tip_coupling(double* tip_pos, Vec2d* coupling_out);
    void get_full_H(double* tip_pos, Vec2d* H_out);
}

#endif // LANDAUER_QD_H
