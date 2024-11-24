import numpy as np
import sys
sys.path.append("../../")
from LandauerQD_py import LandauerQDs as LandauerQDs_py
from pyProbeParticle import LandauerQD as cpp_solver
import test_utils as tu

# Test parameters
TOLERANCE = 1e-8

# ======== Setup System ========
                       
E_sites   = [-0.5, -0.75, -0.65]  # On-site energies
n_qds     = len(E_sites)          # Number of quantum dots
K         = 0.01                  # Coulomb interaction
decay     = 0.7                   # Decay constant
tS        = 0.1                   # Substrate coupling
E_sub     = 0.0                   # Substrate energy
E_tip     = 0.0                   # Tip energy
tA        = 0.1                   # Tip coupling strength
eta       = 0.01                  # Broadening
Gamma_tip = 1.0                   # Tip state broadening
Gamma_sub = 1.0                   # Substrate state broadening

# ---- Quantum dot setup (triangle / ring )
R      = 5.0  # Ring radius
angles = np.linspace(0, 2*np.pi, n_qds, endpoint=False)
QD_pos = np.array([[R*np.cos(phi), R*np.sin(phi), 0.0] for phi in angles])

# ---- Tip setup
tip_pos = np.array([ 1.0, 1.5, 5.0 ]) # Tip position
Q_tip   = 0.6                         # Tip charge
E_test  = 0.0                         # Test energy

# ======== Functions ========

def compare_matrices():
    """Compare matrices saved by Python and C++ implementations"""
    print("\n=== Comparing matrices between Python and C++ implementations ===")
    
    # List of matrices to compare (Python file, C++ file, description)
    matrices_to_compare = [
        ("py_H.txt", "cpp_H.txt", "Hamiltonian"),
        ("py_G.txt", "cpp_G.txt", "Green's function"),
        ("py_Gdag.txt", "cpp_Gdag.txt", "Green's function conjugate transpose"),
        ("py_Gamma_s.txt", "cpp_Gamma_s.txt", "Gamma substrate"),
        ("py_Gamma_t.txt", "cpp_Gamma_t.txt", "Gamma tip"),
        ("py_Gammat_Gdag.txt", "cpp_Gammat_Gdag.txt", "Gamma_t @ Gdag"),
        ("py_G_Gammat_Gdag.txt", "cpp_G_Gammat_Gdag.txt", "G @ Gamma_t @ Gdag"),
        ("py_Tmat.txt", "cpp_Tmat.txt", "Transmission matrix")
    ]
    
    all_match = True
    for py_file, cpp_file, desc in matrices_to_compare:
        print(f"\nComparing {desc}...")
        match = tu.compare_matrix_files(py_file, cpp_file, tol=TOLERANCE)
        if not match:
            all_match = False
            print(f"❌ {desc} matrices DO NOT match within tolerance {TOLERANCE}")
        else:
            print(f"✓ {desc} matrices match within tolerance {TOLERANCE}")
    
    return all_match

def run_comparison():
    """Run and compare both implementations"""
    print("\n=== Running Landauer QD Implementation Comparison ===")
    print(f"System parameters:")
    print(f"Number of QDs: {n_qds}")
    print(f"Test energy: {E_test}")
    print(f"Tip position: {tip_pos}")
    print(f"Tip charge: {Q_tip}")
    
    # Initialize solvers
    #py_solver = setup_python_solver()
    #setup_cpp_solver()

    #cpp_solver = lqd

    debug = True

    py_solver = LandauerQDs_py(QD_pos, E_sites, K=K, decay=decay, tS=tS,   E_sub=E_sub, E_tip=E_tip, tA=tA, eta=eta,  Gamma_tip=Gamma_tip, Gamma_sub=Gamma_sub, debug=debug )
    cpp_solver.init(QD_pos, E_sites, K=K, decay=decay, tS=tS,  E_sub=E_sub, E_tip=E_tip, tA=tA, eta=eta,  Gamma_tip=Gamma_tip, Gamma_sub=Gamma_sub,             debug=debug )  # Enable debug output
    
    # Python implementation
    print("\n ############### Running Python implementation...\n")
    py_transmission  = py_solver.calculate_transmission( tip_pos, E_test, Q_tip=Q_tip, Hqd=None )
    #py_transmission = py_solver.calculate_transmission_from_H(E_test, py_H)
    
    print("\n ############### Running C++ implementation...\n")
    cpp_transmission = cpp_solver.calculate_transmission(E_test, tip_pos, Q_tip=Q_tip, Hqd=None )
    
    # Compare results
    print("\n=== Comparing Results ===")
    print(f"Python transmission: {py_transmission}")
    print(f"C++ transmission: {cpp_transmission}")
    print(f"Transmission difference: {abs(py_transmission - cpp_transmission)}")
        
    all_match = compare_matrices()
    if all_match:
        print("\nSUCCESS: All matrices match within tolerance!")
    else:
        print("\nFAILURE: Some matrices do not match within tolerance!")
        
    cpp_solver.cleanup()


# ======== Body ========

if __name__ == "__main__":
    run_comparison()
