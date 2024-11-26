#!/usr/bin/env python3

import numpy as np
from LandauerQD_py import LandauerQDs as LandauerQDs_py
import sys
sys.path.append("../../")
from pyProbeParticle import LandauerQD as cpp_solver
import matplotlib.pyplot as plt

# Enable more detailed numpy printing
np.set_printoptions(precision=8, suppress=True, linewidth=100)

# ========== Setup minimal test case ==========

# System parameters
nsite = 3
R = 5.0
Q_tip = 0.6
z_tip = 6.0
K = 0.01    # Coulomb interaction between QDs
tS = 0.1    # QD-substrate coupling
tA = 0.1    # Tip coupling strength
decay = 0.7
Gamma_tip = 1.0  # Tip state broadening
Gamma_sub = 1.0  # Substrate state broadening
eta = 0.01      # Small imaginary part for Green's function

# Setup QD positions in a ring
angles = np.linspace(0, 2*np.pi, nsite, endpoint=False)
QDpos       = np.zeros((nsite, 3))  # 3D array for Python implementation
QDpos[:, 0] = R * np.cos(angles)
QDpos[:, 1] = R * np.sin(angles)

# Energy of states on the sites
E0QDs = np.array([-1.0, -1.0, -1.0])

# Test points - we'll test a few strategic positions
test_positions = [
    np.array([-5.0, 0.0, z_tip]),  # Far left
    np.array([0.0, 0.0, z_tip]),   # Center
    np.array([5.0, 0.0, z_tip]),   # Far right
    #np.array([0.87, -0.35, z_tip]),   # Center
]

test_energies = np.array([-0.5, 0.0, 0.5])  # Test at different energies

#test_energies = np.array([0.3])  # Test at different energies

def debug_calculation():
    """Debug calculation steps in detail."""
    print("\n=== Detailed Debugging of Calculations ===")
    
    # Initialize C++ implementation with debug=True
    cpp_solver.init            ( QDpos, E0QDs, K=K, decay=decay, tS=tS, tA=tA, Gamma_tip=Gamma_tip, Gamma_sub=Gamma_sub, eta=eta, debug=1, verbosity=1)
    py_system = LandauerQDs_py ( QDpos, E0QDs, K=K, decay=decay, tS=tS, tA=tA, Gamma_tip=Gamma_tip, Gamma_sub=Gamma_sub, eta=eta, debug=True)   # Initialize Python implementation with debug=True
    
    results = []
    print("\n=== Testing at multiple positions and energies ===")
    for pos in test_positions:
        for E in test_energies:
            print(f"\nPosition: [{pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f}]")
            print(f"Energy: {E:6.2f}")
            print(f"Q_tip: {Q_tip:6.2f}")
            
            print( "\n\n########### C++    calculation: ############## " )
            T_cpp = cpp_solver.calculate_transmission(E, pos, Q_tip);       print(f"C++ transmission:    {T_cpp:10.3e}" )   # Calculate using C++ implementation first
            print( "\n\n########### Python calculation: ############## " )
            T_py  = py_system.calculate_transmission  (pos, E, Q_tip=Q_tip); print(f"Python transmission: {T_py:10.3e}"  ) # Calculate using Python implementation
            
            rel_diff = abs(T_cpp - T_py) / max(abs(T_cpp), abs(T_py))
            print(f"Relative difference: {rel_diff:10.3e}")
            
            results.append({
                'pos': pos,
                'E': E,
                'T_py': T_py,
                'T_cpp': T_cpp,
                'rel_diff': rel_diff
            })
    
    # Find case with largest discrepancy
    max_diff_case = max(results, key=lambda x: x['rel_diff'])
    print("\n=== Case with largest discrepancy ===")
    print(f"Position: {max_diff_case['pos']}")
    print(f"Energy: {max_diff_case['E']}")
    print(f"Python transmission: {max_diff_case['T_py']}")
    print(f"C++ transmission: {max_diff_case['T_cpp']}")
    print(f"Relative difference: {max_diff_case['rel_diff']}")
    
    cpp_solver.cleanup()
    return results

if __name__ == "__main__":
    results = debug_calculation()
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Plot transmission vs position for each energy
    for i, E in enumerate(test_energies):
        plt.subplot(1, 3, i+1)
        positions = np.array([r['pos'][0] for r in results[i::len(test_energies)]])
        T_py = np.array([r['T_py'] for r in results[i::len(test_energies)]])
        T_cpp = np.array([r['T_cpp'] for r in results[i::len(test_energies)]])
        
        plt.semilogy(positions, T_py, 'o-', label='Python')
        plt.semilogy(positions, T_cpp, 's--', label='C++')
        plt.grid(True)
        plt.legend()
        plt.title(f'E = {E:.1f}')
        plt.xlabel('X Position')
        plt.ylabel('Transmission')
    
    plt.tight_layout()
    plt.show()
