#!/usr/bin/env python3

import numpy as np
from LandauerQD_py import LandauerQDs as LandauerQDs_py
import sys
sys.path.append("../../")
from pyProbeParticle import LandauerQD as cpp_solver
import matplotlib.pyplot as plt

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

# Setup QD positions in a ring
angles = np.linspace(0, 2*np.pi, nsite, endpoint=False)
QDpos_py = np.zeros((nsite, 3))  # 3D array for Python implementation
QDpos_py[:, 0] = R * np.cos(angles)
QDpos_py[:, 1] = R * np.sin(angles)

QDpos_cpp = QDpos_py[:, :2]  # 2D array for C++ implementation (x,y only)

# Energy of states on the sites
E0QDs = np.array([-1.0, -1.0, -1.0])

# Minimal test case: 2 energies, 2 positions
energies = np.array([-0.5, 0.5])
x_positions = np.array([-5.0, 5.0])
ps_line = np.zeros((len(x_positions), 3))
ps_line[:, 0] = x_positions
ps_line[:, 2] = z_tip

# ========== Method 1: Pure Python Implementation ==========
def run_python_method():
    print("\n=== Method 1: Pure Python Implementation ===")
    py_system = LandauerQDs_py(QDpos_py, E0QDs, K=K, decay=decay, tS=tS, tA=tA, 
                              Gamma_tip=Gamma_tip, Gamma_sub=Gamma_sub)
    
    transmissions_py = np.zeros((len(ps_line), len(energies)))
    for i, pos in enumerate(ps_line):
        for j, E in enumerate(energies):
            transmissions_py[i,j] = py_system.calculate_transmission(pos, E, Q_tip=Q_tip)
    
    print("Python Implementation Results:")
    print(transmissions_py)
    return transmissions_py

# ========== Method 2: C++ Point-by-Point ==========
def run_cpp_pointwise():
    print("\n=== Method 2: C++ Point-by-Point ===")
    cpp_solver.init(QDpos_cpp, E0QDs, K=K, decay=decay, tS=tS, tA=tA, 
                   Gamma_tip=Gamma_tip, Gamma_sub=Gamma_sub)
    
    transmissions_cpp_point = np.zeros((len(ps_line), len(energies)))
    for i, pos in enumerate(ps_line):
        for j, E in enumerate(energies):
            transmissions_cpp_point[i,j] = cpp_solver.calculate_transmission(E, pos, Q_tip)
    
    cpp_solver.cleanup()
    print("C++ Point-by-Point Results:")
    print(transmissions_cpp_point)
    return transmissions_cpp_point

# ========== Method 3: C++ scan_1D ==========
def run_cpp_scan1d():
    print("\n=== Method 3: C++ scan_1D ===")
    cpp_solver.init(QDpos_cpp, E0QDs, K=K, decay=decay, tS=tS, tA=tA, 
                   Gamma_tip=Gamma_tip, Gamma_sub=Gamma_sub)
    
    transmissions_cpp_scan = cpp_solver.scan_1D(ps_line, energies, np.full(len(ps_line), Q_tip))
    
    cpp_solver.cleanup()
    print("C++ scan_1D Results:")
    print(transmissions_cpp_scan)
    return transmissions_cpp_scan

# ========== Run all methods and compare ==========
if __name__ == "__main__":
    # Run all methods
    trans_py = run_python_method()
    trans_cpp_point = run_cpp_pointwise()
    trans_cpp_scan = run_cpp_scan1d()
    
    # Compare results
    print("\n=== Comparing Results ===")
    print("Max difference Python vs C++ point-by-point:", 
          np.max(np.abs(trans_py - trans_cpp_point)))
    print("Max difference Python vs C++ scan_1D:", 
          np.max(np.abs(trans_py - trans_cpp_scan)))
    print("Max difference C++ methods:", 
          np.max(np.abs(trans_cpp_point - trans_cpp_scan)))
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(121)
    plt.title("E = -0.5")
    plt.plot(x_positions, trans_py[:,0], 'o-', label='Python')
    plt.plot(x_positions, trans_cpp_point[:,0], 's--', label='C++ point')
    plt.plot(x_positions, trans_cpp_scan[:,0], '^:', label='C++ scan')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(122)
    plt.title("E = 0.5")
    plt.plot(x_positions, trans_py[:,1], 'o-', label='Python')
    plt.plot(x_positions, trans_cpp_point[:,1], 's--', label='C++ point')
    plt.plot(x_positions, trans_cpp_scan[:,1], '^:', label='C++ scan')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
