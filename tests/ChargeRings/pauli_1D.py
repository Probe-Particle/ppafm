#!/usr/bin/env python3

# Set up ASan preloading before any imports
import os
bASAN = True
bQmeQ = False
#bQmeQ = True

if bQmeQ:
    import QmeQ_pauli as qmeqp

if bASAN:
    # Get ASan library path
    asan_lib = os.popen('gcc -print-file-name=libasan.so').read().strip()
    print("Preloading ASan library: ", asan_lib)
    # Set LD_PRELOAD environment variable
    os.environ['LD_PRELOAD'] = asan_lib
    os.environ['ASAN_OPTIONS'] = 'detect_leaks=0'

import sys
sys.stdout = sys.stderr = open(sys.stdout.fileno(), mode='w', buffering=1)

import numpy as np
import matplotlib.pyplot as plt
from sys import path
#path.insert(0, '/home/prokop/bin/home/prokop/venvs/ML/lib/python3.12/site-packages/qmeq/')

#path.insert(0, '/home/prokop/bin/home/prokop/venvs/ML/lib/python3.12/site-packages/qmeq/')
path.insert(0, '../../pyProbeParticle')
import pauli as psl
from pauli import PauliSolver

# setup numpy print options to infinite line length
np.set_printoptions(linewidth=256, suppress=True)


# Constants
NSingle = 3  # number of impurity states
NLeads  = 2   # number of leads

# Parameters (in meV) - with small perturbations to break degeneracy
eps1 = -10.0
eps2 = -10.01  # Slightly different
eps3 = -10.02  # Slightly different

VTs = [0.0, 0.05, 0.0]

t     = 0.0      # direct hopping
W     = 3.0      # inter-site coupling (matching compare_solvers.py)
VBias = 0.1  # bias voltage

# Lead parameters
muS    = 0.0    # substrate chemical potential
muT    = 10.0    # tip chemical potential
Temp   = 0.224 # temperature in meV
GammaS = 0.20 # coupling to substrate
GammaT = 0.05 # coupling to tip
DBand  = 10000.0 # lead bandwidth
VTs    = np.ones(3)*np.sqrt(GammaT/np.pi)
VS     = np.sqrt(GammaS/np.pi)

# lead_params = {
#     'muS':   0.0,   # substrate chemical potential
#     'muT':   0.0,   # tip chemical potential
#     'Temp':  0.224, # temperature in meV
#     'VBias': 0.1,   # bias voltage
#     'VTs':   VTs
# }

# Position-dependent coefficients


def run_cpp_solver(pauli, eps1, eps2, eps3):
    """Run C++ solver with the given parameters"""
    if verbosity > 0:
        print( "\n\n" )
        print( "######################################################################" )
        print( "######################################################################" )
        print( "\n### Running C++ solver /home/prokop/git_SW/qmeq/cpp/pauli_solver.hpp \n" )
    
    #Hsingle_, TLeads_, lead_mu, lead_temp, lead_gamma = prepare_cpp_inputs(eps1, eps2, eps3)
    #Hsingle_, TLeads_, lead_mu, lead_temp, lead_gamma = prepare_cpp_inputs_efficient(eps1, eps2, eps3)
    
    # --- prepare static inputs - this does not change when we change eps
    state_order = [0, 4, 2, 6, 1, 5, 3, 7]
    state_order = np.array(state_order, dtype=np.int32)
    TLeads_, lead_mu, lead_temp, lead_gamma = prepare_leads_cpp()
    NStates = 2**NSingle

    # --- prepare dynamic inputs - this changes when we change eps
    Hsingle_ = prepare_hsingle_cpp(eps1, eps2, eps3)
    
    # Create and run solver
    solver = pauli.create_pauli_solver_new(NStates, NLeads, Hsingle_, W, TLeads_, lead_mu, lead_temp, lead_gamma, state_order, verbosity)
    
    # Get energies before solving
    energies = pauli.get_energies(solver, NStates)
    if verbosity > 0:
        print("C++ energies:", energies)
    
    pauli.solve(solver)
    
    # Get detailed results for comparison
    kernel         = pauli.get_kernel(solver, NStates)
    probabilities = pauli.get_probabilities(solver, NStates)
    currents      = [pauli.calculate_current(solver, lead) for lead in range(NLeads)]
    Tba           = pauli.get_coupling(solver, NLeads, NStates)
    pauli_factors = pauli.get_pauli_factors(solver, NLeads, NStates)
    
    if verbosity > 0:
        print("C++ probabilities:", probabilities)
        print("C++ kernel:\n", kernel)
        print("C++ current:", currents[1])
    
    # Create a result dictionary for comparison
    cpp_res = {
        'current': currents[1],
        'energies': energies,
        'probabilities': probabilities,
        'kernel': kernel,
        'pauli_factors': pauli_factors,
        'leads': {
            'mu': lead_mu,
            'temp': lead_temp,
            'gamma': lead_gamma,
            'Tba': Tba
        }
    }
    
    pauli.cleanup(solver)
    return cpp_res
    

# Define a comparison function like in compare_solvers.py
def compare_results(qmeq_res, cpp_res, tol=1e-8, bPrintSame=True):
    print("\n\n### Comparing QmeQ and C++ results")
    
    # Compare current
    qmeq_current = qmeq_res['current']
    cpp_current  = cpp_res['current']
    diff_current = abs(qmeq_current - cpp_current)
    
    # Compare energies
    qmeq_energies = qmeq_res['energies']
    cpp_energies = cpp_res['energies']
    
    print("\nEnergies:")
    for i, (qe, ce) in enumerate(zip(qmeq_energies, cpp_energies)):
        diff = abs(qe - ce)
        if diff > tol or bPrintSame:
            print(f"  State {i}: QmeQ={qe}, C++={ce}, Diff={diff}")
    
    # Compare probabilities
    qmeq_probs = qmeq_res['probabilities']
    cpp_probs  = cpp_res['probabilities']
    
    print("\nProbabilities:")
    for i, (qp, cp) in enumerate(zip(qmeq_probs, cpp_probs)):
        diff = abs(qp - cp)
        if diff > tol or bPrintSame:
            print(f"  State {i}: QmeQ={qp}, C++={cp}, Diff={diff}")

if __name__ == "__main__":
    print( "\n\n" )
    print( "##################################################################################" )
    print( "##################################################################################" )
    print( "### compare_scan_1D.py Compare QmeQ vs C++ Pauli solvers for 1D array of energies " )
    print( "##################################################################################" )
    print( "##################################################################################" )
    
    # Use exact parameters from compare_solvers.py
    bPrint = True
    nstep = 300
    #nstep = 3
    #nstep = 1
    # Instead of generating energy range, use exact values from compare_solvers.py
    eps = np.zeros((nstep,3))
    ts = np.linspace(0, 10.0, nstep)
    eps[:,0] = eps1 + ts  
    eps[:,1] = eps2 + ts  
    eps[:,2] = eps3 + ts  

    verbosity = 0  # Match compare_solvers.py verbosity
    
    # Prepare Pauli solver C++
    state_order = [0, 4, 2, 6, 1, 5, 3, 7]
    state_order = np.array(state_order, dtype=np.int32)
    #TLeads_, lead_mu, lead_temp, lead_gamma = psl.prepare_leads_cpp(**lead_params)
    NStates  = 2**NSingle
    Hsingle_ = psl.prepare_hsingle_cpp(eps1, eps2, eps3)
    pauli  = PauliSolver(NSingle, NLeads, verbosity=verbosity )
    #pauli.set_leads(lead_mu, lead_temp, lead_gamma)
    
    TLeads_ = np.array([ [VS, VS, VS],  VTs ])   
    pauli.set_tunneling(TLeads_)
    pauli.set_lead(0, muS, Temp )
    pauli.set_lead(1, muT, Temp )
    
    Iqmeq = np.zeros(nstep)
    Icpp  = np.zeros(nstep)
    for i in range(nstep):
        epsi = eps[i]
        if verbosity > 0: print(f"\n####### compare_scan_1D.py loop [{i}] epsi: {epsi}")

        if bQmeQ:
            mu_L, Temp_L, TLeads = qmeqp.build_leads(muS, muT, Temp, VS, VTs)
            Hsingle, Hcoulomb    = qmeqp.build_hamiltonian(eps1, eps2, eps3, t, W)
            qmeq_res             = qmeqp.run_QmeQ_solver(NSingle, Hsingle, Hcoulomb, NLeads, TLeads, mu_L, Temp_L, DBand, verbosity)
            Iqmeq[i]             = qmeq_res['current']

        Hsingle_ = psl.prepare_hsingle_cpp(epsi[0], epsi[1], epsi[2])
        I1 = pauli.solve_hsingle(Hsingle_, W, 1, state_order)

        if bQmeQ:
            print(f"Eps[{i}] {epsi} Current: QmeQ: {qmeq_res['current']} C++: {I1} | Diff: {abs(qmeq_res['current'] - I1)}")
        else:
            print(f"Eps[{i}] {epsi} Current: C++: {I1}")
        #compare_results(qmeq_res, cpp_res, tol=1e-8, bPrintSame=True)
        Icpp[i]  = I1

    plt.figure(figsize=(10, 6))
    if bQmeQ:
        plt.plot(ts, Iqmeq, '.-b', label='QmeQ Pauli')
    plt.plot(ts, Icpp,  '.:r', label='C++ Pauli')
    plt.xlabel('Onsite Energy (meV)')
    plt.ylabel('Current (nA)')
    plt.title('Solver Comparison for 1D Energy Scan')
    plt.legend()
    plt.grid(True)
    plt.show()