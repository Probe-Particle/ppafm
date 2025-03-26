#!/usr/bin/env python3

# Set up ASan preloading before any imports
import os
bASAN = True
bQmeQ = False


import sys
sys.stdout = sys.stderr = open(sys.stdout.fileno(), mode='w', buffering=1)
#path.insert(0, '/home/prokop/bin/home/prokop/venvs/ML/lib/python3.12/site-packages/qmeq/')

import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(linewidth=256, suppress=True)

import qmeq
from qmeq import config
from qmeq import indexing as qmqsi
from qmeq.config import verb_print_


def build_hamiltonian(eps1, eps2, eps3, t, W):
    verb_print_(1,"\n#### Building Hamiltonian: eps: ", [eps1, eps2, eps3], " t: ", t, " W: ", W)
    # One-particle Hamiltonian
    hsingle = {(0,0): eps1, (0,1): t, (0,2): t,
               (1,1): eps2, (1,2): t,
               (2,2): eps3}
    
    # Two-particle Hamiltonian: inter-site coupling
    coulomb = {(0,1,1,0): W,
               (1,2,2,1): W,
               (0,2,2,0): W}
    
    return hsingle, coulomb

def build_leads(muS, muT, Temp, VS, VT, coeffT, VBias):
    # Leads: substrate (S) and scanning tip (T)
    mu_L   = {0: muS, 1: muT + VBias}
    Temp_L = {0: Temp, 1: Temp}
    # Coupling between leads (1st number) and impurities (2nd number)
    TLeads = {(0,0): VS,         # S <-- 1
              (0,1): VS,         # S <-- 2
              (0,2): VS,         # S <-- 3
              (1,0): VT,         # T <-- 1
              (1,1): coeffT*VT,  # T <-- 2
              (1,2): coeffT*VT}  # T <-- 3
    
    return mu_L, Temp_L, TLeads

def run_QmeQ_solver(NSingle, Hsingle, Hcoulomb, NLeads, TLeads, mu_L, Temp_L, DBand, verbosity=0):
    """Run QmeQ solver with the given parameters"""
    if verbosity > 0:
        print( "\n\n" )
        print( "######################################################################" )
        print( "######################################################################" )
        print( "\n### Running QmeQ Pauli solver /home/prokop/git_SW/qmeq/qmeq/approach/base/pauli.py " )
    
    #mu_L, Temp_L, TLeads = build_leads(muS, muT, Temp, VS, VT, coeffT, VBias)
    #Hsingle, Hcoulomb    = build_hamiltonian(eps1, eps2, eps3, t, W)
    
    try:
        config.verbosity = verbosity
        system = qmeq.Builder(NSingle, Hsingle, Hcoulomb, NLeads, TLeads, mu_L, Temp_L, DBand,  kerntype='Pauli', indexing='Lin', itype=0, symq=True, solmethod='solve', mfreeq=0)
        system.appr.verbosity = verbosity  # Set verbosity after instance creation
        system.verbosity = verbosity
        if verbosity > 0:
            print( "type(system).__name__ ", type(system).__name__)
        system.solve()
    except Exception as e:
        print(f"Error running QmeQ solver: {e}")
        return None
    
    chargelst = system.si.chargelst
    state_order = qmqsi.get_state_order(chargelst); 
    state_occupancy = qmqsi.get_state_occupancy_strings(chargelst, NSingle); 
    
    if verbosity > 0:
        print("QmeQ state order:", state_order)
        print("QmeQ state occupancy:", state_occupancy)
        print("QmeQ energies:", system.Ea)
        print("QmeQ probabilities:", system.phi0)
        print("QmeQ kernel:\n", system.kern)
        print("QmeQ current:", system.current[1])
    
    # Create a result dictionary for comparison
    qmeq_res = {
        'current': system.current[1],
        'energies': system.Ea,
        'probabilities': system.phi0,
        'kernel': system.kern,
        'pauli_factors': system.appr.paulifct,
        'leads': {
            'mu': system.leads.mulst,
            'temp': system.leads.tlst,
            'gamma': system.leads.dlst[:,0],
            'Tba': system.leads.Tba.real
        }
    }
    
    return qmeq_res