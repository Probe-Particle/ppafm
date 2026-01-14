#!/usr/bin/env python3
"""
Dump PME quantities (Ei, Ri, Es, Ne, K, rho, I_tip) using the Python pauli solver
for direct comparison with the Node/JS dump (run_pme_node.js).
Uses pyProbeParticle.pauli on a simple 2-site system matching the viewer defaults.
"""

import json
import sys
import os
import numpy as np

# Add repository root to path so pyProbeParticle is importable when run from test/
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(repo_root)
from pyProbeParticle import pauli

params_path = os.path.join(os.path.dirname(__file__), 'params.json')
with open(params_path, 'r') as f:
    P = json.load(f)

NSites = int(P['NSites'])
nleads = 2

TempK = float(P['TempK'])
VBias = float(P['VBias'])
GammaS = float(P['GammaS'])
GammaT = float(P['GammaT'])
W = float(P['W'])
sites = [(s['x'], s['y'], s['z'], s['E']) for s in P['sites']]

# Build Pauli solver (nSingle, nleads, verbosity)
solver = pauli.PauliSolver(NSites, nleads, 0)

# Set leads (muS=0, muT=VBias), temperature in eV
kBoltz = 8.617333262e-5
T_eV = TempK * kBoltz
solver.set_lead(0, 0.0, T_eV)
solver.set_lead(1, VBias, T_eV)

# Set tunneling amplitudes: amplitudes, not rates; follow run_cpp_scan convention (sqrt/Gamma/pi)
VS = np.sqrt(GammaS / np.pi)
VT = np.sqrt(GammaT / np.pi)
TLeads = np.zeros((nleads, NSites), dtype=np.float64)
TLeads[0, :] = VS
TLeads[1, :] = VT
solver.set_tunneling(TLeads)

# Set Hsingle (diagonal with site energies)
H = np.zeros((NSites, NSites), dtype=np.float64)
for i, s in enumerate(sites):
    H[i, i] = s[3]
solver.set_hsingle(H)

# Set scalar W (uniform pair)
state_order = pauli.make_state_order(NSites)
solver.generate_pauli_factors(W=W, state_order=state_order)
solver.generate_kernel()
solver.solve()

# Retrieve energies, kernel, probabilities
nStates = 2 ** NSites
energies = np.zeros(nStates, dtype=np.float64)
kernel = np.zeros(nStates * nStates, dtype=np.float64)
probs = np.zeros(nStates, dtype=np.float64)
pauli.lib.get_energies(solver.solver, pauli._np_as(energies, pauli.c_double_p))
pauli.lib.get_kernel(solver.solver, pauli._np_as(kernel, pauli.c_double_p))
pauli.lib.get_probabilities(solver.solver, pauli._np_as(probs, pauli.c_double_p))
current = pauli.lib.calculate_current(solver.solver, 1)  # lead index 1 (tip)

out = {
    "params": P,
    "sites": sites,
    "Es": energies.tolist(),
    "K": kernel[: nStates * nStates].tolist(),
    "rho": probs.tolist(),
    "I_tip": current
}

print(json.dumps(out, indent=2))
