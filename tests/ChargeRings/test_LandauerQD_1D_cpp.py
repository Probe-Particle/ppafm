import os
os.environ['LD_PRELOAD'] = '/usr/lib/x86_64-linux-gnu/libgomp.so.1'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
sys.path.append("../../")
from pyProbeParticle import ChargeRings as chr
from pyProbeParticle import LandauerQD as cpp_solver

# ========== Setup

# System parameters
Q_tip = 0.6
z_tip = 6.0
L     = 20.0
decay = 0.7
T     = 10.0  # Temperature for occupancy calculation

# Occupancy calculation switch
use_occupancy = True  # Set to True to use occupancy solver
cCouling = 0.03       # Coupling parameter for occupancy calculation
E_Fermi = 0.0        # Fermi energy level

# QD system setup
nsite = 3
R = 5.0
phiRot = -1.0
Q0 = 1.0            # Base charge for occupancy calculation
Qzz = 15.0 * 0.0    # z-component of quadrupole moment

# Energy of states on the sites
E0QDs = np.array([-1.0, -1.0, -1.0])

K = 0.01    # Coulomb interaction between QDs
tS = 0.1    # QD-substrate coupling
tA = 0.1    # Tip coupling strength
Gamma_tip = 1.0  # Tip state broadening
Gamma_sub = 1.0  # Substrate state broadening
eta = 1e-8      # Green's function broadening

# Energy range for transmission calculation
Emin = -0.1
Emax = 0.3
energies = np.linspace(Emin, Emax, 100)

site_colors = ['r', 'g', 'b']

# ========== Main

# Setup system geometry
phis = np.linspace(0, 2*np.pi, nsite, endpoint=False)
QDpos = np.zeros((3, 3))
QDpos[:,0] = np.cos(phis)*R
QDpos[:,1] = np.sin(phis)*R

# Setup scanning line
ps_line = chr.getLine(QDpos, [0.5,0.5,-3.0], [-3.0,-3.0,1.0], n=200)
ps_line[:,2] = z_tip
Qtips = np.ones(len(ps_line))*Q_tip

if use_occupancy:
    # Setup site multipoles and rotations for occupancy calculation
    QDrots = chr.makeRotMats(phis + phiRot, nsite)
    QDmpols = np.zeros((3,10))
    QDmpols[:,4] = Qzz
    QDmpols[:,0] = Q0
    
    # Initialize ChargeRings system
    chr.initRingParams(QDpos, E0QDs, rot=QDrots, MultiPoles=QDmpols, E_Fermi=E_Fermi, cCouling=cCouling, temperature=T)
    
    # Calculate occupancies and Hamiltonians
    Q_qds = chr.solveSiteOccupancies(ps_line, Qtips)
    eigenvalues, evecs, H_QDs, Gs = chr.solveHamiltonians(ps_line, Qtips, Qsites=Q_qds, bH=True)
    
    # Initialize C++ solver with occupancy
    cpp_solver.init(QDpos.T, E0QDs, K=K, decay=decay, tS=tS, tA=tA, Gamma_tip=Gamma_tip, Gamma_sub=Gamma_sub, eta=eta)
    
    # Calculate transmission using pre-computed Hamiltonians
    transmissions = []
    for i, (pos, H_qd) in enumerate(zip(ps_line, H_QDs)):
        trans_E = []
        for E in energies:
            trans = cpp_solver.calculate_transmission(E, pos, Qtips[i], H_qd.astype(np.float64))
            trans_E.append(trans)
        transmissions.append(trans_E)
    transmissions = np.array(transmissions)
    
else:
    # Initialize C++ solver without occupancy
    cpp_solver.init(QDpos, E0QDs, K=K, decay=decay, tS=tS, tA=tA, Gamma_tip=Gamma_tip, Gamma_sub=Gamma_sub, eta=eta)
    
    # Calculate transmission directly
    transmissions = []
    for pos in ps_line:
        transmission = cpp_solver.calculate_transmission(pos, energies)
        transmissions.append(transmission)
    transmissions = np.array(transmissions)

# Plot results using GridSpec
if use_occupancy:
    fig = plt.figure(figsize=(12, 12))
    gs = GridSpec(3, 2, width_ratios=[1, 0.05], height_ratios=[1, 1, 1])
else:
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, width_ratios=[1, 0.05], height_ratios=[1, 1])

# Plot transmission
ax1 = fig.add_subplot(gs[0, 0])
X, Y = np.meshgrid(range(len(ps_line)), energies)
im1 = ax1.pcolormesh(X, Y, transmissions.T, shading='auto', cmap='plasma')
plt.colorbar(im1, cax=fig.add_subplot(gs[0, 1]))
ax1.set_xlabel('Position')
ax1.set_ylabel('Energy (eV)')
ax1.set_title('Transmission')

if use_occupancy:
    # Plot QD charges
    ax3 = fig.add_subplot(gs[2, 0])
    for i in range(nsite):
        ax3.plot(range(len(ps_line)), Q_qds[:, i], '-', c=site_colors[i], label=f'QD {i+1}')
    ax3.set_xlabel('Position')
    ax3.set_ylabel('QD charge')
    ax3.set_title('QD Charges')
    ax3.legend()

plt.tight_layout()
plt.show()
