
import os
os.environ['LD_PRELOAD'] = '/usr/lib/x86_64-linux-gnu/libgomp.so.1'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from LandauerQD_py import LandauerQDs
import sys
sys.path.append("../../")
from pyProbeParticle import ChargeRings as chr

# ========== Setup

# System parameters
Q_tip = 0.6
z_tip = 6.0
L     = 20.0
decay = 0.7
T     = 10.0  # Temperature for occupancy calculation

# Occupancy calculation switch
#use_occupancy = True  # Set to True to use occupancy solver
use_occupancy = False  # Set to True to use occupancy solver
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

Emin = -0.1  # Increased range to see effect of broadening
Emax = 0.3

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

# Energy range for transmission calculation
energies = np.linspace(Emin, Emax, 100)
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
    
    # Initialize Landauer system
    system = LandauerQDs(QDpos, E0QDs, K, decay, tS, E_sub=0.0, E_tip=0.0, tA=tA, Gamma_tip=Gamma_tip, Gamma_sub=Gamma_sub)
    
    # Calculate transmission using pre-computed Hamiltonians
    transmissions = system.scan_1D(ps_line, energies, H_QDs=H_QDs)
else:
    # Initialize Landauer system without occupancy
    system = LandauerQDs(QDpos, E0QDs, K, decay, tS, E_sub=0.0, E_tip=0.0, tA=tA, Gamma_tip=Gamma_tip, Gamma_sub=Gamma_sub)
    
    # Calculate transmission and eigenvalues directly
    transmissions = system.scan_1D(ps_line, energies, Qtips=Qtips)
    eigenvalues = system.scan_eigenvalues(ps_line, Qtips=Qtips)

# Plot results using GridSpec
if use_occupancy:
    fig = plt.figure(figsize=(12, 12))
    gs = GridSpec(3, 2, width_ratios=[1, 0.05], height_ratios=[1, 1, 1])
else:
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, width_ratios=[1, 0.05], height_ratios=[1, 1])

#cmap='plasma'
cmap='cividis'
#cmap='afmhot'

# Plot transmission with eigenvalue bands
ax1 = fig.add_subplot(gs[0, 0])
im = ax1.imshow(transmissions.T, aspect='auto', origin='lower', extent=[0, len(ps_line), energies[0], energies[-1]], cmap=cmap, vmin=0.0, vmax=0.0001)

# Overlay eigenvalue bands
if use_occupancy:
    # Plot eigenvalues and diagonal elements of H_QDs
    for i in range(nsite):
        ax1.plot(range(len(ps_line)), eigenvalues[:, i], ':', alpha=0.7, label=f'Eps_{i+1}')
        ax1.plot(range(len(ps_line)), H_QDs[:,i,i], '-', alpha=1.0, lw=0.7, c=site_colors[i], label=f'H({i+1},{i+1})')
else:
    for i in range(nsite):
        ax1.plot(range(len(ps_line)), eigenvalues[:, i], ':', alpha=0.7, label=f'E_{i+1}')

ax1.legend()
ax1.set_xlim(0, len(ps_line))
ax1.set_ylim(energies[0], energies[-1])
ax1.set_xlabel('Position along line')
ax1.set_ylabel('Energy (eV)')
ax1.set_title('Transmission and Energy Levels')

# Add colorbar
cbar_ax1 = fig.add_subplot(gs[0, 1])
plt.colorbar(im, cax=cbar_ax1, label='Transmission')

# Plot transmission at selected energies
ax2 = fig.add_subplot(gs[1, 0])
selected_energies = [-0.02, 0.0, 0.02]
for E in selected_energies:
    idx = np.argmin(np.abs(energies - E))
    print(f"Energy: {E:.1f} eV, Index: {idx}")
    ax2.plot(transmissions[:, idx], label=f'E = {E:.1f} eV')
    ax1.axhline(E, ls=':', c='k', alpha=0.5)
ax2.set_xlabel('Position Index')
ax2.set_ylabel('Transmission')
ax2.legend()
ax2.set_title('Transmission vs Position at Selected Energies')
plt.grid()

if use_occupancy:
    # Plot QD charges
    ax3 = fig.add_subplot(gs[2, 0])
    for i in range(nsite):
        ax3.plot(range(len(ps_line)), Q_qds[:, i], '-', c=site_colors[i], label=f'QD {i+1}')
    ax3.set_xlim(0, len(ps_line))
    ax3.set_xlabel('Position along line')
    ax3.set_ylabel('Charge Occupancy')
    ax3.legend()

plt.tight_layout()
plt.show()
