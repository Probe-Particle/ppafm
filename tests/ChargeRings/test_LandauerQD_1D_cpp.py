import os
os.environ['LD_PRELOAD'] = '/usr/lib/x86_64-linux-gnu/libgomp.so.1'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

#from LandauerQD_py import LandauerQDs
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
use_occupancy = False  # Set to True to use occupancy solver
#use_occupancy = False  # Set to True to use occupancy solver
cCouling = 0.03       # Coupling parameter for occupancy calculation
E_Fermi = 0.0        # Fermi energy level

# QD system setup
nsite = 3
R = 5.0
phiRot = -1.0
Q0 = 1.0            # Base charge for occupancy calculation
Qzz = 15.0 * 0.0    # z-component of quadrupole moment

# Setup QD positions in a ring
angles = np.linspace(0, 2*np.pi, nsite, endpoint=False) + phiRot
QDpos = np.zeros((nsite, 2))
QDpos[:, 0] = R * np.cos(angles)
QDpos[:, 1] = R * np.sin(angles)

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

# Initialize the C++ solver
cpp_solver.init(QDpos, E0QDs, K=K, decay=decay, tS=tS, tA=tA, 
               Gamma_tip=Gamma_tip, Gamma_sub=Gamma_sub)

# Setup scan line
x = np.linspace(-L, L, 100)
ps_line = np.zeros((len(x), 3))
ps_line[:, 0] = x
ps_line[:, 2] = z_tip

# Setup energies for transmission calculation
energies = np.linspace(Emin, Emax, 100)

# Calculate transmissions using C++ implementation
transmissions = cpp_solver.scan_1D(ps_line, energies, np.full(len(x), Q_tip))

# Plot results using GridSpec
fig = plt.figure(figsize=(12, 8))
gs = GridSpec(2, 2, width_ratios=[1, 0.05], height_ratios=[1, 1])

#cmap='plasma'
cmap='cividis'
#cmap='afmhot'

# Plot transmission with eigenvalue bands
ax1 = fig.add_subplot(gs[0, 0])
im = ax1.imshow(transmissions.T, aspect='auto', origin='lower', extent=[0, len(ps_line), energies[0], energies[-1]], cmap=cmap, vmin=0.0, vmax=0.0001)

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

plt.tight_layout()
plt.show()

# Clean up C++ resources
cpp_solver.cleanup()
