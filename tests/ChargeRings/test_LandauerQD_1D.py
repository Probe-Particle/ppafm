import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from LandauerQD import LandauerQDs
import sys
sys.path.append("../../")
from pyProbeParticle import ChargeRings as chr

# ========== Setup

# System parameters (matching test_ChargeRings_1D.py)
Q_tip = 0.6
z_tip = 6.0
L = 20.0
decay = 0.7

# QD system setup
nsite = 3
R = 5.0
phiRot = -1.0

# Energy of states on the sites
Esite = np.array([-1.0, -1.0, -1.0])


K = 0.01  # Coulomb interaction between QDs
tS = 0.01  # QD-substrate coupling
tA = 0.1   # Tip coupling strength

# ========== Main


# Setup system geometry
phis = np.linspace(0, 2*np.pi, nsite, endpoint=False)
QDpos = np.zeros((3, 3))
QDpos[:,0] = np.cos(phis)*R
QDpos[:,1] = np.sin(phis)*R

# Initialize Landauer system

system = LandauerQDs(QDpos, Esite, K, decay, tS, E_sub=0.0, tA=tA)

# Setup scanning line (matching test_ChargeRings_1D.py)
ps_line = chr.getLine(QDpos, [0.5,0.5,-3.0], [-3.0,-3.0,1.0], n=200)
ps_line[:,2] = z_tip

# Energy range for transmission calculation
#energies = np.linspace(-2.0, 2.0, 100)
energies = np.linspace(-0.2, 0.2, 100)

# Perform 1D scan
transmissions = system.scan_1D(ps_line, Q_tip, energies)

# Plot results using GridSpec
fig = plt.figure(figsize=(12, 8))
gs = GridSpec(2, 2, width_ratios=[1, 0.05], height_ratios=[1, 1])

ax1 = fig.add_subplot(gs[0, 0])
im = ax1.imshow(transmissions.T, aspect='auto', origin='lower',
                extent=[0, len(ps_line), energies[0], energies[-1]], cmap='viridis')
ax1.set_ylabel('Energy (eV)')
ax1.set_title('Transmission vs Position and Energy')

cbar = fig.colorbar(im, cax=fig.add_subplot(gs[0, 1]))
cbar.set_label('Transmission')

ax2 = fig.add_subplot(gs[1, 0])
selected_energies = [-0.1, 0.0, 0.1]
for E in selected_energies:
    idx = np.argmin(np.abs(energies - E))
    ax2.plot(transmissions[:, idx], label=f'E = {E:.1f} eV')
ax2.set_xlabel('Position Index')
ax2.set_ylabel('Transmission')
ax2.legend()
ax2.set_title('Transmission vs Position at Selected Energies')

# Leave some space between subplots for alignment
plt.tight_layout()
plt.show()