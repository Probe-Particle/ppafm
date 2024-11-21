import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import sys
sys.path.append("../../")
from pyProbeParticle import LandauerQD as lqd
from pyProbeParticle import ChargeRings as chr

# ========== Setup

Q_tip     = 0.6
cCouling  = 0.03 # * 0.0
E_Fermi   = 0.0
z_tip     = 6.0
L         = 20.0
decay     = 0.7
T         = 10.0

# QD system setup
nsite = 3
R = 5.0
phiRot = -1.0
Q0  = 1.0
Qzz = 15.0 * 0.0

# Energy of states on the sites
E0QDs = np.array([-1.0, -1.0, -1.0])

K  = 0.01  # Coulomb interaction between QDs
tS = 0.1   # QD-substrate coupling
tA = 0.1   # Tip coupling strength
Gamma_tip = 1.0  # Tip state broadening
Gamma_sub = 1.0  # Substrate state broadening

Emin = -0.1  # Increased range to see effect of broadening
Emax =  0.3

site_colors = ['r', 'g', 'b']

# ========== Main

# ---- Setup geometry and electrostatic potential of the quantum dot system
phis = np.linspace(0, 2*np.pi, nsite, endpoint=False)
QDpos = np.zeros((nsite, 3))
QDpos[:,0] = np.cos(phis)*R
QDpos[:,1] = np.sin(phis)*R
QDpos[:,2] = 0.0

# Initialize LandauerQDs system
lqd.init(QDpos, E0QDs, K=K, decay=decay, tS=tS, E_sub=0.0, E_tip=0.0, tA=tA, eta=0.01, Gamma_tip=Gamma_tip, Gamma_sub=Gamma_sub)

# ---- Setup scan line
ps_line = np.zeros((100, 3))
ps_line[:,0] = np.linspace(-L/2, L/2, len(ps_line))
ps_line[:,2] = z_tip
Qtips = np.ones(len(ps_line)) * Q_tip

# ---- Calculate transmission
# Setup site multipoles and rotations for occupancy calculation
QDrots = np.zeros((nsite, 3, 3))
QDmpols = np.zeros((nsite, 10))
QDmpols[:,0] = Q0
QDmpols[:,4] = Qzz

# Initialize ChargeRings system for occupancy calculation
chr.initRingParams(QDpos, E0QDs, rot=QDrots, MultiPoles=QDmpols, E_Fermi=E_Fermi, cCouling=cCouling, temperature=T)

# Calculate occupancies and Hamiltonians using ChargeRings
Q_qds = chr.solveSiteOccupancies(ps_line, Qtips)
eigenvalues, evecs, H_QDs, Gs = chr.solveHamiltonians(ps_line, Qtips, Qsites=Q_qds, bH=True)

# Calculate transmission using LandauerQD with pre-computed Hamiltonians
energies      = np.linspace(Emin, Emax, 100)
transmissions = lqd.calculate_transmissions(ps_line, energies, H_QDs=H_QDs)

# ---- Plot results
fig = plt.figure(figsize=(10, 8))
gs  = GridSpec(2, 2, figure=fig)

# Plot QD positions and scan line
ax = fig.add_subplot(gs[0, 0])
ax.plot(ps_line[:,0], ps_line[:,1], 'k--', label='scan line')
for i in range(nsite):
    ax.plot(QDpos[i,0], QDpos[i,1], 'o', color=site_colors[i], label=f'QD {i+1}')
ax.set_xlabel('x [Å]')
ax.set_ylabel('y [Å]')
ax.legend()
ax.grid(True)
ax.axis('equal')

# Plot site occupancies
ax = fig.add_subplot(gs[0, 1])
for i in range(nsite):
    ax.plot(ps_line[:,0], Q_qds[:,i], color=site_colors[i], label=f'QD {i+1}')
ax.set_xlabel('x [Å]')
ax.set_ylabel('Site occupancy')
ax.legend()
ax.grid(True)

# Plot transmission
ax = fig.add_subplot(gs[1, :])
X, Y = np.meshgrid(ps_line[:,0], energies)
im = ax.pcolormesh(X, Y, transmissions.T, shading='auto')
plt.colorbar(im, ax=ax, label='Transmission')
ax.set_xlabel('x [Å]')
ax.set_ylabel('Energy [eV]')

plt.tight_layout()
plt.show()

# Clean up
lqd.cleanup()
