import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from LandauerQD import LandauerQDs
import sys
sys.path.append("../../")
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
tS = 0.1  # QD-substrate coupling
tA = 0.1   # Tip coupling strength
Gamma_tip = 1.0  # Tip state broadening
Gamma_sub = 1.0  # Substrate state broadening

Emin = -0.1  # Increased range to see effect of broadening
Emax =  0.3


site_colors = ['r', 'g', 'b']


# ========== Main


# ---- Setup geometry and electrostatic potential of the quantum dot system
phis = np.linspace(0, 2*np.pi, nsite, endpoint=False)
QDpos = np.zeros((3, 3))
QDpos[:,0] = np.cos(phis)*R
QDpos[:,1] = np.sin(phis)*R
# rotation of multipoles on sites
QDrots = chr.makeRotMats( phis + phiRot, nsite )
# Setup site multipoles
QDmpols = np.zeros((3,10))
QDmpols[:,4] = Qzz
QDmpols[:,0] = Q0

# ---- Initialize Quantum dot system
chr.initRingParams(QDpos, E0QDs, rot=QDrots, MultiPoles=QDmpols, E_Fermi=E_Fermi, cCouling=cCouling, temperature=T )    # setup Quantum dot system in C++ library
system = LandauerQDs(QDpos, E0QDs, K, decay, tS, E_sub=0.0, E_tip=0.0, tA=tA,  Gamma_tip=Gamma_tip, Gamma_sub=Gamma_sub)  # initialize Landauer system in python

# ---- Setup scanning line (matching test_ChargeRings_1D.py)
ps_line      = chr.getLine(QDpos, [0.5,0.5,-3.0], [-3.0,-3.0,1.0], n=400)      # generate scanning line points
ps_line[:,2] = z_tip                                                           # set scan height above the sample
Qtips        = np.ones(len(ps_line))*Q_tip                                     # Charge of tip along the line ( constant for now )
energies     = np.linspace( Emin, Emax, 100)        # Energy range energies at which we sample the Transmission function in Landauer conductance calculation

# ---- Calculate transmission
Q_qds                         = chr.solveSiteOccupancies( ps_line, Qtips )                        #  solve quantum dot (sites) occupancies (charge) 
eigenvalues, evecs, H_QDs, Gs = chr.solveHamiltonians( ps_line, Qtips, Qsites=Q_qds, bH=True  )   #  get hamiltonian, eigenvalues, and eigenvectors for QD system considering fixed pre-computed charge occupancy
transmissions                 = system.scan_1D(ps_line, energies, H_QDs=H_QDs )                   # calculate transmission function using pre-computed Hamiltonian (for fixed charge occupancy of QDs)


print( "Q_qds ", Q_qds.shape, Q_qds.dtype )

# ---- Plot results using GridSpec
fig = plt.figure(figsize=(12, 12))
gs = GridSpec(3, 2, width_ratios=[1, 0.05], height_ratios=[1, 1, 1])

#cmap='plasma'
cmap='cividis'
#cmap='afmhot'

# ---- Plot Panel 1 - 2D transmission spectrum superimposed with QD levels bands

# Plot transmission with eigenvalue bands
ax1 = fig.add_subplot(gs[0, 0])
im = ax1.imshow(transmissions.T, aspect='auto', origin='lower',  extent=[0, len(ps_line), energies[0], energies[-1]],  vmin=0.0, vmax=0.00001, cmap=cmap )

xs = range(len(ps_line))

# Overlay eigenvalue bands
for i in range(nsite):
    ax1.plot( xs, eigenvalues[:, i], ':', alpha=0.7,                            label=f'Eps_{i+1}'      )   # plot eigenvalues (diagonalized Hamiltonian)
    ax1.plot( xs, H_QDs[:,i,i],      '-', alpha=1.0, lw=0.7, c=site_colors[i] , label=f'H({i+1},{i+1})' )   # plot on-site energies of the Hamiltonian

ax1.legend()
ax1.set_xlim(0, len(ps_line))
ax1.set_ylim(energies[0], energies[-1])
ax1.set_xlabel('Position along line')
ax1.set_ylabel('Energy (eV)')
ax1.set_title('Transmission and Energy Levels')

# Add colorbar
cbar_ax1 = fig.add_subplot(gs[0, 1])
plt.colorbar(im, cax=cbar_ax1, label='Transmission')

# ---- Plot Panel 2 - Total transmission vs position

ax2 = fig.add_subplot(gs[1, 0])
selected_energies = [-0.02, 0.0, 0.02]
for E in selected_energies:
    idx = np.argmin(np.abs(energies - E))
    print( f"Energy: {E:.1f} eV, Index: {idx}" )
    ax2.plot(transmissions[:, idx], label=f'E = {E:.1f} eV')
    ax1.axhline(E, ls=':', c='k', alpha=0.5)
ax2.set_xlabel('Position Index')
ax2.set_ylabel('Transmission')
ax2.legend()
ax2.set_title('Transmission vs Position at Selected Energies')
plt.grid()

# ---- Plot Panel 3 - Charge occupancies
ax3 = fig.add_subplot(gs[2, 0])
for i in range(nsite):
    ax3.plot(xs, Q_qds[:, i], '-', c=site_colors[i], label=f'QD {i+1}')
ax3.set_xlim(0, len(ps_line))
ax3.set_xlabel('Position along line')
ax3.set_ylabel('Charge Occupancy')
ax3.grid(True)
ax3.legend()

plt.tight_layout()
plt.show()