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
L     = 30.0
npix  = 100  # Number of pixels in x and y
decay = 0.7
T     = 10.0

# Occupancy calculation switch
use_occupancy = True  # Set to True to use occupancy solver
cCouling = 0.03       # Coupling parameter for occupancy calculation
E_Fermi = 0.0        # Fermi energy level

# QD system setup
nsite = 3
R = 5.0  # radius of circle on which sites are placed
phiRot = -1.0
Q0 = 1.0            # Base charge for occupancy calculation
Qzz = 15.0 * 0.0    # z-component of quadrupole moment

# Energy of states on the sites
E0QDs = np.array([-1.0, -1.0, -1.0])

# Coupling parameters
K = 0.01    # Coulomb interaction between QDs
tS = 0.1    # QD-substrate coupling
tA = 0.1    # Tip coupling strength
Gamma_tip = 1.0  # Tip state broadening
Gamma_sub = 1.0  # Substrate state broadening
eta = 1e-8      # Green's function broadening

# Energy for transmission calculation
scan_energy = 0.0  # Scan at Fermi level

# ========== Main

# Setup system geometry
phis = np.linspace(0, 2*np.pi, nsite, endpoint=False)
QDpos = np.zeros((3, 3))
QDpos[:,0] = np.cos(phis)*R
QDpos[:,1] = np.sin(phis)*R

# Setup 2D scanning grid
x = np.linspace(-L/2, L/2, npix)
y = np.linspace(-L/2, L/2, npix)
X, Y = np.meshgrid(x, y)
ps = np.zeros((npix, npix, 3))
ps[:,:,0] = X
ps[:,:,1] = Y
ps[:,:,2] = z_tip

# Initialize maps
transmission_map = np.zeros((npix, npix))
eigenvalues_map = np.zeros((npix, npix, nsite))

# Reshape for charge calculation
ps_flat = ps.reshape(-1, 3)
Qtips = np.ones(len(ps_flat)) * Q_tip

if use_occupancy:
    # Setup site multipoles and rotations
    QDrots = chr.makeRotMats(phis + phiRot, nsite)
    QDmpols = np.zeros((3,10))
    QDmpols[:,4] = Qzz
    QDmpols[:,0] = Q0
    
    # Initialize ChargeRings system
    chr.initRingParams(QDpos, E0QDs, rot=QDrots, MultiPoles=QDmpols, E_Fermi=E_Fermi, cCouling=cCouling, temperature=T)
    
    # Calculate occupancies and Hamiltonians
    Q_qds = chr.solveSiteOccupancies(ps_flat, Qtips)
    eigenvalues, evecs, H_QDs, Gs = chr.solveHamiltonians(ps_flat, Qtips, Qsites=Q_qds, bH=True)
    
    # Initialize C++ solver with occupancy
    cpp_solver.init(QDpos.T, E0QDs, K=K, decay=decay, tS=tS, tA=tA, Gamma_tip=Gamma_tip, Gamma_sub=Gamma_sub, eta=eta)
    
    # Reshape results back to 2D
    Q_qds_map = Q_qds.reshape(npix, npix, nsite)
    eigenvalues_map = eigenvalues.reshape(npix, npix, nsite)
    
    # Calculate transmission using pre-computed Hamiltonians
    for i in range(npix):
        for j in range(npix):
            idx = i * npix + j
            Hqd = H_QDs[idx]
            transmission_map[i,j] = cpp_solver.calculate_transmission(scan_energy, ps_flat[idx], Qtips[idx], Hqd.astype(np.float64))
else:
    # Initialize C++ solver without occupancy
    cpp_solver.init(QDpos.T, E0QDs, K=K, decay=decay, tS=tS, tA=tA, Gamma_tip=Gamma_tip, Gamma_sub=Gamma_sub, eta=eta)
    
    # Calculate transmission directly
    for i in range(npix):
        for j in range(npix):
            tip_pos = ps[i,j]
            transmission_map[i,j] = cpp_solver.calculate_transmission(scan_energy, tip_pos, Q_tip)

# Plot results
if use_occupancy:
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05])
else:
    fig = plt.figure(figsize=(12, 5))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05])

# Plot transmission map
ax1 = fig.add_subplot(gs[0])
im1 = ax1.imshow(transmission_map, extent=[-L/2, L/2, -L/2, L/2], origin='lower', cmap='cividis')
ax1.set_title('Transmission Map')
ax1.set_xlabel('X (Å)')
ax1.set_ylabel('Y (Å)')
ax1.plot(QDpos[:,0], QDpos[:,1], 'r.', markersize=10, label='QDs')
ax1.legend()

# Plot average QD energy map
ax2 = fig.add_subplot(gs[1])
im2 = ax2.imshow(np.mean(eigenvalues_map, axis=2), extent=[-L/2, L/2, -L/2, L/2], origin='lower', cmap='cividis')
ax2.set_title('Average QD Energy')
ax2.set_xlabel('X (Å)')
ax2.plot(QDpos[:,0], QDpos[:,1], 'r.', markersize=10, label='QDs')
ax2.legend()

if use_occupancy:
    # Plot average QD charge map
    ax3 = fig.add_subplot(gs[2])
    im3 = ax3.imshow(np.mean(Q_qds_map, axis=2), extent=[-L/2, L/2, -L/2, L/2], origin='lower', cmap='cividis')
    ax3.set_title('Average QD Charge')
    ax3.set_xlabel('X (Å)')
    ax3.plot(QDpos[:,0], QDpos[:,1], 'r.', markersize=10, label='QDs')
    ax3.legend()
    
    # Colorbar
    plt.colorbar(im3, cax=fig.add_subplot(gs[3]))
else:
    # Colorbar
    plt.colorbar(im2, cax=fig.add_subplot(gs[2]))

plt.tight_layout()
plt.show()

# Optional: Save the data
np.savez('landauer_2D_scan.npz', 
         transmission_map=transmission_map,
         eigenvalues_map=eigenvalues_map,
         QDpos=QDpos,
         scan_params={'L': L, 'z_tip': z_tip, 'scan_energy': scan_energy})
