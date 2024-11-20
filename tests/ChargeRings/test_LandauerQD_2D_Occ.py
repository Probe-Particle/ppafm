import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
sys.path.append("../../")
from pyProbeParticle import ChargeRings as chr
from LandauerQD import LandauerQDs

# ========== Setup

# System parameters (matching test_ChargeRings3.py)
Q_tip    = 0.6
cCouling = 0.03
E_Fermi  = 0.0
z_tip    = 6.0
L        = 20.0
npix     = 100  # Number of pixels in x and y
decay    = 0.7
T        = 10.0

# QD system setup
nsite  = 3
R      = 5.0  # radius of circle on which sites are placed
phiRot = -1.0
Q0     = 1.0
Qzz    = 15.0 * 0.0

# Energy of states on the sites
E0QDs = np.array([-1.0, -1.0, -1.0])

# Coupling parameters
K  = 0.01        # Coulomb interaction between QDs
tS = 0.1         # QD-substrate coupling
tA = 0.1         # Tip coupling strength
Gamma_tip = 1.0  # Tip state broadening
Gamma_sub = 1.0  # Substrate state broadening

# Energy for transmission calculation
scan_energy = 0.0  # Scan at Fermi level

# ========== Main

# Setup system geometry
phis = np.linspace(0, 2*np.pi, nsite, endpoint=False)
QDpos = np.zeros((3, 3))
QDpos[:,0] = np.cos(phis)*R
QDpos[:,1] = np.sin(phis)*R

# Setup site multipoles and rotations
QDrots = chr.makeRotMats(phis + phiRot, nsite)
QDmpols = np.zeros((3,10))
QDmpols[:,4] = Qzz
QDmpols[:,0] = Q0

# Initialize systems
chr.initRingParams(QDpos, E0QDs, rot=QDrots, MultiPoles=QDmpols,  E_Fermi=E_Fermi, cCouling=cCouling, temperature=T)
system = LandauerQDs(QDpos, E0QDs, K, decay, tS, E_sub=0.0, E_tip=0.0, tA=tA, Gamma_tip=Gamma_tip, Gamma_sub=Gamma_sub)

# Setup 2D scanning grid
x = np.linspace(-L/2, L/2, npix)
y = np.linspace(-L/2, L/2, npix)
X, Y = np.meshgrid(x, y)
ps = np.zeros((npix, npix, 3))
ps[:,:,0] = X
ps[:,:,1] = Y
ps[:,:,2] = z_tip

# Reshape for charge calculation
ps_flat = ps.reshape(-1, 3)
Qtips = np.ones(len(ps_flat)) * Q_tip

# Calculate charges and Hamiltonians for all positions
print("Calculating charge occupancies...")
Q_qds = chr.solveSiteOccupancies(ps_flat, Qtips)
print("Solving Hamiltonians...")
eigenvalues, evecs, H_QDs, Gs = chr.solveHamiltonians(ps_flat, Qtips, Qsites=Q_qds, bH=True)

# Calculate transmission at scan_energy for each point
print("Calculating transmission...")
transmission_map = np.zeros((npix, npix))
eigenvalues_map = eigenvalues.reshape(npix, npix, nsite)
Q_qds_map = Q_qds.reshape(npix, npix, nsite)

for i in range(npix):
    for j in range(npix):
        idx = i * npix + j
        H_QD = H_QDs[idx]
        transmission_map[i,j] = system.calculate_transmission_single_energy(
            ps_flat[idx], scan_energy, H_QD=H_QD)

# Plot results
fig = plt.figure(figsize=(15, 5))
gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05])

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
im2 = ax2.imshow(np.mean(eigenvalues_map, axis=2),     extent=[-L/2, L/2, -L/2, L/2],   origin='lower', cmap='cividis')
ax2.set_title('Average QD Energy')
ax2.set_xlabel('X (Å)')
ax2.plot(QDpos[:,0], QDpos[:,1], 'r.', markersize=10, label='QDs')
ax2.legend()

# Plot average QD charge map
ax3 = fig.add_subplot(gs[2])
im3 = ax3.imshow(np.mean(Q_qds_map, axis=2),
                 extent=[-L/2, L/2, -L/2, L/2],
                 origin='lower', cmap='cividis')
ax3.set_title('Average QD Charge')
ax3.set_xlabel('X (Å)')
ax3.plot(QDpos[:,0], QDpos[:,1], 'r.', markersize=10, label='QDs')
ax3.legend()

# Add colorbar
cbar_ax = fig.add_subplot(gs[3])
plt.colorbar(im1, cax=cbar_ax, label='Transmission')

plt.tight_layout()
plt.show()

# Optional: Save the data
np.savez('landauer_2D_scan_with_charge.npz', 
         transmission_map=transmission_map,
         eigenvalues_map=eigenvalues_map,
         Q_qds_map=Q_qds_map,
         QDpos=QDpos,
         scan_params={'L': L, 'z_tip': z_tip, 'scan_energy': scan_energy})
