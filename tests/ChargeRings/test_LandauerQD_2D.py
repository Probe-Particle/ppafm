import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
sys.path.append("../../")
from pyProbeParticle import ChargeRings as chr
from LandauerQD import LandauerQDs

# ========== Setup

# # System parameters (matching test_ChargeRings3.py)
# Q_tip = 0.6
# z_tip = 6.0
# L = 20.0
# npix = 100  # Number of pixels in x and y
# decay = 0.7

# # QD system setup
# nsite = 3
# R = 5.0  # radius of circle on which sites are placed
# phiRot = -1.0

# # Energy of states on the sites
# Esite = np.array([-1.0, -1.0, -1.0])

# # Coupling parameters
# K = 0.01   # Coulomb interaction between QDs
# tS = 0.1   # QD-substrate coupling
# tA = 0.1   # Tip coupling strength
# Gamma_tip = 1.0  # Tip state broadening
# Gamma_sub = 1.0  # Substrate state broadening

# # Energy for transmission calculation




# System parameters (matching test_ChargeRings_1D.py)
Q_tip = 0.6
z_tip = 6.0
L = 20.0
decay = 0.7
npix = 100  # Number of pixels in x and y

# QD system setup
nsite = 3
R = 5.0
phiRot = -1.0

# Energy of states on the sites
Esite = np.array([-1.0, -1.0, -1.0])

K  = 0.01  # Coulomb interaction between QDs
tS = 0.1  # QD-substrate coupling
tA = 0.1   # Tip coupling strength
Gamma_tip = 1.0  # Tip state broadening
Gamma_sub = 1.0  # Substrate state broadening

Emin = -0.2  # Increased range to see effect of broadening
Emax =  0.2
scan_energy = 0.0  # Scan at Fermi level


# ========== Main

# Setup system geometry
phis = np.linspace(0, 2*np.pi, nsite, endpoint=False)
QDpos = np.zeros((3, 3))
QDpos[:,0] = np.cos(phis)*R
QDpos[:,1] = np.sin(phis)*R

# Initialize Landauer system
system = LandauerQDs(QDpos, Esite, K, decay, tS, E_sub=0.0, E_tip=0.0, tA=tA,  Gamma_tip=Gamma_tip, Gamma_sub=Gamma_sub)

# Setup 2D scanning grid
x = np.linspace(-L/2, L/2, npix)
y = np.linspace(-L/2, L/2, npix)
X, Y = np.meshgrid(x, y)
ps = np.zeros((npix, npix, 3))
ps[:,:,0] = X
ps[:,:,1] = Y
ps[:,:,2] = z_tip

# Calculate transmission for each point
transmission_map = np.zeros((npix, npix))
eigenvalues_map = np.zeros((npix, npix, nsite))

for i in range(npix):
    for j in range(npix):
        tip_pos = ps[i,j]
        transmission_map[i,j] = system.calculate_transmission(tip_pos, scan_energy, Q_tip)
        eigenvalues_map[i,j]  = system.get_QD_eigenvalues(tip_pos, Q_tip)

# Plot results
fig = plt.figure(figsize=(15, 5))
gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05])

# Plot transmission map
ax1 = fig.add_subplot(gs[0])
im1 = ax1.imshow(transmission_map, extent=[-L/2, L/2, -L/2, L/2],  origin='lower', cmap='cividis')
ax1.set_title('Transmission Map')
ax1.set_xlabel('X (Å)')
ax1.set_ylabel('Y (Å)')

# Plot QD positions
ax1.plot(QDpos[:,0], QDpos[:,1], 'r.', markersize=10, label='QDs')
ax1.legend()

# Plot eigenvalues map (average of all QD energies)
ax2 = fig.add_subplot(gs[1])
im2 = ax2.imshow(np.mean(eigenvalues_map, axis=2),  extent=[-L/2, L/2, -L/2, L/2],  origin='lower', cmap='cividis')
ax2.set_title('Average QD Energy')
ax2.set_xlabel('X (Å)')

# Plot QD positions
ax2.plot(QDpos[:,0], QDpos[:,1], 'r.', markersize=10, label='QDs')
ax2.legend()

# Add colorbar
cbar_ax = fig.add_subplot(gs[2])
plt.colorbar(im1, cax=cbar_ax, label='Transmission')

plt.tight_layout()
plt.show()

# Optional: Save the data
np.savez('landauer_2D_scan.npz', 
         transmission_map=transmission_map,
         eigenvalues_map=eigenvalues_map,
         QDpos=QDpos,
         scan_params={'L': L, 'z_tip': z_tip, 'scan_energy': scan_energy})
