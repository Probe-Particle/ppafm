import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import sys
sys.path.append("../../")
from pyProbeParticle import ChargeRings as chr
from LandauerQD_py import LandauerQDs

# ========== Setup

# System parameters
z_tip    = 6.0
L        = 30.0
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

# Scan parameters
scan_energy = 0.0  # Scan at Fermi level

#Q_tips = np.linspace(-1.0, 1.0, 20+1)  # Range of tip charges to scan
Q_tips = np.linspace( 0.0, 0.8, 40+1)  # Range of tip charges to scan

#use_occupancy = True  # Switch between occupancy and non-occupancy calculation
use_occupancy = False
save_dir = "Qscan_images"  # Directory to save images
cCouling = 0.03 
E_Fermi = 0.0

# Create save directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# ========== Main

# Setup system geometry
phis = np.linspace(0, 2*np.pi, nsite, endpoint=False)
QDpos = np.zeros((3, 3))
QDpos[:,0] = np.cos(phis)*R
QDpos[:,1] = np.sin(phis)*R
QDrots = chr.makeRotMats(phis + phiRot, nsite)

if use_occupancy:
    # Setup site multipoles and rotations
    QDmpols = np.zeros((3,10))
    QDmpols[:,4] = Qzz
    QDmpols[:,0] = Q0
    # Initialize ChargeRings system
    chr.initRingParams(QDpos, E0QDs, rot=QDrots, MultiPoles=QDmpols, E_Fermi=E_Fermi, cCouling=cCouling, temperature=T)

# Initialize Landauer system
system = LandauerQDs(QDpos, E0QDs, K, decay, tS, E_sub=0.0, E_tip=0.0, tA=tA, Gamma_tip=Gamma_tip, Gamma_sub=Gamma_sub)

# Setup 2D scanning grid
x = np.linspace(-L/2, L/2, npix)
y = np.linspace(-L/2, L/2, npix)
X, Y = np.meshgrid(x, y)
ps = np.zeros((npix, npix, 3))
ps[:,:,0] = X
ps[:,:,1] = Y
ps[:,:,2] = z_tip

# Reshape for charge calculation if using occupancy
if use_occupancy:
    ps_flat = ps.reshape(-1, 3)

# Loop over different tip charges
for iq, Q_tip in enumerate(Q_tips):
    print(f"Processing Q_tip = {Q_tip:.3f}")
    
    # Initialize maps
    transmission_map = np.zeros((npix, npix))
    eigenvalues_map = np.zeros((npix, npix, nsite))
    
    if use_occupancy:
        # Occupancy version
        Qtips   = np.ones(len(ps_flat)) * Q_tip
        Q_qds,_,_ = chr.solveSiteOccupancies(ps_flat, Qtips)
        eigenvalues, evecs, H_QDs, Gs = chr.solveHamiltonians(ps_flat, Qtips, Qsites=Q_qds, bH=True)
        Q_qds_map = Q_qds.reshape(npix, npix, nsite)
        eigenvalues_map = eigenvalues.reshape(npix, npix, nsite)
        
        for i in range(npix):
            for j in range(npix):
                idx = i * npix + j
                H_QD = H_QDs[idx]
                transmission_map[i,j] = system.calculate_transmission_single_energy(ps_flat[idx], scan_energy, H_QD=H_QD)
    else:
        # Non-occupancy version
        for i in range(npix):
            for j in range(npix):
                tip_pos = ps[i,j]
                transmission_map[i,j] = system.calculate_transmission(tip_pos, scan_energy, Q_tip)
                eigenvalues_map[i,j] = system.get_QD_eigenvalues(tip_pos, Q_tip)
    
    # Plot results
    fig = plt.figure(figsize=(15, 5))
    if use_occupancy:
        gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05])
    else:
        gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05])
    
    # Plot transmission map
    ax1 = fig.add_subplot(gs[0])
    im1 = ax1.imshow(transmission_map, extent=[-L/2, L/2, -L/2, L/2], origin='lower', cmap='cividis')
    ax1.set_title(f'Transmission Map (Q_tip = {Q_tip:.3f})')
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
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'Qscan_{iq:03d}_Q{Q_tip:.3f}.png'), dpi=150, bbox_inches='tight')
    plt.close()

print(f"Images saved in {save_dir}/")
