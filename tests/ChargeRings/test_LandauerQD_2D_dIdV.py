import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
sys.path.append("../../")
from pyProbeParticle import ChargeRings as chr
from LandauerQD_py import LandauerQDs

# ========== Setup

# System parameters
Q_tip = 0.6
z_tip = 6.0
L     = 30.0
npix  = 100  # Number of pixels in x and y
decay = 0.7
T     = 10.0  # Temperature for occupancy calculation

# Occupancy calculation switch
use_occupancy = True  # Set to True to use occupancy solver
cCouling      = 0.03  # Coupling parameter for occupancy calculation
E_Fermi       = 0.0   # Fermi energy level

# QD system setup
nsite  = 3
R      = 5.0  # radius of circle on which sites are placed
phiRot = -1.0
Q0     = 1.0            # Base charge for occupancy calculation
Qzz    = 15.0 * 0.0    # z-component of quadrupole moment

# Energy of states on the sites
E0QDs = np.array([-1.0, -1.0, -1.0])

# Coupling parameters
K         = 0.01 # Coulomb interaction between QDs
tS        = 0.1  # QD-substrate coupling
tA        = 0.1  # Tip coupling strength
Gamma_tip = 1.0  # Tip state broadening
Gamma_sub = 1.0  # Substrate state broadening

# dI/dV calculation parameters
V_bias = 0.1   # Bias voltage for dI/dV calculation
dV     = 0.01      # Voltage step for finite difference
Emin   = -0.1    # Energy range for transmission integration
Emax   = 0.3
#n_energies = 100
#energies = np.linspace(Emin, Emax, n_energies)

n_energies = 10
energies = np.linspace(Emin, Emax, n_energies)

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
didv_map = np.zeros((npix, npix))
eigenvalues_map = np.zeros((npix, npix, nsite))

if use_occupancy:
    # Setup site multipoles and rotations
    QDrots = chr.makeRotMats(phis + phiRot, nsite)
    QDmpols = np.zeros((3,10))
    QDmpols[:,4] = Qzz
    QDmpols[:,0] = Q0
    
    # Initialize ChargeRings system
    chr.initRingParams(QDpos, E0QDs, rot=QDrots, MultiPoles=QDmpols, E_Fermi=E_Fermi, cCouling=cCouling, temperature=T)
    
    # Reshape for charge calculation
    ps_flat = ps.reshape(-1, 3)
    Qtips = np.ones(len(ps_flat)) * Q_tip
    
    # Calculate occupancies and Hamiltonians
    Q_qds = chr.solveSiteOccupancies(ps_flat, Qtips)
    eigenvalues, evecs, H_QDs, Gs = chr.solveHamiltonians(ps_flat, Qtips, Qsites=Q_qds, bH=True)
    
    # Reshape results back to 2D
    Q_qds_map = Q_qds.reshape(npix, npix, nsite)
    eigenvalues_map = eigenvalues.reshape(npix, npix, nsite)
    
    # Initialize Landauer system
    system = LandauerQDs(QDpos, E0QDs, K, decay, tS, E_sub=0.0, E_tip=0.0, tA=tA, Gamma_tip=Gamma_tip, Gamma_sub=Gamma_sub)
    
    # Calculate dI/dV for each pixel
    for i in range(npix):
        for j in range(npix):
            print(f"Calculating dI/dV for pixel {i}, {j}")
            idx = i * npix + j
            Hqd = H_QDs[idx]
            # Calculate dI/dV using finite difference
            I_plus  = system.calculate_current(ps_flat[idx], energies, V_bias=V_bias + dV/2, Hqd=Hqd, T=T)
            I_minus = system.calculate_current(ps_flat[idx], energies, V_bias=V_bias - dV/2, Hqd=Hqd, T=T)
            didv_map[i,j] = (I_plus - I_minus) / dV
else:
    # Initialize Landauer system without occupancy
    system = LandauerQDs(QDpos, E0QDs, K, decay, tS, E_sub=0.0, E_tip=0.0, tA=tA, Gamma_tip=Gamma_tip, Gamma_sub=Gamma_sub)
    
    # Calculate dI/dV and eigenvalues directly
    Qtips = np.ones(len(ps_flat))*Q_tip  # Create array of tip charges
    for i in range(npix):
        for j in range(npix):
            print(f"Calculating dI/dV for pixel {i}, {j}")
            tip_pos = ps[i,j]
            idx = i * npix + j
            # Calculate dI/dV using finite difference
            I_plus  = system.calculate_current(tip_pos, energies, V_bias + dV/2, Q_tip=Qtips[idx], T=T)
            I_minus = system.calculate_current(tip_pos, energies, V_bias - dV/2, Q_tip=Qtips[idx], T=T)
            didv_map[i,j] = (I_plus - I_minus) / dV
            eigenvalues_map[i,j] = system.get_QD_eigenvalues(tip_pos, Qtips[idx])

# Plot results
if use_occupancy:
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05])
else:
    fig = plt.figure(figsize=(12, 5))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05])

# Plot dI/dV map
ax1 = fig.add_subplot(gs[0])
im1 = ax1.imshow(didv_map, extent=[-L/2, L/2, -L/2, L/2], origin='lower', cmap='cividis')
ax1.set_title(f'dI/dV Map (V_bias = {V_bias:.2f}V)')
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
np.savez('landauer_2D_didv_scan.npz', 
         didv_map=didv_map,
         eigenvalues_map=eigenvalues_map,
         QDpos=QDpos,
         scan_params={'L': L, 'z_tip': z_tip, 'V_bias': V_bias, 'dV': dV})
