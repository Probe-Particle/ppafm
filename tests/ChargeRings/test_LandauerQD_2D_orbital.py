#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
sys.path.append("../../")
from pyProbeParticle import ChargeRings as chr
from pyProbeParticle import GridUtils as GU
from pyProbeParticle import photo
from LandauerQD_py import LandauerQDs

# ========== Setup

# System parameters
Q_tip = 0.6
z_tip = 6.0
#L     = 30.0
decay = 0.7
T     = 10.0

# QD system setup
nsite   = 3
R       = 5.0  # radius of circle on which sites are placed
phiRot  = -1.0
Q0      = 1.0
Qzz     = 15.0 * 0.0

# Energy of states on the sites
E0QDs = np.array([-1.0, -1.0, -1.0])

# Coupling parameters
K         = 0.01 # Coulomb interaction between QDs
tS        = 0.1  # QD-substrate coupling
tA        = 0.1  # Tip coupling strength
Gamma_tip = 1.0  # Tip state broadening
Gamma_sub = 1.0  # Substrate state broadening

# Energy for transmission calculation
scan_energy = 0.0  # Scan at Fermi level

def load_orbital(fname):
    """Load QD orbital from cube file."""
    try:
        orbital_data, lvec, nDim, _ = GU.loadCUBE(fname)
        return orbital_data, lvec
    except Exception as e:
        print(f"Error loading cube file {fname}: {e}")
        raise

def make_tip_field(sh, dd, z0, decay):
    """Create exponentially decaying tip wavefunction."""
    return photo.makeTipField(sh, dd, z0=z0, beta=decay, bSTM=True)

def calculate_orbital_stm(orbital_data, QDpos, angles, canvas_shape, canvas_dd):
    """Calculate STM signal using orbital convolution for all QDs."""
    # Create tip field
    tip_field, _ = make_tip_field(canvas_shape, canvas_dd, z_tip, decay)
    
    # Initialize canvas for all QDs
    canvas = np.zeros(canvas_shape, dtype=np.float64, order='C')
    
    # Place each QD orbital on canvas
    for i in range(len(QDpos)):
        pos = QDpos[i]
        # Extract middle z-slice and ensure it's contiguous
        orbital_slice = np.ascontiguousarray(orbital_data[:,:,orbital_data.shape[2]//2], dtype=np.float64)
        GU.stampToGrid2D(canvas, orbital_slice, pos[:2], angles[i], dd=canvas_dd)
    
    # Convolve with tip field
    stm_map = photo.convFFT(tip_field, canvas)
    return np.real(stm_map * np.conj(stm_map))

def main():
    # Setup system geometry
    phis = np.linspace(0, 2*np.pi, nsite, endpoint=False)
    QDpos = np.zeros((nsite, 3))
    QDpos[:,0] = np.cos(phis)*R
    QDpos[:,1] = np.sin(phis)*R
    
    # Load QD orbital
    orbital_data, orbital_lvec = load_orbital("QD.cub")
    
    # Print orbital information for debugging
    print("Orbital shape:", orbital_data.shape)
    print("Lattice vectors:")
    print(orbital_lvec)
    
    # Get physical dimensions from lattice vectors (already in Angstroms)
    # The first row is the origin offset, rows 1-3 are the lattice vectors
    Lx = abs(orbital_lvec[1,0])  # First lattice vector x component
    Ly = abs(orbital_lvec[2,1])  # Second lattice vector y component
    print(f"Physical dimensions: Lx={Lx:.3f} Å, Ly={Ly:.3f} Å")
    
    # Calculate canvas dimensions based on orbital size
    canvas_shape = orbital_data.shape[:2]  # Use original dimensions
    canvas_dd = [Lx/canvas_shape[1], Ly/canvas_shape[0]]  # Grid spacing in x and y
    print(f"Canvas shape: {canvas_shape}")
    print(f"Grid spacing: dx={canvas_dd[0]:.3f} Å, dy={canvas_dd[1]:.3f} Å")
    
    # Create angles for each QD
    angles = [phi + phiRot for phi in phis]
    
    # Calculate orbital-based STM
    orbital_stm = calculate_orbital_stm(orbital_data, QDpos, angles, canvas_shape, canvas_dd)
    
    # Initialize Landauer system
    system = LandauerQDs(QDpos, E0QDs, K=K, decay=decay, tS=tS, tA=tA, Gamma_tip=Gamma_tip, Gamma_sub=Gamma_sub)
    
    # Calculate Landauer transmission map
    x = np.linspace(0, Lx, canvas_shape[0])  # Use absolute coordinates
    y = np.linspace(0, Ly, canvas_shape[1])
    X, Y = np.meshgrid(x, y, indexing='ij')  # Use 'ij' indexing to match array order
    transmission_map = np.zeros_like(X)
    
    for i in range(canvas_shape[0]):
        for j in range(canvas_shape[1]):
            tip_pos = np.array([X[i,j], Y[i,j], z_tip])
            transmission_map[i,j] = system.calculate_transmission(tip_pos, scan_energy, Q_tip=Q_tip)
    
    # Normalize maps for better visualization
    orbital_stm = (orbital_stm - orbital_stm.min()) / (orbital_stm.max() - orbital_stm.min() + 1e-10)
    transmission_map = (transmission_map - transmission_map.min()) / (transmission_map.max() - transmission_map.min() + 1e-10)
    combined_map = transmission_map * orbital_stm
    
    # Plot results
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3, figure=fig)
    
    extent = [0, Lx, 0, Ly]  # Use absolute coordinates
    
    # Plot Landauer transmission
    ax1 = fig.add_subplot(gs[0])
    im1 = ax1.imshow(transmission_map.T, extent=extent, origin='lower', aspect='equal')
    ax1.set_title('Landauer Transmission')
    plt.colorbar(im1, ax=ax1)
    
    # Plot orbital STM
    ax2 = fig.add_subplot(gs[1])
    im2 = ax2.imshow(orbital_stm.T, extent=extent, origin='lower', aspect='equal')
    ax2.set_title('Orbital STM')
    plt.colorbar(im2, ax=ax2)
    
    # Plot combined signal
    ax3 = fig.add_subplot(gs[2])
    im3 = ax3.imshow(combined_map.T, extent=extent, origin='lower', aspect='equal')
    ax3.set_title('Combined Signal')
    plt.colorbar(im3, ax=ax3)
    
    # Add QD positions to all plots
    for ax in [ax1, ax2, ax3]:
        ax.scatter(QDpos[:,0], QDpos[:,1], c='red', marker='o', label='QDs')
        ax.legend()
        ax.set_xlabel('x [Å]')
        ax.set_ylabel('y [Å]')
    
    plt.tight_layout()
    
    # Save data
    np.savez('landauer_orbital_2D_scan.npz',
             transmission_map=transmission_map,
             orbital_stm=orbital_stm,
             combined_map=combined_map,
             QDpos=QDpos,
             scan_params={'L': Lx, 'z_tip': z_tip, 'scan_energy': scan_energy})
    
    plt.show()

if __name__ == "__main__":
    main()
