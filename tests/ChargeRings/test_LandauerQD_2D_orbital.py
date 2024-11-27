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

def calculate_orbital_stm(orbital_data, QDpos, angles, canvas_shape, canvas_dd, center_region=None):
    """Calculate STM signal using orbital convolution for all QDs."""
    # Create tip field
    tip_field, _ = make_tip_field(canvas_shape, canvas_dd, z_tip, decay)
    
    # Initialize canvas for all QDs
    canvas = np.zeros(canvas_shape, dtype=np.float64, order='C')
    
    # Place each QD orbital on canvas
    for i in range(len(QDpos)):
        pos = QDpos[i]
        # Take xy plane (z-middle slice) and transpose to fix axis orientation
        orbital_slice = np.ascontiguousarray(orbital_data[orbital_data.shape[0]//2,:,:].T, dtype=np.float64)
        print(f"Orbital slice shape: {orbital_slice.shape}")
        print(f"Stamping QD {i} at position: {pos[:2]} Å")
        GU.stampToGrid2D(canvas, orbital_slice, pos[:2]/canvas_dd, angles[i], dd=canvas_dd)
    
    # Print canvas statistics before convolution
    print(f"Canvas before convolution - min: {canvas.min():.3e}, max: {canvas.max():.3e}")
    
    # Plot canvas before convolution (full canvas)
    plt.figure(figsize=(8, 8))
    plt.imshow(canvas.T, origin='lower', aspect='equal')
    # Mark the center and QD positions
    plt.axhline(y=canvas_shape[1]//2, color='r', linestyle='--', alpha=0.3)
    plt.axvline(x=canvas_shape[0]//2, color='r', linestyle='--', alpha=0.3)
    plt.scatter(QDpos[:,0]/canvas_dd[0] + canvas_shape[0]//2, 
               QDpos[:,1]/canvas_dd[1] + canvas_shape[1]//2, 
               c='red', marker='o', label='QDs')
    plt.colorbar(label='Orbital density')
    plt.title('Canvas before convolution (full canvas)')
    plt.xlabel('x [grid points]')
    plt.ylabel('y [grid points]')
    plt.legend()
    plt.tight_layout()
    
    # Convolve with tip field
    stm_map = photo.convFFT(tip_field, canvas)
    result = np.real(stm_map * np.conj(stm_map))
    
    print(f"STM map after convolution - min: {result.min():.3e}, max: {result.max():.3e}")
    
    # Plot full canvas after convolution
    plt.figure(figsize=(8, 8))
    plt.imshow(result.T, origin='lower', aspect='equal')
    # Mark the center region that will be cropped
    if center_region is not None:
        cx, cy = canvas_shape[0]//2, canvas_shape[1]//2
        dx, dy = center_region
        rect = plt.Rectangle((cx-dx, cy-dy), 2*dx, 2*dy, 
                           fill=False, color='red', linestyle='--', label='Crop region')
        plt.gca().add_patch(rect)
    plt.scatter(QDpos[:,0]/canvas_dd[0] + canvas_shape[0]//2, 
               QDpos[:,1]/canvas_dd[1] + canvas_shape[1]//2, 
               c='red', marker='o', label='QDs')
    plt.colorbar(label='STM signal')
    plt.title('Canvas after convolution (full canvas)')
    plt.xlabel('x [grid points]')
    plt.ylabel('y [grid points]')
    plt.legend()
    plt.tight_layout()
    
    # Crop to center region if specified
    if center_region is not None:
        cx, cy = canvas_shape[0]//2, canvas_shape[1]//2
        dx, dy = center_region
        result = result[cx-dx:cx+dx, cy-dy:cy+dy]
        
    return result

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
    # Note: orbital_data shape is (iz, iy, ix)
    Lx = abs(orbital_lvec[1,0])  # First lattice vector x component
    Ly = abs(orbital_lvec[2,1])  # Second lattice vector y component
    print(f"Physical dimensions: Lx={Lx:.3f} Å, Ly={Ly:.3f} Å")
    
    # Calculate canvas dimensions based on orbital size
    canvas_shape = (orbital_data.shape[2], orbital_data.shape[1])  # (ix, iy)
    canvas_dd = [Lx/canvas_shape[0], Ly/canvas_shape[1]]  # Grid spacing in x and y
    print(f"Canvas shape: {canvas_shape}")
    print(f"Grid spacing: dx={canvas_dd[0]:.3f} Å, dy={canvas_dd[1]:.3f} Å")
    
    # Define center region size (in pixels)
    center_pixels = 50  # This will give us a 100x100 pixel region around center
    
    # Create angles for each QD
    angles = [phi + phiRot for phi in phis]
    
    # Calculate orbital-based STM on full canvas and crop to center
    orbital_stm = calculate_orbital_stm(orbital_data, QDpos, angles, canvas_shape, canvas_dd,
                                      center_region=(center_pixels, center_pixels))
    
    # Calculate physical dimensions of center region
    center_Lx = 2 * center_pixels * canvas_dd[0]
    center_Ly = 2 * center_pixels * canvas_dd[1]
    
    # Initialize Landauer system
    system = LandauerQDs(QDpos, E0QDs, K=K, decay=decay, tS=tS, tA=tA, Gamma_tip=Gamma_tip, Gamma_sub=Gamma_sub)
    
    # Calculate Landauer transmission map only for center region
    x = np.linspace(-center_Lx/2, center_Lx/2, 2*center_pixels)
    y = np.linspace(-center_Ly/2, center_Ly/2, 2*center_pixels)
    X, Y = np.meshgrid(x, y, indexing='ij')
    transmission_map = np.zeros_like(X)
    
    for i in range(2*center_pixels):
        for j in range(2*center_pixels):
            tip_pos = np.array([X[i,j], Y[i,j], z_tip])
            transmission_map[i,j] = system.calculate_transmission(tip_pos, scan_energy, Q_tip=Q_tip)
    
    print(f"Landauer transmission - min: {transmission_map.min():.3e}, max: {transmission_map.max():.3e}")
    
    # Normalize maps for better visualization
    orbital_stm = (orbital_stm - orbital_stm.min()) / (orbital_stm.max() - orbital_stm.min() + 1e-10)
    transmission_map = (transmission_map - transmission_map.min()) / (transmission_map.max() - transmission_map.min() + 1e-10)
    combined_map = transmission_map * orbital_stm
    
    # Plot results
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3, figure=fig)
    
    extent = [-center_Lx/2, center_Lx/2, -center_Ly/2, center_Ly/2]  # Center around origin
    
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
             scan_params={'L': center_Lx, 'z_tip': z_tip, 'scan_energy': scan_energy})
    
    plt.show()

if __name__ == "__main__":
    main()
