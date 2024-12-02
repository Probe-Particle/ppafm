#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pyProbeParticle import photo
from orbital_utils import load_orbital, plotMinMax, photonMap2D_stamp
from LandauerQD_py import LandauerQDs

# ========== Setup

# System parameters
Q_tip = 0.6
z_tip = 5.0
#L     = 30.0
decay = 0.5
decay_conv = 1.6
T     = 10.0

# QD system setup
nsite   = 3
R       = 7.0  # radius of circle on which sites are placed
phiRot  = 0.1 + np.pi/2
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

bDebug = True

Lcanv = 60.0
dCanv = 0.2

# ========== Functions

def evalGridStep2D( sh, lvec ):
    dix = lvec[1][0]/sh[0]; #print( "evalGridStep2D dix: ", dix, " lvec: ", lvec[1][0], " sh: ", sh[0] )
    diy = lvec[2][1]/sh[1]; #print( "evalGridStep2D diy: ", diy, " lvec: ", lvec[2][1], " sh: ", sh[1] )
    return (dix,diy)

def calculate_orbital_stm(orbital_data, orbital_lvec, QDpos, angles, canvas_shape, canvas_dd, center_region_size=None, center_region_center=None):
    """Calculate STM signal using orbital convolution for all QDs."""
    
    # Create tip field
    tipWf, _ = photo.makeTipField(canvas_shape, canvas_dd, z0=z_tip, beta=decay_conv, bSTM=True)

    # Prepare orbital data - integrate over z the upper half of the orbital grid
    orbital_2D = np.sum(orbital_data[orbital_data.shape[0]//2:,:,:], axis=0)
    orbital_2D = np.ascontiguousarray(orbital_2D.T, dtype=np.float64)
    print(f"Orbital slice shape: {orbital_2D.shape}")

    # Place all orbitals on canvas
    rhos = [orbital_2D] * len(QDpos)
    lvecs = [orbital_lvec] * len(QDpos)
    poss = [[pos[0], pos[1]] for pos in QDpos]
    canvas = photonMap2D_stamp(rhos, lvecs, canvas_shape, canvas_dd, angles=angles, poss=poss, coefs=[1.0]*len(QDpos), byCenter=True, bComplex=False)

    if bDebug:
        plotMinMax(canvas, label='Canvas (Sum Molecular Wfs)', figsize=(8,8))
        plt.scatter(QDpos[:,0], QDpos[:,1], c='g', marker='o', label='QDs')
    
    # Convolve with tip field
    stm_map = photo.convFFT(tipWf, canvas)
    result = np.real(stm_map * np.conj(stm_map))
    
    if bDebug:
        plotMinMax(result, label='STM Map (After convolution)', figsize=(8,8))
        plt.scatter(QDpos[:,0], QDpos[:,1], c='g', marker='o', label='QDs')
    
    # Crop to center region if specified
    if center_region_size is not None:
        cx, cy = center_region_center
        dx, dy = center_region_size
        result = result[cx-dx:cx+dx, cy-dy:cy+dy].T.copy()
        
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
    print(f"Molecular orbital dimensions: Lx={Lx:.3f} Å, Ly={Ly:.3f} Å")
    
    # Calculate canvas dimensions based on orbital size
    ncanv = int(np.ceil(Lcanv/dCanv))
    canvas_shape = (ncanv, ncanv)  # (ix, iy)
    canvas_dd = [dCanv, dCanv]  # Grid spacing in x and y
    print(f"Canvas shape: {canvas_shape}")
    print(f"Canvas spacing: dx={canvas_dd[0]:.3f} Å, dy={canvas_dd[1]:.3f} Å")
    
    # Create angles for each QD
    angles = np.array([phi + phiRot for phi in phis])
    
    # Define center region
    center_region_center = (canvas_shape[0]//2, canvas_shape[1]//2)
    center_region_size = (canvas_shape[0]//4, canvas_shape[1]//4)
    
    # Calculate physical dimensions of center region
    center_Lx = 2 * center_region_size[0] * canvas_dd[0]
    center_Ly = 2 * center_region_size[1] * canvas_dd[1]
    print(f"Center region physical size: {center_Lx:.3f} x {center_Ly:.3f} Å")
    
    # Calculate orbital-based STM on full canvas and crop to center
    orbital_stm = calculate_orbital_stm(orbital_data, orbital_lvec, QDpos, angles, canvas_shape, canvas_dd,  center_region_size=center_region_size, center_region_center=center_region_center)
    
    # Initialize Landauer system
    system = LandauerQDs(QDpos, E0QDs, K=K, decay=decay, tS=tS, tA=tA, Gamma_tip=Gamma_tip, Gamma_sub=Gamma_sub)
    
    # Calculate Landauer transmission map for center region
    # Create grid matching the center region size and position
    x = np.linspace(-center_Lx/2, center_Lx/2, 2*center_region_size[0])
    y = np.linspace(-center_Ly/2, center_Ly/2, 2*center_region_size[1])
    X, Y = np.meshgrid(x, y, indexing='ij')
    transmission_map = np.zeros_like(X)
    
    for i in range(2*center_region_size[0]):
        for j in range(2*center_region_size[1]):
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
        ax.scatter(QDpos[:,0], QDpos[:,1], c='g', marker='o', label='QDs')
        ax.legend()
        ax.set_xlabel('x [Å]')
        ax.set_ylabel('y [Å]')
    
    plt.tight_layout()
    
    # Save results
    np.savez('landauer_orbital_2D_scan.npz',
             transmission_map=transmission_map,
             orbital_stm=orbital_stm,
             combined_map=combined_map,
             QDpos=QDpos,
             scan_params={'L': center_Lx, 'z_tip': z_tip, 'scan_energy': scan_energy})
    
    plt.savefig('test_LandauerQD_2D_orbital.png', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
