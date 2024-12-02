#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
sys.path.append("../../")
from pyProbeParticle import ChargeRings as chr
from pyProbeParticle import GridUtils as GU
from pyProbeParticle import photo
from orbital_utils import load_orbital, plotMinMax, photonMap2D_stamp

# ========== Setup Parameters

# System parameters from test_ChargeRings_2D.py
V_Bias    = 1.0
Q_tip     = 0.6
cCouling  = 0.02
E_Fermi   = 0.0
z_tip     = 5.0
L         = 20.0
npix      = 400
decay     = 0.5
T         = 10.0

# QD system setup
nsite  = 3
R      = 7.0
phiRot = 0.1 + np.pi/2

Q0  = 1.0
Qzz = 15.0 * 0.0

# Canvas parameters from test_LandauerQD_2D_orbital.py
Lcanv = 60.0
dCanv = 0.2
decay_conv = 1.6

# Energy of states on the sites
Esite = [-1.0, -1.0, -1.0]

#bDebug = False
bDebug = True

# ========== Functions

def calculate_orbital_stm_single(orbital_data, orbital_lvec, QDpos, angle, canvas_shape, canvas_dd, center_region_size=None, center_region_center=None):
    """Calculate STM signal using orbital convolution for a single QD."""
    
    # Create tip field
    tipWf, _ = photo.makeTipField(canvas_shape, canvas_dd, z0=z_tip, beta=decay_conv, bSTM=True)

    # Prepare orbital data - integrate over z the upper half of the orbital grid
    # Note: orbital_data shape is (iz, iy, ix)
    orbital_2D = np.sum(orbital_data[orbital_data.shape[0]//2:,:,:], axis=0)
    orbital_2D = np.ascontiguousarray(orbital_2D.T, dtype=np.float64)  # Now shape is (ix, iy)
    
    # Place orbital on canvas
    canvas = photonMap2D_stamp(
        [orbital_2D], [orbital_lvec], canvas_shape, canvas_dd,
        angles=[angle], poss=[[QDpos[0], QDpos[1]]], coefs=[1.0],
        byCenter=True, bComplex=False
    )
    
    if bDebug:
        plotMinMax(canvas, label='Canvas (Single Molecular Wf)', figsize=(8,8))
        plt.scatter(QDpos[0], QDpos[1], c='g', marker='o', label='QD')
    
    # Convolve with tip field
    stm_map = photo.convFFT(tipWf, canvas)
    result = np.real(stm_map * np.conj(stm_map))
    
    if bDebug:
        plotMinMax(result, label='STM Map (After convolution)', figsize=(8,8))
        plt.scatter(QDpos[0], QDpos[1], c='g', marker='o', label='QD')
    
    # Crop to center region if specified
    if center_region_size is not None:
        cx, cy = center_region_center
        dx, dy = center_region_size
        result = result[cx-dx:cx+dx, cy-dy:cy+dy].T.copy()
        
    return result

def main():
    # Setup system geometry
    phis = np.linspace(0, 2*np.pi, nsite, endpoint=False)
    spos = np.zeros((nsite, 3))
    spos[:,0] = np.cos(phis)*R
    spos[:,1] = np.sin(phis)*R
    angles = phis + phiRot
    
    # Setup site multipoles
    mpols = np.zeros((nsite, 10))
    mpols[:,4] = Qzz
    mpols[:,0] = Q0
    
    # Initialize global parameters for ChargeRings
    rots = chr.makeRotMats(phis + phiRot, nsite)
    chr.initRingParams(spos, Esite, rot=rots, MultiPoles=mpols, E_Fermi=E_Fermi, 
                      cCouling=cCouling, temperature=T, onSiteCoulomb=3.0)
    
    # Set up configuration basis
    confstrs = ["000","001","010","100","110","101","011","111"]
    confs = chr.confsFromStrings(confstrs)
    nconfs = chr.setSiteConfBasis(confs)
    
    # Load QD orbital and print information
    orbital_data, orbital_lvec = load_orbital("QD.cub")
    print("Orbital shape:", orbital_data.shape)
    print("Lattice vectors:")
    print(orbital_lvec)
    
    # Get physical dimensions from lattice vectors (already in Angstroms)
    # Note: orbital_data shape is (iz, iy, ix)
    Lx = abs(orbital_lvec[1,0])  # First lattice vector x component
    Ly = abs(orbital_lvec[2,1])  # Second lattice vector y component
    print(f"Molecular orbital dimensions: Lx={Lx:.3f} Å, Ly={Ly:.3f} Å")
    
    # Setup canvas
    ncanv = int(np.ceil(Lcanv/dCanv))
    canvas_shape = (ncanv, ncanv)  # (ix, iy)
    canvas_dd = [dCanv, dCanv]  # Grid spacing in x and y
    print(f"Canvas shape: {canvas_shape}")
    print(f"Canvas spacing: dx={canvas_dd[0]:.3f} Å, dy={canvas_dd[1]:.3f} Å")
    
    # Define center region
    center_region_center = (canvas_shape[0]//2, canvas_shape[1]//2)
    center_region_size = (canvas_shape[0]//4, canvas_shape[1]//4)
    
    # Calculate physical dimensions of center region
    center_Lx = 2 * center_region_size[0] * canvas_dd[0]
    center_Ly = 2 * center_region_size[1] * canvas_dd[1]
    print(f"Center region physical size: {center_Lx:.3f} x {center_Ly:.3f} Å")
    
    # Create grid for the center region
    x = np.linspace(-center_Lx/2, center_Lx/2, 2*center_region_size[0])
    y = np.linspace(-center_Ly/2, center_Ly/2, 2*center_region_size[1])
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Create tip positions array for the center region
    ps = np.zeros((len(x) * len(y), 3))
    ps[:,0] = X.flatten()
    ps[:,1] = Y.flatten()
    ps[:,2] = z_tip
    Qtips = np.ones(len(ps)) * Q_tip
    
    # Calculate site occupancies using ChargeRings
    Q_1, Es_1, Ec_1 = chr.solveSiteOccupancies(ps, Qtips, bEsite=True, solver_type=2)
    
    # Initialize arrays for storing results
    total_current = np.zeros((2*center_region_size[0], 2*center_region_size[1]))
    site_currents = []
    
    # Calculate current for each site
    for i in range(nsite):
        # Calculate orbital convolution for this site
        orbital_stm = calculate_orbital_stm_single(
            orbital_data, orbital_lvec, spos[i], angles[i],
            canvas_shape, canvas_dd, center_region_size, center_region_center
        )
        
        # Calculate site current coefficients
        site_current = chr.calculate_site_current(
            ps, spos[i], Es_1[:,i],
            E_Fermi + V_Bias, E_Fermi,
            decay=decay, T=T
        )
        
        # Reshape site current to match orbital_stm shape
        site_current = site_current.reshape(orbital_stm.shape)
        
        # Multiply orbital convolution with site current coefficients
        current_contribution = orbital_stm * site_current
        
        # Add to total current and store individual contribution
        total_current += current_contribution
        site_currents.append(current_contribution)
    
    # Plot results
    extent = [-center_Lx/2, center_Lx/2, -center_Ly/2, center_Ly/2]
    
    fig, axs = plt.subplots(1, nsite+1, figsize=(3*(nsite+1), 6))
    for i in range(nsite):
        axs[i].imshow(site_currents[i], extent=extent, origin='lower', aspect='equal')
        axs[i].set_title(f'Site {i+1} Contribution')
        axs[i].scatter(spos[:,0], spos[:,1], c='g', marker='o', label='QDs')
    axs[-1].imshow(total_current, extent=extent, origin='lower', aspect='equal')
    axs[-1].set_title('Total STM Current')
    axs[-1].scatter(spos[:,0], spos[:,1], c='g', marker='o', label='QDs')
    for ax in axs:
        ax.set_xlabel('X [Å]')
        ax.set_ylabel('Y [Å]')
        ax.legend()
    plt.tight_layout()
    plt.show()

def makeCanvas( canvas_shape, canvas_dd, ):
    # Set up canvas parameters
    Lx = canvas_dd[0] * canvas_shape[0]
    Ly = canvas_dd[1] * canvas_shape[1]
    extent = [-Lx/2, Lx/2, -Ly/2, Ly/2]
    canvas = np.zeros(canvas_shape, dtype=np.float64)
    return canvas, extent

def crop_center(result, size=None, center=None):
        cx, cy = center
        dx, dy = size
        result = result[cx-dx:cx+dx, cy-dy:cy+dy].T.copy()

if __name__ == "__main__":
    main()
