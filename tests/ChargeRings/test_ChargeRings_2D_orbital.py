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

def makeCanvas( canvas_shape, canvas_dd ):
    """Create an empty canvas with the given shape and grid spacing."""
    canvas = np.zeros(canvas_shape, dtype=np.float64)
    Lx = canvas_dd[0]*canvas_shape[0]
    Ly = canvas_dd[1]*canvas_shape[1]
    extent = [ Lx*-0.5, Lx*0.5, Ly*-0.5, Ly*0.5 ] 
    return canvas, extent

def crop_central_region(result, center, size):
    """Crop the center region of a 2D array."""
    cx, cy = center
    dx, dy = size
    return result[cx-dx:cx+dx, cy-dy:cy+dy]

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
    chr.initRingParams(spos, Esite, rot=rots, MultiPoles=mpols, E_Fermi=E_Fermi,  cCouling=cCouling, temperature=T, onSiteCoulomb=3.0)
    
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
    canvas_dd = np.array([dCanv, dCanv])  # Grid spacing in x and y
    print(f"Canvas shape: {canvas_shape}")
    print(f"Canvas spacing: dx={canvas_dd[0]:.3f} Å, dy={canvas_dd[1]:.3f} Å")

    canvas_sum, canvas_extent = makeCanvas( canvas_shape, canvas_dd )

    # Create tip field (outside loop since it's the same for all sites)
    tipWf, _ = photo.makeTipField(canvas_shape, canvas_dd, z0=z_tip, beta=decay_conv, bSTM=True)
    
    # Prepare orbital data - integrate over z the upper half of the orbital grid
    orbital_2D = np.sum(orbital_data[orbital_data.shape[0]//2:,:,:], axis=0)
    orbital_2D = np.ascontiguousarray(orbital_2D.T, dtype=np.float64)  # Now shape is (ix, iy)
        
    # Define center region
    crop_center = (canvas_shape[0]//2, canvas_shape[1]//2)
    crop_size = (canvas_shape[0]//4, canvas_shape[1]//4)
    
    # Calculate physical dimensions of center region
    center_Lx = 2 * crop_size[0] * canvas_dd[0]
    center_Ly = 2 * crop_size[1] * canvas_dd[1]
    print(f"Center region physical size: {center_Lx:.3f} x {center_Ly:.3f} Å")
    
    # Create grid for the center region
    x = np.linspace(-center_Lx/2, center_Lx/2, 2*crop_size[0])
    y = np.linspace(-center_Ly/2, center_Ly/2, 2*crop_size[1])
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
    total_current = np.zeros((2*crop_size[0], 2*crop_size[1]))
    site_currents = []


    STM_sum = canvas_sum * 0.0
    site_current_sum = None
    
    # Calculate current for each site
    for i in range(nsite):
        canvas = photonMap2D_stamp(  [orbital_2D], [orbital_lvec], canvas_dd, canvas=canvas_sum.copy(),  angles=[angles[i]], poss=[[spos[i,0], spos[i,1]]], coefs=[1.0],  byCenter=True, bComplex=False )    
        canvas_sum += canvas                                                # Add to sum canvas (for debugging)
        stm_map     = photo.convFFT(tipWf, canvas)                              # Convolve with tip field
        orbital_stm = np.real(stm_map * np.conj(stm_map))
        STM_sum     += orbital_stm
        
        orbital_stm = crop_central_region( orbital_stm, crop_center, crop_size )  # Crop to center region

        
        
        site_current = chr.calculate_site_current(  ps, spos[i], Es_1[:,i], E_Fermi + V_Bias, E_Fermi, decay=decay, T=T  )  # Calculate site current coefficients
        site_current = site_current.reshape(orbital_stm.shape)       # Reshape site current to match orbital_stm shape
        current_contribution = orbital_stm * site_current            # Multiply orbital convolution with site current coefficients
        
        # Add to total current and store individual contribution
        total_current += current_contribution
        site_currents.append(current_contribution)
    
    # Debug: Plot sum of all orbitals before convolution
    if bDebug:
        extent = [-Lcanv/2, Lcanv/2, -Lcanv/2, Lcanv/2]
        # --- canvas_sum
        plotMinMax(canvas_sum, label='Canvas (Sum of Molecular Wfs)', figsize=(8,8), extent=extent)
        plt.scatter(spos[:,0], spos[:,1], c='g', marker='o', label='QDs')
        plt.legend()
        plt.savefig(f"test_ChargeRings_2D_orbital_canvas_Wf_Sum.png", bbox_inches='tight')
        # --- STM_sum
        plotMinMax(STM_sum, label='STM Map (After convolution)', figsize=(8,8), extent=extent)
        plt.scatter(spos[:,0], spos[:,1], c='g', marker='o', label='QDs')
        plt.legend()
        plt.savefig(f"test_ChargeRings_2D_orbital_STM_Sum.png", bbox_inches='tight')
        
    
    # Plot results
    extent = [-center_Lx/2, center_Lx/2, -center_Ly/2, center_Ly/2]
    
    # Plot individual site contributions
    for i in range(nsite):
        plt.figure(figsize=(6, 5))
        plt.imshow(site_currents[i], extent=extent, origin='lower', aspect='equal')
        plt.colorbar(label=f'Site {i+1} Current')
        plt.title(f'Site {i+1} Contribution')
        plt.scatter(spos[:,0], spos[:,1], c='g', marker='o', label='QDs')
        plt.savefig(f"test_ChargeRings_2D_orbital_SiteCurrent_{i}.png", bbox_inches='tight')
        plt.legend()
    
    # Plot total current
    plt.figure(figsize=(8, 6))
    plt.imshow(total_current, extent=extent, origin='lower', aspect='equal')
    plt.colorbar(label='Total Current')
    plt.title('Total STM Current')
    plt.scatter(spos[:,0], spos[:,1], c='g', marker='o', label='QDs')
    plt.legend()
    plt.savefig(f"test_ChargeRings_2D_orbital_final.png", bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
