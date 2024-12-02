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
Q_tip     = 0.13
cCouling  = 0.02*0.0
E_Fermi   = 0.0
z_tip     = 5.0
L         = 20.0
npix      = 400
decay     = 0.5
T         = 2.0

# QD system setup
nsite  = 3
R      = 7.0
phiRot = 0.1 + np.pi/2

Q0  = 1.0
Qzz = 15.0 * 0.0

# Canvas parameters from test_LandauerQD_2D_orbital.py
Lcanv = 60.0
dCanv = 0.2
decay_conv = 0.5

# Energy of states on the sites
Esite = [-0.2, -0.2, -0.2]

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
    
    # Create empty canvas and get extent
    canvas_sum, canvas_extent = makeCanvas(canvas_shape, canvas_dd)
    
    # Create tip field (outside loop since it's the same for all sites)
    tipWf, _ = photo.makeTipField(canvas_shape, canvas_dd, z0=z_tip, beta=decay_conv, bSTM=True)
    
    # Prepare orbital data - integrate over z the upper half of the orbital grid
    # First transpose to (ix, iy, iz) for consistent orientation
    orbital_data = np.transpose(orbital_data, (2, 1, 0))
    orbital_2D = np.sum(orbital_data[:, :, orbital_data.shape[2]//2:], axis=2)
    orbital_2D = np.ascontiguousarray(orbital_2D, dtype=np.float64)
    
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
    X, Y = np.meshgrid(x, y, indexing='xy')  # Changed to 'xy' indexing
    
    # Create tip positions array for the center region
    ps = np.zeros((len(x) * len(y), 3))
    ps[:,0] = X.flatten()
    ps[:,1] = Y.flatten()
    ps[:,2] = z_tip
    Qtips = np.ones(len(ps)) * Q_tip
    
    # Calculate site occupancies using ChargeRings
    Q_1, Es_1, Ec_1 = chr.solveSiteOccupancies(ps, Qtips, bEsite=True, solver_type=2)
    
    # Initialize arrays for storing results
    total_current = np.zeros((2*crop_size[1], 2*crop_size[0]))  # Note: shape matches meshgrid 'xy' indexing
    site_coef_maps = []
    
    M_sum  = canvas_sum * 0.0
    M2_sum = canvas_sum * 0.0
    site_current_sum = None
    
    # Calculate current for each site
    for i in range(nsite):
        # Place orbital on canvas
        canvas = photonMap2D_stamp([orbital_2D], [orbital_lvec], canvas_dd, canvas=canvas_sum*0.0, angles=[angles[i]], poss=[[spos[i,0], spos[i,1]]], coefs=[1.0],  byCenter=True, bComplex=False)
        canvas_sum += canvas
        
        # Convolve with tip field   M_i = < psi_i |H| psi_tip  >
        M_i = photo.convFFT(tipWf, canvas)
        M_i = np.real(M_i)
        M_sum  += M_i
        M2_sum += M_i**2
        
        # Crop to center region
        M_i = crop_central_region(M_i, crop_center, crop_size)
        
        # Calculate coefficient of how open is the channel passing through the site due to energy consideration    c = rho_i rho_j [ f_i - f_j ]
        c_i = chr.calculate_site_current(ps, spos[i], Es_1[:,i], E_Fermi + V_Bias, E_Fermi,  decay=decay*0.0, T=T)
        c_i = c_i.reshape((2*crop_size[1], 2*crop_size[0]))  # Reshape site current to match orbital_stm shape (now in 'xy' indexing)
                
        total_current +=  M_i**2  * c_i    # incoherent sum    I = sum c_i * M_i^2
        site_coef_maps.append(c_i)
    
    # Debug: Plot sum of all orbitals before convolution
    if bDebug:
        plt.figure(figsize=(8,8))
        extent = canvas_extent
        # --- canvas_sum
        plotMinMax(canvas_sum, label='Canvas (Sum of Molecular Wfs)', extent=extent)
        plt.scatter(spos[:,0], spos[:,1], c='g', marker='o', label='QDs')
        plt.legend()
        plt.savefig("test_ChargeRings_2D_orbital_canvas_Wf_Sum.png", bbox_inches='tight')
        plt.close()

        # --- STM_sum
        plt.figure(figsize=(8,8))
        plotMinMax(M_sum, label='Sum of M_i = <psi_i|H|psi_tip> ', extent=extent)
        plt.scatter(spos[:,0], spos[:,1], c='g', marker='o', label='QDs')
        plt.legend()
        plt.savefig("test_ChargeRings_2D_orbital_STM_Sum.png", bbox_inches='tight')
        plt.close()

        # --- STM2_sum
        plt.figure(figsize=(8,8))
        plotMinMax(M2_sum, label='Sum of M_i^2 = <psi_i|H|psi_tip>^2 ', extent=extent)
        plt.scatter(spos[:,0], spos[:,1], c='g', marker='o', label='QDs')
        plt.legend()
        plt.savefig("test_ChargeRings_2D_orbital_STM2_Sum.png", bbox_inches='tight')
        plt.close()
    
    # Plot results
    extent = [-center_Lx/2, center_Lx/2, -center_Ly/2, center_Ly/2]
    
    # Plot sum of site_currents 
    site_coef_sum = np.sum(site_coef_maps, axis=0)
    plt.figure(figsize=(6, 5))
    plt.imshow(site_coef_sum, extent=extent, origin='lower', aspect='equal')
    plt.title('Site Coef sum')
    plt.scatter(spos[:,0], spos[:,1], c='g', marker='o', label='QDs')
    plt.legend()
    plt.savefig("test_ChargeRings_2D_orbital_SiteCoefsum.png", bbox_inches='tight')
    plt.close()
    
    # Plot total current
    plt.figure(figsize=(8, 6))
    plt.imshow(total_current, extent=extent, origin='lower', aspect='equal')
    plt.title('Total STM Current')
    plt.scatter(spos[:,0], spos[:,1], c='g', marker='o', label='QDs')
    plt.legend()
    plt.savefig("test_ChargeRings_2D_orbital_final.png", bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
