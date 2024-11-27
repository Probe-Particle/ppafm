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
z_tip = 5.0
#L     = 30.0
decay = 1.7
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
dCanv = 0.5

# ========== Functions

def load_orbital(fname):
    """Load QD orbital from cube file."""
    try:
        orbital_data, lvec, nDim, _ = GU.loadCUBE(fname)
        return orbital_data, lvec
    except Exception as e:
        print(f"Error loading cube file {fname}: {e}")
        raise

# def make_tipWf(sh, dd, z0, decay):
#     """Create exponentially decaying tip wavefunction."""
#     return photo.makeTipField(sh, dd, z0=z0, beta=decay, bSTM=True)


def plotMinMax( data, label=None, figsize=(5,5), cmap='bwr', extent=None ):
    plt.figure(figsize=figsize); 
    vmin=data.min()
    vmax=data.max()
    absmax = max(abs(vmin),abs(vmax))
    plt.imshow(data, origin='lower', aspect='equal', cmap=cmap, vmin=-absmax, vmax=absmax, extent=extent)
    plt.colorbar(label=label)
    #plt.xlabel('x [grid points]')
    #plt.show()

def evalGridStep2D( sh, lvec ):
    dix = lvec[1][0]/sh[0]; #print( "evalGridStep2D dix: ", dix, " lvec: ", lvec[1][0], " sh: ", sh[0] )
    diy = lvec[2][1]/sh[1]; #print( "evalGridStep2D diy: ", diy, " lvec: ", lvec[2][1], " sh: ", sh[1] )
    return (dix,diy)

def photonMap2D_stamp( rhos, lvecs, canvas_shape, dd_canv, angles=[0.0], poss=[ [0.0,0.0] ], coefs=[ [1.0,0.0] ], byCenter=False, bComplex=False ):
    """Generate 2D photon map by convolving transition densities with tip field.
    
    Computes optical coupling between molecular transitions and tip cavity field
    using FFT-based convolution.
    
    Args:
        rhos (list): Transition densities for each molecule
        lvecs (list): Lattice vectors for each molecule
        dd_canv (tuple): Grid spacing for canvas
        angles (list): Rotation angles for each molecule
        poss (list): Positions of molecules
        coefs (list): Coefficients for each molecule's contribution
        byCenter (bool): If True, positions are relative to molecular centers
        bComplex (bool): If True, return complex values
        
    Returns:
        tuple: Total photon map and individual molecular contributions
    """
    dd_canv = np.array( dd_canv )
    if bComplex:
        dtype=np.complex128
    else:
        dtype=np.float64    
    canvas = np.zeros( canvas_shape, dtype=dtype )
    for i in range(len(poss)):
        coef = coefs[i]
        rho  = rhos[i]
        if len(rho.shape)==3:
            rho  = np.sum( rho, axis=2  )   # integrate over z
        rho     = rho.astype( dtype ) 
        ddi     = np.array(  evalGridStep2D( rhos[i].shape, lvecs[i] ) )
        #print(f"ddi: {ddi}" )
        pos     = np.array( poss[i][:2] ) 
        pos    /= dd_canv
        dd_fac  = ddi/dd_canv
        if not isinstance(coef, float):
            coef = complex( coef[0], coef[1] )
        GU.stampToGrid2D( canvas, rho, pos, angles[i], dd=dd_fac, coef=coef, byCenter=byCenter, bComplex=bComplex)
    return canvas

def calculate_orbital_stm(orbital_data, orbital_lvec, QDpos, angles, canvas_shape, canvas_dd, center_region=None):
    """Calculate STM signal using orbital convolution for all QDs."""
    
    # Create tip field
    tipWf, _ = photo.makeTipField( canvas_shape, canvas_dd, z0=z_tip, beta=decay, bSTM=True)

    #if bDebug: plotMinMax( tipWf, label='Tip Wavefunction' )

    orbital_2D = np.sum( orbital_data[orbital_data.shape[0]//2:,:,:] ,axis=0)   # integrate over z the upper half of the orbital grid
    orbital_2D = np.ascontiguousarray( orbital_2D.T, dtype=np.float64)
    print(f"Orbital slice shape: {orbital_2D.shape}")

    #if bDebug: plotMinMax( orbital_2D, label='Molecular Orbital Wf' )

    nqd = len(QDpos)
    Lx = canvas_dd[0] * canvas_shape[0]
    Ly = canvas_dd[1] * canvas_shape[1]
    extent = [ -Lx/2, Lx/2, -Ly/2, Ly/2 ]
    print(f"Canvas shape: {canvas_shape}, physical dimensions: {Lx:.3f} x {Ly:.3f} Angstrom")
    rhos  = [orbital_2D   ]*len(QDpos)
    lvecs = [orbital_lvec ]*len(QDpos)
    #coefs = [ [1.0,0.0] ]*len(QDpos)
    coefs = [ 1.0 ]*len(QDpos)
    canvas = photonMap2D_stamp( rhos, lvecs, canvas_shape, canvas_dd, angles=angles, poss=QDpos, coefs=coefs, byCenter=True, bComplex=False )
    #canvas = photonMap2D_stamp( rhos, lvecs, canvas_shape, canvas_dd, angles=angles, poss=QDpos[:1], coefs=coefs, byCenter=True, bComplex=False )
    #canvas = photonMap2D_stamp( rhos, lvecs, canvas_shape, canvas_dd, angles=angles, poss=[[0.0,0.0,0.0]], coefs=coefs, byCenter=True, bComplex=False )

    if bDebug: 
        plotMinMax( canvas, label='Canvas (Sum Molecular Wfs)', figsize=(8,8), extent=extent  )
        plt.scatter(QDpos[:,0], QDpos[:,1],  c='g', marker='o', label='QDs')

    #plt.show(); exit()

    # Print canvas statistics before convolution
    print(f"Canvas before convolution( Molecular Wfs ): min: {canvas.min():.3e}, max: {canvas.max():.3e}")
    
    # Convolve with tip field
    stm_map = photo.convFFT(tipWf, canvas)
    result = np.real(stm_map * np.conj(stm_map))
    
    print(f"STM map - i.e. Canvas(after convolution): min: {result.min():.3e}, max: {result.max():.3e}")
    
    if bDebug: 
        plotMinMax( result, label='STM Map (Canvas after convolution)', figsize=(8,8), extent=extent )
        plt.scatter(QDpos[:,0], QDpos[:,1],  c='g', marker='o', label='QDs')
    
    # Crop to center region if specified
    if center_region is not None:
        cx, cy = canvas_shape[0]//2, canvas_shape[1]//2
        dx, dy = center_region
        result = result[cx-dx:cx+dx, cy-dy:cy+dy].copy()


    if bDebug: 
        plotMinMax( result, label='STM Map (Canvas after convolution)', figsize=(8,8) )

    plt.show(); exit()
        
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
    ncanv        = int(np.ceil(Lcanv/dCanv))
    canvas_shape = (ncanv, ncanv)  # (ix, iy)
    canvas_dd    = [dCanv, dCanv]  # Grid spacing in x and y
    print(f"Canvas shape: {canvas_shape}")
    print(f"Canvas spacing: dx={canvas_dd[0]:.3f} Å, dy={canvas_dd[1]:.3f} Å")
    
    # Create angles for each QD
    angles = np.array([phi + phiRot for phi in phis])
    
    # Calculate orbital-based STM on full canvas and crop to center
    #orbital_stm = calculate_orbital_stm( orbital_data, orbital_lvec, QDpos, angles + 0.5*np.pi, canvas_shape, canvas_dd, center_region=(canvas_shape[0]//2, canvas_shape[1]//2))
    orbital_stm = calculate_orbital_stm( orbital_data, orbital_lvec, QDpos, angles            , canvas_shape, canvas_dd, center_region=(canvas_shape[0]//4, canvas_shape[1]//4))

    #plt.show(); exit(); return
    
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
    gs  = GridSpec(1, 3, figure=fig)
    
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
    
    # Save data
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
