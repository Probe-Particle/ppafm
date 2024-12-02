#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
sys.path.append("../../")
from pyProbeParticle import ChargeRings as chr
from pyProbeParticle import GridUtils as GU
from pyProbeParticle import photo

# ========== Setup Parameters

# System parameters from test_ChargeRings_2D.py
V_Bias    = 1.0
Q_tip     = 0.6
cCouling  = 0.02
E_Fermi   = 0.0
z_tip     = 6.0
L         = 20.0
npix      = 400
decay     = 0.2
T         = 10.0

# QD system setup
nsite  = 3
R      = 5.0  # radius of circle on which sites are placed
phiRot = -1.0

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

def load_orbital(fname):
    """Load QD orbital from cube file."""
    try:
        orbital_data, lvec, nDim, _ = GU.loadCUBE(fname)
        return orbital_data, lvec
    except Exception as e:
        print(f"Error loading cube file {fname}: {e}")
        raise

def plotMinMax( data, label=None, figsize=(5,5), cmap='bwr', extent=None, bSave=False ):
    plt.figure(figsize=figsize); 
    vmin=data.min()
    vmax=data.max()
    absmax = max(abs(vmin),abs(vmax))
    plt.imshow(data, origin='lower', aspect='equal', cmap=cmap, vmin=-absmax, vmax=absmax, extent=extent)
    plt.colorbar()
    plt.title(label)
    #plt.xlabel('x [grid points]')
    #plt.show()
    if bSave:
        plt.savefig(label+'.png', bbox_inches='tight')

def crop_center(result, size=None, center=None):
        cx, cy = center
        dx, dy = size
        result = result[cx-dx:cx+dx, cy-dy:cy+dy].T.copy()

def makeCanvas( canvas_shape, canvas_dd, ):
    # Set up canvas parameters
    Lx = canvas_dd[0] * canvas_shape[0]
    Ly = canvas_dd[1] * canvas_shape[1]
    extent = [-Lx/2, Lx/2, -Ly/2, Ly/2]
    canvas = np.zeros(canvas_shape, dtype=np.float64)
    return canvas, extent

def calculate_orbital_stm_single( canvas, orbital_2D, orbital_lvec, tipWf, QDpos, angle, canvas_dd, bTipConv=True):
    """Calculate STM signal using orbital convolution for a single QD."""
    # Place orbital on canvas using GridUtils
    ddi = np.array(photo.evalGridStep2D(orbital_2D.shape, orbital_lvec))
    dd_fac = ddi/np.array(canvas_dd)
    pos = np.array([QDpos[0], QDpos[1]]) / np.array(canvas_dd)
    GU.stampToGrid2D(canvas, orbital_2D, pos, angle, dd=dd_fac, coef=1.0, byCenter=True, bComplex=False)
    #if bDebug: plotMinMax( orbital_2D, label=label+'Molecular Orbital Wf' )
    if bTipConv:
        # Create tip field
        #tipWf, _ = photo.makeTipField(canvas_shape, canvas_dd, z0=z_tip, beta=decay_conv, bSTM=True)
        # Convolve with tip field
        stm_map = photo.convFFT(tipWf, canvas)
        result = np.real(stm_map * np.conj(stm_map))
        #if bDebug: plotMinMax( result, label=label+'STM   Molecular_Wf o tip_Wf' )
        #if bDebug: plotMinMax( result, label=label+'STM cut region' )
        return result
    else:
        return canvas

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
    chr.initRingParams(spos, Esite, rot=rots, MultiPoles=mpols, E_Fermi=E_Fermi, cCouling=cCouling, temperature=T, onSiteCoulomb=3.0)
    
    # Set up configuration basis
    confstrs = ["000","001","010","100","110","101","011","111"]
    confs = chr.confsFromStrings(confstrs)
    nconfs = chr.setSiteConfBasis(confs)
    
    # Load QD orbital
    orbital_data, orbital_lvec = load_orbital("QD.cub")
    
    # Setup canvas
    ncanv = int(np.ceil(Lcanv/dCanv))
    canvas_shape = (ncanv, ncanv)
    canvas_dd = [dCanv, dCanv]
    
    # Define center region
    crop_center = (canvas_shape[0]//2, canvas_shape[1]//2)
    crop_size = (canvas_shape[0]//4, canvas_shape[1]//4)
    
    # Calculate physical dimensions of center region
    center_Lx = 2 * crop_size[0] * canvas_dd[0]
    center_Ly = 2 * crop_size[1] * canvas_dd[1]
    
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

    canvas_sum, extent = makeCanvas( canvas_shape, canvas_dd )
    # Prepare orbital data - integrate over z the upper half of the orbital grid
    orbital_2D = np.sum(  orbital_data[orbital_data.shape[0]//2:,:,:], axis=0)
    orbital_2D = np.ascontiguousarray(orbital_2D.T, dtype=np.float64)

    tipWf, _ = photo.makeTipField(canvas_shape, canvas_dd, z0=z_tip, beta=decay_conv, bSTM=True)

    bTotal   = False
    bTipConv = False

    # Calculate current for each site
    for i in range(nsite):

        canvas = canvas_sum.copy()
        orbital_stm = calculate_orbital_stm_single( canvas, orbital_2D, orbital_lvec, tipWf, spos[i], angles[i], canvas_dd, bTipConv=bTipConv )

        canvas_sum += orbital_stm
            
        if bTotal == True:
            orbital_stm = crop_center(orbital_stm, center_region_size, center_region_center)
            site_current = chr.calculate_site_current( ps, spos[i], Es_1[:,i], E_Fermi + V_Bias, E_Fermi,  decay=decay, T=T, M=1 )
            site_current = site_current.reshape(orbital_stm.shape)   # Reshape site current to match orbital_stm shape
            current_contribution = orbital_stm * site_current      # Multiply orbital convolution with site current coefficients
            total_current += current_contribution
            site_currents.append(current_contribution)

    if bDebug: 
        plotMinMax( canvas_sum, label='Canvas Sum', extent=extent )
        plt.scatter(spos[:,0], spos[:,1],  c='g', marker='o', label='QDs')
        plt.savefig('test_ChargeRings_2D_orbital_canvas_sum_Wfs.png', bbox_inches='tight')

    plt.show(); exit()


    
    # Plot results
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3, figure=fig)
    #extent = [-center_Lx/2, center_Lx/2, -center_Ly/2, center_Ly/2]
    
    # Plot individual site contributions
    for i in range(nsite):
        plt.figure(figsize=(6, 5))
        plt.imshow(site_currents[i], extent=extent, origin='lower', aspect='equal')
        plt.colorbar(label=f'Site {i+1} Current')
        plt.title(f'Site {i+1} Contribution')
        plt.scatter(spos[:,0], spos[:,1], c='g', marker='o', label='QDs')
        plt.legend()
    
    # Plot total current
    plt.figure(figsize=(8, 6))
    plt.imshow(total_current, extent=extent, origin='lower', aspect='equal')
    plt.colorbar(label='Total Current')
    plt.title('Total STM Current')
    plt.scatter(spos[:,0], spos[:,1], c='g', marker='o', label='QDs')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
