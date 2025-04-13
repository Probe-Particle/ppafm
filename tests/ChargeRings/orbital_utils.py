#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from pyProbeParticle import GridUtils as GU
from pyProbeParticle import photo

def load_orbital(fname):
    """Load QD orbital from cube file."""
    try:
        from pyProbeParticle import GridUtils as GU
        orbital_data, lvec, nDim, _ = GU.loadCUBE(fname)
        return orbital_data, lvec
    except Exception as e:
        print(f"Error loading cube file {fname}: {e}")
        raise

def plotMinMax(data, label=None, figsize=(5,5), cmap='bwr', extent=None, bSave=False):
    """Plot data with symmetric colormap centered at zero."""
    plt.figure(figsize=figsize)
    vmax = np.max(np.abs(data))
    plt.imshow(data, origin='lower', cmap=cmap, vmin=-vmax, vmax=vmax, extent=extent)
    plt.colorbar()
    if label is not None:
        plt.title(label)
    if bSave and label is not None:
        plt.savefig(f"{label.replace(' ', '_')}.png", bbox_inches='tight')

def evalGridStep2D(sh, lvec):
    """Calculate grid step size from shape and lattice vectors."""
    dd = np.zeros(2)
    dd[0] = abs(lvec[1,0])/sh[0]  # x dimension
    dd[1] = abs(lvec[2,1])/sh[1]  # y dimension
    return dd

def photonMap2D_stamp( orb2D, lvecs, dd_canv, canvas_shape=None, angles=[0.0], poss=[[0.0,0.0]], coefs=[1.0], byCenter=True, bComplex=False, canvas=None):
    """Generate 2D photon map by placing transition densities on canvas.
    
    Args:
        rhos (list): List of 2D density arrays (ix,iy)
        lvecs (list): List of lattice vectors for each density
        canvas_shape (tuple): Shape of the canvas (ix,iy)
        dd_canv (list): Grid spacing of canvas [dx,dy]
        angles (list): Rotation angles for each density
        poss (list): Positions of densities [[x,y],...]
        coefs (list): Coefficients for each density
        byCenter (bool): If True, positions are relative to density centers
        bComplex (bool): If True, return complex values
    
    Returns:
        numpy.ndarray: Canvas with placed densities
    """
    if canvas is None:
        canvas = np.zeros(canvas_shape, dtype=np.complex128 if bComplex else np.float64)
    for orb2D, lvec, pos, angle, coef in zip(orb2D, lvecs, poss, angles, coefs):
        ddi = evalGridStep2D(orb2D.shape, lvec)
        dd_fac = ddi/np.array(dd_canv)                                                                                # Calculate grid spacing ratio
        pos_grid = np.array([pos[0], pos[1]]) / np.array(dd_canv)                                                     # Convert position to grid coordinates
        GU.stampToGrid2D(canvas, orb2D, pos_grid, angle, dd=dd_fac, coef=coef, byCenter=byCenter, bComplex=bComplex)  # Place density on canvas
    return canvas

def make_tipWf(sh, dd, z0, decay):
    """Create exponentially decaying tip wavefunction."""
    return photo.makeTipField(sh, dd, z0=z0, beta=decay, bSTM=True)

def crop_central_region(result, center, size):
    """Crop the center region of a 2D array.
    
    Args:
        result (ndarray): Input 2D array
        center (tuple): Center point (cx, cy)
        size (tuple): Size of crop region (dx, dy)
    
    Returns:
        ndarray: Cropped array
    """
    cx, cy = center
    dx, dy = size
    return result[cx-dx:cx+dx, cy-dy:cy+dy]

def calculate_simple( spos, ps, Es_1, E_Fermi, V_Bias, decay, T, crop_center=None, crop_size=None): 
    '''
    Calculate coefficient c_i = rho_i rho_j [ f_i - f_j ] with exponential decay as M_i^2
    '''
    Itot = np.zeros((2*crop_size[1], 2*crop_size[0]))
    for i in range(len(spos)):
        c_i = cr.calculate_site_current(ps, spos[i], Es_1[:,i], E_Fermi, E_Fermi, decay=decay, T=T)
        c_i = c_i.reshape((2*crop_size[1], 2*crop_size[0]))
        Itot += c_i  # Direct sum without M_i^2 factor

def calculate_Hopping_maps(orbital_2D, orbital_lvec, spos, angles, canvas_dd, canvas_shape, tipWf, crop_center=None, crop_size=None, bTranspose=False ):
    from pyProbeParticle import photo
    # Initialize arrays
    canvas_sum = np.zeros(canvas_shape, dtype=np.float64)
    Ms = []
    nsite = len(spos)
    for i in range(nsite):
        # Place orbital on canvas
        canvas = photonMap2D_stamp([orbital_2D], [orbital_lvec], canvas_dd, canvas=canvas_sum*0.0, angles=[angles[i]], poss=[[spos[i,0], spos[i,1]]], coefs=[1.0], byCenter=True, bComplex=False)
        #canvas_sum += canvas
        # Convolve with tip field   M_i = < psi_i |H| psi_tip  >
        M_i = photo.convFFT(tipWf, canvas, bNormalize=True)
        M_i = np.real(M_i)      
        # Crop to center region if needed
        if crop_center is not None and crop_size is not None:
            M_i = crop_central_region(M_i, crop_center, crop_size)    
        Ms.append(M_i)
    # tranform to shape (nsite,nx,ny)
    if bTranspose:
        Ms_ = np.empty( (nsite,)+canvas_shape, dtype=np.float64 )
        for i in range(nsite):
            Ms_[i,:,:] = Ms[i][:,:]
        Ms = Ms_
    return Ms

def calculate_stm_maps(orbital_2D, orbital_lvec, spos, angles, canvas_dd, canvas_shape, tipWf, ps, Es_1, E_Fermi, V_Bias, decay, T, crop_center=None, crop_size=None, cis=None ):
    """Calculate STM maps for a given orbital configuration using Fermi Golden Rule.
    I(E,x) = M^2 * rho_i rho_j [ f_i - f_j ] = M(x)^2 * c(E)
    where:
       * M(x) = <psi_i(x)|H|psi_tip(x) >                     is position dependent tunelling matrix element
       * c(E) = 1/2 * rho_i(E) rho_j(E) ( f_i(E) - f_j(E) )  is energy dependent coefficient given by the Fermi occupation function f(E) and the density of states rho(E)
       NOTE: the density of states rho may be itself position dependent as the orbital energy of quantum dots depend on position of the tip which shifts it by electrostatic potential 

    
    Args:
        orbital_2D (ndarray): 2D orbital data
        orbital_lvec (ndarray): Lattice vectors
        spos (ndarray): Site positions
        angles (ndarray): Site angles
        canvas_dd (ndarray): Canvas grid spacing
        canvas_shape (tuple): Canvas shape
        tipWf (ndarray): Tip wavefunction
        ps (ndarray): Tip positions
        Es_1 (ndarray): Site energies
        E_Fermi (float): Fermi energy
        V_Bias (float): Bias voltage
        decay (float): Decay parameter
        T (float): Temperature
        crop_center (tuple): Center point for cropping
        crop_size (tuple): Size of cropped region
    
    Returns:
        tuple: (total_current, M_sum, M2_sum, site_coef_maps)
    """
    from pyProbeParticle import ChargeRings as chr
    from pyProbeParticle import photo
    
    # Initialize arrays
    canvas_sum = np.zeros(canvas_shape, dtype=np.float64)
    M_sum = np.zeros(canvas_shape, dtype=np.float64)
    M2_sum = np.zeros(canvas_shape, dtype=np.float64)
    site_coef_maps = []
    
    if crop_size is not None:
        total_current = np.zeros((2*crop_size[1], 2*crop_size[0]))
    else:
        total_current = np.zeros(canvas_shape)
    
    # Calculate current for each site
    for i in range(len(spos)):
        # Place orbital on canvas
        canvas = photonMap2D_stamp([orbital_2D], [orbital_lvec], canvas_dd, canvas=canvas_sum*0.0,
                                 angles=[angles[i]], poss=[[spos[i,0], spos[i,1]]], coefs=[1.0],
                                 byCenter=True, bComplex=False)
        canvas_sum += canvas
        
        # Convolve with tip field   M_i = < psi_i |H| psi_tip  >
        M_i = photo.convFFT(tipWf, canvas, bNormalize=True)
        M_i = np.real(M_i)
        M_sum += M_i
        M2_sum += M_i**2
        
        # Crop to center region if needed
        if crop_center is not None and crop_size is not None:
            M_i = crop_central_region(M_i, crop_center, crop_size)
        
        if cis is not None:
            # Calculate coefficient c_i = rho_i rho_j [ f_i - f_j ]
            c_i = chr.calculate_site_current(ps, spos[i], Es_1[:,i], E_Fermi + V_Bias, E_Fermi, decay=decay*0.0, T=T)
            
            if crop_size is not None:
                c_i = c_i.reshape((2*crop_size[1], 2*crop_size[0]))
            else:
                c_i = c_i.reshape(canvas_shape)
        else:
            c_i = cis[i] 

        # Incoherent sum    I = sum c_i * M_i^2
        total_current += M_i**2 * c_i   # Fermi Golden Rule   I = sum_i{ c_i * M_i^2 }
        site_coef_maps.append(c_i)
    
    return total_current, M_sum, M2_sum, site_coef_maps

def calculate_didv(orbital_2D, orbital_lvec, spos, angles, canvas_dd, canvas_shape, tipWf, ps,  Q_tip, dQ, E_Fermi, V_Bias, decay, T, crop_center=None, crop_size=None):
    """Calculate dI/dV using numerical differentiation.
    
    Args:
        orbital_2D (ndarray): 2D orbital data
        orbital_lvec (ndarray): Lattice vectors
        spos (ndarray): Site positions
        angles (ndarray): Site angles
        canvas_dd (ndarray): Canvas grid spacing
        canvas_shape (tuple): Canvas shape
        tipWf (ndarray): Tip wavefunction
        ps (ndarray): Tip positions
        Q_tip (float): Tip charge
        dQ (float): Charge difference for numerical differentiation
        E_Fermi (float): Fermi energy
        V_Bias (float): Bias voltage
        decay (float): Decay parameter
        T (float): Temperature
        crop_center (tuple): Center point for cropping
        crop_size (tuple): Size of cropped region
    
    Returns:
        tuple: (dIdQ, I_1, I_2)
    """
    from pyProbeParticle import ChargeRings as chr
    
    # Calculate for Q_tip
    Qtips = np.ones(len(ps)) * Q_tip
    Q_1, Es_1, _ = chr.solveSiteOccupancies(ps, Qtips, bEsite=True, solver_type=2)
    I_1, _, _, _ = calculate_stm_maps(orbital_2D, orbital_lvec, spos, angles, canvas_dd, canvas_shape,tipWf, ps, Es_1, E_Fermi, V_Bias, decay, T, crop_center, crop_size)
    
    # Calculate for Q_tip + dQ
    Q_2, Es_2, _ = chr.solveSiteOccupancies(ps, Qtips + dQ, bEsite=True, solver_type=2)
    I_2, _, _, _ = calculate_stm_maps(orbital_2D, orbital_lvec, spos, angles, canvas_dd, canvas_shape, tipWf, ps, Es_2, E_Fermi, V_Bias, decay, T, crop_center, crop_size)
    
    # Calculate dI/dQ
    dIdQ = (I_2 - I_1) / dQ
    
    return dIdQ, I_1, I_2
