#!/usr/bin/python

import numpy as np
from TipMultipole import (
    makeCircle, makeRotMats, compute_site_energies,
    compute_site_tunelling, makePosXY, compute_V_mirror, occupancy_FermiDirac
)

def calculate_tip_potential(params):
    """
    Calculate tip potential data for X-Z projections
    
    Args:
        params (dict): Dictionary containing simulation parameters
            Required keys:
            - npix: Number of pixels for grid
            - L: Size of simulation box
            - z_tip: Tip height
            - Rtip: Tip radius
            - VBias: Bias voltage
            - zV0: Mirror surface position
            - zQd: Quantum dot height
            - Esite: Site energy
            
    Returns:
        dict: Dictionary containing calculated data:
            - Vtip: Tip potential on X-Z grid
            - Esites: Site energies on X-Z grid
            - ps_xz: X-Z positions
            - V1d: 1D potential along x at z=0
            - extent: Plot extent parameters
    """
    # X-Z grid
    ps_xz, Xs, Zs = makePosXY(n=params['npix'], L=params['L'], axs=(0,2,1))
    
    # Tip position
    zT = params['z_tip'] + params['Rtip']
    tip_pos = np.array([0.0, 0.0, zT])
    
    # Calculate potentials
    return {
        'Vtip': compute_V_mirror( tip_pos, ps_xz, VBias=params['VBias'],   Rtip=params['Rtip'], zV0=params['zV0'] ).reshape(params['npix'], params['npix']),
        'Esites': compute_site_energies( ps_xz, np.array([[0.0,0.0,params['zQd']]]), params['VBias'], params['Rtip'], zV0=params['zV0'] ).reshape(params['npix'], params['npix']),
        'ps_xz': ps_xz,
        'extent': [-params['L'], params['L'], -params['L'], params['L']],
        'V1d': compute_V_mirror( tip_pos,  np.array([[x, 0, params['zQd']] for x in np.linspace(-params['L'], params['L'], params['npix'])]), VBias=params['VBias'],  Rtip=params['Rtip'],  zV0=params['zV0'] )
    }

def calculate_qdot_system(params):
    """
    Calculate quantum dot system properties
    
    Args:
        params (dict): Dictionary containing simulation parameters
            Required keys:
            - nsite: Number of sites
            - radius: Ring radius
            - zQd: Quantum dot height
            - temperature: System temperature
            - onSiteCoulomb: On-site Coulomb interaction
            - phiRot: Rotation angle
            - npix: Number of pixels for grid
            - L: Size of simulation box
            - z_tip: Tip height
            - Rtip: Tip radius
            - VBias: Bias voltage
            - zV0: Mirror surface position
            - Esite: Site energy
            - decay: Tunneling decay parameter
            
    Returns:
        dict: Dictionary containing calculated data:
            - Es: Site energies
            - total_charge: Total charge distribution
            - STM: STM signal
            - pTips: Tip positions
            - extent: Plot extent parameters
            - spos: Site positions
            - rots: Rotation matrices
    """
    nsite = params['nsite']
    R = params['radius']
    
    # Setup multipoles and site energies
    Esite = np.full(nsite, params['Esite'])
    
    # Calculate positions and rotations
    spos, phis = makeCircle(n=nsite, R=R)
    spos[:,2] = params['zQd']  # quantum dots are on the surface
    rots = makeRotMats(phis + params['phiRot'])
    
    # Calculate tip positions
    zT = params['z_tip'] + params['Rtip']
    pTips, Xs, Ys = makePosXY(n=params['npix'], L=params['L'], p0=(0,0,zT))
    
    # Calculate energies
    Es = compute_site_energies(
        pTips, 
        spos,
        VBias=params['VBias'],
        Rtip=params['Rtip'],
        zV0=params['zV0'],
        E0s=Esite
    )
    
    # Calculate tunneling
    Ts = compute_site_tunelling(pTips, spos, beta=params['decay'], Amp=1.0)
    
    # Calculate charges and currents
    Qs = np.zeros(Es.shape)
    Is = np.zeros(Es.shape)
    for i in range(params['nsite']):
        Qs[:,i] = occupancy_FermiDirac(Es[:,i], params['temperature'])
        Is[:,i] = Ts[:,i] * (1-Qs[:,i])
    
    return {
        'Es': Es.reshape(params['npix'], params['npix'], -1),
        'total_charge': np.sum(Qs, axis=1).reshape(params['npix'], params['npix'],-1),
        'STM': np.sum(Is, axis=1).reshape(params['npix'], params['npix'],-1),
        'pTips': pTips,
        'extent': [-params['L'], params['L'], -params['L'], params['L']],
        'spos': spos,
        'rots': rots
    }
