#!/usr/bin/python

import numpy as np
from TipMultipole import (
    makeCircle, makeRotMats, compute_site_energies,
    compute_site_tunelling, makePosXY, compute_V_mirror, occupancy_FermiDirac
)

def calculate_tip_potential(*, npix=100, L=20.0, z_tip=2.0, Rtip=1.0, VBias=1.0, zV0=-2.5, zQd=0.0, Esite=-0.1, **kwargs):
    """
    Calculate tip potential data for X-Z projections
    
    Args:
        npix: Number of pixels for grid
        L: Size of simulation box
        z_tip: Tip height
        Rtip: Tip radius
        VBias: Bias voltage
        zV0: Mirror surface position
        zQd: Quantum dot height
        Esite: Site energy
            
    Returns:
        dict: Dictionary containing calculated data:
            - Vtip: Tip potential on X-Z grid
            - Esites: Site energies on X-Z grid
            - ps_xz: X-Z positions
            - V1d: 1D potential along x at z=0
            - extent: Plot extent parameters
    """
    # X-Z grid
    ps_xz, Xs, Zs = makePosXY(n=npix, L=L, axs=(0,2,1))
    
    # Tip position
    zT = z_tip + Rtip
    tip_pos = np.array([0.0, 0.0, zT])
    
    # Calculate potentials
    return {
        'Vtip': compute_V_mirror(tip_pos, ps_xz, VBias=VBias, Rtip=Rtip, zV0=zV0).reshape(npix, npix),
        'Esites': compute_site_energies(ps_xz, np.array([[0.0,0.0,zQd]]), VBias=VBias, Rtip=Rtip, zV0=zV0).reshape(npix, npix),
        'ps_xz': ps_xz,
        'extent': [-L, L, -L, L],
        'V1d': compute_V_mirror(tip_pos, np.array([[x, 0, zQd] for x in np.linspace(-L, L, npix)]), VBias=VBias, Rtip=Rtip, zV0=zV0)
    }

def calculate_qdot_system(*, nsite=3, radius=5.0, zQd=0.0, temperature=10.0, onSiteCoulomb=3.0, phiRot=-1.0, npix=100, L=20.0,  z_tip=2.0, Rtip=1.0, VBias=1.0, zV0=-2.5, Esite=-0.1, decay=0.3, **kwargs):
    """
    Calculate quantum dot system properties
    
    Args:
        nsite: Number of sites
        radius: Ring radius
        zQd: Quantum dot height
        temperature: System temperature
        onSiteCoulomb: On-site Coulomb interaction
        phiRot: Rotation angle
        npix: Number of pixels for grid
        L: Size of simulation box
        z_tip: Tip height
        Rtip: Tip radius
        VBias: Bias voltage
        zV0: Mirror surface position
        Esite: Site energy
        decay: Tunneling decay parameter
            
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
    # Setup multipoles and site energies
    Esite_arr = np.full(nsite, Esite)
    
    # Calculate positions and rotations
    spos, phis = makeCircle(n=nsite, R=radius, phi0=phiRot)
    spos[:,2] = zQd  # quantum dots are on the surface
    rots = makeRotMats(phis + phiRot)
    
    # Calculate tip positions
    zT = z_tip + Rtip
    pTips, Xs, Ys = makePosXY(n=npix, L=L, p0=(0,0,zT))
    
    # Calculate energies
    Es = compute_site_energies(pTips, spos, VBias=VBias, Rtip=Rtip, zV0=zV0, E0s=Esite_arr)
    
    # Calculate tunneling
    Ts = compute_site_tunelling(pTips, spos, beta=decay, Amp=1.0)
    
    # Calculate charges and currents
    Qs = np.zeros(Es.shape)
    Is = np.zeros(Es.shape)
    for i in range(nsite):
        Qs[:,i] = occupancy_FermiDirac(Es[:,i], temperature)
        Is[:,i] = Ts[:,i] * (1-Qs[:,i])
    
    return {
        'Es': Es.reshape(npix, npix, -1),
        'total_charge': np.sum(Qs, axis=1).reshape(npix, npix,-1),
        'STM': np.sum(Is, axis=1).reshape(npix, npix,-1),
        'pTips': pTips,
        'extent': [-L, L, -L, L],
        'spos': spos,
        'rots': rots
    }
