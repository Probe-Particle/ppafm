import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../../pyProbeParticle')
import pauli
import utils as ut
import plot_utils as pu

def scan_xV(params, ax1=None, ax2=None, ax3=None, nx=100, nV=100):
    """
    Scan voltage dependence above one particle
    
    Args:
        params: Dictionary of parameters
        ax1,ax2,ax3: Optional matplotlib axes for plotting
        
    Returns:
        Tuple of (V1d, V2d, Vtip, Esites)
    """
    L = params['L']
    z_tip = params['z_tip']
    zT    = z_tip + params['Rtip']
    zV0   = params['zV0']
    zQd   = 0.0
    VBias = params['VBias']
    Rtip  = params['Rtip']
    Esite = params['Esite']
    
    # 1D Potential calculations
    pTips_1d = np.zeros((nx, 3))
    x_coords = np.linspace(-L, L, nx)
    pTips_1d[:,0] = x_coords
    pTips_1d[:,2] = zT
    V1d = pauli.evalSitesTipsMultipoleMirror(pTips_1d, pSites=np.array([[0.0,0.0,zQd]]), VBias=VBias, E0=Esite, Rtip=Rtip, zV0=zV0)[:,0]
    V1d_ = V1d - Esite
    
    # 2D Potential calculations
    V_vals = np.linspace(0.0, VBias, nV)
    X_v, V_v = np.meshgrid(x_coords, V_vals)
    pTips_v = np.zeros((nV*nx, 3))
    pTips_v[:,0] = X_v.flatten()
    pTips_v[:,2] = zT
    V2d = pauli.evalSitesTipsMultipoleMirror(pTips_v, pSites=np.array([[0.0,0.0,zQd]]), VBias=V_v.flatten(), E0=Esite, Rtip=Rtip, zV0=zV0)[:,0].reshape(nV, nx)
    V2d_ = V2d - Esite
    
    # XZ grid calculations
    x_xz = np.linspace(-L, L, nx)
    z_xz = np.linspace(-L, L, nV)
    X_xz, Z_xz = np.meshgrid(x_xz, z_xz)
    ps_xz = np.array([X_xz.flatten(), np.zeros_like(X_xz.flatten()), Z_xz.flatten()]).T
    
    Vtip = pauli.evalSitesTipsMultipoleMirror(ps_xz, pSites=np.array([[0.0, 0.0, zT]]),   VBias=VBias, Rtip=Rtip, zV0=zV0)[:,0].reshape(nV, nx)
    Esites = pauli.evalSitesTipsMultipoleMirror(ps_xz, pSites=np.array([[0.0, 0.0, zQd]]), VBias=VBias, Rtip=Rtip, zV0=zV0)[:,0].reshape(nV, nx)
    
    # Plotting if axes provided
    if ax1 is not None:
        pu.plot_imshow(ax1, V2d, title="Esite(tip_x,tip_V)", extent=[-L, L, 0.0, VBias], ylabel="V [V]", cmap='bwr')
        ax1.plot(x_coords, V1d, label='V_tip')
        ax1.plot(x_coords, V1d_, label='V_tip + E_site')
        ax1.plot(x_coords, x_coords*0.0 + VBias, label='VBias')
        ax1.axhline(0.0, ls='--', c='k')
        #ax1.set_title("1D Potential (z=0)")
        # ax1.set_xlabel("x [Å]")
        # ax1.set_ylabel("V [V]")
        # ax1.grid()
        ax1.set_aspect('auto')
        ax1.legend()
    
    if ax2 is not None:
        extent_xz = [-L, L, -L, L]
        pu.plot_imshow(ax2, Vtip, title="Tip Potential", extent=extent_xz, cmap='bwr', vmin=-VBias, vmax=VBias)
        circ1, _ = ut.makeCircle(16, R=Rtip, axs=(0,2,1), p0=(0.0, 0.0, zT))
        circ2, _ = ut.makeCircle(16, R=Rtip, axs=(0,2,1), p0=(0.0, 0.0, 2*zV0-zT))
        ax2.plot(circ1[:,0], circ1[:,2], ':k')
        ax2.plot(circ2[:,0], circ2[:,2], ':k')
        ax2.axhline(zV0, ls='--', c='k', label='mirror surface')
        ax2.axhline(zQd, ls='--', c='g', label='Qdot height')
        ax2.axhline(z_tip, ls='--', c='orange', label='Tip Height')
    
    if ax3 is not None:
        pu.plot_imshow(ax3, Esites, title="Site Potential", extent=extent_xz, cmap='bwr', vmin=-VBias, vmax=VBias)
        ax3.axhline(zV0, ls='--', c='k', label='mirror surface')
        ax3.axhline(zQd, ls='--', c='g', label='Qdot height')
        ax3.legend()
    
    return V1d, V2d, Vtip, Esites

def scan_xy(params, pauli_solver=None, ax4=None, ax5=None, ax6=None):
    """
    Scan tip position in x,y plane for constant Vbias
    
    Args:
        params: Dictionary of parameters
        pauli_solver: Optional pauli solver instance
        ax4,ax5,ax6: Optional matplotlib axes for plotting
        
    Returns:
        Tuple of (STM, Es, Ts)
    """
    L     = params['L']
    nsite = params['nsite']
    
    # Site positions and rotations
    spos, phis = ut.makeCircle(n=nsite, R=params['radius'], phi0=params['phiRot'])
    spos[:,2]  = params['zQd']
    rots       = ut.makeRotMats(phis + params['phiRot'])
    
    # Run pauli scan
    STM, Es, Ts = pauli.run_pauli_scan_top(spos, rots, params, pauli_solver=pauli_solver)
    #print( "min,max Es", np.min(Es), np.max(Es))
    #print( "min,max Ts", np.min(Ts), np.max(Ts))
    #print( "min,max STM", np.min(STM), np.max(STM))

    Ttot = np.max(Ts, axis=2)
    Etot = np.max(Es, axis=2)
    
    # Plotting if axes provided
    extent = [-L, L, -L, L]
    if ax4 is not None: pu.plot_imshow(ax4, Etot, title="Energies (max)",  extent=extent, cmap='bwr')
    if ax5 is not None: pu.plot_imshow(ax5, Ttot, title="Tunneling (max)", extent=extent, cmap='hot')
    if ax6 is not None: pu.plot_imshow(ax6, STM,  title="STM",             extent=extent, cmap='hot')
    
    return STM, Es, Ts

if __name__ == "__main__":
    # Example usage when run as standalone script - using same defaults as GUI
    params = {
        'VBias': 0.6, 'Rtip': 2.5, 'z_tip': 2.0,
        'W': 0.03, 'GammaS': 0.01, 'GammaT': 0.01, 'Temp': 0.224, 'onSiteCoulomb': 3.0,
        'zV0': -3.3, 'zQd': 0.0,
        'nsite': 3, 'radius': 5.2, 'phiRot': 0.8,
        'Esite': -0.04, 'Q0': 1.0, 'Qzz': 0.0,
        'L': 20.0, 'npix': 100, 'decay': 0.3, 'dQ': 0.02
    }
    verbosity = 0
    
    # Initialize pauli solver
    pauli_solver = pauli.PauliSolver( nSingle=3, nleads=2, verbosity=verbosity )
    
    # Create figure
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(15, 10))
    
    # Run scans with plotting
    scan_xV(params, ax1, ax2, ax3)
    scan_xy(params, pauli_solver, ax4, ax5, ax6)
    
    plt.tight_layout()
    plt.show()
