import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../../pyProbeParticle')
import pauli
import utils as ut
import plot_utils as pu

def scan_xV(params, ax_V2d=None, ax_Vtip=None, ax_Esite=None, ax_I2d=None, nx=100, nV=100, bLegend=True):
    """
    Scan voltage dependence above one particle
    
    Args:
        params: Dictionary of parameters
        ax_V2d:   Axis for 2D voltage scan plot (Esite vs x,V)
        ax_Vtip:  Axis for tip potential plot
        ax_Esite: Axis for site potential plot
        nx: Number of x points
        nV: Number of voltage points
    """
    L = params['L']
    z_tip = params['z_tip']
    zT    = z_tip + params['Rtip']
    zV0   = params['zV0']
    zQd   = 0.0
    VBias = params['VBias']
    Rtip  = params['Rtip']
    Esite = params['Esite']

    pTips_1d = np.zeros((nx, 3))
    x_coords = np.linspace(-L, L, nx)
    pTips_1d[:,0] = x_coords
    pTips_1d[:,2] = zT
    V_vals = np.linspace(0.0, VBias, nV)
        
    # XZ grid calculations
    if ax_Vtip is not None or ax_Esite is not None:
        x_xz = np.linspace(-L, L, nx)
        z_xz = np.linspace(-L, L, nV)
        X_xz, Z_xz = np.meshgrid(x_xz, z_xz)
        ps_xz = np.array([X_xz.flatten(), np.zeros_like(X_xz.flatten()), Z_xz.flatten()]).T
    
    # Current calculations if I2d axis provided
    if ax_I2d is not None:
        pSite = np.array([[0.0, 0.0, zQd]])
        current, _, _ = pauli.run_pauli_scan_xV(pTips_1d, V_vals, pSite, params)
        pu.plot_imshow(ax_I2d, current, title="Current", extent=[-L, L, 0.0, VBias], ylabel="V [V]", cmap='hot')
        ax_I2d.set_aspect('auto')
        
    # Plotting if axes provided
    if ax_V2d is not None:
        # 1D Potential calculations
        V1d = pauli.evalSitesTipsMultipoleMirror(pTips_1d, pSites=np.array([[0.0,0.0,zQd]]), VBias=VBias, E0=Esite, Rtip=Rtip, zV0=zV0)[:,0]
        V1d_ = V1d - Esite
        
        # 2D Potential calculations
        X_v, V_v = np.meshgrid(x_coords, V_vals)
        pTips_v = np.zeros((nV*nx, 3))
        pTips_v[:,0] = X_v.flatten()
        pTips_v[:,2] = zT
        V2d = pauli.evalSitesTipsMultipoleMirror(pTips_v, pSites=np.array([[0.0,0.0,zQd]]), VBias=V_v.flatten(), E0=Esite, Rtip=Rtip, zV0=zV0)[:,0].reshape(nV, nx)
        V2d_ = V2d - Esite
        
        pu.plot_imshow(ax_V2d, V2d, title="Esite(tip_x,tip_V)", extent=[-L, L, 0.0, VBias], ylabel="V [V]", cmap='bwr')
        ax_V2d.plot(x_coords, V1d, label='V_tip')
        ax_V2d.plot(x_coords, V1d_, label='V_tip + E_site')
        ax_V2d.plot(x_coords, x_coords*0.0 + VBias, label='VBias')
        ax_V2d.axhline(0.0, ls='--', c='k')
        #ax_V2d.set_title("1D Potential (z=0)")
        # ax_V2d.set_xlabel("x [Å]")
        # ax_V2d.set_ylabel("V [V]")
        # ax_V2d.grid()
        ax_V2d.set_aspect('auto')
        if bLegend:
            ax_V2d.legend()
    else:
        V1d  = None
        V1d_ = None

    
    if ax_Vtip is not None:
        Vtip   = pauli.evalSitesTipsMultipoleMirror(ps_xz, pSites=np.array([[0.0, 0.0, zT]]),   VBias=VBias, Rtip=Rtip, zV0=zV0)[:,0].reshape(nV, nx)
        extent_xz = [-L, L, -L, L]
        pu.plot_imshow(ax_Vtip, Vtip, title="Tip Potential", extent=extent_xz, cmap='bwr', vmin=-VBias, vmax=VBias)
        circ1, _ = ut.makeCircle(16, R=Rtip, axs=(0,2,1), p0=(0.0, 0.0, zT))
        circ2, _ = ut.makeCircle(16, R=Rtip, axs=(0,2,1), p0=(0.0, 0.0, 2*zV0-zT))
        ax_Vtip.plot(circ1[:,0], circ1[:,2], ':k')
        ax_Vtip.plot(circ2[:,0], circ2[:,2], ':k')
        ax_Vtip.axhline(zV0,  ls='--', c='k', label='mirror surface')
        ax_Vtip.axhline(zQd,  ls='--', c='g', label='Qdot height')
        ax_Vtip.axhline(z_tip,ls='--', c='orange', label='Tip Height')
    else:
        Vtip = None
    
    if ax_Esite is not None:
        Esites = pauli.evalSitesTipsMultipoleMirror(ps_xz, pSites=np.array([[0.0, 0.0, zQd]]), VBias=VBias, Rtip=Rtip, zV0=zV0)[:,0].reshape(nV, nx)
        pu.plot_imshow(ax_Esite, Esites, title="Site Potential", extent=extent_xz, cmap='bwr', vmin=-VBias, vmax=VBias)
        ax_Esite.axhline(zV0, ls='--', c='k', label='mirror surface')
        ax_Esite.axhline(zQd, ls='--', c='g', label='Qdot height')
        ax_Esite.legend()
    else:
        Esites = None
    
    return V1d, V2d, Vtip, Esites

def scan_xy(params, pauli_solver=None, ax_Etot=None, ax_Ttot=None, ax_STM=None):
    """
    Scan tip position in x,y plane for constant Vbias
    
    Args:
        params: Dictionary of parameters
        pauli_solver: Optional pauli solver instance
        ax_Etot: Axis for total energies plot
        ax_Ttot: Axis for total tunneling plot
        ax_STM:  Axis for STM current plot
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
    if ax_Etot is not None: pu.plot_imshow(ax_Etot, Etot, title="Energies (max)",  extent=extent, cmap='bwr')
    if ax_Ttot is not None: pu.plot_imshow(ax_Ttot, Ttot, title="Tunneling (max)", extent=extent, cmap='hot')
    if ax_STM  is not None: pu.plot_imshow(ax_STM,  STM,  title="STM",             extent=extent, cmap='hot')
    
    return STM, Es, Ts

def scan_param_sweep(params, scan_params, selected_params=None, nx=100, nV=100, sz=3, bLegend=False):
    """
    Scan over multiple parameters simultaneously and plot V2d and Vtip for each set
    
    Args:
        params: Dictionary of parameters
        scan_params: List of (param_name, values) tuples to sweep
        selected_params: List of other parameters to show in figure title
        nx: Number of x points
        nV: Number of voltage points
        sz: Base size for each subplot
        bLegend: Whether to show legend in plots
    """
    if not scan_params:
        raise ValueError("scan_params cannot be empty")
        
    param_names = [p[0] for p in scan_params]
    values_list = [p[1] for p in scan_params]
    nscan = len(values_list[0])
    
    # Validate all value lists have same length
    for vals in values_list[1:]:
        assert len(vals) == nscan, "All value lists must have same length"
    
    # Create figure with (2, nscan) subplots (columns for each parameter set)
    fig, axes = plt.subplots(2, nscan, figsize=(sz*nscan, sz*2))
    
    # Build figure title with selected parameters (excluding swept ones)
    title = "Parameter sweep\n"
    if selected_params:
        title_params = [p for p in selected_params if p not in param_names]
        if title_params:
            title += "Other params: " + ", ".join([f"{k}={params[k]}" for k in title_params])
    fig.suptitle(title, fontsize=12)
    
    # Make copy of params to avoid modifying original
    params = params.copy()
    
    for i in range(nscan):
        # Update all parameter values
        for param, vals in scan_params:
            params[param] = vals[i]
        
        # Get current column axes
        ax_V2d = axes[0,i] if nscan > 1 else axes[0]
        ax_Vtip = axes[1,i] if nscan > 1 else axes[1]
        
        # Run scan for current parameter values
        scan_xV(params, ax_V2d=ax_V2d, ax_Vtip=ax_Vtip, nx=nx, nV=nV, bLegend=bLegend)
        
        # Build title showing all swept parameters
        title_parts = [f"{name}={params[name]:.3f}" for name in param_names]
        title = ", ".join(title_parts)
        ax_V2d.set_title(title, fontsize=10)
        ax_Vtip.set_title(title, fontsize=10)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Example usage when run as standalone script - using same defaults as GUI
    params = {
        'VBias': 1.0, 'Rtip': 2.5, 'z_tip': 2.0,
        'W': 0.03, 'GammaS': 0.01, 'GammaT': 0.01, 'Temp': 0.224, 'onSiteCoulomb': 3.0,
        'zV0': -3.0, 'zQd': 0.0,
        'nsite': 3, 'radius': 5.2, 'phiRot': 0.8,
        'Esite': -0.04, 'Q0': 1.0, 'Qzz': 0.0,
        'L': 20.0, 'npix': 100, 'decay': 0.1, 'dQ': 0.02
    }
    verbosity = 0

    scan_param_sweep(params, [('z_tip', np.linspace(1.0, 6.0, 5)), ('Esite', np.linspace(-0.20, -0.05, 5))], selected_params=['VBias','zV0'])
    plt.savefig('scan_param_sweep.png')
    plt.show()
    exit()
    
    # Initialize pauli solver
    pauli_solver = pauli.PauliSolver( nSingle=3, nleads=2, verbosity=verbosity )
    
    # Create figure
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(15, 10))
    
    # Run scans with plotting
    #scan_xV(params, ax_V2d=ax1, ax_Vtip=ax2, ax_Esite=ax3)
    scan_xV(params, ax_Vtip=ax1, ax_V2d=ax2, ax_I2d=ax3)
    scan_xy(params, pauli_solver, ax_Etot=ax4, ax_Ttot=ax5, ax_STM=ax6)
    
    plt.tight_layout()
    plt.show()

    # Example parameter sweep
    z_tip_vals = np.linspace(1.0, 3.0, 5)  # Scan 5 tip heights
    fig = scan_param_sweep(params, [('z_tip', z_tip_vals)])
    plt.show()
