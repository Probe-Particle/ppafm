import numpy as np
import matplotlib.pyplot as plt
import os

import sys
sys.path.append('../../')
from pyProbeParticle import pauli
from pyProbeParticle import utils as ut
import plot_utils as pu
import time

import orbital_utils

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

def scan_xy_orb(params, orbital_2D, orbital_lvec, pauli_solver=None, ax_Etot=None, ax_Ttot=None, ax_STM=None, ax_Ms=None, ax_rho=None, ax_dIdV=None, decay=None, bOmp=False):
    """
    Scan tip position in x,y plane for constant Vbias using external hopping Ts
    computed by convolution of orbitals on canvas
    
    Args:
        params: Dictionary of parameters
        orbital_2D: 2D orbital data
        orbital_lvec: Lattice vectors for the orbital
        pauli_solver: Optional pauli solver instance
        ax_Etot: Axis for total energies plot
        ax_Ttot: Axis for total tunneling plot
        ax_STM: Axis for STM current plot
        ax_Ms: Axis for hopping matrix plot
        ax_rho: Axis for density of states plot
        decay: Optional decay parameter for orbital calculations
    """
    #import sys
    #sys.path.insert(0, '../')
    import orbital_utils
    
    # Extract parameters
    #L      = params['L']
    #params['GammaS'] = 1.0
    L = 60.0; params['L'] = L
    nsite  = params['nsite']
    z_tip  = params['z_tip']
    #npix   = params['npix']
    npix = 400; params['npix']=npix
    #decay  = params['decay'] *  #*5.0

    decay = 0.2
    
    # Site positions and rotations
    spos, phis = ut.makeCircle(n=nsite, R=params['radius'], phi0=params['phiRot'])
    spos[:,2]  = params['zQd']
    rots       = ut.makeRotMats(phis + params['phiRot'])
    angles     = phis - np.pi * 0.5 + 0.1 # - 0.05
        
    # Setup canvas for orbital calculations
    dCanv = L/npix
    ncanv = npix  # Same size as output grid
    canvas_shape = (ncanv, ncanv)
    canvas_dd = np.array([dCanv, dCanv])

    T0 = time.perf_counter()    
    # Calculate hopping maps using orbital convolution
    tipWf, shift = orbital_utils.make_tipWf(canvas_shape, canvas_dd, z0=z_tip, decay=decay)
    print( "tipWf shape: ", tipWf.shape )
    Ms, rho = orbital_utils.calculate_Hopping_maps( orbital_2D, orbital_lvec, spos[:,:2], angles, canvas_dd, canvas_shape, tipWf, bTranspose=True )
    T1=time.perf_counter(); print("Time(scan_xy_orb.1 orbital_utils.calculate_Hopping_maps)",  T1-T0 )

    T1 = time.perf_counter()  
    Ts_flat = np.zeros((npix*npix, nsite), dtype=np.float64)
    for i in range(nsite):
        Ts_flat[:,i] = Ms[i].flatten()**2
    Ts_flat[:,:] += 0.0001*0    

    # Create solver if not provided
    if pauli_solver is None:
        pauli_solver = pauli.PauliSolver(nSingle=nsite, nleads=2)

    T2 = time.perf_counter(); print("Time(scan_xy_orb.2 Ts,PauliSolver)",  T2-T1 )     
    bOmp = True
    STM_flat, Es_flat, _ = pauli.run_pauli_scan_top(spos, rots, params, pauli_solver=pauli_solver, Ts=Ts_flat, bOmp=bOmp)
    #STM_flat, Es_flat, _ = pauli.run_pauli_scan_top(spos, rots, params, pauli_solver=pauli_solver )
    if ax_dIdV is not None:
        params_ = params.copy()
        params_['VBias'] += 0.05
        STM_2, Es_flat_, _ = pauli.run_pauli_scan_top(spos, rots, params_, pauli_solver=pauli_solver, Ts=Ts_flat, bOmp=bOmp)
        dIdV = (STM_2 - STM_flat) / 0.01

    T3 = time.perf_counter(); print("Time(scan_xy_orb.3 pauli.run_pauli_scan)",  T3-T2 )
    print("min, max STM_flat", np.min(STM_flat), np.max(STM_flat))
    print("min, max Es_flat", np.min(Es_flat), np.max(Es_flat))
    print("min, max Ts_flat", np.min(Ts_flat), np.max(Ts_flat))
    
    # Reshape the results
    STM = STM_flat.reshape(npix, npix)
    Es  = Es_flat.reshape(npix, npix, nsite)
    Ts  = Ts_flat.reshape(npix, npix, nsite)
    
    # Calculate total Ts (max over sites) and Es (max over sites)
    Ttot = np.sum(Ts, axis=2)
    Etot = np.max(Es, axis=2)
    
    # Plotting if axes provided
    extent = [-L, L, -L, L]

    if ax_Etot is not None: pu.plot_imshow(ax_Etot, Etot, title="Energies max(eps)",      extent=extent, cmap='bwr')
    if ax_rho  is not None: pu.plot_imshow(ax_rho,  rho,  title="sum(Wf)", extent=extent, cmap='bwr')
    if ax_Ttot is not None: pu.plot_imshow(ax_Ttot, Ttot, title="Tunneling sum(M^2)",     extent=extent, cmap='hot')
    if ax_STM  is not None: pu.plot_imshow(ax_STM,  STM,  title="STM",                    extent=extent, cmap='hot')
    if ax_dIdV is not None:
        dIdV = dIdV.reshape(npix, npix)
        pu.plot_imshow(ax_dIdV, dIdV, title="dI/dV", extent=extent, cmap='bwr')
    if ax_Ms   is not None: 
        for i in range(nsite):
            ax_Ms[i].imshow(Ms[i], cmap='bwr', origin='lower', extent=extent)
            ax_Ms[i].set_title(f"Hopping matrix {i}")

    T4 = time.perf_counter(); print("Time(scan_xy_orb.4 plotting)",  T4-T3 )        
    
    return STM, Es, Ts

def run_scan_xy_orb( params, fname="QD.cub" ):
    print("Testing scan_xy_orb with real orbital data...")        
    # Load orbital data from the QD.cub file in the current directory
    orbital_file = fname
    if not os.path.exists(orbital_file):
        print(f"ERROR: Could not find {orbital_file} in the current directory")
        return
        
    print(f"Loading orbital file: {orbital_file}")
    orbital_data, orbital_lvec = orbital_utils.load_orbital(orbital_file)
    
    # Process orbital data similar to ChargeRingsOrbitalGUI.py
    # Transpose and sum over z-direction for 2D representation
    orbital_2D = np.transpose(orbital_data, (2, 1, 0))
    orbital_2D = np.sum(orbital_2D[:, :, orbital_2D.shape[2]//2:], axis=2)
    orbital_2D = np.ascontiguousarray(orbital_2D, dtype=np.float64)
    
    # Initialize pauli solver
    pauli_solver = pauli.PauliSolver(nSingle=params['nsite'], nleads=2, verbosity=0)
    
    # Create figure for visualization
    #fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(20, 10))
    #scan_xy_orb(params, orbital_2D, orbital_lvec, pauli_solver,  ax_Etot=ax1, ax_rho=ax2,  ax_Ttot=ax3, ax_STM=ax4, ax_Ms=[ax5, ax6, ax7], ax_dIdV=ax8 )

    fig, ( ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,5))
    scan_xy_orb(params, orbital_2D, orbital_lvec, pauli_solver,  ax_Etot=ax1, ax_Ttot=ax2, ax_STM=ax3, ax_Ms=None, ax_dIdV=ax4 )
        
    fig.suptitle(f"Scan with Orbital-Based Hopping ({orbital_file})", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
    
    # Save and show the figure
    plt.savefig('scan_xy_orb_test.png')
    print(f"Figure saved as 'scan_xy_orb_test.png'")
    plt.show()

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
        'VBias': 0.3, 'Rtip': 2.5, 'z_tip': 3.0,

        #'W': 0.00,
         #'W': 0.005,
        'W': 0.01,
        #'W': 0.015,
        #'W': 0.02,
        #'W': 0.025,
        #'W': 0.03,
        
        'GammaS': 0.01, 'GammaT': 0.01, 'Temp': 0.224, 'onSiteCoulomb': 3.0,
        'zV0': -10.0, 'zQd': 0.0,
        'nsite': 3, 'radius': 5.2, 'phiRot': 0.8,
        'Esite': -0.04, 'Q0': 1.0, 'Qzz': 0.0,
        'L': 20.0, 'npix': 100, 'decay': 0.1, 'dQ': 0.02
    }
    verbosity = 0
    
    run_scan_xy_orb( params )
    
    exit()

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
