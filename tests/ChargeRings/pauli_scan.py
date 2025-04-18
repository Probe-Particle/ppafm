import numpy as np
import matplotlib.pyplot as plt
import os

import sys
sys.path.append('../../')
from pyProbeParticle import pauli
from pyProbeParticle import utils as ut
import plot_utils as pu
import numpy as np
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
        ax_V2d.plot(x_coords, V1d,  label='V_tip')
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

def scan_xy(params, pauli_solver=None, ax_Etot=None, ax_Ttot=None, ax_STM=None, ax_dIdV=None, bOmp=False):
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
    STM, Es, Ts = pauli.run_pauli_scan_top(spos, rots, params, pauli_solver=pauli_solver, bOmp=bOmp)
    #print( "min,max Es", np.min(Es), np.max(Es))
    #print( "min,max Ts", np.min(Ts), np.max(Ts))
    #print( "min,max STM", np.min(STM), np.max(STM))
    if ax_dIdV is not None:
        params_ = params.copy()
        dQ = params.get('dQ', 0.05)
        params_['VBias'] += dQ
        STM_2, _, _ = pauli.run_pauli_scan_top(spos, rots, params_, pauli_solver=pauli_solver, bOmp=bOmp)       
        dIdV = (STM_2 - STM) / dQ

    Ttot = np.max(Ts, axis=2)
    Etot = np.max(Es, axis=2)
    
    # Plotting if axes provided
    extent = [-L, L, -L, L]
    if ax_Etot is not None: pu.plot_imshow(ax_Etot, Etot, title="Energies  (max)",  extent=extent, cmap='bwr')
    if ax_Ttot is not None: pu.plot_imshow(ax_Ttot, Ttot, title="Tunneling (max)",  extent=extent, cmap='hot')
    if ax_STM  is not None: pu.plot_imshow(ax_STM,  STM,  title="STM",              extent=extent, cmap='hot')
    if ax_dIdV is not None: pu.plot_imshow(ax_dIdV, dIdV, title="dI/dV",            extent=extent, cmap='bwr')
    
    return STM, Es, Ts

# Helpers: crop central region and index/coordinate transforms
def cut_central_region(map_list, dcanv, big_npix, small_npix):
    """Crop central small_npix×small_npix from big_npix maps with pixel size dcanv"""
    # pixel size shared, so just index crop
    center=big_npix//2; half=small_npix//2
    start=center-half; end=start+small_npix
    return [m[start:end, start:end] for m in map_list]

def pixel_to_coord(i,j,dcanv,npix): 
    """Map pixel idx to coord centered at 0"""
    return ((i+0.5-npix/2)*dcanv, (j+0.5-npix/2)*dcanv)

def coord_to_pixel(x,y,dcanv,npix): 
    """Map coord to nearest pixel idx"""
    return (int(x/dcanv + npix/2), int(y/dcanv + npix/2))

def generate_central_hops(orb2D, orb_lvec, spos_xy, angles, z0, dcanv, big_npix, small_npix, decay=0.2):
    """Compute big canvas hops and crop central small canvas"""
    big_dd=[dcanv,dcanv]; big_shape=(big_npix,big_npix)
    tipWf,shift=orbital_utils.make_tipWf(big_shape,big_dd,z0=z0,decay=decay)
    Ms_big,rho_big=orbital_utils.calculate_Hopping_maps(orb2D,orb_lvec,spos_xy,angles,big_dd,big_shape,tipWf,bTranspose=True)
    Ms_small=cut_central_region(Ms_big,dcanv,big_npix,small_npix)
    rho_small=cut_central_region([rho_big],dcanv,big_npix,small_npix)[0]
    return Ms_small, rho_small

def scan_xy_orb(params, orbital_2D=None, orbital_lvec=None, pauli_solver=None, ax_Etot=None, ax_Ttot=None, ax_STM=None, ax_Ms=None, ax_rho=None, ax_dIdV=None, decay=None, bOmp=False, Tmin=0.0, EW=2.0):
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
        ax_dIdV: Axis for dI/dV plot
        decay: Optional decay parameter for orbital calculations
    """
    #import sys
    #sys.path.insert(0, '../')
    import orbital_utils
    
    # small canvas & site setup
    L=params['L']; npix=params['npix']
    nsite=params['nsite']; z_tip=params['z_tip']
    spos,phis=ut.makeCircle(n=nsite,R=params['radius'],phi0=params['phiRot'])
    angles=phis-np.pi*0.5+0.1
    # big-to-small hopping computation
    #dcanv=params['L']/params['npix']

    T1=time.perf_counter()
    if orbital_2D is not None:
        dcanv=2*L/npix
        big_npix=int(params.get('big_npix',400))
        Ms,rho=generate_central_hops(orbital_2D,orbital_lvec,spos[:,:2],angles,z_tip,dcanv,big_npix,npix,decay=decay or params.get('decay',0.2))
        Ts_flat = np.zeros((npix*npix, nsite), dtype=np.float64)
        for i in range(nsite): Ts_flat[:,i]=Ms[i].flatten()**2
    else:
        Ts_flat = None

    # Create solver if not provided
    if pauli_solver is None:
        pauli_solver = pauli.PauliSolver(nSingle=nsite, nleads=2)
    pauli.set_valid_point_cuts(Tmin, EW)
    
    T2 = time.perf_counter(); print("Time(scan_xy_orb.2 Ts,PauliSolver)",  T2-T1 )     
    #bOmp = True
    STM_flat, Es_flat, Ts_flat_ = pauli.run_pauli_scan_top(spos, ut.makeRotMats(phis + params['phiRot']), params, pauli_solver=pauli_solver, Ts=Ts_flat, bOmp=bOmp)
    #STM_flat, Es_flat, _ = pauli.run_pauli_scan_top(spos, rots, params, pauli_solver=pauli_solver )
    if ax_dIdV is not None:
        params_ = params.copy()
        dQ = params.get('dQ', 0.005)
        params_['VBias'] += dQ
        STM_2, _, _ = pauli.run_pauli_scan_top(spos, ut.makeRotMats(phis + params['phiRot']), params_, pauli_solver=pauli_solver, Ts=Ts_flat, bOmp=bOmp)
        dIdV = (STM_2 - STM_flat) / dQ

    T3 = time.perf_counter(); print("Time(scan_xy_orb.3 pauli.run_pauli_scan)",  T3-T2 )
    print("min, max STM_flat", np.min(STM_flat), np.max(STM_flat))
    print("min, max Es_flat", np.min(Es_flat), np.max(Es_flat))
    print("min, max Ts_flat", np.min(Ts_flat), np.max(Ts_flat))
    
    # Reshape the results
    STM = STM_flat.reshape(npix, npix)
    Es  = Es_flat .reshape(npix, npix, nsite)
    Ts  = Ts_flat_.reshape(npix, npix, nsite)
    
    # Calculate total Ts (max over sites) and Es (max over sites)
    Ttot = np.sum(Ts, axis=2)
    Etot = np.max(Es, axis=2)
    
    # Plotting if axes provided
    extent = [-L,L, -L, L]

    if ax_Etot is not None: pu.plot_imshow(ax_Etot, Etot, title="Energies max(eps)",      extent=extent, cmap='bwr')
    if ax_rho  is not None and rho is not None: pu.plot_imshow(ax_rho,  rho,  title="sum(Wf)", extent=extent, cmap='bwr')
    if ax_Ttot is not None: pu.plot_imshow(ax_Ttot, Ttot, title="Tunneling sum(M^2)",     extent=extent, cmap='hot')
    if ax_STM  is not None: pu.plot_imshow(ax_STM,  STM,  title="STM",                    extent=extent, cmap='hot')
    if ax_dIdV is not None:
        dIdV = dIdV.reshape(npix, npix)
        pu.plot_imshow(ax_dIdV, dIdV, title="dI/dV", extent=extent, cmap='bwr')
    if ax_Ms   is not None and Ms is not None: 
        for i in range(nsite):
            ax_Ms[i].imshow(Ms[i], cmap='bwr', origin='lower', extent=extent)
            ax_Ms[i].set_title(f"Hopping matrix {i}")

    T4 = time.perf_counter(); print("Time(scan_xy_orb.4 plotting)",  T4-T3 )        
    
    return STM, Es, Ts

def run_scan_xy_orb( params, orbital_file="QD.cub" ):
    print("Testing scan_xy_orb with real orbital data...")        
    # Load orbital data from the QD.cub file in the current directory
    if orbital_file is not None:
        if not os.path.exists(orbital_file):
            print(f"ERROR: Could not find {orbital_file} in the current directory")
            return
        print(f"Loading orbital file: {orbital_file}")
        orbital_data, orbital_lvec = orbital_utils.load_orbital(orbital_file)
        orbital_2D = np.transpose(orbital_data, (2, 1, 0))
        orbital_2D = np.sum(orbital_2D[:, :, orbital_2D.shape[2]//2:], axis=2)
        orbital_2D = np.ascontiguousarray(orbital_2D, dtype=np.float64)
    else:
        orbital_2D = None
        orbital_lvec = None

    # Initialize pauli solver
    pauli_solver = pauli.PauliSolver(nSingle=params['nsite'], nleads=2, verbosity=0)

    fig, ( ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,5))
    scan_xy_orb(params, orbital_2D=orbital_2D, orbital_lvec=orbital_lvec, pauli_solver=pauli_solver, ax_Etot=ax1, ax_Ttot=ax2, ax_STM=ax3, ax_Ms=None, ax_dIdV=ax4 )
        
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

def scan_param_sweep_xy_orb(params, scan_params, selected_params=None, orbital_2D=None, orbital_lvec=None, orbital_file=None, pauli_solver=None, bOmp=False, Tmin=0.0, EW=2.0, fig=None):
    """
    Scan parameter values while keeping tip position fixed, precomputing hopping maps once.
    Generates nscan+1 columns of plots with specified layout.
    
    Args:
        params: Dictionary of parameters
        scan_params: List of (param_name, values) tuples to sweep
        selected_params: List of other parameters to show in figure title
        orbital_2D: 2D orbital data
        orbital_lvec: Lattice vectors for the orbital
        pauli_solver: Optional pauli solver instance
        decay: Optional decay parameter for orbital calculations
        fig: Optional figure object to plot into
    """
    
    bOrbGiven = (orbital_2D is not None) or (orbital_lvec is not None)
    bDoOrb    = ( orbital_file is not None ) or bOrbGiven
    if bDoOrb:
        if not bOrbGiven:
            print(f"Loading orbital file: {orbital_file}")
            orbital_data, orbital_lvec = orbital_utils.load_orbital(orbital_file)
        # Process orbital data similar to ChargeRingsOrbitalGUI.py
        # Transpose and sum over z-direction for 2D representation
        orbital_2D = np.transpose(orbital_data, (2, 1, 0))
        orbital_2D = np.sum(orbital_2D[:, :, orbital_2D.shape[2]//2:], axis=2)
        orbital_2D = np.ascontiguousarray(orbital_2D, dtype=np.float64)
        # Precompute hopping maps once
        dcanv = 2*L/npix; big_npix = int(params.get('big_npix',400))
        Ms, rho = generate_central_hops(orbital_2D, orbital_lvec, params['spos'][:,:2], params['angles'], params['z_tip'], dcanv, big_npix, npix, decay)
        Ts_flat = np.zeros((npix*npix, nsite), dtype=np.float64)
        for i in range(nsite): Ts_flat[:,i] = Ms[i].flatten()**2
    else:
        Ts_flat = None

    if not scan_params:
        raise ValueError("scan_params cannot be empty")
        
    param_names = [p[0] for p in scan_params]
    values_list = [p[1] for p in scan_params]
    nscan = len(values_list[0])
    
    # Validate all value lists have same length
    for vals in values_list[1:]:
        assert len(vals) == nscan, "All value lists must have same length"
    
    # Extract parameters
    L     = params['L']
    nsite = params['nsite']
    z_tip = params['z_tip']
    npix  = params['npix']
    decay = params['decay']
    

    # Site positions and rotations
    spos, phis = ut.makeCircle(n=nsite, R=params['radius'], phi0=params['phiRot'])
    spos[:,2]  = params['zQd']
    angles     = phis  + params['phiRot']
    rots       = ut.makeRotMats( angles )
    
    
    # Store in params for later use
    params['spos'] = spos
    params['rots'] = rots
    params['angles'] = angles
    
    # Setup canvas for orbital calculations
    #dCanv = L/npix
    #ncanv = npix
    #canvas_shape = (ncanv, ncanv)
    #canvas_dd = np.array([dCanv, dCanv])
    extent = [-L, L, -L, L]
    


    # Create solver if not provided
    if pauli_solver is None:
        pauli_solver = pauli.PauliSolver(nSingle=nsite, nleads=2)
    pauli.set_valid_point_cuts(Tmin, EW)
    
    # Create figure if not provided
    if fig is None:
        fig = plt.figure(figsize=(5*(nscan+1), 5*3))
    
    # Build figure title with selected parameters (excluding swept ones)
    title = "Parameter sweep: "
    if selected_params:
        title_params = [p for p in selected_params if p not in param_names]
        if title_params:
            title += ", ".join([f"{k}={params[k]}" for k in title_params])
    fig.suptitle(title, fontsize=12)
    
    # First column: rho (top), Ttot (middle)
    ax_rho  = fig.add_subplot(3, nscan+1, 1)
    ax_Ttot = fig.add_subplot(3, nscan+1, nscan+2)
    
    # Plot rho and Ttot (precomputed)
    if bDoOrb:
        pu.plot_imshow(ax_rho, rho, title="sum(Wf)", extent=extent, cmap='bwr')
        Ttot = np.sum(Ts_flat_.reshape(npix, npix, nsite), axis=2)
        pu.plot_imshow(ax_Ttot, Ttot, title="Tunneling sum(M^2)", extent=extent, cmap='hot')
    
    # Process each parameter value
    for i in range(nscan):
        # Update all parameter values
        for param, vals in scan_params:
            params[param] = vals[i]
        
        # Create axes for this parameter value
        ax_Etot = fig.add_subplot(3, nscan+1, i+2)
        ax_STM  = fig.add_subplot(3, nscan+1, i +nscan+3)
        ax_dIdV = fig.add_subplot(3, nscan+1, i+2*nscan+4)
        
        # Run pauli scan and plot results (same as before)
        STM_flat, Es_flat, Ts_flat_ = pauli.run_pauli_scan_top(params['spos'], params['rots'], params, pauli_solver=pauli_solver, Ts=Ts_flat, bOmp=bOmp)

        if not bDoOrb:
            Ttot = np.sum(Ts_flat_.reshape(npix, npix, nsite), axis=2)
            pu.plot_imshow(ax_Ttot, Ttot, title="Tunneling sum(M^2)", extent=extent, cmap='hot')
        
        # Compute dIdV
        params_ = params.copy()
        params_['VBias'] += 0.05
        STM_2, _, _ = pauli.run_pauli_scan_top(params['spos'], params['rots'], params_, pauli_solver=pauli_solver, Ts=Ts_flat, bOmp=bOmp)
        dIdV = (STM_2 - STM_flat) / 0.01
        
        # Reshape and plot
        STM = STM_flat.reshape(npix, npix)
        Etot = np.max(Es_flat.reshape(npix, npix, nsite), axis=2)
        dIdV = dIdV.reshape(npix, npix)
        
        # Build title showing all swept parameters
        title_parts = [f"{name}={params[name]:.3f}" for name in param_names]
        title = ", ".join(title_parts)
        
        pu.plot_imshow(ax_Etot, Etot, title=title+"\nEnergies max(eps)", extent=extent, cmap='bwr'); #ax_Etot.set_aspect('auto')
        pu.plot_imshow(ax_STM,  STM,  title="STM",                       extent=extent, cmap='hot'); #ax_STM.set_aspect('auto')
        pu.plot_imshow(ax_dIdV, dIdV, title="dI/dV",                     extent=extent, cmap='bwr'); #ax_dIdV.set_aspect('auto')
    
    plt.tight_layout()
    return fig

def calculate_1d_scan(params, start_point, end_point, pointPerAngstrom=5):
    """Calculate 1D scan between two points using run_pauli_scan"""
    x1, y1 = start_point
    x2, y2 = end_point
    dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    npoints = max(100, int(dist * pointPerAngstrom))
    t = np.linspace(0, 1, npoints)
    x = x1 + (x2 - x1) * t
    y = y1 + (y2 - y1) * t
    distance = np.linspace(0, dist, npoints)

    zT = params['z_tip'] + params['Rtip']
    pTips = np.zeros((npoints, 3))
    pTips[:, 0] = x
    pTips[:, 1] = y
    pTips[:, 2] = zT

    nsite = int(params['nsite'])
    spos, phis = ut.makeCircle(n=nsite, R=params['radius'], phi0=params['phiRot'])
    spos[:, 2] = params['zQd']
    rots = ut.makeRotMats(phis + params['phiRot'])

    Vtips = np.full(npoints, params['VBias'])
    cpp_params = np.array([ params['Rtip'], params['zV0'], params['Esite'], params['decay'], params['GammaT'], params['W'] ])
    order = params.get('order', 1)
    cs = params.get('cs', np.array([1.0, 0.0, 0.0, 0.0]))
    state_order = np.array([0, 4, 2, 6, 1, 5, 3, 7], dtype=np.int32)

    # Run scan
    solver = pauli.PauliSolver(nSingle=nsite, nleads=2, verbosity=0)
    current, Es, Ts = solver.scan_current_tip( pTips, Vtips, spos, cpp_params, order, cs, state_order, rots=rots, bOmp=False, bMakeArrays=True )
    return distance, Es, Ts, current, x, y, x1, y1, x2, y2

def plot_1d_scan_results(distance, Es, Ts, STM, nsite, ref_data_line=None, ref_columns=None):
    """Plot results of 1D scan"""
    scan_fig = plt.figure(figsize=(10, 12))
    bRef = (ref_data_line is not None and ref_columns is not None)

    ax1 = scan_fig.add_subplot(311)
    clrs = ['r', 'g', 'b']
    for i in range(nsite):
        ax1.plot(distance, Es[:, i], '-', linewidth=0.5, color=clrs[i], label=f'E_{i+1}')
        if bRef:
            icol = ref_columns[f'Esite_{i+1}']
            ax1.plot( ref_data_line[:, 0], ref_data_line[:, icol], ':', color=clrs[i], alpha=0.7, label=f'Ref E_{i+1}')
    ax1.set_ylabel('Energy [eV]')
    ax1.legend()
    ax1.grid(True)

    ax2 = scan_fig.add_subplot(312)
    for i in range(nsite):
        ax2.plot(distance, Ts[:, i], '-', linewidth=0.5, color=clrs[i], label=f'T_{i+1}')
        if bRef:
            icol = ref_columns[f'Tsite_{i+1}']
            ax2.plot( ref_data_line[:, 0], ref_data_line[:, icol], ':', color=clrs[i], alpha=0.7, label=f'Ref T_{i+1}' )
    ax2.set_ylabel('Hopping T [a.u.]')
    ax2.legend()
    ax2.grid(True)

    ax3 = scan_fig.add_subplot(313)
    ax3.plot(distance, STM, '.-', color='k', linewidth=0.5, markersize=1.5, label='STM')
    ax3.set_ylabel('Current [a.u.]')
    ax3.legend()
    ax3.grid(True)

    scan_fig.tight_layout()
    plt.show()
    return scan_fig

def save_1d_scan_data(params, distance, x, y, Es, Ts, STM, nsite, x1, y1, x2, y2):
    """Save 1D scan data to file"""
    # Prepare header with parameters
    param_header = "# Calculation parameters:\n"
    for key, val in params.items():
        param_header += f"# {key}: {val}\n"
    # Add column descriptions
    param_header += "\n# Column descriptions:\n"
    param_header += "# 0: Distance (Angstrom)\n"
    param_header += "# 1: X coordinate\n"
    param_header += "# 2: Y coordinate\n"
    for i in range(nsite):
        param_header += f"# {i+3}: Esite_{i+1}\n"
    for i in range(nsite):
        param_header += f"# {i+3+nsite}: Tsite_{i+1}\n"
    param_header += f"# {3+2*nsite}: STM_total\n"
    # Add line coordinates
    param_header += f"\n# Line scan from ({x1:.2f}, {y1:.2f}) to ({x2:.2f}, {y2:.2f})"
    # Stack data
    save_data = np.column_stack( [distance, x, y] + [Es[:, i] for i in range(nsite)] + [Ts[:, i] for i in range(nsite)] + [STM] )
    filename = f"line_scan_{x1:.1f}_{y1:.1f}_to_{x2:.1f}_{y2:.1f}.dat"
    np.savetxt(filename, save_data, header=param_header)
    print(f"Data saved to {filename}")
    return filename
    
def calculate_xV_scan(params, start_point, end_point, ax_Emax=None, ax_STM=None, ax_dIdV=None, nx=100, nV=100, Vmin=0.0,Vmax=None, bLegend=True):
    """Scan tip along a line for a range of voltages and plot Emax, STM, dI/dV."""
    print("calculate_xV_scan()", start_point, end_point,  )
    # Line geometry
    x1, y1 = start_point; x2, y2 = end_point
    dist = np.hypot(x2-x1, y2-y1)
    npts = nx
    t = np.linspace(0,1,npts)
    x = x1 + (x2-x1)*t; y = y1 + (y2-y1)*t
    if Vmax is None: Vmax = params['VBias']
    Vbiases = np.linspace(Vmin, Vmax, nV)

    # Tip positions and voltages grid
    pTips = np.zeros((npts*nV,3))

    # Site & rotation
    nsite = int(params['nsite'])
    spos, phis = ut.makeCircle(n=nsite, R=params['radius'], phi0=params['phiRot'])
    spos[:,2] = params['zQd']
    rots = ut.makeRotMats(phis + params['phiRot'])

    pTips = np.zeros((npts,3))
    pTips[:,0] = x; pTips[:,1] = y
    zT = params['z_tip'] + params['Rtip']
    pTips[:,2] = zT
    
    state_order = np.array( [0,4,2,6,1,5,3,7] )
    current, Es, Ts = pauli.run_pauli_scan_xV( pTips, Vbiases, spos,  params, rots=rots, order=1, cs=None, bOmp=False, state_order=state_order, Ts=None )

    # reshape
    STM = current.reshape(nV,npts)
    Es  = Es.reshape(nV,npts,nsite)
    # max energy per bias
    Emax = Es.max(axis=2)
    # dI/dV
    dIdV = np.gradient(STM, Vbiases, axis=0)

    # Plot
    extent = [0, dist, Vmin, Vmax]
    if ax_Emax is not None:
        pu.plot_imshow(ax_Emax, Emax, title='Emax', extent=extent, cmap='bwr')
        ax_Emax.set_aspect('auto')
        if bLegend: ax_Emax.set_ylabel('V [V]')
    if ax_STM is not None:
        pu.plot_imshow(ax_STM, STM, title='STM', extent=extent, cmap='hot')
        ax_STM.set_aspect('auto')
        if bLegend: ax_STM.set_ylabel('V [V]')
    if ax_dIdV is not None:
        pu.plot_imshow(ax_dIdV, dIdV, title='dI/dV', extent=extent, cmap='bwr')
        ax_dIdV.set_aspect('auto')
        if bLegend: ax_dIdV.set_ylabel('V [V]')

    print("calculate_xV_scan() DONE")
    return x, Vbiases, Emax, STM, dIdV

if __name__ == "__main__":
    # Example usage when run as standalone script - using same defaults as GUI
    params = {
        'VBias': 0.3, 'Rtip': 2.5, 'z_tip': 3.0,

        #'W': 0.00,
         #'W': 0.005,
        #'W': 0.01,
        #'W': 0.015,
        #'W': 0.02,
        #'W': 0.025,
        'W': 0.03,
        
        'GammaS': 0.01, 'GammaT': 0.01, 'Temp': 0.224, 'onSiteCoulomb': 3.0,
        #'zV0': -10.0, 'zQd': 0.0,
        'zV0': -3.3, 'zQd': 0.0,
        'nsite': 3, 'radius': 5.2, 
        'phiRot': np.pi*0.5 + 0.2 ,
        'Esite': -0.04, 'Q0': 1.0, 'Qzz': 0.0,
        #'L': 20.0, 
        #'npix': 100, 
        
        'L': 30.0, 
        'npix': 150,
        'dQ': 0.001,
        #'decay': 0.05, 
        'decay': 0.3,
    }
    verbosity = 0

    #run_scan_xy_orb(params, orbital_file='QD.cub')
    #scan_xy_orb(params, scan_params=[('VBias', np.linspace(0.0, 0.5, 5))], selected_params=['VBias','zV0', 'z_tip', 'W', 'Esite' ], orbital_file='QD.cub')
    # plt.savefig('scan_param_sweep_xy_orb.png')
    # plt.show()
    # exit()

    # #scan_param_sweep(params, scan_params=[('z_tip', np.linspace(1.0, 6.0, 5)), ('Esite', np.linspace(-0.20, -0.05, 5))], selected_params=['VBias','zV0'], orbital_file='QD.cub')
    #scan_param_sweep_xy_orb(params, scan_params=[('VBias', np.linspace(0.0, 0.5, 5))], selected_params=['VBias','zV0', 'z_tip', 'W', 'Esite' ], orbital_file='QD.cub')
    scan_param_sweep_xy_orb(params, scan_params=[('VBias', np.linspace(0.05, 0.2, 6))], selected_params=['VBias','zV0', 'z_tip', 'W', 'Esite' ], orbital_file=None)
    plt.savefig('scan_param_sweep_xy_orb.png')
    plt.show()
    # exit()
    
    #run_scan_xy_orb( params )
    
    exit()

    # scan_param_sweep(params, [('z_tip', np.linspace(1.0, 6.0, 5)), ('Esite', np.linspace(-0.20, -0.05, 5))], selected_params=['VBias','zV0'])
    # plt.savefig('scan_param_sweep.png')
    # plt.show()
    # exit()
    
    # Initialize pauli solver
    pauli_solver = pauli.PauliSolver( nSingle=3, nleads=2, verbosity=verbosity )
    
    # Create figure
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(15, 10))
    
    # Run scans with plotting
    #scan_xV(params, ax_V2d=ax1, ax_Vtip=ax2, ax_Esite=ax3)
    scan_xV(params, ax_Vtip=ax1,  ax_V2d=ax2,  ax_I2d=ax3)
    scan_xy(params, pauli_solver, ax_Etot=ax4, ax_Ttot=ax5, ax_STM=ax6)
    
    plt.tight_layout()
    plt.show()

    # Example parameter sweep
    #z_tip_vals = np.linspace(1.0, 3.0, 5)  # Scan 5 tip heights
    #fig = scan_param_sweep(params, [('z_tip', z_tip_vals)])
    #plt.show()
