import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import math
import numpy as _np  # for flattening axes

import sys
sys.path.append('../../')
from pyProbeParticle import pauli
from pyProbeParticle import utils as ut
import plot_utils as pu
import numpy as np
import time

import orbital_utils
from scipy.interpolate import RectBivariateSpline


# ===========================================
# ============= Utility functions
# ===========================================

def validate_probabilities(probs, tol=-1e-12):
    """Check if any probabilities are negative below tolerance threshold.
    
    Args:
        probs: Probability array (can be 1D, 2D or 3D)
        tol: Negative values above this threshold will trigger an error
        
    Raises:
        ValueError if any probabilities are below tolerance
    """
    if probs is None: 
        return
    # Reshape to ensure we always have at least 2 dimensions
    probs = np.atleast_2d(probs)
    
    # Find minimum along all dimensions except the last (state dimension)
    min_vals = np.min(probs, axis=tuple(range(probs.ndim-1)))
    #print(f"Validating probabilities: min_vals {min_vals} tol {tol}")
    for i, min_val in enumerate(min_vals):
        if min_val < tol: 
            print(f"ERROR in validate_probabilities() min_val {min_val} < tol {tol}")
            raise ValueError(f"State {i} has negative probability ({min_val:.2e}) below tolerance ({tol:.2e})")

def make_site_geom( params ):
    nsite=params['nsite']
    spos, phis = ut.makeCircle(n=nsite,R=params['radius'],phi0=params['phiRot'])
    spos[:,2]  = params['zQd']
    angles     = phis+params['phi0_ax']
    rots       = ut.makeRotMats( angles )
    return spos, rots, angles

def make_grid_axes(fig, nplots):
    """
    Create a near-square grid of subplots for nplots axes.
    rows = ceil(sqrt(nplots)), cols = ceil(nplots/rows).
    Returns flat array of axes of length rows*cols.
    """
    nrows = math.ceil(math.sqrt(nplots))
    ncols = math.ceil(nplots / nrows)
    axs = fig.subplots(nrows, ncols)
    # flatten axes array
    return _np.array(axs).flatten()

def cut_central_region(map_list, dcanv, big_npix, small_npix):
    """Crop central small_npix×small_npix from big_npix maps with pixel size dcanv"""
    center=big_npix//2; half=small_npix//2; start=center-half; end=start+small_npix
    return [m[start:end, start:end] for m in map_list]

def generate_central_hops(orb2D, orb_lvec, spos_xy, angles, z0, dcanv, big_npix, small_npix, decay=0.2):
    """Compute big canvas hops and crop central small canvas"""
    big_dd=[dcanv,dcanv]; big_shape=(big_npix,big_npix)
    tipWf,shift=orbital_utils.make_tipWf(big_shape,big_dd,z0=z0,decay=decay)
    Ms_big,rho_big=orbital_utils.calculate_Hopping_maps(orb2D,orb_lvec,spos_xy,angles,big_dd,big_shape,tipWf,bTranspose=True)
    Ms_small=cut_central_region(Ms_big,dcanv,big_npix,small_npix)
    rho_small=cut_central_region([rho_big],dcanv,big_npix,small_npix)[0]
    return Ms_small, rho_small

# ===========================================
# ============= Plot & Save functions
# ===========================================

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

def plot_1d_scan_results(distance, Es, Ts, STM, nsite, probs=None, ref_data_line=None, ref_columns=None, fig=None):
    """Plot results of 1D scan"""
    bProbs = (probs is not None)
    nsub   = 3 + bProbs

    if fig is None:
        fig = plt.figure(figsize=(10, 12))
    bRef = (ref_data_line is not None and ref_columns is not None)

    ax1 = fig.add_subplot(nsub,1,1)
    clrs = ['r', 'g', 'b']
    for i in range(nsite):
        ax1.plot(distance, Es[:, i], '-', linewidth=0.5, color=clrs[i], label=f'E_{i+1}')
        if bRef:
            icol = ref_columns[f'Esite_{i+1}']
            ax1.plot( ref_data_line[:, 0], ref_data_line[:, icol], ':', color=clrs[i], alpha=0.7, label=f'Ref E_{i+1}')
    ax1.set_ylabel('Energy [eV]')
    ax1.legend()
    ax1.grid(True)

    ax2 = fig.add_subplot(nsub,1,2)
    for i in range(nsite):
        ax2.plot(distance, Ts[:, i], '-', linewidth=0.5, color=clrs[i], label=f'T_{i+1}')
        if bRef:
            icol = ref_columns[f'Tsite_{i+1}']
            ax2.plot( ref_data_line[:, 0], ref_data_line[:, icol], ':', color=clrs[i], alpha=0.7, label=f'Ref T_{i+1}' )
    ax2.set_ylabel('Hopping T [a.u.]')
    ax2.legend()
    ax2.grid(True)

    ax3 = fig.add_subplot(nsub,1,3)
    ax3.plot(distance, STM, '.-', color='k', linewidth=0.5, markersize=1.5, label='STM')
    ax3.set_ylabel('Current [a.u.]')
    ax3.legend()
    ax3.grid(True)

    if probs is not None:
        ax4 = fig.add_subplot(nsub,1,4)
        nsite = probs.shape[1]
        state_order = pauli.make_state_order(nsite)
        labels = pauli.make_state_labels(state_order)
        for idx in range(probs.shape[1]): ax4.plot(distance, probs[:,idx], label=labels[idx])
        ax4.set_xlabel('Distance'); ax4.set_ylabel('Probability')
        ax4.legend()
        ax4.grid(True)

    fig.tight_layout()
    #plt.show()
    return fig

def plot_state_probabilities(probs, extent, axs=None, fig=None, labels=None, aspect='auto'):
    """
    Plot multiple state probability maps. Handles single or array of axes.
    probs: 3D array shape (nV, nx, n_states)
    extent: sequence of 4 [xmin, xmax, ymin, ymax]
    """
    # Number of states to plot
    n_states = probs.shape[-1]
    # Create default figure/axes if needed
    ncols = min(2, n_states)
    nrows = int(np.ceil(n_states / ncols))
    if fig is None:
        fig = plt.figure( figsize=(4*n_states, 3*nrows) )
    if axs is None:
        axs = fig.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))

    # Flatten axes into list
    try:
        axs_flat = axs.flatten()
    except AttributeError:
        axs_flat = [axs]
    # Ensure fig is set
    if fig is None and hasattr(axs_flat[0], 'figure'):
        fig = axs_flat[0].figure
    # Plot each state's probability
    for idx in range(n_states):
        ax = axs_flat[idx]
        ax.clear()
        title = labels[idx] if labels and idx < len(labels) else f"P{idx}"
        im = ax.imshow(probs[:,:,idx], origin='lower', extent=extent, cmap='viridis', interpolation='nearest')
        ax.set_aspect(aspect)
        ax.set_title(title)
        ax.set_xlabel('x [Å]')
        ax.set_ylabel('V_bias [V]')
        fig.colorbar(im, ax=ax, label='Probability')
    # Hide extra axes
    # for extra_ax in axs_flat[n_states:]:
    #     extra_ax.clear()
    #     extra_ax.set_visible(False)
    fig.tight_layout()
    return fig, axs

# ===========================================
# ============= Scan functions
# ===========================================

# ============= Simulate Tip Field ( No Pauli solver simulation )

def scan_xV(params, ax_xV=None, ax_Esite=None, ax_I2d=None, nx=100, nV=100, ny=100, bLegend=True, scV=1.0, Woffsets=None, pSites=None):
    """
    Scan voltage dependence above one particle
    
    Args:
        params: Dictionary of parameters
        ax_V2d:   Axis for 2D voltage scan plot (Esite vs x,V)
        ax_Vtip:  Axis for tip potential plot
        ax_Esite: Axis for site potential plot
        nx: Number of x points
        nV: Number of voltage points
        ny: Number of y points
    """
    L = params['L']
    z_tip = params['z_tip']
    zT    = z_tip + params['Rtip']
    zV0   = params['zV0']
    zVd   = params['zVd']
    
    VBias = params['VBias']
    Rtip  = params['Rtip']
    Esite = params['Esite']
    bMirror = params.get('bMirror', True)
    bRamp   = params.get('bRamp',   True)

    pTips_1d = np.zeros((nx, 3))
    x_coords = np.linspace(-L, L, nx)
    pTips_1d[:,0] = x_coords
    pTips_1d[:,2] = zT
    V_vals = np.linspace(0.0, VBias, nV)

    if pSites is None:
        zQd   = params['zQd']
        pSites = np.array([[0.0, 0.0, zQd]])
    # Plotting if axes provided
    if ax_xV is not None:
        # 1D Potential calculations
        V1d = pauli.evalSitesTipsMultipoleMirror(pTips_1d, pSites=pSites, VBias=VBias, E0=Esite, Rtip=Rtip, zV0=zV0, zVd=zVd, bMirror=bMirror, bRamp=bRamp)
        V1d_ = V1d - Esite
        
        # 2D Potential calculations
        X_v, V_v = np.meshgrid(x_coords, V_vals)
        pTips_v = np.zeros((nV*nx, 3))
        pTips_v[:,0] = X_v.flatten()
        pTips_v[:,2] = zT
        V2d = pauli.evalSitesTipsMultipoleMirror(pTips_v, pSites=pSites, VBias=V_v.flatten(), E0=Esite, Rtip=Rtip, zV0=zV0, zVd=zVd, bMirror=bMirror, bRamp=bRamp).reshape(nV, nx)
        
        pu.plot_imshow(ax_xV, V2d, title="Esite(tip_x,tip_V)", extent=[-L, L, 0.0, VBias], ylabel="V [V]", cmap='bwr')
        ax_xV.plot(x_coords, V1d,  label='V_tip')
        ax_xV.plot(x_coords, V1d_, label=f'V_tip-E_site({Esite:.3f})')
        ax_xV.plot(x_coords, x_coords*0.0 + VBias, label='VBias')
        ax_xV.axhline(0.0, ls='--', c='k')
        #ax_V2d.set_title("1D Potential (z=0)")
        # ax_V2d.set_xlabel("x [Å]")
        # ax_V2d.set_ylabel("V [V]")
        # ax_V2d.grid()
        ax_xV.set_aspect('auto')
        if bLegend:
            ax_xV.legend()
    else:
        V1d  = None
        V1d_ = None
    I = None
    if ax_I2d is not None:
        #current, _, _, probs = pauli.run_pauli_scan_xV(pTips_1d, V_vals, pSites=pSites, params=params)
        if Woffsets is None:
            I, _, _, probs = pauli.run_pauli_scan_xV(pTips_1d, V_vals, pSites=pSites, params=params)
            validate_probabilities(probs)
            #pu.plot_imshow(ax_I2d, Ts, title="Current", extent=[-L, L, 0.0, VBias], ylabel="V [V]", cmap='hot')
            #I = Ts
        else:
            Ts = pauli.evalSitesTipsTunneling(pTips_v, pSites=pSites, beta=params['Temp'], Amp=1.0).reshape(nV, nx)
            I = Ts*0.0
            for i, W in enumerate(Woffsets):
                #occup = 1-1.0/(1.0 + np.exp(( V2d-Woffset )/(0.1*params['Temp'])))
                occup = np.ones_like(Ts); occup[ V2d-W<0.0 ] = 0.0
                I += Ts * occup
            pu.plot_imshow(ax_I2d, I, title="Current", extent=[-L, L, 0.0, VBias], ylabel="V [V]", cmap='hot')
        ax_I2d.set_aspect('auto')
    
    if ax_Esite is not None:
        pTip = np.array([[0.0, 0.0, zT]])
        x_xz = np.linspace(-L, L, nx)
        z_xz = np.linspace(-L, L, ny)
        X_xz, Z_xz = np.meshgrid(x_xz, z_xz)
        ps_xz = np.array([X_xz.flatten(), np.zeros_like(X_xz.flatten()), Z_xz.flatten()]).T
        Esites   = pauli.evalSitesTipsMultipoleMirror( pTip, pSites=ps_xz, VBias=VBias, Rtip=Rtip, zV0=zV0, zVd=zVd, bMirror=bMirror, bRamp=bRamp, bSiteScan=True).reshape(ny,nx)
        extent_xz = [-L, L, -L, L]
        pu.plot_imshow(ax_Esite, Esites, title="Esite(x,z)", extent=extent_xz, cmap='bwr', vmin=-VBias*scV, vmax=VBias*scV)
        circ1, _ = ut.makeCircle(16, R=Rtip, axs=(0,2,1), p0=(0.0, 0.0, zT))
        circ2, _ = ut.makeCircle(16, R=Rtip, axs=(0,2,1), p0=(0.0, 0.0, 2*zV0-zT))
        ax_Esite.plot(circ1[:,0], circ1[:,2], ':k')
        ax_Esite.plot(circ2[:,0], circ2[:,2], ':k')
        ax_Esite.axhline(zV0,       ls='--', c='k', label=f'zV0 {zV0:.3f}')
        ax_Esite.axhline(zQd,       ls='--', c='g', label=f'zQd {zQd:.3f}')
        ax_Esite.axhline(z_tip,     ls='--', c='m', label=f'z_tip {z_tip:.3f}')
        ax_Esite.axhline(z_tip+zVd, ls='--', c='b', label=f'z_tip+zVd {zVd+z_tip:.3f}')
        if bLegend:
            ax_Esite.legend()
    else:
        Esites = None
        
    return V1d, V2d, Esites, I


# ============= 1D scan with Pauli Master-equation simulation

def calculate_1d_scan(params, start_point, end_point, pointPerAngstrom=5, ax_probs=None, pauli_solver=None ):
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
    # Site geometry (positions, rotations, angles)
    spos, rots, angles = make_site_geom(params)

    Vtips = np.full(npoints, params['VBias'])
    cpp_params = pauli.make_cpp_params(params)
    cs, order  = pauli.make_quadrupole_Coeffs(params['Q0'], params['Qzz'])
    state_order = pauli.make_state_order(nsite)
    # Run scan
    if pauli_solver is None:
        pauli_solver = pauli.PauliSolver(nSingle=nsite, nleads=2, verbosity=0)
    current, Es, Ts, probs = pauli_solver.scan_current_tip( pTips, Vtips, spos,  cpp_params, order, cs, state_order, rots=rots, bOmp=False, bMakeArrays=True )
    validate_probabilities(probs)
    if ax_probs:
        axp = ax_probs
        for i in range(probs.shape[1]): axp.plot(distance, probs[:,i], label=f"P{i}")
        axp.legend()
    return distance, Es, Ts, current, x, y, x1, y1, x2, y2, probs    

# ============= 2D (x,y) scan with Pauli Master-equation simulation

def scan_xy(params, pauli_solver=None, ax_Etot=None, ax_Ttot=None, ax_STM=None, ax_dIdV=None, bOmp=False, sdIdV=0.5, fig_probs=None):
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
    
    # Site geometry (positions, rotations, angles)
    spos, rots, angles = make_site_geom(params)
    
    # Run pauli scan
    STM, Es, Ts, probs = pauli.run_pauli_scan_top(spos, rots, params, pauli_solver=pauli_solver, bOmp=bOmp)
    validate_probabilities(probs)
    #print( "min,max Es", np.min(Es), np.max(Es))
    #print( "min,max Ts", np.min(Ts), np.max(Ts))
    #print( "min,max STM", np.min(STM), np.max(STM))
    dIdV = None
    if ax_dIdV is not None:
        params_ = params.copy()
        dQ = params.get('dQ', 0.05)
        params_['VBias'] += dQ
        STM_2, _, _, probs = pauli.run_pauli_scan_top(spos, rots, params_, pauli_solver=pauli_solver, bOmp=bOmp)       
        dIdV = (STM_2 - STM) / dQ

    Ttot = np.max(Ts, axis=2)
    Etot = np.max(Es, axis=2)
    
    # Plotting if axes provided
    extent = [-L, L, -L, L]
    if ax_Etot is not None: pu.plot_imshow(ax_Etot, Etot, title="Energies  (max)",  extent=extent, cmap='bwr')
    if ax_Ttot is not None: pu.plot_imshow(ax_Ttot, Ttot, title="Tunneling (max)",  extent=extent, cmap='hot')
    if ax_STM  is not None: pu.plot_imshow(ax_STM,  STM,  title="STM",              extent=extent, cmap='hot')
    if ax_dIdV is not None: pu.plot_imshow(ax_dIdV, dIdV, title="dI/dV",            extent=extent, cmap='bwr', scV=sdIdV)
    if fig_probs is not None:
        # dynamic grid axes for probabilities
        n_states = probs.shape[2]
        state_order = pauli.make_state_order(params['nsite'])
        labels = pauli.make_state_labels(state_order)
        axs = make_grid_axes(fig_probs, n_states)
        plot_state_probabilities(probs, extent=extent, axs=axs[:n_states], fig=fig_probs, labels=labels, aspect='equal')
    
    probs = probs.reshape(params['npix'], params['npix'], -1)
    return STM, dIdV, Es, Ts, probs, spos, rots

def scan_xy_orb(params, orbital_2D=None, orbital_lvec=None, pauli_solver=None, ax_Etot=None, ax_Ttot=None, ax_STM=None, ax_Ms=None, ax_rho=None, ax_dIdV=None, decay=None, bOmp=False, Tmin=0.0, EW=2.0, sdIdV=0.5, fig_probs=None):
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

    T0 = time.perf_counter()

    # small canvas & site setup
    L=params['L']; npix=params['npix']
    nsite=params['nsite']; z_tip=params['z_tip']
    # Site geometry (positions, rotations, angles)
    spos, rots, angles = make_site_geom(params)
    # big-to-small hopping computation
    #dcanv=params['L']/params['npix']

    #T1=time.perf_counter()
    if orbital_2D is not None:
        dcanv=2*L/npix
        big_npix=int(params.get('big_npix',400))
        Ms,rho=generate_central_hops(orbital_2D,orbital_lvec,spos[:,:2],angles,z_tip,dcanv,big_npix,npix,decay=decay or params.get('decay',0.2))
        Ts_flat = np.zeros((npix*npix, nsite), dtype=np.float64)
        for i in range(nsite): Ts_flat[:,i]=Ms[i].flatten() #**2
    else:
        Ts_flat = None

    # Create solver if not provided
    if pauli_solver is None:
        pauli_solver = pauli.PauliSolver(nSingle=nsite, nleads=2)
    pauli.set_valid_point_cuts(Tmin, EW)
    
    #T2 = time.perf_counter(); print("Time(scan_xy_orb.2 Ts,PauliSolver)",  T2-T1 )     
    #bOmp = True
    STM_flat, Es_flat, Ts_flat_, probs = pauli.run_pauli_scan_top(spos, rots, params, pauli_solver=pauli_solver, Ts=Ts_flat, bOmp=bOmp)
    validate_probabilities(probs)
    #STM_flat, Es_flat, _ = pauli.run_pauli_scan_top(spos, rots, params, pauli_solver=pauli_solver )
    dIdV = None
    if ax_dIdV is not None:
        params_ = params.copy()
        dQ = params.get('dQ', 0.005)
        params_['VBias'] += dQ
        STM_2, _, _, probs = pauli.run_pauli_scan_top(spos, rots, params_, pauli_solver=pauli_solver, Ts=Ts_flat, bOmp=bOmp)       
        dIdV = (STM_2 - STM_flat) / dQ

    #T3 = time.perf_counter(); print("Time(scan_xy_orb.3 pauli.run_pauli_scan)",  T3-T2 )
    #print("min, max STM_flat", np.min(STM_flat), np.max(STM_flat))
    #print("min, max Es_flat", np.min(Es_flat), np.max(Es_flat))
    #print("min, max Ts_flat", np.min(Ts_flat), np.max(Ts_flat))
    
    # Reshape the results
    STM = STM_flat.reshape(npix, npix)
    Es  = Es_flat .reshape(npix, npix, nsite)
    Ts  = Ts_flat_.reshape(npix, npix, nsite)
    
    # Calculate total Ts (max over sites) and Es (max over sites)
    Ttot = np.sum(Ts**2, axis=2)
    Etot = np.max(Es, axis=2)
    
    # Plotting if axes provided
    extent = [-L,L, -L, L]
    probs = probs.reshape(npix, npix, -1)

    T_calc = time.perf_counter(); print(f"scan_xy_orb() calc time: {T_calc-T0:.5f} [s]")

    if ax_Etot is not None: pu.plot_imshow(ax_Etot, Etot, title="Energies max(eps)",      extent=extent, cmap='bwr')
    if ax_rho  is not None and rho is not None: pu.plot_imshow(ax_rho,  rho,  title="sum(Wf)", extent=extent, cmap='bwr')
    if ax_Ttot is not None: pu.plot_imshow(ax_Ttot, Ttot, title="Tunneling sum(M^2)",     extent=extent, cmap='hot')
    if ax_STM  is not None: pu.plot_imshow(ax_STM,  STM,  title="STM",                    extent=extent, cmap='hot')
    if ax_dIdV is not None:
        dIdV = dIdV.reshape(npix, npix)
        pu.plot_imshow(ax_dIdV, dIdV, title="dI/dV", extent=extent, cmap='bwr', scV=sdIdV)
    if ax_Ms   is not None and Ms is not None: 
        for i in range(nsite):
            ax_Ms[i].imshow(Ms[i], cmap='bwr', origin='lower', extent=extent)
            ax_Ms[i].set_title(f"Hopping matrix {i}")
    if fig_probs is not None:
        # plot site probabilities in separate window
        n_states = probs.shape[2]
        state_order = pauli.make_state_order(params['nsite'])
        labels = pauli.make_state_labels(state_order)
        axs_all = make_grid_axes(fig_probs, n_states)
        plot_state_probabilities(probs, extent=extent, axs=axs_all[:n_states], fig=fig_probs, labels=labels, aspect='equal')
    #T4 = time.perf_counter(); print("Time(scan_xy_orb.4 plotting)",  T4-T3 )        
    return STM, dIdV, Es, Ts, probs, spos, rots

def run_scan_xy_orb( params, orbital_file="QD.cub" ):
    print(f"run_scan_xy_orb() ... testing scan_xy_orb() with orbital loading from file {orbital_file}")        
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

    if pauli_solver is None:
        pauli_solver = pauli.PauliSolver(nSingle=params['nsite'], nleads=2, verbosity=0)

    fig, ( ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,5))
    scan_xy_orb(params, orbital_2D=orbital_2D, orbital_lvec=orbital_lvec, pauli_solver=pauli_solver, ax_Etot=ax1, ax_Ttot=ax2, ax_STM=ax3, ax_Ms=None, ax_dIdV=ax4)
        
    fig.suptitle(f"Scan with Orbital-Based Hopping ({orbital_file})", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle

    # Save and show the figure
    plt.savefig('scan_xy_orb_test.png')
    print(f"Figure saved as 'scan_xy_orb_test.png'")
    #plt.show()


# ============= 2D (x,V)-plane scan with Pauli Master-equation simulation

def calculate_xV_scan(params, start_point, end_point, ax_Emax=None, ax_STM=None, ax_dIdV=None, nx=100, nV=100, Vmin=0.0,Vmax=None, bLegend=True, sdIdV=0.5, fig_probs=None, bOmp=False, pauli_solver=None):
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

    # Site & rotation
    nsite = int(params['nsite'])
    # Site geometry (positions, rotations, angles)
    spos, rots, angles = make_site_geom(params)

    pTips = np.zeros((npts,3))
    pTips[:,0] = x; pTips[:,1] = y
    zT = params['z_tip'] + params['Rtip']
    pTips[:,2] = zT
    
    cpp_params = pauli.make_cpp_params(params)
    state_order = pauli.make_state_order(nsite)
    current, Es, Ts, probs = pauli.run_pauli_scan_xV( pTips, Vbiases, spos,  cpp_params, order=1, cs=None, rots=rots, bOmp=bOmp, state_order=state_order, Ts=None, pauli_solver=pauli_solver )
    validate_probabilities(probs)
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
        ax_Emax.set_aspect('auto');
        if bLegend: ax_Emax.set_ylabel('V [V]')
    if ax_STM is not None:
        pu.plot_imshow(ax_STM, STM, title='STM', extent=extent, cmap='hot')
        ax_STM.set_aspect('auto');
        if bLegend: ax_STM.set_ylabel('V [V]')
    if ax_dIdV is not None:
        pu.plot_imshow(ax_dIdV, dIdV, title='dI/dV', extent=extent, cmap='bwr', scV=sdIdV)
        ax_dIdV.set_aspect('auto');
        if bLegend: ax_dIdV.set_ylabel('V [V]')

    probs = probs.reshape(nV, nx, -1)
    if fig_probs is not None:
        # dynamic grid axes for probabilities
        n_states = probs.shape[2]
        state_order = pauli.make_state_order(params['nsite'])
        labels = pauli.make_state_labels(state_order)
        axs = make_grid_axes(fig_probs, n_states)
        plot_state_probabilities(probs, extent=[0,dist,Vmin,Vmax], axs=axs[:n_states], fig=fig_probs, labels=labels)
    print("calculate_xV_scan() DONE")
    return STM, dIdV, Es, Ts, probs, x, Vbiases, spos, rots

def calculate_xV_scan_orb(params, start_point, end_point, orbital_2D=None, orbital_lvec=None, pauli_solver=None, bOmp=False, ax_Emax=None, ax_STM=None, ax_dIdV=None, nx=100, nV=100, Vmin=0.0, Vmax=None, bLegend=True, sdIdV=0.5, decay=None, fig_probs=None):
    """Scan voltage dependence along a line using orbital-based hopping Ts"""
    T0 = time.perf_counter()
    # Line geometry
    x1, y1 = start_point; x2, y2 = end_point
    dist = np.hypot(x2-x1, y2-y1)
    npts = nx
    t = np.linspace(0,1,npts)
    x = x1 + (x2-x1)*t; y = y1 + (y2-y1)*t
    if Vmax is None: Vmax = params['VBias']
    Vbiases = np.linspace(Vmin, Vmax, nV)

    # Site & rotation
    nsite = int(params['nsite'])
    # Site geometry (positions, rotations, angles)
    spos, rots, angles = make_site_geom(params)

    # Compute hopping Ts along line from orbital data
    if orbital_2D is not None:
        L = params['L']; npix = params['npix']
        dcanv = 2*L/npix
        big_npix = int(params.get('big_npix',400))
        Ms, _ = generate_central_hops(orbital_2D, orbital_lvec, spos[:,:2], angles, params['z_tip']+params['Rtip'], dcanv, big_npix, npix, decay=decay or params.get('decay',0.2))
        # Prepare grid for interpolation
        coords = (np.arange(npix) + 0.5 - npix/2)*dcanv
        Ts_line = np.zeros((npts, nsite))
        for i in range(nsite):
            Ts_map = Ms[i] #**2
            interp = RectBivariateSpline(coords, coords, Ts_map)
            Ts_line[:,i] = interp(y, x, grid=False)
        Ts_input = Ts_line
    else:
        Ts_input = None

    # Prepare tip positions
    pTips = np.zeros((npts,3))
    pTips[:,0] = x; pTips[:,1] = y; pTips[:,2] = params['z_tip']+params['Rtip']

    state_order = pauli.make_state_order(nsite)
    # Run scan using parameter dict (wrapper generates C++ params internally)
    current, Es, Ts, probs = pauli.run_pauli_scan_xV(pTips, Vbiases, spos, params, order=1, cs=None, rots=rots, state_order=state_order, Ts=Ts_input, bOmp=bOmp, pauli_solver=pauli_solver )
    validate_probabilities(probs)
    # reshape and compute
    STM = current.reshape(nV, npts)
    Es  = Es.reshape(nV, npts, nsite)
    Emax = Es.max(axis=2)
    dIdV = np.gradient(STM, Vbiases, axis=0)

    #T_calc = time.perf_counter(); print(f"calculate_xV_scan_orb() calc time: {T_calc-T0:.5f} [s]")

    # Plot results
    extent = [0, dist, Vmin, Vmax]
    if ax_Emax is not None:
        pu.plot_imshow(ax_Emax, Emax, title='Emax', extent=extent, cmap='bwr')
        ax_Emax.set_aspect('auto');
        if bLegend: ax_Emax.set_ylabel('V [V]')
    if ax_STM is not None:
        pu.plot_imshow(ax_STM, STM, title='STM', extent=extent, cmap='hot')
        ax_STM.set_aspect('auto');
        if bLegend: ax_STM.set_ylabel('V [V]')
    if ax_dIdV is not None:
        pu.plot_imshow(ax_dIdV, dIdV, title='dI/dV', extent=extent, cmap='bwr', scV=sdIdV)
        ax_dIdV.set_aspect('auto');
        if bLegend: ax_dIdV.set_ylabel('V [V]')

    probs = probs.reshape(nV, nx, -1)
    if fig_probs is not None:
        # dynamic grid axes for probabilities
        n_states = probs.shape[2]
        state_order = pauli.make_state_order(params['nsite'])
        labels = pauli.make_state_labels(state_order)
        axs = make_grid_axes(fig_probs, n_states)
        plot_state_probabilities(probs, extent=[0,dist,Vmin,Vmax], axs=axs[:n_states], fig=fig_probs, labels=labels)
    return STM, dIdV, Es, Ts, probs, x, Vbiases, spos, rots


# ===========================================
# ============= Sweep functions
# ===========================================

def sweep_param_xV(params, scan_params, selected_params=None, nx=100, nV=100, sz=3, bLegend=False):
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
    title = "sweep_param_xV: "
    if selected_params:
        title_params = [p for p in selected_params if p not in param_names]
        if title_params:
            title += ", ".join([f"{k}={params[k]}" for k in title_params])
    fig.suptitle(title, fontsize=12)
    
    # Make copy of params to avoid modifying original
    params = params.copy()
    
    for i in range(nscan):
        # Update all parameter values
        for param, vals in scan_params:
            params[param] = vals[i]
        
        # Get current column axes
        ax_xV    = axes[0,i] if nscan > 1 else axes[0]
        ax_Esite = axes[1,i] if nscan > 1 else axes[1]
        
        # Run scan for current parameter values
        scan_xV(params, ax_xV=ax_xV, ax_Esite=ax_Esite, nx=nx, nV=nV, bLegend=bLegend)
        
        # Build title showing all swept parameters
        title_parts = [f"{name}={params[name]:.3f}" for name in param_names]
        title = ", ".join(title_parts)
        ax_xV   .set_title(title, fontsize=10)
        ax_Esite.set_title(title, fontsize=10)
    
    plt.tight_layout()
    return fig

def scan_param_sweep_xy_orb(params, scan_params, selected_params=None, orbital_2D=None, orbital_lvec=None, orbital_file=None, pauli_solver=None, bOmp=False, Tmin=0.0, EW=2.0, sdIdV=0.5, fig=None):
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
        for i in range(nsite): Ts_flat[:,i] = Ms[i].flatten()   #**2
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
    

    # Site geometry (positions, rotations, angles)
    spos, rots, angles = make_site_geom(params)
    
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
        Ttot = np.sum(Ts_flat_.reshape(npix, npix, nsite)**2, axis=2)
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
        STM_flat, Es_flat, Ts_flat_, probs_ = pauli.run_pauli_scan_top(params['spos'], params['rots'], params, pauli_solver=pauli_solver, Ts=Ts_flat, bOmp=bOmp )
        validate_probabilities(probs_)
        dIdV = None
        if ax_dIdV is not None:
            params_ = params.copy()
            params_['VBias'] += 0.05
            STM_2, _, _, probs = pauli.run_pauli_scan_top(params['spos'], params['rots'], params_, pauli_solver=pauli_solver, Ts=Ts_flat, bOmp=bOmp)
            validate_probabilities(probs)
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
        pu.plot_imshow(ax_dIdV, dIdV, title="dI/dV",                     extent=extent, cmap='bwr', scV=sdIdV); #ax_dIdV.set_aspect('auto')
    
    plt.tight_layout()
    return fig










if __name__ == "__main__":


    mpl.rcParams.update({
        # Font sizes
        'font.size': 8,                # Default font size
        'axes.titlesize': 8,           # Axes title size
        'axes.labelsize': 8,           # Axes labels (x and y)
        'xtick.labelsize': 7,          # x-axis tick labels
        'ytick.labelsize': 7,          # y-axis tick labels
        'legend.fontsize': 7,          # Legend font size
        'figure.titlesize': 9,         # Figure title (suptitle)
        'figure.labelsize': 8,         # Figure labels (for colorbars, etc.)
        
        # Margins and spacing
        'figure.autolayout': False,    # We'll control layout manually
        'figure.subplot.left': 0.08,   # Left margin
        'figure.subplot.right': 0.95,  # Right margin
        'figure.subplot.bottom': 0.08, # Bottom margin
        'figure.subplot.top': 0.90,    # Top margin
        'figure.subplot.wspace': 0.05,  # Horizontal space between subplots
        'figure.subplot.hspace': 0.05,  # Vertical space between subplots
        
        # Lines and markers
        'lines.linewidth': 0.8,        # Line width
        'lines.markersize': 3,         # Marker size
        'patch.linewidth': 0.6,        # Patch (bars, etc.) edge width
        
        # Legend
        'legend.frameon': False,       # Remove legend frame
        'legend.borderpad': 0.1,       # Padding inside legend border
        'legend.labelspacing': 0.1,    # Vertical space between legend entries
        'legend.handlelength': 1.5,    # Length of legend handles
        'legend.handletextpad': 0.4,   # Space between handle and text
        
        # Axes
        'axes.linewidth': 0.6,         # Axis line width
        'axes.labelpad': 2,            # Space between label and axis
        'xtick.major.pad': 1.5,        # Padding between x-ticks and label
        'ytick.major.pad': 1.5,        # Padding between y-ticks and label
        'xtick.major.size': 2,         # x-tick length
        'ytick.major.size': 2,         # y-tick length
        'xtick.major.width': 0.6,      # x-tick width
        'ytick.major.width': 0.6,      # y-tick width
        
        # Saving figures
        'savefig.bbox': 'tight',       # Remove extra whitespace around figure
        'savefig.pad_inches': 0.02,    # Padding when bbox='tight'
        #'savefig.dpi': 300,            # Higher DPI for quality when small
    })


    # Example usage when run as standalone script - using same defaults as GUI
    params = {
        'VBias': 0.6, 
        'Rtip':  3.0, 
        'z_tip': 5.0,

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
        'zVd': 2.0,
        'zQd': 0.0,
        'nsite': 3, 'radius': 5.2, 
        'phiRot': np.pi*0.5 + 0.2 ,
        
        #'Esite': -0.04,
        'Esite': -0.150, 
        
        'Q0': 1.0, 'Qzz': 0.0,
        #'L': 20.0, 
        #'npix': 100, 
        
        'L': 30.0, 
        'npix': 150,
        'dQ': 0.001,
        #'decay': 0.05, 
        'decay': 0.3,
    }
    verbosity = 0

    # #run_scan_xy_orb(params, orbital_file='QD.cub')
    # #scan_xy_orb(params, scan_params=[('VBias', np.linspace(0.0, 0.5, 5))], selected_params=['VBias','zV0', 'z_tip', 'W', 'Esite' ], orbital_file='QD.cub')
    # # plt.savefig('scan_xy_orb_test.png')
    # # plt.show()
    # # exit()

    # # #scan_param_sweep(params, scan_params=[('z_tip', np.linspace(1.0, 6.0, 5)), ('Esite', np.linspace(-0.20, -0.05, 5))], selected_params=['VBias','zV0'], orbital_file='QD.cub')
    # #scan_param_sweep_xy_orb(params, scan_params=[('VBias', np.linspace(0.0, 0.5, 5))], selected_params=['VBias','zV0', 'z_tip', 'W', 'Esite' ], orbital_file='QD.cub')
    # scan_param_sweep_xy_orb(params, scan_params=[('VBias', np.linspace(0.05, 0.2, 6))], selected_params=['VBias','zV0', 'z_tip', 'W', 'Esite' ], orbital_file=None)
    # plt.savefig('scan_param_sweep_xy_orb.png')
    # plt.show()
    # # exit()
    
    # #run_scan_xy_orb( params )
    
    #exit()

    # scan_param_sweep(params, [('z_tip', np.linspace(1.0, 6.0, 5)), ('Esite', np.linspace(-0.20, -0.05, 5))], selected_params=['VBias','zV0'])
    # plt.savefig('scan_param_sweep.png')
    # plt.show()
    # exit()
    
    # # Initialize pauli solver
    # pauli_solver = pauli.PauliSolver( nSingle=3, nleads=2, verbosity=verbosity )
    # # Create figure
    # fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(15, 10))
    # # Run scans with plotting
    # #scan_xV(params, ax_V2d=ax1, ax_Vtip=ax2, ax_Esite=ax3)
    # scan_xV(params, ax_Vtip=ax1,  ax_V2d=ax2,  ax_I2d=ax3)
    # #scan_xy(params, pauli_solver, ax_Etot=ax4, ax_Ttot=ax5, ax_STM=ax6)    
    # plt.tight_layout()
    # plt.show()

    # Example parameter sweep
    #z_tip_vals = np.linspace(1.0, 3.0, 5)  # Scan 5 tip heights
    #fig = sweep_param_xV(params, [('z_tip', z_tip_vals)])

    #zVd_vals = np.linspace(5.0, 15.0, 5)  # Scan 5 tip heights
    #zV0_vals = np.linspace(-1.0, -3.0, 5)  # Scan 5 tip heights
    #zV0_vals = np.linspace(-3.0, -0.5, 5)  # Scan 5 tip heights
    #fig = sweep_param_xV(params, [('zVd', zVd_vals), ('zV0', zV0_vals)])

    #selected_params=['VBias', 'Esite', 'z_tip', 'zV0', 'zVd' ]
    #sweep_param_xV(params, [('zVd',   np.linspace( 2.0,  16.0, 5))], selected_params=selected_params); plt.savefig('sweep_xV_zVd.png')
    #sweep_param_xV(params, [('zV0',   np.linspace(-5.0, -0.5,  5))], selected_params=selected_params); plt.savefig('sweep_xV_zV0.png')
    #sweep_param_xV(params, [('z_tip', np.linspace( 1.0,  6.0,  5))], selected_params=selected_params); plt.savefig('sweep_xV_ztip.png')


    selected_params=[ 'Esite' ]
    calculate_xV_scan_orb(params, (0,0), (10,10))

    plt.show()
