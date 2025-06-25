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


import colormaps

import orbital_utils
from scipy.interpolate import RectBivariateSpline

const_hbar    = 6.582119569e-16  # [ eV*s ]
const_me      = 9.1093837015e-31  # [ kg   ]
#e_charge = 1.602176634e-19   # [ C    ]
eV_A2           =  16.02176634
hbar2_2me       =  const_hbar*const_hbar / (2.0 * const_me )
hbar2_2me_eVA2  = hbar2_2me * eV_A2

print("hbar2_2me      =",hbar2_2me,      "[ eV^2 s^2/kg ]")
print("hbar2_2me_eVA2 =",hbar2_2me_eVA2, "[ eV A^2 ]")

#cmap_dIdV = 'bwr'
#cmap_dIdV = 'PiYG'
cmap_dIdV = 'PiYG_r'
#cmap_dIdV = 'vanimo'
#cmap_dIdV = 'vanimo_inv'

#cmap_STM  = 'hot'
#map_STM  = 'afmhot'
#cmap_STM = 'gnuplot2'
#cmap_STM = 'seismic'
cmap_STM = 'inferno'

# ===========================================
# ============= Utility functions
# ===========================================

def make_site_geom(params):
    """
    Generates ring geometry for charge sites.
    Used by all scanning functions that require multiple sites.
    """
    nsite=params['nsite']
    spos, phis = ut.makeCircle(n=nsite,R=params['radius'],phi0=params['phiRot'])
    spos[:,2]  = params['zQd']
    angles     = phis+params['phi0_ax']
    rots       = ut.makeRotMats( angles )
    return spos, rots, angles

def make_grid_axes(fig, nplots, figsize=(12, 8)):
    """
    Create a near-square grid of subplots for nplots axes.
    rows = ceil(sqrt(nplots)), cols = ceil(nplots/rows).
    Returns flat array of axes of length rows*cols.
    """
    nrows = math.ceil(math.sqrt(nplots))
    ncols = math.ceil(nplots / nrows)
    axs = fig.subplots(nrows, ncols, figsize=figsize)
    # flatten axes array
    return np.array(axs).flatten()

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

# def calculate_tunneling_gauss( pTips, p0=[0.,0.,0.], E0=1.0, Amp=1.0, w=1.0):
#     r2  = (pTips[:,0] - p0[0])**2 + (pTips[:,1] - p0[1])**2 
#     V_gauss = Amp * np.exp(-r2 / (2.0 * w**2))
#     barrier = E0 + V_gauss
#     beta = np.sqrt( barrier / hbar2_2me_eVA2 )
#     r = np.sqrt(r2)
#     outTs = np.exp(-beta * r)
#     return outTs

def generate_hops_gauss( spos, params, pTips=None, bBarrier=True ):
    nsite  = len(spos)
    if pTips is None:
        # Generate grid of tip positions
        npix   = params['npix']
        L      = params['L']
        z_tip  = params['z_tip']
        coords = (np.arange(npix) + 0.5 - npix/2)*(2*L/npix)
        xx, yy = np.meshgrid(coords, coords)
        pTips = np.zeros((npix*npix, 3))
        pTips[:,0] = xx.flatten()
        pTips[:,1] = yy.flatten()
        pTips[:,2] = z_tip + params['Rtip']
        npoints = npix*npix
    else:
        npoints = pTips.shape[0]  # Use actual tip count
    Ts = np.zeros((npoints, nsite))
    # Calculate barrier and beta
    if bBarrier:
        barrier = np.zeros(npoints)  # Correct size
        Amp = params['At']
        E0  = params['Et0']
        w   = params['wt']
        if abs(Amp) > 1e-12:
            # Vectorized calculation for all points
            for i in range(nsite):
                p = spos[i]
                r2 = (pTips[:,0] - p[0])**2 + (pTips[:,1] - p[1])**2 + (pTips[:,2] - p[2])**2 
                barrier[:] += Amp * np.exp(-r2 / (2.0 * w**2))
        barrier += E0
        print( "generate_hops_gauss() barrier (min, max) ", np.min(barrier), np.max(barrier) )
        beta = np.sqrt(barrier / hbar2_2me_eVA2)
    else:
        beta = np.ones(npoints) * params['decay']
    print(     "generate_hops_gauss() beta    (min, max) ", np.min(beta), np.max(beta) )
    # Now compute Ts for each site
    for i in range(nsite):
        p = spos[i]
        r2 = (pTips[:,0] - p[0])**2 + (pTips[:,1] - p[1])**2 + (pTips[:,2] - p[2])**2
        r  = np.sqrt(r2)
        Ts[:,i] = np.exp(-beta * r)
    return Ts, pTips, beta, barrier

# ===========================================
# ============= Plot & Save functions
# ===========================================

def save_scan_results(result_dir, params, scan_params=None, fig=None, summary_file='summary.jsonl'):
    """
    Save scan results to a flat file structure in the specified directory.
    
    Args:
        result_dir: Path to directory where results will be saved
        params: The initial parameters dictionary before scanning, including start/end points if applicable
        scan_params: List of (param_name, values) tuples from the scan
        fig: Matplotlib figure to save
        summary_file: Name of the summary file to append results to
    
    Returns:
        Tuple of (json_path, fig_path) paths to the saved files
    """
    from pathlib import Path
    import json
    from datetime import datetime
    import numpy as np
    
    # Create output directory if needed
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    
    # Prepare the single consolidated parameters dictionary
    combined_params = params.copy()
    if scan_params is not None:
        for param_name, values_list_np in scan_params:
            combined_params[param_name] = values_list_np.tolist() if isinstance(values_list_np, np.ndarray) else values_list_np

    # Save figure
    fig_path = Path(result_dir)/f"scan_{timestamp}.png"
    fig.savefig(fig_path, dpi=300)

    # Prepare output data for both files
    output_data = {
        'timestamp': timestamp,
        'parameters': combined_params,   
    }
    if scan_params is not None:  output_data['scan_parameter_names'] =  [p[0] for p in scan_params]

    # Save main JSON file
    json_path = Path(result_dir)/f"scan_{timestamp}.json"
    with open(json_path, 'w')    as f: json.dump(output_data, f, indent=2)
    if summary_file is not None:
        summary_path = Path(result_dir)/summary_file
        with open(summary_path, 'a') as f: f.write(json.dumps(output_data) + '\n')
    
    print(f"Scan results saved to {json_path} and {fig_path}")
    return json_path, fig_path

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

def plot_1d_scan_results(distance, Es, Ts, STM, nsite, probs=None, stateEs=None, ref_data_line=None, ref_columns=None, fig=None, V_slice=None):
    """Plot results of 1D scan"""
    bProbs = (probs is not None)
    bStateEs = (stateEs is not None)
    nsub   = 3 + bProbs + bStateEs

    if fig is None:
        fig = plt.figure(figsize=(10, 12))
    bRef = (ref_data_line is not None and ref_columns is not None)

    ax_Es = fig.add_subplot(nsub,1,1)
    clrs = ['r', 'g', 'b']
    for i in range(nsite):
        ax_Es.plot(distance, Es[:, i], '-', linewidth=0.5, color=clrs[i], label=f'E_{i+1}')
        if bRef:
            icol = ref_columns[f'Esite_{i+1}']
            ax_Es.plot( ref_data_line[:, 0], ref_data_line[:, icol], ':', color=clrs[i], alpha=0.7, label=f'Ref E_{i+1}')
    ax_Es.set_ylabel('Energy [eV]')
    ax_Es.legend()
    ax_Es.grid(True)

    ax_Ts = fig.add_subplot(nsub,1,2)
    # ax_Ts = fig.add_subplot(nsub,1,2)
    for i in range(nsite):
        ax_Ts.plot(distance, Ts[:, i], '-', linewidth=0.5, color=clrs[i], label=f'T_{i+1}')
        if bRef:
            icol = ref_columns[f'Tsite_{i+1}']
            ax_Ts.plot( ref_data_line[:, 0], ref_data_line[:, icol], ':', color=clrs[i], alpha=0.7, label=f'Ref T_{i+1}' )
    ax_Ts.set_ylabel('Hopping T [a.u.]')
    ax_Ts.legend()
    ax_Ts.grid(True)

    ax_STM = fig.add_subplot(nsub,1,3)
    ax_STM.plot(distance, STM, '.-', color='k', linewidth=0.5, markersize=1.5, label='STM')
    ax_STM.set_ylabel('Current [a.u.]')
    ax_STM.legend()
    ax_STM.grid(True)

    current_subplot_idx = 4
    if bProbs:
        ax_probs = fig.add_subplot(nsub,1,current_subplot_idx)
        nsite = probs.shape[1] # Assuming probs is 2D (npoints, nstates)
        state_order = pauli.make_state_order(nsite)
        labels = pauli.make_state_labels(state_order)
        for idx in range(probs.shape[1]): ax_probs.plot(distance, probs[:,idx], label=labels[idx])
        ax_probs.set_xlabel('Distance'); ax_probs.set_ylabel('Probability')
        ax_probs.legend()
        ax_probs.grid(True)
        current_subplot_idx += 1

    if bStateEs:
        ax_stateEs = fig.add_subplot(nsub,1,current_subplot_idx)
        nsite = stateEs.shape[1] # Assuming stateEs is 2D (npoints, nstates)
        state_order = pauli.make_state_order(nsite)
        labels = pauli.make_state_labels(state_order)
        for idx in range(stateEs.shape[1]):
            ax_stateEs.plot(distance, stateEs[:,idx], label=labels[idx])
        if V_slice is not None:
            ax_stateEs.set_title(f'State Energies and Probabilities along Scan Line (at V={V_slice:.2f}V)')
        ax_stateEs.set_xlabel('Distance'); ax_stateEs.set_ylabel('State Energy [eV]')
        ax_stateEs.legend()
        ax_stateEs.grid(True)

    fig.tight_layout()
    #plt.show()
    return fig

def plot_state_maps(data_3d, extent, axs=None, fig=None, labels=None, aspect='auto', map_type='probability', V_slice=None):
    """
    Plot multiple state maps (probabilities or energies). Handles single or array of axes.
    data_3d: 3D array shape (nV, nx, n_states)
    extent: sequence of 4 [xmin, xmax, ymin, ymax]
    map_type: 'probability' or 'energy'
    """
    # Number of states to plot
    n_states = data_3d.shape[-1]
    # Create default figure/axes if needed
    ncols = min(2, n_states)
    nrows = int(np.ceil(n_states / ncols))
    if fig is None: # Use default figsize if not provided
        fig = plt.figure(figsize=(4 * ncols, 3 * nrows))
    if axs is None:
        axs = fig.subplots(nrows, ncols)

    # Flatten axes into list
    try:
        axs_flat = axs.flatten()
    except AttributeError:
        axs_flat = [axs]
    # Ensure fig is set
    if fig is None and hasattr(axs_flat[0], 'figure'):
        fig = axs_flat[0].figure
    # Plot each state's probability
    cmap = 'viridis' if map_type == 'probability' else 'bwr'
    colorbar_label = 'Probability' if map_type == 'probability' else 'Energy [eV]'
    title_prefix = 'P' if map_type == 'probability' else 'E'

    for idx in range(n_states):
        ax = axs_flat[idx]
        ax.clear()
        title = labels[idx] if labels and idx < len(labels) else f"{title_prefix}{idx}"
        im = ax.imshow(data_3d[:, :, idx], origin='lower', extent=extent, cmap=cmap, interpolation='nearest')
        ax.set_aspect(aspect)
        ax.set_title(title)
        if V_slice is not None:
            ax.axhline(V_slice, color='cyan', linestyle=':', linewidth=1.5, label=f'V={V_slice:.2f}V', zorder=10)
            ax.legend(fontsize='x-small', loc='upper right')
        fig.colorbar(im, ax=ax, label=colorbar_label)
    fig.tight_layout()
    return fig, axs

def plot_column(fig, ncols, col_idx, data1, data2, extent, title='', cmap1=cmap_STM, cmap2=cmap_dIdV, bCbar=False, xlabel='', ylabel='', aspectEqual=False):
    """
    Plot a complete column of two 2D datasets (top and bottom).
    
    Args:
        fig: Figure object
        ncols: Total columns in figure
        col_idx: Column index (0-based)
        data1: 2D data for the top plot
        data2: 2D data for the bottom plot
        extent: Plot extent [xmin, xmax, ymin, ymax]
        title: Title for the column
        cmap1: Colormap for the top plot
        cmap2: Colormap for the bottom plot
        bCbar: Whether to show colorbars
        xlabel: Label for the x-axis
        ylabel: Label for the y-axis
    """
    # Top plot
    ax1 = fig.add_subplot(2, ncols, col_idx + 1)
    im1 = ax1.imshow(data1, extent=extent, aspect='auto',  cmap=cmap1, origin='lower', interpolation='nearest')
    ax1.set(title=f'{title}', xlabel=xlabel)
    if bCbar:
       fig.colorbar(im1, ax=ax1 )
    if aspectEqual:
        ax1.set_aspect('equal')
    
    # Bottom plot
    ax2 = fig.add_subplot(2, ncols, col_idx + ncols + 1)
    vmax = np.max(np.abs(data2))
    im2 = ax2.imshow(data2, extent=extent, aspect='auto',cmap=cmap2, origin='lower',  vmin=-vmax, vmax=vmax, interpolation='nearest')
    ax2.set(title=f'dI/dV {title}', xlabel=xlabel)
    if bCbar:
       fig.colorbar(im2, ax=ax2)
    if aspectEqual:
        ax2.set_aspect('equal')
    
    # Set common ylabel for first column only
    if col_idx == 0:
        ax1.set_ylabel (ylabel)
        ax2.set_ylabel(ylabel)
    
    return ax1, ax2

# ===========================================
# ============= Scan functions
# ===========================================

# ============= Simulate Tip Field ( No Pauli solver simulation )

def scan_tipField_xV(params, ax_xV=None, ax_Esite=None, ax_I2d=None, nx=100, nV=100, ny=100, bLegend=True, scV=1.0, Woffsets=None, pSites=None):
    """
    Simulates tip potential and current for voltage scanning experiments.
    
    Core function for visualizing potential landscapes without Pauli solver.
    Used by CombinedChargeRingsGUI_v5.py for basic potential simulations.
    For Pauli solver simulations, see calculate_xV_scan() functions.
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
            pauli.validate_probabilities(probs)
            #pu.plot_imshow(ax_I2d, Ts, title="Current", extent=[-L, L, 0.0, VBias], ylabel="V [V]", cmap='hot')
            #I = Ts
        else:
            Ts = pauli.evalSitesTipsTunneling(pTips_v, pSites=pSites, beta=params['decay'], Amp=1.0).reshape(nV, nx)
            I = Ts*0.0
            for i, W in enumerate(Woffsets):
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
    current, Es, Ts, probs, stateEs = pauli_solver.scan_current_tip( pTips, Vtips, spos,  cpp_params, order, cs, state_order, rots=rots, bOmp=False, bMakeArrays=True, return_state_energies=True )
    pauli.validate_probabilities(probs)
    if ax_probs:
        axp = ax_probs
        for i in range(probs.shape[1]): axp.plot(distance, probs[:,i], label=f"P{i}")
        axp.legend()
    return distance, Es, Ts, current, x, y, x1, y1, x2, y2, probs, stateEs    

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
    STM, Es, Ts, probs, stateEs = pauli.run_pauli_scan_top(spos, rots, params, pauli_solver=pauli_solver, bOmp=bOmp, return_state_energies=True)
    pauli.validate_probabilities(probs)
    #print( "min,max Es", np.min(Es), np.max(Es))
    #print( "min,max Ts", np.min(Ts), np.max(Ts))
    #print( "min,max STM", np.min(STM), np.max(STM))
    dIdV = None
    if ax_dIdV is not None:
        params_ = params.copy()
        dQ = params.get('dQ', 0.05)
        params_['VBias'] += dQ
        STM_2, Es_2, Ts_2, probs_2, stateEs_2 = pauli.run_pauli_scan_top(spos, rots, params_, pauli_solver=pauli_solver, bOmp=bOmp, return_state_energies=True)       
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
    stateEs = stateEs.reshape(params['npix'], params['npix'], -1)
    return STM, dIdV, Es, Ts, probs, spos, rots

def interpolate_hopping_maps(T1, T2, c=1.0, T0=1.0):
    # Combine tunneling models
    if T2 is None: 
        print("interpolate_hopping_maps(): Warrning: No T2 provided, returning T1")
        return T1
    max1 = T1.max()
    max2 = T2.max()
    print( "interpolate_hopping_maps() max1, max2: ", max1, max2, " c: ", c, " T0: ", T0)
    return T0 * ( (c/max2)*T2 + ((1-c)/max1)*T1 )


def scan_xy_orb(params, orbital_2D=None, orbital_lvec=None, pauli_solver=None, ax_Etot=None, ax_Ttot=None, ax_STM=None, ax_Ms=None, ax_rho=None, ax_dIdV=None, decay=None, bOmp=False, Tmin=0.0, EW=2.0, sdIdV=1.0, fig_probs=None, bdIdV=False):
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

    #print( " type(rots) ",  type(rots) )
    # big-to-small hopping computation
    #dcanv=params['L']/params['npix']

    #T1=time.perf_counter()

    # Use gaussian tunneling model with parameters from GUI
    Ts_gauss, _, _, _ = generate_hops_gauss( spos, params)

    c_orb = params['c_orb']  # Default to 1.0 if not specified
    #Ts_orb = Ts_gauss
    Ts_orb = None
    if orbital_2D is not None:
        dcanv=2*L/npix
        big_npix=int(params.get('big_npix',400))
        Ms,rho=generate_central_hops(orbital_2D,orbital_lvec,spos[:,:2],angles,z_tip,dcanv,big_npix,npix,decay=decay or params.get('decay',0.2))
        print("calculate_xV_scan_orb() Ms (min, max) ", np.min(Ms), np.max(Ms))
        Ts_orb = np.zeros((npix*npix, nsite), dtype=np.float64)
        for i in range(nsite): Ts_orb[:,i]=np.abs(Ms[i].flatten()) #**2
    #Ts_flat = Ts_orb

    Ts_flat = interpolate_hopping_maps(Ts_gauss, Ts_orb, c=c_orb, T0=params['T0'])
    
    # Create solver if not provided
    if pauli_solver is None:
        pauli_solver = pauli.PauliSolver(nSingle=nsite, nleads=2)
    pauli.set_valid_point_cuts(Tmin, EW)
    
    #T2 = time.perf_counter(); print("Time(scan_xy_orb.2 Ts,PauliSolver)",  T2-T1 )     
    #bOmp = True
    STM_flat, Es_flat, Ts_flat_, probs, stateEs_flat = pauli.run_pauli_scan_top(spos, rots, params, pauli_solver=pauli_solver, Ts=Ts_flat, bOmp=bOmp)
    pauli.validate_probabilities(probs)
    #STM_flat, Es_flat, _ = pauli.run_pauli_scan_top(spos, rots, params, pauli_solver=pauli_solver )
    dIdV = None
    if ax_dIdV is not None: bdIdV=True
    if bdIdV:
        params_ = params.copy()
        dQ = params.get('dQ', 0.01)
        params_['VBias'] += dQ
        STM_2, _, _, probs, _ = pauli.run_pauli_scan_top(spos, rots, params_, pauli_solver=pauli_solver, Ts=Ts_flat, bOmp=bOmp)       
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
    stateEs = stateEs_flat.reshape(npix, npix, -1)

    T_calc = time.perf_counter(); print(f"scan_xy_orb() calc time: {T_calc-T0:.5f} [s]")

    if ax_Etot is not None:                     pu.plot_imshow(ax_Etot, Etot, title="Energies max(eps)",  extent=extent, cmap='bwr', )
    if ax_rho  is not None and rho is not None: pu.plot_imshow(ax_rho,  rho,  title="sum(Wf)",            extent=extent, cmap='bwr')
    if ax_Ttot is not None:                     pu.plot_imshow(ax_Ttot, Ttot, title="Tunneling sum(M^2)", extent=extent, cmap=cmap_STM)
    if ax_STM  is not None:                     pu.plot_imshow(ax_STM,  STM,  title="STM",                extent=extent, cmap=cmap_STM)
    if ax_dIdV is not None:
        dIdV = dIdV.reshape(npix, npix)
        pu.plot_imshow(ax_dIdV, dIdV, title="dI/dV", extent=extent, cmap=cmap_dIdV, scV=sdIdV)
    if ax_Ms   is not None and Ms is not None: 
        for i in range(nsite):
            ax_Ms[i].imshow(Ms[i], cmap=cmap_STM, origin='lower', extent=extent)
            ax_Ms[i].set_title(f"Hopping matrix {i}")
    if fig_probs is not None:
        # plot site probabilities in separate window
        n_states = probs.shape[2]
        state_order = pauli.make_state_order(params['nsite'])
        labels = pauli.make_state_labels(state_order)
        axs_all = make_grid_axes(fig_probs, n_states)
        plot_state_probabilities(probs, extent=extent, axs=axs_all[:n_states], fig=fig_probs, labels=labels, aspect='equal')
    #T4 = time.perf_counter(); print("Time(scan_xy_orb.4 plotting)",  T4-T3 )        
    return STM, dIdV, Es, Ts, probs, stateEs, spos, rots

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
    current, Es, Ts, probs, stateEs = pauli.run_pauli_scan_xV( pTips, Vbiases, spos,  cpp_params, order=1, cs=None, rots=rots, bOmp=bOmp, state_order=state_order, Ts=None, pauli_solver=pauli_solver, return_state_energies=True )
    pauli.validate_probabilities(probs)
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


def calculate_xV_scan_orb(params, start_point, end_point, orbital_2D=None, orbital_lvec=None, pauli_solver=None, bOmp=False, ax_Emax=None, ax_STM=None, ax_dIdV=None, nx=100, nV=100, Vmin=0.0, Vmax=None, bLegend=True, sdIdV=0.5, decay=None, fig_probs=None, fig_energies=None, V_slice=None, ax_current_components=None):
    """
    Voltage scan along a line with orbital-based tunneling calculations.
    
    Orbital-aware version of calculate_xV_scan(). Uses external orbital data if provided,
    otherwise falls back to internal tunneling calculations.
    Key function used in fit_sim_exp_general.py and CombinedChargeRingsGUI_v5.py.
    """
    T0 = time.perf_counter()
    # Line geometry
    x1, y1 = start_point; x2, y2 = end_point
    dist = np.hypot(x2-x1, y2-y1)
    npts = nx
    t = np.linspace(0,1,npts)
    x = x1 + (x2-x1)*t; y = y1 + (y2-y1)*t
    if Vmax is None: Vmax = params['VBias']
    Vbiases = np.linspace(Vmin, Vmax, nV)

    # Prepare tip positions
    pTips = np.zeros((npts,3))
    pTips[:,0] = x; pTips[:,1] = y; pTips[:,2] = params['z_tip']+params['Rtip']

    # Site & rotation
    nsite = int(params['nsite'])
    # Site geometry (positions, rotations, angles)
    spos, rots, angles = make_site_geom(params)
    # Compute hopping Ts along line from orbital data
    c_orb = params['c_orb']  # Default to 1.0 if not specified
    print("calculate_xV_scan_orb() c_orb: ", c_orb)
    Ts_orb = None
    if orbital_2D is not None:
        L = params['L']; npix = params['npix']
        dcanv = 2*L/npix
        big_npix = int(params.get('big_npix',400))
        Ms, _ = generate_central_hops(orbital_2D, orbital_lvec, spos[:,:2], angles, params['z_tip']+params['Rtip'], dcanv, big_npix, npix, decay=decay or params.get('decay',0.2))
        print("calculate_xV_scan_orb() Ms (min, max) ", np.min(Ms), np.max(Ms))
        # Prepare grid for interpolation
        coords = (np.arange(npix) + 0.5 - npix/2)*dcanv
        Ts_line = np.zeros((npts, nsite))
        for i in range(nsite):
            Ts_map = np.abs(Ms[i]) #**2
            interp = RectBivariateSpline(coords, coords, Ts_map)
            Ts_line[:,i] = interp(y, x, grid=False)
        Ts_orb = Ts_line
    
    # Compute Gaussian tunneling
    Ts_gauss, _, _, _ = generate_hops_gauss(spos, params, pTips=pTips, bBarrier=True)
    Ts_input = interpolate_hopping_maps(Ts_gauss, Ts_orb, c=c_orb, T0=params['T0'])
    
    # # Combine tunneling models
    # if Ts_orb is not None and c_orb > 1e-9:
    #     max1 = Ts_orb.max()
    #     max2 = Ts_gauss.max()
    #     T0 = params['T0']
    #     Ts_input = T0 * ( (c_orb/max1) * Ts_orb + ((1 - c_orb)/max2) * Ts_gauss )
    #     #Ts_input = c_orb * Ts_orb + (1-c_orb) * Ts_gauss
    #     print(f"Using combined tunneling model with c_orb={c_orb:.2f}")
    # else:
    #     Ts_input = Ts_gauss
    #     print("No orbital used, falling back to Gaussian tunneling")


    state_order = pauli.make_state_order(nsite)
    nstate = len(state_order)
    # allocate buffer for current components: flat array [nV*npts, nstate*nstate]
    current_matrix_flat = np.zeros((nV * npts, nstate*nstate), dtype=float)
    pauli.set_current_matrix_export_pointer(current_matrix_flat)
    current, Es, Ts, probs, stateEs = pauli.run_pauli_scan_xV( pTips, Vbiases, spos, params, order=1, cs=None, rots=rots, state_order=state_order, Ts=Ts_input, bOmp=bOmp, pauli_solver=pauli_solver)
    pauli.validate_probabilities(probs) # Validate probabilities
    # reshape and compute
    STM = current.reshape(nV, npts)
    Es  = Es.reshape(nV, npts, nsite)
    stateEs = stateEs.reshape(nV, npts, -1)
    Ts  = Ts.reshape(nV, npts, nsite)
    stateEs = stateEs.reshape(nV, npts, -1)
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
        pu.plot_imshow(ax_STM, STM, title='STM', extent=extent, cmap=cmap_STM)
        ax_STM.set_aspect('auto');
        if bLegend: ax_STM.set_ylabel('V [V]')
        if V_slice is not None:
            ax_STM.axhline(V_slice, color='cyan', linestyle=':', linewidth=1.5, label=f'V={V_slice:.2f}V', zorder=10)
            ax_STM.legend(fontsize='x-small', loc='upper right')
    if ax_dIdV is not None:
        pu.plot_imshow(ax_dIdV, dIdV, title='dI/dV', extent=extent, cmap=cmap_dIdV, scV=sdIdV)
        ax_dIdV.set_aspect('auto');
        if bLegend: ax_dIdV.set_ylabel('V [V]')
        if V_slice is not None:
            ax_dIdV.axhline(V_slice, color='cyan', linestyle=':', linewidth=1.5, label=f'V={V_slice:.2f}V', zorder=10)
            ax_dIdV.legend(fontsize='x-small', loc='upper right')

    probs = probs.reshape(nV, nx, -1)
    state_order_labels = pauli.make_state_order(params['nsite'])
    labels = pauli.make_state_labels(state_order_labels)

    if fig_probs is not None:
        plot_state_maps(probs, extent=[0,dist,Vmin,Vmax], fig=fig_probs, labels=labels, map_type='probability', V_slice=V_slice)

    if fig_energies is not None:
        plot_state_maps(stateEs, extent=[0,dist,Vmin,Vmax], fig=fig_energies, labels=labels, map_type='energy', V_slice=V_slice)

    # Plot individual current components if requested
    # reshape flat buffer to [nV, npts, nstate, nstate]
    flat = current_matrix_flat.reshape((nV, npts, nstate, nstate))
    # find closest voltage slice index
    iv = np.argmin(np.abs(Vbiases - V_slice))
    # extract and transpose to [nstates, nstates, npts]
    slice_cm = flat[iv]  # shape (npts, nstate, nstate)
    cm = slice_cm.transpose((1,2,0))
    if ax_current_components is not None:
        plot_current_components(cm, ax_current_components, x, V_slice, labels)
    # Prepare 1D currents and components for return
    currents_1d = STM[iv, :]
    current_components = cm

    return STM, dIdV, Es, Ts, probs, stateEs, x, Vbiases, spos, rots, current_components, currents_1d

def plot_current_components(current_matrix, ax, distance, V_slice, state_labels):
    """
    Plots the current contribution matrix for a given point in the scan.
    """
    ax.clear()
    nstates = current_matrix.shape[1]

    # Plot the contribution of each state transition
    for i in range(nstates):
        for j in range(nstates):
            if np.max(np.abs(current_matrix[i, j])) > 1e-12:  # Only plot significant contributions
                ax.plot(distance, current_matrix[i, j], label=f'{state_labels[i]} -> {state_labels[j]}')

    ax.set_xlabel('Distance [Å]')
    ax.set_ylabel('Current Contribution')
    ax.set_title(f'Current Contributions at V={V_slice:.2f}V')
    ax.legend(loc='upper right', fontsize='x-small')
    ax.grid(True)



def plot_state_scan_1d(distance, stateEs, probs, nsite, currents=None, current_components=None, fig=None, prob_scale=200.0, V_slice=None, alpha_prob=0.25, comp_thresh=1e-12):
    """
    Plots many-body state energies and their probabilities along a 1D scan line.
    Energies are plotted as lines, and probabilities are represented by the size of scatter points on top.
    """
    if fig is None:
        fig = plt.figure(figsize=(12, 8))
    if currents is not None and current_components is not None:
        # Two-panel plot: component stackplot above, state energies below
        fig = plt.figure(figsize=(12, 8)) if fig is None else fig
        ax1 = fig.add_subplot(2, 1, 1)
        # Stackplot of components
        nstates = current_components.shape[0]
        comp_list = []
        comp_labels = []
        for i in range(nstates):
            for j in range(nstates):
                comp = current_components[i, j]
                if np.max(np.abs(comp)) > comp_thresh:
                    comp_list.append(comp)
                    comp_labels.append(f'{i}->{j}')
        if comp_list:
            ax1.stackplot(distance, *comp_list, labels=comp_labels)
        ax1.set_title(f'Current Components (V={V_slice:.2f}V)')
        ax1.set_xlabel('Distance [Å]'); ax1.set_ylabel('Current')
        ax1.legend(loc='upper right', fontsize='x-small')
        ax1.grid(True)
        # Bottom panel: state energies and probabilities
        ax = fig.add_subplot(2, 1, 2)
        # Plot energies and probs
        n_states = stateEs.shape[1]
        state_order = pauli.make_state_order(nsite)
        labels = pauli.make_state_labels(state_order)
        # Reuse style mapping
        state_sytels={
            '000': ('gray', '-'), '111': ('gray', '-'),
            '100': ('b', '--'), '010': ('b', '-'), '001': ('b', '-'),
            '011': ('r', '--'), '101': ('r', '-'), '110': ('r', '-'),
        }
        colors = [ state_sytels[label][0] for label in labels]
        styles = [ state_sytels[label][1] for label in labels]
        for i in range(n_states):
            prob_mask = probs[:, i] > 1e-3
            if np.any(prob_mask):
                ax.scatter(distance[prob_mask], stateEs[prob_mask, i], s=probs[prob_mask, i] * prob_scale, color=colors[i], alpha=alpha_prob, edgecolors='none')
            ax.plot(distance, stateEs[:, i], ls=styles[i], color=colors[i], label=labels[i], lw=1.0)
        ax.set_xlabel('Distance [Å]')
        ax.set_ylabel('Many-Body State Energy [eV]')
        ax.set_title(f'State Energies and Probabilities along Scan Line (V={V_slice:.2f}V)')
        ax.legend(title='States', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
        fig.tight_layout(rect=[0, 0, 0.85, 1])
        return fig


    ax = fig.add_subplot(1, 1, 1)
    n_states = stateEs.shape[1]
    state_order = pauli.make_state_order(nsite)
    labels = pauli.make_state_labels(state_order)

    state_sytels={
      # label, color, line style  
      '000': ('gray', '-'),
      '111': ('gray', '-'),

      '100': ('b', '--'),
      '010': ('b', '-'),
      '001': ('b', '-'),

      '011': ('r', '--'),
      '101': ('r', '-'),
      '110': ('r', '-'),

    }
    colors = [ state_sytels[label][0] for label in labels]
    styles = [ state_sytels[label][1] for label in labels]

    #colors = plt.cm.jet(np.linspace(0, 1, n_states))
    for i in range(n_states):
        prob_mask = probs[:, i] > 1e-3
        if np.any(prob_mask):
            ax.scatter(distance[prob_mask], stateEs[prob_mask, i], s=probs[prob_mask, i] * prob_scale, color=colors[i], alpha=alpha_prob, edgecolors='none')
            #ax.scatter(distance[prob_mask], stateEs[prob_mask, i], s=probs[prob_mask, i] * prob_scale, color=colors[i], alpha=0.3, edgecolors=None, linewidth=0.5)
        ax.plot(distance, stateEs[:, i], ls=styles[i], color=colors[i], label=labels[i], lw=1.0)
       
    ax.set_xlabel('Distance [Å]')
    ax.set_ylabel('Many-Body State Energy [eV]')
    ax.set_title(f'State Energies and Probabilities along Scan Line (V={V_slice:.2f}V)')
    ax.legend(title='States', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    return fig

# ===========================================
# ============= Sweep functions
# ===========================================

def sweep_param_tipField_xV(params, scan_params, selected_params=None, nx=100, nV=100, sz=3, bLegend=False):
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
        scan_tipField_xV(params, ax_xV=ax_xV, ax_Esite=ax_Esite, nx=nx, nV=nV, bLegend=bLegend)
        
        # Build title showing all swept parameters
        title_parts = [f"{name}={params[name]:.3f}" for name in param_names]
        title = ", ".join(title_parts)
        ax_xV   .set_title(title, fontsize=10)
        ax_Esite.set_title(title, fontsize=10)
    
    plt.tight_layout()
    return fig

def sweep_scan_param_pauli_xy_orb_old(params, scan_params, selected_params=None, orbital_2D=None, orbital_lvec=None, orbital_file=None, pauli_solver=None, bOmp=False, Tmin=0.0, EW=2.0, sdIdV=0.5, fig=None):
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
        STM_flat, Es_flat, Ts_flat_, probs_ = pauli.run_pauli_scan_top(params['spos'], params['rots'], params, pauli_solver=pauli_solver, Ts=Ts_flat, bOmp=bOmp)       
        pauli.validate_probabilities(probs_)
        dIdV = None
        if ax_dIdV is not None:
            params_ = params.copy()
            params_['VBias'] += 0.05
            STM_2, _, _, probs = pauli.run_pauli_scan_top(params['spos'], params['rots'], params_, pauli_solver=pauli_solver, Ts=Ts_flat, bOmp=bOmp)
            pauli.validate_probabilities(probs)
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

def sweep_scan_param_pauli_xy_orb(params, scan_params, view_params=None, 
                                 orbital_2D=None, orbital_lvec=None, 
                                orbital_file=None, pauli_solver=None, bOmp=False, sdIdV=0.5, ExpRef=None, fig=None,
                                result_dir=None):
    """
    Scan parameter values in xy plane (distance vs distance) using scan_xy_orb.
    Generates nscan columns of plots with specified layout, with optional experimental reference.
    
    Args:
        params: Dictionary of parameters
        scan_params: List of (param_name, values) tuples to sweep
        selected_params: List of other parameters to show in figure title
        orbital_2D: 2D orbital data
        orbital_lvec: Lattice vectors for the orbital
        orbital_file: Path to orbital file to load if orbital_2D and orbital_lvec not provided
        pauli_solver: Optional pauli solver instance
        bOmp: Whether to use OpenMP for calculations
        sdIdV: Scaling factor for dI/dV plots
        ExpRef: Optional experimental reference data
        fig: Optional figure to plot on
        result_dir: Path to directory where results will be saved in flat structure
    
    Returns:
        Tuple of (fig, all_results) containing the figure and all results
    """
    from pathlib import Path
    import json
    from datetime import datetime
    
    # Process orbital data if provided
    if orbital_file is not None and orbital_2D is None:
        print(f"Loading orbital file: {orbital_file}")
        orbital_data, orbital_lvec = orbital_utils.load_orbital(orbital_file)
        # Process orbital data
        orbital_2D = np.transpose(orbital_data, (2, 1, 0))
        orbital_2D = np.sum(orbital_2D[:, :, orbital_2D.shape[2]//2:], axis=2)
        orbital_2D = np.ascontiguousarray(orbital_2D, dtype=np.float64)

    if not scan_params:
        raise ValueError("scan_params cannot be empty")
    
    param_names = [p[0] for p in scan_params]
    values_list = [p[1] for p in scan_params]
    nscan = len(values_list[0])
    
    # Validate all value lists have same length
    for vals in values_list[1:]:
        assert len(vals) == nscan, "All value lists must have same length"
    
    # Create solver if not provided
    if pauli_solver is None:
        nsite = int(params['nsite'])
        pauli_solver = pauli.PauliSolver(nSingle=nsite, nleads=2)
    
    # Create figure if not provided
    if fig is None:
        # If we have experimental reference, add one more column for it
        ncols = nscan + (1 if ExpRef is not None else 0)
        fig = plt.figure(figsize=(5*ncols, 10))
    
    # Build figure title with selected parameters
    
    if view_params: 
        title = "params "
        title += " ".join([f"{k}: {params[k]:.4g}" for k in view_params if k in params])
        fig.suptitle(title, fontsize=12)
    
    # Set up plot grid - 2 rows (STM, dIdV), nscan+1 columns if ExpRef provided
    nrows, ncols = 2, nscan + (1 if ExpRef is not None else 0)
    
    # Plot experimental reference if provided
    col_offset = 0
    if ExpRef is not None:
        # Extract experimental data
        exp_STM = ExpRef['STM']
        exp_dIdV = ExpRef['dIdV']
        exp_x = ExpRef['x']
        exp_y = ExpRef['y']
        exp_extent = [min(exp_x), max(exp_x), min(exp_y), max(exp_y)]
        plot_column(fig, ncols, 0, exp_STM, exp_dIdV, exp_extent, title='Experimental', xlabel='x (Å)', ylabel='y (Å)', aspectEqual=True)
        col_offset = 1
    
    # Initialize results collection
    all_results = []
    
    # Process each parameter value
    for i in range(nscan):
        # Update all parameter values for this iteration
        params_i = params.copy()
        for param, vals in scan_params:
            params_i[param] = vals[i]
        
        # Store parameters for this run
        run_params = params_i.copy()
        run_params.update({ 'scan_index': i, 'scan_params': {param: vals[i] for param, vals in scan_params} })

        # Run xy scan with orbital data
        STM, dIdV, Es, Ts, probs, spos, rots = scan_xy_orb(
            params_i,
            orbital_2D=orbital_2D, orbital_lvec=orbital_lvec,
            pauli_solver=pauli_solver, bOmp=bOmp,
            sdIdV=sdIdV, fig_probs=None, bdIdV=True
        )

        # Collect results for return
        all_results.append({
            'STM': STM, 'dIdV': dIdV, 'Es': Es, 'Ts': Ts, 'probs': probs,
            'x': np.linspace(-params_i['L']/2, params_i['L']/2, params_i['npix']), 
            'y': np.linspace(-params_i['L']/2, params_i['L']/2, params_i['npix']),
            'params': run_params,
            'spos': spos,
            'rots': rots
        })

        # Plot results
        col_title  = " ".join([f"{param}: {vals[i]:.4g}" for param, vals in scan_params])
        sim_extent = [-params_i['L']/2, params_i['L']/2, -params_i['L']/2, params_i['L']/2]
        plot_column(fig, ncols, i + col_offset, STM, dIdV, sim_extent, title=col_title, xlabel='x (Å)', ylabel='y (Å)', aspectEqual=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
    
    # Save results using the modular function
    if result_dir:
        save_scan_results( result_dir=result_dir, params=params, scan_params=scan_params, fig=fig )
    
    return fig, all_results  # Return results data


def sweep_scan_param_pauli_xV_orb(params, scan_params, view_params=None, 
                                selected_params=None, orbital_2D=None, orbital_lvec=None, 
                                orbital_file=None, nx=100, nV=100, Vmin=0.0, Vmax=None, 
                                pauli_solver=None, bOmp=False, sdIdV=0.5, ExpRef=None, fig=None,
                                result_dir=None):
    """
    Scan parameter values in xV plane (distance vs voltage) using calculate_xV_scan_orb.
    Generates nscan columns of plots with specified layout, with optional experimental reference.
    
    Args:
        params: Dictionary of parameters
        scan_params: List of (param_name, values) tuples to sweep
        selected_params: List of other parameters to show in figure title
        orbital_2D: 2D orbital data
        orbital_lvec: Lattice vectors for the orbital
        orbital_file: Path to orbital file to load if orbital_2D and orbital_lvec not provided
        nx: Number of x points
        nV: Number of voltage points
        Vmin: Minimum voltage
        Vmax: Maximum voltage (default: params['VBias'])
        pauli_solver: Optional pauli solver instance
        bOmp: Whether to use OpenMP for calculations
        sdIdV: Scaling factor for dI/dV plots
        ExpRef: Optional experimental reference data
        fig: Optional figure to plot on
        result_dir: Path to directory where results will be saved in flat structure
    
    Returns:
        Tuple of (fig, all_results) containing the figure and all results
    """
    from pathlib import Path
    import json
    from datetime import datetime
    
    # Process orbital data if provided
    if orbital_file is not None and orbital_2D is None:
        print(f"Loading orbital file: {orbital_file}")
        orbital_data, orbital_lvec = orbital_utils.load_orbital(orbital_file)
        # Process orbital data
        orbital_2D = np.transpose(orbital_data, (2, 1, 0))
        orbital_2D = np.sum(orbital_2D[:, :, orbital_2D.shape[2]//2:], axis=2)
        orbital_2D = np.ascontiguousarray(orbital_2D, dtype=np.float64)

    if not scan_params:
        raise ValueError("scan_params cannot be empty")
    
    param_names = [p[0] for p in scan_params]
    values_list = [p[1] for p in scan_params]
    nscan = len(values_list[0])
    
    # Validate all value lists have same length
    for vals in values_list[1:]:
        assert len(vals) == nscan, "All value lists must have same length"
    
    # Set up voltage range
    if Vmax is None:
        Vmax = params['VBias']
    
    # Create solver if not provided
    if pauli_solver is None:
        nsite = int(params['nsite'])
        pauli_solver = pauli.PauliSolver(nSingle=nsite, nleads=2)
    
    # Create figure if not provided
    if fig is None:
        # If we have experimental reference, add one more column for it
        ncols = nscan + (1 if ExpRef is not None else 0)
        fig = plt.figure(figsize=(5*ncols, 10))
    
    # Build figure title with selected parameters
    
    if view_params: 
        title = "params "
        title += " ".join([f"{k}: {params[k]:.4g}" for k in view_params if k in params])
        fig.suptitle(title, fontsize=12)
    
    # Set up plot grid - 2 rows (STM, dIdV), nscan+1 columns if ExpRef provided
    nrows, ncols = 2, nscan + (1 if ExpRef is not None else 0)
    
    # Plot experimental reference if provided
    col_offset = 0
    if ExpRef is not None:
        # Extract experimental data
        exp_STM = ExpRef['STM']
        exp_dIdV = ExpRef['dIdV']
        exp_x = ExpRef['x']
        exp_voltages = ExpRef['voltages']
        exp_extent = [0, max(exp_x), min(exp_voltages), max(exp_voltages)]
        plot_column(fig, ncols, 0, exp_STM, exp_dIdV, exp_extent, title='Experimental', xlabel='Distance (Å)', ylabel='Voltage (V)')
        col_offset = 1
    
    # Initialize results collection
    all_results = []
    
    # Process each parameter value
    for i in range(nscan):
        # Update all parameter values for this iteration
        params_i = params.copy()
        for param, vals in scan_params:
            params_i[param] = vals[i]
        
        # Store parameters for this run
        run_params = params_i.copy()
        run_params.update({ 'scan_index': i, 'scan_params': {param: vals[i] for param, vals in scan_params} })

        x1, y1 = run_params['p1_x'], run_params['p1_y']
        x2, y2 = run_params['p2_x'], run_params['p2_y']
        start_point = (x1, y1)
        end_point   = (x2, y2)
        dist = np.hypot(x2-x1, y2-y1)
                
        # Run xV scan with orbital data
        STM, dIdV, Es, Ts, probs, x, voltages, spos, rots = calculate_xV_scan_orb(
            params_i, start_point, end_point,
            orbital_2D=orbital_2D, orbital_lvec=orbital_lvec,
            pauli_solver=pauli_solver, bOmp=bOmp,
            ax_Emax=None, ax_STM=None, ax_dIdV=None,  # Don't plot in the function
            nx=nx, nV=nV, Vmin=Vmin, Vmax=Vmax,
            sdIdV=sdIdV, fig_probs=None
        )
        all_results.append({ 'parameters': run_params, 'timestamp': datetime.now().isoformat() })
        col_title  = " ".join([f"{param}: {vals[i]:.4g}" for param, vals in scan_params])
        sim_extent = [0, dist, Vmin, Vmax]
        plot_column(fig, ncols, i + col_offset, STM, dIdV, sim_extent, title=col_title, xlabel='Distance (Å)', ylabel='Voltage (V)')
    
    # Set consistent voltage limits across all plots
    for ax in fig.get_axes():
        ax.set_ylim(Vmin, Vmax)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
    
    # Save results using the modular function
    if result_dir:
        params_ = params.copy()
        params_['start_point'] = start_point
        params_['end_point'  ] = end_point
        save_scan_results( result_dir=result_dir, params=params_, scan_params=scan_params, fig=fig )
    
    return fig, all_results  # Return results data


if __name__ == "__main__":


    # mpl.rcParams.update({
    #     # Font sizes
    #     'font.size': 8,                # Default font size
    #     'axes.titlesize': 8,           # Axes title size
    #     'axes.labelsize': 8,           # Axes labels (x and y)
    #     'xtick.labelsize': 7,          # x-axis tick labels
    #     'ytick.labelsize': 7,          # y-axis tick labels
    #     'legend.fontsize': 7,          # Legend font size
    #     'figure.titlesize': 9,         # Figure title (suptitle)
    #     'figure.labelsize': 8,         # Figure labels (for colorbars, etc.)
        
    #     # Margins and spacing
    #     'figure.autolayout': False,    # We'll control layout manually
    #     'figure.subplot.left': 0.08,   # Left margin
    #     'figure.subplot.right': 0.95,  # Right margin
    #     'figure.subplot.bottom': 0.08, # Bottom margin
    #     'figure.subplot.top': 0.90,    # Top margin
    #     'figure.subplot.wspace': 0.05,  # Horizontal space between subplots
    #     'figure.subplot.hspace': 0.05,  # Vertical space between subplots
        
    #     # Lines and markers
    #     'lines.linewidth': 0.8,        # Line width
    #     'lines.markersize': 3,         # Marker size
    #     'patch.linewidth': 0.6,        # Patch (bars, etc.) edge width
        
    #     # Legend
    #     'legend.frameon': False,       # Remove legend frame
    #     'legend.borderpad': 0.1,       # Padding inside legend border
    #     'legend.labelspacing': 0.1,    # Vertical space between legend entries
    #     'legend.handlelength': 1.5,    # Length of legend handles
    #     'legend.handletextpad': 0.4,   # Space between handle and text
        
    #     # Axes
    #     'axes.linewidth': 0.6,         # Axis line width
    #     'axes.labelpad': 2,            # Space between label and axis
    #     'xtick.major.pad': 1.5,        # Padding between x-ticks and label
    #     'ytick.major.pad': 1.5,        # Padding between y-ticks and label
    #     'xtick.major.size': 2,         # x-tick length
    #     'ytick.major.size': 2,         # y-tick length
    #     'xtick.major.width': 0.6,      # x-tick width
    #     'ytick.major.width': 0.6,      # y-tick width
        
    #     # Saving figures
    #     'savefig.bbox': 'tight',       # Remove extra whitespace around figure
    #     'savefig.pad_inches': 0.02,    # Padding when bbox='tight'
    #     #'savefig.dpi': 300,            # Higher DPI for quality when small
    # })


    # Example usage when run as standalone script - using same defaults as GUI
    params = {
       # Geometry
        'nsite': 3,
        'radius': 5.2,
        'phiRot': 1.3,
        'phi0_ax': 0.2,

        # Electrostatic Field
        'VBias': 1.0,
        'V_slice': 0.6,
        'Rtip': 3.0,
        'z_tip': 5.0,
        'zV0': -1.0,
        'zVd': 15.0,
        'zQd': 0.0,
        'Q0': 1.0,
        'Qzz': 10.0,

        # Transport Solver
        'Esite': -0.100,
        'W': 0.05,
        'Temp': 3.00,
        'decay': 0.3,
        'GammaS': 0.01,
        'GammaT': 0.01,
        'onSiteCoulomb': 3.0, # Explicitly included, matches GUI's internal default

        # Barrier
        'Et0': 0.2,
        'wt': 8.0,
        'At': -0.1,
        'c_orb': 1.0,
        'T0': 1.0,

        # Visualization
        'L': 20.0,
        'npix': 200,
        'dQ': 0.02,

        # Data Cuts (simulation end-points for 1D scan)
        'p1_x': 9.72,
        'p1_y': -9.96,
        'p2_x': -11.0,
        'p2_y': 12.0,

        # Data Cuts (Experimental end-points for 1D scan)
        'exp_slice': 10,
        'ep1_x': 9.72,
        'ep1_y': -6.96,
        'ep2_x': -11.0,
        'ep2_y': 15.0,
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
    # #scan_tipField_xV(params, ax_V2d=ax1, ax_Vtip=ax2, ax_Esite=ax3)
    # scan_tipField_xV(params, ax_Vtip=ax1,  ax_V2d=ax2,  ax_I2d=ax3)
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




    # ---------- From Here

    # Define start and end points for the scan from params, mirroring GUI logic
    start_point = (params['p1_x'], params['p1_y'])
    end_point   = (params['p2_x'], params['p2_y'])
    kBoltz = 8.617333262e-5 # eV/K
    # Define Vmax for the scan, mirroring GUI's default logic
    Vmax_scan = params['VBias']
    V_slice_scan = params.get('V_slice', 0.5)  # Initialize Pauli solver once, mirroring GUI logic (moved from previous position)
    pauli_solver = pauli.PauliSolver(nSingle=params['nsite'], nleads=2, verbosity=verbosity)
    
    # Set lead temperatures and check_prob_stop, mirroring GUI logic
    T_eV = params['Temp'] * kBoltz
    pauli_solver.set_lead(0, 0.0, T_eV) # for lead 0 (substrate)
    pauli_solver.set_lead(1, 0.0, T_eV) # for lead 1 (tip)
    # Set check_prob_stop and bValidateProbabilities as in GUI
    pauli_solver.set_check_prob_stop(bCheckProb=False, bCheckProbStop=False, CheckProbTol=1e-12)
    pauli.bValidateProbabilities = False # Set global flag for probability validation

    # Create figures for main plots, probabilities, and state energies
    fig, (ax_Emax, ax_STM, ax_dIdV) = plt.subplots(1, 3, figsize=(15, 5))
    fig_probs = plt.figure(figsize=(12, 8)) # Separate figure for probabilities
    fig_energies = plt.figure(figsize=(12, 8)) # Separate figure for state energies
    # Separate figure for current component slice
    fig_curr = plt.figure(figsize=(12, 6))
    ax_current = fig_curr.add_subplot(1,1,1)

    STM, dIdV, Es, Ts, probs, stateEs, x, Vbiases, spos, rots, curr_comps, curr_1d = calculate_xV_scan_orb(params, start_point, end_point, ax_Emax=ax_Emax, ax_STM=ax_STM, ax_dIdV=ax_dIdV, Vmax=Vmax_scan, V_slice=V_slice_scan, ax_current_components=ax_current, pauli_solver=pauli_solver, fig_probs=fig_probs, fig_energies=fig_energies)
    fig.tight_layout() # Adjust layout to prevent overlapping titles/labels
    fig_curr.tight_layout() # Adjust layout for current components figure

    # Create a separate figure for the 1D line plot of many-body states
    fig_1d_states = plt.figure()
    # Take a slice at the desired bias voltage
    idx = np.argmin(np.abs(Vbiases - V_slice_scan))
    stateEs_1d = stateEs[idx, :, :]
    probs_1d   = probs[idx, :, :]
    plot_state_scan_1d(x, stateEs_1d, probs_1d, params['nsite'], currents=curr_1d, current_components=curr_comps, fig=fig_1d_states, V_slice=V_slice_scan)

    print("HERE - DONE, show()")
    plt.show()
