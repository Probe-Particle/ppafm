#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../../')

import pauli_xv_sweep_cli as xv
import pauli_scan as ps


def build_output_dir(root, geom_name, solver_mode, W):
    root = Path(root)
    sub = root / 'NTCDA' / geom_name / f'solver_{solver_mode}' / f'W_{W:.3f}'
    sub.mkdir(parents=True, exist_ok=True)
    return sub


def run_single_case(config_path, geometry_file, geom_name, solver_mode, W, out_root, nx=100, nV=120, Vmin=0.0, Vmax=None, parallel=False):
    params = xv.load_params(Path(config_path))
    params['geometry_file'] = xv.resolve_geometry_file(Path(geometry_file))
    xv.sync_nsite_with_geometry(params)
    params['solver_mode'] = int(solver_mode)
    params['W'] = float(W)
    params['nx'] = int(nx)
    params['nV'] = int(nV)
    params['Vmin'] = float(Vmin)
    params['Vmax'] = Vmax

    start_point, end_point = xv.DEFAULT_LINE
    arrays = ps.run_xv_scan_case( params, start_point=start_point, end_point=end_point, geometry_file=params['geometry_file'], nx=int(nx), nV=int(nV), Vmin=float(Vmin), Vmax=Vmax if Vmax is None else float(Vmax), parallel=bool(parallel), bCurrentComponents=True,)

    out_dir = build_output_dir(out_root, geom_name, solver_mode, W)
    ps.save_scan_case(out_dir, params, arrays)
    V_slice = params.get('V_slice', params.get('VBias', None))
    x_slice = params.get('x_slice', None)
    ps.export_xv_state_and_current_decomposition_plots(out_dir, params, arrays, V_slice=V_slice, x_slice=x_slice, threshold=1e-12, max_transitions=40)

    STM = arrays['STM']
    dIdV = arrays['dIdV']
    Vbiases = arrays['Vbiases']
    pTips = arrays['pTips']
    distance = np.hypot(pTips[:, 0] - pTips[0, 0], pTips[:, 1] - pTips[0, 1])
    extent = (float(distance.min()), float(distance.max()), float(Vbiases.min()), float(Vbiases.max()))
    if V_slice is None:  V_slice = float(Vbiases[-1])
    if x_slice is None:  x_slice = float(distance[len(distance)//2])
    iv = int(np.argmin(np.abs(Vbiases - V_slice)))
    ix = int(np.argmin(np.abs(distance - x_slice)))
    V_slice = float(Vbiases[iv])
    x_slice = float(distance[ix])

    fig, (ax_stm, ax_didv) = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
    im0 = ax_stm.imshow(STM, origin='lower', aspect='auto', extent=extent, cmap='inferno')
    ax_stm.set_ylabel('V [V]')
    ax_stm.set_title(f'{geom_name}, solver={solver_mode}, W={W:.3f} eV (STM)')
    fig.colorbar(im0, ax=ax_stm, fraction=0.046, pad=0.04)
    # show both cut directions (horizontal=V slice, vertical=x slice)
    ax_stm.axhline(V_slice, color='cyan', lw=1.5, ls='--', zorder=10)
    ax_stm.axvline(x_slice, color='magenta', lw=1.5, ls='--', zorder=10)

    DIDV_CMAP = 'bwr'
    DIDV_OVERSAT = 3.0
    abs_max = float(np.max(np.abs(dIdV)))
    if abs_max == 0.0: abs_max = 1e-9
    vmax = abs_max / DIDV_OVERSAT
    im1 = ax_didv.imshow(dIdV, origin='lower', aspect='auto', extent=extent, cmap=DIDV_CMAP, vmin=-vmax, vmax=vmax)
    ax_didv.set_ylabel('V [V]')
    ax_didv.set_xlabel('Distance [Å]')
    ax_didv.set_title('dI/dV')
    fig.colorbar(im1, ax=ax_didv, fraction=0.046, pad=0.04)
    ax_didv.axhline(V_slice, color='cyan', lw=1.5, ls='--', zorder=10)
    ax_didv.axvline(x_slice, color='magenta', lw=1.5, ls='--', zorder=10)

    fig.suptitle(f"Cuts: V={V_slice:.3f} V, x={x_slice:.3f} Å", fontsize=10)

    fig.tight_layout()
    (out_dir / 'scan.png').parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / 'scan.png', dpi=150)
    plt.close(fig)
    print(f'Wrote results to {out_dir}')

if __name__ == '__main__':
    base = Path(__file__).resolve().parent
    config_path = base / 'example_pauli_params.json'

    geometries = [
        ('2site_Ruslan_long',  base / 'Ruslan_long.txt'),
       # ('2site_Ruslan_short', base / 'Ruslan_short.txt'),
       # ('4site_Ruslan_kite',  base / 'Ruslan_kite.txt'),
    ]
    #solvers  = [0, -1]  # PME, Ground-state
    #W_values = [0.0, 0.02, 0.05, 0.10]

    solvers  = [0]  # PME, Ground-state
    W_values = [0.02]

    out_root = base / 'results'

    for geom_name, geom_file in geometries:
        for solver_mode in solvers:
            for W in W_values:
                print(f'Running {geom_name}, solver={solver_mode}, W={W:.3f}')
                run_single_case( config_path=config_path,geometry_file=geom_file,geom_name=geom_name,solver_mode=solver_mode,W=W,out_root=out_root)