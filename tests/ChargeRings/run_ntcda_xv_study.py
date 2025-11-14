#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../../')

import pauli_xv_sweep_cli as xv


def build_output_dir(root, geom_name, solver_mode, W):
    root = Path(root)
    sub = root / 'NTCDA' / geom_name / f'solver_{solver_mode}' / f'W_{W:.3f}'
    sub.mkdir(parents=True, exist_ok=True)
    return sub


def run_single_case(config_path, geometry_file, geom_name, solver_mode, W, out_root,
                     nx=100, nV=120, Vmin=0.0, Vmax=None, parallel=False):
    params = xv.load_params(Path(config_path))
    params['geometry_file'] = xv.resolve_geometry_file(Path(geometry_file))
    xv.sync_nsite_with_geometry(params)
    params['solver_mode'] = int(solver_mode)
    params['W'] = float(W)

    start_point, end_point = xv.DEFAULT_LINE

    STM, dIdV, Vbiases, distance, extent = xv.run_single_scan(
        params,
        start_point=start_point,
        end_point=end_point,
        nx=int(nx),
        nV=int(nV),
        Vmin=float(Vmin),
        Vmax=Vmax if Vmax is None else float(Vmax),
        parallel=bool(parallel),
    )

    out_dir = build_output_dir(out_root, geom_name, solver_mode, W)

    # Save NPZ
    npz_path = out_dir / 'scan.npz'
    np.savez_compressed(
        npz_path,
        STM=STM,
        dIdV=dIdV,
        Vbiases=Vbiases,
        distance=distance,
        extent=np.array(extent),
    )

    # Save params JSON (including solver/W and geometry)
    params_out = params.copy()
    params_out['solver_mode'] = int(solver_mode)
    params_out['W'] = float(W)
    params_out['geometry_file'] = str(params['geometry_file'])
    params_out['nx'] = int(nx)
    params_out['nV'] = int(nV)
    params_out['Vmin'] = float(Vmin)
    params_out['Vmax'] = Vmax
    json_path = out_dir / 'params.json'
    with json_path.open('w') as f:
        json.dump(params_out, f, indent=2)

    # Simple plot: STM and dIdV vs distance,V
    fig, (ax_stm, ax_didv) = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
    im0 = ax_stm.imshow(
        STM,
        origin='lower',
        aspect='auto',
        extent=extent,
        cmap='inferno',
    )
    ax_stm.set_ylabel('V [V]')
    ax_stm.set_title(f'{geom_name}, solver={solver_mode}, W={W:.3f} eV (STM)')
    fig.colorbar(im0, ax=ax_stm, fraction=0.046, pad=0.04)

    vmax = float(np.max(np.abs(dIdV)))
    if vmax == 0.0:
        vmax = 1e-9
    im1 = ax_didv.imshow(
        dIdV,
        origin='lower',
        aspect='auto',
        extent=extent,
        cmap='PiYG_r',
        vmin=-vmax,
        vmax=vmax,
    )
    ax_didv.set_ylabel('V [V]')
    ax_didv.set_xlabel('Distance [Å]')
    ax_didv.set_title('dI/dV')
    fig.colorbar(im1, ax=ax_didv, fraction=0.046, pad=0.04)

    fig.tight_layout()
    png_path = out_dir / 'scan.png'
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    print(f'Wrote results to {out_dir}')


def main():
    base = Path(__file__).resolve().parent
    config_path = base / 'example_pauli_params.json'

    geometries = [
        ('2site_Ruslan_long',  base / 'Ruslan_long.txt'),
        ('2site_Ruslan_short', base / 'Ruslan_short.txt'),
        ('4site_Ruslan_kite',  base / 'Ruslan_kite.txt'),
    ]
    solvers = [0, -1]  # PME, Ground-state
    W_values = [0.0, 0.02, 0.05, 0.10]

    out_root = base / 'results'

    for geom_name, geom_file in geometries:
        for solver_mode in solvers:
            for W in W_values:
                print(f'Running {geom_name}, solver={solver_mode}, W={W:.3f}')
                run_single_case(
                    config_path=config_path,
                    geometry_file=geom_file,
                    geom_name=geom_name,
                    solver_mode=solver_mode,
                    W=W,
                    out_root=out_root,
                )


if __name__ == '__main__':
    main()
