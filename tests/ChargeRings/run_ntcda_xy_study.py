#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../../')

import pauli_xy_cli as xy
import pauli_scan as ps


def build_output_dir(root, geom_name, solver_mode, W, Vbias):
    root = Path(root)
    sub = root / 'NTCDA_xy' / geom_name / f'solver_{solver_mode}' / f'W_{W:.3f}' / f'VBias_{Vbias:.2f}'
    sub.mkdir(parents=True, exist_ok=True)
    return sub


def load_params(config_path: Path) -> dict:
    params = xy.DEFAULT_PARAMS.copy()
    if config_path.is_file():
        with config_path.open() as fp:  cfg = json.load(fp)
        params.update(cfg)
    return params


def run_single_case(config_path, geometry_file, geom_name, solver_mode, W, Vbias, out_root, parallel=False, compute_didv=True):
    params = load_params(Path(config_path))
    params['geometry_file'] = xy.resolve_geometry_file(Path(geometry_file))
    xy.sync_nsite_with_geometry(params)
    params['solver_mode'] = int(solver_mode)
    params['W'] = float(W)
    params['VBias'] = float(Vbias)

    arrays = ps.run_xy_scan_case(params, geometry_file=params['geometry_file'], parallel=bool(parallel), compute_didv=bool(compute_didv))

    out_dir = build_output_dir(out_root, geom_name, solver_mode, W, Vbias)
    ps.save_scan_case(out_dir, params, arrays)

    STM = arrays['STM']
    dIdV = arrays.get('dIdV')

    L = float(params.get('L', 20.0))
    extent = np.array([-L, L, -L, L], dtype=float)

    fig, axes = plt.subplots(1, 2 if dIdV is not None else 1, figsize=(8, 4))
    if not isinstance(axes, np.ndarray):  axes = np.array([axes])

    ax0 = axes[0]
    im0 = ax0.imshow(STM, origin='lower', extent=extent, cmap='inferno')
    ax0.set_xlabel('x [Å]')
    ax0.set_ylabel('y [Å]')
    ax0.set_title(f'STM, V={Vbias:.2f} V, W={W:.3f} eV')
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    if dIdV is not None:
        DIDV_CMAP = 'bwr'
        DIDV_OVERSAT = 3.0
        abs_max = float(np.max(np.abs(dIdV)))
        if abs_max == 0.0: abs_max = 1e-9
        vmax = abs_max / DIDV_OVERSAT
        ax1 = axes[1]
        im1 = ax1.imshow(dIdV, origin='lower', extent=extent, cmap=DIDV_CMAP, vmin=-vmax, vmax=vmax)
        ax1.set_xlabel('x [Å]')
        ax1.set_ylabel('y [Å]')
        ax1.set_title('dI/dV')
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_dir / 'scan.png', dpi=150)
    plt.close(fig)
    print(f'Wrote xy results to {out_dir}')


if __name__ == '__main__':
    base = Path(__file__).resolve().parent
    config_path = base / 'example_pauli_params.json'

    geometries = [
        ('2site_Ruslan_long',  base / 'Ruslan_long.txt'),
        ('2site_Ruslan_short', base / 'Ruslan_short.txt'),
        ('4site_Ruslan_kite',  base / 'Ruslan_kite.txt'),
    ]
    solvers = [0, -1]  # PME, Ground-state
    W_values = [0.02, 0.05]
    Vbias_values = [0.5 + 0.1 * i for i in range(11)]  # 0.5 .. 1.5 step 0.1

    out_root = base / 'results'

    for geom_name, geom_file in geometries:
        for solver_mode in solvers:
            for W in W_values:
                for Vbias in Vbias_values:
                    print(f'Running xy {geom_name}, solver={solver_mode}, W={W:.3f}, V={Vbias:.2f}')
                    run_single_case( config_path=config_path, geometry_file=geom_file, geom_name=geom_name, solver_mode=solver_mode, W=W, Vbias=Vbias, out_root=out_root)