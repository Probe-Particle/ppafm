#!/usr/bin/env python3
"""
Quick XY scan collector: saves flat-named outputs into a single folder.

Example filename pattern:
  Ruslan_kite_solver0_W20meV_V1200meV.png
  Ruslan_kite_solver0_W20meV_V1200meV.json
  Ruslan_kite_solver0_W20meV_V1200meV.npz

Defaults: only PNG, no dIdV (pass --compute-didv to include), single-threaded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../../")
import pauli_xy_cli  # noqa: E402
import pauli_scan  # noqa: E402


def fmt_meV(val_ev: float) -> int:
    """Return rounded meV integer from eV value."""
    return int(round(1000.0 * float(val_ev)))

def build_params(args: argparse.Namespace) -> Dict[str, object]:
    # Start from defaults provided by pauli_xy_cli, then override
    params = pauli_xy_cli.DEFAULT_PARAMS.copy()

    if args.config:
        with args.config.open() as fp: cfg = json.load(fp)
        params.update(cfg)

    if args.geometry:
        params["geometry_file"] = pauli_xy_cli.resolve_geometry_file(args.geometry)
        pauli_xy_cli.sync_nsite_with_geometry(params)
    elif "geometry_file" in params:
        params["geometry_file"] = pauli_xy_cli.resolve_geometry_file(Path(params["geometry_file"]))
        pauli_xy_cli.sync_nsite_with_geometry(params)

    if args.solver_mode is not None: params["solver_mode"] = int(args.solver_mode)
    if args.W           is not None: params["W"] = float(args.W)
    if args.decay       is not None: params["decay"] = float(args.decay)
    if args.z_tip       is not None: params["z_tip"] = float(args.z_tip)
    if args.npix        is not None: params["npix"] = int(args.npix)

    # Force Qzz = 0 (no quadrupole)
    params["Qzz"] = 0.0

    # If no external geometry, install rectangle geometry into params
    if args.geometry is None:
        z_plane = float(params.get("zQd", 0.0))
        Esite = float(params.get("Esite", 0.0))
        params["rect_sites"] = make_rect_geometry(float(args.rect_x), float(args.rect_y), z_plane, Esite)
        params["nsite"] = 4

    # Anisotropic couplings (two dimers along x, vertical y, diagonals)
    Wx = float(args.W_x)
    Wy = float(args.W_y)
    Wd = float(args.W_diag)
    params["W_x"] = Wx
    params["W_y"] = Wy
    params["W_diag"] = Wd
    params["W"] = max(params.get("W", 0.0), Wx, Wy, Wd)  # keep scalar for reference
    Wij = np.zeros((4, 4), dtype=np.float64)
    # site order: [ x,  y], [-x,  y], [ x, -y], [-x, -y]
    Wij[0, 1] = Wij[1, 0] = Wx
    Wij[2, 3] = Wij[3, 2] = Wx
    Wij[0, 2] = Wij[2, 0] = Wy
    Wij[1, 3] = Wij[3, 1] = Wy
    Wij[0, 3] = Wij[3, 0] = Wd
    Wij[1, 2] = Wij[2, 1] = Wd
    params["Wij_matrix"] = Wij

    return params


def bias_grid(start: float, stop: float, step: float | None, n: int | None) -> Iterable[float]:
    if step is not None: return np.arange(start, stop + 0.5 * step, step)
    if n is None: n = 5
    return np.linspace(start, stop, n)

def make_rect_geometry(x: float, y: float, z: float, E: float) -> np.ndarray:
    """Return 4-site rectangle geometry array (x,y,-x,y,x,-y,-x,-y) with given z and onsite E."""
    # columns: x,y,z,E
    return np.array([
        [ x,  y, z, E],
        [-x,  y, z, E],
        [ x, -y, z, E],
        [-x, -y, z, E],
    ], dtype=np.float64)

def save_png(path: Path, STM: np.ndarray, dIdV: np.ndarray | None, L: float) -> None:
    fig, axes = plt.subplots(1, 2 if dIdV is not None else 1, figsize=(8, 4))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    extent = [-L, L, -L, L]

    im0 = axes[0].imshow(STM, origin="lower", extent=extent, cmap="hot")
    axes[0].set_title("STM")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    if dIdV is not None:
        abs_max = float(np.max(np.abs(dIdV)))
        vmax = abs_max / 3.0 if abs_max > 0 else 1e-9
        im1 = axes[1].imshow(dIdV, origin="lower", extent=extent, cmap="bwr", vmin=-vmax, vmax=vmax)
        axes[1].set_title("dI/dV")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def _jsonable_params(params: Dict[str, object]) -> Dict[str, object]:
    out = {}
    for k, v in params.items():
        if isinstance(v, Path):
            out[k] = str(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out

def save_montage(path: Path, stms: list[np.ndarray], didvs: list[np.ndarray | None], L: float, V_list: list[float]) -> None:
    nv = len(stms)
    has_didv = any(d is not None for d in didvs)
    nrows = 2 if has_didv else 1
    fig, axes = plt.subplots(nrows, nv, figsize=(3*nv, 3*nrows), squeeze=False)
    extent = [-L, L, -L, L]
    for i in range(nv):
        ax0 = axes[0, i]
        im0 = ax0.imshow(stms[i], origin="lower", extent=extent, cmap="hot")
        ax0.set_title(f"STM @ {V_list[i]:.3f} V", fontsize=8)
        ax0.set_xticks([]); ax0.set_yticks([])
        fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
        if has_didv:
            ax1 = axes[1, i]
            d = didvs[i]
            if d is None:
                ax1.axis("off")
            else:
                abs_max = float(np.max(np.abs(d)))
                vmax = abs_max / 3.0 if abs_max > 0 else 1e-9
                im1 = ax1.imshow(d, origin="lower", extent=extent, cmap="bwr", vmin=-vmax, vmax=vmax)
                ax1.set_title("dI/dV", fontsize=8)
                ax1.set_xticks([]); ax1.set_yticks([])
                fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Quick flat XY scan collector (single folder outputs).")
    ap.add_argument("--config",       type=Path, help="JSON params file (optional).")
    ap.add_argument("--geometry",     type=Path, help="Geometry file (optional, overrides config).")
    ap.add_argument("--solver-mode",  type=int, default=None, help="Solver mode (e.g., 0, -1).")
    ap.add_argument("--W",            type=float, default=0.02, help="W (eV, legacy scalar).")
    ap.add_argument("--W-x",          dest="W_x", type=float, default=0.03, help="Dimer coupling along +x/-x (eV).")
    ap.add_argument("--W-y",          dest="W_y", type=float, default=0.01, help="Coupling along y (vertical) (eV).")
    ap.add_argument("--W-diag",       dest="W_diag", type=float, default=0.0, help="Diagonal coupling (eV).")
    ap.add_argument("--decay",        type=float, default=None, help="decay (overrides params).")
    ap.add_argument("--z_tip",        type=float, default=None, help="tip height z (overrides params).")
    ap.add_argument("--rect-x",       type=float, default=6.0, help="Rectangle half-width x (Å).")
    ap.add_argument("--rect-y",       type=float, default=5.0, help="Rectangle half-height y (Å).")
    ap.add_argument("--npix",         type=int, default=None, help="Grid size (overrides params).")
    ap.add_argument("--out-dir",      type=Path, default=Path("results_quick"), help="Output directory.")
    ap.add_argument("--geom-name",    type=str, default=None, help="Name used in filenames (default: geometry stem or 'geom').")
    ap.add_argument("--Vstart",       type=float, default=0.5, help="Start VBias (eV).")
    ap.add_argument("--Vend",         type=float, default=1.2, help="End VBias (eV).")
    ap.add_argument("--Vstep",        type=float, default=None, help="VBias step (eV).")
    ap.add_argument("--nV",           type=int, default=10, help="Number of VBias subdivisions (linspace).")
    ap.add_argument("--compute-didv", type=int, default=1, help="Compute dI/dV via finite difference.")
    ap.add_argument("--save-json",    type=int, default=1, help="Also save JSON metadata.")
    ap.add_argument("--save-npz",     type=int, default=0, help="Also save NPZ arrays.")
    ap.add_argument("--parallel",     type=int, default=1, help="Enable OpenMP in solver.")
    ap.add_argument("--montage",      type=int, default=2, help="Save a montage (STM[/dIdV]) over all V into one PNG.")
    args = ap.parse_args()

    params = build_params(args)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    geom_name = args.geom_name
    if geom_name is None:
        if "geometry_file" in params:
            geom_name = Path(params["geometry_file"]).stem
        else:
            geom_name = "rect"

    W_meV = fmt_meV(params["W"])
    V_list = list(bias_grid(args.Vstart, args.Vend, args.Vstep, args.nV))
    stms_all: list[np.ndarray] = []
    didvs_all: list[np.ndarray | None] = []

    for V in V_list:
        params_run = params.copy()
        params_run["VBias"] = float(V)
        results, meta = pauli_xy_cli.run_xy_scan(params_run, parallel=args.parallel, compute_didv=args.compute_didv)
        STM = results["STM"]
        dIdV = results.get("dIdV")
        stms_all.append(STM)
        didvs_all.append(dIdV)

        if args.montage != 2:
            V_meV = fmt_meV(V)
            Wx_meV = fmt_meV(params["W_x"])
            Wy_meV = fmt_meV(params["W_y"])
            Wd_meV = fmt_meV(params["W_diag"])
            stem = (
                f"{geom_name}_solver{int(params_run['solver_mode'])}"
                f"_Wx{Wx_meV}meV_Wy{Wy_meV}meV_Wd{Wd_meV}meV_V{V_meV}meV"
            )
            png_path = out_dir / f"{stem}.png"
            save_png(png_path, STM, dIdV, params_run["L"])
            print(f"[xy_quickflat] saved PNG  -> {png_path}")

            if args.save_json:
                json_path = out_dir / f"{stem}.json"
                payload = {
                    "VBias": V,
                    "params": _jsonable_params(params_run),
                    "STM_shape": list(np.shape(STM)),
                    "dIdV_shape": list(np.shape(dIdV)) if dIdV is not None else None,
                }
                json_path.write_text(json.dumps(payload, indent=2))
                print(f"[xy_quickflat] saved JSON -> {json_path}")

            if args.save_npz:
                npz_path = out_dir / f"{stem}.npz"
                params_json=json.dumps({k: (str(v) if isinstance(v, Path) else v) for k, v in params_run.items()})
                np.savez_compressed( npz_path,  STM=STM, dIdV=dIdV if dIdV is not None else np.array([]),  Es=results.get("Es"),Ts=results.get("Ts"), spos=meta.get("spos"),rots=meta.get("rots"),VBias=V,params_json=params_json)
                print(f"[xy_quickflat] saved NPZ  -> {npz_path}")

    if args.montage:
        m_path = out_dir / f"{geom_name}_solver{int(params['solver_mode'])}_W{W_meV}meV_montage.png"
        save_montage(m_path, stms_all, didvs_all, params["L"], V_list)
        print(f"[xy_quickflat] saved montage -> {m_path}")
        if args.save_json:
            m_json = out_dir / f"{geom_name}_solver{int(params['solver_mode'])}_W{W_meV}meV_montage.json"
            payload = {
                "VBias_list": list(V_list),
                "params": _jsonable_params(params),
                "STM_shape": list(np.shape(stms_all[0])) if stms_all else None,
                "dIdV_shape": list(np.shape(didvs_all[0])) if didvs_all and didvs_all[0] is not None else None,
            }
            m_json.write_text(json.dumps(payload, indent=2))
            print(f"[xy_quickflat] saved montage JSON -> {m_json}")