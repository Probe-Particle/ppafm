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
    #if args.W           is not None: params["W"] = float(args.W)
    if args.decay       is not None: params["decay"] = float(args.decay)
    if args.z_tip       is not None: params["z_tip"] = float(args.z_tip)
    if args.L           is not None: params["L"   ]  = float(args.L)
    if args.npix        is not None: params["npix"]  = int(args.npix)

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

def save_montage(path: Path, stms: list[np.ndarray], didvs: list[np.ndarray | None], L: float, V_list: list[float], params: Dict[str, object], caption: str = "") -> None:
    nv = len(stms)
    has_didv = any(d is not None for d in didvs)
    nrows = 2 if has_didv else 1
    fig, axes = plt.subplots(nrows, nv, figsize=(3*nv, 3*nrows), squeeze=False)
    extent = [-L, L, -L, L]
    for i in range(nv):
        ax0 = axes[0, i]
        im0 = ax0.imshow(stms[i], origin="lower", extent=extent, cmap="hot")
        ax0.set_title(f"V={V_list[i]:.3f} V", fontsize=8)
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
    if caption:
        fig.suptitle(caption, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_and_plot_xv(start_point: np.ndarray, end_point: np.ndarray, label: str, params: Dict[str, object], out_dir: Path, geom_name: str, Wx_meV: int, Wy_meV: int, Wd_meV: int, dx: float, dy: float, caption_base: str, Vmin: float = 0.0, Vmax: float = 2.0, nx: int = 200, nV: int = 200, parallel: bool = False) -> None:
    """Run an xV cut and save STM/dIdV vs distance,V."""
    arrays = pauli_scan.run_xv_scan_case(
        params,
        start_point=start_point,
        end_point=end_point,
        geometry_file=params.get("geometry_file"),
        nx=int(nx),
        nV=int(nV),
        Vmin=float(Vmin),
        Vmax=float(Vmax),
        parallel=bool(parallel),
        bCurrentComponents=False,
    )
    STM = arrays["STM"]
    dIdV = arrays["dIdV"]
    Vbiases = arrays["Vbiases"]
    pTips = arrays["pTips"]
    # distance along path
    distance = np.hypot(pTips[:, 0] - pTips[0, 0], pTips[:, 1] - pTips[0, 1])
    extent = (float(distance.min()), float(distance.max()), float(Vbiases.min()), float(Vbiases.max()))

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
    im0 = ax0.imshow(STM, origin="lower", aspect="auto", extent=extent, cmap="inferno")
    ax0.set_ylabel("V [V]")
    ax0.set_title("STM")
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    abs_max = float(np.max(np.abs(dIdV)))
    if abs_max == 0.0:
        abs_max = 1e-9
    vmax = abs_max / 3.0
    im1 = ax1.imshow(dIdV, origin="lower", aspect="auto", extent=extent, cmap="bwr", vmin=-vmax, vmax=vmax)
    ax1.set_ylabel("V [V]")
    ax1.set_xlabel("Distance [Å]")
    ax1.set_title("dI/dV")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    fig.suptitle(f"{label} {caption_base}", fontsize=9)
    fig.tight_layout()

    stem = (
        f"{geom_name}_solver{int(params['solver_mode'])}"
        f"_Wx{Wx_meV}meV_Wy{Wy_meV}meV_Wd{Wd_meV}meV_dx{dx:.2f}_dy{dy:.2f}_{label}_xV"
    )
    png_path = out_dir / f"{stem}.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"[xy_rect] saved xV cut ({label}) -> {png_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Quick flat XY scan collector (single folder outputs).")
    ap.add_argument("--config",       type=Path, help="JSON params file (optional).")
    ap.add_argument("--geometry",     type=Path, help="Geometry file (optional, overrides config).")
    ap.add_argument("--solver-mode",  type=int, default=None, help="Solver mode (e.g., 0, -1).")
    #ap.add_argument("--W",            type=float, default=0.02, help="W (eV, legacy scalar).")
    ap.add_argument("--W-x",          dest="W_x",    type=float, default=0.020, help="Dimer coupling along +x/-x (eV).")
    ap.add_argument("--W-y",          dest="W_y",    type=float, default=0.005, help="Coupling along y (vertical) (eV).")
    ap.add_argument("--W-diag",       dest="W_diag", type=float, default=0.0, help="Diagonal coupling (eV).")
    ap.add_argument("--decay",        type=float, default=None, help="decay (overrides params).")
    ap.add_argument("--z_tip",        type=float, default=None, help="tip height z (overrides params).")
    ap.add_argument("--rect-x",       type=float, default=6.0, help="Rectangle half-width x (Å).")
    ap.add_argument("--rect-y",       type=float, default=5.0, help="Rectangle half-height y (Å).")
    ap.add_argument("--L",            type=float, default=10.0, help="Scan window L (Å).")
    ap.add_argument("--npix",         type=int, default=None, help="Grid size (overrides params).")
    ap.add_argument("--out-dir",      type=Path, default=Path("results_quick"), help="Output directory.")
    ap.add_argument("--geom-name",    type=str, default=None, help="Name used in filenames (default: geometry stem or 'geom').")
    ap.add_argument("--Vstart",       type=float, default=0.7, help="Start VBias (eV).")
    ap.add_argument("--Vend",         type=float, default=1.2, help="End VBias (eV).")
    ap.add_argument("--Vstep",        type=float, default=None, help="VBias step (eV).")
    ap.add_argument("--nV",           type=int, default=20, help="Number of VBias subdivisions (linspace).")
    ap.add_argument("--compute-didv", type=int, default=1, help="Compute dI/dV via finite difference.")
    ap.add_argument("--save-json",    type=int, default=1, help="Also save JSON metadata.")
    ap.add_argument("--save-npz",     type=int, default=0, help="Also save NPZ arrays.")
    ap.add_argument("--parallel",     type=int, default=0, help="Enable OpenMP in solver.")
    ap.add_argument("--montage",      type=int, default=2, help="Save a montage (STM[/dIdV]) over all V into one PNG.")
    ap.add_argument("--xv-vmax",      type=float, default=2.0, help="Vmax for xV cuts (eV).")
    ap.add_argument("--xv-vmin",      type=float, default=0.2, help="Vmin for xV cuts (eV).")
    ap.add_argument("--xv-nx",        type=int, default=200, help="Number of spatial samples for xV cuts.")
    ap.add_argument("--xv-nV",        type=int, default=200, help="Number of voltage samples for xV cuts.")
    ap.add_argument("--tip-orb",      nargs=4, type=float, default=[0.1, 0.1, 1.0, 0.0], help="Tip orbital components [px, py, pz, s].")
    ap.add_argument("--tip-orb-power",type=int, default=None, help="Power for angular factor (tipOrb_power).")
    ap.add_argument("--tip-orb-abs",  type=int, default=None, help="Use absolute value in angular factor (tipOrb_abs: 1/0).")
    args = ap.parse_args()

    params = build_params(args)
    # tip orbital asymmetry (px/py/pz mix), matches example_pauli_params.json
    if args.tip_orb is not None:
        params["tipOrb"] = list(args.tip_orb)
    if args.tip_orb_power is not None:
        params["tipOrb_power"] = int(args.tip_orb_power)
    if args.tip_orb_abs is not None:
        params["tipOrb_abs"] = bool(args.tip_orb_abs)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    geom_name = args.geom_name
    if geom_name is None:
        if "geometry_file" in params:
            geom_name = Path(params["geometry_file"]).stem
        else:
            geom_name = "rect"

    W_meV = fmt_meV(params["W"])
    Wx_meV = fmt_meV(params["W_x"])
    Wy_meV = fmt_meV(params["W_y"])
    Wd_meV = fmt_meV(params["W_diag"])
    V_list = list(bias_grid(args.Vstart, args.Vend, args.Vstep, args.nV))
    stms_all: list[np.ndarray] = []
    didvs_all: list[np.ndarray | None] = []

    if "rect_sites" in params:
        dx = float(params["rect_sites"][0, 0])
        dy = float(params["rect_sites"][0, 1])
    else:
        dx = float(getattr(args, "rect_x", 0.0))
        dy = float(getattr(args, "rect_y", 0.0))
    caption_parts = [
        f"Wx:{params['W_x']:.3f}",
        f"Wy:{params['W_y']:.3f}",
        f"Wd:{params['W_diag']:.3f}",
        f"dx:{dx:.2f}",
        f"dy:{dy:.2f}",
        f"V:{args.Vstart:.2f}-{args.Vend:.2f}",
        f"z:{params.get('z_tip', 0.0):.2f}",
    ]
    if params.get("decay") is not None:
        caption_parts.append(f"decay:{params['decay']:.3f}")
    caption_base = " ".join(caption_parts)

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
        m_path = out_dir / (
            f"{geom_name}_solver{int(params['solver_mode'])}"
            f"_Wx{Wx_meV}meV_Wy{Wy_meV}meV_Wd{Wd_meV}meV_dx{dx:.2f}_dy{dy:.2f}_montage.png"
        )
        save_montage(m_path, stms_all, didvs_all, params["L"], V_list, params, caption=caption_base)
        print(f"[xy_quickflat] saved montage -> {m_path}")
        if args.save_json:
            m_json = out_dir / (
                f"{geom_name}_solver{int(params['solver_mode'])}"
                f"_Wx{Wx_meV}meV_Wy{Wy_meV}meV_Wd{Wd_meV}meV_dx{dx:.2f}_dy{dy:.2f}_montage.json"
            )
            payload = {
                "VBias_list": list(V_list),
                "params": _jsonable_params(params),
                "STM_shape": list(np.shape(stms_all[0])) if stms_all else None,
                "dIdV_shape": list(np.shape(didvs_all[0])) if didvs_all and didvs_all[0] is not None else None,
            }
            m_json.write_text(json.dumps(payload, indent=2))
            print(f"[xy_quickflat] saved montage JSON -> {m_json}")

    # xV cuts: four lines (center x/y, site x/y), spanning full window [-L,+L]
    Lspan = float(params["L"])
    Vmax_xv = float(args.xv_vmax)
    Vmin_xv = float(args.xv_vmin)
    nx_xv = int(args.xv_nx)
    nV_xv = int(args.xv_nV)
    # through center
    run_and_plot_xv(np.array([-Lspan, 0.0]), np.array([ Lspan, 0.0]), "center_x", params, out_dir, geom_name, Wx_meV, Wy_meV, Wd_meV, dx, dy, caption_base, Vmin=Vmin_xv, Vmax=Vmax_xv, nx=nx_xv, nV=nV_xv, parallel=bool(args.parallel))
    run_and_plot_xv(np.array([0.0, -Lspan]), np.array([0.0,  Lspan]), "center_y", params, out_dir, geom_name, Wx_meV, Wy_meV, Wd_meV, dx, dy, caption_base, Vmin=Vmin_xv, Vmax=Vmax_xv, nx=nx_xv, nV=nV_xv, parallel=bool(args.parallel))
    # through first site
    run_and_plot_xv(np.array([-Lspan, dy]), np.array([ Lspan, dy]), "site_x", params, out_dir, geom_name, Wx_meV, Wy_meV, Wd_meV, dx, dy, caption_base, Vmin=Vmin_xv, Vmax=Vmax_xv, nx=nx_xv, nV=nV_xv, parallel=bool(args.parallel))
    run_and_plot_xv(np.array([dx, -Lspan]), np.array([dx,  Lspan]), "site_y", params, out_dir, geom_name, Wx_meV, Wy_meV, Wd_meV, dx, dy, caption_base, Vmin=Vmin_xv, Vmax=Vmax_xv, nx=nx_xv, nV=nV_xv, parallel=bool(args.parallel))