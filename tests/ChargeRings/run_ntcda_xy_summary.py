#!/usr/bin/env python3
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def load_W_dirs_xy(solver_dir: Path):
    if not solver_dir.is_dir():
        return []
    w_dirs = []
    for d in sorted(solver_dir.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if not name.startswith("W_"):
            continue
        try:
            w_val = float(name.split("_")[1])
        except (IndexError, ValueError):
            continue
        w_dirs.append((w_val, d))
    w_dirs.sort(key=lambda x: x[0])
    return w_dirs


def load_V_dirs(W_dir: Path):
    if not W_dir.is_dir():
        return []
    v_dirs = []
    for d in sorted(W_dir.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if not name.startswith("VBias_"):
            continue
        try:
            v_val = float(name.split("_")[1])
        except (IndexError, ValueError):
            continue
        npz_path = d / "scan.npz"
        if npz_path.is_file():
            v_dirs.append((v_val, d, npz_path))
    v_dirs.sort(key=lambda x: x[0])
    return v_dirs


def make_summary_for_solver_and_W(geom_label: str, solver_mode: int, W: float, W_dir: Path):
    v_dirs = load_V_dirs(W_dir)
    if not v_dirs:
        print(f"[xy-summary] No VBias directories for {geom_label}, solver={solver_mode}, W={W:.3f} at {W_dir}")
        return

    STMs = []
    dIdVs = []
    exts = []
    VBs = []

    for VBias, vdir, npz_path in v_dirs:
        data = np.load(npz_path)
        STM = data["STM"]
        dIdV = data.get("dIdV")
        if dIdV is None:
            continue
        extent = tuple(data["extent"].tolist()) if "extent" in data else None
        STMs.append(STM)
        dIdVs.append(dIdV)
        exts.append(extent)
        VBs.append(VBias)

    if not STMs or not dIdVs:
        print(f"[xy-summary] No STM/dIdV arrays found for {geom_label}, solver={solver_mode}, W={W:.3f}")
        return

    stm_min = min(stm.min() for stm in STMs)
    stm_max = max(stm.max() for stm in STMs)
    didv_min = min(d.min() for d in dIdVs)
    didv_max = max(d.max() for d in dIdVs)
    didv_abs = max(abs(didv_min), abs(didv_max))

    ncols = len(VBs)
    fig, axes = plt.subplots(2, ncols, figsize=(3.8 * ncols, 6.0), squeeze=False)

    for idx, (VBias, STM, dIdV, extent) in enumerate(zip(VBs, STMs, dIdVs, exts)):
        ax_stm = axes[0, idx]
        ax_didv = axes[1, idx]

        im0 = ax_stm.imshow(
            STM,
            origin="lower",
            aspect="equal",
            extent=extent,
            cmap="inferno",
            vmin=stm_min,
            vmax=stm_max,
        )
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-didv_abs, vmax=didv_abs)
        im1 = ax_didv.imshow(
            dIdV,
            origin="lower",
            aspect="equal",
            extent=extent,
            cmap="PiYG_r",
            norm=norm,
        )

        ax_stm.set_title(f"V={VBias:.2f} V")
        ax_stm.set_xlabel("x [Å]")
        ax_stm.set_ylabel("y [Å]")
        ax_didv.set_xlabel("x [Å]")
        ax_didv.set_ylabel("y [Å]")

        fig.colorbar(im0, ax=ax_stm, fraction=0.046, pad=0.04)
        fig.colorbar(im1, ax=ax_didv, fraction=0.046, pad=0.04)

    fig.suptitle(f"{geom_label}, solver={solver_mode}, W={W:.3f} eV")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_path = W_dir / "summary_xy.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[xy-summary] Saved {out_path}")


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    base = Path(__file__).resolve().parent
    base_results = base / "results" / "NTCDA_xy"

    geom_labels = [
        "2site_Ruslan_long",
        "2site_Ruslan_short",
        "4site_Ruslan_kite",
    ]
    solvers = [0, -1]

    for geom in geom_labels:
        for solver in solvers:
            solver_dir = base_results / geom / f"solver_{solver}"
            w_dirs = load_W_dirs_xy(solver_dir)
            if not w_dirs:
                print(f"[xy-summary] No W dirs for {geom}, solver={solver} at {solver_dir}")
                continue
            for W, W_dir in w_dirs:
                make_summary_for_solver_and_W(geom, solver, W, W_dir)


if __name__ == "__main__":
    main()
