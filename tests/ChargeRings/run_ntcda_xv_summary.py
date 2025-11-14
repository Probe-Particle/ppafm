#!/usr/bin/env python3
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def load_W_dirs(solver_dir: Path):
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
        npz_path = d / "scan.npz"
        if npz_path.is_file():
            w_dirs.append((w_val, d, npz_path))
    w_dirs.sort(key=lambda x: x[0])
    return w_dirs


def make_summary_for_solver(geom_label: str, solver_mode: int, base_results: Path):
    solver_dir = base_results / "NTCDA" / geom_label / f"solver_{solver_mode}"
    w_dirs = load_W_dirs(solver_dir)
    if not w_dirs:
        print(f"[summary] No W directories found for {geom_label}, solver={solver_mode} at {solver_dir}")
        return

    STMs = []
    dIdVs = []
    exts = []
    Ws = []

    for W, wdir, npz_path in w_dirs:
        data = np.load(npz_path)
        STM = data["STM"]
        dIdV = data["dIdV"]
        extent = tuple(data["extent"].tolist()) if "extent" in data else None
        STMs.append(STM)
        dIdVs.append(dIdV)
        exts.append(extent)
        Ws.append(W)

    # global color scales
    stm_min = min(stm.min() for stm in STMs)
    stm_max = max(stm.max() for stm in STMs)
    didv_min = min(d.min() for d in dIdVs)
    didv_max = max(d.max() for d in dIdVs)
    didv_abs = max(abs(didv_min), abs(didv_max))

    ncols = len(Ws)
    fig, axes = plt.subplots(2, ncols, figsize=(4.5 * ncols, 6.0), squeeze=False)

    for idx, (W, STM, dIdV, extent) in enumerate(zip(Ws, STMs, dIdVs, exts)):
        ax_stm = axes[0, idx]
        ax_didv = axes[1, idx]

        im0 = ax_stm.imshow(
            STM,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="inferno",
            vmin=stm_min,
            vmax=stm_max,
        )
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-didv_abs, vmax=didv_abs)
        im1 = ax_didv.imshow(
            dIdV,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="PiYG_r",
            norm=norm,
        )

        ax_stm.set_title(f"W={W:.3f} eV")
        ax_stm.set_ylabel("V [V]")
        ax_didv.set_ylabel("V [V]")
        ax_didv.set_xlabel("Distance [Å]")

        fig.colorbar(im0, ax=ax_stm, fraction=0.046, pad=0.04)
        fig.colorbar(im1, ax=ax_didv, fraction=0.046, pad=0.04)

    fig.suptitle(f"{geom_label}, solver={solver_mode}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_path = solver_dir / "summary.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[summary] Saved {out_path}")


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    base = Path(__file__).resolve().parent
    # NOTE: current data live under results/NTCDA-/...
    base_results = base / "results" 

    # Known geometry labels that follow our directory convention
    geom_labels = [
        "2site_Ruslan_long",
        "2site_Ruslan_short",
        "4site_Ruslan_kite",   # for future 4-site study
    ]
    solvers = [0, -1]

    for geom in geom_labels:
        for solver in solvers:
            make_summary_for_solver(geom, solver, base_results)


if __name__ == "__main__":
    main()
