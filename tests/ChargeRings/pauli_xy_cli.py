#!/usr/bin/env python3
"""Command-line helper for running Pauli STM x-y scans.

This script wraps the existing :mod:`pauli_scan` utilities and exposes a
light-weight CLI for computing STM and dI/dV maps on a regular grid.  It keeps
only the essentials needed for the simulation so we can easily toggle between
serial and threaded execution (``bOmp`` flag) while experimenting from the
shell or CI scripts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../../')
import pauli_scan  # noqa: E402  (local module adjusts sys.path for pyProbeParticle)
from pyProbeParticle import pauli

# Default parameter set mirrors the useful subset of GUI defaults so the CLI can
# be used without supplying a configuration file.
DEFAULT_PARAMS: Dict[str, object] = {
    # Geometry
    "nsite": 4,
    "radius": 5.2,
    "phiRot": 1.3,
    "phi0_ax": 0.2,
    "zQd": 0.0,
    # Tip bias / electrostatics
    "VBias": 0.70,
    "dQ": 0.02,  # finite-difference step for dI/dV
    # Tip shape
    "Rtip": 3.0,
    "z_tip": 5.0,
    "zV0": -1.0,
    "zVd": 15.0,
    # Multipole coefficients
    "Q0": 1.0,
    "Qzz": 10.0,
    # Transport solver
    "Esite": -0.100,
    "W": 0.05,
    "Temp": 3.0,
    "decay": 0.3,
    "GammaS": 0.01,
    "GammaT": 0.01,
    # Barrier / tunnelling model
    "Et0": 0.2,
    "wt": 8.0,
    "At": 0.0,
    "c_orb": 1.0,
    "T0": 1.0,
    # Scan window
    "L": 20.0,
    "npix": 200,
    # Misc flags
    "bMirror": True,
    "bRamp": True,
    "bWijDistance": False,
    "verbosity": 0,
}

# Keys that can be overridden directly from the CLI (value: argparse attribute).
CLI_OVERRIDE_KEYS = {
    "nsite": "nsite",
    "radius": "radius",
    "phiRot": "phiRot",
    "phi0_ax": "phi0ax",
    "VBias": "VBias",
    "dQ": "dQ",
    "Rtip": "Rtip",
    "z_tip": "zTip",
    "zV0": "zV0",
    "zVd": "zVd",
    "zQd": "zQd",
    "Q0": "Q0",
    "Qzz": "Qzz",
    "Esite": "Esite",
    "W": "W",
    "Temp": "Temp",
    "decay": "decay",
    "GammaS": "GammaS",
    "GammaT": "GammaT",
    "Et0": "Et0",
    "wt": "wt",
    "At": "At",
    "c_orb": "cOrb",
    "T0": "T0",
    "L": "L",
    "npix": "npix",
}


def build_params(args: argparse.Namespace) -> Dict[str, object]:
    params: Dict[str, object] = DEFAULT_PARAMS.copy()

    # Load config overrides if provided.
    if args.config:
        with args.config.open() as fp:
            config_data = json.load(fp)
        params.update(config_data)

    # Apply CLI overrides.
    for key, attr in CLI_OVERRIDE_KEYS.items():
        value = getattr(args, attr)
        if value is not None:
            params[key] = value

    if args.verbosity is not None:
        params["verbosity"] = int(args.verbosity)
    params["bWijDistance"] = bool(args.wijDistance)
    params["bMirror"] = bool(args.mirror)
    params["bRamp"] = bool(args.ramp)

    # Ensure integer fields stay ints.
    params["nsite"] = int(params["nsite"])
    params["npix"] = int(params["npix"])

    return params

def run_xy_scan(params: Dict[str, object], *, parallel: bool, compute_didv: bool) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Execute the STM scan (and optional dI/dV) returning result arrays."""
    params = params.copy()
    nsite = int(params["nsite"])
    verbosity = int(params.get("verbosity", 0))

    # Geometry must be determined before solver setup (for Wij configuration).
    spos, rots, _angles = pauli_scan.make_site_geom(params)

    solver = pauli.PauliSolver(nSingle=nsite, nleads=2, verbosity=verbosity)
    T_eV = params["Temp"] * 8.617333262e-5  # k_B in eV/K
    solver.set_lead(0, 0.0, T_eV)
    solver.set_lead(1, 0.0, T_eV)
    solver.set_check_prob_stop(bCheckProb=False, bCheckProbStop=False, CheckProbTol=1e-12)
    pauli.bValidateProbabilities = False
    pauli_scan._apply_wij_config(solver, spos, params)

    Tmin = float(params.get("Tmin", 0.0))
    EW = float(params.get("EW", 2.0))
    pauli.set_valid_point_cuts(Tmin, EW)

    Ts_gauss, _pTips, _beta, _barrier = pauli_scan.generate_hops_gauss(spos, params)
    Ts_flat = pauli_scan.interpolate_hopping_maps(
        Ts_gauss,
        None,
        c=params.get("c_orb", 1.0),
        T0=params.get("T0", 1.0),
    )
    Ts_flat = np.ascontiguousarray(Ts_flat, dtype=np.float64)

    STM, Es, Ts, _probs, _stateEs = pauli.run_pauli_scan_top(
        spos,
        rots,
        params,
        pauli_solver=solver,
        bOmp=parallel,
        Ts=Ts_flat,
    )

    results = {"STM": STM, "Es": Es, "Ts": Ts}

    dIdV = None
    if compute_didv:
        dQ = float(params.get("dQ", 0.0))
        if dQ <= 0.0:
            raise ValueError("dQ must be positive to compute dI/dV. Override with --dq.")
        params_shifted = params.copy()
        params_shifted["VBias"] = float(params["VBias"]) + dQ
        solver_shifted = pauli.PauliSolver(nSingle=nsite, nleads=2, verbosity=verbosity)
        solver_shifted.set_lead(0, 0.0, T_eV)
        solver_shifted.set_lead(1, 0.0, T_eV)
        solver_shifted.set_check_prob_stop(bCheckProb=False, bCheckProbStop=False, CheckProbTol=1e-12)
        pauli_scan._apply_wij_config(solver_shifted, spos, params_shifted)
        STM_shifted, _Es2, _Ts2, _probs2, _stateEs2 = pauli.run_pauli_scan_top(
            spos,
            rots,
            params_shifted,
            pauli_solver=solver_shifted,
            bOmp=parallel,
            Ts=Ts_flat,
        )
        dIdV = (STM_shifted - STM) / dQ
        results["dIdV"] = dIdV

    metadata = {
        "spos": spos,
        "rots": rots,
    }
    return results, metadata


def maybe_save(output_path: Path, params: Dict[str, object], results: Dict[str, np.ndarray]) -> None:
    output_path = output_path.with_suffix(".npz")
    save_kwargs = {name: array for name, array in results.items() if array is not None}
    save_kwargs["params_json"] = json.dumps(params)
    np.savez_compressed(output_path, **save_kwargs)
    print(f"Saved results to {output_path}")


def maybe_plot(results: Dict[str, np.ndarray]) -> None:
    STM = results["STM"]
    dIdV = results.get("dIdV")

    fig, axes = plt.subplots(1, 2 if dIdV is not None else 1, figsize=(10, 4))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    ax = axes[0]
    im0 = ax.imshow(STM, origin="lower", cmap="inferno")
    ax.set_title("STM")
    plt.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)

    if dIdV is not None:
        ax = axes[1]
        im1 = ax.imshow(dIdV, origin="lower", cmap="PiYG_r")
        ax.set_title("dI/dV")
        plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Pauli STM x-y scan and compute STM & dI/dV maps")
    parser.add_argument("--config",      type=Path,  help="Optional JSON file with parameter overrides")
    parser.add_argument("--nsite",       type=int,   help="Number of sites")
    parser.add_argument("--radius",      type=float, help="Site ring radius [Å]")
    parser.add_argument("--phiRot",      type=float, help="Rotation angle of ring [rad]")
    parser.add_argument("--phi0ax",      type=float, help="Azimuthal offset for local axes [rad]")
    parser.add_argument("--VBias",       type=float, help="Tip bias voltage [V]")
    parser.add_argument("--dQ",          type=float, help="Finite-difference step for dI/dV [V]")
    parser.add_argument("--Rtip",        type=float, help="Tip radius [Å]")
    parser.add_argument("--zTip",        type=float, help="Tip base height [Å]")
    parser.add_argument("--zV0",         type=float, help="Electrostatic zV0 parameter [Å]")
    parser.add_argument("--zVd",         type=float, help="Electrostatic zVd parameter [Å]")
    parser.add_argument("--zQd",         type=float, help="Site plane z coordinate [Å]")
    parser.add_argument("--Q0",          type=float, help="Multipole coefficient Q0")
    parser.add_argument("--Qzz",         type=float, help="Multipole coefficient Qzz")
    parser.add_argument("--Esite",       type=float, help="On-site energy [eV]")
    parser.add_argument("--W",           type=float, help="Coulomb constant W [eV]")
    parser.add_argument("--Temp",        type=float, help="Temperature [K]")
    parser.add_argument("--decay",       type=float, help="Exponential decay for tunnelling")
    parser.add_argument("--GammaS",      type=float, help="Gamma substrate [eV]")
    parser.add_argument("--GammaT",      type=float, help="Gamma tip [eV]")
    parser.add_argument("--Et0",         type=float, help="Barrier parameter Et0 [eV]")
    parser.add_argument("--wt",          type=float, help="Barrier width parameter wt [Å]")
    parser.add_argument("--At",          type=float, help="Barrier amplitude parameter At [eV]")
    parser.add_argument("--cOrb",        type=float, help="Orbital mixing coefficient")
    parser.add_argument("--T0",          type=float, help="Overall tunnelling scale T0")
    parser.add_argument("--L",           type=float, help="Half-width of scan window [Å]")
    parser.add_argument("--npix",        type=int,   help="Number of pixels per axis in scan grid")
    parser.add_argument("--wijDistance", type=int,   default=0, help="Use distance-dependent Wij (1=yes, 0=no)")
    parser.add_argument("--mirror",      type=int,   default=1, help="Mirror image term (1=on, 0=off)")
    parser.add_argument("--ramp",        type=int,   default=1, help="Ramp flag (1=on, 0=off)")
    parser.add_argument("--parallel",    type=int,   default=1, help="Enable threaded scan (1=yes)")
    parser.add_argument("--verbosity",   type=int,   default=None, help="Pass-through verbosity for PauliSolver")
    parser.add_argument("--output",      type=Path,  help="Optional .npz file to store STM/dI/dV arrays")
    parser.add_argument("--plot",        type=int,   default=1, help="Display STM and dI/dV maps (1=yes)")
    parser.add_argument("--didv",        type=int,   default=1, help="Compute dI/dV (1=yes, 0=no)")
    args = parser.parse_args()
    params = build_params(args)

    running_mode = "parallel" if args.parallel else "serial"
    print(f"Running Pauli xy scan with nsite={params['nsite']} in {running_mode} mode")

    results, _meta = run_xy_scan(
        params,
        parallel=bool(args.parallel),
        compute_didv=bool(args.didv),
    )

    stm = results["STM"]
    print(
        f"STM stats: min={stm.min():.4e}, max={stm.max():.4e}, shape={stm.shape}"
    )
    if "dIdV" in results:
        didv = results["dIdV"]
        print(
            f"dI/dV stats: min={didv.min():.4e}, max={didv.max():.4e}, shape={didv.shape}"
        )

    if args.output:
        maybe_save(args.output, params, results)

    if args.plot:
        maybe_plot(results)
        plt.show()
