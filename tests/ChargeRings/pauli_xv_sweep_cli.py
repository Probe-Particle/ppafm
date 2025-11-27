#!/usr/bin/env python3
"""Command-line helper for sweeping Pauli xV scans over a 1D parameter path.

The script wraps :func:`pauli_scan.calculate_xV_scan` so we can evaluate a
sequence of voltage-line scans while interpolating a chosen set of input
parameters.  Results are collated into a two-row figure (STM on top, dI/dV
below) and can optionally be saved to disk for post-processing.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR.parent.parent) not in sys.path:
    sys.path.append('../../')

import pauli_scan  # noqa: E402
from pyProbeParticle import pauli  # noqa: E402

# Boltzmann constant in eV/K for solver setup when we create PauliSolver
K_BOLTZ_EV = 8.617333262e-5

# Baseline parameter dictionary mirroring the useful GUI defaults so the CLI can
# run without external JSON configuration. Users are expected to override these
# via --config.
DEFAULT_PARAMS: Dict[str, object] = {
    "nsite": 2,
    "radius": 5.2,
    "phiRot": 1.3,
    "phi0_ax": 0.2,
    "VBias": 0.70,
    "Rtip": 3.0,
    "z_tip": 5.0,
    "zV0": -1.0,
    "zVd": 15.0,
    "zQd": 0.0,
    "Q0": 1.0,
    "Qzz": 10.0,
    "Esite": -0.100,
    "W": 0.05,
    "Temp": 3.0,
    "decay": 0.3,
    "GammaS": 0.01,
    "GammaT": 0.01,
    "solver_mode": 0,
    "Et0": 0.2,
    "wt": 8.0,
    "At": 0.0,
    "c_orb": 1.0,
    "T0": 1.0,
    "L": 20.0,
    "npix": 200,
    "bMirror": True,
    "bRamp": True,
    "bWijDistance": False,
    "verbosity": 0,
    "p1_x": -10.0,
    "p1_y":  0.0,
    "p2_x": 10.0,
    "p2_y":  0.0,
}

# Default scan line (Å)
DEFAULT_LINE: Tuple[Tuple[float, float], Tuple[float, float]] = ((9.72, -9.96), (-11.0, 12.0))

DEFAULT_SWEEP_SAMPLES = 11


@dataclass
class SweepSpec:
    key: str
    values: np.ndarray

    def value_at(self, index: int) -> float:
        return float(self.values[index])


def resolve_geometry_file(path: Path) -> str:
    """Resolve geometry file path relative to CWD or script directory."""
    path = path.expanduser()
    if path.exists():
        return str(path.resolve())
    script_path = (SCRIPT_DIR / path).expanduser()
    if script_path.exists():
        return str(script_path.resolve())
    raise FileNotFoundError(f"Geometry file '{path}' not found (checked {path} and {script_path})")


def sync_nsite_with_geometry(params: Dict[str, object]) -> None:
    """Update nsite to match geometry file if provided."""
    geometry_path = params.get("geometry_file")
    if not geometry_path:
        return
    spos, _rots, _angles = pauli_scan.load_site_geometry(geometry_path)
    params["nsite"] = int(spos.shape[0])


def load_params(config_path: Path | None) -> Dict[str, object]:
    params = DEFAULT_PARAMS.copy()
    if config_path is not None:
        if not config_path.is_absolute():
            config_path = SCRIPT_DIR / config_path
        if not config_path.exists():
            raise FileNotFoundError(f"Parameter file '{config_path}' not found")
        with config_path.open() as fp:
            cfg = json.load(fp)
        params.update(cfg)
    if "geometry_file" in params:
        params["geometry_file"] = resolve_geometry_file(Path(params["geometry_file"]))
        sync_nsite_with_geometry(params)
    params["nsite"] = int(params["nsite"])
    params["npix"] = int(params["npix"])
    params["solver_mode"] = int(params.get("solver_mode", 0))
    return params


def parse_linear_sweep(path: Path, samples: int | None) -> Tuple[List[SweepSpec], int]:
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    with path.open() as fp:
        raw = json.load(fp)
    if not isinstance(raw, dict):
        raise ValueError("Sweep specification must be a JSON object mapping parameter names to [start, stop]")

    sample_count = int(samples) if samples is not None else DEFAULT_SWEEP_SAMPLES
    specs: List[SweepSpec] = []
    for key, value in raw.items():
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"Parameter '{key}' must be a list of two values [start, stop]")
        start, stop = map(float, value)
        values = np.linspace(start, stop, sample_count)
        specs.append(SweepSpec(key=key, values=values))
    if not specs:
        raise ValueError("Sweep specification is empty; provide at least one parameter to vary")
    return specs, sample_count


def parse_table_sweep(path: Path) -> Tuple[List[SweepSpec], int]:
    if not path.is_absolute():
        path = SCRIPT_DIR / path

    def _load(delim: str | None) -> np.ndarray:
        return np.genfromtxt(path, names=True, delimiter=delim, dtype=float)

    try:
        data = _load(',')
        if data.size == 0:
            raise ValueError
    except Exception:
        data = _load(None)

    if data.size == 0:
        raise ValueError(f"Sweep table '{path}' is empty")

    if data.ndim == 0:
        data = data.reshape(1)

    names = data.dtype.names
    if not names:
        raise ValueError(f"Sweep table '{path}' must contain a header row with parameter names")

    sample_count = int(data.shape[0])
    specs: List[SweepSpec] = []
    for name in names:
        column = np.asarray(data[name], dtype=float)
        if column.ndim == 0:
            column = column.reshape(1)
        if len(column) != sample_count:
            raise ValueError(f"Column '{name}' has inconsistent length in sweep table")
        specs.append(SweepSpec(key=name, values=column))
    return specs, sample_count


@dataclass
class SweepResult:
    label: str
    params: Dict[str, object]
    STM: np.ndarray
    dIdV: np.ndarray
    Vbiases: np.ndarray
    distance: np.ndarray
    extent: Tuple[float, float, float, float]


def compute_distance_axis(pTips: np.ndarray) -> np.ndarray:
    base = pTips[0]
    deltas = pTips - base
    return np.sqrt(deltas[:, 0] ** 2 + deltas[:, 1] ** 2)


def run_single_scan(
    params: Dict[str, object],
    start_point: Tuple[float, float],
    end_point: Tuple[float, float],
    nx: int,
    nV: int,
    Vmin: float,
    Vmax: float,
    parallel: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[float, float, float, float]]:
    if Vmax is None:
        Vmax = float(params["VBias"])
    params_local = params.copy()
    nsite = int(params_local["nsite"])
    verbosity = int(params_local.get("verbosity", 0))
    solver = pauli.PauliSolver(nSingle=nsite, nleads=2, verbosity=verbosity)
    which_solver = int(params_local.get("solver_mode", 0))
    solver.setLinSolver(1, 50, 1e-12, which_solver)
    T_eV = float(params_local["Temp"]) * K_BOLTZ_EV
    solver.set_lead(0, 0.0, T_eV)
    solver.set_lead(1, 0.0, T_eV)
    solver.set_check_prob_stop(bCheckProb=False, bCheckProbStop=False, CheckProbTol=1e-12)
    pauli.bValidateProbabilities = False
    output = pauli_scan.calculate_xV_scan_orb(
        params_local,
        start_point=start_point,
        end_point=end_point,
        nx=nx,
        nV=nV,
        Vmin=Vmin,
        Vmax=Vmax,
        bLegend=False,
        bOmp=parallel,
        orbital_2D=None,
        orbital_lvec=None,
        pauli_solver=solver,
    )
    STM, dIdV, _Es, _Ts, _probs, _stateEs, pTips, Vbiases, _spos, _rots, *_ = output
    distance = compute_distance_axis(pTips)
    extent = (distance[0], distance[-1], Vmin, Vmax)
    return STM, dIdV, Vbiases, distance, extent


def perform_sweep(
    base_params: Dict[str, object],
    sweep_specs: List[SweepSpec],
    samples: int,
    start_point: Tuple[float, float],
    end_point: Tuple[float, float],
    nx: int,
    nV: int,
    Vmin: float,
    Vmax: float | None,
    parallel: bool,
) -> List[SweepResult]:
    results: List[SweepResult] = []
    frac_iter: Iterable[float]
    if samples == 1:
        frac_iter = (0.0,)
    else:
        frac_iter = (i / (samples - 1) for i in range(samples))

    pauli.bValidateProbabilities = False

    for idx, frac in enumerate(frac_iter):
        params = base_params.copy()
        label_parts = []
        for spec in sweep_specs:
            value = spec.value_at(idx)
            params[spec.key] = value
            label_parts.append(f"{spec.key}={value:.4g}")
        label = ", ".join(label_parts)
        print(f"Running sweep sample {idx + 1}/{samples}: {label}")
        STM, dIdV, Vbiases, distance, extent = run_single_scan(params, start_point=start_point, end_point=end_point, nx=nx, nV=nV, Vmin=Vmin, Vmax=Vmax, parallel=parallel)
        results.append(SweepResult(label=label, params=params, STM=STM, dIdV=dIdV, Vbiases=Vbiases,   distance=distance, extent=extent, ))
    return results


def plot_results(results: List[SweepResult], *, save_path: Path | None, show_plot: bool) -> None:
    ncols = len(results)
    if ncols == 0:
        print("No results to plot")
        return

    fig, axes = plt.subplots(2, ncols, figsize=(4.5 * ncols, 6.0), squeeze=False)

    stm_min = min(res.STM.min() for res in results)
    stm_max = max(res.STM.max() for res in results)
    didv_min = min(res.dIdV.min() for res in results)
    didv_max = max(res.dIdV.max() for res in results)
    didv_abs = max(abs(didv_min), abs(didv_max))

    for idx, res in enumerate(results):
        ax_stm = axes[0, idx]
        ax_didv = axes[1, idx]
        im0 = ax_stm.imshow(res.STM, origin="lower", aspect="auto", extent=res.extent, cmap="inferno", vmin=stm_min, vmax=stm_max)
        im1 = ax_didv.imshow(
            res.dIdV,
            origin="lower",
            aspect="auto",
            extent=res.extent,
            cmap="PiYG_r",
            norm=TwoSlopeNorm(vcenter=0.0, vmin=-didv_abs, vmax=didv_abs),
        )
        ax_stm.set_title(res.label)
        ax_stm.set_ylabel("V [V]")
        ax_didv.set_ylabel("V [V]")
        ax_didv.set_xlabel("Distance [Å]")
        fig.colorbar(im0, ax=ax_stm, fraction=0.046, pad=0.04)
        fig.colorbar(im1, ax=ax_didv, fraction=0.046, pad=0.04)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def save_results(path: Path, results: List[SweepResult], sweep_specs: List[SweepSpec], base_params: Dict[str, object]) -> None:
    payload = {
        "labels": np.array([res.label for res in results]),
        "distance": results[0].distance,
        "Vbiases": results[0].Vbiases,
        "params_json": json.dumps(base_params),
        "sweep_params": json.dumps({spec.key: spec.values.tolist() for spec in sweep_specs}),
    }
    for idx, res in enumerate(results):
        payload[f"stm_{idx}"] = res.STM
        payload[f"didv_{idx}"] = res.dIdV
    path = path.with_suffix(".npz")
    np.savez_compressed(path, **payload)
    print(f"Saved sweep data to {path}")

def resolve_line(args: argparse.Namespace, params: Dict[str, object]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    p1 = (
        float(args.p1x) if args.p1x is not None else float(params.get("p1_x", DEFAULT_LINE[0][0])),
        float(args.p1y) if args.p1y is not None else float(params.get("p1_y", DEFAULT_LINE[0][1])),
    )
    p2 = (
        float(args.p2x) if args.p2x is not None else float(params.get("p2_x", DEFAULT_LINE[1][0])),
        float(args.p2y) if args.p2y is not None else float(params.get("p2_y", DEFAULT_LINE[1][1])),
    )
    return p1, p2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep Pauli xV scans along a 1D parameter path")
    parser.add_argument("--config",    type=Path,  default=Path("example_pauli_params.json"), help="JSON file with baseline parameters")
    parser.add_argument("--sweep",     type=Path,  default=Path("example_pauli_sweep.json"), help="JSON file describing linear sweep endpoints [start, stop]")
    parser.add_argument("--table",     type=Path,  default=None, help="Optional CSV/whitespace table specifying explicit parameter values (overrides --sweep)")
    #parser.add_argument("--geometry",  type=Path,  default="Ruslan_short.txt", help="Path to custom site geometry file (x y angle columns)")
    parser.add_argument("--geometry",  type=Path,  default="Ruslan_kite.txt", help="Path to custom site geometry file (x y angle columns)")
    parser.add_argument("--solverMode",type=int,   choices=[0, -1, -2], help="Select solver mode: 0=PME, -1=Ground State, -2=Boltzmann")
    parser.add_argument("--samples",   type=int,   default=5, help="Number of sweep samples (ignored when --table is used)")
    parser.add_argument("--nx",        type=int,   default=100,  help="Number of points along the spatial line")
    parser.add_argument("--nV",        type=int,   default=120,  help="Number of bias samples")
    parser.add_argument("--Vmin",      type=float, default=0.0,  help="Minimum bias [V]")
    parser.add_argument("--Vmax",      type=float, default=None, help="Maximum bias [V] (defaults to each sample's VBias)")
    parser.add_argument("--p1x",       type=float, default=None, help="Start point x-coordinate [Å]")
    parser.add_argument("--p1y",       type=float, default=None, help="Start point y-coordinate [Å]")
    parser.add_argument("--p2x",       type=float, default=None, help="End point x-coordinate [Å]")
    parser.add_argument("--p2y",       type=float, default=None, help="End point y-coordinate [Å]")
    parser.add_argument("--parallel",  type=int,   default=0,    help="Enable threaded execution (1=yes, 0=no)")
    parser.add_argument("--plot",      type=int,   default=1,    help="Display the summary figure (1=yes, 0=no)")
    parser.add_argument("--figure",    type=Path,  default=None, help="Optional path to save the summary figure")
    parser.add_argument("--output",    type=Path,  default=None, help="Optional .npz path to store raw results")
    parser.add_argument("--verbosity", type=int,   default=0,    help="Pass-through solver verbosity")
    args = parser.parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be a positive integer")

    params = load_params(args.config)
    if args.geometry is not None:
        params["geometry_file"] = resolve_geometry_file(args.geometry)
        sync_nsite_with_geometry(params)
    if args.solverMode is not None:
        params["solver_mode"] = int(args.solverMode)
    if args.verbosity is not None:
        params["verbosity"] = int(args.verbosity)
    if args.table is not None:
        sweep_specs, sample_count = parse_table_sweep(args.table)
    else:
        sweep_specs, sample_count = parse_linear_sweep(args.sweep, samples=args.samples)
    start_point, end_point = resolve_line(args, params)

    parallel = bool(args.parallel)
    results = perform_sweep(base_params=params, sweep_specs=sweep_specs, samples=sample_count, start_point=start_point, end_point=end_point, nx=int(args.nx), nV=int(args.nV), Vmin=float(args.Vmin), Vmax=None if args.Vmax is None else float(args.Vmax), parallel=parallel)

    if args.output:
        save_results(args.output, results, sweep_specs, params)
    plot_results(results, save_path=args.figure, show_plot=bool(args.plot))
