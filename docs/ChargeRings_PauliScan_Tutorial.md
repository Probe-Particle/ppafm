# Charge-Ring Pauli Scan Tutorial

This guide explains how to run the Pauli master-equation simulations that live under `tests/ChargeRings/`. It covers the physical idea, the available entry points, how they differ, and what to expect from each run—both numerically and on disk.

## 1. Physical background

The charge-ring model treats a handful of localized orbitals arranged on (or near) a ring above the sample surface. A metallic STM tip is scanned either across the surface (x–y maps) or along a prescribed line while the bias voltage is ramped (x–V maps). Transport is evaluated with a Pauli master equation implemented in `pyProbeParticle/pauli` and exposed to Python through `pauli_scan.py`:

- **STM current (``STM``)** corresponds to the steady-state current collected by the tip for a fixed bias/position.
- **dI/dV maps (``dIdV``)** estimate local density-of-states features by finite differences in bias voltage.
- **State energies / probabilities** track occupation statistics of the master equation and are useful when comparing to experimental resonances.

The scripts documented below assemble geometries, tunneling matrices, and solver inputs before launching the C++ backend (`pauli_lib.so`).

## 2. Key files and their roles

| File | Purpose |
|------|---------|
| `pauli_scan.py` | Core library with geometry builders, tunneling models, and high-level scan helpers (XY grid scans, x–V line scans, sweep orchestration). |
| `sweep_pauli_xy.py` | Simple Python script that sweeps one or more parameters and produces tiled 2D STM/dI/dV maps for each sample. Good for quick exploratory studies. |
| `sweep_pauli_xV.py` | Example script that sweeps along an x–V line scan, optionally overlaying experimental reference data (`exp_rings_data.npz`). |
| `pauli_xy_cli.py` | Command-line interface (CLI) for a **single** x–y scan with optional dI/dV. Supports config files and ad-hoc overrides. |
| `pauli_xv_sweep_cli.py` | CLI for x–V line scans, optionally sweeping several parameters defined in JSON/CSV tables. |
| `CombinedChargeRingsGUI_v5.py` | PyQt GUI that wraps the same library routines for interactive parameter tuning and experimental overlays. |

All entry points call into `pauli_scan.py`, so adding functionality there makes it available to both scripts and the GUI.

## 3. Prerequisites and environment

1. Activate your Python environment (NumPy, Matplotlib, SciPy available). No extra dependencies beyond the repository are needed.
2. Work from `tests/ChargeRings/`. The scripts automatically compile the required C++ helpers (`pauli_lib.so`, `GridUtils_lib.so`) on first use.
3. Optional: place experimental data files (e.g. `exp_rings_data.npz`) in the same directory when using examples that expect them.

## 4. XY-plane parameter sweeps (`sweep_pauli_xy.py`)

**When to use:** produce a grid of STM/dI/dV images while sweeping geometric or electronic parameters.

```bash
cd tests/ChargeRings
python sweep_pauli_xy.py
```

- Parameters are defined near the top of the file (`params` dict). Geometry (`nsite`, `radius`, `phiRot`), tip shape (`Rtip`, `z_tip`), tunnelling barrier (`Et0`, `At`, `wt`), and solver settings (`GammaS/T`, `Temp`) can be edited in-place.
- `scan_params` lists tuples `(parameter_name, values)` describing the sweep. Each entry creates a column in the output figure.
- On completion you will see console summaries plus:
  - A Matplotlib window showing the STM + dI/dV panels.
  - Saved artefacts in `results_xy_scan/` (`scan_YYYY_MM_DD_HHMMSS.png` and `.json`). The JSON stores metadata for downstream analysis.

Expected numeric ranges (default setup): STM current in the 10⁻⁶ range, dI/dV in the 10⁻⁵ range, probabilities between 0 and 1.

## 5. XY CLI (`pauli_xy_cli.py`)

**When to use:** single run from the command line with reproducible I/O.

Example command:
```bash
cd tests/ChargeRings
python pauli_xy_cli.py --config charge_ring_params.json --VBias 0.6 --npix 150 --plot 1 --output xy_scan
```

Key features:
- Loads defaults (`DEFAULT_PARAMS`) then merges JSON configuration and CLI overrides.
- Optional `--didv 0` skips the finite-difference pass for speed.
- Saves compressed `.npz` payloads when `--output` is given (arrays: `STM`, `dIdV`, `Es`, `Ts`, plus `params_json`).
- `--parallel 1` toggles OpenMP inside the C++ solver.

Expect textual output with run times and array statistics; plots mirror the figure produced by the sweep script but for a single configuration.

## 6. x–V sweeps with experimental reference (`sweep_pauli_xV.py`)

**When to use:** compare simulated line scans against measured data along the same spatial path.

```bash
cd tests/ChargeRings
python sweep_pauli_xV.py
```

Highlights:
- Loads experimental STM/dI/dV from `exp_rings_data.npz` and defines matching simulation line endpoints (`p1_x`, `p2_x`, ...).
- `scan_params` typically varies one bias-control parameter (e.g. `zV0`). Each sample produces a column with simulated STM (top) and dI/dV (bottom).
- Results are saved to `pauli_scan_results/` (PNG + JSON). The JSON captures timestamps, parameter settings, and references to result arrays.

Console output prints min/max values for STM, dI/dV, site energies, tunnelling amplitudes, and state probabilities so you can confirm the solver is well-behaved.

## 7. x–V CLI (`pauli_xv_sweep_cli.py`)

**When to use:** batch x–V sweeps with flexible parameter scheduling (linear ramps or explicit tables).

Example workflow:
```bash
cd tests/ChargeRings
python pauli_xv_sweep_cli.py \
    --config example_pauli_params.json \
    --sweep example_pauli_sweep.json \
    --samples 7 \
    --nx 120 --nV 140 \
    --figure sweep_summary.png \
    --output sweep_dump
```

What it does:
- Reads linear sweep endpoints (`--sweep`) or a CSV/whitespace table (`--table`) to set parameter values per sample.
- Runs `calculate_xV_scan_orb` for each sample, collecting STM and dI/dV panels along the line defined by `p1/p2` (overrides via CLI).
- Produces a two-row figure (STM over distance, dI/dV over distance) with shared color scales; optionally saved via `--figure`.
- Persists raw arrays to `.npz` when `--output` is provided, along with sweep metadata for post-processing.

## 8. Library reference (`pauli_scan.py`)

`pauli_scan.py` is the backbone for every interface. Notable functions:

- `make_site_geom(params)`: builds site positions/rotations either from a circular template or a user-supplied geometry file.
- `generate_hops_gauss(spos, params, pTips=None, bBarrier=True)`: computes tunnelling amplitudes with a Gaussian barrier model driven by `Et0`, `At`, `wt`.
- `interpolate_hopping_maps(Ts_gauss, Ts_orb, c, T0)`: blends analytic tunnelling with orbital-derived maps.
- `scan_xy_orb(...)`: runs a full 2D grid scan, returning STM, dI/dV, energy tensors, probabilities, and solver metadata.
- `calculate_xV_scan_orb(...)`: performs 1D line scans across a range of voltages; supports orbital inputs and current decomposition exports.
- `sweep_scan_param_pauli_xy_orb(...)` / `sweep_scan_param_pauli_xV_orb(...)`: high-level sweep drivers used by the example scripts.

The module also configures Pauli solver safety flags (disabling probability-stop checks, matching CLI defaults) before calling the C++ backend.

## 9. Interactive GUI (`CombinedChargeRingsGUI_v5.py`)

Run the GUI to explore parameters visually:

```bash
cd tests/ChargeRings
python CombinedChargeRingsGUI_v5.py
```

Features:
- Live control over geometry, bias, and tunnelling settings.
- Tabs for 2D STM maps, x–V panels, and experimental overlays (when `exp_*` arrays are loaded).
- Uses the same `scan_xy_orb` and `calculate_xV_scan_orb` routines. Returned arrays (STM, dI/dV, `stateEs`) are shown within the UI and can be exported for offline analysis.

## 10. Outputs and expected locations

| Component | Location | Contents |
|-----------|----------|----------|
| XY sweep figures | `results_xy_scan/` | Timestamped PNG + JSON summarizing each sweep. |
| x–V sweep figures | `pauli_scan_results/` | PNG + JSON with per-column parameter metadata. |
| CLI dumps (`--output`) | user-specified `.npz` | Arrays: STM, dI/dV, Es, Ts, distance/bias axes, parameter JSON. |

These artefacts make it easy to plot in notebooks or compare against experimental line traces.

## 11. Troubleshooting tips

- **Missing tunnel parameters (`Et0`, `At`, `wt`)**: ensure your parameter dictionaries include all barrier keys; defaults in the CLI scripts are a good template.
- **Probability validation errors**: the scripts already disable strict checks; if you re-enable them, be prepared to adjust tolerances when currents are very small.
- **Large runtimes**: reduce `npix` (XY) or `nx`/`nV` (x–V) while prototyping. Re-enable `--parallel 1` once the setup is stable.
- **Experimental overlays**: verify the experimental start/end points (`ep1_*`, `ep2_*`) match the simulation path.

## 12. Further exploration

- Extend `pauli_scan.py` with alternative tunnelling models (e.g., tip–orbital integrals) and reuse them from the CLI/GUI.
- Add new sweep dimensions by editing `scan_params` in the scripts or expanding the JSON/CVS used by `pauli_xv_sweep_cli.py`.
- Use the saved `.json`/`.npz` artefacts to build comparison notebooks or machine-learning surrogates.

With these tools you can reproduce the physics explored in the Charge Rings project—ranging from qualitative STM imaging to quantitative fits against experimental line scans—using consistent parameters across scripts, CLIs, and the GUI.
