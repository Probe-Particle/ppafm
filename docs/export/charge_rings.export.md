# Pauli Master Equation Charge Rings — Export Document

## Purpose & Goals

This project simulates **scanning tunneling microscopy (STM)** and **dI/dV spectroscopy** of molecular charge-transfer systems (e.g., NTCDA brickwall, PTCDA) on surfaces. The physical model is the **Pauli Master Equation (PME)**: a rate-equation solver for the steady-state occupation probabilities of many-body charge states on a set of molecular sites (quantum dots) coupled to two electrodes (substrate + tip). The tip position is scanned in 2D (xy maps) or along a line with voltage sweep (xV maps), producing simulated STM current and dI/dV maps for direct comparison with experimental data.

The system has **three solver backends**:
1. **C++ CPU solver** (`PauliSolver` in `pauli.hpp`) — serial or OpenMP-parallel
2. **OpenCL GPU solver** (`PME.cl` + `pauli_ocl.py`) — parallelized on GPU via PyOpenCL
3. **Python reference** (legacy, in `ChargeRings.py`) — older occupancy-based solver, not PME

All three produce identical physics (validated to ~1e-12 relative error).

## General Mechanics

1. **Geometry**: N molecular sites arranged in a circle, rectangle, or loaded from file. Each site has position (x,y,z), onsite energy E, and orientation (rotation matrix for multipole).
2. **Tip interaction**: For each tip position, site energies are shifted by the tip's electrostatic potential (multipole expansion with optional mirror charge). Tunneling amplitudes decay exponentially with tip-site distance.
3. **Many-body states**: 2^N states (all occupation combinations). State energies = sum of onsite + Coulomb interaction (W or Wij matrix) between occupied sites.
4. **Pauli factors**: Forward/backward transition rates between states differing by one electron, computed from Fermi functions and tunneling couplings.
5. **Kernel matrix**: Rate matrix built from Pauli factors; normalized (sum of probabilities = 1) and solved via Gaussian elimination or SVD.
6. **Current**: Computed from probability-weighted transition rates for a given lead.
7. **Scanning**: The solver loops over tip positions (and optionally bias voltages), computing current at each point.

## File Listing (Core Modules & Entry Points Only)

### C++ Solver Core

- `/home/prokop/git/ppafm/cpp/pauli.hpp` — **Main solver**: `PauliSolver` class with state enumeration, energy calculation, Pauli factor generation, kernel build, linear solve, current computation. ~1600 lines, header-only.
- `/home/prokop/git/ppafm/cpp/pauli_lib.cpp` — **C wrapper** for ctypes: `create_solver`, `set_lead`, `set_hsingle`, `set_wij`, `generate_pauli_factors`, `generate_kernel`, `solve_pauli`, `solve_hsingle`, `scan_current`, `scan_current_tip`. Also contains `is_valid_point` cutoff logic.
- `/home/prokop/git/ppafm/cpp/TipField.h` — Tip electrostatic potential: multipole expansion with mirror charge (`evalMultipoleMirror`), tunneling amplitude evaluation.
- `/home/prokop/git/ppafm/cpp/Multipoles.cpp` — Multipole coefficient evaluation (monopole/dipole/quadrupole).
- `/home/prokop/git/ppafm/cpp/gauss_solver.hpp` — Gaussian elimination linear solver (used by `solve_kern`).
- `/home/prokop/git/ppafm/cpp/SVD.h` — SVD-based least-squares solver (alternative to Gaussian).
- `/home/prokop/git/ppafm/cpp/Vec3.h` — 3D vector math (dot, cross, norm) used throughout.
- `/home/prokop/git/ppafm/cpp/Vec2.h` — 2D vector math.
- `/home/prokop/git/ppafm/cpp/Mat3.h` — 3x3 matrix operations (rotation matrices).
- `/home/prokop/git/ppafm/cpp/print_utils.hpp` — Debug print helpers for arrays/matrices.
- `/home/prokop/git/ppafm/cpp/fastmath.h` — Fast math approximations.
- `/home/prokop/git/ppafm/cpp/integerOps.h` — Bit operations: `popcount`, `ctz` (count trailing zeros) for state transition detection.
- `/home/prokop/git/ppafm/cpp/Makefile` — Build system for C++ libraries (targets: `pauli`, `chrings`, `landauer`).

### OpenCL GPU Solver

- `/home/prokop/git/ppafm/cl/PME.cl` — **OpenCL kernels**: `compute_tip_interaction` (kernel 1: site energy shifts + tunneling factors) and `solve_pme` (kernel 2: many-body energy assembly, rate matrix build in shared memory, parallel Gauss-Jordan elimination, current extraction). Hardcoded for N_STATES=16 (4 sites).

### Python Bindings

- `/home/prokop/git/ppafm/pyProbeParticle/pauli.py` — **Python ctypes wrapper** for C++ `PauliSolver`: `PauliSolver` class with `set_lead`, `set_tunneling`, `set_hsingle`, `set_Wij`, `generate_pauli_factors`, `generate_kernel`, `solve`, `scan_current`, `scan_current_tip`. Auto-compiles C++ library on import.
- `/home/prokop/git/ppafm/pyProbeParticle/pauli_ocl.py` — **PyOpenCL wrapper**: `PauliSolverCL` class with `scan_current_tip` method. Handles device setup, buffer management, kernel dispatch, and post-fetch CPU-side `is_valid_point` cut to match C++ behavior.
- `/home/prokop/git/ppafm/pyProbeParticle/cpp_utils.py` — C++ compilation utilities: `compile_lib`, path management, ctypes pointer helpers.
- `/home/prokop/git/ppafm/pyProbeParticle/utils.py` — Utility functions: `makeCircle`, `makeRotMats`, `makePosXY`.
- `/home/prokop/git/ppafm/pyProbeParticle/ChargeRings.py` — Legacy Python ctypes binding for older C++ `ChargeRings.cpp` solver (occupancy-based, not PME). Kept for backward compatibility.

### Core Application Scripts (tests/ChargeRings/)

- `/home/prokop/git/ppafm/tests/ChargeRings/pauli_scan.py` — **Core scanning engine**: `make_site_geom`, `make_configured_solver`, `calculate_xV_scan`, `calculate_xy_scan`, `calculate_xV_scan_orb`, `sweep_scan_param_pauli_xV_orb`, `sweep_scan_param_pauli_xy_orb`, `run_xv_scan_case`, `run_xy_scan_case`, `save_scan_case`, `export_xv_state_and_current_decomposition_plots`. ~3000 lines, the central library all CLIs and GUIs depend on.
- `/home/prokop/git/ppafm/tests/ChargeRings/pauli_xv_sweep_cli.py` — **CLI entry point** for xV (position-voltage) line scans with parameter sweeps. Loads JSON config, calls `pauli_scan` functions, saves NPZ/PNG/JSON.
- `/home/prokop/git/ppafm/tests/ChargeRings/pauli_xy_cli.py` — **CLI entry point** for xy 2D STM/dI/dV maps. Similar structure to xV CLI.
- `/home/prokop/git/ppafm/tests/ChargeRings/sweep_pauli_xV.py` — **Example script**: xV sweep with experimental reference data comparison.
- `/home/prokop/git/ppafm/tests/ChargeRings/sweep_pauli_xy.py` — **Example script**: xy sweep with orbital data.
- `/home/prokop/git/ppafm/tests/ChargeRings/run_ntcda_all_studies.py` — **Batch driver**: runs xV study, xV summary, xy study, xy summary in sequence.
- `/home/prokop/git/ppafm/tests/ChargeRings/run_ntcda_xv_study.py` — Runs xV scans over W parameter for each geometry/solver.
- `/home/prokop/git/ppafm/tests/ChargeRings/run_ntcda_xy_study.py` — Runs xy maps over W and VBias for each geometry/solver.
- `/home/prokop/git/ppafm/tests/ChargeRings/run_ntcda_xv_summary.py` — Generates summary panels from xV study results.
- `/home/prokop/git/ppafm/tests/ChargeRings/run_ntcda_xy_summary.py` — Generates summary panels from xy study results.

### GUI Applications

- `/home/prokop/git/ppafm/tests/ChargeRings/PauliFastGUI.py` — **Main GUI**: PyQt5 application supporting both CPU and GPU solvers, xy and xV scans, real-time parameter adjustment. Primary interactive tool.
- `/home/prokop/git/ppafm/tests/ChargeRings/CombinedChargeRingsGUI_pauli.py` — Extended GUI with Pauli solver integration, line scan export, state decomposition visualization.
- `/home/prokop/git/ppafm/tests/ChargeRings/GUITemplate.py` — **Reusable GUI framework**: `PlotConfig`, `PlotManager`, `GUITemplate` base class for all GUIs. Handles parameter widgets, plot management, animations.
- `/home/prokop/git/ppafm/tests/ChargeRings/ChargeRingsGUI_python.py` — Python-backend-only GUI (no C++ Pauli solver, uses older occupancy model).
- `/home/prokop/git/ppafm/tests/ChargeRings/ChargeRingsOrbitalGUI.py` — GUI variant for orbital/photon map calculations.

### Supporting Modules

- `/home/prokop/git/ppafm/tests/ChargeRings/TipMultipole.py` — Python tip multipole energy calculation: `multipole_energy`, `compute_site_energies`, `compute_site_tunelling`, `makeCircle`, `makeRotMats`, `makePosXY`, `compute_V_mirror`, `occupancy_FermiDirac`.
- `/home/prokop/git/ppafm/tests/ChargeRings/orbital_utils.py` — Orbital loading from Gaussian cube files, 2D photon map generation using `pyProbeParticle.photo`.
- `/home/prokop/git/ppafm/tests/ChargeRings/exp_utils.py` — Experimental data loading from `exp_rings_data.npz`, interpolation, denoising, extraction of line scans from 2D maps.
- `/home/prokop/git/ppafm/tests/ChargeRings/plot_utils.py` — Plotting helpers: `plot_imshow` with auto-diverging colormap detection.
- `/home/prokop/git/ppafm/tests/ChargeRings/colormaps.py` — Custom matplotlib colormaps (e.g., `PuRdR-w-BuGn`).
- `/home/prokop/git/ppafm/tests/ChargeRings/charge_rings_core.py` — Core calculation functions for older GUIs (tip potential, QD system).
- `/home/prokop/git/ppafm/tests/ChargeRings/charge_rings_plotting.py` — Plotting helpers for older GUIs.

### Fitting / Optimization

- `/home/prokop/git/ppafm/tests/ChargeRings/fit_sim_exp_general.py` — **Fitting entry point**: Monte Carlo optimization of simulation parameters against experimental STM/dI/dV data.
- `/home/prokop/git/ppafm/tests/ChargeRings/MonteCarloOptimizer.py` — `MonteCarloOptimizer` class and `FitDatasetManager` for parameter sweeps and run management.
- `/home/prokop/git/ppafm/tests/ChargeRings/fitting_plots.py` — Plotting for optimization progress and comparison results.
- `/home/prokop/git/ppafm/tests/ChargeRings/wasserstein_distance.py` — 1D/2D Wasserstein distance for image comparison (used as fitting metric).
- `/home/prokop/git/ppafm/tests/ChargeRings/distance_metrics.py` — Additional distance metrics (Wasserstein along voltage axis, cross-correlation).

### Validation / Comparison

- `/home/prokop/git/ppafm/tests/ChargeRings/compare_pme_solvers.py` — Compare CPU vs GPU PME solvers using shared params.json and geometry files.
- `/home/prokop/git/ppafm/tests/ChargeRings/compare_cpu_gpu_bias.py` — Compare CPU/GPU currents at fixed bias voltage.
- `/home/prokop/git/ppafm/tests/ChargeRings/compare_cpu_gpu_line.py` — Compare CPU/GPU currents along a line scan.

### Documentation & User Guides

- `/home/prokop/git/ppafm/tests/ChargeRings/CONVENTIONS.md` — Coding conventions (minimal interference, formatting).
- `/home/prokop/git/ppafm/tests/ChargeRings/Ruslan_PTCDA.md` — **User guide**: NTCDA brickwall Coulomb study plan — geometry files, scan protocol, parameter conventions, solver modes, output organization.
- `/home/prokop/git/ppafm/tests/ChargeRings/fit_dataset.md` — Fitting dataset documentation: how to organize experimental vs simulated data for fitting.
- `/home/prokop/git/ppafm/tests/ChargeRings/temperature.md` — **User guide**: analysis of temperature units propagation from GUI through `pauli_scan.py` to C++ `pauli.hpp` Fermi-Dirac statistics. Documents a known issue where GUI temperature may not reach the C++ solver correctly.
- `/home/prokop/git/ppafm/tests/ChargeRings/qmeq_integration_notes.md` — **User guide**: how to recalculate ChargeRings results using the external QmeQ (Quantum Master Equation) library. Documents required inputs, data sufficiency, and standalone cluster execution.
- `/home/prokop/git/ppafm/tests/ChargeRings/monte_carlo_optimization_plan.md` — **User guide**: plan for Monte Carlo parameter optimization against experimental STM images using Wasserstein distance. References key files and functions.
- `/home/prokop/git/ppafm/tests/ChargeRings/load_data_implementation_plan.md` — Implementation plan for loading saved simulation data (NPZ/JSON) back into the GUI.
- `/home/prokop/git/ppafm/tests/ChargeRings/manifest_CombinedChargeRingsGUI.md` — Short manifest of planned features and known problems for the CombinedChargeRingsGUI.
- `/home/prokop/git/ppafm/tests/ChargeRings/plot_exp_3D.md` — **User guide**: how to visualize experimental 3D STM/dI/dV data using Plotly isosurfaces and Mayavi volume rendering.
- `/home/prokop/git/ppafm/tests/ChargeRings/plot_exp_denoise.md` — **User guide**: denoising methods for low-resolution experimental 3D STM data (bilateral filter, non-local means, TV denoising via scikit-image).

### Data Files (Essential)

- `/home/prokop/git/ppafm/tests/ChargeRings/exp_rings_data.npz` — Experimental STM/dI/dV ring data (29 MB).
- `/home/prokop/git/ppafm/tests/ChargeRings/QD.cub` — Quantum dot orbital Gaussian cube file (7.6 MB).
- `/home/prokop/git/ppafm/tests/ChargeRings/Ruslan_long.txt` — NTCDA geometry: 2 molecules, long separation.
- `/home/prokop/git/ppafm/tests/ChargeRings/Ruslan_short.txt` — NTCDA geometry: 2 molecules, short separation.
- `/home/prokop/git/ppafm/tests/ChargeRings/Ruslan_kite.txt` — NTCDA geometry: 4 molecules, kite/brickwall pattern.
- `/home/prokop/git/ppafm/tests/ChargeRings/dimer.txt` — Dimer geometry file.
- `/home/prokop/git/ppafm/tests/ChargeRings/hexamer.txt` — Hexamer geometry file.
- `/home/prokop/git/ppafm/tests/ChargeRings/example_pauli_params.json` — Example parameter JSON for PME solver.

## Dependency Graph

```
C++ Core                    Python Bindings              Application Layer
─────────────               ─────────────────             ──────────────────
pauli.hpp                   pauli.py (ctypes)             pauli_scan.py (engine)
  ├─ TipField.h             pauli_ocl.py (PyOpenCL)       ├─ pauli_xv_sweep_cli.py
  ├─ gauss_solver.hpp       cpp_utils.py (compile)        ├─ pauli_xy_cli.py
  ├─ SVD.h                  utils.py (geometry)           ├─ sweep_pauli_xV.py
  ├─ Vec3.h, Vec2.h                                       ├─ sweep_pauli_xy.py
  ├─ Mat3.h                                               ├─ run_ntcda_*.py
  ├─ integerOps.h                                         └─ PauliFastGUI.py
  └─ print_utils.hpp                                         ├─ GUITemplate.py
                                                             ├─ TipMultipole.py
pauli_lib.cpp (C wrapper)                                   ├─ orbital_utils.py
  └─ → pauli.hpp                                             ├─ exp_utils.py
                                                             ├─ fit_sim_exp_general.py
PME.cl (OpenCL kernels)                                     │   ├─ MonteCarloOptimizer.py
  └─ → pauli_ocl.py                                          │   ├─ fitting_plots.py
                                                             │   └─ wasserstein_distance.py
                                                             ├─ compare_*.py
                                                             └─ plot_utils.py, colormaps.py
```

## External Dependencies

- **NumPy** — array operations throughout
- **SciPy** — interpolation, KDTree, linear algebra
- **Matplotlib** — all plotting
- **PyQt5** — all GUI applications
- **PyOpenCL** — GPU solver (`pauli_ocl.py`)
- **ctypes** — C++ binding (`pauli.py`)
- **OpenMP** — parallel CPU scans (`pauli_lib.cpp`)
- **GCC (C++20)** — compiles `pauli_lib.so` automatically on import

## Build Instructions

The C++ library auto-compiles on first `import pyProbeParticle.pauli`:
- Compiles `cpp/pauli_lib.cpp` → `cpp/pauli_lib.so` using `cpp_utils.compile_lib()`
- Flags: `-std=c++20 -fPIC -Ofast -march=native -fopenmp`
- No manual build step needed; alternatively `make pauli` in `cpp/`

The OpenCL kernel compiles at runtime from `cl/PME.cl` when `PauliSolverCL` is instantiated.
