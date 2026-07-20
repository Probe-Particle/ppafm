# Kiring Interpolation of AFM Data — Export Document

## Purpose & Goals

This project interpolates **DFT-calculated AFM (Atomic Force Microscopy) z-scan data** sampled at arbitrary points over molecules into a **regular 3D Cartesian grid** suitable for use with the PPAFM (Probe-Particle AFM) relaxation pipeline. The input data consists of force/energy values measured at non-uniformly distributed sampling points (atoms, bonds, critical points) at multiple z-heights. The output is a 3D grid of energy (E) and force (Fx, Fy, Fz) that can be fed into PPAFM's `relaxed_scan.py` to simulate realistic AFM images with a flexible probe particle.

The interpolation uses two methods:
1. **Kriging** (Ordinary Kriging with compact Wendland C2 covariance) — preferred, smoother results
2. **RBF** (Radial Basis Function with Wendland C2 compact support) — alternative

Both support **per-point adaptive support radii** to handle highly non-uniform sampling densities (dense near molecule, sparse in vacuum).

## General Mechanics

1. **Input data**: 
   - **Points file**: `type x y` (or `index type x y`) listing 2D sampling positions around a molecule
   - **Z-scan file**: `pNNN zMM value` — force/energy at each point index and z-height
2. **Point augmentation**: Raw sampling points are sparse around the molecule. The augmenter adds grid points in a parabolic envelope around the molecule to fill gaps and prevent edge artifacts.
3. **Interpolation setup**: Build the Kriging/RBF matrix from pairwise distances between data points using the Wendland C2 compact-support covariance function. The matrix is (N+1)x(N+1) for Kriging (with Lagrange multiplier for unbiasedness) or NxN for RBF.
4. **Per-z-slice solve**: For each z-height, solve the linear system to get interpolation coefficients, then evaluate at all grid points.
5. **Force computation**: Analytical gradients of the covariance function give Fx, Fy directly. Fz is computed by finite differences between z-slices.
6. **PPAFM integration**: The 3D grid (Fx, Fy, Fz, E) is saved in PPAFM's npy format and loaded by `perform_relaxation()` for probe-particle relaxation.

## File Listing (Core Modules & Entry Points Only)

### Core Interpolation Library (pyProbeParticle/)

- `/home/prokop/git/ppafm/pyProbeParticle/interpy.py` — **Shared basis functions**: `wendland_c2`, `wendland_c2_varR` (per-point radii), `compact_c2_covariance`, `wendland_c2_deriv`, `wendland_c2_deriv_varR` (analytical gradients), `pairwise_distances`. Foundation for both Kriging and RBF.
- `/home/prokop/git/ppafm/pyProbeParticle/InterpolatorKriging.py` — **Kriging interpolator**: `InterpolatorKriging` class with `update_weights` (solve for coefficients), `evaluate` (interpolate at query points), `evaluate_gradient` (analytical gradient for force computation). Supports global and local (KDTree) evaluation modes, per-point radii, and nugget regularization.
- `/home/prokop/git/ppafm/pyProbeParticle/InterpolatorRBF.py` — **RBF interpolator**: `InterpolatorRBF` class with same API as Kriging. Uses Wendland C2 as RBF basis. Supports per-point radii, normalized mode, and analytical gradients.

### Application Scripts (tests/Interpolation/)

- `/home/prokop/git/ppafm/tests/Interpolation/interp_zscan_to_grid.py` — **Main interpolation script**: `load_clean_points`, `load_point_info`, `load_zscan`, `build_grid`, `make_interpolator`, `auto_support_radii`, `interpolate_volume`, `plot_z_sequence`, `plot_single_slice`. CLI with options for kind (rbf/kriging), grid size, R-basis, z-range, scatter overlay. ~470 lines.
- `/home/prokop/git/ppafm/tests/Interpolation/interp_zscan_to_grid_and_ff.py` — **GridFF generator**: `interpolate_volume_and_forces` (interpolate + compute Fx,Fy,Fz,E), `save_gridff_ppafm` (save in PPAFM npy format). CLI entry point for generating PPAFM-compatible force fields from DFT data. Unit conversion: kcal/mol → eV.
- `/home/prokop/git/ppafm/tests/Interpolation/augment_points_with_grid_8.py` — **Latest point augmenter**: generates surrounding grid points using a parabolic envelope + radial layers to fill missing outer points. Key functions: `get_smooth_parabolic_envelope`, layer generation with staggering. Writes `_smooth.txt` (original + grid points) and optional `_grid_only.txt`.
- `/home/prokop/git/ppafm/tests/Interpolation/clean_point_info.py` — **Point file converter**: parses raw `*_point_info.txt` files (format: `index type[x y][]`) into clean `type x y` tables. Used as preprocessing step before augmentation/interpolation.

### End-to-End Test & Integration

- `/home/prokop/git/ppafm/tests/Interpolation/test_relax_kriging.py` — **End-to-end test**: loads points + z-scan → interpolates to GridFF → runs PPAFM relaxation (`perform_relaxation`) → plots OutFz, PPpos, comparison. Tests different klat stiffness values. Caches GridFF for re-use.
- `/home/prokop/git/ppafm/tests/Interpolation/test_gradient_HHO.py` — **Gradient validation**: compares analytical Kriging gradients against numerical derivatives. Produces E, Fx, Fy, Fz images.

### Batch Runners

- `/home/prokop/git/ppafm/tests/Interpolation/run_all_hho.py` — **Python batch runner**: processes all HHO-h-* datasets through `test_relax_kriging.py`. Supports precomputed GridFF caching and dataset prefix filtering.
- `/home/prokop/git/ppafm/tests/Interpolation/run_all_localR8.sh` — **Bash batch script**: runs kriging interpolation (R=8, nugget=0) with scatter overlay for all scans. Cleans points, interpolates, saves volumes + slices.
- `/home/prokop/git/ppafm/tests/Interpolation/plot_points_on_molecule.py` — **Visualization**: plots sampling points overlaid on molecular geometry. Uses `AtomicSystem` for molecule loading.

### Documentation & User Guides

- `/home/prokop/git/ppafm/tests/Interpolation/PPAFM_KiringInterpolation_Integration.md` — **User guide / integration guide**: detailed analysis of PPAFM system (GridUtils, HighLevel, core, ProbeParticle.cpp) and how to feed interpolated grids into the relaxation pipeline. Data formats, axis ordering, stride conventions, file responsibilities.
- `/home/prokop/git/ppafm/tests/Interpolation/BetterInterpolation_discussion.md` — **User guide / design discussion**: diagnosis of "dotted vacuum" artifacts, non-uniform sampling density problem, auto-support-radius solution, per-point radii implementation. Includes recommended CLI commands and workflow steps.
- `/home/prokop/git/ppafm/tests/Interpolation/BetterInterpolation_progress.md` — **User guide / progress & tutorial**: retrospective of development, current best parameters (global R=25 nugget=0 or local R=8 nugget=0), step-by-step tutorial for clean artifact-free interpolation, batch script guidance.

### Data Directories (Structure Reference)

- `/home/prokop/git/ppafm/tests/Interpolation/data_Mithun/` — Old dataset (has artifacts from missing outer points). Contains `points/`, `points_new/`, `points_outer/`, `scans/`, `OHO-h_1-CO_O_slice/`.
- `/home/prokop/git/ppafm/tests/Interpolation/data_Mithun_new/` — New dataset with supplemented outer points. Contains `endgroup_points/`, `results/`, `points_clean/`, `volumes/`, `slices/`.
- `/home/prokop/git/ppafm/tests/Interpolation/data_Mithun_flat` — Flattened data structure (symlink or reference to new data).

## Dependency Graph

```
Core Library                 Application Layer              Integration
──────────────               ──────────────────             ──────────────
interpy.py                   interp_zscan_to_grid.py         test_relax_kriging.py
  ├─ wendland_c2               ├─ load_clean_points            ├─ interp_zscan_to_grid_and_ff.py
  ├─ wendland_c2_varR          ├─ load_zscan                   ├─ pyProbeParticle.HighLevel
  ├─ wendland_c2_deriv         ├─ build_grid                   ├─ pyProbeParticle.GridUtils
  ├─ pairwise_distances        ├─ auto_support_radii           └─ pyProbeParticle (PPU)
                               ├─ make_interpolator
InterpolatorKriging.py        ├─ interpolate_volume
  └─ → interpy.py              └─ plot_z_sequence
                               augment_points_with_grid_8.py
InterpolatorRBF.py            clean_point_info.py
  └─ → interpy.py              
                               run_all_hho.py / run_all_localR8.sh
                               plot_points_on_molecule.py
```

## External Dependencies

- **NumPy** — array operations, linear algebra
- **SciPy** — `scipy.spatial.KDTree` (neighbor search), `scipy.linalg.solve` (linear system)
- **Matplotlib** — plotting (slices, scatter overlay, comparison)

### PPAFM Integration Dependencies (for relaxation only)

- `/home/prokop/git/ppafm/pyProbeParticle/GridUtils.py` — `save_vec_field`, `save_scal_field` for PPAFM format output
- `/home/prokop/git/ppafm/pyProbeParticle/HighLevel.py` — `perform_relaxation` for probe-particle relaxation
- `/home/prokop/git/ppafm/pyProbeParticle/core.py` — ctypes interface to C++ `ProbeParticle.cpp`
- `/home/prokop/git/ppafm/pyProbeParticle/common.py` — parameter management (`PPU.params`)

These are **not needed** for the interpolation itself — only for the end-to-end PPAFM relaxation test (`test_relax_kriging.py`).

## Data Formats

### Input: Points file
```
type x y           (clean format)
index type x y     (with index)
```
Or raw `*_point_info.txt`:
```
0 C[ 0.413 -1.211][][]
```

### Input: Z-scan file
```
p001 z00  -0.123456
p001 z01  -0.122345
...
pNNN zMM   value
```

### Output: GridFF (PPAFM npy format)
- `prefix_FF_x.npy`, `prefix_FF_y.npy`, `prefix_FF_z.npy` — force components [nz, ny, nx]
- `prefix_E.npy` — energy [nz, ny, nx]
- `lvec` — 4x3 lattice vectors defining the grid cell
- Axis ordering: **[z, y, x]** (Fortran-order convention from PPAFM)
- Units: eV (energy), eV/Å (force) — converted from kcal/mol input

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--kind` | `rbf` | Interpolator: `rbf` or `kriging` |
| `-r / --R-basis` | 1.2 | Support radius for covariance function |
| `--kriging-nugget` | 0.0 | Diagonal regularization for Kriging matrix |
| `--kriging-global` | 0 | 1=global evaluation (no KDTree cutoff) |
| `--autoR-k` | 0 | >0: auto-compute R from k-th neighbor distance |
| `--autoR-scale` | 1.3 | Multiplier for auto-computed R |
| `-s / --z0` | 1.6 | Starting z-height (Å) |
| `-d / --dz` | 0.1 | Z-step (Å) |
| `--dx / --dy` | None | XY grid spacing (overrides nx/ny) |

## Recommended Workflow

1. **Clean points**: `python clean_point_info.py raw_point_info.txt clean_points.txt`
2. **Augment**: `python augment_points_with_grid_8.py -p points_dir -o outer_dir --new-dir new_dir`
3. **Interpolate**: `python interp_zscan_to_grid.py -k kriging -r 8.0 --kriging-nugget 0.0 -p points_smooth.txt -z scan.dat -o volume.npy`
4. **Generate GridFF**: `python interp_zscan_to_grid_and_ff.py -k kriging -r 8.0 --out-ppafm-prefix out_prefix`
5. **Relax (optional)**: `python test_relax_kriging.py --points_file ... --zscan_file ...`
