# Interpolating DFT z‑scans to a Regular 3D Grid

This tutorial explains how to go from raw DFT data (irregular lateral sampling + z‑scans) to a regular 3D grid that can be used for AFM/STM analysis. It reflects the latest fixes for avoiding vacuum bumps via adaptive radii, optional global kriging, and a scatter overlay for checking data fidelity.

It uses two helper scripts in `tests/Interpolation/`:

- `clean_point_info.py` – parses `*_point_info.txt` from the DFT workflow and produces a clean list of lateral sampling points.
- `interp_zscan_to_grid.py` – takes the clean points and the z‑scan file and interpolates onto a regular `(x, y, z)` grid using RBF or Kriging, with optional adaptive radii, nugget, global evaluation, and scatter overlays.

The target audience is someone who already has DFT z‑scan data but has not worked with this interpolation tooling before.

---

## 1. Input Files

You need two kinds of inputs, typically under `tests/Interpolation/data_Mithun_new/` (new data with supplemented outer points) or `data_Mithun/` (legacy):

- **Point info file** (irregular lateral sampling positions):

  - Example: `data_Mithun/points/OHO-h_1_point_info.txt`
  - Contains one line per sampling point, with an index, a point “type” (atom, bond, center, etc.), and coordinates inside square brackets:
    - e.g. `0 C[ 0.41315336 -1.21060971][]`

- **z‑scan file** (1D scans for each point):

  - Example: `data_Mithun/scans/OHO-h_1-CO_O.dat`
  - Each line has the form:

    ```
    pNNN zMM value
    ```

    where:

    - `pNNN` is the point index (matching the ordering of the cleaned points file),
    - `zMM` is the z‑index (integer step),
    - `value` is the scalar quantity (e.g. potential, energy, field component).

The scripts convert this irregular `{point, z}` data into a regular 3D numpy array `vol[nz, ny, nx]`.

---

## 2. Cleaning the Point Info File (`clean_point_info.py`)

`clean_point_info.py` turns a verbose `*_point_info.txt` into a compact table of lateral coordinates:

- **Input**: `*_point_info.txt`
- **Output (clean)**: text file with header `type x y`
- **Output (indexed)**: text file with header `index type x y`

### 2.1 Script behavior

The parser does the following for each non‑empty line:

- Strips the leading index.
- Extracts the point “type” (e.g. `C`, `center`, `bond`, ...).
- Reads the first coordinate bracket `[ x y ]` and takes the first two numbers as `x` and `y`.
- Ignores any extra information (neighbor lists, atom indices, etc.).

If a line cannot be parsed, it is skipped. If no valid rows are found, the script raises an error.

### 2.2 Command‑line usage

From the repository root:

```bash
cd tests/Interpolation

# 1) Create an indexed version (keeps an explicit point index column)
python clean_point_info.py \
    data_Mithun/points/OHO-h_1_point_info.txt \
    data_Mithun/points/OHO-h_1_points_indexed.txt \
    --include-index

# 2) Create a clean version (only type and coordinates)
python clean_point_info.py \
    data_Mithun/points/OHO-h_1_point_info.txt \
    data_Mithun/points/OHO-h_1_points_clean.txt
```

The **clean** file is what `interp_zscan_to_grid.py` expects via `--points`.

- Example header of the clean file:

  ```text
  type x y
  C 0.41315336 -1.21060971
  O -0.372... 0.000...
  ...
  ```

- Example header of the indexed file:

  ```text
  index type x y
  0 C 0.41315336 -1.21060971
  1 O -0.372... 0.000...
  ...
  ```

Both formats are supported by the loader in `interp_zscan_to_grid.py`.

---

## 3. Interpolating to a Regular 3D Grid (`interp_zscan_to_grid.py`)

Once you have a clean points file and the z‑scan data, you can interpolate onto a regular 3D grid. The key to avoiding “dotted” vacuum artifacts is to ensure sufficient kernel overlap (larger R or adaptive radii) or use the global/dense kriging evaluation.

### 3.1 What the script does

`interp_zscan_to_grid.py` performs the following steps:

1. **Load lateral points** via `--points`:

   - Supports headers `type x y` (clean file) or `index type x y` (indexed file).
   - Reads all rows into an `(N, 2)` array of `(x, y)` coordinates.

2. **Load z‑scan data** via `--zscan`:

   - Reads all lines `pNNN zMM value`.
   - Builds an array `values[n_points, n_z]`, ordered by increasing point index and z‑index.
   - Checks that every `pNNN zMM` combination is present.

3. **Build a regular grid** in lateral coordinates:

   - Extracts `x_min, x_max, y_min, y_max` from the input points.
   - Adds a small padding (default `pad = 0.1` of the lateral box size).
   - Generates regular `xs`, `ys` with `nx`, `ny` samples using `np.linspace`.
   - Forms a flat list of 2D grid points `(x, y)` for interpolation.

4. **Interpolate for each z‑slice**:

   - For each z index `iz`, takes the corresponding `values[:, iz]`.
   - Uses either:
     - `InterpolatorRBF` (radial basis functions; `--kind rbf`), or
     - `InterpolatorKriging` (kriging; `--kind kriging`).
   - Optional robustness knobs:
     - Adaptive radii: `--autoR-k`, `--autoR-scale`, `--autoR-percentile`, `--autoR-rmin`, `--autoR-rmax` (per-point or global radius from k-NN distances).
     - Nugget for kriging: `--kriging-nugget` (diagonal regularization; 0 keeps exact fit).
     - Global evaluation: `--kriging-global 1` (dense evaluation; choose large R to act truly global).
     - Scatter overlay: `--scatter-overlay 1` to draw raw points on saved PNGs with the same colormap limits.
   - Calls `update_weights()` with the 1D data for this z.
   - Evaluates the interpolator on all grid points.
   - Reshapes the result to `(ny, nx)` and stores into `vol[iz, :, :]`.

5. **Save the 3D volume** to `--out-npy` as `vol[nz, ny, nx]`.

6. **Optional visualization**:

   - Either plots a **single slice** near a specified z value (`--plot-slice-z`, with `--show` to display or `--save-prefix` to write PNGs).
   - Or plots a **sequence of z heights** between `--zmin` and `--zmax` (step `--zstep`), with linear interpolation between the stored z slices.

### 3.2 Command‑line options

Run from `tests/Interpolation/` (defaults work, but override paths/resolution explicitly):

```bash
python interp_zscan_to_grid.py \
    --points data_Mithun/points/OHO-h_1_points_clean.txt \
    --zscan  data_Mithun/scans/OHO-h_1-CO_O.dat \
    --out-npy data_Mithun/OHO-h_1-CO_O_volume_rbf.npy \
    --nx 50 --ny 50 \
    --kind rbf --R-basis 1.2 \
    --z0 1.6 --dz 0.1 \
    --zmin 2.0 --zmax 4.0 --zstep 0.1 \
    --save-prefix data_Mithun/OHO-h_1-CO_O_slice
```

Key flags (common):

- `--points`: path to clean (or indexed) points file.
- `--zscan`: path to DFT z‑scan file.
- `--out-npy`: output `.npy` file for the 3D volume.
- `--nx`, `--ny` or `--dx`, `--dy`: grid resolution or spacing.
- `--nz`, `--z0`, `--dz`/`--dz-grid`: z sampling.
- `--kind`: `rbf` or `kriging`.
- `--R-basis`: support radius (scalar) unless adaptive radii are enabled.
- Adaptive radii: `--autoR-k`, `--autoR-scale`, `--autoR-percentile`, `--autoR-rmin`, `--autoR-rmax`.
- Kriging robustness: `--kriging-nugget`, `--kriging-global`.
- Visualization: `--plot-slice-z` or `--zmin/--zmax/--zstep`, `--show`, `--save-prefix`.
- Scatter check: `--scatter-overlay 1 --scatter-size 10 --scatter-skip 1` (uses same vmin/vmax as the imshow).

A second example (different `R_basis`):

```bash
python interp_zscan_to_grid.py \
    --points data_Mithun/points/OHO-h_1_points_clean.txt \
    --zscan  data_Mithun/scans/OHO-h_1-CO_O.dat \
    --out-npy data_Mithun/OHO-h_1-CO_O_volume_rbf_R2p0.npy \
    --save-prefix data_Mithun/OHO-h_1-CO_O_slice_R2p0 \
    --nx 50 --ny 50 \
    --kind rbf --R-basis 2.0 \
    --z0 1.6 --dz 0.1 \
    --zmin 2.0 --zmax 4.0 --zstep 0.1
```

---

## 4. Choosing Between RBF and Kriging (and what fixes “dotted” vacuum)

Both interpolators share the same interface but behave differently:

- **RBF (`--kind rbf`)**
  - Uses a compactly supported Wendland C² kernel `wendland_c2(r, R_basis)`.
  - Interpolation is based on solving `Phi * w = z` for each z‑slice and then evaluating `sum_i w_i * phi(||q - p_i||)`.
  - Sensitive to the choice of `R_basis` and kernel shape.

- **Kriging (`--kind kriging`)**
  - Uses the same Wendland C² as a covariance kernel.
  - Solves a kriging system to obtain weights that tend to behave more robustly in inter‑stationary regions.
  - Often gives smoother and more physically plausible behavior between data points.

Practical guidance to remove bumps in vacuum:

1) Ensure overlap: use a **larger R** or adaptive radii so outer points overlap several neighbors (k≈6). Small R produces isolated bumps.
2) If you want a reference “ground truth”: use **global kriging** (`--kriging-global 1`) with a large R (e.g., 20–25 Å) and nugget=0. This matches the smooth baseline we observed (`global_R25_nug0`).
3) A fast and good local setting observed: **local kriging, R=8.0, nugget=0, dx=dy=dz=0.10**, with scatter overlay to confirm fidelity (`local_R8_nug0`).
4) Nugget can smooth spikes if needed; start with 1e-3–1e-2.
5) Use scatter overlays to verify the interpolant passes through raw points without rims.

---

## 5. Advanced: RBF Normalization and Kernel Peak Parameter

The underlying RBF implementation (`pyProbeParticle/InterpolatorRBF.py`) provides two advanced tuning knobs:

- A **kernel peak parameter** `C_peak` for the Wendland function.
- An optional **normalized RBF** evaluation.

### 5.1 Wendland C² with configurable peak

The Wendland C² kernel is implemented as:

```python
wendland_c2(r, R_basis, C=1.0)
```

For `C = 1.0` this reproduces the standard Wendland C²:

\[
\phi(r) = (1 - r/R)^4 (4 r / R + 1), \quad 0 \le r < R
\]

`C` mainly changes the value near the center (`r ≈ 0`), i.e. how strongly each data point influences its immediate neighborhood, while the cutoff behavior remains almost unchanged.

In `InterpolatorRBF`, this is exposed as `C_peak` in the constructor.

### 5.2 Normalized RBF (Shepard‑like) with regularization

Evaluation in `InterpolatorRBF` can optionally use a normalized form:

- **Unnormalized (default)**:

  \[
  E(\mathbf{q}) = \sum_i w_i \, \phi(||\mathbf{q} - \mathbf{p}_i||)
  \]

- **Normalized (optional)**:

  \[
  E(\mathbf{q}) = \frac{\sum_i w_i \, \phi(||\mathbf{q} - \mathbf{p}_i||)}{S(\mathbf{q}) + \varepsilon},
  \quad S(\mathbf{q}) = \sum_i \phi(||\mathbf{q} - \mathbf{p}_i||)
  \]

  where `eps_norm = ε` is a small regularization constant to avoid division by very small `S(\mathbf{q})` far from all data points.

This makes the interpolated value depend more on the **relative** kernel values between neighbors and less on the absolute magnitude of the kernel, which can improve behavior in regions between sampling points.

At the moment, these options are controlled at the `InterpolatorRBF` level. The CLI `interp_zscan_to_grid.py` script currently constructs `InterpolatorRBF(points_xy, R_basis)` with its default behavior (no normalization, `C_peak = 1.0`), so the tutorial above matches the default setup.

If you need to experiment with these advanced options, you can:

- Modify `make_interpolator()` in `interp_zscan_to_grid.py` to pass custom `C_peak`, `normalized`, and `eps_norm` into `InterpolatorRBF`.
- Re‑run the script and compare slices visually to the Kriging results.

---

## 6. Suggested Workflow (concise)

1) **Clean points**: `python clean_point_info.py endgroup_points/<TIP>_point_info.txt points_clean/<TIP>_points_clean.txt`

2) **Pick settings** (recommendation):
   - Fast/local: `--kind kriging -r 8.0 --kriging-nugget 0 --dx 0.10 --dy 0.10 --dz-grid 0.10`
   - Reference/global: `--kriging-global 1 -r 25.0 --kriging-nugget 0`
   - Adaptive option: `--autoR-k 6 --autoR-scale 1.8` (per-point radii) or `--autoR-percentile 90 --autoR-scale 3.0` (single large R)
   - Scatter check: `--scatter-overlay 1 --scatter-size 10 --scatter-skip 1`

3) **Run one scan** (example): see Section 3.2; adjust `--points`, `--zscan`, and output paths.

4) **Batch all scans** (example script): `tests/Interpolation/run_all_localR8.sh` (R=8, nugget=0, scatter overlay).
   - Outputs: `data_Mithun_new/volumes/local_R8/` and `data_Mithun_new/slices/local_R8/`
   - Robustness: if any `.dat` has missing z entries, either fix the file or modify the script to skip on error (wrap the python call with `|| { echo "[WARN]..."; continue; }`).

5) **Inspect**: open a few `slice_z*.png` (with scatter overlay) to ensure the interpolant passes through data and vacuum is smooth (no isolated bumps).

This workflow yields smooth, artifact‑free interpolations while remaining fast for batch processing.
