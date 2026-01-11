# Interpolating DFT z‑scans to a Regular 3D Grid

This tutorial explains how to go from raw DFT data (irregular lateral sampling + z‑scans) to a regular 3D grid that can be used for AFM/STM analysis.

It uses two small helper scripts in `tests/Interpolation/`:

- `clean_point_info.py` – parses `*_point_info.txt` from the DFT workflow and produces a clean list of lateral sampling points.
- `interp_zscan_to_grid.py` – takes the clean points and the z‑scan file and interpolates onto a regular `(x, y, z)` grid using RBF or Kriging.

The target audience is someone who already has DFT z‑scan data but has not worked with this interpolation tooling before.

---

## 1. Input Files

You need two kinds of inputs, typically under `tests/Interpolation/data_Mithun/`:

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

The scripts will convert this irregular `{point, z}` data into a regular 3D numpy array `vol[nz, ny, nx]`.

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

Once you have a clean points file and the z‑scan data, you can interpolate onto a regular 3D grid.

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
   - Calls `update_weights()` with the 1D data for this z.
   - Evaluates the interpolator on all grid points.
   - Reshapes the result to `(ny, nx)` and stores into `vol[iz, :, :]`.

5. **Save the 3D volume** to `--out-npy` as `vol[nz, ny, nx]`.

6. **Optional visualization**:

   - Either plots a **single slice** near a specified z value (`--plot-slice-z`, with `--show` to display or `--save-prefix` to write PNGs).
   - Or plots a **sequence of z heights** between `--zmin` and `--zmax` (step `--zstep`), with linear interpolation between the stored z slices.

### 3.2 Command‑line options

Run from `tests/Interpolation/` (the script ships with sensible defaults so `python interp_zscan_to_grid.py` works out of the box, but you usually override the paths/resolution explicitly):

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

Key flags:

- `--points`: path to clean (or indexed) points file.
- `--zscan`: path to DFT z‑scan file.
- `--out-npy`: output `.npy` file for the 3D volume.
- `--nx`, `--ny`: lateral grid resolution.
- `--nz`: optional limit on number of z steps used (otherwise all from input).
- `--z0`, `--dz`:
  - `z0`: starting z coordinate (offset).
  - `dz`: spacing between z layers in the output grid.
- `--R-basis`: support radius for the interpolation kernel (RBF or Kriging).
- `--kind`: `rbf` or `kriging`.
- `--plot-slice-z`: optional physical z at which to show/plot a single slice.
- `--zmin`, `--zmax`, `--zstep`: optional parameters to plot a sequence of z slices (linearly interpolated in z).
- `--show`: actually show matplotlib windows for the requested plots.
- `--save-prefix`: directory prefix for saving PNG slices; if set, images are saved instead of or in addition to showing.

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

## 4. Choosing Between RBF and Kriging

Both interpolators share the same interface but behave differently:

- **RBF (`--kind rbf`)**
  - Uses a compactly supported Wendland C² kernel `wendland_c2(r, R_basis)`.
  - Interpolation is based on solving `Phi * w = z` for each z‑slice and then evaluating `sum_i w_i * phi(||q - p_i||)`.
  - Sensitive to the choice of `R_basis` and kernel shape.

- **Kriging (`--kind kriging`)**
  - Uses the same Wendland C² as a covariance kernel.
  - Solves a kriging system to obtain weights that tend to behave more robustly in inter‑stationary regions.
  - Often gives smoother and more physically plausible behavior between data points.

A good workflow is:

1. Start with **Kriging** and tune `R_basis` to get a reasonable field.
2. Compare to **RBF** with the same `R_basis`.
3. Use visual inspection of slices (`--show` or `--save-prefix`) to decide which behaves better for your system.

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

## 6. Suggested Workflow Summary

1. **Prepare points**:

   - Run `clean_point_info.py` on your `*_point_info.txt` to obtain
     `*_points_clean.txt` and (optionally) `*_points_indexed.txt`.

2. **Inspect points** (optional but recommended):

   - Plot the `x, y` positions from the clean file to verify the sampling pattern (atoms, bonds, centers, etc.).

3. **Interpolate with Kriging**:

   - Run `interp_zscan_to_grid.py` with `--kind kriging` and a reasonable `--R-basis`.
   - Save the 3D volume to `.npy` and visualize a few slices.

4. **Compare with RBF**:

   - Run the script with `--kind rbf` and the same `R-basis`.
   - Compare slices to identify differences, especially in inter‑stationary regions.

5. **(Advanced) Tune RBF kernel / normalization**:

   - If needed, adjust `InterpolatorRBF` construction to use non‑default `C_peak`, `normalized=True`, and a small `eps_norm`.
   - Use this to reduce sensitivity to the exact kernel shape and improve behavior between sampling points.

Following these steps, you can go from raw DFT z‑scan output to a clean, regular 3D representation suitable for further AFM/STM analysis and visualization.
