# 2D Interpolation of Planar Molecular Data (RBF & Kriging)

## 1. Problem Setting

We consider a planar molecule lying in the \(x,y\)-plane. At a finite set of points
\(\{\mathbf{p}_i\}_{i=1}^N\),

\[
\mathbf{p}_i = (x_i, y_i) \in \mathbb{R}^2,
\]

we have 1D scans of some potential (or any scalar field) along the \(z\)-direction:

\[
V_i(z_k) = V(x_i, y_i, z_k), \quad k = 1,\dots, N_z.
\]

The data points are *irregularly* distributed: typically above atoms, bond centers, and other chemically relevant positions. There is no underlying regular grid and no explicit neighbor topology.

Our goal is to interpolate the potential in the lateral \(x,y\)-plane so that for any pixel \(\mathbf{q} = (x,y)\) of a 2D image we can obtain an interpolated lateral value

\[
V(\mathbf{q}, z_k) \approx \tilde V(\mathbf{q}, z_k)
\]

for all \(z_k\). Because the interpolation in \(x,y\) is *linear* in the data values, we can precompute lateral interpolation weights and reuse them for all \(z\) values of a 1D scan.

---

## 2. Core Building Blocks

### 2.1 Compactly Supported Wendland C² Function

Defined in `pyProbeParticle/interpy.py` as `wendland_c2`:

```python
def wendland_c2(r, R_basis):
    r     = np.abs(r)
    mask  = r < R_basis
    t     = r[mask] / R_basis
    t1    = 1.0 - t
    t2    = t1 * t1
    t4    = t2 * t2
    out = np.zeros_like(r)
    out[mask] = t4 * (4.0 * t + 1.0)
    return out
```

Mathematically, for \(0 \le r < R_\text{basis}\):

\[
\phi(r; R_\text{basis}) = \bigl(1 - r/R_\text{basis}\bigr)^4 \bigl(4\,r/R_\text{basis} + 1\bigr),
\]

and \(\phi(r) = 0\) for \(r \ge R_\text{basis}\).

Properties:

- **Compact support**: \(\phi(r) = 0\) for \(r \ge R_\text{basis}\), which induces a natural cutoff.
- **C² smoothness**: second-order continuous derivatives.
- **Radial**: depends only on distance \(r = \|\mathbf{x} - \mathbf{x}'\|\).

This is used both as

- an **RBF basis** in `InterpolatorRBF`, and
- a **covariance kernel** \(C(r)\) in `InterpolatorKriging` (`compact_c2_covariance`).

Tunable parameter:

- **`R_basis`**: support radius of the kernel. Larger values give smoother, more global interpolation but higher cost; smaller values give more local behavior and sparser neighborhoods.

### 2.2 Pairwise Distances

Defined in `interpy.py` as `pairwise_distances`:

```python
def pairwise_distances(points1, points2):
    return np.sqrt(np.sum((points1[:, np.newaxis, :] - points2[np.newaxis, :, :])**2, axis=-1))
```

For \(\{\mathbf{p}_i\}\) and \(\{\mathbf{q}_j\}\) this computes the matrix

\[
D_{ij} = \|\mathbf{p}_i - \mathbf{q}_j\|_2.
\]

Used to build the RBF and Kriging matrices.

---

## 3. Global RBF Interpolator (`InterpolatorRBF`)

### 3.1 Setup

Class in `pyProbeParticle/InterpolatorRBF.py`:

```python
class InterpolatorRBF:
    def __init__(self, data_points, R_basis):
        self.data_points = np.asarray(data_points, dtype=float)  # (N,2)
        self.ndata       = self.data_points.shape[0]
        self.R_basis     = float(R_basis)
        ...
        distances = pairwise_distances(self.data_points, self.data_points)
        self.phi_matrix = wendland_c2(distances, self.R_basis)  # (N×N)
```

Let the data be \(\{(\mathbf{p}_i, z_i)\}_{i=1}^N\) with \(\mathbf{p}_i \in \mathbb{R}^2\), \(z_i \in \mathbb{R}\).

- Build the **RBF matrix** \(\Phi \in \mathbb{R}^{N\times N}\):

  \[
  \Phi_{ij} = \phi\bigl(\|\mathbf{p}_i - \mathbf{p}_j\|; R_\text{basis}\bigr).
  \]

- This is generally **dense**, but many entries are zero because of compact support if the point cloud is sparse relative to \(R_\text{basis}\).

### 3.2 Weight Solve (`update_weights`)

```python
self.weights = solve(self.phi_matrix, z)
```

Given data values \(z = (z_1,\dots,z_N)^T\), the weights \(w = (w_1,\dots,w_N)^T\) are obtained from

\[
\Phi\,w = z.
\]

- This is a *global* system: each new scalar field on the same geometry (e.g. a different \(z\)-slice of the potential) reuses the same \(\Phi\) but solves with a new right-hand side \(z\).

### 3.3 Evaluation (`evaluate`)

For a query point \(\mathbf{q}\), the interpolant is

\[
\tilde z(\mathbf{q}) = \sum_{i=1}^N w_i\,\phi\bigl(\|\mathbf{q} - \mathbf{p}_i\|; R_\text{basis}\bigr).
\]

Implementation uses a KD-tree for **local evaluation**:

```python
data_kdtree = KDTree(self.data_points)
neighbor_indices_list = data_kdtree.query_ball_point(query_points, r=self.R_basis)
...
phi_vals = wendland_c2(dists, self.R_basis)
interpolated_values[i] = np.sum(neighbor_weights * phi_vals)
```

- For each query point, only data points within distance \(R_\text{basis}\) are used (cutoff from compact support).

### 3.4 Tunable Parameters / Possible CLI Options

- **`R_basis`** (existing): global support radius of the RBF.
- **Kernel type** (future): currently fixed to Wendland C². Could be extended by adding options:
  - Wendland C⁰, C⁴, etc. (different smoothness & support polynomials).
  - Gaussian: \(\phi(r) = \exp[-(r/\epsilon)^2]\).
  - Multiquadric: \(\phi(r) = \sqrt{1 + (r/\epsilon)^2}\).
  - Inverse multiquadric, thin-plate spline, etc.
- **Regularization** (future): solve \((\Phi + \lambda I)w = z\) to improve conditioning; expose \(\lambda\) as a parameter for noisy data.
- **Neighborhood restriction** (future): evaluation already uses \(r \le R_\text{basis}\), but one could add:
  - minimum number of neighbors,
  - maximum number of neighbors (k-NN),
  - separate `R_eval` vs `R_basis`.

CLI flags could look like:

- `--interpolator rbf`
- `--rbf-kernel wendland_c2` (later also `gaussian`, `mq`, ...)
- `--rbf-R 1.0`
- `--rbf-lambda 1e-6` (Tikhonov regularization).

---

## 4. Global Ordinary Kriging (`InterpolatorKriging`)

### 4.1 Setup

Class in `pyProbeParticle/InterpolatorKriging.py`:

```python
class InterpolatorKriging:
    def __init__(self, data_points, R_basis):
        self.data_points = np.asarray(data_points, dtype=float)  # (N,2)
        self.ndata       = self.data_points.shape[0]
        self.R_basis     = float(R_basis)
        ...
        distances = pairwise_distances(self.data_points, self.data_points)
        covariance_matrix = compact_c2_covariance(distances, self.R_basis)
        self.kriging_matrix = np.zeros((self.ndata + 1, self.ndata + 1))
        self.kriging_matrix[:N, :N] = covariance_matrix
        self.kriging_matrix[:N,  N] = 1.0
        self.kriging_matrix[ N, :N] = 1.0
        self.kriging_matrix[ N,  N] = 0.0
```

Using \(C(r) = \phi(r; R_\text{basis})\) as covariance, we build

\[
C_{ij} = C\bigl(\|\mathbf{p}_i - \mathbf{p}_j\|\bigr),
\]

and the Kriging system matrix

\[
K = \begin{bmatrix}
C & \mathbf{1} \\
\mathbf{1}^T & 0
\end{bmatrix} \in \mathbb{R}^{(N+1)\times(N+1)},
\]

where \(\mathbf{1} = (1,\dots,1)^T\).

### 4.2 Coefficient Solve (`update_weights`)

```python
rhs = np.zeros(self.ndata + 1)
rhs[:self.ndata] = z
rhs[self.ndata]  = 0.0
self.coefficients = solve(self.kriging_matrix, rhs)
```

We solve

\[
\begin{bmatrix} C & \mathbf{1} \\ \mathbf{1}^T & 0 \end{bmatrix}
\begin{bmatrix} \mathbf{c} \\ \mu \end{bmatrix}
=
\begin{bmatrix} \mathbf{z} \\ 0 \end{bmatrix},
\]

where:

- \(\mathbf{c} = (c_1,\dots,c_N)^T\) are the Kriging weights,
- \(\mu\) is the Lagrange multiplier associated with the unbiasedness constraint.

This is **ordinary Kriging** with unknown constant mean.

### 4.3 Evaluation (`evaluate`)

From the solution vector:

```python
c_coeffs = self.coefficients[:self.ndata]
mu       = self.coefficients[self.ndata]
...
val = mu + sum_i c_i C(||q - p_i||)
```

For any query \(\mathbf{q}\):

\[
\tilde z(\mathbf{q}) = \mu + \sum_{i=1}^N c_i\,C\bigl(\|\mathbf{q} - \mathbf{p}_i\|\bigr).
\]

- Evaluation again uses KD-tree with radius \(R_\text{basis}\), so only nearby data points contribute.

### 4.4 Tunable Parameters / Possible CLI Options

Similar to RBF, with covariance instead of basis function:

- **`R_basis`** (existing): correlation length / support.
- **Covariance model** (future): currently `compact_c2_covariance` = Wendland C².
  Possible alternatives:
  - Exponential: \(C(r) = \sigma^2 \exp(-r/a)\).
  - Gaussian: \(C(r) = \sigma^2 \exp(-(r/a)^2)\).
  - Matérn family with different smoothness parameters.
  - Other compactly supported Wendland covariances.
- **Nugget / regularization** (future): \(C \to C + \tau^2 I\) to handle noise.

Example CLI:

- `--interpolator kriging`
- `--cov-kernel wendland_c2`
- `--cov-R 1.0`
- `--cov-nugget 0.0`.

---

## 5. Mesh Refinement & Triangulation (`tests/mesh_refine.py`)

While not directly used by the RBF/Kriging interpolators, `mesh_refine.py` provides a way to generate and refine a 2D triangular mesh that can serve as an alternative interpolation backbone.

### 5.1 Equilateral Triangle Grid

Function `generate_equilateral_triangle_grid(nx, ny, L)`:

- Creates a regular 2D lattice with basis vectors
  \(\mathbf{a}_1 = (1,0)\), \(\mathbf{a}_2 = (-1/2, \sqrt{3}/2)\), scaled by \(L\).
- Returns:
  - `vertices`: array of vertex coordinates \((N,2)\),
  - `triangles`: connectivity \((M,3)\) as vertex indices.

### 5.2 Mesh Refinement by Point Insertion

Function `refine_mesh(vertices, triangles, points, tol_edge, tol_vertex_dist)`:

For each point \(\mathbf{p}\) to insert:

1. Find containing triangle via barycentric coordinates \((u,v,w)\).
2. Compute distances to the triangle vertices.
3. Decide:
   - **Snap to vertex** if \(\min\text{dist} < \text{tol_vertex_dist}\): move that vertex to \(\mathbf{p}\).
   - **Split edge** if close to one edge (one barycentric coordinate below `tol_edge`): insert a new vertex on that edge and split adjacent triangles.
   - **Triangle split** otherwise: insert vertex inside triangle and split into three.

This can be used to refine a coarse triangular mesh so that important data points (atoms, bonds) become vertices of the mesh.

### 5.3 Edge Flipping (`flip_long_edges`)

Function `flip_long_edges(vertices, triangles)`:

- Builds an edge map and, for interior edges shared by two triangles, compares the length of the shared edge vs. the opposite diagonal.
- If the diagonal is significantly shorter (`len_cd2 < len_ab2 * 0.99`), the edge is flipped to improve triangle quality (reduce long, skinny triangles).

**Use case for interpolation:** once you have a high-quality mesh whose vertices coincide with sampling points, you can use simple **barycentric interpolation** within each triangle as an alternative or complement to RBF/Kriging.

---

## 6. Test & Visualization Script (`tests/test_interpy.py`)

This script demonstrates how to:

- Initialize `InterpolatorRBF` and `InterpolatorKriging` on toy data.
- Check interpolation accuracy at data points.
- Sample and visualize the interpolated field on a regular 2D grid.

Key entry points:

- `check_interp_accuracy(interp, data_vals, points)` verifies that interpolation reproduces values at the data points.
- `plot_interpolator_grid(...)` samples the interpolator on a uniform grid and plots with `imshow`, overlaying the original data points.

This is a good starting point for testing different kernel choices and parameter settings.

---

## 7. Application to Planar Molecule + z-Scan

### 7.1 Separating Lateral and Vertical Directions

Let the measured data be \(V_i(z_k) = V(x_i,y_i,z_k)\). For each fixed \(z_k\):

1. Define **lateral data values** \(z_i^{(k)} := V_i(z_k)\).
2. Solve the global RBF or Kriging system once per \(z_k\):
   - RBF: \(\Phi w^{(k)} = z^{(k)}\).
   - Kriging: \(K [c^{(k)}; \mu^{(k)}]^T = [z^{(k)}; 0]^T\).
3. Evaluate \(\tilde V(\mathbf{q}, z_k)\) for all image pixels \(\mathbf{q}\).

This is **exactly what current classes support**: each interpolator is a purely 2D object acting on \(x,y\); \(z\) enters only through the data values.

### 7.2 Precomputing Per-Pixel Interpolation Coefficients

Because the interpolation is **linear in the data values**, we can precompute lateral interpolation coefficients for each pixel and reuse them over all \(z\) values.

#### 7.2.1 Linear RBF Combination

At a query point \(\mathbf{q}\), the RBF interpolant is

\[
\tilde z(\mathbf{q}) = \sum_{i \in N(\mathbf{q})} w_i\,\phi\bigl(\|\mathbf{q} - \mathbf{p}_i\|\bigr),
\]

where \(N(\mathbf{q})\) are neighbors within cutoff.

If we conceptually factor this as a **linear operator** acting on the data values \(z\):

- Solve once for a basis of unit vectors (or equivalently invert \(\Phi\) / \(K\)) and build a matrix \(W\) such that

  \[
  \tilde z(\mathbf{q}_j) = \sum_i W_{ji}\,z_i.
  \]

- For the z-scan: for each \(z_k\), just compute

  \[
  V(\mathbf{q}_j, z_k) \approx \sum_i W_{ji}\,V_i(z_k).
  \]

Importantly, **the neighbor structure** (from KD-trees and cutoff) is the same for all \(z_k\); only the scalar data \(z_i^{(k)}\) change. So we can:

- Precompute, for each pixel \(j\):
  - the indices of nearby data points \(i\),
  - the associated kernel values / weights.
- Store these as sparse rows of \(W\).
- For each z-slice, do a sparse matrix–vector product.

This matches your idea: *"precalculate interpolation coefficients of linear combination from the nearby points, for given pixel and then use these coefs for weighting the whole 1D line (z-scan) at once"*.

#### 7.2.2 Rational / Normalized RBF (Shepard-like, NURBS Analogy)

Right now, the RBF and Kriging implementations use **unnormalized** linear combinations.
You suggested considering **rational RBFs**, analogous to NURBS:

- **Linear RBF (current):**

  \[
  \tilde z(\mathbf{q}) = \sum_i w_i \phi_i(\mathbf{q}).
  \]

- **Rational / normalized RBF (possible extension):**

  \[
  \tilde z(\mathbf{q}) = \frac{\sum_i w_i \phi_i(\mathbf{q})}{\sum_i w_i' \phi_i(\mathbf{q})},
  \]

  or simpler Shepard-type normalization

  \[
  \tilde z(\mathbf{q}) = \frac{\sum_i \phi_i(\mathbf{q})\,z_i}{\sum_i \phi_i(\mathbf{q})}.
  \]

Here, the denominator acts like the "R" in NURBS, normalizing the influence so that interpolation behaves more like a weighted average.

For the precomputation strategy, this means we would store both numerator and denominator coefficient patterns per pixel.

---

## 8. Performance Considerations & Optimizations

### 8.1 Current Implementation

- **Setup cost (once per geometry):**
  - Build \(\Phi\) or \(C\) using `pairwise_distances` (\(O(N^2)\)).
  - Store the matrix and reuse it for different data values.
- **Per data set (e.g. per z-slice):**
  - Solve a dense linear system (\(O(N^3)\) direct, though `solve` may reuse internal factorization if reused carefully; explicit LU would give full reuse).
- **Evaluation:**
  - KD-tree search (`query_ball_point`) to find neighbors within `R_basis`.
  - Per query point: compute distances, kernel values, and a weighted sum.

### 8.2 Suggested Optimizations

1. **Precompute Evaluation Coefficients per Pixel**
   - For a fixed geometry and fixed kernel/cutoff:
     - Use KD-tree once to build, for all pixels, their neighbor lists and kernel values.
     - Assemble a sparse weight matrix \(W\) representing the linear map from data values \(z\) to interpolated values on the pixel grid.
   - Then, per z-slice, compute \(W z^{(k)}\) via sparse matvec; no per-pixel KD-tree or distance computations.

2. **Use Cutoff Distance Explicitly**
   - Already done via `R_basis` and Wendland compact support.
   - Could further restrict neighbors to e.g. k nearest within `R_basis` to bound worst-case cost when points cluster.

3. **Factorization Reuse**
   - Instead of calling `solve` repeatedly on the same matrix object, explicitly store an LU factorization once:
     - \(\Phi = L U\) or the LU of \(K\), and reuse via forward/back substitution for each \(z^{(k)}\).
   - This is conceptually already sketched in the comments (`lu_factor`, `lu_solve`).

4. **Mesh-Based Alternative**
   - Use `mesh_refine.py` to build a refined triangular mesh with vertices at data points and meaningful additional points (bond midpoints, etc.).
   - Then use piecewise-linear barycentric interpolation within each triangle.
   - Advantage: very fast evaluation once the mesh is built; disadvantage: less smooth than RBF/Kriging.

---

## 9. RBF Kernels: Current and Possible Choices

### 9.1 Currently Used

- **Wendland C²** (compact, radial, C² smooth) via `wendland_c2`.
- Used as:
  - `InterpolatorRBF`: basis function \(\phi(r)\).
  - `InterpolatorKriging`: covariance kernel \(C(r)\).

### 9.2 Possible Alternatives

To support switching kernels (e.g. via CLI):

- **Compactly supported Wendland family** (various smoothness orders C⁰, C², C⁴, ...).
- **Gaussian RBF:** \(\phi(r) = \exp\bigl(- (r/\epsilon)^2\bigr)\).
- **Multiquadric / Inverse multiquadric:** \(\phi(r) = \sqrt{1 + (r/\epsilon)^2}\), \(\phi(r) = 1/\sqrt{1 + (r/\epsilon)^2}\).
- **Thin-plate splines:** \(\phi(r) = r^2 \log r\) (if non-compact support is acceptable).
- **Shepard / normalized RBF:** using rational combinations as described above.
- For Kriging: standard geostatistical models (exponential, Gaussian, Matérn, spherical, etc.) with tunable range, sill, and nugget.

To expose these, you might define a small kernel registry in `interpy.py` and drive it from CLI flags.

---

## 10. How to Use These Tools Efficiently

1. **Choose geometry:** data points \(\{(x_i,y_i)\}\) from atoms, bonds, etc.
2. **Pick an interpolator:**
   - Use `InterpolatorRBF` for smooth, deterministic interpolation.
   - Use `InterpolatorKriging` when a geostatistical interpretation (covariance model, unbiasedness) is desired.
3. **Tune `R_basis`:**
   - Start with a radius on the order of nearest-neighbor distance to several neighbors.
   - For planar molecules, something like 1–2 bond lengths is a reasonable first guess.
4. **Precompute per-pixel weights (future extension):**
   - For fixed \(x,y\) pixel grid, build and store neighbor indices and kernel values.
   - Reuse for all z-slices of the scan.
5. **(Optional) Use mesh refinement:**
   - Build a triangular mesh covering the molecule.
   - Insert atom/bond points using `refine_mesh`.
   - Use barycentric interpolation as a fast fall-back or comparison method.

This document should give a clear overview of what is implemented now and where to plug in additional kernels, normalization schemes, and cutoffs, as well as how to adapt the existing classes to the planar molecule + z-scan workflow and prepare for the optimizations you outlined.
