# AFM H‑Bond Point Generation (`tests/Interpolation/afm_hbond_modified.py`)

This note documents how the script `afm_hbond_modified.py` generates 2D sampling points above planar molecules, and how these points appear in the `*_point_info.txt` files (e.g. `OHO-h_1_point_info.txt`).

## Overview

For each sample molecule (e.g. `OHO-h_1.xyz`) the script:

- **Finds atoms and bonds** from the input geometry.
- **Adds auxiliary “kink” atoms** for strongly bent geometries.
- **Builds ring centers** and **molecular polygons** from bonds.
- **Constructs contours** following these polygons.
- **Adds bond centers** and **contour points**.
- **Combines everything into a 2D point cloud** in the molecular plane (x,y), later used as lateral sampling positions for AFM/DFT z‑scans.

The same point cloud is used for all tips when generating z‑scan geometries.

---

## Point Construction Steps

All point generation is done in `makeSamplePoints(mol, fname=..., bPlot=True, bPointInfo=True)`.

### 1. Input Molecule & Bonds

- The molecule is loaded as a `pyBall.AtomicSystem`:
  - `mol.apos` – atomic positions (x,y,z).
  - `mol.enames` – element names (`C`, `N`, `O`, `H`, ...).
- The script calls:
  - `mol.findBonds()` – detects bonds between atoms.
  - `mol.neighs()` – builds neighbor lists.

Result:

- A list of bonds `mol.bonds` as pairs of atom indices `(i,j)`.

### 2. Kink Dummy Atoms (`kink`)

To better sample regions near strongly bent geometries (e.g. hydrogen bonds), the script calls

```python
anew, bnew = mm.makeKinkDummy(mol.apos, mol.ngs, angMin=10.0, l=1.0)
```

- `anew` – additional 2D points (in x,y) representing **kink positions**.
- `bnew` – associated bond indices for each kink.

If at least one kink point is created, the x,y point array starts as

- `apos = concat(mol.apos[:,:2], anew)` – all atom projections + kink points.

### 3. Bonds and Bond Samples

The script constructs a clean, ordered bond list

```python
bonds = [(i, j) if i < j else (j, i) for i, j in mol.bonds]

bonds_bak = bonds.copy()

bonds = bonds + bnew
```

- `bonds_bak` – original atom–atom bonds.
- `bonds` – original + auxiliary bonds involving kinks.

It then builds bond samples and collapses them into polygons via `pyBall.atomicUtils`:

```python
binds  = np.repeat(np.arange(len(bonds)), 2)  # bond index for each endpoint
bsamp  = au.makeBondSamples(bonds, apos, where=None)
centers, polygons = au.colapse_to_means(bsamp, R=0.7, binds=binds)
```

- `centers` – mean positions of grouped bond samples; effectively **ring centers / polygon centers** in 2D.
- `polygons` – for each center, a list of bond indices forming a polygon around it.

At this stage:

- `apos` – atom + kink points (2D).
- `centers` – 2D ring/polygon centers.

The combined early point set is

```python
points = concat(apos, centers[:,:2])
na  = len(apos)
nc  = len(centers)
nac = na + nc
```

### 4. Polygons, Contours, and Contour Points (`cp`)

From the polygons and bonds, the script builds contour structures using `molmesh2d` (`mm`):

```python
polys  = [set(binds[p] for p in poly) for poly in polygons]
conts, cont_closed = mm.polygonContours(polys, bonds)
cps_, cpis         = mm.controusPoints(conts, points, centers, beta=0.3)
cps                = concat(cps_, axis=0)
```

- `conts` – contours: ordered lists of bond indices tracing around polygons/rings.
- `cont_closed` – flags for whether a contour is closed.
- `cps_` – contour points per contour.
- `cpis` – for each contour point, associated **bond indices**.
- `cps` – all contour points concatenated.

Intuitively:

- **Contour points (`cp`)** lie on the *outer side of bonds*, sampling the molecular “contours” around rings/atoms.

### 5. Bond Centers (`bond`)

The script builds pure bond-center points:

```python
bss = mm.contours2bonds(conts, bonds)

bcs = au.makeBondSamples(bonds, apos, where=[0.0])
```

- `bcs` – **bond-center points** (midpoints of bonds) in 2D.
- `bss` – mapping from contours to bonds (used for indexing but not written directly).

### 6. Final Point Set and Index Ranges

The final `points` array is assembled as

```python
bss0  = len(points)         # start index of bond centers
cont0 = bss0 + len(bcs)     # start index of contour points

points = concat(points, bcs, cps)
```

Index ranges (0‑based indices inside `points`):

- `0 .. nao-1`      – **original atoms** (projections to x,y plane).
- `nao .. na-1`     – **kink points** (if any).
- `na .. nac-1`     – **ring / polygon centers** (`center`).
- `bss0 .. cont0-1` – **bond centers** (`bond`).
- `cont0 .. end`    – **contour points** (`cp`).

Here `nao = len(mol.apos)` (original atoms) and `na`, `nac`, `bss0`, `cont0` are returned in the list `ns`.

### 7. Point Types (`ptyps`) and Neighbor Info (`pngs`)

Each point gets a **type label** and an associated neighbor list:

```python
ptyps = mol.enames + ["kink"]*len(anew) + ["center"]*len(conts) + ['bond']*len(bcs) + ['cp']*len(cps)
pngs  = [ [] ]*nao + [[j] for i,j in bnew]*len(anew) + conts + bonds + cpis
```

- For the first `nao` points, `ptyps` is the element symbol (`C`, `N`, `O`, `H`, ...), `pngs` is an empty list.
- For kink points, `ptyps = "kink"`, `pngs` references their associated bonds.
- For ring centers, `ptyps = "center"`, `pngs` lists the bonds forming that polygon.
- For bond centers, `ptyps = "bond"`, `pngs` lists the corresponding atom–atom bond `(i,j)`.
- For contour points, `ptyps = "cp"`, `pngs` lists the relevant bond indices on the contour.

These `(points, ptyps, pngs)` triples are written to file by `printPointInfo`.

---

## `*_point_info.txt` File Format

Example: `tests/Interpolation/data_Mithun/points/OHO-h_1_point_info.txt`.

Each line has the form

```text
<index> <type>[x y]<neighbors>
```

Concrete examples:

- Atom centers:

  ```text
  0 C[ 0.41315336 -1.21060971][]
  1 N[-0.98362505 -1.21060971][]
  ```

  - `0`, `1` – point indices.
  - `C`, `N` – point types (atomic elements).
  - `[ 0.41315336 -1.21060971]` – `(x,y)` coordinates in the molecular plane.
  - `[]` – empty neighbor list (no extra bonds).

- Ring/polygon centers:

  ```text
  12 center[-0.3790283  0.0082955][1, 3, 10, 4, 2, 0]
  ```

  - Type `center` – ring / polygon center.
  - Coordinates in square brackets.
  - Neighbor list `[1, 3, 10, 4, 2, 0]` – indices of bonds bounding the polygon.

- Bond centers:

  ```text
  19 bond[-0.28523585 -1.21060971](0, 1)
  ```

  - Type `bond` – midpoint of bond connecting two atoms.
  - Coordinates in square brackets.
  - Neighbor specification `(0, 1)` – the atom indices forming this bond.

- Contour points:

  ```text
  31 cp[-0.80224602 -0.84493815](0, 1)
  ```

  - Type `cp` – contour point on the outer side of bond(s).
  - Neighbor spec `(0, 1)` – bond index / atom indices associated with this contour point.

The exact neighbor syntax (`[]` vs `(i, j)` vs `[i, j, ...]`) is not needed for simple interpolation; for interpolation we only require **type** and **(x,y)**.

---

## Point Types: Physical Meaning

- **Atomic points (`C`, `N`, `O`, `H`, ...)**
  - Projections of real nuclei into the molecular plane.
  - Capture strong local variations of potential directly above atoms.

- **Kink points (`kink`)**
  - Auxiliary points near sharp angles or “kinks” in the local bonding network.
  - Help to sample regions where geometry (and thus potential) may vary rapidly.

- **Ring/Polygon centers (`center`)**
  - Centers of aromatic rings or other closed polygons of bonds.
  - Provide sampling inside rings, where potential may differ from atom centers.

- **Bond centers (`bond`)**
  - Midpoints between bonded atoms.
  - Particularly important for hydrogen bonds and covalent bonds where AFM contrast is strong between atoms.

- **Contour points (`cp`)**
  - Points following the outer contour of the molecule or rings.
  - Sample regions “between” bonds or at the molecular edge to stabilize interpolation and capture contrast away from atoms.

This construction yields a chemically informed, non‑uniform set of sampling points that is well suited for RBF/Kriging interpolation of DFT z‑scan data in the x,y plane.
