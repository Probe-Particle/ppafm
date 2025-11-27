# NTCDA Brickwall Coulomb Study – Plan

## 0. System & Files

- **Geometry files**
  - **[Ruslan_long.txt](cci:7://file:///home/prokop/git/ppafm/tests/ChargeRings/home/prokop/git/ppafm/tests/ChargeRings/Ruslan_long.txt:0:0-0:0)** – 2 A‑type molecules, long separation (experimental “long bond”).
  - **[Ruslan_short.txt](cci:7://file:///home/prokop/git/ppafm/tests/ChargeRings/home/prokop/git/ppafm/tests/ChargeRings/Ruslan_short.txt:0:0-0:0)** – 2 A‑type molecules, short separation (experimental “short bond”).
  - **[Ruslan_kite.txt](cci:7://file:///home/prokop/git/ppafm/tests/ChargeRings/Ruslan_kite.txt:0:0-0:0)** – 4 molecules: A along ±y, B along ±x (kite / brickwall patch).
- **Lattice (full system, for interpretation)**
  - Rectangular cell:  
    - `a = (11.57, 0.00, 0.0)` Å  
    - `b = (0.00, 15.04, 0.0)` Å  
  - Two molecular species:
    - **A (bright, charging)** – corners of the supercell.
    - **B (dark, essentially neutral)** – center / brickwall positions.
- **Scan protocol**
  - Tip scan along **x** through the geometric center (our line scans).
  - Tools:
    - [pauli_xy_cli.py](cci:7://file:///home/prokop/git/ppafm/tests/ChargeRings/pauli_xy_cli.py:0:0-0:0) – 2D xy maps.
    - [pauli_xv_sweep_cli.py](cci:7://file:///home/prokop/git/ppafm/tests/ChargeRings/pauli_xv_sweep_cli.py:0:0-0:0) – xV line scans and parameter sweeps.
    - [pauli_scan.py](cci:7://file:///home/prokop/git/ppafm/tests/ChargeRings/pauli_scan.py:0:0-0:0) – shared machinery (geometry, Wij, etc.).

---

## 1. Global Conventions / Parameters

- **Solvers**
  - **PME solver** (`solver_mode = 0`).
  - **Ground-state solver** (`solver_mode = -1`).
- **Coulomb coupling**
  - Parameter `W` (eV) defines **nearest-neighbour** interaction via normalized distance kernel.
  - Distance kernel options: `Wij_mode ∈ {const, dipole, coulomb, exp}`, with `Wij_power`, `Wij_beta`.
- **On-site energies**
  - Use `Esite` (global) plus per‑site offsets from geometry (4th column).
  - For A/B cases: interpret 4th column as per‑site E, or later introduce type‑based construction.
- **Output organization**
  - For each simulation, store:
    - **NPZ**: STM, dIdV, auxiliary arrays.
    - **JSON**: all parameters (including geometry file, W, solver mode, kernel params, etc.).
    - **PNG**: corresponding plots (STM maps, xV maps, line cuts).

We’ll standardize directory structure, e.g.:

- `results/NTCDA/2site_circle/...`
- `results/NTCDA/2site_Ruslan_long/...`
- `results/NTCDA/2site_Ruslan_short/...`
- `results/NTCDA/4site_kite/...`

---

## 2. 2‑Site Reference: Ideal Circle Geometry

Goal: understand baseline behavior vs **distance** and **solver** with the simplest symmetric dimer.

### 2.1 Setup

- [ ] Define a simple 2‑site circular geometry using `makeCircle`:
  - Vary distance `d` (by changing `R`).
  - Single species A (identical onsite energy).
- [ ] Choose baseline parameters:
  - `Esite_A` (e.g. −0.10 eV).
  - `W` (a few representative values).
  - `Wij_mode` (e.g. `'dipole'` or `'exp'`) and kernel parameters.
- [ ] Fix scan line: x through origin, tip height, bias range (Vmin→Vmax).

### 2.2 Distance dependence (circle dimer)

For each `d` in a small grid (e.g. 5–10 distances):

- [ ] Run PME solver (`solver_mode = 0`) xV line scan (using [pauli_xv_sweep_cli.py](cci:7://file:///home/prokop/git/ppafm/tests/ChargeRings/pauli_xv_sweep_cli.py:0:0-0:0) or direct call in a notebook).
- [ ] Run Ground-state solver (`solver_mode = -1`) for the same `d`.
- [ ] Save:
  - NPZ: STM(x,V), dIdV(x,V).
  - JSON: all parameters incl. `d`, `W`, `Wij_mode`, `Wij_beta/power`, solver mode.
  - PNG: xV color maps (STM and dIdV), and optionally 1D cuts at chosen biases.

### 2.3 Compare solvers (circle dimer)

- [ ] For each `d`, plot PME vs GS:
  - Overlay 1D cuts (e.g. STM at fixed V, or dIdV at resonance).
  - Mark systematic differences (e.g. peak position, width, height).
- [ ] Summarize in a small table: for each `d`, key metrics (peak bias, splitting, etc.).

---

## 3. 2‑Site: Ruslan Long / Short Dimers (A‑type only)

Goal: map how **Coulomb coupling W** influences the experimentally motivated long and short dimers.

### 3.1 Setup for [Ruslan_long.txt](cci:7://file:///home/prokop/git/ppafm/tests/ChargeRings/home/prokop/git/ppafm/tests/ChargeRings/Ruslan_long.txt:0:0-0:0) and [Ruslan_short.txt](cci:7://file:///home/prokop/git/ppafm/tests/ChargeRings/home/prokop/git/ppafm/tests/ChargeRings/Ruslan_short.txt:0:0-0:0)

- [ ] Use `geometry_file = Ruslan_long.txt` and [Ruslan_short.txt](cci:7://file:///home/prokop/git/ppafm/tests/ChargeRings/home/prokop/git/ppafm/tests/ChargeRings/Ruslan_short.txt:0:0-0:0); ensure:
  - Per‑site `Esite` are taken from 4th column.
  - Scan line along x through center (already default in current CLI).
- [ ] Fix:
  - Solver modes: PME (0) and GS (−1).
  - Distance kernel (`Wij_mode`, `Wij_beta/power`).

### 3.2 W‑dependence scans (2‑site Ruslan dimers)

For each geometry `{long, short}`:

- [ ] Choose a set of `W` values (e.g. 0, 0.01, 0.02, 0.05, 0.1 eV).
- [ ] For each `W`:
  - [ ] Run PME xV scan.
  - [ ] Run GS xV scan.
  - [ ] Save NPZ / JSON / PNG into `results/NTCDA/2site_{long|short}/W_{value}/solver_{mode}/...`.

### 3.3 Analysis

- [ ] For each geometry, plot how spectra change with `W`:
  - xV maps for a grid of W.
  - Extract quantities: energy shifts, splitting between states, asymmetry (only one molecule charging).
- [ ] Compare long vs short:
  - [ ] Identify W range where only one site charges vs both.
  - [ ] Relate to real lattice distances and charge localization observed by Ruslan.

---

## 4. 4‑Site Kite Geometry – A/B System ([Ruslan_kite.txt](cci:7://file:///home/prokop/git/ppafm/tests/ChargeRings/Ruslan_kite.txt:0:0-0:0))

Goal: use the 4‑site kite to explore how **A/B onsite energies** and **Coulomb couplings** lead to only A charging.

### 4.1 Setup

- [ ] Use `geometry_file = Ruslan_kite.txt`:
  - A‑like sites along ±y.
  - B‑like sites along ±x.
  - Per‑site onsite energies currently encoded in column 4 (e.g. A and B slightly different).
- [ ] Define parameterization for A/B:
  - [ ] Introduce offsets `ΔE_A`, `ΔE_B` around some base `E0`, or directly accept `E_A`, `E_B`.
  - [ ] Optionally later: derive type labels from file (e.g. we know which indices are A vs B for this specific geometry).

### 4.2 W‑dependence for kite (4 sites)

- [ ] Fix a baseline pair of (E_A, E_B) from geometry.
- [ ] Sweep `W` as for dimers:
  - Maybe different ranges for nearest neighbours vs others (via distance kernel).
- [ ] For each W:
  - [ ] Run PME xV line scan.
  - [ ] Run GS xV line scan.
  - [ ] Save NPZ / JSON / PNG: `results/NTCDA/4site_kite/W_{value}/solver_{mode}/...`.

### 4.3 On‑site energy variations (A/B contrast)

- [ ] For a selected W (or small set), sweep:
  - E.g. `E_A` fixed, vary `E_B` (or vice versa).
- [ ] For each (E_A, E_B):
  - [ ] Run PME and GS scans, save outputs like above.
- [ ] Analyze:
  - [ ] When does charge stay on A only?
  - [ ] When does B start to charge?
  - [ ] Map out a “phase diagram” in (ΔE, W) space qualitatively.

---

## 5. Organization & Reproducibility

- [ ] Define a **small helper module or notebook cells** that:
  - [ ] Construct a full parameter dict with explicit entries for:
    - Geometry file, solver mode, W, Wij_mode, Wij_beta, Wij_power, E_A/B (if used), etc.
  - [ ] Build consistent output paths from those parameters.
  - [ ] Save:
    - NPZ (`STM`, `dIdV`, `Es`, `Ts`, etc.),
    - JSON (`params_json`),
    - PNG (plots).
- [ ] Version control:
  - [ ] Record git commit hash in JSON so simulations can be tied to code state.

---

## 6. Notebook / Labbook Structure

In a Jupyter notebook (or set of notebooks) we will:

1. **Introduction & System description**
   - [ ] Summarize geometry, experimental context, and our effective model.
2. **Section 1: Circle dimer (reference)**
   - [ ] Run 2‑site circle scans and document results with inline plots.
3. **Section 2: Ruslan long/short dimers**
   - [ ] Run W‑sweeps and compare with experiment‑motivated expectations.
4. **Section 3: Kite geometry (A/B system)**
   - [ ] Explore (E_A, E_B, W) space; highlight regimes matching “only A charges”.
5. **Section 4: Summary**
   - [ ] Collect key plots and a short scientific interpretation.
