# Cube Audit Validation Report (generic framework)

This report documents the **generic cube validation framework** implemented in `cube_audit.py` and the results of running it on one concrete dataset (anion, charge = -1) that contains both cube files and a PySCF log.

- **Audit script**: `tests/PhotonMap/test_indranil/interaction_framework/cube_audit.py`
- **Dataset directory audited**:
  - `/home/indranil/git/ppafm/tests/PhotonMap/test_indranil/interaction_framework/M4_S0.2_optimised_structure_wb97x_d3bj_6_31g__gpu_charge-1`

The goal is for this audit to work for **any molecule/cluster**, **any charge state**, and **any electronic-structure code**, as long as the cubes are self-consistent.

---

## 1) What the audit framework checks (generic)

### 1.1 Per-cube generic sanity checks (`report_cube`)

For any cube file:

- Detect/print:
  - grid shape (`npts`), origin, step vectors, voxel volume
  - min/max/mean/RMS and sign fractions
  - **cube type heuristic**:
    - `density` (electron density, integral ~ N_e)
    - `signed_density` (TDM / difference density / any signed density; integral often expected ~0)
    - `potential` (electrostatic potential)
    - `unknown`

- Generic warnings:
  - NaN/Inf in grid
  - atoms outside cube bounds
  - **min atom-to-box margin** (helps catch too-small boxes)
  - negative values in a cube classified as `density`
  - non-negligible integral in a cube classified as `signed_density`

### 1.2 Directory-wide summarizer (`--dir-summary`)

For large directories (many `.cube` files), `--dir-summary` prints one line per cube:

- filename, detected type, npts, integral, min, max, RMS

This is intended to quickly spot:

- “wrong units” cubes
- cubes with huge values
- signed densities that do not integrate to ~0
- potential cubes accidentally saved as densities, etc.

### 1.3 Electrostatic potential cube audit (`--audit-potential`)

Given:

- electron density cube `n(r)` (expected to integrate to N_e)
- electrostatic potential cube `V_total(r)`

The audit does:

- **Axis tail fit** along one axis (quick diagnostic):
  - fits the tail to `V(R) = beta + q/R` (offset + monopole)

- **Potential-only multipole fit** (robust):
  - sample many random boundary points
  - reject points too close to nuclei
  - fit `beta`, monopole `q`, dipole `mu`, and quadrupole `Q` directly from the potential cube values

This provides an *independent* estimate of the net charge implied by the potential cube, even when the density cube is truncated.

### 1.4 Poisson consistency audit (`--audit-poisson`)

Checks the differential relationship away from nuclei:

- `∇²V_total(r) ≈ ±4π n(r)`

The audit:

- computes a finite-difference Laplacian of the potential cube
- compares against `±4π n` (both signs)
- reports the best-matching sign and residual norms

This is generic and does not require knowing the “true” charge; it tests **internal consistency** of `V` and `n`.

---

## 2) Validation run: anion directory with PySCF log

### 2.1 Command used

```bash
python cube_audit.py \
  --dir /home/indranil/git/ppafm/tests/PhotonMap/test_indranil/interaction_framework/M4_S0.2_optimised_structure_wb97x_d3bj_6_31g__gpu_charge-1 \
  --no-dir-scan \
  --parse-log \
  --audit-potential --audit-poisson \
  --density ground_state_density.cube \
  --potential electrostatic_potential.cube \
  --expected-charge -1 \
  --audit-axis x --audit-rmin-ang 6.0 --audit-nsamples 30 \
  --audit-fit-samples 800 --audit-fit-min-atom-dist-ang 2.0 \
  --poisson-exclude-nuc-radius-ang 0.8 --poisson-edge-pad 1
```

### 2.2 Values extracted from the log (`--parse-log`)

From:

- `GPU_charge-1_20260225_215342.log`

Parsed key fields:

- `charge`: `-1`
- `molecular_charge`: `-1`
- `n_electrons`: `201`
- `expected_electrons`: `201`
- `nelec_numeric_total`: `201.00042911` (from `nelec by numeric integration = [101.00020516 100.00022395]`)

Note:

- The log also contained a line `Total electrons: 139.57`. This is **inconsistent** with the PySCF-reported electron counts above; treat it as *not the authoritative electron count*.

### 2.3 Density cube electron-count check

Using the cube grid integral:

- `∫n dV (box) = 190.74717385 e`
- `Z_total = 200`
- `q_from_box = Z_total - ∫n dV = +9.25282615 e`

Expected for the anion:

- `q_expected = -1`

So the density cube is missing electron density in the tails (finite-box truncation), causing a large spurious net charge when integrated.

### 2.4 Potential cube: inferred charge from potential-only multipole fit

From `--audit-potential` (multipole fit directly to `electrostatic_potential.cube`):

- `q_fit(pot) = -7.633098e-01 e`
- `q_fit(pot) - q_expected = +2.366902e-01 e`
- `q_fit(pot) - q_from_box = -1.001614e+01 e`

Interpretation:

- The **potential cube** implies a net charge closer to the physically expected anion (`-1`) than the density-box integral does.
- The remaining deviation from `-1` is expected because the cube box is not fully in the asymptotic far-field and because higher multipoles still contribute.

### 2.5 Poisson consistency between `V_total` and `n`

From `--audit-poisson`:

- best sign: `+4πn`
- `RMS(best resid) = 1.204438e-02 Ha/Bohr^2`
- `RMS(resid for +4πn) = 1.204438e-02 Ha/Bohr^2`
- `RMS(resid for -4πn) = 3.180875e-01 Ha/Bohr^2`

Interpretation:

- The potential cube and density cube are **differentially consistent** under the convention:
  - `∇²V_total ≈ +4π n` (away from nuclei)

This is exactly what you expect for:

- `V_total = V_nuc - V_elec`
- `∇²V_elec = -4π n`
- away from nuclei: `∇²V_total = -∇²V_elec = +4π n`

---

## 3) Conclusions for this dataset

- **Density cube is truncated** (electron tail missing):
  - leads to large spurious `q_from_box`.

- **Potential cube is much closer to the expected charge state** when analyzed via potential-only multipole fitting.

- **Poisson consistency holds strongly** for the sign convention `∇²V_total ≈ +4π n` away from nuclei.

---

## 4) Recommended usage for other molecules/codes

### Large directory quick triage

```bash
python cube_audit.py --dir /path/to/cubes --dir-summary
```

### Full density/potential cross-check

```bash
python cube_audit.py \
  --dir /path/to/cubes \
  --no-dir-scan \
  --audit-potential --audit-poisson \
  --density ground_state_density.cube \
  --potential electrostatic_potential.cube \
  --expected-charge 0 \
  --audit-axis x --audit-rmin-ang 10 --audit-nsamples 40 \
  --audit-fit-samples 1500 --audit-fit-min-atom-dist-ang 2.0 \
  --poisson-exclude-nuc-radius-ang 0.8
```

If you want the potential-based charge fit (`q_fit(pot)`) to be very accurate, generate the potential cube with a **larger margin** so the boundary samples are truly far-field.
