Got it. I’ll do two things:

1) Map the dependency chain from the C++ solver (`pauli.hpp` → `pauli_lib.cpp`) through the Python wrapper (`pyProbeParticle/pauli.py`) to the high-level scan utilities (`tests/ChargeRings/pauli_scan.py`) and the CLI/study scripts you listed, and then draft a clean Markdown doc explaining what each script does and how they fit together.

2) Trace the `bAux`/aux-output path specifically (the `out_prob_*`, `out_fct*`, `out_inds`, and the `current_matrix_ptr` layout), and connect those arrays to what `plot_state_scan_1d` currently plots. Then I’ll assess what needs to change (if anything) to reuse `plot_state_scan_1d` for 2- and 4-site systems and how to make state ordering/labels robust.

I’m going to inspect the relevant files now.


I’ve located the main call path for auxiliary current decomposition already (`set_current_matrix_pointer` → `scan_current_tip_` → `generate_current` with `bAux`). Next I’ll read the missing `run_ntcda_*` scripts and the key parts of `pauli.hpp` that define `generate_current`, state ordering, and index mapping, plus the two existing markdown docs—then I’ll summarize dependencies and explain exactly how the exported arrays map to plotted transitions and how to generalize `plot_state_scan_1d` safely to 2/4 sites (fixing the hard-coded reorder/labels issue).













## Dependency / responsibility map (what calls what)

### Top-level study/CLI scripts (your list)
- **`tests/ChargeRings/pauli_xy_cli.py`**
  - Builds `params` (defaults + JSON + CLI overrides).
  - Builds geometry (`make_site_geom`) and a [pauli.PauliSolver](cci:2://file:///home/prokop/git/ppafm/cpp/pauli.hpp:186:0-1589:1).
  - Calls **`pauli.run_pauli_scan_top(...)`** to do an XY grid scan.
  - Optionally computes `dIdV` by rerunning at `VBias + dQ` and finite-differencing.

- **`tests/ChargeRings/pauli_xv_sweep_cli.py`**
  - Loads params and possibly a `geometry_file`.
  - Calls into **`tests/ChargeRings/pauli_scan.py`** (line scan helpers) and uses [pyProbeParticle.pauli.PauliSolver](cci:2://file:///home/prokop/git/ppafm/cpp/pauli.hpp:186:0-1589:1).
  - It is the “batch driver” for repeated x–V line scans.

- **`tests/ChargeRings/sweep_pauli_xy.py`, `sweep_pauli_xV.py`**
  - “Example notebooks as scripts”: they mostly call high-level sweep helpers from **`pauli_scan.py`**.

- **NTCDA studies**
  - **[run_ntcda_xv_study.py](cci:7://file:///home/prokop/git/ppafm/tests/ChargeRings/run_ntcda_xv_study.py:0:0-0:0)** calls [pauli_xv_sweep_cli.load_params(...)](cci:1://file:///home/prokop/git/ppafm/tests/ChargeRings/run_ntcda_xy_study.py:20:0-26:17) then calls `xv.run_single_scan(...)` (defined inside `pauli_xv_sweep_cli.py`) to produce STM+dIdV for each `(geometry, solver_mode, W)`.
  - **[run_ntcda_xy_study.py](cci:7://file:///home/prokop/git/ppafm/tests/ChargeRings/run_ntcda_xy_study.py:0:0-0:0)** calls `pauli_xy_cli.run_xy_scan(...)` for each `(geometry, solver_mode, W, VBias)`.
  - **`*_summary.py`** scripts just load `.npz` outputs and tile plots.
  - **[run_ntcda_all_studies.py](cci:7://file:///home/prokop/git/ppafm/tests/ChargeRings/run_ntcda_all_studies.py:0:0-0:0)** just orchestrates the above.

### High-level Python library layer
- **`tests/ChargeRings/pauli_scan.py`**
  - Geometry (`make_site_geom`, `load_site_geometry`)
  - Tunneling (`generate_hops_gauss`, interpolation with orbital maps)
  - Scan drivers:
    - `calculate_1d_scan(...)` (pure 1D spatial line at fixed `VBias`)
    - `calculate_xV_scan_orb(...)` (2D: distance × VBias)
    - `run_xv_scan_case(...)`, `run_xy_scan_case(...)` (reusable scan wrappers returning a consistent dict of arrays)
    - `save_scan_case(...)` (stores params + arrays in a standard layout)
  - Analysis/plotting:
    - `plot_state_scan_1d(...)` (your many-body energies + probabilities + optional current decomposition stackplot)
    - `plot_state_probabilities_stack_1d(...)` (1D probability stackplot)
    - `plot_state_probabilities_stack_with_energies_1d(...)` (1D probability stackplot with many-body energy lines overlaid)
    - `plot_current_decomposition_stack_1d(...)` (1D stackplot of selected transition contributions)
    - `export_xv_state_and_current_decomposition_plots(...)` (exports both cut directions for xV scans)

### Python wrapper over C++
- **`pyProbeParticle/pauli.py`**
  - `ctypes` bindings into `pauli_lib.so`
  - Exposes:
    - [PauliSolver](cci:2://file:///home/prokop/git/ppafm/cpp/pauli.hpp:186:0-1589:1) object with `scan_current_tip(...)`
    - helper functions: `make_state_order(nsite)`, `make_state_labels(order)`, etc.
    - the global pointer hook: `set_current_matrix_export_pointer(...)` → C function `set_current_matrix_pointer(...)`
    - tip-orbital angular prefactor helper: `evalSitesTipsAngularFactor(...)` (see below)

### C interface + solver core
- **`cpp/pauli_lib.cpp`** (C ABI)
  - Stores global export pointers:
    - `g_current_matrix_ptr`, `g_out_prob_b_enter`, `g_out_prob_c_leave`, `g_out_fct1_b_enter`, `g_out_fct2_c_leave`, `g_out_inds`
  - In `scan_current_tip_` loop (per scan point `ip`):
    - Sets solver’s member pointers to the correct **offset** inside the big export arrays:
      - `solver->current_matrix_ptr = g_current_matrix_ptr + ip*nstate2`
      - similarly for `out_prob_*`, `out_fct*`
    - `solver->out_inds = g_out_inds` but only enables index writeout for `ip==0` using `bOutIndex` (so indices are written once; that’s fine because indices don’t depend on tip position)

- **[cpp/pauli.hpp](cci:7://file:///home/prokop/git/ppafm/cpp/pauli.hpp:0:0-0:0)** (actual solver logic)
  - [PauliSolver::generate_current(int lead_idx)](cci:1://file:///home/prokop/git/ppafm/cpp/pauli.hpp:1444:4-1505:5) is where `bAux` matters.

---

## Exported xV “cuts” and plotting (reproducible post-processing)

For xV scans (distance × voltage), there are two common 1D cuts:

- **Distance cut at fixed voltage**
  - x-axis: distance `[Å]`
  - controlled by parameter: `V_slice`

- **Voltage cut at fixed distance**
  - x-axis: voltage `[V]`
  - controlled by parameter: `x_slice`

These are exported by `export_xv_state_and_current_decomposition_plots(...)`.

### Reproducible selection of cut positions

- If `V_slice` is present in `params.json`, it is used for the distance-cut.
- If `x_slice` is present in `params.json`, it is used for the voltage-cut.
- If either is missing, the exporter falls back to a default (e.g. midpoint distance for `x_slice`).

### Overview plot with cut markers

In the NTCDA xV study script, the 2D scan overview (`scan.png`) overlays:

- a **horizontal line** at `V=V_slice`
- a **vertical line** at `x=x_slice`

This makes it obvious which 1D cuts correspond to the exported 1D plots.

### Titles and annotations

The 1D exported plots include the cut value in the title so the plots remain interpretable even when moved out of the result folder.

---

## Tip-orbital angular prefactor for tunneling (“tilted” images)

Experimental images can appear tilted (max hopping not exactly above a site). A simple and efficient model is to multiply the base tunneling by an angular factor derived from the direction between tip and site.

### Mathematical definition

Let:

- `dr = r_site - r_tip`
- `r = |dr|`
- `rhat = dr/r`

Define the tip-orbital amplitude (global Cartesian frame) as a linear combination of `s,px,py,pz`:

- `A = px*rhat.x + py*rhat.y + pz*rhat.z + s`

Then the multiplicative angular factor is:

- `fac = (|A| if tipOrb_abs else A) ^ tipOrb_power`

and the tunneling map is modified by:

- `T_eff = T_base * fac`

The coefficient vector is passed as a `Quat4d` container in the convention:

- `tipOrb = (px, py, pz, s)`

### Parameters

Add to `params.json`:

- `tipOrb`: list of 4 floats `[px, py, pz, s]`
- `tipOrb_power`: integer power (typically `1` or `2`)
- `tipOrb_abs`: bool (if `True`, uses `|A|`)

### Implementation path (zero-copy)

- **C++ implementation**: `cpp/TipField.h`
  - `evalSitesTipsAngularFactor( nTips, pTips, nSites, pSites, tipOrb, power, bAbs, outFac )`
  - Input arrays are passed as raw pointers and cast to `(Vec3d*)` / `(Quat4d*)` (no copies).

- **C ABI export**: `cpp/pauli_lib.cpp`
  - wrapper `evalSitesTipsAngularFactor(...)` exposing `double*` pointers for ctypes.

- **Python binding**: `pyProbeParticle/pauli.py`
  - `evalSitesTipsAngularFactor(pTips, pSites, tipOrb, power, bAbs)` returns an `(nTips,nSites)` factor array.

- **Scan integration**: `tests/ChargeRings/pauli_scan.py`
  - applied in:
    - `generate_hops_gauss(...)`
    - `scan_xy_orb(...)` (after gaussian/orbital mixing)
    - `calculate_xV_scan_orb(...)` (after gaussian/orbital mixing)

---

---

## The `bAux` path: how the plotted “transition currents” map to internal arrays

Inside **[PauliSolver::generate_current(lead_idx)](cci:1://file:///home/prokop/git/ppafm/cpp/pauli.hpp:1444:4-1505:5)** ([pauli.hpp](cci:7://file:///home/prokop/git/ppafm/cpp/pauli.hpp:0:0-0:0), lines ~1445+ in your snippet):

- `bAux = (bAuxOutput && current_matrix_ptr != nullptr)`
- For each allowed charge-sector pair transition `(b -> c)` (charge `Q` to `Q+1`):
  - It computes ordered indices:
    - `bb = state_order2[b]`
    - `cc = state_order2[c]`
  - It computes `cb = get_ind_dm1(c, b, charge)` (QmeQ-style “valid transition index”)
  - It looks up two Pauli factors (rates-ish):
    - `fct_enter = pauli_factors[idx]`  (entering, `b -> c`)
    - `fct_leave = pauli_factors[idx+1]` (leaving, `c -> b`)
  - It forms the **current contribution** for this ordered pair `(bb,cc)`:
    - `contrib = probabilities[bb]*fct_enter - probabilities[cc]*fct_leave`

If `bAux`:
- `current_matrix_ptr[ bb*nstates + cc ] = contrib`
- `out_prob_b_enter[...] = probabilities[bb]`
- `out_prob_c_leave[...] = probabilities[cc]`
- `out_fct1_b_enter[...] = fct_enter`
- `out_fct2_c_leave[...] = fct_leave`
- `out_inds[...]` stores metadata (charge, raw state indices `b,c`, ordered indices `bb,cc`, transition index `cb`, factor index `idx`)

### Key practical consequence for plotting
- The “channel current” that you stackplot as `110 -> 100` etc is **exactly** an element of `current_matrix_ptr` in **ordered index space** `(bb,cc)`.
- The label `110 -> 100` only makes sense if you map `bb` and `cc` back to the corresponding **bitstring state** in the same order you used to build `labels`.

So for robust labeling:
- Your plotting must use the *same* `state_order` / ordered-index convention as the solver used when it filled `current_matrix_ptr`.

---

## Important warning: `plot_state_scan_1d` is currently 3-site specific

In `plot_state_scan_1d(...)` you have:

```python
re_order = [0,1,2,4,3,5,6,7]
```

That implicitly assumes:
- `nsite=3`
- `nstates=8`

For **2 sites** you have 4 states, for **4 sites** you have 16 states, so this will either crash or silently mislabel/misorder the plot.

Also: the `state_styles` dict you pass is almost certainly tailored to 8 labels; for 16 labels you need either:
- autogenerated colors/styles, or
- a consistent rule-based styling (by charge sector, etc).

This hard-coded reorder is the #1 reason you previously had “serious problems with ordering/labeling”.

---

## Task (2): How to reuse `plot_state_scan_1d` for 2- and 4-site systems

### A) 1D scan along **VBias** at fixed `(x,y)`
You can reuse the function almost directly by treating:
- `distance` axis = `Vbiases`
- `stateEs` = energies vs V
- `probs` = probabilities vs V
- optionally `currents` = `STM` vs V
- optionally `current_components` = transition currents vs V

You already have the right raw arrays produced by `calculate_xV_scan_orb(...)` / `pauli.run_pauli_scan_xV(...)`, but you must:
- run with `nx=1` (single tip point) or extract one x-position index from a line scan
- reshape the current decomposition buffer correctly

Concrete reshaping logic (conceptually):
- Suppose you run `calculate_xV_scan_orb(..., bCurrentComponents=True, nx=1, nV=nV)`
- Returned:
  - `STM` shape `(nV, 1)`
  - `stateEs` shape `(nV, 1, nstates)`
  - `probs` shape `(nV, 1, nstates)`
  - `current_decomp[0]` (“current_matrix”) was allocated in `pauli_scan.py` as `(nxV, nstate2)` where `nxV = npts*nV`
- For `nx=1`, `npts=1`, `nxV=nV`
  - reshape `current_matrix.reshape(nV, nstates, nstates)`
  - then transpose to what `plot_state_scan_1d` expects:
    - `current_components = np.transpose(current_matrix_reshaped, (1,2,0))` giving `(nstates, nstates, nV)`

Then call:
- `plot_state_scan_1d(distance=Vbiases, stateEs=stateEs[:,0,:], probs=probs[:,0,:], currents=STM[:,0], current_components=current_components, V_slice=None or V_slice=...)`

**But** you must fix the hard-coded `re_order` and style assumptions first.

### B) 1D scan along a **line in XY** at fixed `VBias` (what you did before)
This is already what `plot_state_scan_1d` is designed for, provided you pass:
- `distance` from `make_pTips_line(...)`
- `stateEs` and `probs` sliced at a chosen `V_slice` index from the 2D x–V run

For 2/4 sites, the same corrections apply:
- robust reorder
- robust labeling

---

## What should change (minimal + safe) to generalize and fix ordering/labels

You likely need **small edits** in `plot_state_scan_1d` (and maybe a helper) rather than rewriting scan code.

### 1) Replace hard-coded `re_order` with computed ordering
Best option (stable and interpretable):
- Compute charge for each state bitmask in the *state order list used for labeling*
- Sort by `(charge, state_bitmask)` or by `(charge, energy_at_midpoint)` depending on what you want

Conceptually:
- `state_order = pauli.make_state_order(nsite)` returns an array of bitmask integers in the intended order
- define:
  - `charge[i] = popcount(state_order[i])`
- choose reorder indices:
  - `re_order = np.argsort( list(zip(charge, state_order)) )`

That will work for 2, 3, 4 sites consistently and makes legends sane (0e sector first, then 1e, …).

### 2) Make `state_styles` optional
For 4 sites (16 states) it’s painful to maintain a dict of 16 styles manually.
Minimal improvement:
- if `state_styles is None`:
  - choose colors from a colormap (e.g. `tab20`)
  - choose line styles by charge sector (e.g. solid for Q=0, dashed for Q=1, etc)

### 3) Ensure current-component labels are derived from the same ordering
When you label a component as `labels[i] -> labels[j]`, `i,j` must match the index convention of the exported `current_matrix_ptr`.

From [pauli.hpp](cci:7://file:///home/prokop/git/ppafm/cpp/pauli.hpp:0:0-0:0), the export uses:
- `ij = bb*nstates + cc` where `bb=state_order2[b]`

So: you must confirm that `bb` corresponds to the same ordering as `labels` in Python.
Right now, in [pauli.hpp](cci:7://file:///home/prokop/git/ppafm/cpp/pauli.hpp:0:0-0:0):
- `state_order2` is set to identity in [generate_kern()](cci:1://file:///home/prokop/git/ppafm/cpp/pauli.hpp:1183:4-1248:5) (see line ~1189-1193), i.e. `state_order2[i]=i`.
But other parts of kernel generation use `state_order_inv[state]` when iterating.
This suggests:
- the *actual ordering used internally* is controlled by `state_order` / `state_order_inv` (coming from Python), but `state_order2` currently being identity is a red flag if you expected a nontrivial mapping.

So the safest approach is:
- In Python, treat exported `current_matrix_ptr` indices as **the solver’s “ordered index space”**, i.e. the same indexing used by `probabilities[]` and `energies[]` arrays returned.
- Since `Probs` and `StateEnergies` are copied out as [solver->get_probabilities()](cci:1://file:///home/prokop/git/ppafm/cpp/pauli.hpp:1549:4-1549:69) and `solver->energies`, the `labels` you use should correspond to that same index order.
- Therefore, your `labels = pauli.make_state_labels(pauli.make_state_order(nsite))` must match the solver’s internal order.

If you are currently passing a custom `state_order` into `scan_current_tip(...)` (you do), then **labels must be built from that exact `state_order`**, not recomputed inside plotting with `make_state_order(nsite)` again (unless you guarantee it’s identical).

So a minimal but important plotting change is:
- change `plot_state_scan_1d(..., nsite, ...)` to accept `state_order` explicitly (optional)
  - if not provided, compute `make_state_order(nsite)`
  - if provided, use it for labels and charge sorting

This avoids the “serious problems” when someone used a different `state_order` upstream.

---

## NTCDA-specific: what’s missing to do your state-probability maps there
The NTCDA study scripts were extended to run scans via the reusable wrappers in `pauli_scan.py` and to save and export the additional diagnostics needed for state/transition analysis.

In particular, xV studies can now:

- save `probs` and `stateEs` (many-body probabilities and energies)
- optionally export current decomposition components when the scan is run with `bCurrentComponents=True`
- export 1D cut plots (probability+energies, and current decomposition)

This makes “deep-dive” analysis for selected NTCDA cases reproducible from the same `params.json` without rewriting the study scripts.

---

## Documentation: you already have a starting point
[docs/ChargeRings_PauliScan_Tutorial.md](cci:7://file:///home/prokop/git/ppafm/docs/ChargeRings_PauliScan_Tutorial.md:0:0-0:0) is already “nice” for the scan scripts, but it doesn’t cover:
- NTCDA study workflow and directory layout
- current decomposition export (`bAux`) and how to interpret `out_inds`, `current_matrix_ptr`
- state ordering caveats

The clean docs structure I’d propose (new markdown files, minimal disruption):
- `docs/ChargeRings_NTCDA_Studies.md`
  - what each `run_ntcda_*` script produces
  - where outputs are written
  - how to rerun a single case for deeper state analysis
- add a short section to [ChargeRings_PauliScan_Tutorial.md](cci:7://file:///home/prokop/git/ppafm/docs/ChargeRings_PauliScan_Tutorial.md:0:0-0:0) pointing to:
  - the `bAux` export mechanism
  - the “state ordering” rule (labels must use the same `state_order` as the solver)

I can draft that Markdown next, but you’ll need to switch to **Code mode** if you want me to actually add/edit files in-repo.

---

## What I need from you / next step
- If you want **only plotting fixes** (generalize `plot_state_scan_1d` to 2/4 sites + robust labeling), tell me which direction you prefer for `re_order`:
  - **(A)** by charge then by bitmask (most stable)
  - **(B)** by charge then by energy at each x (can reorder along scan, can confuse legends)

And please switch to **Code mode** if you want me to implement:
- `plot_state_scan_1d(..., state_order=None, state_styles=None, re_order=None)` generalization
- a helper like `extract_current_components(current_decomp, nV, nx, nstates, ...)`
- optional: a small “NTCDA state-cut” script that loads `params.json` and produces the probability/energy/channel plots.

## Status
- **Completed:** dependency map + `bAux` propagation mapping.
- **Completed:** xV cut export infrastructure (configurable cuts, 2D overview cut markers, 1D plot titles).
- **Completed:** 1D probability stackplots with overlaid many-body energies for exported xV cuts.
- **Completed:** tip-orbital angular prefactor for tunneling (`tipOrb`) implemented from C++ through Python and integrated into scan tunneling map generation.