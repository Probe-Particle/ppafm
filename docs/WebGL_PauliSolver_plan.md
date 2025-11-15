# WebGL Pauli Master Equation Viewer – Design & Plan

## Goal

Implement a simplified, self-contained WebGL visualization of Pauli Master Equation (PME) transport for few-site quantum dots (nsites ≤ 4, nstates ≤ 16). The whole app lives in a **single HTML file with embedded JavaScript and GLSL**, similar in spirit to `web/ppafm_web/index.html` + `PP_AFM_shader.glslf`.

Each **pixel** corresponds to a tip position `(x, y)` (and possibly a fixed or scanned bias). In the **fragment shader**, we compute:

- on-site energies including tip-induced gating (from a simplified `TipField` model),
- many-body state energies,
- (later) PME kernel and steady-state probabilities,
- current and dI/dV.

CPU-side JS will manage UI, parameter uniforms, and possibly some precomputation, but as much as feasible runs in GLSL.

---

## Reference Implementations & Files

- **Theory / documentation**
  - [x] `docs/ChargeRings_documentation_2.md`

- **Core PME implementation**
  - [x] `cpp/pauli.hpp` – `PauliSolver` class; state energies, lead couplings, Pauli factors, kernel, solver, current
  - [x] `cpp/pauli_lib.cpp` – C API, scan helpers (`scan_current_tip`, `solve_hsingle`, etc.)
  - [x] `pyProbeParticle/pauli.py` – Python wrapper, helpers (`make_state_order`, `run_pauli_scan_top`, `run_pauli_scan_xV`, etc.)
  - [x] `tests/ChargeRings/pauli_scan.py` – high-level scan workflows, GUI-style parameters, analysis utilities

- **Tip potential & tunneling**
  - [x] `cpp/TipField.h` – `evalMultipoleMirror`, `evalSitesTipsMultipoleMirror`, `evalSitesTipsTunneling`

- **Linear solver backend**
  - [x] `cpp/gauss_solver.hpp` – Gaussian / regularized least-squares solver used by PME kernel

- **WebGL / shader pattern**
  - [x] `web/ppafm_web/index.html` – single-page WebGL app, GUI controls, Three.js/matrix of uniforms
  - [x] `web/ppafm_web/PP_AFM_shader.glslf` – AFM fragment shader; example of per-pixel iterative relaxation & field evaluation

---

## Physics/Numerics We Need to Reproduce

### Notation and basic quantities

- **Sites / single-particle orbitals:** `i = 1..N` (we use `N = nSingle = nsites ≤ 4`).
- **Many-body states:** `s ∈ {0,1}^N` or integer bitmask `state ∈ [0,2^N)`; bit `i` indicates occupation of site `i`.
- **Charges:**
  - Site charge/occupancy in state `s`: `n_i(s) ∈ {0,1}`.
  - Total charge (electron number) in state `s`:

    $$
    Q_s = \sum_{i=1}^N n_i(s)
    $$

- **Single-particle Hamiltonian:** `H_single` with diagonal on-site energies `H_{ii}` and (optional) hopping `H_{ij}`.
- **Coulomb interaction:**
  - Simple uniform pair interaction `W` (energy cost per occupied pair), or
  - Full matrix `W_{ij}` (off-diagonal only, `W_{ii}=0`).
- **Leads:** `l ∈ {S,T}` (substrate, tip) with chemical potentials `μ_l` and temperature `T_l`.

### Many-body state energies (PME/QmeQ-style)

From `docs/ChargeRings_documentation_2.md` and `calculate_state_energy` in `cpp/pauli.hpp`.

For a given many-body state `s` (bitmask `state`):

1. **Single-particle contribution**

   $$
   E_s^{\text{sp}} = \sum_{i \in s} H_{ii}
   $$

2. **Hopping contribution (optional)**

   $$
   E_s^{\text{hop}} = \sum_{i<j,\; i,j \in s} \bigl( H_{ij} + H_{ji} \bigr)
   $$

3. **Coulomb interaction**

   - With **scalar** `W`:

    $$
     E_s^{\text{Coul}} = N_{\text{pairs}}(s)\, W\,, \qquad
     N_{\text{pairs}}(s) = \sum_{i<j} n_i(s) n_j(s)
    $$

   - With **matrix** `W_{ij}`:

    $$
     E_s^{\text{Coul}} = \sum_{i<j} n_i(s) n_j(s)\, W_{ij}
    $$

4. **Total many-body energy**

   $$
   E_s = E_s^{\text{sp}} + E_s^{\text{hop}} + E_s^{\text{Coul}}.
   $$

In the **simplified WebGL version** we can start with:

- `H_{ij}=0` for `i≠j` (no hopping),
- either `W=0` (non-interacting) or constant `W` or distance-based `W_{ij}` precomputed on CPU.

### Tip-induced on-site energies (multipole/ramp model)

From `docs/ChargeRings_documentation.md` and `cpp/TipField.h` (`Emultipole`, `evalMultipoleMirror`). For a site at position `\mathbf{r}_i` and tip at `\mathbf{r}_\text{tip}` with bias `V_\text{tip}` and tip radius `R_\text{tip}`:

1. **Displacements**

   - Direct tip displacement:

     $\mathbf{d} = \mathbf{r}_\text{tip} - \mathbf{r}_i.$

   - Mirror image tip position about plane `z = z_{V0}`:

     $z_\text{tip}^{\text{mir}} = 2 z_{V0} - z_\text{tip},\quad
        \mathbf{r}_\text{tip}^{\text{mir}} = (x_\text{tip}, y_\text{tip}, z_\text{tip}^{\text{mir}}),
     $

     with displacement

     $\mathbf{d}_\text{mir} = \mathbf{r}_\text{tip}^{\text{mir}} - \mathbf{r}_i.$

2. **Multipole energy kernel** (schematic, up to quadrupole; cf. eq. in `ChargeRings_documentation.md`):


   $$
     E_{\text{multipole}}(\mathbf{d}) = 
     \frac{Q_0}{r}
     + \frac{\mathbf{p}\ \cdot \mathbf{d}}{r^3}
     + \frac{1}{2}\sum_{ij} Q_{ij} \frac{3 d_i d_j - r^2\delta_{ij}}{r^5}
    $$

   In C++ this is implemented in a compact polynomial form with coefficients `cs[k]` and order flag.

3. **Mirror + ramp combination** (`evalMultipoleMirror`)

   - Define $\mathbf{z}_V = (z_{V0}, z_{Vd})$, and $z_{V1} = z_\text{tip} + z_{Vd}`.
   - Core multipole part (before bias scaling):

    $$
    E_0(\mathbf{r}_\text{tip}, \mathbf{r}_i)
       = E_{\text{multipole}}(\mathbf{d})
       - E_{\text{multipole}}(\mathbf{d}_\text{mir})\,.
    $$

   - Multiply by $V_\text{tip} R_\text{tip}$:

    $$
     E_{\text{multi}} = V_\text{tip} R_\text{tip} \; E_0(\mathbf{r}_\text{tip}, \mathbf{r}_i)
    $$

   - Add linear ramp between planes $z=z_{V0}$ and $z=z_{V1}$:

    $$
     \text{ramp}(z_i) = \frac{z_i - z_{V0}}{z_{V1} - z_{V0}}\,,\quad
     0 \le \text{ramp} \le 1\;\text{(clamped)},
    $$

    $$
     V_{\text{lin}} = V_\text{tip} \; \text{ramp}(z_i),\qquad
     E_{\text{lin}} = Q_0 \; V_{\text{lin}}.
    $$

4. **Total on-site energy shift**

   For site `i` with base energy `E_i^0`:

   \[
   E_i(\mathbf{r}_\text{tip}) = E_i^0 + E_{\text{multi}} + E_{\text{lin}}.
   \]

In the **simplified WebGL implementation** we can start with:

- Monopole-only (`Q_0` + ramp),
- No per-site rotations, i.e. use lab-frame `\mathbf{d}` directly.

### Total electrostatic energy and charging force (ChargeRings picture)

From `ChargeRings_documentation.md`:

1. **Total energy** for continuous site charges `Q_i` and tip charge `Q_\text{tip}`:

   $$
   U_{\text{total}} = \sum_{i=1}^N \left( \frac{E_i Q_i}{2} + \frac{Q_i Q_{\text{tip}}}{4 \pi \epsilon_0 r_{i\text{tip}}} - \mu Q_i \right)
   + \sum_{i=1}^N \sum_{j=i+1}^N \frac{Q_i Q_j}{4 \pi \epsilon_0 r_{ij}}.
   $$

2. **Charging force** (functional derivative w.r.t. `Q_i`):

   $$
   \frac{\delta U_{\text{total}}}{\delta Q_i}
     = \frac{E_i}{2} + \frac{Q_{\text{tip}}}{4 \pi \epsilon_0 r_{i\text{tip}}}
     + \sum_{j \ne i} \frac{Q_j}{4 \pi \epsilon_0 r_{ij}} - \mu.
   $$

In our PME implementation we work with **discrete** charges `Q_s` and many-body energies `E_s`, but the underlying electrostatics correspond to this functional form.

### Tunneling amplitudes & leads

We follow the PME documentation and `eval_lead_coupling_natural` logic in `cpp/pauli.hpp`.

1. **Local tunneling amplitudes** (`TLeads[l,i]`):

   - Substrate (lead `S`):

     \[T_{S,i} = V_S \quad (\text{constant across sites}).\]

   - Tip (lead `T`): exponentially decaying with tip–site distance:

     \[
     T_{T,i}(\mathbf{r}_\text{tip}) = V_T \; \exp(-\beta\, r_{i\text{tip}}),
     \]

     where `β` is a decay constant.

2. **Many-body tunneling matrix elements** `T_{cb}^{(l)}`

   For a base state `b` and target state `c` that differ by one electron at site `j`:

   - Let `site = j` (natural bit mapping).
   - Compute fermion sign from occupied sites with index `< site`:

     \[
     f_{\text{sign}}(b, j) = (-1)^{\sum_{k<j} n_k(b)}.
     \]

   - If `b` has `n_j(b)=0` and `c` has `n_j(c)=1` (electron enters the dot):

     \[
     T_{cb}^{(l)} \propto f_{\text{sign}}(b,j)\, T_{l,j}.
     \]

   - If `b` has `n_j(b)=1` and `c` has `n_j(c)=0` (electron leaves the dot):

     \[
     T_{cb}^{(l)} \propto f_{\text{sign}}(b,j)\, T_{l,j}^*.
     \]

In code we store a dense matrix `T^{(l)}[c,b]` for each lead `l`.

### Fermi-Dirac distribution (leads)

For a given lead `l` and energy argument `E`:

\[
f_l(E) = \frac{1}{1 + \exp\left(\dfrac{E - \mu_l}{k_B T_l}\right)}.
\]

In our C++ implementation we work in meV and use `KB` in matching units. For WebGL we can absorb `k_B` into the temperature scale and use the same functional form.

### Pauli transition rates (Pauli factors)

For a transition between many-body states `b` and `c` differing by one electron, via lead `l`, with energy difference

\[
\Delta E_{cb} = E_c - E_b,
\]

and tunneling amplitude `T_{cb}^{(l)}`, the **forward** (electron entering dot) and **backward** (electron leaving dot) rates are (from `ChargeRings_documentation_2.md`):

\[
\Gamma_{cb}^{(l,\text{forward})} = 2\pi \; \bigl|T_{cb}^{(l)}\bigr|^2\; f_l(\Delta E_{cb}),
\]
\[
\Gamma_{cb}^{(l,\text{backward})} = 2\pi \; \bigl|T_{cb}^{(l)}\bigr|^2\; \bigl[1 - f_l(\Delta E_{cb})\bigr].
\]

In WebGL we can work with rates **up to a global prefactor** (set `2π/ħ = 1` in units of our choice) since we mostly care about normalized probabilities and relative currents.

### Kernel matrix (Pauli master equation)

The PME for probabilities `\rho_s` is

\[
\frac{d\rho_s}{dt} = \sum_{a \ne s} \bigl(K_{sa} \rho_a - K_{as} \rho_s\bigr),
\]

which in steady state becomes

\[
\sum_a K_{sa} \rho_a = 0.
\]

We construct `K` from Pauli rates `\Gamma`:

- **Diagonal elements**

  For a given state `b` (with total charge `Q_b`):

  \[
  K_{bb} = - \sum_l \Biggl( \sum_{a, Q_a=Q_b-1} \Gamma_{ba}^{(l,\text{backward})}
                                + \sum_{c, Q_c=Q_b+1} \Gamma_{cb}^{(l,\text{forward})}\Biggr).
  \]

- **Off-diagonal elements**

  For `a ≠ b` with `Q_b = Q_a+1` (one more electron in `b`):

  \[
  K_{ba} = \sum_l \Gamma_{ba}^{(l,\text{forward})}.
  \]

  For `c ≠ b` with `Q_c = Q_b+1` (one fewer electron in `b`):

  \[
  K_{bc} = \sum_l \Gamma_{cb}^{(l,\text{backward})}.
  \]

To obtain a unique solution, we replace one row of `K` (e.g. the first) by a **normalization condition**:

\[
\sum_s \rho_s = 1.
\]

In practice:

- Replace row `s0` by ones: `K'_{s0,a} = 1` for all `a`,
- Set RHS vector `\text{rhs}'_{s0} = 1`, all others zero.

We then solve

\[
K' \rho = \text{rhs}'.
\]

### Current through a given lead

From `ChargeRings_documentation_2.md` and `generate_current` in `cpp/pauli.hpp`, for lead `l`:

\[
I_l = \sum_{Q} \sum_{b:Q_b=Q} \sum_{c:Q_c=Q+1}
  \Bigl( \rho_b\, \Gamma_{cb}^{(l,\text{forward})}
       - \rho_c\, \Gamma_{bc}^{(l,\text{backward})} \Bigr).
\]

In the WebGL app we will typically focus on the **tip current** `I_T` and visualize either:

- `I_T(x,y)` (STM-like current map), or
- its finite-difference derivative `dI_T/dV` w.r.t. bias.

### Differential conductance dI/dV

We approximate the differential conductance by a finite difference in bias `V`:

\[
\frac{dI}{dV}(V) \approx \frac{I(V+\delta V) - I(V)}{\delta V},
\]

using a small `\delta V` (e.g. a few percent of `V`) and recomputing PME/current for both biases.

In WebGL:

- Either perform **two passes** (two frames, different `uVBias`) and combine on CPU,
- Or run **two internal solves** in the fragment shader with `uVBias` and `uVBias+δV` and take the difference.

---

## Overall Architecture (WebGL App)

### CPU side (JavaScript)

- Single HTML file with:
  - `<canvas>` or Three.js plane (similar to `web/ppafm_web/index.html`).
  - GUI controls (inputs/sliders) for parameters (adapted from `params` in `tests/ChargeRings/pauli_scan.py`).

- Responsibilities:
  - Initialize WebGL/Three.js scene and full-screen quad.
  - Define GLSL fragment shader source (inline `<script type="x-shader/x-fragment">` or JS string).
  - Define uniforms:
    - **Geometry of sites**: positions, onsite base energies.
    - **Lead parameters**: μ substrate, μ tip (VBias), T.
    - **Coulomb parameters**: W or Wij.
    - **Tip-field parameters**: Rtip, zV0, zVd, multipole coefficients cs.
    - **Tunneling parameters**: VS, VT, β.
    - **Control flags**: stage selection (ground-state, PME, show I, show dIdV, etc.).
  - Map GUI controls → uniforms.

### GPU side (GLSL fragment shader)

Per pixel (x,y) we:

1. Map `gl_FragCoord` to **tip position** `(x, y, z_tip)`.
2. Compute **onsite energies** `E_i` using simplified `TipField` formula.
3. Build **Hsingle** (diagonal from E_i, optional off-diagonal hopping terms from JS uniforms).
4. Enumerate many-body states (bitmasks) up to `2^nsites`.
5. Compute **state energies** `E_s`.
6. Stage-dependent:
   - Stage 1: choose ground state s₀ (min E_s), define occupancy per site (bit pattern), visualize charge density.
   - Stage 2: compute PME steady-state ρ_s:
     - Build tunneling amplitudes `T_{cb}^{(l)}`.
     - Compute Pauli rates Γ.
     - Build kernel K.
     - Solve Kρ = rhs with Gaussian elimination.
     - Optionally add simple checks (sum ρ≈1, ρ≥0 within tolerance).
   - Stage 3: compute current I_tip.
   - Stage 4: compute dI/dV by re-solving with slightly different μ_tip or VBias.

7. Map final scalar quantity (charge, current, dI/dV, etc.) to color.

---

## Phased Implementation Plan (with Checkboxes)

### Phase 0 – Planning & stubs

- [ ] Create single-file HTML skeleton `web/pauli_web/pauli_pme_viewer.html` (or similar) with:
  - [ ] Canvas + minimal Three.js setup (copy pattern from `web/ppafm_web/index.html`).
  - [ ] Minimal fragment shader that just draws a test gradient.
  - [ ] Simple GUI inputs for `nsites`, `L`, `z_tip`, `VBias`, `Rtip`, etc., but not yet wired.

- [ ] Decide **hard limits** and encode in shader:
  - [ ] `MAX_SITES` (likely 4).
  - [ ] `MAX_STATES` = `2^MAX_SITES` (16).


### Phase 1 – Only onsite energies & ground-state occupancy (no PME)

**Goal:** reproduce steps (1)–(2) of your phased list.

1. **Data representation & uniforms**
   - [ ] In JS, define `nsites` and an array of site positions `spos[i].xyz` and base energies `Esite[i]` (by default from a simple ring as in `make_site_geom`).
   - [ ] Pass those into shader as `uniform int uNSites`, `uniform vec4 uSites[MAX_SITES]` (xyz, Esite).
   - [ ] Pass tip parameters: `uL`, `uZTip`, `uRtip`, `uZV0`, `uZVd`, `uCs[10]`, `uOrder`, `uVBias`.

2. **Simplified tip-field in GLSL**
   - [ ] Port a **minimal** version of `evalMultipoleMirror` logic per site:
     - [ ] Compute tip and mirror displacements.
     - [ ] Use monopole or quadrupole-only cs.
     - [ ] Implement linear ramp in z between `zV0` and `zV1=z_tip+zVd`.
   - [ ] For each site i, compute `E_i(tip)` and set `Hsingle[i,i] = E_i`.

3. **Many-body energies**
   - [ ] Encode `nsites`-bit states 0..(2^nsites-1) in shader.
   - [ ] Implement `count_electrons(state)` and state-energy formula matching `calculate_state_energy_simple` (initially without hopping, Wij=W const or 0).
   - [ ] Compute state energies `E_s` for all states.

4. **Ground state & occupancy visualization**
   - [ ] Find index `s0` of minimal `E_s`.
   - [ ] Convert `s0` bits to per-site occupancy `occ_i`.
   - [ ] Aggregate occupancy or charge map `Q = Σ_i occ_i` or show e.g. occupancy of a chosen site via a GUI toggle.
   - [ ] Map Q or occ_i to grayscale/colormap and render.

5. **Validation / debugging hooks**
   - [ ] Add debug mode that encodes `E_s` of a selected state as intensity to sanity-check geometry.
   - [ ] Compare one tip position against C++/Python results for same parameters (offline, plotted in Python) to validate state energies.


### Phase 2 – PME kernel & stationary non-equilibrium occupancy

**Goal:** reproduce step (3) of your list: stationary PME solution per pixel.

1. **Leads & tunneling in GLSL**
   - [ ] Add uniforms for lead parameters: `uMuSub`, `uMuTip`, `uTemp` (energies in meV or eV consistent with site energies).
   - [ ] Add uniforms for tunneling constants `uVS`, `uVT`, `uBeta`.
   - [ ] Implement `evalSitesTipsTunneling` in GLSL: per site i, compute `T_tip[i] = VS` (substrate) and `T_tip[i] = VT*exp(-β*|tip-spos[i]|)` for lead 1.

2. **State-to-state tunneling amplitudes**
   - [ ] Implement QmeQ-style `eval_lead_coupling_natural` in GLSL:
     - [ ] For each lead l, and base state b, loop sites j.
     - [ ] Determine whether adding or removing electron at site j leads to a valid state c.
     - [ ] Compute fermion sign from occupations of bits < j.
     - [ ] Accumulate `T_{cb}^{(l)}` into a dense `T_lead[l][c][b]` matrix.

3. **Pauli factors (transition rates)**
   - [ ] Implement Fermi-Dirac function `f_l(E)` with given μ_l, T.
   - [ ] For each pair (b,c) differing by 1 electron and each lead l, compute:
     - [ ] ΔE = E_c - E_b.
     - [ ] Γ_forward(cb,l) ∝ |T_{cb}^{(l)}|^2 f_l(ΔE).
     - [ ] Γ_backward(bc,l) ∝ |T_{cb}^{(l)}|^2 (1 - f_l(ΔE)).
   - [ ] Use a simple prefactor (e.g. absorb 2π/ħ into units) – match C++ up to a global scaling.

4. **Kernel construction**
   - [ ] Build full K[nstates][nstates] using simplified rules from documentation:
     - [ ] Off-diagonals: sum appropriate Γ’s for transitions into state s.
     - [ ] Diagonals: minus sum of all rates leaving s.
   - [ ] Enforce normalization by replacing one row with ones and RHS=1; other RHS entries zero.

5. **Linear solver in GLSL**
   - [ ] Implement basic Gaussian elimination with partial pivoting for dense real matrix up to 16×16.
   - [ ] Solve `K ρ = rhs` per pixel.
   - [ ] Normalize ρ again in shader for safety.

6. **Visualization**
   - [ ] Visualize total charge `Q = Σ_s ρ_s * N_electrons(s)` per pixel.
   - [ ] Add GUI toggle to show specific state probabilities ρ_s for selected s (e.g. by mapping to hue or separate mode).


### Phase 3 – Current map

**Goal:** reproduce step (4): compute current I_tip per pixel.

1. **Current expression**
   - [ ] Using ρ_s and Γ from Phase 2, compute tip current:
     - [ ] For each charge sector Q states b, and Q+1 states c:
       - [ ] Accumulate `I_tip += ρ_b * Γ_forward(cb,tip) - ρ_c * Γ_backward(bc,tip)`.
   - [ ] Use units consistent with C++ up to factor; we mainly care about relative spatial/dV patterns.

2. **Visualization**
   - [ ] Add mode to color by I_tip (e.g. colormap similar to `cmap_STM` in Python).
   - [ ] Optionally compute log-scaled or normalized current for better contrast.


### Phase 4 – dI/dV

**Goal:** reproduce step (5): compute dI/dV by finite difference over VBias.

1. **Dual-bias computation strategy**
   - Option A (two passes):
     - [ ] Render I_tip at VBias and VBias+δV by re-drawing scene twice and combining on CPU or in a second pass.
   - Option B (single-pass, duplicated kernels):
     - [ ] In shader, run PME + current twice with `uVBias` and `uVBias+δV`, storing both I(V) and I(V+δV).

2. **Finite difference**
   - [ ] Compute `dIdV ≈ (I(V+δV) - I(V)) / δV`.
   - [ ] Visualize with diverging colormap (like `PiYG_r` used for dIdV in Python scripts).


---

## GUI Controls & Parameters

We should mimic / subset `params` from `tests/ChargeRings/pauli_scan.py` and GUI elements from `CombinedChargeRingsGUI_v5.py` for familiarity.

**Core controls (first version):**

- **Geometry / scan**
  - [ ] `nsites` (1–4).
  - [ ] `npix` or canvas size (fixed 512×512 like AFM viewer).
  - [ ] `L` – half-size of scan area in Å.
  - [ ] `z_tip` – tip height above QD plane.
  - [ ] A few predefined geometries (ring of nsites, line, triangle) with dropdown.

- **Tip-field** (subset of `params` used in `scan_tipField_xV` and `run_pauli_scan_top`):
  - [ ] `VBias` – tip bias.
  - [ ] `Rtip` – tip radius.
  - [ ] `zV0`, `zVd` – mirror and ramp parameters.
  - [ ] `Q0`, `Qzz` – monopole/quadrupole moments; we can pre-bake cs from these.
  - [ ] `Esite` base onsite energy.

- **Tunneling & PME**
  - [ ] `GammaS`, `GammaT` → VS, VT via `sqrt(Gamma/pi)` (like in `run_cpp_scan`).
  - [ ] Decay `beta` for tunneling `exp(-beta*r)`.
  - [ ] `W` Coulomb interaction strength (simple uniform W at first).
  - [ ] `Temp` (K or meV – we can match Python meaning).

- **Display modes**
  - [ ] Dropdown or radio buttons:
    - [ ] `Mode = ground_state_charge`.
    - [ ] `Mode = PME_charge`.
    - [ ] `Mode = current`.
    - [ ] `Mode = dIdV`.
    - [ ] (later) `Mode = state_prob[s]`.

- **Debug**
  - [ ] Toggle `Show energies` (e.g. encode min E_s or selected E_s).
  - [ ] Toggle `Show occupied site index`.

---

## Numerical & Implementation Considerations

- **Performance:**
  - nstates ≤ 16 → K is at most 16×16; Gaussian elimination per pixel is borderline but still plausible for modern GPUs at 512×512, especially if we optimize loops and use low precision where safe.
  - We can provide a **"low-res preview"** option (reduce resolution) if performance is an issue.

- **Stability:**
  - In C++/`gauss_solver.hpp` there is significant work to regularize near-singular matrices and renormalize probabilities. In GLSL we start with a simpler version but should:
    - Clamp small pivots.
    - Renormalize ρ.
    - Optionally skip pixels where condition numbers blow up (e.g. if W→0 and couplings vanish).

- **State ordering:**
  - Use **natural order** 0..(2^nsites-1) throughout (matching `eval_lead_coupling_natural` and `make_state_order` for `nsite!=3`).
  - Avoid QmeQ’s historical ordering issues.

- **Units:**
  - Match existing convention where all energies are in meV or eV consistently; internal scaling can be arbitrary for visualization but we should note what we choose in code comments.

---

## Immediate Next Steps

- [ ] Decide final filename and path for the WebGL HTML prototype (e.g. `web/pauli_web/Pauli_STM_viewer.html`).
- [ ] Implement Phase 0 skeleton: basic Three.js quad + dummy shader + a few numeric inputs.
- [ ] Implement Phase 1 ground-state-only logic in GLSL and validate against a single-point Python/C++ computation.

Once Phase 1 is working and validated, we can iterate with you on how much of the full PME & current pipeline we want to push into GLSL vs approximate/CPU-side.
