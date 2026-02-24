# USER

I have solver of many body charge stats on mutiple quantum dots (QD) interaction with STM tip (lead1) and surface (lead2), the tip both gat the voltage and hop in and out the QD, there is no direct hopping between the QDs. I have two implementations - CPU/C++ reference, and newer pyOpenCL reimplementation 

### C++ implementation

@pauli.hpp @TipField.h @gauss_solver.hpp 
library interface
@pauli_lib.cpp 
python wrapper
@pauli.py 
high levels utilities and test runs
@pauli_scan.py 
GUI

### pyOpenCL implementation

@PME.cl @pauli_ocl.py 
documentation of reimplementation
@PME_OpenCL.md 


### Consolidated teds and GUI of both

@PauliFastGUI.py @debug_W_Esite_propagation.py @compare_pme_solvers.py @compare_cpu_gpu_line.py @compare_cpu_gpu_bias.py 


Read all relevant files, make deteild notes what is in which file what are the importint functions and what are the equavalent (simular responsibility) functions between GPU and CPU. the goal is to make comprenesive deteild documentation

---

# GPT-5.1-Codex

## Gemeral Physics

### Many-body states and energies
- Single-particle basis of size `nSingle`; many-body states labeled by bitmask `state` (`site i` occupied if `(state>>i)&1`). Helpers in CPU @cpp/pauli.hpp#81-145.
- Many-body energy \(E_b\) for state \(b\):
  \[
  E_b = \sum_{i\in b} (\varepsilon_i) \;+\; \sum_{i<j\in b} (W_{ij}\ \text{or}\ W)
        \;+\; \sum_{i<j\in b} (t_{ij}+t_{ji})\ (\text{if hopping present})
  \]
  implemented by [calculate_state_energy](cci:1://file:///home/prokop/git/ppafm/cpp/pauli.hpp:108:0-144:1) (full) and [calculate_state_energy_simple](cci:1://file:///home/prokop/git/ppafm/cpp/pauli.hpp:82:0-106:1) (diag-only) @cpp/pauli.hpp#81-145. In OpenCL, phase 1 of `solve_pme` uses `H_single_base + H_shifts` and optional `Wij` to produce `Energies[tid]` @cl/PME.cl#185-209.
- Onsite shifts from tip electrostatics: `evalMultipoleMirror` (mirror plane + linear ramp) gives per-site \(\Delta \varepsilon_i\) @cpp/TipField.h#36-85; OpenCL kernel `compute_tip_interaction` fills `H_shifts` similarly @pyProbeParticle/pauli_ocl.py#140-157.

### Lead coupling and tunneling
- Two leads (substrate l=0, tip l=1) with chemical potentials \(\mu_l\) and temperatures \(T_l\); stored in `leads` (CPU) and `lead_params` (GPU) @cpp/pauli.hpp#147-153, @pyProbeParticle/pauli_ocl.py#83-91.
- Tunneling amplitudes \(t_{l,i}\) (lead l ↔ site i). CPU builds `TLeads` via [evalSitesTipsTunneling](cci:1://file:///home/prokop/git/ppafm/pyProbeParticle/pauli.py:169:0-178:16) (exp decay) or passed in; OpenCL gets per-pixel factors `T_factors` from kernel 1. In CPU, coupling matrices are built by [calculate_tunneling_amplitudes](cci:1://file:///home/prokop/git/ppafm/cpp/pauli.hpp:656:4-665:5) with fermionic sign @cpp/pauli.hpp#645-667; in GPU, `solve_pme` uses \(t_{l,i}\) directly @cl/PME.cl#246-269.
- Gamma convention: code uses \(T_{l,i} = \Gamma/\pi \times \text{exp}(\cdot)\); rates multiply \(T^2\cdot 2\pi\) to form \(\Gamma\)-scaled rates (see comments @pyProbeParticle/pauli_ocl.py#168-195).

### Transition rates (Pauli master equation)
- Allowed transitions flip exactly one site between many-body states \(b\) and \(c\); diff bit found by XOR/popcount in CPU [generate_coupling_terms](cci:1://file:///home/prokop/git/ppafm/cpp/pauli.hpp:1109:4-1185:5) @cpp/pauli.hpp#1110-1185 and GPU `solve_pme` @cl/PME.cl#215-273.
- Energy difference \(\Delta E = E_c - E_b\) (for adding electron) or \(E_b - E_c\) (for removing).
- Fermi factor \(f_l(\Delta E) = [1 + \exp((\Delta E - \mu_l)/T_l)]^{-1}\) @cpp/pauli.hpp#707-711, @cl/PME.cl#254-267.
- Rate contributions per lead l:
  - Entering (lead → dot): \( \Gamma_{l,enter} \propto t_{l,i}^2 \, f_l(\Delta E) \, 2\pi \)
  - Leaving (dot → lead): \( \Gamma_{l,leave} \propto t_{l,i}^2 \, [1-f_l(\Delta E)] \, 2\pi \)
- Total inflow/outflow accumulated into kernel matrix rows:
  - Off-diagonal \(K_{b,c}\) adds inflow to \(b\) from \(c\).
  - Diagonal \(K_{b,b}\) subtracts total outflow from \(b\).
  CPU: [generate_coupling_terms](cci:1://file:///home/prokop/git/ppafm/cpp/pauli.hpp:1109:4-1185:5) fills kernel using precomputed `pauli_factors`; GPU: computes directly in kernel @cl/PME.cl#246-273.

### Master equation and normalization
- Steady-state Pauli ME: \( \sum_c K_{b,c} P_c = 0\) for all \(b\), with normalization \(\sum_b P_b = 1\).
- Implementation replaces one row with normalization (row 0 set to ones) and RHS = [1,0,0,...]^T.
- Solve linear system \(K P = RHS\). CPU uses Gauss/LS ([gauss_solver.hpp](cci:7://file:///home/prokop/git/ppafm/cpp/gauss_solver.hpp:0:0-0:0)) @cpp/gauss_solver.hpp#6-239; GPU uses parallel Gauss-Jordan in `solve_pme` (local memory) @cl/PME.cl#276-288 and subsequent elimination (continues below shown snippet).

### Current calculation
- Current from lead l is the net rate-weighted probability flow along transitions involving that lead; in C++ computed after probabilities ([calculate_current](cci:1://file:///home/prokop/git/ppafm/pyProbeParticle/pauli.py:371:4-372:55), not shown) using stored coupling/rates; in GPU, the kernel accumulates per-pixel current (writes to `out_current`) after solving probabilities @cl/PME.cl#151-155 (write), later in kernel body.
- Optional exports: current matrix per transition, kernel matrix, probabilities, state energies (CPU via external buffers; GPU when requested) @cpp/pauli.hpp#223-229, @pyProbeParticle/pauli_ocl.py#207-285.

### Valid-point cut (zeroing unphysical points)
- Condition: zero current if \( \max_i E_i + W \cdot \text{EW\_cut} < 0\) or \( \max_i | \sqrt{\gamma}\, T_i | < \text{Tmin\_cut}\).
- Applied in CPU scan loop `scan_current_tip_` before solve @cpp/pauli_lib.cpp#445-484; mirrored post-kernel in GPU wrapper @pyProbeParticle/pauli_ocl.py#261-276.

### Key stored quantities (where)
- Single-particle Hamiltonian `Hsingle` / `H_single_base`: diagonal onsite energies (plus off-diag hopping if used); tip shifts added per-pixel (`H_shifts`) in GPU.
- Coulomb `Wij` or scalar `W`: pair interactions.
- Tunneling amplitudes `TLeads` (CPU) / `T_factors` (GPU) per pixel/site.
- Lead params `mu`, `temperature` per lead.
- Kernel matrix `kernel` (CPU) / local `Mat` (GPU) representing PME coefficients with normalization row.
- Probabilities `probabilities` (CPU) / local `Probs`/output `out_probs` (GPU).
- Energies per state `energies` (CPU) / `Energies`/`out_stateEs` (GPU).
- Current outputs: scalar per pixel (`out_current`), optional per-transition current matrix (`current_matrix_ptr` CPU, `out_curmat` GPU).

This captures the governing equations and how each physical quantity is represented and solved in the CPU and OpenCL paths.

## Importaint Files

Here’s a one-liner role/relationship summary for each mentioned (and closely related) file:

- [cpp/pauli.hpp](cci:7://file:///home/prokop/git/ppafm/cpp/pauli.hpp:0:0-0:0): Core CPU PauliSolver (state energies, tunneling, kernel assembly, linsolve) that the Python wrapper and tests drive.  
- `cpp/TipField.h`: Tip–site electrostatics and tunneling helpers used by CPU scan routines to build Hsingle/TLeads.  
- [cpp/gauss_solver.hpp](cci:7://file:///home/prokop/git/ppafm/cpp/gauss_solver.hpp:0:0-0:0): Gaussian/least-squares solvers used by PauliSolver to solve the PME kernel (probabilities).  
- `cpp/pauli_lib.cpp`: C++/ctypes bridge exposing scan/solve functions (including threaded tip scans) to Python.  
- [pyProbeParticle/pauli.py](cci:7://file:///home/prokop/git/ppafm/pyProbeParticle/pauli.py:0:0-0:0): Python ctypes wrapper and convenience class around the C++ solver; sets leads, Hsingle, Wij, TLeads, runs scans.  
- [tests/ChargeRings/pauli_scan.py](cci:7://file:///home/prokop/git/ppafm/tests/ChargeRings/pauli_scan.py:0:0-0:0): High-level CPU scan utilities/GUI backing; builds geometry, configures PauliSolver, runs xy/xV scans.  
- `cl/PME.cl`: OpenCL kernels (`compute_tip_interaction`, `solve_pme`) implementing tip-induced shifts/tunneling and PME solve on GPU.  
- `pyProbeParticle/pauli_ocl.py`: Python OpenCL wrapper `PauliSolverCL`; uploads data, launches PME kernels, mirrors CPU outputs for parity.  
- [docs/PME_OpenCL.md](cci:7://file:///home/prokop/git/ppafm/docs/PME_OpenCL.md:0:0-0:0): Design/notes for the OpenCL reimplementation, conventions, and parity requirements versus C++.  
- `tests/ChargeRings/PauliFastGUI.py`: GUI frontend that can call CPU or OpenCL scan paths for interactive visualization.  
- `tests/ChargeRings/debug_W_Esite_propagation.py`: Debug helper tracing Wij/Esite propagation into CPU/GPU solvers.  
- `tests/ChargeRings/compare_pme_solvers.py`: Head-to-head CPU vs OpenCL comparison for a pixel; can dump kernel/current matrices.  
- `tests/ChargeRings/compare_cpu_gpu_line.py`: Compares CPU vs GPU currents along a spatial line at fixed bias.  
- `tests/ChargeRings/compare_cpu_gpu_bias.py`: Compares CPU vs GPU currents across bias sweep at a fixed point.

## CPU / C++ reference

- **[cpp/pauli.hpp](cci:7://file:///home/prokop/git/ppafm/cpp/pauli.hpp:0:0-0:0)**
  - Core solver class [PauliSolver](cci:2://file:///home/prokop/git/ppafm/cpp/pauli.hpp:186:0-1604:1): allocation, state ordering, energies, coupling, kernel, probabilities.
  - Energy calculation: [calculate_state_energy](cci:1://file:///home/prokop/git/ppafm/cpp/pauli.hpp:108:0-144:1) (full, with off-diagonal hopping and Coulomb) and simplified [calculate_state_energy_simple](cci:1://file:///home/prokop/git/ppafm/cpp/pauli.hpp:82:0-106:1) for diagonal-only cases @cpp/pauli.hpp#81-145.
  - State/lead setup: setters [setHsingle](cci:1://file:///home/prokop/git/ppafm/cpp/pauli.hpp:429:4-437:5), [setW](cci:1://file:///home/prokop/git/ppafm/cpp/pauli.hpp:439:4-447:5), [setWij](cci:1://file:///home/prokop/git/ppafm/cpp/pauli.hpp:449:4-467:5), [setTLeads](cci:1://file:///home/prokop/git/ppafm/cpp/pauli.hpp:469:4-478:5), [setStateOrder](cci:1://file:///home/prokop/git/ppafm/cpp/pauli.hpp:480:4-489:5), [setLeadParams](cci:1://file:///home/prokop/git/ppafm/cpp/pauli.hpp:491:4-504:5); flags to invalidate cached energies/kernel @cpp/pauli.hpp#428-505.
  - Tunneling amplitudes: [calculate_tunneling_amplitudes](cci:1://file:///home/prokop/git/ppafm/cpp/pauli.hpp:656:4-665:5) / [updateTunnelingAmplitudes](cci:1://file:///home/prokop/git/ppafm/cpp/pauli.hpp:644:4-654:5) choose QmeQ vs natural ordering @cpp/pauli.hpp#645-667.
  - Energies update: [calculate_state_energies](cci:1://file:///home/prokop/git/ppafm/cpp/pauli.hpp:526:4-539:5) and [updateStateEnergies](cci:1://file:///home/prokop/git/ppafm/cpp/pauli.hpp:669:4-681:5) (optionally simple) @cpp/pauli.hpp#527-540, #668-681.
  - Indexing helpers: [init_states_by_charge](cci:1://file:///home/prokop/git/ppafm/cpp/pauli.hpp:872:4-904:5), [init_indexing_maps](cci:1://file:///home/prokop/git/ppafm/cpp/pauli.hpp:769:4-816:5), `get_ind_dm0/dm1`, [count_valid_transitions](cci:1://file:///home/prokop/git/ppafm/cpp/pauli.hpp:683:4-693:5) mirror QmeQ conventions @cpp/pauli.hpp#684-799.
  - Kernel assembly: [generate_fct](cci:1://file:///home/prokop/git/ppafm/cpp/pauli.hpp:943:4-1038:5) (not shown in excerpts), [generate_coupling_terms](cci:1://file:///home/prokop/git/ppafm/cpp/pauli.hpp:1109:4-1185:5) builds row b from ±1-charge neighbors using precomputed `pauli_factors`; [generate_kern](cci:1://file:///home/prokop/git/ppafm/cpp/pauli.hpp:1198:4-1263:5) orchestrates pauli_factors, state ordering, filling kernel @cpp/pauli.hpp#1110-1264.
  - Linear solve: mode set via [setLinSolver](cci:1://file:///home/prokop/git/ppafm/cpp/pauli.hpp:330:4-334:5); Gauss/SVD implementations live in [gauss_solver.hpp](cci:7://file:///home/prokop/git/ppafm/cpp/gauss_solver.hpp:0:0-0:0). Prob checks & optional exports (current matrix, factor/prob buffers) are wired via external pointers @cpp/pauli.hpp#206-229, #645-667, #1200-1264.
- **[cpp/gauss_solver.hpp](cci:7://file:///home/prokop/git/ppafm/cpp/gauss_solver.hpp:0:0-0:0)**
  - Gaussian elimination with scaling/pivoting ([GaussElimination](cci:1://file:///home/prokop/git/ppafm/cpp/gauss_solver.hpp:6:0-70:1), [linSolve_gauss](cci:1://file:///home/prokop/git/ppafm/cpp/gauss_solver.hpp:72:0-128:1)) plus least-squares fallback [linSolve_lstsq](cci:1://file:///home/prokop/git/ppafm/cpp/gauss_solver.hpp:130:0-131:60); normalizes probabilities to sum 1 @cpp/gauss_solver.hpp#1-239.
- **`cpp/TipField.h`**
  - Tip–site electrostatics and tunneling helpers:
    - `evalMultipoleMirror` computes site energy shift with mirror plane + linear ramp @cpp/TipField.h#36-85.
    - [evalSitesTipsMultipoleMirror](cci:1://file:///home/prokop/git/ppafm/pyProbeParticle/pauli.py:197:0-214:15) vectorized over tips/sites @cpp/TipField.h#87-113.
    - [evalSitesTipsTunneling](cci:1://file:///home/prokop/git/ppafm/pyProbeParticle/pauli.py:169:0-178:16) exponential decay tunneling map @cpp/TipField.h#115-133.
- **`cpp/pauli_lib.cpp` (Python wrapper functions)**
  - `scan_current_tip_` (single-thread) and `scan_current_tip_threaded`: loop over tip positions, build per-point `Hsingle` and `TLeads`, apply valid-point cut, call [solve_hsingle](cci:1://file:///home/prokop/git/ppafm/pyProbeParticle/pauli.py:281:4-282:114), optionally export Es/Ts/probabilities/state energies and current matrix @cpp/pauli_lib.cpp#400-530.
- **Python wrapper [pyProbeParticle/pauli.py](cci:7://file:///home/prokop/git/ppafm/pyProbeParticle/pauli.py:0:0-0:0)**
  - ctypes bindings for C++ solver lifecycle and getters/setters: [create_solver](cci:1://file:///home/prokop/git/ppafm/pyProbeParticle/pauli.py:249:4-250:72), [set_lead](cci:1://file:///home/prokop/git/ppafm/pyProbeParticle/pauli.py:252:4-253:54), [set_tunneling](cci:1://file:///home/prokop/git/ppafm/pyProbeParticle/pauli.py:255:4-257:80), [set_hsingle](cci:1://file:///home/prokop/git/ppafm/pyProbeParticle/pauli.py:259:4-261:65), `set_wij`, [generate_pauli_factors](cci:1://file:///home/prokop/git/ppafm/pyProbeParticle/pauli.py:272:4-273:80), [generate_kernel](cci:1://file:///home/prokop/git/ppafm/pyProbeParticle/pauli.py:275:4-276:40), `solve_pauli`, [solve_hsingle](cci:1://file:///home/prokop/git/ppafm/pyProbeParticle/pauli.py:281:4-282:114), [scan_current](cci:1://file:///home/prokop/git/ppafm/pyProbeParticle/pauli.py:299:4-322:26), [scan_current_tip](cci:1://file:///home/prokop/git/ppafm/pyProbeParticle/pauli.py:324:4-354:45), probability/current exports, and [set_valid_point_cuts](cci:1://file:///home/prokop/git/ppafm/pyProbeParticle/pauli.py:74:0-75:38) @pyProbeParticle/pauli.py#38-355.
  - Convenience [PauliSolver](cci:2://file:///home/prokop/git/ppafm/cpp/pauli.hpp:186:0-1604:1) class wraps above; utilities to compute tip-site tunneling and multipole energies via C++ helpers @pyProbeParticle/pauli.py#236-390.
- **GUI/test harness [tests/ChargeRings/pauli_scan.py](cci:7://file:///home/prokop/git/ppafm/tests/ChargeRings/pauli_scan.py:0:0-0:0)**
  - High-level scan runners (x–V scans, xy scans) building geometry, configuring solver via [make_configured_solver](cci:1://file:///home/prokop/git/ppafm/tests/ChargeRings/pauli_scan.py:126:0-142:17), applying Wij config, assembling hopping maps, and invoking [pauli.run_pauli_scan_top](cci:1://file:///home/prokop/git/ppafm/pyProbeParticle/pauli.py:606:0-655:38) / [scan_current_tip](cci:1://file:///home/prokop/git/ppafm/pyProbeParticle/pauli.py:324:4-354:45). Packs outputs (STM, dIdV, Es, Ts, probs, state energies, current decomposition) @tests/ChargeRings/pauli_scan.py#127-248.

## pyOpenCL implementation

- **`pyProbeParticle/pauli_ocl.py`**
  - Class `PauliSolverCL` mirrors C++ solver at nSingle=4 (nStates=16).
  - OpenCL setup: builds `cl/PME.cl`, caches kernels `compute_tip_interaction` and `solve_pme` @pyProbeParticle/pauli_ocl.py#29-74.
  - State order default = identity, stored on device @pyProbeParticle/pauli_ocl.py#75-81.
  - [scan_current_tip](cci:1://file:///home/prokop/git/ppafm/pyProbeParticle/pauli.py:324:4-354:45):
    - Packs tips/sites to float4, uploads rotations, Vtips, multipole coefficients, params; allocates device buffers for H_shifts (Es) and T_factors (Ts) @pyProbeParticle/pauli_ocl.py#92-138.
    - Kernel 1 `compute_tip_interaction` fills per-pixel site energy shifts and tunneling factors @pyProbeParticle/pauli_ocl.py#140-157.
    - Kernel 2 `solve_pme` builds PME matrix, solves collaboratively (1 workgroup/pixel, 16 threads), outputs current, optional probs/state energies/kernel/current matrix; supports Wij buffer and scalar W @pyProbeParticle/pauli_ocl.py#159-286.
    - Post-processing applies CPU-style valid-point cut (Emax + W*EW_cut < 0 or Tmax < Tmin_cut) before returning currents; always returns Ts, optionally Es/Probs/StateEs/K/CurMat @pyProbeParticle/pauli_ocl.py#261-285.
- **`cl/PME.cl`**
  - Kernel `solve_pme`: in local memory builds many-body energies from `H_single_base + H_shifts`, Coulomb via Wij or scalar W; constructs rate matrix using Fermi functions with lead params and per-site tunneling; enforces normalization row; parallel Gauss-Jordan elimination; optional exports of kernel matrix/current matrix/probs/state energies; writes per-pixel current @cl/PME.cl#123-299 (further code continues solve/reduction).
  - Kernel `compute_tip_interaction` (not shown in snippet) computes tip-induced shifts (E) and tunneling factors (T) matching C++ multipole/tunneling conventions.

- **[docs/PME_OpenCL.md](cci:7://file:///home/prokop/git/ppafm/docs/PME_OpenCL.md:0:0-0:0)**
  - Design doc describing 1-workgroup-per-pixel layout, Gauss-Jordan solver, state ordering, sign conventions, and parity with C++ (E calc, rates, normalization row).

## Consolidated tests / comparison scripts

- **`tests/ChargeRings/compare_pme_solvers.py`**
  - Loads params/geometry, runs CPU (pauli) vs GPU (pauli_ocl) for single pixel; compares current, Es, Ts, Probs, StateEs; optional dump of kernel/current matrix @tests/ChargeRings/compare_pme_solvers.py#144-201.
- **`tests/ChargeRings/compare_cpu_gpu_line.py` / `compare_cpu_gpu_bias.py`**
  - 1D comparisons along spatial line or bias sweep; report per-point currents and diffs (CPU vs GPU).
- **`tests/ChargeRings/debug_W_Esite_propagation.py`**
  - Helper to trace how Wij/Esite propagate into solvers.
- **`tests/ChargeRings/PauliFastGUI.py`**
  - GUI wiring to choose CPU/GPU backends and visualize scans.

## CPU ↔ GPU equivalence map (key responsibilities)

- Many-body energies: C++ [calculate_state_energy](cci:1://file:///home/prokop/git/ppafm/cpp/pauli.hpp:108:0-144:1) / [calculate_state_energies](cci:1://file:///home/prokop/git/ppafm/cpp/pauli.hpp:526:4-539:5) ↔ OpenCL `solve_pme` phase 1 using `H_single_base + H_shifts` and Wij/W @cl/PME.cl#185-209.
- State ordering: C++ `state_order` (default identity; may follow charge order) ↔ `state_order_host/dev` (identity) in PauliSolverCL @pyProbeParticle/pauli_ocl.py#75-81.
- Tunneling amplitude construction: C++ [evalSitesTipsTunneling](cci:1://file:///home/prokop/git/ppafm/pyProbeParticle/pauli.py:169:0-178:16) + [calculate_tunneling_amplitudes](cci:1://file:///home/prokop/git/ppafm/cpp/pauli.hpp:656:4-665:5) (or prefilled TLeads) ↔ kernel 1 `compute_tip_interaction` producing `T_factors` and kernel 2 consuming `t_val` in rates @cl/PME.cl#246-269.
- Rate matrix: C++ [generate_coupling_terms](cci:1://file:///home/prokop/git/ppafm/cpp/pauli.hpp:1109:4-1185:5)+[generate_kern](cci:1://file:///home/prokop/git/ppafm/cpp/pauli.hpp:1198:4-1263:5) using `pauli_factors` ↔ OpenCL `solve_pme` loop over single-bit transitions with Fermi factors and lead couplings @cl/PME.cl#215-273.
- Normalization: C++ [normalize_kernel](cci:1://file:///home/prokop/git/ppafm/cpp/pauli.hpp:1187:4-1195:5) (row of ones) and Gauss/SVD solve ↔ OpenCL row-0 replacement before Gauss-Jordan @cl/PME.cl#276-288.
- Linear solve: C++ [gauss_solver.hpp](cci:7://file:///home/prokop/git/ppafm/cpp/gauss_solver.hpp:0:0-0:0) (or SVD) on host ↔ in-kernel parallel Gauss-Jordan (local memory).
- Valid point cut: C++ `is_valid_point` in `scan_current_tip_` zeroes current when Emax+W*EW_cut<0 or Tmax<Tmin_cut ↔ GPU post-filter in PauliSolverCL @pyProbeParticle/pauli_ocl.py#261-276.
- Outputs: CPU wrappers export current, Es, Ts, probabilities, kernel/current matrix via buffers; GPU path optionally returns the same (Es=H_shifts, Ts=T_factors, Probs, StateEs, K, CurMat) for single-pixel debug.
