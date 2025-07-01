# Pauli Master Equation Solver Documentation

The Pauli Master Equation (PME) solver in `ppafm` is a C++ implementation designed to simulate quantum transport phenomena in nanoscale systems, particularly for calculating current and state probabilities in quantum dots coupled to leads. It leverages `ctypes` for seamless integration with Python, providing a powerful tool for analyzing scanning tunneling microscopy (STM) data.

The core of the solver resides in the `PauliSolver` C++ class, which encapsulates the system's quantum states, energies, and coupling parameters, and performs the necessary linear algebra to solve the PME.

## Core Concepts

*   **Many-body states:** Combinations of occupied single-particle states in the quantum dot.
*   **Single-particle energies ($H_{ii}$):** Energies of individual electron orbitals in the quantum dot.
*   **Hopping terms ($H_{ij}$):** Tunneling amplitudes between different single-particle orbitals.
*   **Coulomb interaction ($W$):** Energy cost for having multiple electrons on the quantum dot.
*   **Leads:** External reservoirs (e.g., substrate, tip) that exchange electrons with the quantum dot. Each lead has a chemical potential ($\mu$) and temperature ($T$).
*   **Tunneling amplitudes ($T_{ba}^{(l)}$):** Matrix elements describing the rate at which an electron tunnels between a many-body state $b$ and another many-body state $a$ via lead $l$.
*   **Pauli factors ($\Gamma_{ba}^{(l)}$):** Transition rates between many-body states due to tunneling through a specific lead, incorporating Fermi-Dirac statistics.
*   **Kernel matrix ($K_{ab}$):** The rate matrix of the PME, describing the net flow of probability between all many-body states.
*   **Probabilities ($\rho_s$):** The steady-state probabilities of finding the system in a particular many-body state $s$.
*   **Current ($I_l$):** The net flow of electrons through a specific lead.

## Calculation Flow (Scheme)

The calculation proceeds in a series of steps, orchestrated by the `PauliSolver` class methods. The typical workflow for a single point calculation (e.g., within a scan) is as follows:

1.  **System Initialization:**
    *   The `PauliSolver` object is created, allocating memory for internal arrays (`energies`, `coupling`, `kernel`, `probabilities`, `Hsingle`, `TLeads`, `leads`, `state_order`, `state_order_inv`).
    *   Initial lead parameters (mu, temperature) are set.

2.  **Input Parameter Setup:**
    *   The single-particle Hamiltonian (`Hsingle`), Coulomb interaction strength (`W`), and lead tunneling amplitudes (`TLeads`) are provided.
    *   The many-body states are organized by charge (`init_states_by_charge()`) and an ordering is established (`init_state_ordering()`).

3.  **Calculate State Energies:**
    *   The `calculate_state_energies()` method computes the energy $E_s$ for each many-body state $s$.

4.  **Calculate Lead Coupling Matrix Elements:**
    *   The `eval_lead_coupling()` method computes the tunneling amplitudes $T_{ba}^{(l)}$ between many-body states $b$ and $a$ via each lead $l$.

5.  **Generate Pauli Factors:**
    *   The `generate_fct()` method calculates the forward and backward transition rates ($\Gamma_{cb}^{(l, \text{forward})}$ and $\Gamma_{cb}^{(l, \text{backward})}$) for all relevant transitions between states differing by one electron, considering the Fermi-Dirac distribution of each lead.

6.  **Construct Kernel Matrix:**
    *   The `generate_kern()` method, by calling `generate_coupling_terms()` for each state, populates the kernel matrix $K_{ab}$ using the pre-calculated Pauli factors.

7.  **Solve Linear System for Probabilities:**
    *   The `solve_kern()` method solves the linear system $K \rho = \text{rhs}$ to find the steady-state probabilities $\rho_s$ of each many-body state. A normalization condition ($\sum_s \rho_s = 1$) is applied by modifying the kernel matrix and RHS vector.

8.  **Calculate Current:**
    *   The `generate_current()` method computes the net current $I_l$ flowing through a specified lead $l$, based on the steady-state probabilities and the Pauli factors.

```mermaid
graph TD
    A[PauliSolver Initialization] --> B{Input Parameters Set?};
    B -- Yes --> C[Calculate State Energies $E_s$];
    C --> D[Calculate Lead Coupling $T_{ba}^{(l)}$];
    D --> E[Generate Pauli Factors $\Gamma_{ba}^{(l)}$];
    E --> F[Construct Kernel Matrix $K_{ab}$];
    F --> G[Solve $K \rho = \text{rhs}$ for $\rho_s$];
    G --> H[Calculate Current $I_l$];
    H --> I[Results];
    B -- No --> J[Update Parameters];
    J --> C;
```

## Key Functions and Mathematical Meaning

### `calculate_state_energy(int state, int nSingle, const double* Hsingle, double W)`

This function calculates the total energy of a given many-body state. A many-body state is represented by an integer where each bit corresponds to the occupancy of a single-particle orbital.

*   **Physical Meaning:** The total energy of a specific configuration of electrons in the quantum dot, considering single-particle energies, hopping between occupied orbitals, and Coulomb repulsion between occupied pairs.
*   **Mathematical Expression:**
    $$E_s = \sum_{i \in s} H_{ii} + \sum_{\substack{i,j \in s \\ i<j}} (H_{ij} + H_{ji}) + N_p W$$
    Where:
    *   $s$: The set of occupied single-particle states in the many-body state `state`.
    *   $H_{ii}$: Diagonal elements of the single-particle Hamiltonian, representing the energy of orbital $i$.
    *   $H_{ij}$: Off-diagonal elements of the single-particle Hamiltonian, representing hopping between orbitals $i$ and $j$.
    *   $N_p$: The number of pairs of occupied single-particle states in `state`.
    *   $W$: The Coulomb interaction strength between any two occupied sites.

### `eval_lead_coupling(int lead, const double* TLead)`

This method computes the tunneling amplitudes between many-body states for a specific lead. It considers transitions where only one electron is added or removed from a single-particle site. The fermionic sign is crucial for maintaining proper quantum statistics.

*   **Physical Meaning:** Quantifies the strength of electron tunneling between the quantum dot and a specific lead, leading to a change in the many-body state of the dot.
*   **Mathematical Expression (Conceptual, simplified):**
    The code calculates $T_{final, initial}^{(l)}$, where $final$ and $initial$ are many-body states.
    $$T_{final, initial}^{(l)} \propto (-1)^{\sum_{k=0}^{site-1} \text{occupied}_k} \cdot t_{l,site}$$
    Where:
    *   $t_{l,site}$: The tunneling amplitude between lead $l$ and single-particle site `site`.
    *   $(-1)^{\sum_{k=0}^{site-1} \text{occupied}_k}$: The fermionic sign, which depends on the number of occupied single-particle states *before* the `site` involved in the tunneling event. This ensures the Pauli exclusion principle is respected.

### `generate_fct()`

This function calculates the Pauli factors, which are the fundamental transition rates between many-body states. These rates depend on the tunneling amplitudes, the energy difference between states, and the Fermi-Dirac distribution of the leads.

*   **Physical Meaning:** The rate at which the quantum dot transitions from one many-body state to another by exchanging an electron with a specific lead. This includes both electron "entering" (forward) and "leaving" (backward) processes.
*   **Mathematical Expression:**
    For a transition from state $b$ to state $c$ (where $c$ has one more electron than $b$):
    $$\Gamma_{cb}^{(l, \text{forward})} = |T_{cb}^{(l)}|^2 \cdot f_l(E_c - E_b) \cdot 2\pi$$
    $$\Gamma_{cb}^{(l, \text{backward})} = |T_{cb}^{(l)}|^2 \cdot (1 - f_l(E_c - E_b)) \cdot 2\pi$$
    Where:
    *   $T_{cb}^{(l)}$: The tunneling amplitude between state $c$ and $b$ via lead $l$.
    *   $f_l(E) = \frac{1}{1 + e^{(E - \mu_l)/k_B T_l}}$: The Fermi-Dirac distribution function for lead $l$.
    *   $E_c - E_b$: The energy difference between the final state $c$ and the initial state $b$.
    *   $\mu_l$: Chemical potential of lead $l$.
    *   $T_l$: Temperature of lead $l$ (in energy units, $k_B T_{Kelvin}$).
    *   $k_B$: Boltzmann constant.

### `generate_kern()` and `generate_coupling_terms(int b)`

These methods construct the kernel matrix $K$, which is the central component of the Pauli Master Equation. The kernel matrix describes the net flow of probability between all many-body states.

*   **Physical Meaning:** The kernel matrix represents the system of linear equations that, when solved, yield the steady-state probabilities of the quantum dot's many-body states. Diagonal elements represent the total rate of leaving a state, while off-diagonal elements represent the rate of entering a state from another.
*   **Mathematical Expression:**
    The system of equations is given by:
    $$\frac{d\rho_s}{dt} = \sum_{a \neq s} (K_{sa} \rho_a - K_{as} \rho_s)$$
    In steady-state, $\frac{d\rho_s}{dt} = 0$, leading to $\sum_a K_{sa} \rho_a = 0$.
    The elements of the kernel matrix are constructed from the Pauli factors:
    $$K_{ss} = - \sum_{a \neq s} \Gamma_{as}$$
    $$K_{sa} = \Gamma_{sa} \quad \text{for } s \neq a$$
    More specifically, for a state $b$:
    *   **Diagonal element $K_{bb}$:** Sum of all rates *leaving* state $b$ (negative contribution) and all rates *entering* state $b$ (positive contribution from other states).
        $$K_{bb} = - \sum_{l} \left( \sum_{a \text{ (charge } Q-1)} \Gamma_{ba}^{(l, \text{backward})} + \sum_{c \text{ (charge } Q+1)} \Gamma_{cb}^{(l, \text{forward})} \right)$$
    *   **Off-diagonal element $K_{bb, aa}$ (transition $a \to b$):** Rate of transition from state $a$ to state $b$.
        $$K_{bb, aa} = \sum_{l} \Gamma_{ba}^{(l, \text{forward})}$$
    *   **Off-diagonal element $K_{bb, cc}$ (transition $c \to b$):** Rate of transition from state $c$ to state $b$.
        $$K_{bb, cc} = \sum_{l} \Gamma_{cb}^{(l, \text{backward})}$$
    *   **Note on `state_order2`:** The C++ code in `generate_kern()` hardcodes `state_order2 = {0,1,2,4,3,5,6,7};`. This is likely a debug value and should ideally use the dynamically generated `state_order` array (which maps original state indices to ordered indices for the kernel matrix) to ensure correct indexing for arbitrary `nSingle` values. The intended behavior is to use the ordered indices for `bb` and `aa` when setting kernel elements.

### `solve_kern()`

This method solves the linear system defined by the kernel matrix to obtain the steady-state probabilities of the many-body states.

*   **Physical Meaning:** Determines the long-term average occupation probability of each many-body state in the quantum dot under steady-state conditions.
*   **Mathematical Expression:**
    The system of equations is $\sum_a K_{sa} \rho_a = 0$.
    To ensure a unique solution, the normalization condition $\sum_s \rho_s = 1$ is applied. This is typically done by replacing one row of the kernel matrix (e.g., the first row) with all ones, and setting the corresponding element of the RHS vector to 1 (and others to 0).
    $$K' \rho = \text{rhs}'$$
    Where $K'$ is the modified kernel matrix and $\text{rhs}'$ is the modified RHS vector. This system is then solved using a linear solver (Gaussian elimination or SVD).

### `generate_current(int lead_idx)`

This function calculates the net current flowing through a specified lead.

*   **Physical Meaning:** The measurable electrical current flowing between the quantum dot and an external lead, resulting from the dynamic exchange of electrons.
*   **Mathematical Expression:**
    $$I_l = \sum_{\substack{\text{charge } Q \\ \text{state } b}} \sum_{\substack{\text{charge } Q+1 \\ \text{state } c}} \left( \rho_b \cdot \Gamma_{cb}^{(l, \text{forward})} - \rho_c \cdot \Gamma_{bc}^{(l, \text{backward})} \right)$$
    Where:
    *   $\rho_b$, $\rho_c$: Steady-state probabilities of states $b$ and $c$.
    *   $\Gamma_{cb}^{(l, \text{forward})}$: Rate of electron entering the dot from lead $l$, transitioning from state $b$ to $c$.
    *   $\Gamma_{bc}^{(l, \text{backward})}$: Rate of electron leaving the dot to lead $l$, transitioning from state $c$ to $b$.


## State Ordering and Data Consistency in the Pauli Solver

During analysis of the simulation results, a critical inconsistency was discovered between the ordering of the many-body state energies (`stateEs`) and the steady-state probabilities (`probabilities`) returned by the solver. This document details the problem, its root cause in the C++ backend, and the implications for data analysis.

### The Observed Problem

When running simulations and plotting the results, it was observed that the `stateEs` and `probabilities` arrays did not align correctly, even after applying a permutation based on the expected state labels.

For a symmetric system (e.g., a scan over one vertex of a triangular arrangement of sites), one would expect that states with equivalent energies (e.g., `|100>` and `|010>`) would also have similar probabilities. However, the output showed a contradiction:

*   Based on the **energies**, it appeared that site `0` (the first bit in `|100>`) was the "special" one, as its energy was distinct from the other two singly-occupied states.
*   Based on the **probabilities**, it appeared that site `2` (the last bit in `|001>`) was the "special" one, as its probability was orders of magnitude different from the others.

This pointed to a fundamental mismatch in how the states were being ordered for each calculation.

### Investigation and Root Cause in `pauli.hpp`

A detailed review of the C++ solver code in `/home/prokop/git/ppafm/cpp/pauli.hpp` revealed the source of the inconsistency. The `PauliSolver` class uses **two different and conflicting state ordering schemes** for different parts of the calculation.

#### 1. `state_order` (The "QmeQ" or "Set" Order)

*   **Source:** This is an integer array pointer (`int* state_order`) within the `PauliSolver` class.
*   **Origin:** It is correctly set from Python via the `setStateOrder(const int* newStateOrder)` method. In our test case, this corresponds to the `labels_set` order (`[0, 4, 2, 6, 1, 5, 3, 7]`).
*   **Usage:** The `calculate_state_energies()` function **correctly uses this `state_order`** to compute the energy for each state.
*   **Result:** The returned `stateEs` array is ordered according to the `labels_set` provided from Python.

#### 2. `state_order2` (A Hardcoded Order)
*   **Note:** This was an initial finding. The hardcoded vector has since been corrected to a natural order (`{0,1,2,3,4,5,6,7}`). However, a more fundamental issue remained.

*   **Source:** Inside the `generate_kern()` function, a local `std::vector<int>` named `state_order2` was **hardcoded** with the value:
    `state_order2 = {0,1,2,4,3,5,6,7};`
*   **Usage:** The `generate_coupling_terms()` function, which is responsible for building the kernel matrix, **exclusively uses this hardcoded `state_order2`** to map state indices to kernel matrix indices (e.g., `int bb = state_order2[b];`).
*   **Result:** Because the kernel matrix is constructed in the basis defined by `state_order2`, the resulting `probabilities` array (which is the solution to the master equation `Kernel * p = rhs`) is also ordered according to this hardcoded `state_order2`.

#### 3. Hardcoded Bit-to-Site Mapping in `eval_lead_coupling`

Even after correcting `state_order2`, simulations still failed when using a natural state order (`[0, 1, 2, 3, 4, 5, 6, 7]`). This pointed to another, more subtle hardcoded assumption.

*   **Source:** The `eval_lead_coupling` function, which calculates the tunneling amplitudes between many-body states.
*   **Problem:** This function contains a hardcoded **reversed bit mapping** when calculating which site is affected by tunneling:
    `int site = nSingle - 1 - j2;`
*   **Implication:** This logic assumes that the physical site index `j2` (from the `TLeads` array) corresponds to a reversed bit position in the many-body state's integer representation. For a 3-site system, this means:
    *   Site 0 (from `TLeads`) maps to bit 2 (e.g., `|100>`).
    *   Site 1 maps to bit 1 (e.g., `|010>`).
    *   Site 2 maps to bit 0 (e.g., `|001>`).
*   **Consequence:** This convention is only compatible with a state ordering that also follows this bit-significance grouping, such as the QmeQ `labels_set`. When a different ordering (like the natural one) is used, `eval_lead_coupling` applies the tunneling amplitudes to the wrong bit positions relative to the state energies, breaking the physical model and producing incorrect results.

### Conclusion and Implications

The solver contains multiple hardcoded assumptions that prevent it from working correctly with arbitrary state orderings. The `state_order` passed from Python is only partially respected.

*   **`stateEs` is ordered by `state_order`** (from Python, e.g., `[0, 4, 2, 6, 1, 5, 3, 7]`).
*   **`probabilities` is ordered by the hardcoded `state_order2`** (now fixed to natural order) before was `{0,1,2,4,3,5,6,7}` which was causing some problems.
*   **Tunneling Amplitudes (`coupling` matrix)** are calculated with a hardcoded reversed bit-to-site mapping, making them incompatible with a natural state ordering.

This explains the misalignment perfectly. To correctly analyze the data in Python, one must be aware of these two different orderings and apply the correct permutations to align the data for plotting and analysis.

For a long-term solution, the C++ code should be modified to consistently use a single state ordering scheme (ideally the one passed from Python) throughout the entire calculation pipeline.






## Usage from Python

The C++ `PauliSolver` is exposed to Python via `ctypes` bindings in `pyProbeParticle/pauli.py`. The `PauliSolver` Python class acts as a wrapper, providing methods that directly call the underlying C++ functions.

A typical Python workflow involves:

1.  **Instantiating the solver:**
    ```python
    import pauli as psl
    solver = psl.PauliSolver(nSingle=3, nleads=2, verbosity=0)
    ```
2.  **Setting lead parameters:**
    ```python
    solver.set_lead(0, mu_substrate, temp_substrate)
    solver.set_lead(1, mu_tip, temp_tip)
    ```
3.  **Setting system parameters:**
    ```python
    solver.set_hsingle(H_single_matrix)
    solver.setW(coulomb_interaction_W)
    solver.set_tunneling(T_leads_matrix)
    ```
4.  **Running the calculation:**
    For a single point, the `solve_hsingle` C++ function (wrapped by Python) can be used, which orchestrates the calculation of Pauli factors, kernel, and probabilities:
    ```python
    current = solver.solve_hsingle(H_single_matrix, W, lead_idx_for_current, state_order_array)
    ```
    For scanning over multiple points (e.g., tip positions), higher-level functions like `pauli.run_pauli_scan_top` or `pauli.run_pauli_scan_xV` (from `pauli_scan.py`) are used. These functions prepare input arrays for many points and efficiently call the C++ `scan_current_tip` (or its threaded variants) which handles the loop and parallelization.

5.  **Retrieving results:**
    ```python
    probabilities = solver.get_probabilities(nstates)
    kernel = solver.get_kernel(nstates)
    # etc.
    ```
6.  **Cleaning up:**
    ```python
    solver.cleanup() # Frees C++ memory
    ```