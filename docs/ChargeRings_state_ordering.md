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
