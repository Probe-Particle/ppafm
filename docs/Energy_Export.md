# Plan for Implementing Many-Body State Energy Export and Visualization

## Introduction

This document outlines a step-by-step plan to extend the Pauli Master Equation (PME) simulator to export and visualize the energies of the many-body electronic states. This functionality will mirror the existing implementation for state probabilities, allowing for the display of state energies across scan points (e.g., tip positions or bias voltages).

## 1. C++ Backend Modifications (`/home/prokop/git/ppafm/cpp/pauli_lib.cpp`)

The primary goal in the C++ backend is to ensure that the calculated many-body state energies are copied into an output buffer provided by the Python interface during a scan.

*   **Function to Modify**: The core scanning function `scan_current_tip_` needs to be updated. This function is responsible for iterating through scan points and performing the Pauli solver calculations for each. A new argument should be added to its threaded variant `scan_current_tip_threaded` as well.

*   **New Argument**: Add a new argument to the `scan_current_tip_` function signature. This argument should be a pointer to a double, for example, `double* out_state_energies`. This pointer will point to a memory buffer pre-allocated in Python, where the energies for all states and all scan points will be stored.

*   **Data Source**: The `PauliSolver` class instance, accessible via the `solver` pointer within `scan_current_tip_`, already calculates and stores the many-body state energies in its `energies` member variable. This `energies` array holds the energies for the current scan point.

*   **Copying Data**: Inside the main loop of `scan_current_tip_` (where `ip` represents the current scan point index), after the Pauli solver has completed its calculations for that point (including the determination of state energies), copy the contents of `solver->energies` into the `out_state_energies` buffer. The correct offset for the current point `ip` will be `ip * solver->nstates`, where `solver->nstates` is the total number of many-body states.

*   **Threaded Versions**: The parallelized version of the scanning function, `scan_current_tip_threaded`, must also be updated to correctly pass and handle the `out_state_energies` buffer to its respective worker threads, which in turn call `scan_current_tip_`.

## 2. Python `ctypes` Interface Modifications (`/home/prokop/git/ppafm/pyProbeParticle/pauli.py`)

This layer acts as the bridge between Python and C++, so it needs to reflect the changes in the C++ function signature and manage the new data buffer.

*   **`ctypes` Binding Update**:
    *   Locate the `lib.scan_current_tip.argtypes` definition. This tuple specifies the data types of the arguments expected by the C++ `scan_current_tip` function.
    *   Append `c_double_p` (which represents a pointer to a double) to this tuple. This addition must correspond to the position of the new `out_state_energies` argument in the C++ function signature.

*   **`PauliSolver.scan_current_tip` Method**:
    *   **New Keyword Argument**: Add a new boolean keyword argument to the `scan_current_tip` method signature, for example, `return_state_energies=False`. This argument will allow Python callers to specify whether they want the state energy data returned.
    *   **Buffer Allocation**: Inside the `scan_current_tip` method, add a conditional block. If `return_state_energies` is `True`:
        *   Allocate a new NumPy array, for example, named `StateEnergies`. Its shape should be `(npoints, nstates)` and its data type `np.float64`. Initialize it with zeros.
    *   **Passing to C++**: In the call to `lib.scan_current_tip`, pass the `_np_as(StateEnergies, c_double_p)` as the argument corresponding to `out_state_energies`. If `return_state_energies` is `False`, pass `None` or a null pointer equivalent for this argument.
    *   **Return Value**: Modify the `return` statement of the `scan_current_tip` method. The `StateEnergies` array should be included in the returned tuple, similar to how `Probs` is currently returned.

## 3. High-Level Scan Logic Modifications (`/home/prokop/git/ppafm/tests/ChargeRings/pauli_scan.py`)

These functions orchestrate the calls to the `PauliSolver` for different types of scans (e.g., 2D XY plane, 1D xV line). They need to request and handle the new energy data.

*   **Functions to Modify**: `run_pauli_scan_top` (for XY scans) and `run_pauli_scan_xV` (for xV scans).

*   **Calling `PauliSolver.scan_current_tip`**:
    *   In both `run_pauli_scan_top` and `run_pauli_scan_xV`, locate the call to `pauli_solver.scan_current_tip`.
    *   Modify this call to pass `return_state_energies=True`. This ensures that the underlying C++ function calculates and returns the energy data.

*   **Receiving and Reshaping Data**:
    *   Update the unpacking of the return tuple from `pauli_solver.scan_current_tip` to capture the new `StateEnergies` array.
    *   Reshape the `StateEnergies` array from its flattened `(npoints, nstates)` form to the appropriate multi-dimensional shape for the scan. For `run_pauli_scan_top`, this would typically be `(npix, npix, nstates)`. For `run_pauli_scan_xV`, it would be `(nV, nx, nstates)`.

*   **Returning Data**: Include the reshaped `StateEnergies` array in the return tuple of both `run_pauli_scan_top` and `run_pauli_scan_xV`.

## 4. GUI Application Modifications (`/home/prokop/git/ppafm/tests/ChargeRings/CombinedChargeRingsGUI_v5.py`)

The GUI is the user-facing component, requiring new controls and plotting capabilities.

*   **Class Member**: Add a new member variable to the `ApplicationWindow` class, for example, `self.StateEnergies`, to store the calculated many-body state energies after a scan.

*   **GUI Checkbox**:
    *   Add a new `QCheckBox` widget to the GUI layout, for example, `self.cbPlotStateEnergies`. This checkbox will allow users to toggle the plotting of state energies. Place it logically near the existing `self.cbShowProbs` checkbox.
    *   Connect the `stateChanged` signal of this new checkbox to the `self.run` method, so that the plots update when the checkbox state changes.

*   **`run` Method**:
    *   **Calling Scan Function**: Update the calls to `pauli_scan.scan_xy_orb` (for XY scans) and `pauli_scan.calculate_xV_scan_orb` (for xV scans) to capture the new `StateEnergies` array from their return values.
    *   **Storing Data**: Store the received `StateEnergies` array in the `self.StateEnergies` member variable.
    *   **Conditional Plotting**: Add an `if self.cbPlotStateEnergies.isChecked():` block.
        *   Inside this block, create a new Matplotlib figure and axes specifically for plotting the state energies.
        *   Implement a new plotting function, for example, `plot_state_energies` (for XY scans) or `plot_state_energies_xV` (for xV scans). This function should be modeled closely after the existing `pauli_scan.plot_state_probabilities` function, but it will plot the energy of each state instead of its probability. It will take `self.StateEnergies` as input.
        *   Manage the new figure window using an existing utility like `self.manage_prob_window` to ensure it appears correctly and is associated with the main GUI.

*   **`save_everything` Method**:
    *   Modify the `data` dictionary that is passed to `np.savez`. Include `self.StateEnergies` in this dictionary under a suitable key, such as `'StateEnergies'`. This will ensure that the energy data is saved alongside other simulation results.

*   **`load_everything` Method (Future/Implicit)**:
    *   If a `load_everything` function is implemented (as outlined in `load_data_implementation_plan.md`), ensure that it can correctly load the data associated with the `'StateEnergies'` key from the `.npz` file and populate `self.StateEnergies` accordingly.

*   **`manage_prob_window`**: Verify that this utility function can handle the new figure created for plotting state energies, ensuring it is properly displayed and managed.