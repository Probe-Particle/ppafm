# Pauli PME WebGL Implementation & Debugging Report

## Overview
This document summarizes the development and debugging session focused on enhancing the Pauli Master Equation (PME) WebGL shader and viewer. The goal was to improve physical correctness, numerical stability, and debugging capabilities for the Scanning Tunneling Microscopy (STM) simulation.

## Key Features Implemented

### 1. Dynamic Shader Compilation
*   **Problem:** The shader used hardcoded array sizes (e.g., `[4]`, `[16]`) and loop limits, making it inflexible for different system sizes (`NSites`).
*   **Solution:** Introduced `#define NSITE` and `#define NSTATE` macros in the GLSL code. The JavaScript viewer now dynamically replaces these macros with the actual values from the UI before compiling the shader.
*   **Benefit:** Optimizes memory usage and performance for small systems while allowing scaling to larger ones without manual code changes.

### 2. Enhanced Debugging Tools
*   **Granular Debug Modes:** Added specific render modes to visualize internal shader state:
    *   `DEBUG: K[idx]`, `DEBUG: rhs[idx]`: Inspect elements of the rate matrix and RHS vector.
    *   `DEBUG: uW`: Verify uniform transmission.
    *   `DEBUG: nStates`, `DEBUG: Es[idx]`: Verify state counting and energy calculations.
*   **Debug Indexing:** Added `uDebugI` and `uDebugJ` uniforms (controlled via UI) to select specific matrix/vector elements for visualization.
*   **Raw Data Readback:** Implemented `readPixelsDebug` using a `THREE.FloatType` render target to read exact float values from the GPU, bypassing the color map. This allows quantitative verification of the simulation.

### 3. Physics & Numerical Improvements
*   **Row Normalization:**
    *   **Problem:** Large `decay` values (weak coupling) caused numerical noise/artifacts in the PME solution due to floating-point underflow/instability in the linear solver.
    *   **Solution:** Implemented row normalization in `solve_pme`. Each row of the rate matrix `K` is normalized by its maximum element before solving. This stabilizes the Gaussian elimination process.
*   **Tunneling Decay Fix:**
    *   **Problem:** The tunneling rate calculation incorrectly used `uRtip` instead of the explicit `uDecay` parameter in some functions.
    *   **Solution:** Updated `ground_state_tip_current` and `compute_tip_current` to correctly use `exp(-uDecay * r)`.
*   **Coulomb Interaction (W) Fix:**
    *   **Problem:** The `W` parameter appeared to have no effect.
    *   **Root Cause:** Hardcoded loop limits (e.g., `s < 16`) in `compute_many_body_energies` prevented the loop from running correctly when `NSTATE` was dynamically defined or small, leading to zeroed energies for higher states.
    *   **Solution:** Replaced all hardcoded limits with the `NSTATE` macro.

### 4. UI Refinements
*   **Logarithmic Scaling:** Added `OutScale (10^x)` to allow visualizing data over many orders of magnitude (crucial for tunneling currents).
*   **Step Sizes:** Added `step` attributes to input fields for finer control with the mouse wheel.
*   **Defaults:** Updated default Mode to PME Tip Current (5) and optimized default parameters.

### 5. XV Scan Mode (Position vs Bias)
*   **Functionality:** Added a new visualization mode that scans spatially along a line (X-axis) while varying the bias voltage (Y-axis). This is critical for observing Coulomb blockade "diamonds" and resonant tunneling features.
*   **Controls:**
    *   **Scan Mode:** Toggle between `XY` (Top-down spatial map) and `XV` (Position vs Bias).
    *   **P1 / P2:** Define the start and end points of the spatial line scan (X-axis).
    *   **VBiasMin / Max:** Define the voltage range for the Y-axis.
*   **Implementation:**
    *   The shader now accepts `uScanMode`, `uP1`, `uP2`, `uVBiasMin`, and `uVBiasMax` uniforms.
    *   In `XV` mode, the shader interpolates the tip position between `P1` and `P2` based on the pixel's x-coordinate (`vUv.x`) and interpolates the bias voltage between `VBiasMin` and `VBiasMax` based on the y-coordinate (`vUv.y`).
    *   This allows the same physics core to render both spatial maps and spectroscopic plots without code duplication.

## Debugging Strategies & Lessons Learned

### 1. Granular Visual Debugging
*   **Strategy:** When a complex output (like Current) is wrong, don't just stare at the final image. Create temporary debug modes to visualize intermediate variables (e.g., "Is `nStates` correct?", "Is `Es[3]` non-zero?").
*   **Example:** We confirmed `W` was broken by visualizing `Es[3]` directly, which showed `0.0` despite `W > 0`. This immediately pointed to the energy calculation loop.

### 2. CPU Reference Implementation
*   **Strategy:** Maintain a JavaScript/CPU version of the physics logic (`pauli_pme_cpu.js`) that mirrors the GLSL code.
*   **Benefit:** This allows step-by-step debugging (breakpoints, console logs) that is impossible in a shader. If the CPU version works and the GPU version doesn't, the issue is likely in the GLSL implementation details (precision, types, loop limits) rather than the physics equations.

### 3. Raw Data Verification
*   **Strategy:** Never trust colors alone. A "black" pixel could be `0.0`, `-1e-20`, or `NaN`.
*   **Benefit:** Reading back the raw float values proved that the "noise" was numerical instability (values jumping wildly) rather than a logic error.

### 4. Common GLSL Pitfalls
*   **Hardcoded Loops:** In GLSL 3.0, loops often need constant bounds. However, relying on "magic numbers" (like 16) inside functions makes the code brittle when dimensions change. **Takeaway:** Use preprocessor macros for array sizes and loop bounds.
*   **Parameter Propagation:** It's easy to add a `uniform` but forget to use it in one specific helper function (as happened with `uDecay`). **Takeaway:** Search for the variable name globally to ensure it's used consistently in all relevant contexts.
*   **Numerical Stability:** Linear solvers (Gaussian elimination) on the GPU (float precision) are prone to instability with ill-conditioned matrices (e.g., very small rates). **Takeaway:** Always normalize rows/columns for rate equation solvers.

## Future Development Recommendations
1.  **Automated Tests:** Implement a mode that runs both CPU and GPU solvers on the same input and alerts if the difference exceeds a tolerance.
2.  **Modular Shader Code:** As the shader grows, consider splitting it into chunks (strings in JS) that are assembled, to avoid copy-pasting common logic between "Phase 1" and "Phase 2" shaders.
3.  **Visualizing Matrices:** The "Debug K[idx]" mode is useful, but a mode that renders the entire $N \times N$ matrix as a small texture overlay could be even better for quick diagnostics.
