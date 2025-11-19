# Analysis of Pauli_PME_phase1.glslf Shader Issues

## Executive Summary
The "black screen" and zero/NaN results in the shader are likely caused by **numerical underflow/instability** in the 4x4 Cramer's rule solver using 32-bit floats, exacerbated by a **discrepancy in the tunneling model** between JS and GLSL. Additionally, the large register pressure from 16x16 arrays may be a secondary factor, but the numerical issue is the primary suspect for the "zero vector" result.

## Key Findings

### 1. Discrepancy in Tunneling Model (Critical)
The JavaScript and GLSL implementations calculate the tunneling rates differently, which leads to vastly different matrix element magnitudes.

*   **JavaScript (`solvePmeCpu`):** Uses the `uDecay` parameter (via `useDecay=true`).
    ```javascript
    Ttun = Math.exp(-decay * r);
    ```
*   **GLSL (`build_pme_kernel`):** Ignores `uDecay` and uses a heuristic based on `uRtip`.
    ```glsl
    float Ttun = exp(-2.0 * r / max(uRtip, 1e-3));
    ```
    **Impact:** If `uDecay` (e.g., 1.0-2.0 Å⁻¹) differs significantly from `2.0/uRtip`, the rates $e^{-\beta r}$ will differ exponentially. If the shader rates are extremely small (e.g., $10^{-10}$), the determinant in Cramer's rule ($rate^4$) will underflow to zero in 32-bit floating point arithmetic ($10^{-40} < 10^{-38}$), causing the solver to return zeros.

### 2. Solver Stability: Cramer's Rule vs. Gaussian Elimination
*   **GLSL:** Uses **Cramer's Rule** for $N \le 4$. Cramer's rule involves computing determinants (products of $N$ terms). With small rates (tunneling), these products easily underflow 32-bit float precision.
    *   *Check:* `if (abs(detA) < 1e-20) ... return 0;`
    *   If rates are $\sim 10^{-5}$ or smaller, `detA` becomes $< 10^{-20}$, triggering the zero return.
*   **JavaScript:** Uses **Gaussian Elimination** (via `numeric.solve` or `solveLinearSystem_our`) with 64-bit doubles. Gaussian elimination is much more stable and handles scaling better.
    *   Double precision ($10^{-308}$) avoids underflow far better than single precision ($10^{-38}$).

### 3. Suspicious Matrix Normalization Logic
Both JS and GLSL contain a questionable step in `solve_pme` / `solvePmeCpu`:
```glsl
// Replace last row with normalization
int normRow = nStates - 1;
// ... set row normRow to 1.0 ...
// Clear the last column of OTHER rows?
for (int r = 0; r < nStates - 1; ++r) { K[r*16 + normRow] = 0.0; }
```
**Analysis:** Clearing the column `normRow` in other rows removes the contribution of transitions *from* the last state *to* other states. This effectively decouples the last state from the rest in the balance equations, which is mathematically incorrect for a general Master Equation.
*   **Why JS works:** It might be that `numeric.solve` is robust enough, or the specific physical parameters make those rates negligible, or the error is masked.
*   **Recommendation:** This should likely be removed in both, but for now, ensure the GLSL matches the JS behavior exactly if we want to reproduce JS results.

### 4. Memory / Register Pressure
*   The shader declares `float K[256]` (16x16) plus other arrays. This consumes significant register space (~1KB+).
*   While modern GPUs can handle this, it might reduce occupancy or cause compilation issues on some drivers.
*   **Mitigation:** Since `uNSites` is small (2 or 4), you can use `#define MAX_STATES 4` (or 8) to reduce array sizes if the shader is recompiled for specific cases, or simply trust the compiler. However, the "Zero/NaN" result points to numerical issues rather than a crash.

## Recommendations for Debugging & Fixes

### Step 1: Fix the Tunneling Model
Update `Pauli_PME_phase1.glslf` to use `uDecay` matching the JS implementation.

```glsl
// In build_pme_kernel
// Old: float Ttun = exp(-2.0 * r / max(uRtip, 1e-3));
// New:
float Ttun = exp(-uDecay * r); 
```
*Note: Ensure `uDecay` is passed correctly to `build_pme_kernel`.*

### Step 2: Switch to Gaussian Solver
Disable the Cramer's rule branch and force the use of the Gaussian elimination solver for all $N$. The Gaussian solver in the shader (lines 326+) handles pivoting and is more robust.

```glsl
// In solve_linear_system
// if (n <= 4) { ... }  <-- Comment out or remove this block to force Gaussian
```

### Step 3: Verify Matrix Values
Use the existing debug modes (Mode 9/10) to visualize `K` and `rhs`.
*   If `K` values are very small ($< 10^{-10}$), multiply the entire matrix and RHS by a scaling factor (e.g., $10^{10}$) before solving. The solution $\rho$ is invariant to scaling the system $K \rho = 0$.

### Step 4: Address Normalization Logic
If the above steps don't fix it, try removing the column-clearing loop:
```glsl
// for (int r = 0; r < nStates - 1; ++r) { K[r*16 + normRow] = 0.0; } // Try commenting this out
```
(Do this in JS first to see if it changes the result).

## Proposed Code Changes
I can apply a patch to `Pauli_PME_phase1.glslf` that:
1.  Corrects the `Ttun` calculation.
2.  Bypasses `solve_4x4_cramer`.
