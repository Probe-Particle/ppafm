# WebGL STM Visualizer - Development Summary

## Project Overview
This project aimed to build a robust, interactive web-based Scanning Tunneling Microscopy (STM) visualizer. The goal was to simulate and render STM images of molecules (like PTCDA) in real-time using a simplified Tight-Binding model and WebGL acceleration.

## Architecture
The application is built using a hybrid CPU-GPU approach:
- **Physics Engine (CPU)**: JavaScript (`numeric.js`) handles the construction of the Tight-Binding Hamiltonian and its diagonalization to find eigenvalues and eigenvectors.
- **Rendering Engine (GPU)**: WebGL2 (via `three.js`) renders the STM heatmap. A custom fragment shader performs the heavy lifting of calculating the wavefunction density $|\psi|^2$ at every pixel.

## Key Features
- **Real-time Physics**: Adjustable parameters ($V_{pp\pi}$, $\beta$, Bias) with immediate re-calculation.
- **Shader-Based Rendering**: High-performance rendering using a full-screen quad.
- **Interactive UI**: Controls for orbital selection, color scale, and atom visualization.
- **Modes**:
    - **Full STM**: Sums contributions from all states within the bias window.
    - **Single Orbital**: Visualizes individual eigenstates (HOMO, LUMO, etc.).

## Implementation Details

### 1. Data Transfer (CPU $\to$ GPU)
To efficiently pass physics data to the shader, we used Data Textures:
- **`uAtomTexture`**: Stores atom positions $(x, y, z)$.
- **`uAtomParams`**: Stores decay parameters.
- **`uEigenvectors`**: A $N \times N$ texture storing eigenvector coefficients.
    - *Crucial Detail*: We store the $s$-orbital coefficient in the **Red channel**.

### 2. Shader Logic (`stm_shader.glslf`)
The shader computes the Local Density of States (LDOS) dynamically:
- **`eval_atom_decay(j, pos)`**: Computes $e^{-\beta |r - r_j|}$ for atom $j$.
- **`eval_orbital(k, pos)`**: Sums contributions from all atoms weighted by eigenvectors $C_{kj}$ to get $\psi_k(r)$.
- **Main Loop**: Iterates over relevant eigenstates (based on Bias or Selection) and sums $|\psi_k|^2$.

## Challenges & Solutions

### Challenge 1: The "Black Screen" & Texture Channels
**Problem**: Initial attempts resulted in a black screen despite correct logic. Debugging revealed that the shader was reading zeros for the orbital coefficients.
**Root Cause**: We were initially storing coefficients in the **Alpha channel** of the texture. On some WebGL implementations/drivers, reading from the Alpha channel of a `FloatType` texture can be problematic or clamped.
**Solution**: We moved the coefficient storage to the **Red channel** (`data[idx + 0]`). This immediately resolved the data transfer issue and made the orbitals visible.

### Challenge 2: 3D vs. 2D Complexity
**Problem**: The initial 3D implementation (volumetric raymarching or point clouds) was overly complex for the required "top-down" STM view and harder to debug.
**Solution**: We refactored the project to a **2D Shader-Based approach**. We used a single full-screen quad. This simplified the coordinate mapping (`vUv` $\to$ Angstroms) and allowed us to focus on the physics correctness without 3D camera artifacts.

### Challenge 3: Startup Synchronization
**Problem**: The simulation would not render immediately upon loading; the user had to toggle a control to trigger the first frame.
**Root Cause**: The physics calculation is asynchronous (or heavy), and the initial render call happened before the uniforms were fully populated or the solver had finished.
**Solution**:
1.  **Dummy Initialization**: We initialized `uEigenvalues` with a dummy `Float32Array` to prevent WebGL errors before the first solve.
2.  **Explicit Update Chain**: We ensured that `solvePhysics()` explicitly calls `updateSTM()` and then `updateVisuals()` upon completion.
3.  **Event Listener Fixes**: We corrected ID mismatches (e.g., `chkShowAtoms` vs `showAtoms`) that were breaking the initialization chain.

### Challenge 4: Verification
**Problem**: Ensuring the complex sum-of-exponentials math was correct in the shader.
**Solution**: We adopted a **Stepwise Verification** strategy:
1.  **Step 1**: Visualize raw exponential decay (atom "clouds") $\to$ Confirmed geometry.
2.  **Step 2**: Visualize a single orbital $\to$ Confirmed eigenvector lookup.
3.  **Step 3**: Full STM loop $\to$ Confirmed energy window summation.

## Future Improvements
- **P-Orbitals**: The current shader supports extending to p-orbitals by using the Green and Blue channels of the `uEigenvectors` texture.
- **Advanced Hamiltonians**: Support for different atom types with distinct on-site energies.
- **3D Isosurfaces**: Re-introducing 3D rendering for orbital isosurfaces now that the core logic is robust.
