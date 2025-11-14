# Design: GLSL-Based Probe-Particle (PP) Model for Web / WebGL

This document outlines how to implement a **simplified, GLSL-based** version of the Probe-Particle AFM/STM model, computing forces *on the fly* instead of precomputing force grids (as in the current C++ / OpenCL code).

The implementation is intended for WebGL (GLSL fragment shaders) driven from JavaScript.

---

## 1. Relevant Existing Files / References

### Core description

- **[README.md](cci:7://file:///home/prokophapala/git/tmp/ppafm/README.md:0:0-0:0)**  
  High-level description of the Probe-Particle Model, references to the original papers, and notes on the OpenCL GUI.

### C++ (reference, grid-based)

- **[cpp/ProbeParticle.cpp](cci:7://file:///home/prokophapala/git/tmp/ppafm/cpp/ProbeParticle.cpp:0:0-0:0)**  
  - Complete CPU implementation:
    - Classical force field between PP and sample (Morse, Coulomb, LJ, etc.).
    - Tip model (radial + angular springs, optional spline-based radial spring).
    - Relaxation algorithms: simple damped MD and FIRE.
    - Force field precomputation on regular 3D grid (`gridF`, `gridE`).
    - Relaxation of PP at each tip position using **interpolation** in this precomputed grid.
  - Important functions / concepts:
    - [addAtomMorse](cci:1://file:///home/prokophapala/git/tmp/ppafm/cpp/ProbeParticle.cpp:185:0-193:1), [addAtomCoulomb](cci:1://file:///home/prokophapala/git/tmp/ppafm/cpp/ProbeParticle.cpp:195:0-203:1), etc.: pairwise interactions.
    - [forceRSpring](cci:1://file:///home/prokophapala/git/tmp/ppafm/cpp/ProbeParticle.cpp:123:0-128:1), [forceRSpline](cci:1://file:///home/prokophapala/git/tmp/ppafm/cpp/ProbeParticle.cpp:130:0-143:1), [forceSpringRotated](cci:1://file:///home/prokophapala/git/tmp/ppafm/cpp/ProbeParticle.cpp:145:0-159:1): tip mechanics.
    - [relaxProbe](cci:1://file:///home/prokophapala/git/tmp/ppafm/cpp/ProbeParticle.cpp:302:0-316:1), [relaxTipStroke](cci:1://file:///home/prokophapala/git/tmp/ppafm/cpp/ProbeParticle.cpp:453:0-499:1): iterative MD / FIRE-based relaxation using the grid field.

### OpenCL (GPU, grid-based)

- **[cl/FF.cl](cci:7://file:///home/prokophapala/git/tmp/ppafm/cl/FF.cl:0:0-0:0)** (Force Field kernels)  
  - Kernels to compute **force + energy on a 3D grid**:
    - `evalLJ`, `evalCoulomb`, `evalLJC_Q`, `evalLJC_Q_noPos`, etc.
    - Combined Lennard-Jones + Coulomb, vectorized in local memory.
  - Also contains various rendering-oriented kernels (spheres, bonds, maps).

- **[cl/relax.cl](cci:7://file:///home/prokophapala/git/tmp/ppafm/cl/relax.cl:0:0-0:0)** (Relaxation & convolution on precomputed grid)
  - Takes the 3D force field as an OpenCL 3D image.
  - Interpolates force at PP positions, performs **dynamic relaxation** (FIRE / damped MD).
  - Integrates forces along oscillation amplitude using the **Giessibl formula**:
    - Convolution along z (`convolveZ`, `relaxStrokesTilted_convZ`, etc.).
  - Outputs relaxed forces / Δf maps.

### “Original” GPU strategy vs new GLSL strategy

- **Original** (C++/OpenCL):  
  1. Compute 3D force field on a grid.  
  2. Relax PP using interpolation in that precomputed field.  
  3. Convolve along oscillation amplitude to obtain Δf / STM / IETS.

- **New (this design)**:  
  - **No precomputed grid**.  
  - PP force is computed **directly from atoms** in the fragment shader at each sampled tip position.

---

## 2. Goals for the GLSL / Web Implementation

We want a **simplified, interactive** PP model suitable for a WebGL demo:

1. **No force field grid**:
   - Compute pairwise interactions atom–PP **on-the-fly** in the fragment shader.

2. **Approximate relaxation**:
   - Instead of an iterative MD/FIRE loop for each pixel, approximate relaxation via a **Hooke’s law correction**:
     - `Δr ≈ F / k` (or a few iterations of this).
   - Greatly reduces per-pixel cost; more appropriate for real-time rendering on the web.

3. **Staged implementation**:

   1. **Stage 1: Framework + simple visualization**
      - Load shader(s) and molecule geometry (`.xyz`) in JS.
      - Pass parameters from HTML input boxes to GLSL via uniforms.
      - Fragment shader draws atoms as simple radial blobs (e.g. `exp(-r^2 / σ^2)`).
      - Use this to verify geometry, tip height / imaging plane, coordinate conventions.

   2. **Stage 2: Realistic pair potentials**
      - Implement a combined **Morse + electrostatic** interaction:
        - `V_total = Σ_atoms [ V_Morse + V_Coulomb ]`
        - `F = -∇V_total` computed analytically per pair.
      - No relaxation yet, just forces at fixed PP position.

   3. **Stage 3: Hooke-law PP relaxation**
      - For each pixel:
        1. Start from “bare” PP position (tip apex + equilibrium offset).
        2. Evaluate force from sample (`F_sample`).
        3. Compute displacement `Δr = F_sample / K` (Hooke’s law).
        4. Optionally repeat 2–3 iterations (re-evaluating `F` at displaced position).
      - Use displaced PP position and/or force in rendering.

   4. **Stage 4: Giessibl formula (amplitude integration)**
      - Approximate Δf by sampling the force along a short segment of the oscillation path:
        - `z(t) = z0 + A cos(t)`
        - Δf ∝ ⟨∂Fz/∂z⟩ or the standard Giessibl convolution.
      - Numerically implement this as a small convolution over a few z-samples per pixel in GLSL or in JS.

---

## 3. Data Flow and Representation in the WebGL Version

### 3.1 Inputs

- **Atomic geometry**
  - Loaded from `.xyz` file via JavaScript.
  - Data: positions and types/charges.
  - Transferred to GLSL as:
    - Uniform arrays (for small systems), or
    - A texture buffer / uniform buffer (for larger, but we can start with simple arrays).

- **Physical parameters**
  - From HTML controls (input boxes, sliders):
    - Morse parameters (r₀, ε, α) per atom type or global.
    - Coulomb scaling, atomic charges.
    - Tip stiffness `Kx, Ky, Kz` (or scalar `K`).
    - PP equilibrium offset relative to tip apex.
    - Imaging plane z, resolution, zoom, etc.
  - Passed as uniforms.

- **Tip / imaging setup**
  - We conceptually render a 2D image of the scan:
    - Each fragment corresponds to a lateral tip position `(x_tip, y_tip)` at fixed `z_tip`.
    - The PP base position is `r_PP0 = (x_tip, y_tip, z_tip) + offset`.

### 3.2 Fragment shader per pixel

At each fragment (representing a pixel in the simulated AFM/STM image):

1. Compute **tip/PP position** from fragment coordinates.
2. For Stage 1:
   - Render atoms as radial blobs:
     - For each atom:
       - `r = |r_PP0 - r_atom|`
       - Accumulate something like `signal += exp(-r * decay)` or Gaussian.
   - Use this as brightness to check geometry and image scaling.

3. For Stage 2:
   - Evaluate **pairwise potential and force** at PP position:
     - For each atom:
       - `dp = r_PP - r_atom`
       - **Morse**:
         - `r = |dp|`
         - `expar = exp(α (r - r0))`
         - `E_Morse = ε (expar^2 - 2 expar)`
         - `F_Morse = ε * 2 α (expar^2 - expar) * dp / r`
       - **Coulomb**:
         - `ir2 = 1 / (|dp|^2 + R2SAFE)`
         - `ir = sqrt(ir2)`
         - `E_C = k_coulomb * Qa*Qb * ir`
         - `F_C = k_coulomb * Qa*Qb * ir2 * dp * ir` (or equivalent).
     - Sum over atoms:
       - `F_sample = Σ_i (F_Morse_i + F_C_i)`
       - Optionally also accumulate total energy.

4. For Stage 3 (Hooke-law relaxation):
   - Start with `r_PP = r_PP0`.
   - For iteration = 1..N_iter (e.g. 1–3):
     - Compute `F_sample(r_PP)`.
     - Compute elastic restoring force from tip:
       - Simplified isotropic Hooke:
         - `F_tip = -K * (r_PP - r_PP0)` (vector).
       - Total force: `F_total = F_sample + F_tip`.
     - Update PP position:
       - `Δr = F_total / K_eff` (Hessian approximated by K)
       - Or more precisely: `r_PP = r_PP0 + F_sample / K` (if we treat K as dominating).
   - After convergence/iterations, we have:
     - Relaxed position `r_PP_relaxed`.
     - Possibly use `F_total` or `F_sample` in the signal.

5. For Stage 4 (Giessibl / amplitude integration):
   - For a few z-samples `z_j = z0 + A * f(j)` around `z0`:
     - Repeat the above steps to compute `F_z(z_j)` or `F_total(z_j)`.
   - Convolve with precomputed weights (Giessibl kernel) to approximate Δf.
   - This can be done:
     - Directly in the fragment shader (loop over j), or
     - In multiple passes, or
     - In JS for small image sizes.

---

## 4. Detailed Staging / Implementation Plan

### Stage 1 – Framework + atom visualization

**Objectives:**

- WebGL/JS side:
  - Load `.xyz` file.
  - Parse atom positions and types.
  - Pass them to a **simple fragment shader**.
  - Hook HTML input elements to uniforms (tip height, zoom, viewing transform).
- GLSL side:
  - Render a 2D “height map” or “density map” of atoms using a simple radial basis.

**Rendering idea:**

- Fragment coords `(u, v)` map to `(x, y)` in sample coordinates.
- Fixed `z_plane`.
- For each atom: accumulate `exp(-β |r - r_atom|)` or `exp(-β |r_xy - r_atom_xy|^2)`.
- Color scale the result (grayscale or false color).

**Usefulness:**

- Check that:

  - Geometry is loaded correctly.
  - Tip plane is at reasonable height above molecule.
  - Coordinate transformations are correct.

### Stage 2 – Add realistic potentials

**Objectives:**

- Implement **Morse + Coulomb** pairwise interaction (in GLSL) based on [ProbeParticle.cpp](cci:7://file:///home/prokophapala/git/tmp/ppafm/cpp/ProbeParticle.cpp:0:0-0:0) / [FF.cl](cci:7://file:///home/prokophapala/git/tmp/ppafm/cl/FF.cl:0:0-0:0).
- For each pixel:
  - Evaluate total force and/or energy at a fixed PP position (no relaxation yet).
- Expose:
  - Global or per-type Morse parameters.
  - Charge scaling factors.

**Simplifications:**

- Start with **global** Morse parameters for all atoms.
- Use a single effective charge model (e.g. charge per atom from `.xyz` or typed separately).

### Stage 3 – Hooke-law PP relaxation

**Objectives:**

- Implement approximate relaxation without an explicit time-stepping loop:
  - Solve a simple nonlinear equation with **few fixed-point iterations**:
    - `r_PP = r_PP0 + F_sample(r_PP) / K`.
- Implementation detail in fragment shader:

```glsl
vec3 rPP = rPP0;
for (int iter = 0; iter < N_ITER; iter++) {
    vec3 F_sample = computeSampleForce(rPP); // from Stage 2
    vec3 d  = F_sample / K;                  // isotropic spring for now
    rPP = rPP0 + d;
}
```*