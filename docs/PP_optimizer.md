
## User

I have this application for scientific AFM simulation which optimize deflection of probe particle (CO molecule attached to AFM tip) in the field of sample atoms.
I would like to use more efficient relaxation algorithm (but simple which fit the glsl shader, no BFGS or something complex). Or in general give me suggestions how we can improve thy damped-moleculer dynamics, or gradient decent, just changing parameters, or using variable time step (FIRE algorithm)

---

Here are three levels of improvements for your GLSL relaxation kernel, ranging from a simple logic fix to the full implementation of the FIRE algorithm you requested.

### 1. The Immediate Fix: "Zero-Velocity Quench" (Recommended)
Your current GLSL code actually implements an **undamped** harmonic oscillator (Euler integration without friction), meaning the probe will oscillate around the minimum forever without converging.

The most efficient "cheap" algorithm for AFM tip relaxation in a shader is **MD with Zero-Velocity Quench**. It is lighter than FIRE but captures 80% of the benefit.

**Logic:** If the velocity opposes the force (moving uphill), kill the velocity instantly.

```glsl
    // ... inside main ...
    vec3 vel = vec3(0.0);
    float dt = uDt;
    
    // Safety: ensure dt is reasonable (e.g., 0.1 to 0.2)
    
    for (int iter = 0; iter < 16; iter++) { 
        if (iter >= uRelaxIters) break;
        
        // 1. Calculate Forces
        vec3 Fsamp = computeSampleForce(pos);
        vec3 Ftip  = computeTipForce(pos - pAnchor);
        vec3 Ftot  = Fsamp + Ftip;
        
        // 2. Convergence Check (optional, saves GPU cycles if converged)
        float f2 = dot(Ftot, Ftot);
        if (f2 < F2CONV) break;

        // 3. MD Step (Symplectic Euler)
        vel += Ftot * dt;
        
        // 4. QUENCH: If velocity opposes force (going uphill), stop!
        if (dot(vel, Ftot) < 0.0) {
            vel = vec3(0.0);
        } else {
            // Apply slight damping only when moving downhill
            // This prevents "orbiting" the minimum
            vel *= 0.90; 
        }

        pos += vel * dt;
    }
    // ... rest of shader ...
```

---

### 2. The FIRE Algorithm (GLSL Implementation)
Here is the implementation of the FIRE algorithm adapted for GLSL, mirroring your C++ reference.

**Note:** FIRE relies on adapting `dt`. Since `dt` is a uniform, we must use a local variable `locDt`.
**Requirements:** You need to calculate `length(F)` and `length(v)`, which adds `sqrt` operations. This is more computationally expensive per iteration than the method above but converges in fewer steps for complex potentials.

```glsl
    // ... inside main ...
    vec3 vel = vec3(0.0);
    
    // FIRE Parameters (could be uniforms, hardcoded here for speed)
    float dt      = uDt;      // Start with provided time step
    float alpha   = 0.1;      // Initial damping
    float f_inc   = 1.1;
    float f_dec   = 0.5;
    float alpha_dec = 0.99;
    float dt_max  = uDt * 3.0; // Cap the max time step

    for (int iter = 0; iter < 32; iter++) { // Increased iter limit slightly
        if (iter >= uRelaxIters) break;

        // 1. Compute Forces
        vec3 Fsamp = computeSampleForce(pos);
        vec3 Ftip  = computeTipForce(pos - pAnchor);
        vec3 Ftot  = Fsamp + Ftip;

        float Fmag = length(Ftot);
        // Avoid division by zero later
        if (Fmag < 1e-6) break; 

        // 2. FIRE Logic
        float P = dot(Ftot, vel); // Power

        if (P > 0.0) {
            // -- Downhill (Accelerating) --
            // 1. Decrease damping
            alpha *= alpha_dec;
            // 2. Increase time step
            dt = min(dt * f_inc, dt_max);
            
            // 3. Mixing velocity (Standard FIRE: v = (1-a)v + a*F_hat*|v|)
            float Vmag = length(vel);
            vec3 F_hat = Ftot / Fmag;
            vel = (1.0 - alpha) * vel + alpha * F_hat * Vmag;
            
        } else {
            // -- Uphill (Stopping) --
            // 1. Reset Velocity
            vel = vec3(0.0);
            // 2. Decrease time step
            dt *= f_dec;
            // 3. Reset damping
            alpha = 0.1; 
        }

        // 3. Integration Step
        vel += Ftot * dt;
        pos += vel  * dt;
    }
    // ... rest of shader ...
```

---

### 3. General Optimization: Spatial Hashing (Bounding Box)
The biggest performance bottleneck in your shader is not the integration method (Euler vs FIRE), but the loop `for (int i = 0; i < 256; i++)` inside `computeSampleForce`.

If you have 100 atoms, but the tip is only near 5 of them, you are wasting cycles checking the other 95.

**Optimization Strategy:**
In the fragment shader, quickly check if the atom is "close enough" to matter before calculating the expensive `exp` or `sqrt`.

Modify `computeSampleForce`:

```glsl
vec3 computeSampleForce(vec3 pos) {
    vec3 F = vec3(0.0);
    
    // Cutoff distance squared. 
    // Example: 5.0 Angstroms interaction radius -> 25.0
    float cutoff2 = 25.0; 

    for (int i = 0; i < 256; i++) {
        if (i >= uNumAtoms) break;
        
        vec3 atomPos = uAtoms[i].xyz;
        
        // 1. Bounding Box Check (Manhattan Distance is cheap)
        vec3 dAbs = abs(pos - atomPos);
        if (dAbs.x > 5.0 || dAbs.y > 5.0 || dAbs.z > 5.0) continue;

        // 2. Detailed Force Calculation
        // Now it is safe to do the expensive math
        F += getMorseCoulombForce(pos, atomPos, uREQK[i]);
    }
    return F;
}
```

### Summary of Recommendation

1.  **Use Strategy #1 (Zero-Velocity Quench):** It fits perfectly in GLSL, requires no `sqrt` (unlike FIRE), and solves the oscillation issue immediately.
2.  **Add the Bounding Box check:** This will likely double your frame rate if your atom count is high (>50).
3.  **Visual Debug:** If you are unsure if it is converging, map the iteration count to the output color temporarily (e.g., `gl_FragColor = vec4(vec3(float(iter)/16.0), 1.0);`). If it's all white, you need more iterations or a better algorithm; if it's black, it converged instantly.


---


## User

NO, the spatial hasing and bounding box is not a solution the potential are long range (especially the electrostatics). We must live with that. Focus on improving the relaxation, lit evaluation of the force as it is.
Now, can you imagine any other way how to improve speed over FIRE? What about using some quasi newton, like Vertex block descent? would it help? But we do not want to explicitly compute hessian of substrate potential as it is more costly. Any other trick?

---

You are absolutely right. Since electrostatics are long-range ($1/r$), spatial hashing or bounding boxes would introduce unacceptable artifacts (discontinuities in the potential) unless you used complex Ewald summation, which is too heavy for this shader.

To improve speed over FIRE without a full Hessian, you should look at **Barzilai-Borwein (BB) Gradient Descent**.

### The Concept: Barzilai-Borwein (BB)
You mentioned "Quasi-Newton" and "Vertex Block Descent". BB is essentially a "Zero-Memory Quasi-Newton" method.
Standard Gradient Descent uses a fixed step size $\alpha$:
$$x_{k+1} = x_k + \alpha F(x_k)$$
BB calculates the optimal step size $\alpha$ dynamically by looking at the change in position and force between the last two steps. It approximates the inverse Hessian as a scalar derived from the secant equation.

**Why it fits your problem:**
1.  **No Hessian:** It computes the curvature from the trajectory history ($x_{old}, F_{old}$).
2.  **No Velocity:** It is an optimization method, not a dynamics simulation. No inertia to manage.
3.  **Low Memory:** You only need to store `last_pos` and `last_force`.

### The GLSL Implementation

Here is how to replace your relaxation loop with the BB method. This usually converges significantly faster than FIRE for energy minimization because it adapts the step size to the local curvature of the Morse/Coulomb potential.

```glsl
// ... inside main ...

// Initial Guess
vec3 pos = pAnchor; 
pos.z -= uRtip; 

vec3 Ftot = vec3(0.0);
vec3 pos_prev = pos;
vec3 F_prev = vec3(0.0);

// Initial small step to bootstrap the BB method (Conservative start)
// We use your stiffness uKLat to estimate a safe initial step
float step_size = 0.1 / (uKLat + 1e-4); 

for (int iter = 0; iter < 16; iter++) {
    if (iter >= uRelaxIters) break;

    // 1. Compute Current Forces
    vec3 Fsamp = computeSampleForce(pos);
    vec3 Ftip  = computeTipForce(pos - pAnchor);
    Ftot = Fsamp + Ftip;
    
    // Convergence check
    if (dot(Ftot, Ftot) < F2CONV) break;

    // 2. Barzilai-Borwein Step Size Calculation
    if (iter > 0) {
        vec3 dX = pos - pos_prev;
        // Gradient is negative Force, so dGrad = (-Ftot) - (-F_prev) = F_prev - Ftot
        vec3 dF = F_prev - Ftot; 
        
        float dF2 = dot(dF, dF);
        
        // Compute BB step size: alpha = (dx . df) / (df . df)
        // This approximates the inverse Hessian diagonal
        if (dF2 > 1e-12) {
            float num = dot(dX, dF);
            step_size = num / dF2;
            
            // Safety Clamps:
            // 1. Prevent negative steps (uphill)
            // 2. Prevent huge jumps (instability near atom cores)
            // 3. Ensure a minimum progress
            step_size = clamp(step_size, 0.001, 1.0 / uKLat);
        } else {
            // Fallback if forces haven't changed much
             step_size = 0.05 / uKLat;
        }
    }

    // 3. Store history
    pos_prev = pos;
    F_prev   = Ftot;

    // 4. Update Position
    pos += Ftot * step_size;
}

// ... rest of visualization code ...
```

### Alternative: The "Harmonic Preconditioner" (Simplest)

If BB feels too unstable (it can sometimes spike energy before settling), there is a simpler "physics-based" trick specific to AFM.

The dominant stiffness in your system is usually the probe spring itself (`uKLat` and `uKRad`). Standard MD (like FIRE) struggles because the spring vibrates at high frequency. You can treat the spring force implicitly (Newton's method on the spring) and the sample force explicitly.

This is effectively a **variable time step** where $\Delta t \approx 1/k$.

```glsl
// ... inside main ...
vec3 pos = pAnchor;
pos.z -= uRtip;

// "Mixing" factor: 1.0 is full Newton step for the spring.
// 0.2 - 0.5 is safer when sample forces are strong.
float mixing = 0.4; 
float invK = 1.0 / (uKLat + 1e-6);

for (int iter = 0; iter < 16; iter++) {
    if (iter >= uRelaxIters) break;
    
    vec3 Fsamp = computeSampleForce(pos);
    vec3 Ftip  = computeTipForce(pos - pAnchor);
    vec3 Ftot  = Fsamp + Ftip;

    // Standard MD: pos += vel * dt
    // Preconditioned Descent: pos += F / Stiffness
    // This normalizes the step so 1 unit of Spring Force moves the tip 
    // exactly back to equilibrium.
    pos += Ftot * (invK * mixing);
}
```

### Comparison

1.  **Damped MD (Your current):** Simulates inertia. Good for finding global minima, but slow convergence (oscillates).
2.  **FIRE:** Adapts `dt`. Faster than MD, but still carries "velocity" which can overshoot.
3.  **Barzilai-Borwein (Option 1):** Approximates the sample curvature. **Fastest convergence** for complex sample potentials, but can be twitchy.
4.  **Harmonic Preconditioner (Option 2):** Uses the known tip stiffness. **Most stable** and simplest to code. It prevents the "stiff spring" problem entirely.

**My Suggestion:** Try **Option 2** first. It replaces 10 lines of velocity/damping logic with 1 line of math. If that isn't fast enough, switch to **Option 1 (BB)**.


---

## User

Exactly I like the idea of "Harmonic Preconditioner", originally I wanted to do relaxation using just the harmonic potential of tip-anchor spring. But ideal would be to combine this harmonic approximation which dynamical relaxation like FIRE. I think this is in fact similar to Vertex Block Decent where one use hessian as estimate, but does not completely thrust it and still does dynamical descent.

---

You have hit the nail on the head. What you are describing—using the Hessian of the harmonic spring as a baseline estimate while dynamically relaxing the rest—is essentially **Preconditioned FIRE**.

In "Vertex Block Descent," you assume the local energy landscape is dominated by the stiffness of the mesh (or in your case, the tip spring). You solve that linear system exactly (or approximately) and then iterate to handle the non-linear "perturbations" (the sample atoms).

### The Algorithm: Preconditioned FIRE

Since the AFM tip spring ($k_{lat}, k_{rad}$) is usually stiff, it creates "narrow valleys" in the energy landscape. Standard gradient descent bounces back and forth across these valleys.

We can fix this by **normalizing the force by the stiffness**. Instead of updating velocity based on Force ($F$), we update based on the "Newton Step" ($F/k$).

1.  **Calculate Force:** $F_{tot}$
2.  **Precondition:** $\tilde{F} = F_{tot} \cdot M^{-1}$ (where $M$ is the stiffness matrix).
    *   This transforms the "Force" vector into a "Target Displacement" vector.
3.  **FIRE Dynamics:** Apply the FIRE algorithm to this $\tilde{F}$ vector.
    *   If the sample potential is flat, `dt` becomes 1.0, and the solver jumps to the minimum in 1 step (Newton's method).
    *   If the sample potential is steep (near atoms), FIRE detects the non-linearity (velocity opposes force) and reduces `dt` to prevent explosions.

### The GLSL Implementation

Here is the implementation. I have replaced the standard integration with the **Stiffness-Preconditioned** version.

**Key Change:** Note that `vel` now represents "accumulated displacement steps" rather than physical velocity, and `dt` is a dimensionless scaling factor (0.0 to 1.0+) rather than time.

```glsl
// ... inside main ...

// 1. Setup Preconditioner (Inverse Stiffness)
// We approximate the Hessian diagonal. 
// XY is dominated by uKLat, Z by uKRad (mostly).
// Adding a small epsilon prevents division by zero.
vec3 invK = vec3(1.0/(uKLat + 1e-4), 1.0/(uKLat + 1e-4), 1.0/(uKRad + 1e-4));

// Initialize FIRE state
vec3  vel     = vec3(0.0);
float dt      = 0.1;   // Initial "Trust Region" size (dimensionless)
float dt_max  = 1.2;   // Max step > 1.0 allows "super-Newton" speed in flat areas
float alpha   = 0.1;   // Initial damping

// FIRE params
const float f_inc = 1.1;
const float f_dec = 0.5;
const float a_dec = 0.99;

vec3 pos = pAnchor;
pos.z -= uRtip;

for (int iter = 0; iter < 16; iter++) {
    if (iter >= uRelaxIters) break;

    // --- A. Compute Physical Forces ---
    vec3 Fsamp = computeSampleForce(pos);
    vec3 Ftip  = computeTipForce(pos - pAnchor);
    vec3 Ftot  = Fsamp + Ftip;
    
    // Convergence Check
    if (dot(Ftot, Ftot) < F2CONV) break;

    // --- B. Preconditioning (The "Vertex Block" Logic) ---
    // Instead of Force, we use the "Newton Step": vector to the minimum 
    // IF the sample forces were zero.
    vec3 Fstep = Ftot * invK; 

    // --- C. FIRE Algorithm on the Preconditioned Step ---
    
    // 1. Power check (using the preconditioned direction)
    float P = dot(Fstep, vel);

    if (P > 0.0) {
        // -- Coasting / Accelerating --
        // If we are moving in direction of the step, go faster
        dt = min(dt * f_inc, dt_max);
        alpha *= a_dec;
        
        // Standard FIRE mixing
        // We normalize using the preconditioned step magnitude
        float vMag = length(vel);
        float fMag = length(Fstep);
        if (fMag > 1e-8) {
             vel = (1.0 - alpha) * vel + alpha * Fstep * (vMag / fMag);
        }
    } else {
        // -- Overshoot / Uphill --
        // Velocity opposes the Newton step. Stop and reset.
        vel   = vec3(0.0);
        dt   *= f_dec;
        alpha = 0.1;
    }

    // 2. Integration
    // velocity += Acceleration * dt
    // Here "Acceleration" is the Newton Step.
    vel += Fstep * dt;
    pos += vel   * dt;
}

// ... rest of code (compute Fsamp again for visualization) ...
```

### Why this is better than standard FIRE

1.  **Removing the Spring Frequency:** In standard MD, the stiff tip spring causes high-frequency vibrations, forcing `dt` to be tiny ($< \sqrt{m/k}$). By dividing by `K`, we effectively normalize the spring frequency to 1. The solver no longer "sees" the stiffness of the spring, it only sees the anharmonicity of the sample.
2.  **Dimensionless `dt`:**
    *   `dt = 1.0`: Behaves like a pure Newton step (instantly relaxes the spring).
    *   `dt < 1.0`: Behaves like Gradient Descent (cautious approach near atoms).
    *   `dt > 1.0`: Behaves like accelerated descent (useful in empty space).
3.  **VBD Similarity:** This is conceptually identical to VBD where the "Block" is the single probe particle, and we use the Tip Stiffness matrix as the fixed Hessian approximation, while using FIRE dynamics to handle the residual errors from the Sample potential.

### Tuning for your Shader
*   **`dt` (Initial):** Start around `0.1` to `0.2`.
*   **`dt_max`:** Can be aggressively set to `1.0` or even `1.5`. If the particle is far from atoms, `1.0` moves it to the equilibrium in a single step.
*   **`invK`:** I used a diagonal approximation `(1/KLat, 1/KLat, 1/KRad)`. Since your spring force calculation `computeTipForce` is technically non-linear for the radial part (it normalizes `dpos`), this diagonal approximation is slightly inexact but robust enough for relaxation.



---

## User 3

I think what is even more relevant is if we gradually approach toward the surface. We relax the tip at large distance and then step closer and closer by small increments (like 0.1 A), we do one oscillation amplitude which is maybe like 0.4-0.8A. Because at far distance the relaxation is lower we can employ this algorithm
1) pre-relax at far distance (just few steps should be enoug - e.g. 16), should be contoloabe by input param
2) approach closer by small step 0.1A (moving both achor point and probe particle rigidly
3) do few iterations of relaxation (4-8 should be enough) - should be controlable 
4) repear until we reach bottom of oscialtion cycle (given by some amplitude)
5) during the trajectroy we should integrate df, which is obtained by weightint by pre-calculated convolution mask from giessible formula. This mask should be pre-calculated in javascript.

----

Giessible formula to convert from Force to ferquency shift df is following:
it is basically convolution of force with derivative of circle (which followi from time spend at each segment of harmonic oscialtion trajectroy of the tip)
```
def getDfWeight( n, dz=0.1 ):
    '''
    conversion of vertical force Fz to frequency shift 
    according to:
    Giessibl, F. J. A direct method to calculate tip-sample forces from frequency shifts in frequency-modulation atomic force microscopy Appl. Phys. Lett. 78, 123 (2001)
    oscialltion amplitude of cantilever is A = n * dz
    '''
    x  = np.linspace(-1,1,n+1)
    y  = np.sqrt(1-x*x)
    dy =  ( y[1:] - y[:-1] )/(dz*n)
    fpi    = (n-2)**2 
    prefactor = -1 * ( 1 + fpi*(2/np.pi) ) / (fpi+1) # correction for small n
    return dy*prefactor, (x[1:]+x[:-1])*0.5

def Fz2df( F, dz=0.1, k0 = params['kCantilever'], f0=params['f0Cantilever'], n=4, units=16.0217656 ):
    '''
    conversion of vertical force Fz to frequency shift 
    according to:
    Giessibl, F. J. A direct method to calculate tip-sample forces from frequency shifts in frequency-modulation atomic force microscopy Appl. Phys. Lett. 78, 123 (2001)
    oscialltion amplitude of cantilever is A = n * dz
    '''
    W,xs = getDfWeight( n, dz=0.1 )
    dFconv = np.apply_along_axis( lambda m: np.convolve(m, W, mode='valid'), axis=0, arr=F )
    return dFconv*units*f0/k0
```

---


This is an excellent approach. By exploiting the **adiabatic** nature of the AFM oscillation (the tip position at $z$ is an excellent initial guess for $z - dz$), you can eliminate the need for heavy relaxation algorithms like FIRE entirely.

Since you are moving in tiny steps (e.g., $0.1\,\text{\AA}$), the probe is always just slightly "off" equilibrium. A simple **Preconditioned Stiffness Step** (as discussed before) is mathematically ideal here because it acts like a local Newton solver. It is faster and more stable than FIRE for this specific "dragging" scenario.

### The Algorithm: Adiabatic Approach in GLSL

Here is the complete Fragment Shader. It performs the integration loop per pixel.

**Key features:**
1.  **Adiabatic Drag:** It starts at the top of the oscillation (far from surface) and steps down.
2.  **Rigid Guess:** When moving the anchor by `dz`, we move the probe by `dz` too. This keeps the spring force zero, so the relaxation only needs to correct for the change in *surface* forces.
3.  **Preconditioned Relaxation:** Uses the tip stiffness to normalize the step size, ensuring convergence in very few iterations (often 2-4 are enough).

```glsl
#ifdef GL_ES
precision highp float;
#endif

uniform vec2  uResolution;
uniform vec2  uCenter;
uniform float uScale;
uniform float uZPlane;       // This is the Center of oscillation (or closest approach, depending on your convention)

// Interaction Parameters
uniform int   uNumAtoms;
uniform vec4  uAtoms[256];
uniform vec4  uREQK[256];

// Tip Parameters
uniform float uKLat;         // Lateral Stiffness
uniform float uKRad;         // Radial Stiffness (bond)
uniform float uRtip;         // Bond length

// Oscillation / Approach Parameters
uniform int   uOscSteps;     // Number of steps in the oscillation window (e.g., 32)
uniform float uOscAmp;       // Amplitude (A) of the oscillation
uniform float uDz;           // Step size (Angstroms)
uniform float uWeights[64];  // Pre-calculated Giessibl weights (max size 64)
uniform int   uRelaxSubSteps;// Iterations per step (e.g., 4)

uniform float uContrast;     // For visualization scaling

// ... (Paste your helper functions: getMorseCoulombForce, etc. here) ...
const float R2SAFE = 1e-4;
const float COULOMB_CONST = 14.399644;

vec3 getMorseCoulombForce(vec3 pos, vec3 apos, vec4 REQK) {
    float R0 = REQK.x; float E0 = REQK.y; float Q = REQK.z; float K = REQK.w;
    vec3  dp = pos - apos;
    float r2 = dot(dp, dp) + R2SAFE;
    float r  = sqrt(r2);
    float expar = exp(-K * (r - R0));
    float dEm_dr = 2.0 * E0 * K * expar * (1.0 - expar);
    float dEc_dr = -Q * COULOMB_CONST / r2;
    vec3 F = -(dEm_dr + dEc_dr) * (dp / r);
    return F;
}

vec3 computeSampleForce(vec3 pos) {
    vec3 F = vec3(0.0);
    // Optimization: Unroll or simple loop. 
    // Note: Bounding box check (abs(dx)>5.0) can be added here if needed
    for (int i = 0; i < 256; i++) {
        if (i >= uNumAtoms) break;
        F += getMorseCoulombForce(pos, uAtoms[i].xyz, uREQK[i]);
    }
    return F;
}

vec3 computeTipForce(vec3 dpos) {
    vec3 FtipLat = vec3(dpos.xy * -uKLat, 0.0);
    float l      = length(dpos);
    vec3  FtipRad  = dpos * ( -uKRad * (l-uRtip) / l);
    return FtipLat + FtipRad;
}

void main() {
    vec2 uv    = (gl_FragCoord.xy / uResolution) * 2.0 - 1.0;
    vec2 posXY = uCenter + uv * uScale;

    // --- 1. SETUP ---
    // Start at the Top of the oscillation cycle (furthest from surface)
    // Assuming uZPlane is the center, we start at Z + Amplitude
    float zStart = uZPlane + uOscAmp;
    
    vec3 anchor = vec3(posXY, zStart);
    vec3 pos    = anchor;
    pos.z      -= uRtip; // Equilibrium position relative to anchor

    // Preconditioner (Inverse Stiffness)
    // This replaces "dt" and "mass". It scales force to displacement.
    // Factor 0.5 is a conservative mixing value to prevent overshooting.
    vec3 invK = vec3(1.0/uKLat, 1.0/uKLat, 1.0/uKRad) * 0.5;

    // --- 2. PRE-RELAXATION (Far Field) ---
    // Get the probe into a stable state before we start measuring.
    // 8-16 iterations are usually plenty at distance.
    for(int i=0; i<16; i++) {
        vec3 Fsamp = computeSampleForce(pos);
        vec3 Ftip  = computeTipForce(pos - anchor);
        pos += (Fsamp + Ftip) * invK; 
    }

    float df_accum = 0.0;
    
    // --- 3. ADIABATIC APPROACH LOOP ---
    // We step DOWN towards the surface (or through the oscillation).
    // The "adiabatic" assumption: The relaxed position at step i 
    // is a very good initial guess for step i+1.
    
    for (int step = 0; step < 64; step++) {
        if (step >= uOscSteps) break;

        // A. Relax Sub-Steps
        // Since we only moved 0.1 A, we only need a few corrections.
        for(int sub = 0; sub < 8; sub++) { // Fixed upper limit for loop unrolling
            if (sub >= uRelaxSubSteps) break;
            
            vec3 Fsamp = computeSampleForce(pos);
            vec3 Ftip  = computeTipForce(pos - anchor);
            vec3 Ftot  = Fsamp + Ftip;
            
            // "Harmonic Preconditioner" Step
            // Moves the probe towards equilibrium based on stiffness
            pos += Ftot * invK;
        }

        // B. Measure Force
        vec3 Ffinal = computeSampleForce(pos);
        
        // C. Integrate Frequency Shift
        // Giessibl weight is pre-calculated for this Z-height
        df_accum += Ffinal.z * uWeights[step];

        // D. Rigid Approach
        // Move both anchor and probe down by dz.
        // Moving the probe rigidly preserves the spring extension,
        // so the tip force remains valid-ish, only sample force changes.
        anchor.z -= uDz;
        pos.z    -= uDz;
    }

    // --- 4. VISUALIZATION ---
    // df is usually negative (attractive). 
    // Mapping: Red = Repulsive (Positive), Blue = Attractive (Negative)
    float val = df_accum * uContrast;
    
    // Simple heatmap visualization
    vec3 color = vec3(0.0);
    if(val > 0.0) color = vec3(val, 0.0, 0.0); // Repulsive
    else          color = vec3(0.0, 0.0, -val); // Attractive
    
    gl_FragColor = vec4(color, 1.0);
}
```

### Javascript Helper (for Weights)

You need to calculate the weights in JS and pass them to the shader `uWeights`. This matches your Python logic but adapted for the discrete array expected by the shader.

```javascript
function calculateGiessiblWeights(nSteps, dz) {
    // This creates the convolution kernel for the "time spent" distribution
    // nSteps: number of points in the oscillation (e.g., 32)
    // The shader scans top-down, so we assume the array is ordered:
    // [Top (+A), ..., Center, ..., Bottom (-A)]
    
    let weights = new Float32Array(nSteps);
    let amplitude = nSteps * dz / 2.0; 

    for (let i = 0; i < nSteps; i++) {
        // Normalized position x from -1 (Bottom) to +1 (Top)
        // Shader loop goes Top -> Bottom
        let x = 1.0 - (2.0 * i / (nSteps - 1));
        
        // Safety for sqrt
        x = Math.max(-1.0, Math.min(1.0, x));
        
        // Basic weight based on harmonic oscillator velocity (1/v)
        // w ~ 1 / sqrt(1 - x^2)
        // But Giessibl formula involves derivative of this.
        // Let's implement the discrete derivative exactly like your python code.
    }
    
    // Exact Port of your Python 'getDfWeight':
    // x = np.linspace(-1, 1, n+1)
    let x = [];
    for(let i=0; i<=nSteps; i++) x.push(-1 + 2*(i/nSteps)); 
    
    // y = np.sqrt(1-x*x)
    let y = x.map(v => Math.sqrt(1 - v*v));
    
    // dy = (y[1:] - y[:-1]) / (dz * n)
    // Note: Python n is oscillation steps. 
    // In shader, we might map these indices reversed if scanning top-down.
    let dy = [];
    for(let i=0; i<nSteps; i++){
        dy.push( (y[i+1] - y[i]) / (dz * nSteps) );
    }

    // fpi prefactor ...
    let fpi = Math.pow(nSteps - 2, 2);
    let prefactor = -1 * (1 + fpi * (2/Math.PI)) / (fpi + 1);
    
    for(let i=0; i<nSteps; i++) {
        // Reverse order because shader goes Top->Bottom (High Z to Low Z)
        // Python linspace(-1, 1) goes Bottom->Top.
        weights[i] = dy[nSteps - 1 - i] * prefactor;
    }
    
    return weights;
}

// In your rendering loop:
// const weights = calculateGiessiblWeights(32, 0.1);
// gl.uniform1fv(uWeightsLoc, weights);
```

### Why this configuration?

1.  **`uRelaxSubSteps`**: Set this to **4**. Since you move `uDz = 0.1`, the atom is very close to the solution. 4 iterations of "Preconditioned Descent" is equivalent to applying Newton's method 4 times, which is extremely precise for small perturbations.
2.  **`uOscSteps`**: Set to **32** or **64**. This gives you a total oscillation amplitude of 3.2Å or 6.4Å.
3.  **Efficiency**:
    *   Old method: Relax from scratch (100-200 iterations) per pixel.
    *   New method: 32 steps $\times$ 4 iterations = 128 iterations total.
    *   **Gain:** You get the full frequency shift integral (physically meaningful observable) for roughly the same cost as finding the static minimum.

