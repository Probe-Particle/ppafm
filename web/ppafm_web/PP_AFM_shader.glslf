// Fragment shader for Stage 1: render atoms as radial blobs to verify geometry / z-plane
#ifdef GL_ES
precision highp float;
#endif

uniform vec2  uResolution;   // viewport size in pixels
uniform float uZPlane;       // imaging plane z
uniform float uScale;        // lateral scale (Å per screen unit)
uniform vec2  uCenter;       // center of the view in sample coordinates

uniform int   uNumAtoms;
uniform vec4  uAtoms[256];   // xyz in Å, w = charge (e)

// Per-atom Morse+Coulomb parameters: (R0, E0, Q, K)
// Potential:  E(r) = E0 * (1.0 - exp(-K*(r-R0)))^2 + Q * COULOMB_CONST / r
uniform vec4  uREQK[256];

// Linear contrast scaling for visualization of scalar fields (used for Fz)
uniform float uContrast;

// Lateral stiffness of the probe particle (Hooke law, in eV/Å^2)
uniform float uKLat;
// Radial stiffness of the probe bond (in eV/Å^2) and equilibrium bond length R_tip (Å)
uniform float uKRad;
uniform float uRtip;
// Maximum number of relaxation iterations per pixel
uniform int   uRelaxIters;
// Relaxation parameters (time step and damping) for simple velocity-based dynamics
// v = (1-damp)*v + f*dt; pos += v*dt;
uniform float uDt;
uniform float uDamp;
uniform float uF2Conv;
uniform int   uAlgo;        // 0 = basic Euler, 1 = zero-velocity quench
uniform int   uRenderMode;  // 0 = df, 1 = Fz, 2 = residual |F|, 3 = iter count
uniform int   uOscSteps;    // number of approach/oscillation steps (<=128)
uniform float uDz;          // step size for adiabatic approach
uniform float uOscAmp;      // oscillation amplitude (peak offset)
uniform int   uPreRelax;    // iterations for far-field pre-relax
uniform int   uRelaxSub;    // sub-iterations per step
uniform float uWeights[32];// Giessibl weights (top -> bottom)

varying vec2 vUv;

// Numerical safety for r^2
const float R2SAFE = 1e-4;
// Coulomb constant in eV·Å/e^2 (same as COULOMB_CONST in FF.cl)
const float COULOMB_CONST = 14.399644;
// Convergence threshold for |F|^2 during relaxation
const float F2CONV = 1e-6;

// --- Basic building blocks (mirroring FF.cl) ---

// Coulomb: returns (Fx,Fy,Fz,E)
vec4 getCoulomb(vec4 atom, vec3 pos) {
    vec3 dp  = pos - atom.xyz;
    float r2 = dot(dp, dp) + R2SAFE;
    float ir2 = 1.0 / r2;
    float ir  = sqrt(ir2);
    float E   = atom.w * COULOMB_CONST * ir;          // atom.w = charge in e
    vec3 F    = dp * (E * ir2);                       // F = grad(E)
    return vec4(F, E);
}

// Combined Morse + Coulomb force for a single atom using REQK = (R0, E0, Q, K)
// Returns force vector F = -dE/dr (in eV/Å) in world coordinates.
vec3 getMorseCoulombForce(vec3 pos, vec3 apos, vec4 REQK) {
    float R0 = REQK.x;
    float E0 = REQK.y;
    float Q  = REQK.z;
    float K  = REQK.w;

    vec3  dp = pos - apos;
    float r2 = dot(dp, dp) + R2SAFE;
    float r  = sqrt(r2);

    float expar = exp(-K * (r - R0));
    // dEm/dr for Em = E0 * (1 - exp(-K (r-R0)))^2
    float dEm_dr = 2.0 * E0 * K * expar * (1.0 - expar);
    // Coulomb: Ec = Q * COULOMB_CONST / r  => dEc/dr = -Q*C / r^2
    float dEc_dr = -Q * COULOMB_CONST / r2;
    float dE_dr  = dEm_dr + dEc_dr;

    // Force is negative gradient: F = -dE/dr * (dp / r)
    vec3 F = -dE_dr * (dp / r);
    return F;
}

// Convert HSV color (h in [0,1], s in [0,1], v in [0,1]) to RGB
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

// Lennard-Jones: cLJ.x = C6, cLJ.y = C12 (same as float2 in FF.cl)
// returns (Fx,Fy,Fz,E)
vec4 getLJ(vec3 apos, vec2 cLJ, vec3 pos) {
    vec3 dp  = pos - apos;
    float r2 = dot(dp, dp) + R2SAFE;
    float ir2 = 1.0 / r2;
    float ir6 = ir2 * ir2 * ir2;
    float ELJ = (cLJ.y * ir6 - cLJ.x) * ir6;
    vec3 FLJ = ((12.0 * cLJ.y * ir6 - 6.0 * cLJ.x) * ir6 * ir2) * dp;
    return vec4(FLJ, ELJ);
}

// Morse potential: REA.x = R0, REA.y = E0, REA.z = beta
// dp = pos - apos; returns (Fx,Fy,Fz,E)
vec4 getMorse(vec3 dp, vec3 REA) {
    float r    = sqrt(dot(dp, dp) + R2SAFE);
    float expar = exp(REA.z * (r - REA.x));
    float E     = REA.y * expar * (expar - 2.0);
    float fr    = REA.y * expar * (expar - 1.0) * 2.0 * REA.z;
    vec3 F      = dp * (fr / r);
    return vec4(F, E);
}

// Combined Lennard-Jones + Coulomb for a single atom
// atom.w = charge (e), cLJ = LJ parameters
vec4 getLJC(vec4 atom, vec2 cLJ, vec3 pos) {
    vec4 lj  = getLJ(atom.xyz, cLJ, pos);
    vec4 col = getCoulomb(atom, pos);
    return vec4(lj.xyz + col.xyz, lj.w + col.w);
}

// Combined Morse + Coulomb energy for a single atom using REQK = (R0, E0, Q, K)
float getMorseCoulomb(vec3 pos, vec3 apos, vec4 REQK) {
    float R0 = REQK.x;
    float E0 = REQK.y;
    float Q  = REQK.z;
    float K  = REQK.w;

    vec3  dp = pos - apos;
    float r2 = dot(dp, dp) + R2SAFE;
    float r  = sqrt(r2);

    float expar = exp(-K * (r - R0));
    float Em    = E0 * (1.0 - expar) * (1.0 - expar);
    float Ec    = (r > 0.0) ? Q * COULOMB_CONST / r : 0.0;
    return Em + Ec;
}

// Sum of sample forces from all atoms at a given probe position
vec3 computeSampleForce(vec3 pos) {
    vec3 F = vec3(0.0);
    for (int i = 0; i < 256; i++) {
        if (i >= uNumAtoms) break;
        F += getMorseCoulombForce(pos, uAtoms[i].xyz, uREQK[i]);
    }
    return F;
}

// Tip spring force: lateral (bending) + radial (bond-length) components
// pos  - current probe position
// pos0 - anchoring point (equilibrium lateral position, center of radial sphere)
vec3 computeTipForce(vec3 dpos) {
    vec3 FtipLat = vec3(dpos.xy * -uKLat, 0.0);
    // Radial spring along bond direction to keep |dpos| ~ uRtip
    float l      = length(dpos);
    vec3  FtipRad  = dpos * ( -uKRad * (l-uRtip) / l);
    return FtipLat + FtipRad;
}


// --- MAIN KERNEL: Morse+Coulomb via REQK ---

void main() {
    // Map screen coordinates to sample coordinates
    vec2 uv    = (gl_FragCoord.xy / uResolution) * 2.0 - 1.0;
    vec2 posXY = uCenter + uv * uScale;

    // --- Adiabatic approach with preconditioned relaxation ---
    float zStart = uZPlane + uOscAmp;
    vec3 anchor  = vec3(posXY, zStart);
    vec3 pos     = anchor;
    pos.z       -= uRtip;   // equilibrium bond offset

    float kx = max(uKLat, 1e-6);
    float kz = max(uKRad, 1e-6);
    vec3 invK = vec3(1.0/kx, 1.0/kx, 1.0/kz) * 0.5; // mild mixing

    // Pre-relax in far field
    int preIters = 0;
    for (int i = 0; i < 32; i++) {
        if (i >= uPreRelax) break;
        preIters = i;
        vec3 Fsamp = computeSampleForce(pos);
        vec3 Ftip  = computeTipForce(pos - anchor);
        vec3 Ftot  = Fsamp + Ftip;
        if (dot(Ftot, Ftot) < uF2Conv) break;
        pos += Ftot * invK;
    }

    float dfAccum = 0.0;
    float fzFinal = 0.0;
    float resF2   = 0.0;
    int   iters   = 0;

    for (int step = 0; step < 32; step++) {
        if (step >= uOscSteps) break;

        vec3 Ftot = vec3(0.0);
        // Sub-iterations at this Z
        for (int sub = 0; sub < 16; sub++) {
            if (sub >= uRelaxSub) break;
            iters++;
            vec3 Fsamp = computeSampleForce(pos);
            vec3 Ftip  = computeTipForce(pos - anchor);
            Ftot = Fsamp + Ftip;
            resF2 = dot(Ftot, Ftot);
            if (resF2 < uF2Conv) break;
            pos += Ftot * invK;
        }

        // Measure force after relaxation at this step
        vec3 FsampFinal = computeSampleForce(pos);
        fzFinal = FsampFinal.z;
        float w = (step < 32) ? uWeights[step] : 0.0;
        dfAccum += FsampFinal.z * w;

        // Rigid advance towards surface
        anchor.z -= uDz;
        pos.z    -= uDz;
    }

    vec3 rgb;
    if (uRenderMode == 0) {
        // df from Giessibl convolution (default)
        dfAccum *= -1.0; // seems to be inverted for some reason (perhaps the convolution mask ?, we go down in z )
        rgb = vec3(0.5 + dfAccum * uContrast);
    } else if (uRenderMode == 1) {
        // relaxed vertical force
        rgb = vec3(0.5 + fzFinal * uContrast);
    } else if (uRenderMode == 2) {
        // residual force norm after last relaxation step
        float res = sqrt(resF2);
        rgb = vec3(0.5 + res * uContrast);
    } else {
        // relative iteration count
        float denom = max(float(uOscSteps * uRelaxSub + uPreRelax), 1.0);
        float t = float(iters) / denom;
        rgb = vec3(t);
    }
    // // Map |F| to saturation with a soft roll-off
    // float s = clamp(fmag * 0.5, 0.0, 1.0);
    // float v = 0.5;
    // //vec3 rgb = hsv2rgb(vec3(hue, s, v));

    gl_FragColor = vec4(rgb, 1.0);
}


// ---- OLD KERNEL ----
// void main() {
//     vec2 uv  = (gl_FragCoord.xy / uResolution) * 2.0 - 1.0;
//     vec2 posXY = uCenter + uv * uScale;
//     float density = 0.0;
//     float sigma   = 0.7;
//     float sigma2  = sigma * sigma;

//     for (int i = 0; i < 256; i++) {
//         if (i >= uNumAtoms) break;
//         vec2 dr = posXY - uAtoms[i].xy;        // 2D distance only
//         float r2 = dot(dr, dr);
//         density += exp(-r2 / (2.0 * sigma2));
//     }
//     // Stronger tone mapping
//     float val = 1.0 - exp(-density * 3.0);
//     gl_FragColor = vec4(vec3(val), 1.0);
// }

// ---- DEBUG KERNEL ----
// void main() {
//     vec2 uv  = (gl_FragCoord.xy / uResolution) * 2.0 - 1.0;
//     vec2 posXY = uCenter + uv * uScale;
//     if (uNumAtoms > 0) {
//         vec2 dr = posXY - uAtoms[0].xy;
//         float r2 = dot(dr, dr);
//         float sigma = 1.0;
//         // One bright blob around atom 0
//         float val = exp(-r2 / (2.0 * sigma * sigma));
//         gl_FragColor = vec4(vec3(val), 1.0);
//     } else {
//         //gl_FragColor = vec4(0.0,0.5,0.0,1.0);
//         gl_FragColor = vec4( posXY,0.0,1.0);
//     }
// }