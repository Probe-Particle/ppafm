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
    float z    = uZPlane;

    // Dynamic relaxation in XY: damped velocity + lateral spring to original posXY
    vec3 pAnchor = vec3(posXY, z);
    vec3 pos     = pAnchor; 
    pos.z       -= uRtip;   // anchor is at some height above the sample
    vec3 vel  = vec3(0.0);
    vec3 Fsamp;
    float dt =uDt;
    //float damp = clamp(uDamp, 0.0, 0.999);
    for (int iter = 0; iter < 16; iter++) { // hard cap for safety
        if (iter >= uRelaxIters) break;
        Fsamp = computeSampleForce(pos);
        vec3 Ftip = computeTipForce( pos-pAnchor );
        vec3 Ftot  = Fsamp + Ftip;
        //if (dot(Ftot, Ftot) < F2CONV) break;
        //float cVF = dot(Ftot, vel); if(cVF < 0.0){ vel = vec3(0.0); }
        vel += Ftot * dt;
        pos += vel  * dt;
    }
    //Fsamp = computeSampleForce(pos);
    //pos += Fsamp / uKLat;  // Hook law

    Fsamp = computeSampleForce(pos);
    vec3 rgb = vec3( (0.5+Fsamp.z*uContrast));

    // // Use the final force F (at relaxed position) for visualization
    // float fmag = length(F.xy);
    // float angle = atan(F.y, F.x); // [-pi,pi]
    // float hue   = (angle / (2.0 * 3.14159265)) + 0.5; // map to [0,1]
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