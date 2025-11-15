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

// Linear contrast scaling for visualization: val = clamp(0.5 + E * uContrast, 0, 1)
uniform float uContrast;

varying vec2 vUv;

// Numerical safety for r^2
const float R2SAFE = 1e-4;
// Coulomb constant in eV·Å/e^2 (same as COULOMB_CONST in FF.cl)
const float COULOMB_CONST = 14.399644;

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

// Combined Morse + Coulomb for a single atom using REQK = (R0, E0, Q, K)
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


// --- MAIN KERNEL: Morse+Coulomb via REQK ---

void main() {
    // Map screen coordinates to sample coordinates
    vec2 uv    = (gl_FragCoord.xy / uResolution) * 2.0 - 1.0;
    vec2 posXY = uCenter + uv * uScale;
    float z    = uZPlane;
    vec3 pos   = vec3(posXY, z);

    float E = 0.0;
    for (int i = 0; i < 256; i++) {
        if (i >= uNumAtoms) break;
        E += getMorseCoulomb(pos, uAtoms[i].xyz, uREQK[i]);
    }

    // Pure linear grayscale mapping around E = 0, user-controlled contrast
    float val = clamp(0.5 + E * uContrast, 0.0, 1.0);
    gl_FragColor = vec4(vec3(val), 1.0);
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