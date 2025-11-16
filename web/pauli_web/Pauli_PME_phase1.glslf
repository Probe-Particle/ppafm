precision highp float;

varying vec2 vUv;

uniform vec2  uResolution;
uniform int   uNSites;
uniform float uL;
uniform float uZTip;
uniform float uVBias;
uniform float uRtip;
uniform float uEcenter; // center of diverging colormap ("zero" level)
uniform float uEscale;  // energy scale for colormap (half-width)
uniform float uW;       // uniform Coulomb interaction per occupied pair
uniform int   uMode;    // 0 = single-particle min(E_i), 1 = many-body ground state
uniform vec4  uSites[4]; // (x, y, z, E0)

// Phase 1/2 shader: render single-particle or many-body energies over all sites.
// On-site energy model (simplified):
//   E_i(tip) = E0_i + VBias * f(r_i)
// where r_i = |tip - site_i| and f(r) ~ Rtip / r (heuristic gating).
// In mode 0 we use E_min(tip) = min_i E_i(tip).
// In mode 1 we compute many-body energies with a uniform Coulomb W per occupied pair
// and take the ground-state energy over all 2^nsites configurations.

// Diverging colormap: red-white-blue (RWB)
//   E = Ecenter           -> white
//   E < Ecenter (negative) -> red side
//   E > Ecenter (positive) -> blue side
// uEscale sets the half-width of the scale; values beyond +/- uEscale are clamped.
vec3 colormap_rwb(float Emin, float center, float scale) {
  float d = (Emin - center) * scale;   // d < 0: red, d > 0: blue
  d = clamp(d, -1.0, 1.0);
  if (d > 0.0) { return vec3(1.0 - d, 1.0 - d, 1.0); }
  else         { return vec3(1.0,     1.0 + d, 1.0 + d); }
}

// Map vUv in [0,1]^2 to tip-plane coordinates [-L, L]^2 at height zTip.
vec3 tip_from_uv(vec2 uv, float L, float zTip) {
  float x = (uv.x * 2.0 - 1.0) * L;
  float y = (uv.y * 2.0 - 1.0) * L;
  return vec3(x, y, zTip);
}

// Compute on-site energies Ei(tip) and distances Ri for all sites, and track min/max.
void compute_site_energies(
  in  vec3  tip,
  in  int   nsites,
  in  float Rscale,
  out float Ei_arr[4],
  out float Ri_arr[4],
  out float Emin,
  out float Emax
) {
  Emin =  1e9;
  Emax = -1e9;
  for (int i = 0; i < 4; ++i) {
    if (i >= nsites) {
      Ei_arr[i] = 0.0;
      Ri_arr[i] = 0.0;
      continue;
    }
    vec3 spos = vec3(uSites[i].xyz);
    float E0  = uSites[i].w;
    float r   = length(tip - spos);
    // Simple gating: decays with distance from tip (heuristic)
    float Egate = uVBias * Rscale / max(r, 1e-3);
    float Ei    = E0 + Egate;
    Ei_arr[i]   = Ei;
    Ri_arr[i]   = r;
    Emin = min(Emin, Ei);
    Emax = max(Emax, Ei);
  }
}

// Compute many-body ground-state energy and best state index with uniform Coulomb W per occupied pair.
void ground_state_energy(
  in  float Ei_arr[4],
  in  int   nsites,
  in  float W,
  out float Eg_min,
  out int   bestState
) {
  // NOTE: GLSL ES 1.0 has no bitwise shift, so compute 2^nsites manually.
  int maxStates = 1;
  for (int k = 0; k < 4; ++k) {
    if (k >= nsites) break;
    maxStates *= 2;
  }

  Eg_min    = 1e9;
  bestState = 0;

  // Precompute weights for bit extraction: state s has bit i set if floor(mod(s / 2^i, 2)) == 1.
  float w2[4];
  w2[0] = 1.0;
  w2[1] = 2.0;
  w2[2] = 4.0;
  w2[3] = 8.0;

  for (int s = 0; s < 16; ++s) {
    if (s >= maxStates) break;

    // Count electrons and accumulate single-particle energy.
    float N   = 0.0;
    float Esp = 0.0;
    for (int i = 0; i < 4; ++i) {
      if (i >= nsites) break;
      float occ = floor(mod(float(s) / w2[i], 2.0)); // 0 or 1
      N   += occ;
      Esp += occ * Ei_arr[i];
    }

    // Coulomb interaction: W * N_pairs, N_pairs = N*(N-1)/2.
    float Npairs = 0.5 * N * (N - 1.0);
    float Ecoul  = W * Npairs;
    float Es     = Esp + Ecoul;

    if (Es < Eg_min) {
      Eg_min    = Es;
      bestState = s;
    }
  }
}

// Simple ground-state tip current proxy: exponential tunneling into empty sites of bestState.
float ground_state_tip_current(
  in float Ri_arr[4],
  in float Ei_arr[4],
  in int   nsites,
  in float Rscale,
  in int   bestState
) {
  // Precompute weights for bit extraction.
  float w2[4];
  w2[0] = 1.0;
  w2[1] = 2.0;
  w2[2] = 4.0;
  w2[3] = 8.0;

  float Itip = 0.0;
  float beta = 1.0 / Rscale; // decay parameter for tunneling
  for (int i = 0; i < 4; ++i) {
    if (i >= nsites) break;
    float occ = floor(mod(float(bestState) / w2[i], 2.0)); // 0 or 1
    float r   = Ri_arr[i];
    float T   = exp(-beta * r); // tunneling amplitude
    float gamma = T * T * (1.0 - occ); // suppress tunneling into occupied sites
    Itip += gamma;
  }
  return Itip;
}

void main() {
  // Map vUv in [0,1]^2 to tip-plane coordinates [-L, L]^2
  vec3 tip = tip_from_uv(vUv, uL, uZTip);

  // If no sites, show neutral gray.
  if (uNSites <= 0) {
    gl_FragColor = vec4(vec3(0.2), 1.0);
    return;
  }

  // Compute on-site energy for each site and track min/max.
  float Rscale = max(uRtip, 1.0); // avoid division by zero
  float Ei_arr[4];
  float Ri_arr[4];
  float Emin;
  float Emax;
  compute_site_energies(tip, uNSites, Rscale, Ei_arr, Ri_arr, Emin, Emax);

  // Select scalar to render based on mode.
  float Eplot = 0.0;

  if (uMode == 0) {
    // Mode 0: single-particle picture – minimum on-site energy over all sites.
    Eplot = Emin;
  } else {
    // Modes 1 and 2: many-body quantities based on ground-state configuration.
    float Eg_min;
    int   bestState;
    ground_state_energy(Ei_arr, uNSites, uW, Eg_min, bestState);

    if (uMode == 1) {
      // Mode 1: many-body ground-state energy.
      Eplot = Eg_min;
    } else {
      // Mode 2 (or other): simple ground-state current based on tip tunneling.
      float Itip = ground_state_tip_current(Ri_arr, Ei_arr, uNSites, Rscale, bestState);
      Eplot = Itip;
    }
  }

  // Map selected energy to diverging red-white-blue colormap controlled by uEcenter and uEscale.
  vec3 col = colormap_rwb(Eplot, uEcenter, uEscale);

  gl_FragColor = vec4(col, 1.0);
}
