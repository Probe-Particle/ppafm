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
  if (d > 0.0) { return vec3(1.-d, 1.-d, 1.0 );}
  else         { return vec3(1.,   1.+d, 1.+d);}
}

void main() {
  // Map vUv in [0,1]^2 to tip-plane coordinates [-L, L]^2
  float L = uL;
  float x = (vUv.x * 2.0 - 1.0) * L;
  float y = (vUv.y * 2.0 - 1.0) * L;
  float z = uZTip;
  vec3 tip = vec3(x, y, z);

  // If no sites, show neutral gray.
  if (uNSites <= 0) {
    gl_FragColor = vec4(vec3(0.2), 1.0);
    return;
  }

  // Compute on-site energy for each site and track min/max.
  float Rscale = max(uRtip, 1.0); // avoid division by zero
  float Ei_arr[4];
  float Emin   =  1e9;
  float Emax   = -1e9;
  for (int i = 0; i < 4; ++i) {
    if (i >= uNSites) {
      Ei_arr[i] = 0.0;
      continue;
    }
    vec3 spos = vec3(uSites[i].xyz);
    float E0  = uSites[i].w;
    float r   = length(tip - spos);
    // Simple gating: decays with distance from tip (heuristic)
    float Egate = uVBias * Rscale / max(r, 1e-3);
    float Ei    = E0 + Egate;
    Ei_arr[i]   = Ei;
    Emin = min(Emin, Ei);
    Emax = max(Emax, Ei);
  }

  // Select scalar to render based on mode.
  float Eplot = 0.0;

  if (uMode == 0) {
    // Mode 0: single-particle picture – minimum on-site energy over all sites.
    Eplot = Emin;
  } else {
    // Mode 1: many-body ground-state energy with uniform Coulomb interaction.
    // NOTE: GLSL ES 1.0 has no bitwise shift, so compute 2^nsites manually.
    int   maxStates = 1;
    for (int k = 0; k < 4; ++k) {
      if (k >= uNSites) break;
      maxStates *= 2;
    }
    float Eg_min    =  1e9;

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
        if (i >= uNSites) break;
        float occ = floor(mod(float(s) / w2[i], 2.0)); // 0 or 1
        N   += occ;
        Esp += occ * Ei_arr[i];
      }

      // Coulomb interaction: W * N_pairs, N_pairs = N*(N-1)/2.
      float Npairs = 0.5 * N * (N - 1.0);
      float Ecoul  = uW * Npairs;
      float Es     = Esp + Ecoul;

      Eg_min = min(Eg_min, Es);
    }

    Eplot = Eg_min;
  }

  // Map selected energy to diverging red-white-blue colormap controlled by uEcenter and uEscale.
  vec3 col = colormap_rwb(Eplot, uEcenter, uEscale);

  gl_FragColor = vec4(col, 1.0);
}
