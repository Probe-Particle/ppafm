precision highp float;
precision highp int;

in vec2 vUv;
out vec4 outColor;

uniform vec2  uResolution;
uniform int   uNSites;
uniform float uL;
uniform float uZTip;
uniform float uVBias;
uniform float uRtip;
uniform float uEcenter; // center of diverging colormap ("zero" level)
uniform float uEscale;  // energy scale for colormap (half-width)
uniform float uW;         // uniform Coulomb interaction per occupied pair
uniform int   uMode;      // 0 = single-particle, 1 = many-body GS, 2 = GS current, 3–5 = PME
uniform int   uSiteIndex; // for PME site occupancy visualization
uniform vec4  uSites[4];  // (x, y, z, E0)

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

// === PME solver helpers (modes 3–5) ===

// Compute many-body energies Es[s] and electron counts Ne[s] for all configurations s < 2^nsites.
void compute_many_body_energies(
  in  float Ei_arr[4],
  in  int   nsites,
  in  float W,
  out float Es[16],
  out float Ne[16],
  out int   nStates
) {
  int maxStates = 1;
  for (int k = 0; k < 4; ++k) {
    if (k >= nsites) break;
    maxStates *= 2;
  }
  nStates = maxStates;

  float w2[4];
  w2[0] = 1.0;
  w2[1] = 2.0;
  w2[2] = 4.0;
  w2[3] = 8.0;

  for (int s = 0; s < 16; ++s) {
    if (s >= maxStates) {
      Es[s] = 0.0;
      Ne[s] = 0.0;
      continue;
    }
    float N   = 0.0;
    float Esp = 0.0;
    for (int i = 0; i < 4; ++i) {
      if (i >= nsites) break;
      float occ = floor(mod(float(s) / w2[i], 2.0));
      N   += occ;
      Esp += occ * Ei_arr[i];
    }
    float Npairs = 0.5 * N * (N - 1.0);
    float Ecoul  = W * Npairs;
    Es[s] = Esp + Ecoul;
    Ne[s] = N;
  }
}

// Simple Fermi-Dirac function for energy difference dE at temperature kT.
float fermi(float dE, float kT) {
  float x = dE / max(kT, 1e-6);
  x = clamp(x, -40.0, 40.0);
  float ex = exp(-x);
  return 1.0 / (1.0 + ex);
}

// Solve K rho = rhs for rho via Gaussian elimination; n is number of states (<=16).
void solve_linear_system(
  inout float K[16*16],
  inout float rhs[16],
  int  n
) {
  for (int i = 0; i < 16; ++i) {
    if (i >= n) break;
    float pivot = K[i*16 + i];
    if (abs(pivot) < 1e-12) pivot = (pivot < 0.0 ? -1.0 : 1.0) * 1e-12;
    float invPivot = 1.0 / pivot;
    for (int j = 0; j < 16; ++j) {
      if (j >= n) break;
      K[i*16 + j] *= invPivot;
    }
    rhs[i] *= invPivot;

    for (int r = 0; r < 16; ++r) {
      if (r >= n || r == i) break;
      float factor = K[r*16 + i];
      if (abs(factor) == 0.0) continue;
      for (int c = 0; c < 16; ++c) {
        if (c >= n) break;
        K[r*16 + c] -= factor * K[i*16 + c];
      }
      rhs[r] -= factor * rhs[i];
    }
  }
}

// PME steady-state solver: fills rho_s for s < nStates and returns a simple tip current.
void solve_pme(
  in  float Ei_arr[4],
  in  float Ri_arr[4],
  in  int   nsites,
  in  float W,
  out float rho[16],
  out float I_tip
) {
  float Es[16];
  float Ne[16];
  int   nStates;
  compute_many_body_energies(Ei_arr, nsites, W, Es, Ne, nStates);

  float w2[4];
  w2[0] = 1.0;
  w2[1] = 2.0;
  w2[2] = 4.0;
  w2[3] = 8.0;

  float kT   = 0.01;      // heuristic temperature in eV
  float muS  = 0.0;       // substrate chemical potential
  float muT  = uVBias;    // tip chemical potential ~ bias
  float gammaS0 = 1.0;    // base coupling substrate
  float gammaT0 = 1.0;    // base coupling tip

  float K[16*16];
  float rhs[16];
  for (int i = 0; i < 16; ++i) {
    rhs[i] = 0.0;
    for (int j = 0; j < 16; ++j) {
      K[i*16 + j] = 0.0;
    }
  }

  // Build K from transition rates for both leads.
  for (int s = 0; s < 16; ++s) {
    if (s >= nStates) break;

    for (int i = 0; i < 4; ++i) {
      if (i >= nsites) break;
      float occ = floor(mod(float(s) / w2[i], 2.0));

      for (int l = 0; l < 2; ++l) {
        float mu     = (l == 0) ? muS : muT;
        float gamma0 = (l == 0) ? gammaS0 : gammaT0;
        float r      = Ri_arr[i];
        float Ttun   = exp(-2.0 * r / max(uRtip, 1e-3));
        float gamma  = gamma0 * Ttun;

        int t = s;
        float dN = 0.0;
        if (occ < 0.5) {
          // Electron tunnels in: s -> t with bit i set.
          t = s + int(w2[i]);
          if (t >= nStates) continue;
          dN = 1.0;
        } else {
          // Electron tunnels out: s -> t with bit i cleared.
          t = s - int(w2[i]);
          if (t < 0) continue;
          dN = -1.0;
        }

        float dE = Es[t] - Es[s] - mu * dN;
        float f  = fermi(dE, kT);

        float G_forward  = gamma * f;
        float G_backward = gamma * (1.0 - f);

        // Outflow from s due to s->t, inflow to t.
        K[s*16 + s] += G_forward;
        K[t*16 + s] -= G_forward;
        // Outflow from t due to t->s, inflow to s.
        K[t*16 + t] += G_backward;
        K[s*16 + t] -= G_backward;
      }
    }
  }

  // Replace last equation by normalization: sum_s rho_s = 1.
  int normRow = nStates - 1;
  for (int j = 0; j < 16; ++j) {
    if (j < nStates) {
      K[normRow*16 + j] = 1.0;
    } else {
      K[normRow*16 + j] = 0.0;
    }
  }
  for (int r = 0; r < nStates - 1; ++r) {
    K[r*16 + normRow] = 0.0;
  }
  for (int i = 0; i < nStates; ++i) {
    rhs[i] = 0.0;
  }
  rhs[normRow] = 1.0;

  solve_linear_system(K, rhs, nStates);

  for (int s = 0; s < 16; ++s) {
    rho[s] = (s < nStates) ? rhs[s] : 0.0;
  }

  // PME tip current: sum over transitions involving tip only.
  I_tip = 0.0;
  for (int s = 0; s < 16; ++s) {
    if (s >= nStates) break;

    for (int i = 0; i < 4; ++i) {
      if (i >= nsites) break;
      float occ = floor(mod(float(s) / w2[i], 2.0));
      float r   = Ri_arr[i];
      float Ttun = exp(-2.0 * r / max(uRtip, 1e-3));
      float gammaT = gammaT0 * Ttun;

      int t = s;
      float dN = 0.0;
      if (occ < 0.5) {
        t = s + int(w2[i]);
        if (t >= nStates) continue;
        dN = 1.0; // electron enters system from tip
      } else {
        t = s - int(w2[i]);
        if (t < 0) continue;
        dN = -1.0; // electron leaves system to tip
      }

      float dE = Es[t] - Es[s] - muT * dN;
      float f  = fermi(dE, kT);
      float G_forward  = gammaT * f;
      float G_backward = gammaT * (1.0 - f);

      float sign = (dN > 0.0) ? 1.0 : -1.0;
      float contrib = 0.0;
      contrib += rho[s] * sign * G_forward;
      if (t < nStates) {
        contrib += rho[t] * (-sign) * G_backward;
      }
      I_tip += contrib;
    }
  }
}

// PME observables
float pme_site_occupancy(float rho[16], int nsites, int siteIdx) {
  float w2[4];
  w2[0] = 1.0;
  w2[1] = 2.0;
  w2[2] = 4.0;
  w2[3] = 8.0;
  if (siteIdx < 0) siteIdx = 0;
  if (siteIdx >= nsites) siteIdx = nsites - 1;

  float occSite = 0.0;
  int nStates = 1;
  for (int k = 0; k < 4; ++k) {
    if (k >= nsites) break;
    nStates *= 2;
  }
  for (int s = 0; s < 16; ++s) {
    if (s >= nStates) break;
    float occ = floor(mod(float(s) / w2[siteIdx], 2.0));
    occSite += rho[s] * occ;
  }
  return occSite;
}

float pme_total_charge(float rho[16], int nsites) {
  float w2[4];
  w2[0] = 1.0;
  w2[1] = 2.0;
  w2[2] = 4.0;
  w2[3] = 8.0;
  float Q = 0.0;
  int nStates = 1;
  for (int k = 0; k < 4; ++k) {
    if (k >= nsites) break;
    nStates *= 2;
  }
  for (int s = 0; s < 16; ++s) {
    if (s >= nStates) break;
    float N = 0.0;
    for (int i = 0; i < 4; ++i) {
      if (i >= nsites) break;
      float occ = floor(mod(float(s) / w2[i], 2.0));
      N += occ;
    }
    Q += rho[s] * N;
  }
  return Q;
}

void main() {
  // Map vUv in [0,1]^2 to tip-plane coordinates [-L, L]^2
  vec3 tip = tip_from_uv(vUv, uL, uZTip);

  // If no sites, show neutral gray.
  if (uNSites <= 0) {
    outColor = vec4(vec3(0.2), 1.0);
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
  } else if (uMode == 1 || uMode == 2) {
    // Modes 1 and 2: many-body quantities based on ground-state configuration.
    float Eg_min;
    int   bestState;
    ground_state_energy(Ei_arr, uNSites, uW, Eg_min, bestState);

    if (uMode == 1) {
      // Mode 1: many-body ground-state energy.
      Eplot = Eg_min;
    } else {
      // Mode 2: simple ground-state current based on tip tunneling.
      float Itip = ground_state_tip_current(Ri_arr, Ei_arr, uNSites, Rscale, bestState);
      Eplot = Itip;
    }
  } else {
    // Modes 3–5: PME steady-state observables.
    float rho[16];
    float I_tip;
    solve_pme(Ei_arr, Ri_arr, uNSites, uW, rho, I_tip);

    if (uMode == 3) {
      Eplot = pme_site_occupancy(rho, uNSites, uSiteIndex);
    } else if (uMode == 4) {
      Eplot = pme_total_charge(rho, uNSites);
    } else {
      Eplot = I_tip;
    }
  }

  // Map selected energy to diverging red-white-blue colormap controlled by uEcenter and uEscale.
  vec3 col = colormap_rwb(Eplot, uEcenter, uEscale);

  outColor = vec4(col, 1.0);
}
