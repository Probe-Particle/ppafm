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
uniform float uTemp;    // temperature in Kelvin (from GUI)
uniform float uGammaS;  // substrate coupling (dimensionless)
uniform float uGammaT;  // tip coupling (dimensionless)
uniform float uDecay;   // tunneling decay beta [1/Å]
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

// --- Small 4x4 Cramer's rule solver helpers ---

// 3x3 determinant from a 4x4 block stored row-major in A4 (only first 4x4 used).
// Rows are r0,r1,r2 and columns c0,c1,c2.
float det3_sub4(float A4[16], int r0, int r1, int r2, int c0, int c1, int c2) {
  float a00 = A4[r0*4 + c0];
  float a01 = A4[r0*4 + c1];
  float a02 = A4[r0*4 + c2];
  float a10 = A4[r1*4 + c0];
  float a11 = A4[r1*4 + c1];
  float a12 = A4[r1*4 + c2];
  float a20 = A4[r2*4 + c0];
  float a21 = A4[r2*4 + c1];
  float a22 = A4[r2*4 + c2];

  return a00*(a11*a22 - a12*a21)
       - a01*(a10*a22 - a12*a20)
       + a02*(a10*a21 - a11*a20);
}

// 4x4 determinant of A4 using expansion along first row.
float det4(float A4[16]) {
  float d0 = det3_sub4(A4, 1, 2, 3, 1, 2, 3);
  float d1 = det3_sub4(A4, 1, 2, 3, 0, 2, 3);
  float d2 = det3_sub4(A4, 1, 2, 3, 0, 1, 3);
  float d3 = det3_sub4(A4, 1, 2, 3, 0, 1, 2);

  return  A4[0] * d0
        - A4[1] * d1
        + A4[2] * d2
        - A4[3] * d3;
}

// Solve a 4x4 system A4 * x = b4 using Cramer's rule.
void solve_4x4_cramer(in float A4[16], in float b4[4], out float x4[4]) {
  float detA = det4(A4);
  if (abs(detA) < 1e-20) {
    for (int i = 0; i < 4; ++i) x4[i] = 0.0;
    return;
  }

  float Ai[16];
  // Column 0 replaced by b
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      Ai[i*4 + j] = (j == 0) ? b4[i] : A4[i*4 + j];
    }
  }
  x4[0] = det4(Ai) / detA;

  // Column 1
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      Ai[i*4 + j] = (j == 1) ? b4[i] : A4[i*4 + j];
    }
  }
  x4[1] = det4(Ai) / detA;

  // Column 2
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      Ai[i*4 + j] = (j == 2) ? b4[i] : A4[i*4 + j];
    }
  }
  x4[2] = det4(Ai) / detA;

  // Column 3
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      Ai[i*4 + j] = (j == 3) ? b4[i] : A4[i*4 + j];
    }
  }
  x4[3] = det4(Ai) / detA;
}

// Solve K rho = rhs for rho. For n<=4 use 4x4 Cramer's rule, otherwise
// Gaussian elimination with simple partial pivoting; n <= 16.
void solve_linear_system(
  inout float K[16*16],
  inout float rhs[16],
  int  n
) {
  if (n <= 4) {
    // Copy top-left n x n block into a 4x4 buffer and solve via Cramer.
    float A4[16];
    float b4[4];
    float x4[4];

    for (int i = 0; i < 4; ++i) {
      if (i < n) {
        b4[i] = rhs[i];
      } else {
        b4[i] = 0.0;
      }
      for (int j = 0; j < 4; ++j) {
        if (i < n && j < n) {
          A4[i*4 + j] = K[i*16 + j];
        } else {
          A4[i*4 + j] = (i == j) ? 1.0 : 0.0;
        }
      }
    }

    solve_4x4_cramer(A4, b4, x4);

    for (int i = 0; i < 4; ++i) {
      if (i < n) {
        rhs[i] = x4[i];
      }
    }
    return;
  }

  // Forward elimination with partial pivoting for larger n
  for (int k = 0; k < 16; ++k) {
    if (k >= n) break;

    int   pivotRow = k;
    float pivotVal = abs(K[k*16 + k]);
    for (int i = k + 1; i < 16; ++i) {
      if (i >= n) break;
      float val = abs(K[i*16 + k]);
      if (val > pivotVal) {
        pivotVal = val;
        pivotRow = i;
      }
    }

    if (pivotVal < 1e-20) {
      continue;
    }

    if (pivotRow != k) {
      for (int j = 0; j < 16; ++j) {
        if (j >= n) break;
        float tmp = K[k*16 + j];
        K[k*16 + j]   = K[pivotRow*16 + j];
        K[pivotRow*16 + j] = tmp;
      }
      float tmpR = rhs[k];
      rhs[k]     = rhs[pivotRow];
      rhs[pivotRow] = tmpR;
    }

    float pivot = K[k*16 + k];
    if (abs(pivot) < 1e-20) continue;
    float invPivot = 1.0 / pivot;

    for (int j = k; j < 16; ++j) {
      if (j >= n) break;
      K[k*16 + j] *= invPivot;
    }
    rhs[k] *= invPivot;

    for (int i = k + 1; i < 16; ++i) {
      if (i >= n) break;
      float factor = K[i*16 + k];
      if (abs(factor) < 1e-20) continue;
      for (int j = k; j < 16; ++j) {
        if (j >= n) break;
        K[i*16 + j] -= factor * K[k*16 + j];
      }
      rhs[i] -= factor * rhs[k];
    }
  }

  // Back substitution
  for (int i = n - 1; i >= 0; --i) {
    float sum = rhs[i];
    for (int j = i + 1; j < n; ++j) {
      sum -= K[i*16 + j] * rhs[j];
    }
    float diag = K[i*16 + i];
    if (abs(diag) < 1e-20) {
      continue;
    }
    rhs[i] = sum / diag;
  }
}

// Build PME kernel K from transition rates for both substrate and tip leads.
void build_pme_kernel(
  in  float Es[16],
  in  float Ri_arr[4],
  in  int   nsites,
  in  int   nStates,
  in  float muS,
  in  float muT,
  in  float gammaS0,
  in  float gammaT0,
  in  float w2[4],
  in  float kT,
  inout float K[16*16]
) {
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
}

// Compute PME tip current from solved rho using tip lead only.
float compute_tip_current(
  in float Es[16],
  in float Ri_arr[4],
  in float rho[16],
  in int   nStates,
  in int   nsites,
  in float muT,
  in float gammaT0,
  in float w2[4]
) {
  float I_tip = 0.0;
  float kT   = 0.01;
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
      if (t < nStates) { contrib += rho[t] * (-sign) * G_backward;   }
      I_tip += contrib;
    }
  }
  return I_tip;
}

// PME steady-state solver: fills rho_s for s < nStates and returns a simple tip current.
// K and rhs are provided as out parameters so they can be inspected by the caller for debugging.
void solve_pme(
  in  float Ei_arr[4],
  in  float Ri_arr[4],
  in  int   nsites,
  in  float W,
  out float rho[16],
  out float I_tip,
  out float K[16*16],
  out float rhs[16]
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

  // Temperature kT in eV from uTemp [K]
  float kBoltz = 8.617333262e-5; // eV/K
  float kT   = max(uTemp * kBoltz, 1e-6);
  float muS  = 0.0;        // substrate chemical potential
  float muT  = uVBias;     // tip chemical potential ~ bias
  float gammaS0 = max(uGammaS, 0.0);
  float gammaT0 = max(uGammaT, 0.0);

  // Initialize K and rhs to zero before building the kernel.
  for (int i = 0; i < 16; ++i) {
    rhs[i] = 0.0;
    for (int j = 0; j < 16; ++j) {
      K[i*16 + j] = 0.0;
    }
  }

  // Build K from transition rates for both leads.
  build_pme_kernel(Es, Ri_arr, nsites, nStates, muS, muT, gammaS0, gammaT0, w2, kT, K);

  // Replace last equation by normalization: sum_s rho_s = 1.
  int normRow = nStates - 1;
  for (int j = 0; j < 16; ++j) {
    if (j < nStates) { K[normRow*16 + j] = 1.0; } 
    else             { K[normRow*16 + j] = 0.0; }
  }
  for (int r = 0; r < nStates - 1; ++r) { K[r*16 + normRow] = 0.0;}
  for (int i = 0; i < nStates; ++i)     { rhs[i] = 0.0;}
  rhs[normRow] = 1.0;

  //solve_linear_system(K, rhs, nStates);

  for (int s = 0; s < 16; ++s) { rho[s] = (s < nStates) ? rhs[s] : 0.0; }

  // Compute PME tip current from solved rho.
  I_tip = compute_tip_current(Es, Ri_arr, rho, nStates, nsites, muT, gammaT0, w2);
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
    float K  [16*16];
    float rhs[16];
    solve_pme(Ei_arr, Ri_arr, uNSites, uW, rho, I_tip, K, rhs);

    if (uMode == 3) {
      Eplot = pme_site_occupancy(rho, uNSites, uSiteIndex);
    } else if (uMode == 4) {
      Eplot = pme_total_charge(rho, uNSites);
    } else if (uMode == 5) {
      Eplot = I_tip;
    } else if (uMode == 6) {
      // Debug mode: visualize Es[0]
      float Es_dbg[16];
      float Ne_dbg[16];
      int   nStates_dbg;
      compute_many_body_energies(Ei_arr, uNSites, uW, Es_dbg, Ne_dbg, nStates_dbg);
      Eplot = Es_dbg[0];
    } else if (uMode == 7) {
      // Debug mode: visualize rho[0]
      Eplot = rho[0];
    } else if (uMode == 8) {
      // Debug mode: visualize sum_s rho[s]
      float sumR = 0.0;
      for (int s = 0; s < 16; ++s) {
        sumR += rho[s];
      }
      Eplot = sumR;
    } else if (uMode == 9 || uMode == 10) {
      // Debug modes: inspect raw K and rhs elements from solve_pme.
      // Recompute nStates from uNSites (same logic as compute_many_body_energies).
      int nStates_dbg = 1;
      for (int k = 0; k < 4; ++k) {
        if (k >= uNSites) break;
        nStates_dbg *= 2;
      }

      int maxK = nStates_dbg * nStates_dbg;
      int idx  = uSiteIndex; //clamp(uSiteIndex, 0, (uMode == 9 ? maxK - 1 : nStates_dbg - 1));

      if (uMode == 9) {
        // Treat K as row-major nStates_dbg x nStates_dbg matrix.
        Eplot = K[idx];
      } else {
        // rhs element.
        Eplot = rhs[idx];
      }
    } else {
      Eplot = 0.0;
    }
  }

  // Map selected energy to diverging red-white-blue colormap controlled by uEcenter and uEscale.
  vec3 col = colormap_rwb(Eplot, uEcenter, uEscale);

  outColor = vec4(col, 1.0);
}
