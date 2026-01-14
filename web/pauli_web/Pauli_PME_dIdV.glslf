precision highp float;
precision highp int;

#ifndef NSITE
#define NSITE 4
#endif
#ifndef NSTATE
#define NSTATE 16 
#endif

in vec2 vUv;
out vec4 outColor;

uniform vec2  uResolution;
uniform int   uNSites;
uniform float uL;
uniform float uZTip;
uniform float uVBias;
uniform float uRtip;
uniform float uTemp;
uniform float uGammaS;
uniform float uGammaT;
uniform float uDecay;
uniform int   uOrder;
uniform vec2  uZV;
uniform bool  uMirror;
uniform bool  uRamp;
uniform float uCs[10];
uniform bool  uUseRot;
uniform mat3  uRot[NSITE];
uniform float uDV;
uniform float uEcenter;
uniform float uEscale;
uniform float uW;
uniform int   uMode;
uniform int   uSiteIndex;
uniform int   uSolverMode;
uniform int   uDebugI;
uniform int   uDebugJ;
uniform bool  uOutputRaw;
uniform vec4  uSites[NSITE];
uniform vec2  uP1;
uniform vec2  uP2;
uniform float uVBiasMin;
uniform float uVBiasMax;
uniform int   uScanMode;

vec3 colormap_rwb(float Emin, float center, float scale) {
  float d = (Emin - center) * scale;
  d = clamp(d, -1.0, 1.0);
  if (d > 0.0) { return vec3(1.0 - d, 1.0 - d, 1.0); }
  else         { return vec3(1.0,     1.0 + d, 1.0 + d); }
}

vec3 tip_from_uv(vec2 uv, float L, float zTip) {
  float x = (uv.x * 2.0 - 1.0) * L;
  float y = (uv.y * 2.0 - 1.0) * L;
  return vec3(x, y, zTip);
}

const float PI = 3.1415926535897932384626433832795;

float Emultipole(vec3 d, int order, float cs[10]) {
  float ir2 = 1.0 / max(dot(d, d), 1e-12);
  float E = cs[0];
  if (order > 0) {
    E += ir2 * (cs[1] * d.x + cs[2] * d.y + cs[3] * d.z);
  }
  return sqrt(ir2) * E;
}

float evalMultipoleMirror(
  vec3 pTip,
  vec3 pSite,
  float VBias,
  float Rtip,
  vec2 zV,
  int order,
  float cs[10],
  float E0,
  mat3 rotSite,
  bool bUseRot,
  bool bMirror,
  bool bRamp
) {
  float zV0 = zV.x;
  float zVd = zV.y;
  float orig_z = pTip.z;
  float zV1 = orig_z + zVd;

  vec3 pTipMirror = pTip;
  pTipMirror.z = 2.0 * zV0 - orig_z;

  vec3 d  = pTip - pSite;
  vec3 dm = pTipMirror - pSite;
  if (bUseRot) {
    d  = rotSite * d;
    dm = rotSite * dm;
  }

  int ord = (order > 1) ? 1 : order;

  float E = Emultipole(d, ord, cs);
  if (bMirror) {
    E -= Emultipole(dm, ord, cs);
  }
  float VR = VBias * Rtip;
  E *= VR;

  if (bRamp) {
    float ramp = (pSite.z - zV0) / (zV1 - zV0);
    if (ramp > 1.0) ramp = 1.0;
    if (pSite.z < zV0) ramp = 0.0;
    float V_lin = VBias * ramp;
    float E_lin = cs[0] * V_lin;
    E += E_lin;
  }
  return E + E0;
}

void compute_site_energies(
  in  vec3  tip,
  in  int   nsites,
  in  float Rscale,
  in  float VBias,
  out float Ei_arr[NSITE],
  out float Ri_arr[NSITE],
  out float Emin,
  out float Emax
) {
  Emin =  1e9;
  Emax = -1e9;
  for (int i = 0; i < NSITE; ++i) {
    if (i >= nsites) {
      Ei_arr[i] = 0.0;
      Ri_arr[i] = 0.0;
      continue;
    }
    vec3 spos = vec3(uSites[i].xyz);
    float E0  = uSites[i].w;
    vec3 d0 = tip - spos;
    float r = length(d0);
    mat3 rot = (uUseRot) ? uRot[i] : mat3(1.0);
    float Ei = evalMultipoleMirror(tip, spos, VBias, uRtip, uZV, uOrder, uCs, E0, rot, uUseRot, uMirror, uRamp);
    Ei_arr[i]   = Ei;
    Ri_arr[i]   = r;
    Emin = min(Emin, Ei);
    Emax = max(Emax, Ei);
  }
}

float fermi(float dE, float kT) {
  float x = dE / kT;
  x = clamp(x, -60.0, 60.0);
  return 1.0 / (1.0 + exp(x));
}

void compute_many_body_energies(
  in  float Ei_arr[NSITE],
  in  int   nsites,
  in  float W,
  out float Es[NSTATE],
  out float Ne[NSTATE],
  out int   nStates
) {
  nStates = 1;
  for (int k = 0; k < NSITE; ++k) {
    if (k >= nsites) break;
    nStates *= 2;
  }

  float w2[NSITE];
  w2[0] = 1.0;
#if NSITE > 1
  w2[1] = 2.0;
#endif
#if NSITE > 2
  w2[2] = 4.0;
#endif
#if NSITE > 3
  w2[3] = 8.0;
#endif

  for (int s = 0; s < NSTATE; ++s) {
    if (s >= nStates) {
      Es[s] = 0.0;
      Ne[s] = 0.0;
      continue;
    }
    float N = 0.0;
    float Esp = 0.0;
    for (int i = 0; i < NSITE; ++i) {
      if (i >= nsites) break;
      float occ = floor(mod(float(s) / w2[i], 2.0));
      N += occ;
      Esp += occ * Ei_arr[i];
    }
    float Npairs = 0.5 * N * (N - 1.0);
    float Ecoul  = W * Npairs;
    Es[s] = Esp + Ecoul;
    Ne[s] = N;
  }
}

void build_pme_kernel(
  in  float Es[NSTATE],
  in  float Ri_arr[NSITE],
  in  int   nsites,
  in  int   nStates,
  in  float muS,
  in  float muT,
  in  float gammaS0,
  in  float gammaT0,
  in  float w2[NSITE],
  in  float kT,
  out float K[NSTATE*NSTATE]
) {
  for (int s = 0; s < NSTATE; ++s) {
    if (s >= nStates) break;

    for (int i = 0; i < NSITE; ++i) {
      if (i >= nsites) break;

      float occ = floor(mod(float(s) / w2[i], 2.0));

      for (int l = 0; l < 2; ++l) {
        float mu     = (l == 0) ? muS : muT;
        float r      = Ri_arr[i];
        float Tamp;
        if (l == 0) {
          Tamp = gammaS0;
        } else {
          Tamp = gammaT0 * exp(-uDecay * r);
        }
        float rate0 = 2.0 * PI * (Tamp * Tamp);

        int t = s;
        float dN = 0.0;
        if (occ < 0.5) {
          t = s + int(w2[i]);
          if (t >= nStates) continue;
          dN = 1.0;
        } else {
          t = s - int(w2[i]);
          if (t < 0) continue;
          dN = -1.0;
        }

        float dE = Es[t] - Es[s] - mu * dN;
        float f  = fermi(dE, kT);

        float G_forward  = rate0 * f;
        float G_backward = rate0 * (1.0 - f);

        K[s*NSTATE + s] -= G_forward;
        K[t*NSTATE + s] += G_forward;
        K[t*NSTATE + t] -= G_backward;
        K[s*NSTATE + t] += G_backward;
      }
    }
  }
}

float compute_tip_current(
  in float Es[NSTATE],
  in float Ri_arr[NSITE],
  in float rho[NSTATE],
  in int   nStates,
  in int   nsites,
  in float muT,
  in float gammaT0,
  in float w2[NSITE],
  in float kT
) {
  float I_tip = 0.0;
  for (int s = 0; s < NSTATE; ++s) {
    if (s >= nStates) break;

    for (int i = 0; i < NSITE; ++i) {
      if (i >= nsites) break;
      float occ = floor(mod(float(s) / w2[i], 2.0));
      float r   = Ri_arr[i];
      float Tamp = gammaT0 * exp(-uDecay * r);
      float rate0 = 2.0 * PI * (Tamp * Tamp);

      int t = s;
      float dN = 0.0;
      if (occ < 0.5) {
        t = s + int(w2[i]);
        if (t >= nStates) continue;
        dN = 1.0;
      } else {
        t = s - int(w2[i]);
        if (t < 0) continue;
        dN = -1.0;
      }

      float dE = Es[t] - Es[s] - muT * dN;
      float f  = fermi(dE, kT);
      float G_forward  = rate0 * f;
      float G_backward = rate0 * (1.0 - f);

      float sign = (dN > 0.0) ? 1.0 : -1.0;
      float contrib = 0.0;
      contrib += rho[s] * sign * G_forward;
      if (t < nStates) { contrib += rho[t] * (-sign) * G_backward; }
      I_tip += contrib;
    }
  }
  return I_tip;
}

void solve_linear_system(inout float A[NSTATE*NSTATE], inout float b[NSTATE], int n) {
  for (int i = 0; i < NSTATE; ++i) {
    if (i >= n) break;

    float piv = A[i*NSTATE + i];
    int pivRow = i;
    float maxAbs = abs(piv);
    for (int r = i + 1; r < NSTATE; ++r) {
      if (r >= n) break;
      float v = abs(A[r*NSTATE + i]);
      if (v > maxAbs) { maxAbs = v; pivRow = r; }
    }
    if (pivRow != i) {
      for (int c = i; c < NSTATE; ++c) {
        if (c >= n) break;
        float tmp = A[i*NSTATE + c];
        A[i*NSTATE + c] = A[pivRow*NSTATE + c];
        A[pivRow*NSTATE + c] = tmp;
      }
      float tb = b[i];
      b[i] = b[pivRow];
      b[pivRow] = tb;
    }

    piv = A[i*NSTATE + i];
    float invP = 1.0 / ((abs(piv) > 1e-20) ? piv : (piv >= 0.0 ? 1e-20 : -1e-20));

    for (int c = i; c < NSTATE; ++c) {
      if (c >= n) break;
      A[i*NSTATE + c] *= invP;
    }
    b[i] *= invP;

    for (int r = 0; r < NSTATE; ++r) {
      if (r >= n) break;
      if (r == i) continue;
      float f = A[r*NSTATE + i];
      if (abs(f) < 1e-20) continue;
      for (int c = i; c < NSTATE; ++c) {
        if (c >= n) break;
        A[r*NSTATE + c] -= f * A[i*NSTATE + c];
      }
      b[r] -= f * b[i];
    }
  }
}

float solve_pme_current(
  in  float Ei_arr[NSITE],
  in  float Ri_arr[NSITE],
  in  int   nsites,
  in  float W,
  in  float VBias
) {
  float Es[NSTATE];
  float Ne[NSTATE];
  int   nStates;
  compute_many_body_energies(Ei_arr, nsites, W, Es, Ne, nStates);

  float w2[NSITE];
  w2[0] = 1.0;
#if NSITE > 1
  w2[1] = 2.0;
#endif
#if NSITE > 2
  w2[2] = 4.0;
#endif
#if NSITE > 3
  w2[3] = 8.0;
#endif

  float kBoltz = 8.617333262e-5;
  float kT   = max(uTemp * kBoltz, 1e-6);
  float muS  = 0.0;
  float muT  = VBias;
  float gammaS0 = max(uGammaS, 0.0) / PI;
  float gammaT0 = max(uGammaT, 0.0) / PI;

  float K[NSTATE*NSTATE];
  float rhs[NSTATE];

  for (int i = 0; i < NSTATE; ++i) {
    rhs[i] = 0.0;
    for (int j = 0; j < NSTATE; ++j) {
      K[i*NSTATE + j] = 0.0;
    }
  }

  build_pme_kernel(Es, Ri_arr, nsites, nStates, muS, muT, gammaS0, gammaT0, w2, kT, K);

  int normRow = 0;
  for (int j = 0; j < NSTATE; ++j) {
    if (j < nStates) { K[normRow*NSTATE + j] = 1.0; }
    else             { K[normRow*NSTATE + j] = 0.0; }
  }
  for (int i = 0; i < nStates; ++i) { rhs[i] = 0.0; }
  rhs[normRow] = 1.0;

  solve_linear_system(K, rhs, nStates);

  float rho[NSTATE];
  for (int s = 0; s < NSTATE; ++s) { rho[s] = (s < nStates) ? rhs[s] : 0.0; }

  return compute_tip_current(Es, Ri_arr, rho, nStates, nsites, muT, gammaT0, w2, kT);
}

void main() {
  vec3 tip;
  float localVBias;

  if (uScanMode == 1) {
    float t = vUv.x;
    vec2 pos2d = mix(uP1, uP2, t);
    tip = vec3(pos2d, uZTip);
    localVBias = mix(uVBiasMin, uVBiasMax, vUv.y);
  } else {
    tip = tip_from_uv(vUv, uL, uZTip);
    localVBias = uVBias;
  }

  if (uNSites <= 0) {
    outColor = vec4(vec3(0.2), 1.0);
    return;
  }

  float Rscale = max(uRtip, 1.0);
  float Ei_arr[NSITE];
  float Ri_arr[NSITE];
  float Emin;
  float Emax;

  float dV = max(abs(uDV), 1e-6);

  float IP;
  float IM;

  {
    float Vp = localVBias + dV;
    compute_site_energies(tip, uNSites, Rscale, Vp, Ei_arr, Ri_arr, Emin, Emax);
    IP = solve_pme_current(Ei_arr, Ri_arr, uNSites, uW, Vp);
  }

  {
    float Vm = localVBias - dV;
    compute_site_energies(tip, uNSites, Rscale, Vm, Ei_arr, Ri_arr, Emin, Emax);
    IM = solve_pme_current(Ei_arr, Ri_arr, uNSites, uW, Vm);
  }

  float Eplot = (IP - IM) / (2.0 * dV);

  if (uOutputRaw) {
    outColor = vec4(Eplot, 0.0, 0.0, 1.0);
    return;
  }

  vec3 col = colormap_rwb(Eplot, uEcenter, uEscale);
  outColor = vec4(col, 1.0);
}
