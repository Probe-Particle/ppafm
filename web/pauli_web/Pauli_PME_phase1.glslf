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
uniform vec4  uSites[4]; // (x, y, z, E0)

// Phase 1 shader: render minimum on-site energy over all sites.
// On-site energy model (simplified):
//   E_i(tip) = E0_i + VBias * exp(-r_i / Rscale)
// where r_i = |tip - site_i| and Rscale ~ Rtip.
// We then take E_min(tip) = min_i E_i(tip) and map it to color.

// Diverging colormap: red-white-blue (RWB)
//   E = Ecenter           -> white
//   E < Ecenter (negative) -> red side
//   E > Ecenter (positive) -> blue side
// uEscale sets the half-width of the scale; values beyond +/- uEscale are clamped.
vec3 colormap_rwb(float Emin, float center, float scale) {
  float s = max(scale, 1e-6);
  float d = (Emin - center) / s;   // d < 0: red, d > 0: blue
  d = clamp(d, -1.0, 1.0);

  vec3 white = vec3(1.0);
  vec3 red   = vec3(0.8, 0.1, 0.1);
  vec3 blue  = vec3(0.1, 0.1, 0.8);

  if (d > 0.0) {
    // Positive side: blend from white to blue as d goes 0 -> 1
    return mix(white, blue, d);
  } else {
    // Negative side: blend from white to red as d goes 0 -> -1
    return mix(white, red, -d);
  }
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

  // Compute on-site energy for each site and track minimum.
  float Rscale = max(uRtip, 1.0); // avoid division by zero
  float Emin   =  1e9;
  float Emax   = -1e9;

  for (int i = 0; i < 4; ++i) {
    if (i >= uNSites) break;
    vec3 spos = vec3(uSites[i].xyz);
    float E0  = uSites[i].w;
    float r   = length(tip - spos);
    // Simple gating: exponential decay with distance
    float Egate = uVBias * exp(-r / Rscale);
    float Ei    = E0 + Egate;
    Emin = min(Emin, Ei);
    Emax = max(Emax, Ei);
  }

  // Map Emin to diverging red-white-blue colormap controlled by uEcenter and uEscale.
  vec3 col = colormap_rwb(Emin, uEcenter, uEscale);

  gl_FragColor = vec4(col, 1.0);
}
