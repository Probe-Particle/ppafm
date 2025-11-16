// CPU-side PME debug solver for a single tip position.
// This mirrors the GLSL logic in Pauli_PME_phase1.glslf as closely as possible
// so we can inspect Es, K, rhs, rho, and I_tip via console logging.

'use strict';

// Map tip-plane coordinates from (x, y, zTip) and sites -> Ei_arr, Ri_arr
function computeSiteEnergies(tip, sites, nsites, Rscale, VBias) {
  const Ei = new Array(4).fill(0.0);
  const Ri = new Array(4).fill(0.0);
  let Emin = 1e9;
  let Emax = -1e9;
  for (let i = 0; i < 4; i++) {
    if (i >= nsites) continue;
    const sx = sites[i].x;
    const sy = sites[i].y;
    const sz = sites[i].z;
    const E0 = sites[i].E;
    const dx = tip.x - sx;
    const dy = tip.y - sy;
    const dz = tip.z - sz;
    const r = Math.sqrt(dx*dx + dy*dy + dz*dz);
    const Egate = VBias * Rscale / Math.max(r, 1e-3);
    const Ei_i  = E0 + Egate;
    Ei[i] = Ei_i;
    Ri[i] = r;
    if (Ei_i < Emin) Emin = Ei_i;
    if (Ei_i > Emax) Emax = Ei_i;
  }
  return { Ei, Ri, Emin, Emax };
}

// Many-body energies Es[s] and electron counts Ne[s]
function computeManyBodyEnergies(Ei, nsites, W) {
  let maxStates = 1;
  for (let k = 0; k < 4; k++) {
    if (k >= nsites) break;
    maxStates *= 2;
  }
  const nStates = maxStates;

  const Es = new Array(16).fill(0.0);
  const Ne = new Array(16).fill(0.0);
  const w2 = [1, 2, 4, 8];

  for (let s = 0; s < 16; s++) {
    if (s >= maxStates) {
      Es[s] = 0.0;
      Ne[s] = 0.0;
      continue;
    }
    let N = 0.0;
    let Esp = 0.0;
    for (let i = 0; i < 4; i++) {
      if (i >= nsites) break;
      const occ = Math.floor((s / w2[i]) % 2);
      N   += occ;
      Esp += occ * Ei[i];
    }
    const Npairs = 0.5 * N * (N - 1.0);
    const Ecoul  = W * Npairs;
    Es[s] = Esp + Ecoul;
    Ne[s] = N;
  }
  return { Es, Ne, nStates };
}

function fermi(dE, kT) {
  const kTeff = Math.max(kT, 1e-6);
  let x = dE / kTeff;
  if (x < -40.0) x = -40.0;
  if (x >  40.0) x =  40.0;
  const ex = Math.exp(-x);
  return 1.0 / (1.0 + ex);
}

// Build PME kernel K (16x16) in row-major form, same as GLSL build_pme_kernel
function buildPmeKernel(Es, Ri, nsites, nStates, muS, muT, gammaS0, gammaT0, kT, Rtip, useDecay, decay) {
  const K = new Array(16 * 16).fill(0.0);
  const w2 = [1, 2, 4, 8];

  for (let s = 0; s < 16; s++) {
    if (s >= nStates) break;

    for (let i = 0; i < 4; i++) {
      if (i >= nsites) break;
      const occ = Math.floor((s / w2[i]) % 2);

      for (let l = 0; l < 2; l++) {
        const mu     = (l === 0) ? muS : muT;
        const gamma0 = (l === 0) ? gammaS0 : gammaT0;
        const r      = Ri[i];
        let Ttun;
        if (useDecay) {
          Ttun = Math.exp(-decay * r);
        } else {
          Ttun = Math.exp(-2.0 * r / Math.max(Rtip, 1e-3));
        }
        const gamma  = gamma0 * Ttun;

        let t = s;
        let dN = 0.0;
        if (occ < 0.5) {
          // electron in: s -> t with bit i set
          t = s + w2[i];
          if (t >= nStates) continue;
          dN = 1.0;
        } else {
          // electron out: s -> t with bit i cleared
          t = s - w2[i];
          if (t < 0) continue;
          dN = -1.0;
        }

        const dE = Es[t] - Es[s] - mu * dN;
        const f  = fermi(dE, kT);

        const G_forward  = gamma * f;
        const G_backward = gamma * (1.0 - f);

        // Outflow from s due to s->t, inflow to t
        K[s*16 + s] += G_forward;
        K[t*16 + s] -= G_forward;
        // Outflow from t due to t->s, inflow to s
        K[t*16 + t] += G_backward;
        K[s*16 + t] -= G_backward;
      }
    }
  }

  return K;
}

// Simple dense Gaussian elimination with partial pivoting on an nStates x nStates block
function solveLinearSystem(K, rhs, nStates) {
  const n = nStates;
  const A = K.slice(); // copy
  const b = rhs.slice();

  for (let k = 0; k < n; k++) {
    // pivot
    let pivotRow = k;
    let pivotVal = Math.abs(A[k*16 + k]);
    for (let i = k+1; i < n; i++) {
      const val = Math.abs(A[i*16 + k]);
      if (val > pivotVal) {
        pivotVal = val;
        pivotRow = i;
      }
    }
    if (pivotVal < 1e-20) continue;

    if (pivotRow !== k) {
      for (let j = 0; j < n; j++) {
        const tmp = A[k*16 + j];
        A[k*16 + j] = A[pivotRow*16 + j];
        A[pivotRow*16 + j] = tmp;
      }
      const tmpb = b[k];
      b[k] = b[pivotRow];
      b[pivotRow] = tmpb;
    }

    const pivot = A[k*16 + k];
    if (Math.abs(pivot) < 1e-20) continue;
    const invP = 1.0 / pivot;

    for (let j = k; j < n; j++) {
      A[k*16 + j] *= invP;
    }
    b[k] *= invP;

    for (let i = k+1; i < n; i++) {
      const factor = A[i*16 + k];
      if (Math.abs(factor) < 1e-20) continue;
      for (let j = k; j < n; j++) {
        A[i*16 + j] -= factor * A[k*16 + j];
      }
      b[i] -= factor * b[k];
    }
  }

  // back substitution
  for (let i = n-1; i >= 0; i--) {
    let sum = b[i];
    for (let j = i+1; j < n; j++) {
      sum -= A[i*16 + j] * b[j];
    }
    const diag = A[i*16 + i];
    if (Math.abs(diag) < 1e-20) continue;
    b[i] = sum / diag;
  }

  return b; // solution
}

function computeTipCurrent(Es, Ri, rho, nStates, nsites, muT, gammaT0, Rtip, useDecay, decay, kT) {
  const w2 = [1, 2, 4, 8];
  let I_tip = 0.0;

  for (let s = 0; s < 16; s++) {
    if (s >= nStates) break;
    for (let i = 0; i < 4; i++) {
      if (i >= nsites) break;
      const occ = Math.floor((s / w2[i]) % 2);
      const r   = Ri[i];
      let Ttun;
      if (useDecay) {
        Ttun = Math.exp(-decay * r);
      } else {
        Ttun = Math.exp(-2.0 * r / Math.max(Rtip, 1e-3));
      }
      const gammaT = gammaT0 * Ttun;

      let t = s;
      let dN = 0.0;
      if (occ < 0.5) {
        t = s + w2[i];
        if (t >= nStates) continue;
        dN = 1.0;
      } else {
        t = s - w2[i];
        if (t < 0) continue;
        dN = -1.0;
      }

      const dE = Es[t] - Es[s] - muT * dN;
      const f  = fermi(dE, kT);
      const G_forward  = gammaT * f;
      const G_backward = gammaT * (1.0 - f);

      const sign = (dN > 0.0) ? 1.0 : -1.0;
      let contrib = 0.0;
      contrib += rho[s] * sign * G_forward;
      if (t < nStates) contrib += rho[t] * (-sign) * G_backward;
      I_tip += contrib;
    }
  }

  return I_tip;
}

// PME observables on CPU, mirroring GLSL helpers.
function pmeSiteOccupancy(rho, nsites, siteIdx) {
  const w2 = [1, 2, 4, 8];
  if (siteIdx < 0) siteIdx = 0;
  if (siteIdx >= nsites) siteIdx = nsites - 1;
  let occSite = 0.0;
  let nStates = 1;
  for (let k = 0; k < 4; k++) {
    if (k >= nsites) break;
    nStates *= 2;
  }
  for (let s = 0; s < 16; s++) {
    if (s >= nStates) break;
    const occ = Math.floor((s / w2[siteIdx]) % 2);
    occSite += rho[s] * occ;
  }
  return occSite;
}

function pmeTotalCharge(rho, nsites) {
  const w2 = [1, 2, 4, 8];
  let Q = 0.0;
  let nStates = 1;
  for (let k = 0; k < 4; k++) {
    if (k >= nsites) break;
    nStates *= 2;
  }
  for (let s = 0; s < 16; s++) {
    if (s >= nStates) break;
    let N = 0.0;
    for (let i = 0; i < 4; i++) {
      if (i >= nsites) break;
      const occ = Math.floor((s / w2[i]) % 2);
      N += occ;
    }
    Q += rho[s] * N;
  }
  return Q;
}

// High-level: solve PME for a single tip point and return debug info
function solvePmeCpu(params) {
  const {
    tip,          // {x,y,z}
    sites,        // array of {x,y,z,E}, length <= 4
    nsites,
    Rtip,
    VBias,
    TempK,
    GammaS,
    GammaT,
    decay,
    W
  } = params;

  const Rscale = Math.max(Rtip, 1.0);
  const { Ei, Ri } = computeSiteEnergies(tip, sites, nsites, Rscale, VBias);

  const { Es, Ne, nStates } = computeManyBodyEnergies(Ei, nsites, W);

  const kBoltz = 8.617333262e-5; // eV/K
  const kT     = Math.max(TempK * kBoltz, 1e-6);
  const muS    = 0.0;
  const muT    = VBias;
  const gammaS0 = Math.max(GammaS, 0.0);
  const gammaT0 = Math.max(GammaT, 0.0);

  const useDecay = true; // mirror planned decay-based tunneling

  let K = buildPmeKernel(Es, Ri, nsites, nStates, muS, muT, gammaS0, gammaT0, kT, Rtip, useDecay, decay);

  const rhs = new Array(16).fill(0.0);
  const normRow = nStates - 1;
  for (let j = 0; j < 16; j++) {
    if (j < nStates) K[normRow*16 + j] = 1.0;
    else             K[normRow*16 + j] = 0.0;
  }
  for (let r = 0; r < nStates - 1; r++) {
    K[r*16 + normRow] = 0.0;
  }
  for (let i = 0; i < nStates; i++) rhs[i] = 0.0;
  rhs[normRow] = 1.0;

  const rhoSol = solveLinearSystem(K, rhs, nStates);
  const rho = new Array(16).fill(0.0);
  for (let s = 0; s < 16; s++) {
    rho[s] = (s < nStates) ? rhoSol[s] : 0.0;
  }

  const I_tip = computeTipCurrent(Es, Ri, rho, nStates, nsites, muT, gammaT0, Rtip, useDecay, decay, kT);

  return { Ei, Ri, Es, Ne, nStates, K, rhs: rhoSol, rho, I_tip };
}

// Convenience debug entry: call from console with current GUI-like params
// Example usage in browser devtools:
//   const tip = {x: 0, y: 0, z: 8};
//   const sites = [
//     {x: -5.785, y: 0, z: 0, E: -0.1},
//     {x:  5.785, y: 0, z: 0, E: -0.1}
//   ];
//   pauliPmeDebug({ tip, sites, nsites: 2, Rtip: 4.0, VBias: 0.3, TempK: 3.0, GammaS: 0.01, GammaT: 0.01, decay: 0.3, W: 0.0 });

function pauliPmeDebug(params) {
  const out = solvePmeCpu(params);
  console.log('PME CPU debug result:', out);
  return out;
}

// Expose to global scope if running in browser
if (typeof window !== 'undefined') {
  window.solvePmeCpu  = solvePmeCpu;
  window.pauliPmeDebug = pauliPmeDebug;

  function getCurrentPmeUniformParams() {
    var uniforms = window.uniforms || {};
    return {
      nsites:   uniforms.uNSites  ? uniforms.uNSites.value   : 2,
      Rtip:     uniforms.uRtip    ? uniforms.uRtip.value     : 4.0,
      zTip:     uniforms.uZTip    ? uniforms.uZTip.value     : 8.0,
      VBias:    uniforms.uVBias   ? uniforms.uVBias.value    : 0.3,
      TempK:    uniforms.uTemp    ? uniforms.uTemp.value     : 3.0,
      GammaS:   uniforms.uGammaS  ? uniforms.uGammaS.value   : 0.01,
      GammaT:   uniforms.uGammaT  ? uniforms.uGammaT.value   : 0.01,
      decay:    uniforms.uDecay   ? uniforms.uDecay.value    : 0.3,
      W:        uniforms.uW       ? uniforms.uW.value        : 0.0,
      mode:     uniforms.uMode    ? uniforms.uMode.value     : 3,
      siteIdx:  uniforms.uSiteIndex ? uniforms.uSiteIndex.value : 0
    };
  }

  // CPU-side PME line scan between the first two sites using solvePmeCpu.
  // Uses current GUI values and sample count from inpLineScanPoints.
  window.runCpuLineScan = function runCpuLineScan() {
    if (typeof window.solvePmeCpu !== 'function') {
      console.warn('solvePmeCpu not available (pauli_pme_cpu.js not loaded?)');
      return;
    }

    if (typeof window.uniforms === 'undefined') {
      console.warn('uniforms not available; runCpuLineScan expects globals from pauli_pme_viewer.html');
      return;
    }

    var txt = document.getElementById('txtSites').value || '';
    var lines = txt.split(/\r?\n/).map(function (l) { return l.trim(); }).filter(function (l) { return l.length > 0; });
    var parsed = [];
    for (var i = 0; i < lines.length && parsed.length < 4; i++) {
      var parts = lines[i].split(/\s+/);
      if (parts.length < 4) continue;
      var x = parseFloat(parts[0]);
      var y = parseFloat(parts[1]);
      var z = parseFloat(parts[2]);
      var E = parseFloat(parts[3]);
      if (!isFinite(x) || !isFinite(y) || !isFinite(z) || !isFinite(E)) continue;
      parsed.push({ x: x, y: y, z: z, E: E });
    }
    if (parsed.length < 2) {
      console.warn('CPU line scan requires at least two sites. Parsed:', parsed.length);
      return;
    }

    var p = getCurrentPmeUniformParams();

    var nsites = p.nsites;
    if (nsites < 2) nsites = 2;
    if (nsites > parsed.length) nsites = parsed.length;

    var nPts = parseInt(document.getElementById('inpLineScanPoints').value, 10);
    if (!isFinite(nPts) || nPts < 2) nPts = 20;

    var Rtip   = p.Rtip;
    var zTip   = p.zTip;
    var VBias  = p.VBias;
    var TempK  = p.TempK;
    var GammaS = p.GammaS;
    var GammaT = p.GammaT;
    var decay  = p.decay;
    var W      = p.W;
    var mode   = p.mode;
    var siteIdx = p.siteIdx;

    var s0 = parsed[0];
    var s1 = parsed[1];

    var dx = s1.x - s0.x;
    var dy = s1.y - s0.y;
    var dz = s1.z - s0.z;

    var xs = [];
    var Ys = [];

    for (var iPt = 0; iPt < nPts; iPt++) {
      var t = (nPts === 1) ? 0.0 : iPt / (nPts - 1);
      var tip = {
        x: s0.x + t * dx,
        y: s0.y + t * dy,
        z: zTip
      };

      var out = window.solvePmeCpu({
        tip: tip,
        sites: parsed,
        nsites: nsites,
        Rtip: Rtip,
        VBias: VBias,
        TempK: TempK,
        GammaS: GammaS,
        GammaT: GammaT,
        decay: decay,
        W: W
      });

      var val;
      if (mode === 3) {
        val = pmeSiteOccupancy(out.rho, nsites, siteIdx);
      } else if (mode === 4) {
        val = pmeTotalCharge(out.rho, nsites);
      } else {
        val = out.I_tip;
      }

      xs.push(t);
      Ys.push(val);
    }

    var ctx = document.getElementById('cpuLineScanCanvas');
    if (!ctx) return;

    if (window.cpuLineScanChart) {
      window.cpuLineScanChart.destroy();
      window.cpuLineScanChart = null;
    }

    var label;
    if (mode === 3) label = 'PME occupancy (CPU)';
    else if (mode === 4) label = 'PME total charge (CPU)';
    else label = 'PME I_tip (CPU)';

    window.cpuLineScanChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: xs,
        datasets: [{
          label: label,
          data: Ys,
          borderColor: 'rgba(0, 192, 255, 1.0)',
          backgroundColor: 'rgba(0, 192, 255, 0.25)',
          pointRadius: 1,
          tension: 0.0
        }]
      },
      options: {
        animation: false,
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            title: { display: true, text: 't along line (0 = site 0, 1 = site 1)' }
          },
          y: {
            title: { display: true, text: label }
          }
        },
        plugins: {
          legend: { display: false }
        }
      }
    });
  };
}
