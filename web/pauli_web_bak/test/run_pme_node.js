// Simple Node.js runner to compute PME many-body energies and tip current
// using the browser-oriented pauli_pme_cpu.js. This evaluates a single tip
// point with the current default params (2-dot system) and prints JSON so it
// can be compared against Python/C++ outputs.

const fs = require('fs');
const path = require('path');
const vm = require('vm');

// Load the browser JS into this context
const pmePath = path.join(__dirname, '..', 'pauli_pme_cpu.js');
const code = fs.readFileSync(pmePath, 'utf8');
vm.runInThisContext(code, { filename: pmePath });

// Load shared params from params.json
const paramsPath = path.join(__dirname, 'params.json');
const params = JSON.parse(fs.readFileSync(paramsPath, 'utf8'));

// Default tip position if not present
if (!params.tip) {
  params.tip = { x: 0.0, y: 0.0, z: params.ZTip || 8.0 };
}
const sites = params.sites;

// Compute
const out = solvePmeCpu(params);
const energies = computeManyBodyEnergies(out.Ei, params.NSites, params.W);

const result = {
  params: {
    NSites: params.NSites,
    Rtip: params.Rtip,
    VBias0: params.VBias0,
    VBias: params.VBias,
    TempK: params.TempK,
    GammaS: params.GammaS,
    GammaT: params.GammaT,
    decay: params.decay,
    W: params.W
  },
  sites,
  Ei: out.Ei,
  Ri: out.Ri,
  Es: energies.Es.slice(0, energies.nStates),
  Ne: energies.Ne.slice(0, energies.nStates),
  nStates: energies.nStates,
  K: out.K.slice(0, energies.nStates * energies.nStates),
  rho: out.rho.slice(0, energies.nStates),
  I_tip: out.I_tip
};

console.log(JSON.stringify(result, null, 2));
