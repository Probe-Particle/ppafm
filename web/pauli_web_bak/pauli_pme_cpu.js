// CPU-side PME debug solver for a single tip position.
// This mirrors the GLSL logic in Pauli_PME_phase1.glslf as closely as possible
// so we can inspect Es, K, rhs, rho, and I_tip via console logging.

'use strict';

// Minimal debug printing helpers for vectors and matrices.
// Returns current verbosity level (numeric). Default 0 if unset.
function getPmeVerbosity() {
    if (typeof window !== 'undefined' && typeof window.pmeVerbosity === 'number') {
        return window.pmeVerbosity;
    }
    return 0;
}

function dbgPrintVector(label, v, n) {
    if (getPmeVerbosity() < 4) return;
    var m = (typeof n === 'number') ? n : v.length;
    var row = [];
    for (var i = 0; i < m; i++) {
        var x = v[i];
        row.push((isFinite(x) ? x : NaN).toFixed ? x.toFixed(6) : String(x));
    }
    console.log(label + ' [' + row.join(', ') + ']');
}

function dbgPrintMatrix(label, K, nStates) {
    if (getPmeVerbosity() < 4) return;
    console.log(label + ' (size ' + nStates + 'x' + nStates + '):');
    for (var r = 0; r < nStates; r++) {
        var row = [];
        for (var c = 0; c < nStates; c++) {
            var x = K[r * 16 + c];
            row.push((isFinite(x) ? x : NaN).toFixed ? x.toFixed(6) : String(x));
        }
        console.log('  ' + row.join(' '));
    }
}

// Map tip-plane coordinates from (x, y, zTip) and sites -> Ei_arr, Ri_arr
function computeSiteEnergies(tip, sites, NSites, Rscale, VBias) {
    const Ei = new Array(4).fill(0.0);
    const Ri = new Array(4).fill(0.0);
    let Emin = 1e9;
    let Emax = -1e9;
    for (let i = 0; i < 4; i++) {
        if (i >= NSites) continue;
        const sx = sites[i].x;
        const sy = sites[i].y;
        const sz = sites[i].z;
        const E0 = sites[i].E;
        const dx = tip.x - sx;
        const dy = tip.y - sy;
        const dz = tip.z - sz;
        const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
        const Egate = VBias * Rscale / Math.max(r, 1e-3);
        const Ei_i = E0 + Egate;
        Ei[i] = Ei_i;
        Ri[i] = r;
        if (Ei_i < Emin) Emin = Ei_i;
        if (Ei_i > Emax) Emax = Ei_i;
    }
    return { Ei, Ri, Emin, Emax };
}

// Many-body energies Es[s] and electron counts Ne[s]
function computeManyBodyEnergies(Ei, NSites, W) {
    let maxStates = 1;
    for (let k = 0; k < 4; k++) {
        if (k >= NSites) break;
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
            if (i >= NSites) break;
            const occ = Math.floor((s / w2[i]) % 2);
            N += occ;
            Esp += occ * Ei[i];
        }
        const Npairs = 0.5 * N * (N - 1.0);
        const Ecoul = W * Npairs;
        Es[s] = Esp + Ecoul;
        Ne[s] = N;
    }
    return { Es, Ne, nStates };
}

function fermi(dE, kT) {
    const kTeff = Math.max(kT, 1e-6);
    let x = dE / kTeff;
    if (x < -40.0) x = -40.0;
    if (x > 40.0) x = 40.0;
    const ex = Math.exp(-x);
    return 1.0 / (1.0 + ex);
}

// Build PME kernel K (16x16) in row-major form, same as GLSL build_pme_kernel
function buildPmeKernel(Es, Ri, NSites, nStates, muS, muT, gammaS0, gammaT0, kT, Rtip, useDecay, decay) {
    const K = new Array(16 * 16).fill(0.0);
    const w2 = [1, 2, 4, 8];

    for (let s = 0; s < 16; s++) {
        if (s >= nStates) break;

        for (let i = 0; i < 4; i++) {
            if (i >= NSites) break;
            const occ = Math.floor((s / w2[i]) % 2);

            for (let l = 0; l < 2; l++) {
                const mu = (l === 0) ? muS : muT;
                const gamma0 = (l === 0) ? gammaS0 : gammaT0;
                const r = Ri[i];
                let Ttun;
                if (useDecay) {
                    Ttun = Math.exp(-decay * r);
                } else {
                    Ttun = Math.exp(-2.0 * r / Math.max(Rtip, 1e-3));
                }
                const gamma = gamma0 * Ttun;

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
                const f = fermi(dE, kT);

                const G_forward = gamma * f;
                const G_backward = gamma * (1.0 - f);

                // Outflow from s due to s->t, inflow to t
                K[s * 16 + s] += G_forward;
                K[t * 16 + s] -= G_forward;
                // Outflow from t due to t->s, inflow to s
                K[t * 16 + t] += G_backward;
                K[s * 16 + t] -= G_backward;
            }
        }
    }

    return K;
}

// Ported from GLSL
function det3_sub4(A4, r0, r1, r2, c0, c1, c2) {
    var a00 = A4[r0 * 4 + c0];
    var a01 = A4[r0 * 4 + c1];
    var a02 = A4[r0 * 4 + c2];
    var a10 = A4[r1 * 4 + c0];
    var a11 = A4[r1 * 4 + c1];
    var a12 = A4[r1 * 4 + c2];
    var a20 = A4[r2 * 4 + c0];
    var a21 = A4[r2 * 4 + c1];
    var a22 = A4[r2 * 4 + c2];

    return a00 * (a11 * a22 - a12 * a21)
        - a01 * (a10 * a22 - a12 * a20)
        + a02 * (a10 * a21 - a11 * a20);
}

function det4(A4) {
    var d0 = det3_sub4(A4, 1, 2, 3, 1, 2, 3);
    var d1 = det3_sub4(A4, 1, 2, 3, 0, 2, 3);
    var d2 = det3_sub4(A4, 1, 2, 3, 0, 1, 3);
    var d3 = det3_sub4(A4, 1, 2, 3, 0, 1, 2);

    return A4[0] * d0
        - A4[1] * d1
        + A4[2] * d2
        - A4[3] * d3;
}

function solve_4x4_cramer(A4, b4) {
    var x4 = [0, 0, 0, 0];
    var detA = det4(A4);
    if (Math.abs(detA) < 1e-20) {
        return x4;
    }

    var Ai = new Array(16);
    // Column 0
    for (var i = 0; i < 4; ++i) {
        for (var j = 0; j < 4; ++j) {
            Ai[i * 4 + j] = (j == 0) ? b4[i] : A4[i * 4 + j];
        }
    }
    x4[0] = det4(Ai) / detA;

    // Column 1
    for (var i = 0; i < 4; ++i) {
        for (var j = 0; j < 4; ++j) {
            Ai[i * 4 + j] = (j == 1) ? b4[i] : A4[i * 4 + j];
        }
    }
    x4[1] = det4(Ai) / detA;

    // Column 2
    for (var i = 0; i < 4; ++i) {
        for (var j = 0; j < 4; ++j) {
            Ai[i * 4 + j] = (j == 2) ? b4[i] : A4[i * 4 + j];
        }
    }
    x4[2] = det4(Ai) / detA;

    // Column 3
    for (var i = 0; i < 4; ++i) {
        for (var j = 0; j < 4; ++j) {
            Ai[i * 4 + j] = (j == 3) ? b4[i] : A4[i * 4 + j];
        }
    }
    x4[3] = det4(Ai) / detA;

    return x4;
}

function solveLinearSystem_cramer(K, rhs, nStates) {
    // Copy top-left n x n block into a 4x4 buffer and solve via Cramer.
    var A4 = new Array(16).fill(0.0);
    var b4 = new Array(4).fill(0.0);

    for (var i = 0; i < 4; ++i) {
        if (i < nStates) {
            b4[i] = rhs[i];
        } else {
            b4[i] = 0.0;
        }
        for (var j = 0; j < 4; ++j) {
            if (i < nStates && j < nStates) {
                A4[i * 4 + j] = K[i * 16 + j];
            } else {
                A4[i * 4 + j] = (i == j) ? 1.0 : 0.0;
            }
        }
    }

    var x4 = solve_4x4_cramer(A4, b4);

    var out = new Array(16).fill(0.0);
    for (var i = 0; i < 4; ++i) {
        if (i < nStates) out[i] = x4[i];
    }
    return out;
}

// Simple dense Gaussian elimination with partial pivoting on an nStates x nStates block
function solveLinearSystem_our(K, rhs, nStates) {
    const n = nStates;

    const A = K.slice(); // copy
    const b = rhs.slice();

    for (let k = 0; k < n; k++) {
        // pivot
        let pivotRow = k;
        let pivotVal = Math.abs(A[k * 16 + k]);
        for (let i = k + 1; i < n; i++) {
            const val = Math.abs(A[i * 16 + k]);
            if (val > pivotVal) {
                pivotVal = val;
                pivotRow = i;
            }
        }
        if (pivotVal < 1e-20) continue;

        if (pivotRow !== k) {
            for (let j = 0; j < n; j++) {
                const tmp = A[k * 16 + j];
                A[k * 16 + j] = A[pivotRow * 16 + j];
                A[pivotRow * 16 + j] = tmp;
            }
            const tmpb = b[k];
            b[k] = b[pivotRow];
            b[pivotRow] = tmpb;
        }

        const pivot = A[k * 16 + k];
        if (Math.abs(pivot) < 1e-20) continue;
        const invP = 1.0 / pivot;

        for (let j = k; j < n; j++) {
            A[k * 16 + j] *= invP;
        }
        b[k] *= invP;

        for (let i = k + 1; i < n; i++) {
            const factor = A[i * 16 + k];
            if (Math.abs(factor) < 1e-20) continue;
            for (let j = k; j < n; j++) {
                A[i * 16 + j] -= factor * A[k * 16 + j];
            }
            b[i] -= factor * b[k];
        }
    }

    // back substitution
    for (let i = n - 1; i >= 0; i--) {
        let sum = b[i];
        for (let j = i + 1; j < n; j++) {
            sum -= A[i * 16 + j] * b[j];
        }
        const diag = A[i * 16 + i];
        if (Math.abs(diag) < 1e-20) continue;
        b[i] = sum / diag;
    }

    return b; // solution
}

function solveLinearSystem_numericjs(K, rhs, nStates) {
    const n = nStates;

    if (typeof numeric === 'undefined' || typeof numeric.solve !== 'function') {
        return solveLinearSystem_our(K, rhs, nStates);
    }

    // Build a compact n x n matrix view from the fixed 16x16 row-major storage.
    const A = new Array(n);
    for (let i = 0; i < n; i++) {
        const row = new Array(n);
        const base = i * 16;
        for (let j = 0; j < n; j++) {
            row[j] = K[base + j];
        }
        A[i] = row;
    }

    const b = new Array(n);
    for (let i = 0; i < n; i++) {
        b[i] = rhs[i];
    }

    const x = numeric.solve(A, b);

    // Return a 16-element solution vector to match existing callers, zero-padded beyond nStates.
    const out = new Array(16).fill(0.0);
    for (let i = 0; i < n; i++) {
        out[i] = x[i];
    }
    return out;
}

function solveLinearSystem_wraper(K, rhs, nStates, solver) {
    // Explicit choice if provided.
    if (solver === 'cramer') {
        return solveLinearSystem_cramer(K, rhs, nStates);
    }
    if (solver === 'our') {
        return solveLinearSystem_our(K, rhs, nStates);
    }
    if (solver === 'numericjs') {
        return solveLinearSystem_numericjs(K, rhs, nStates);
    }

    // Default: prefer numeric.js when available, otherwise use our implementation.
    if (typeof numeric !== 'undefined' && typeof numeric.solve === 'function') {
        return solveLinearSystem_numericjs(K, rhs, nStates);
    }
    return solveLinearSystem_our(K, rhs, nStates);
}

function computeTipCurrent(Es, Ri, rho, nStates, NSites, muT, gammaT0, Rtip, useDecay, decay, kT) {
    const w2 = [1, 2, 4, 8];
    let I_tip = 0.0;

    for (let s = 0; s < 16; s++) {
        if (s >= nStates) break;
        for (let i = 0; i < 4; i++) {
            if (i >= NSites) break;
            const occ = Math.floor((s / w2[i]) % 2);
            const r = Ri[i];
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
            const f = fermi(dE, kT);
            const G_forward = gammaT * f;
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
function pmeSiteOccupancy(rho, NSites, siteIdx) {
    const w2 = [1, 2, 4, 8];
    if (siteIdx < 0) siteIdx = 0;
    if (siteIdx >= NSites) siteIdx = NSites - 1;
    let occSite = 0.0;
    let nStates = 1;
    for (let k = 0; k < 4; k++) {
        if (k >= NSites) break;
        nStates *= 2;
    }
    for (let s = 0; s < 16; s++) {
        if (s >= nStates) break;
        const occ = Math.floor((s / w2[siteIdx]) % 2);
        occSite += rho[s] * occ;
    }
    return occSite;
}

function pmeTotalCharge(rho, NSites) {
    const w2 = [1, 2, 4, 8];
    let Q = 0.0;
    let nStates = 1;
    for (let k = 0; k < 4; k++) {
        if (k >= NSites) break;
        nStates *= 2;
    }
    for (let s = 0; s < 16; s++) {
        if (s >= nStates) break;
        let N = 0.0;
        for (let i = 0; i < 4; i++) {
            if (i >= NSites) break;
            const occ = Math.floor((s / w2[i]) % 2);
            N += occ;
        }
        Q += rho[s] * N;
    }
    return Q;
}

// Pure numerical 1D line scan between two tip positions p0 and p1.
// p0, p1: [x, y, z] endpoints in real space.
// nPts: number of sampling points along the line.
// sites: array of {x,y,z,E} site objects.
// params: central params object with uniform-style keys (uRtip, uVBias, uTemp, uGammaS, uGammaT, uDecay, uW, uZTip, uNSites, solver).
// NSites, mode, siteIdx: prevalidated scalars.
// Returns { xs: [], Ys: [] } where xs are t in [0,1] and Ys are the selected observable.
function runCpuLineScanNumeric(p0, p1, nPts, sites, params, NSites, mode, siteIdx) {
    var xs = new Array(nPts);
    var Ys = new Array(nPts);

    var dx = p1[0] - p0[0];
    var dy = p1[1] - p0[1];
    var dz = p1[2] - p0[2];

    for (var iPt = 0; iPt < nPts; iPt++) {
        var t = (nPts === 1) ? 0.0 : iPt / (nPts - 1);
        var tip = {
            x: p0[0] + t * dx,
            y: p0[1] + t * dy,
            z: params.ZTip
        };

        // Mutate the shared params object with geometry-dependent fields.
        params.tip = tip;
        params.sites = sites;
        params.NSites = NSites;

        var out = solvePmeCpu(params);

        var val;
        if (mode === 3) {
            val = pmeSiteOccupancy(out.rho, NSites, siteIdx);
        } else if (mode === 4) {
            val = pmeTotalCharge(out.rho, NSites);
        } else {
            val = out.I_tip;
        }

        xs[iPt] = t;
        Ys[iPt] = val;
    }

    console.log('runCpuLineScanNumeric:', {
        solver: params.solver,
        mode: mode,
        NSites: NSites,
        nPts: nPts,
        YsSample: Ys.slice(0, Math.min(5, Ys.length))
    });

    return { xs: xs, Ys: Ys };
}

// High-level: solve PME for a single tip point and return debug info
function solvePmeCpu(params) {
    const {
        tip,          // {x,y,z}
        sites,        // array of {x,y,z,E}, length <= 4
        NSites,
        Rtip,
        VBias,
        TempK,
        GammaS,
        GammaT,
        decay,
        W
    } = params;

    if (getPmeVerbosity() >= 3) {
        console.log('solvePmeCpu params:', {
            NSites: NSites,
            sites: sites,
            tip: tip,
            Rtip: Rtip,
            VBias: VBias,
            TempK: TempK,
            GammaS: GammaS,
            GammaT: GammaT,
            decay: decay,
            W: W,
            solver: params.solver
        });
    }

    const Rscale = Math.max(Rtip, 1.0);
    const { Ei, Ri } = computeSiteEnergies(tip, sites, NSites, Rscale, VBias);

    const { Es, Ne, nStates } = computeManyBodyEnergies(Ei, NSites, W);

    const kBoltz = 8.617333262e-5; // eV/K
    const kT = Math.max(TempK * kBoltz, 1e-6);
    const muS = 0.0;
    const muT = VBias;
    const gammaS0 = Math.max(GammaS, 0.0);
    const gammaT0 = Math.max(GammaT, 0.0);

    const useDecay = true; // mirror planned decay-based tunneling

    let K = buildPmeKernel(Es, Ri, NSites, nStates, muS, muT, gammaS0, gammaT0, kT, Rtip, useDecay, decay);

    const rhs = new Array(16).fill(0.0);
    const normRow = nStates - 1;
    for (let j = 0; j < 16; j++) {
        if (j < nStates) K[normRow * 16 + j] = 1.0;
        else K[normRow * 16 + j] = 0.0;
    }
    for (let i = 0; i < nStates; i++) rhs[i] = 0.0;
    rhs[normRow] = 1.0;
    const solver = params.solver || 'numericjs';
    const rhoSol = solveLinearSystem_wraper(K, rhs, nStates, solver);
    const rho = new Array(16).fill(0.0);
    for (let s = 0; s < 16; s++) {
        rho[s] = (s < nStates) ? rhoSol[s] : 0.0;
    }

    const I_tip = computeTipCurrent(Es, Ri, rho, nStates, NSites, muT, gammaT0, Rtip, useDecay, decay, kT);
    if (getPmeVerbosity() >= 3) {
        console.log('solvePmeCpu debug: solver=' + solver + ' NSites=' + NSites + ' nStates=' + nStates + ' I_tip=' + I_tip);
    }
    dbgPrintMatrix('  K', K, nStates);
    dbgPrintVector('  rhs', rhs, nStates);
    dbgPrintVector('  rho', rho, nStates);

    return { Ei, Ri, Es, Ne, nStates, K, rhs: rhoSol, rho, I_tip };
}

// Convenience debug entry: call from console with current GUI-like params
// Example usage in browser devtools:
//   const tip = {x: 0, y: 0, z: 8};
//   const sites = [
//     {x: -5.785, y: 0, z: 0, E: -0.1},
//     {x:  5.785, y: 0, z: 0, E: -0.1}
//   ];
//   pauliPmeDebug({ tip, sites, NSites: 2, Rtip: 4.0, VBias: 0.3, TempK: 3.0, GammaS: 0.01, GammaT: 0.01, decay: 0.3, W: 0.0 });

function pauliPmeDebug(params) {
    const out = solvePmeCpu(params);
    if (getPmeVerbosity() >= 2) {
        console.log('PME CPU debug result:', out);
    }
    return out;
}

// -----------------------------------------------------------------------------
// 2D CPU PME scans (XY and XV) mirroring the GLSL shader logic
// -----------------------------------------------------------------------------

// Evaluate a single PME observable at a given tip position and bias, using the
// same parameters/observables as the shader. This reuses solvePmeCpu and the
// existing pmeSiteOccupancy / pmeTotalCharge helpers.
function evalPmeAtTipAndBias(tip, VBias, sites, params, NSites, mode, siteIdx) {
    const p = Object.assign({}, params, {
        tip: tip,
        sites: sites,
        NSites: NSites,
        VBias: VBias
    });

    const out = solvePmeCpu(p);

    if (mode === 3) {
        return pmeSiteOccupancy(out.rho, NSites, siteIdx);
    } else if (mode === 4) {
        return pmeTotalCharge(out.rho, NSites);
    } else {
        // Default: PME tip current, matching shader uMode==5
        return out.I_tip;
    }
}

// 2D XY scan: map a regular grid in uv \in [0,1]^2 to real-space tip positions
// using the same mapping as the shader:
//   tip = ((2*u-1)*L, (2*v-1)*L, ZTip),  VBias = params.VBias
// Returns { xs, ys, Z } where Z[iy][ix] is the scalar observable.
function runCpuScan2D_XY(nx, ny, sites, params, NSites, mode, siteIdx) {
    if (nx < 1 || ny < 1) return { xs: [], ys: [], Z: [] };

    const L = params.L;
    const ZTip = params.ZTip;
    const VBias = params.VBias;

    const xs = new Array(nx);
    const ys = new Array(ny);
    const Z  = new Array(ny);

    for (let iy = 0; iy < ny; iy++) {
        // Center of pixel in [0,1]
        const v = (iy + 0.5) / ny;
        ys[iy] = v;
        const row = new Array(nx);

        for (let ix = 0; ix < nx; ix++) {
            const u = (ix + 0.5) / nx;
            if (iy === 0) xs[ix] = u;

            const x = (u * 2.0 - 1.0) * L;
            const y = (v * 2.0 - 1.0) * L;
            const tip = { x: x, y: y, z: ZTip };

            row[ix] = evalPmeAtTipAndBias(tip, VBias, sites, params, NSites, mode, siteIdx);
        }
        Z[iy] = row;
    }

    return { xs: xs, ys: ys, Z: Z };
}

// 2D XV scan: x-axis is position along P1-P2 (t \in [0,1]), y-axis is bias
// between VBiasMin and VBiasMax, mirroring the shader's uScanMode==1 branch.
//   pos2d = mix(P1, P2, t)
//   tip   = (pos2d.x, pos2d.y, ZTip)
//   VBias = mix(VBiasMin, VBiasMax, v)
// Returns { ts, Vs, Z } where Z[iy][ix] corresponds to (t, V).
function runCpuScan2D_XV(nx, ny, sites, params, NSites, mode, siteIdx) {
    if (nx < 1 || ny < 1) return { ts: [], Vs: [], Z: [] };

    const ZTip     = params.ZTip;
    const P1x      = params.P1x;
    const P1y      = params.P1y;
    const P2x      = params.P2x;
    const P2y      = params.P2y;
    const VBiasMin = params.VBiasMin;
    const VBiasMax = params.VBiasMax;

    const ts = new Array(nx);   // position parameter t in [0,1]
    const Vs = new Array(ny);   // bias values
    const Z  = new Array(ny);

    for (let iy = 0; iy < ny; iy++) {
        const v = (iy + 0.5) / ny;
        const VBias = VBiasMin + (VBiasMax - VBiasMin) * v;
        Vs[iy] = VBias;
        const row = new Array(nx);

        for (let ix = 0; ix < nx; ix++) {
            const t = (ix + 0.5) / nx;
            if (iy === 0) ts[ix] = t;

            const px = P1x + (P2x - P1x) * t;
            const py = P1y + (P2y - P1y) * t;
            const tip = { x: px, y: py, z: ZTip };

            row[ix] = evalPmeAtTipAndBias(tip, VBias, sites, params, NSites, mode, siteIdx);
        }
        Z[iy] = row;
    }

    return { ts: ts, Vs: Vs, Z: Z };
}
