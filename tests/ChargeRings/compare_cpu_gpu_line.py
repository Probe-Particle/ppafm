#!/usr/bin/python
import os
import sys
import argparse
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from pyProbeParticle import pauli
from pyProbeParticle import pauli_ocl
import pauli_scan

kBoltz = 8.617333262e-5


def _build_Wij_matrix(spos, params):
    Wij_matrix = params.get('Wij_matrix', None)
    if Wij_matrix is not None:
        return np.ascontiguousarray(np.array(Wij_matrix, dtype=np.float64))
    W0 = float(params.get('W', 0.0))
    if W0 == 0.0:
        n = int(spos.shape[0])
        return np.zeros((n, n), dtype=np.float64)
    Wij_file = params.get('Wij_file', None)
    if Wij_file:
        Wij = np.loadtxt(Wij_file)
        return np.ascontiguousarray(Wij, dtype=np.float64)
    use_distance = bool(params.get('bWijDistance', False))
    mode = params.get('Wij_mode', None)
    if use_distance or (mode is not None and mode != 'const'):
        mode = mode or 'dipole'
        beta = float(params.get('Wij_beta', 1.0))
        power = float(params.get('Wij_power', 3.0))
        Wij = pauli_scan.make_Wij_distance(spos, W=W0, mode=mode, beta=beta, power=power)
        return np.ascontiguousarray(Wij, dtype=np.float64)
    return pauli.setWijConstant(int(spos.shape[0]), pauli_solver=None, W0=W0)


def _default_params():
    return {
        'radius': 5.2,
        'phiRot': 1.3,
        'phi0_ax': 0.2,
        'VBias': 1.00,
        'Rtip': 3.0,
        'z_tip': 5.0,
        'zV0': -1.0,
        'zVd': 15.0,
        'zQd': 0.0,
        'Q0': 1.0,
        'Qzz': 10.0,
        'Esite': -0.090,
        'W': 0.02,
        'bWijDistance': False,
        'Temp': 3.0,
        'decay': 0.3,
        'GammaS': 0.01,
        'GammaT': 0.01,
        'Et0': 0.2,
        'wt': 8.0,
        'At': 0.0,
        'c_orb': 1.0,
        'T0': 1.0,
        'L': 20.0,
        'npix': 200,
        'dQ': 0.02,
        'nsite': 4,
        'bMirror': True,
        'bRamp': True,
        'p1_x': 9.72,
        'p1_y': -9.96,
        'p2_x': -11.0,
        'p2_y': 12.0,
        'geometry_file': os.path.join(_REPO_ROOT, 'tests', 'ChargeRings', 'Ruslan_kite.txt'),
    }


def run_cpu(params, spos, rots, pTips, Vtips, cs, order, state_order):
    nsite = int(params['nsite'])
    solver_mode = int(params.get('solver_mode', 0))
    verbosity = int(params.get('verbosity', 0))
    ps = pauli.PauliSolver(nSingle=nsite, nleads=2, verbosity=verbosity)
    ps.setLinSolver(1, 50, 1e-12, solver_mode)
    T_eV = float(params.get('Temp', 0.0)) * kBoltz
    ps.set_lead(0, 0.0, T_eV)
    ps.set_lead(1, 0.0, T_eV)
    ps.set_check_prob_stop(bCheckProb=False, bCheckProbStop=False, CheckProbTol=1e-12)
    pauli_scan._apply_wij_config(ps, spos, params)
    cpp_params = pauli.make_cpp_params(params)
    current, *_ = ps.scan_current_tip(
        pTips,
        Vtips,
        spos,
        cpp_params,
        int(order),
        cs,
        state_order,
        rots=rots,
        bOmp=False,
        bMakeArrays=False,
        Ts=None,
        return_probs=False,
        return_state_energies=False,
    )
    return np.asarray(current, dtype=np.float64)


def run_gpu(params, spos, rots, pTips, Vtips, cs, order, Wij=None):
    nsite = int(params['nsite'])
    verbosity = int(params.get('verbosity', 0))
    pcl = pauli_ocl.PauliSolverCL(nSingle=nsite, nLeads=2, verbosity=verbosity)
    T_eV = float(params.get('Temp', 0.0)) * kBoltz
    pcl.set_lead(0, 0.0, T_eV)
    pcl.set_lead(1, 0.0, T_eV)
    cpp_params = pauli.make_cpp_params(params).astype(np.float32)
    currents, *_ = pcl.scan_current_tip(
        pTips.astype(np.float32),
        Vtips.astype(np.float32),
        spos.astype(np.float32),
        cpp_params,
        int(order),
        np.asarray(cs, dtype=np.float32),
        rots=rots.astype(np.float32) if rots is not None else None,
        return_probs=False,
        return_state_energies=False,
        Wij=Wij.astype(np.float32) if Wij is not None else None,
    )
    return np.asarray(currents, dtype=np.float64)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--nx', type=int, default=200)
    ap.add_argument('--VBias', type=float, default=1.00)
    ap.add_argument('--verbosity', type=int, default=0)
    ap.add_argument('--geometry', type=str, default='')
    ap.add_argument('--p1_x', type=float, default=np.nan)
    ap.add_argument('--p1_y', type=float, default=np.nan)
    ap.add_argument('--p2_x', type=float, default=np.nan)
    ap.add_argument('--p2_y', type=float, default=np.nan)
    args = ap.parse_args()

    params = _default_params()
    params['VBias'] = float(args.VBias)
    params['verbosity'] = int(args.verbosity)
    if args.geometry:
        params['geometry_file'] = args.geometry
    if np.isfinite(args.p1_x): params['p1_x'] = float(args.p1_x)
    if np.isfinite(args.p1_y): params['p1_y'] = float(args.p1_y)
    if np.isfinite(args.p2_x): params['p2_x'] = float(args.p2_x)
    if np.isfinite(args.p2_y): params['p2_y'] = float(args.p2_y)

    spos, rots, _angles = pauli_scan.make_site_geom(params)
    if spos.shape[1] >= 4:
        spos = np.array(spos, dtype=np.float64, copy=True)
        spos[:, 3] = float(params.get('Esite', 0.0))

    zT = float(params['z_tip']) + float(params['Rtip'])
    pTips, _, dist = pauli_scan.make_pTips_line(
        (float(params['p1_x']), float(params['p1_y'])),
        (float(params['p2_x']), float(params['p2_y'])),
        int(args.nx),
        zT=zT,
    )
    if not hasattr(dist, '__len__'):
        dist = pTips[:, 0]
    Vtips = np.full((len(pTips),), float(params['VBias']), dtype=np.float64)

    cs, order = pauli.make_quadrupole_Coeffs(float(params['Q0']), float(params['Qzz']))
    state_order = pauli.make_state_order(int(params['nsite']))
    Wij = _build_Wij_matrix(spos, params)

    cur_cpu = run_cpu(params, spos, rots, pTips, Vtips, cs, order, state_order)
    cur_gpu = run_gpu(params, spos, rots, pTips, Vtips, cs, order, Wij=Wij)

    diff = cur_gpu - cur_cpu
    max_abs = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))

    print(f"nx={len(pTips)} VBias={params['VBias']} p1=({params['p1_x']:.6f},{params['p1_y']:.6f}) p2=({params['p2_x']:.6f},{params['p2_y']:.6f})")
    print(f"diff: max_abs={max_abs:.6e} mean_abs={mean_abs:.6e}")
    print("x I_CPU I_GPU I_diff")
    for x, Ic, Ig, Id in zip(dist, cur_cpu, cur_gpu, diff):
        print(f"{x:.6e} {Ic:.6e} {Ig:.6e} {Id:.6e}")
