#!/usr/bin/python

import os
import sys
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from pyProbeParticle import pauli
from pyProbeParticle import pauli_ocl
import pauli_scan

kBoltz = 8.617333262e-5  # eV/K


def _stats(name, a):
    a = np.asarray(a)
    print(f"{name}: shape={a.shape} min={a.min():.6e} max={a.max():.6e} mean={a.mean():.6e} std={a.std():.6e}")


def _diff_stats(name, a, b):
    d = np.asarray(b) - np.asarray(a)
    _stats(name + " diff", d)
    print(f"{name} diff: max_abs={np.max(np.abs(d)):.6e} mean_abs={np.mean(np.abs(d)):.6e}")


def _make_xy(params):
    npix = int(params['npix'])
    L = float(params['L'])
    zT = float(params['z_tip']) + float(params['Rtip'])
    coords = (np.arange(npix) + 0.5 - npix / 2) * (2 * L / npix)
    xx, yy = np.meshgrid(coords, coords)
    pTips = np.zeros((npix * npix, 3), dtype=np.float64)
    pTips[:, 0] = xx.ravel()
    pTips[:, 1] = yy.ravel()
    pTips[:, 2] = zT
    Vtips = np.full((pTips.shape[0],), float(params['VBias']), dtype=np.float64)
    return pTips, Vtips


def _make_xv(params):
    p1 = (float(params['p1_x']), float(params['p1_y']))
    p2 = (float(params['p2_x']), float(params['p2_y']))
    nx = int(params.get('nx', 120))
    nV = int(params.get('nV', 120))
    zT = float(params['z_tip']) + float(params['Rtip'])
    pTips_line, _, _dist = pauli_scan.make_pTips_line(p1, p2, nx, zT=zT)
    Vbiases = np.linspace(0.0, float(params['VBias']), nV).astype(np.float64)
    pTips = np.tile(pTips_line, (nV, 1)).astype(np.float64)
    Vtips = np.repeat(Vbiases, nx).astype(np.float64)
    return pTips, Vtips, (nV, nx)


def _configure_common(params, solver_cpu, solver_ocl=None):
    T_eV = float(params['Temp']) * kBoltz
    solver_cpu.set_lead(0, 0.0, T_eV)
    solver_cpu.set_lead(1, 0.0, T_eV)
    if solver_ocl is not None:
        solver_ocl.set_lead(0, 0.0, T_eV)
        solver_ocl.set_lead(1, 0.0, T_eV)


def _run_case(params, use_ocl=False):
    # geometry
    spos, rots, _angles = pauli_scan.make_site_geom(params)
    if spos.shape[1] >= 4:
        spos = np.array(spos, dtype=np.float64, copy=True)
        spos[:, 3] = float(params.get('Esite', 0.0))

    cs, order = pauli.make_quadrupole_Coeffs(float(params['Q0']), float(params['Qzz']))
    state_order = pauli.make_state_order(int(params['nsite']))
    cpp_params = pauli.make_cpp_params(params)

    solver_cpu = pauli.PauliSolver(nSingle=int(params['nsite']), nleads=2, verbosity=0)
    solver_cpu.set_check_prob_stop(bCheckProb=False, bCheckProbStop=False, CheckProbTol=1e-12)
    solver_cpu.setLinSolver(1, 50, 1e-12, 0)

    solver_ocl = None
    if use_ocl:
        solver_ocl = pauli_ocl.PauliSolverCL(nSingle=int(params['nsite']), nLeads=2, verbosity=0)

    _configure_common(params, solver_cpu, solver_ocl)

    # Match reference pipeline: configure Wij on CPU solver
    pauli_scan._apply_wij_config(solver_cpu, spos, params)
    Wij = None
    if use_ocl:
        # Build the same Wij as CPU uses (constant/distance/file/matrix) and pass to OpenCL.
        Wij_matrix = params.get('Wij_matrix', None)
        if Wij_matrix is not None:
            Wij = np.ascontiguousarray(np.array(Wij_matrix, dtype=np.float32))
        else:
            W0 = float(params.get('W', 0.0))
            if W0 == 0.0:
                Wij = np.zeros((int(params['nsite']), int(params['nsite'])), dtype=np.float32)
            else:
                Wij_file = params.get('Wij_file', None)
                if Wij_file:
                    Wij = np.ascontiguousarray(np.loadtxt(Wij_file), dtype=np.float32)
                else:
                    use_distance = bool(params.get('bWijDistance', False))
                    mode = params.get('Wij_mode', None)
                    if use_distance or (mode is not None and mode != 'const'):
                        mode = mode or 'dipole'
                        beta = float(params.get('Wij_beta', 1.0))
                        power = float(params.get('Wij_power', 3.0))
                        Wij = pauli_scan.make_Wij_distance(spos, W=W0, mode=mode, beta=beta, power=power).astype(np.float32)
                    else:
                        Wij = pauli.setWijConstant(int(params['nsite']), pauli_solver=None, W0=W0).astype(np.float32)

    # XY
    pTips_xy, Vtips_xy = _make_xy(params)
    if use_ocl:
        cur_xy, *_ = solver_ocl.scan_current_tip(
            pTips_xy.astype(np.float32),
            Vtips_xy.astype(np.float32),
            spos.astype(np.float32),
            cpp_params.astype(np.float32),
            int(order),
            np.asarray(cs, dtype=np.float32),
            return_probs=False,
            return_state_energies=False,
            Wij=Wij,
        )
        cur_xy = cur_xy.reshape(int(params['npix']), int(params['npix']))
    else:
        cur_xy, *_ = solver_cpu.scan_current_tip(
            pTips_xy,
            Vtips_xy,
            spos,
            cpp_params,
            int(order),
            cs,
            state_order,
            rots=rots,
            bOmp=False,
            bMakeArrays=False,
            return_probs=False,
            return_state_energies=False,
        )
        cur_xy = cur_xy.reshape(int(params['npix']), int(params['npix']))

    # xV
    pTips_xv, Vtips_xv, shape_xv = _make_xv(params)
    if use_ocl:
        cur_xv, *_ = solver_ocl.scan_current_tip(
            pTips_xv.astype(np.float32),
            Vtips_xv.astype(np.float32),
            spos.astype(np.float32),
            cpp_params.astype(np.float32),
            int(order),
            np.asarray(cs, dtype=np.float32),
            return_probs=False,
            return_state_energies=False,
            Wij=Wij,
        )
        cur_xv = cur_xv.reshape(*shape_xv)
    else:
        cur_xv, *_ = solver_cpu.scan_current_tip(
            pTips_xv,
            Vtips_xv,
            spos,
            cpp_params,
            int(order),
            cs,
            state_order,
            rots=rots,
            bOmp=False,
            bMakeArrays=False,
            return_probs=False,
            return_state_energies=False,
        )
        cur_xv = cur_xv.reshape(*shape_xv)

    return cur_xy, cur_xv


if __name__ == "__main__":
    # Start from the known-working Ruslan_kite defaults used by compare_pme_solvers
    base_params_path = os.path.join(
        os.path.dirname(__file__),
        'results_test_xy/NTCDA_xy/4site_Ruslan_kite/solver_0/W_0.020/VBias_1.00/params.json'
    )
    params = pauli_scan.load_json_params(base_params_path)

    # Ensure required GUI-like keys exist
    params.setdefault('p1_x', -10.0)
    params.setdefault('p1_y', 0.0)
    params.setdefault('p2_x', 10.0)
    params.setdefault('p2_y', 0.0)
    params.setdefault('nx', 120)
    params.setdefault('nV', 120)

    paramsA = dict(params)
    paramsB = dict(params)

    # Test knobs
    paramsA['W'] = float(params.get('W', 0.02))
    paramsB['W'] = paramsA['W'] * 2.0

    paramsA['Esite'] = float(params.get('Esite', -0.090))
    paramsB['Esite'] = paramsA['Esite'] + 0.02

    # Ensure site count
    paramsA['nsite'] = int(paramsA.get('nsite', 4))
    paramsB['nsite'] = int(paramsB.get('nsite', 4))

    print("=== CPU compare ===")
    xyA, xvA = _run_case(paramsA, use_ocl=False)
    xyB, xvB = _run_case(paramsB, use_ocl=False)
    _stats("CPU XY A", xyA)
    _stats("CPU XY B", xyB)
    _diff_stats("CPU XY", xyA, xyB)
    _stats("CPU xV A", xvA)
    _stats("CPU xV B", xvB)
    _diff_stats("CPU xV", xvA, xvB)

    print("\n=== OpenCL compare ===")
    xyA, xvA = _run_case(paramsA, use_ocl=True)
    xyB, xvB = _run_case(paramsB, use_ocl=True)
    _stats("OCL XY A", xyA)
    _stats("OCL XY B", xyB)
    _diff_stats("OCL XY", xyA, xyB)
    _stats("OCL xV A", xvA)
    _stats("OCL xV B", xvB)
    _diff_stats("OCL xV", xvA, xvB)
