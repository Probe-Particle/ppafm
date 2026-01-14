import json
import os
import sys
import numpy as np

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), '../..')))

from pyProbeParticle import pauli
from pyProbeParticle import pauli_ocl
from pyProbeParticle import utils as ut


def load_params(path):
    with open(path, 'r') as f:
        return json.load(f)


def load_geometry(path, default_E=0.0):
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"Geometry file '{path}' must have >=3 columns")

    n = data.shape[0]
    spos = np.zeros((n, 4), dtype=np.float64)
    spos[:, 0:2] = data[:, 0:2]

    angles_deg = data[:, 2]
    angles_rad = np.radians(angles_deg)

    if data.shape[1] >= 4:
        spos[:, 3] = data[:, 3]
    else:
        spos[:, 3] = float(default_E)

    rots = ut.makeRotMats(angles_rad, nsite=n)
    return spos, rots


def make_line_tips(params):
    npix = int(params['npix'])
    x1, y1 = float(params['p1_x']), float(params['p1_y'])
    x2, y2 = float(params['p2_x']), float(params['p2_y'])

    zT = float(params['z_tip']) + float(params['Rtip'])

    xs = np.linspace(x1, x2, npix)
    ys = np.linspace(y1, y2, npix)

    pTips = np.zeros((npix, 3), dtype=np.float64)
    pTips[:, 0] = xs
    pTips[:, 1] = ys
    pTips[:, 2] = zT

    return pTips


def run_cpu(params, spos, rots, pTips, Vtips, cs, order, state_order):
    nsite = int(params['nsite'])
    solver_mode = int(params.get('solver_mode', 0))
    verbosity = int(params.get('verbosity', 0))

    ps = pauli.PauliSolver(nSingle=nsite, nleads=2, verbosity=verbosity)
    ps.setLinSolver(1, 50, 1e-12, solver_mode)

    T_eV = float(params.get('Temp', 0.0)) * 8.617333262e-5
    ps.set_lead(0, 0.0, T_eV)
    ps.set_lead(1, 0.0, T_eV)

    ps.set_check_prob_stop(bCheckProb=False, bCheckProbStop=False, CheckProbTol=1e-12)

    cpp_params = pauli.make_cpp_params(params)

    current, Es, Ts, Probs, StateEs = ps.scan_current_tip(
        pTips,
        Vtips,
        spos,
        cpp_params,
        int(order),
        cs,
        state_order,
        rots=rots,
        bOmp=False,
        bMakeArrays=True,
        Ts=None,
        return_probs=True,
        return_state_energies=True,
    )

    return current, Es, Ts, Probs, StateEs


def run_cpu_single_with_matrix(params, spos, rots, pTip, Vtip, cs, order, state_order):
    nsite = int(params['nsite'])
    solver_mode = int(params.get('solver_mode', 0))
    verbosity = int(params.get('verbosity', 0))

    ps = pauli.PauliSolver(nSingle=nsite, nleads=2, verbosity=verbosity)
    ps.setLinSolver(1, 50, 1e-12, solver_mode)
    T_eV = float(params.get('Temp', 0.0)) * 8.617333262e-5
    ps.set_lead(0, 0.0, T_eV)
    ps.set_lead(1, 0.0, T_eV)
    ps.set_check_prob_stop(bCheckProb=False, bCheckProbStop=False, CheckProbTol=1e-12)
    cpp_params = pauli.make_cpp_params(params)

    nstates = 2 ** int(nsite)
    K_flat = np.zeros((nstates * nstates,), dtype=np.float64)
    pauli.set_current_matrix_export_pointer(K_flat)

    current, Es, Ts, Probs, StateEs = ps.scan_current_tip(
        np.asarray(pTip, dtype=np.float64).reshape(1, 3),
        np.asarray([Vtip], dtype=np.float64),
        spos,
        cpp_params,
        int(order),
        cs,
        state_order,
        rots=rots,
        bOmp=False,
        bMakeArrays=True,
        Ts=None,
        return_probs=True,
        return_state_energies=True,
    )

    CurMat = K_flat.reshape(nstates, nstates)

    # Export Pauli rate kernel matrix (pre-normalization row; C++ applies normalization in solve_kern via kern_copy)
    K_rate = ps.get_kernel(nstates)
    # Make it consistent with GPU export which includes normalization row
    K_rate_norm = K_rate.copy()
    K_rate_norm[0, :] = 1.0

    return (
        float(current[0]),
        Es.reshape(nsite),
        Ts.reshape(nsite),
        Probs.reshape(nstates),
        StateEs.reshape(nstates),
        CurMat,
        K_rate_norm,
    )


def run_gpu(params, spos, pTips, Vtips, cs, order):
    nsite = int(params['nsite'])

    pcl = pauli_ocl.PauliSolverCL(nSingle=nsite, nLeads=2, verbosity=int(params.get('verbosity', 0)))

    T_eV = float(params.get('Temp', 0.0)) * 8.617333262e-5
    VBias = float(params['VBias'])
    pcl.set_lead(0, 0.0, T_eV)
    pcl.set_lead(1, VBias, T_eV)

    cpp_params = pauli.make_cpp_params(params)

    currents, Es, Ts, Probs, StateEs, K, CurMat = pcl.scan_current_tip(
        pTips,
        Vtips,
        spos,
        cpp_params.astype(np.float32),
        int(order),
        np.asarray(cs, dtype=np.float32),
        return_probs=True,
        return_state_energies=True,
        return_curmat=(len(pTips) == 1),
    )

    return currents, Es, Ts, Probs, StateEs, K, CurMat


def print_stats(name, arr):
    arr = np.asarray(arr)
    print(f"{name}: vmin={arr.min():.6e} vmax={arr.max():.6e}")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('params', nargs='?', default=None)
    ap.add_argument('--tol', type=float, default=1e-6)
    ap.add_argument('--debug', action='store_true')
    ap.add_argument('--ip', type=int, default=None)
    ap.add_argument('--dumpK', action='store_true')
    args = ap.parse_args()

    if args.params is None:
        args.params = os.path.join(
            os.path.dirname(__file__),
            'results_test_xy/NTCDA_xy/4site_Ruslan_kite/solver_0/W_0.020/VBias_1.00/params.json',
        )
        args.params = os.path.normpath(args.params)

    params = load_params(args.params)

    geom = params.get('geometry_file', None)
    if geom is None:
        raise ValueError('params.json must contain geometry_file for this runner')

    spos, rots = load_geometry(geom, default_E=float(params.get('Esite', 0.0)))
    params['nsite'] = int(params.get('nsite', spos.shape[0]))

    pTips = make_line_tips(params)

    npix = int(params['npix'])
    VBias = float(params['VBias'])
    Vtips = np.full(npix, VBias, dtype=np.float64)

    cs, order = pauli.make_quadrupole_Coeffs(float(params['Q0']), float(params['Qzz']))
    state_order = pauli.make_state_order(int(params['nsite']))

    print(f"params={args.params}")
    print(f"nsite={params['nsite']} npix={npix} VBias={VBias} Temp[K]={params.get('Temp', None)}")

    print("-- CPU (C++ PauliSolver) --")
    cur_cpu, Es_cpu, Ts_cpu, Probs_cpu, StateEs_cpu = run_cpu(params, spos, rots, pTips, Vtips, cs, order, state_order)
    print_stats('CPU current', cur_cpu)

    ip_def = int(np.argmax(np.abs(cur_cpu)))

    print("-- GPU (OpenCL) --")
    cur_gpu, Es_gpu, Ts_gpu, Probs_gpu, StateEs_gpu, K_gpu, CurMat_gpu = run_gpu(params, spos, pTips.astype(np.float32), Vtips.astype(np.float32), cs, order)
    print_stats('GPU current', cur_gpu)

    diff = np.asarray(cur_gpu) - np.asarray(cur_cpu)
    print_stats('diff (GPU-CPU)', diff)
    print(f"diff: max_abs={np.max(np.abs(diff)):.6e} mean_abs={np.mean(np.abs(diff)):.6e}")

    if (np.max(np.abs(diff)) > args.tol) or args.debug or args.dumpK:
        print("-- diagnostics --")
        if Es_cpu is not None:
            print(f"CPU Es: min={Es_cpu.min():.6e} max={Es_cpu.max():.6e}")
        if Ts_cpu is not None:
            print(f"CPU Ts: min={Ts_cpu.min():.6e} max={Ts_cpu.max():.6e}")
        if Es_gpu is not None:
            print(f"GPU Es(shifts): min={Es_gpu.min():.6e} max={Es_gpu.max():.6e}")
        if Ts_gpu is not None:
            print(f"GPU Ts(factors): min={Ts_gpu.min():.6e} max={Ts_gpu.max():.6e}")

        if Probs_gpu is not None:
            print(f"GPU Probs: min={Probs_gpu.min():.6e} max={Probs_gpu.max():.6e}")
        if StateEs_gpu is not None:
            print(f"GPU StateEs: min={StateEs_gpu.min():.6e} max={StateEs_gpu.max():.6e}")

        if Probs_cpu is not None:
            print(f"CPU Probs: min={Probs_cpu.min():.6e} max={Probs_cpu.max():.6e}")
            ip0 = ip_def if args.ip is None else int(args.ip)
            print(f"CPU Probs[ip={ip0}]:")
            print(np.asarray(Probs_cpu[ip0]))

            if Probs_gpu is not None:
                print(f"GPU Probs[ip={ip0}]:")
                print(np.asarray(Probs_gpu[ip0]))
                print(f"Probs diff (GPU-CPU) ip={ip0}: min={float((Probs_gpu[ip0]-Probs_cpu[ip0]).min()):.6e} max={float((Probs_gpu[ip0]-Probs_cpu[ip0]).max()):.6e}")

        if StateEs_cpu is not None:
            print(f"CPU StateEs: min={StateEs_cpu.min():.6e} max={StateEs_cpu.max():.6e}")

        # Optional kernel matrix dump for one pixel
        if args.dumpK:
            ip1 = ip_def if args.ip is None else int(args.ip)
            print(f"-- current contribution matrix dump (single pixel ip={ip1}) --")
            cur1, Es1, Ts1, P1, SE1, CurMat_cpu, Kcpu_rate = run_cpu_single_with_matrix(
                params, spos, rots, pTips[ip1], Vtips[ip1], cs, order, state_order
            )

            cur1g, Es1g, Ts1g, P1g, SE1g, K_g, CurMat_g = run_gpu(
                params,
                spos,
                pTips[ip1:ip1+1].astype(np.float32),
                Vtips[ip1:ip1+1].astype(np.float32),
                cs,
                order,
            )
            K_gpu1 = K_g[0] if K_g is not None else None
            CurMat_gpu1 = CurMat_g[0] if CurMat_g is not None else None

            print(f"CPU(single) current={cur1:.6e} GPU(single) current={float(cur1g[0]):.6e}")
            print(f"CPU CurMat: min={CurMat_cpu.min():.6e} max={CurMat_cpu.max():.6e}")
            if CurMat_gpu1 is None:
                print("GPU CurMat: None (not returned)")
            else:
                print(f"GPU CurMat: min={CurMat_gpu1.min():.6e} max={CurMat_gpu1.max():.6e}")
                dK = CurMat_gpu1 - CurMat_cpu
                print(f"dCurMat (GPU-CPU): min={dK.min():.6e} max={dK.max():.6e} max_abs={np.max(np.abs(dK)):.6e}")

            if K_gpu1 is not None:
                print(f"CPU K(rate,norm): min={Kcpu_rate.min():.6e} max={Kcpu_rate.max():.6e}")
                print(f"GPU K(export):    min={K_gpu1.min():.6e} max={K_gpu1.max():.6e}")
                dKr = K_gpu1 - Kcpu_rate
                print(f"dKrate (GPU-CPU): min={dKr.min():.6e} max={dKr.max():.6e} max_abs={np.max(np.abs(dKr)):.6e}")

            print(f"CPU StateEs[ip={ip1}]:")
            print(SE1)
            if SE1g is not None:
                print(f"GPU StateEs[ip={ip1}]:")
                print(SE1g[0])
                dSe = SE1g[0] - SE1
                print(f"dStateEs (GPU-CPU): min={dSe.min():.6e} max={dSe.max():.6e} max_abs={np.max(np.abs(dSe)):.6e}")

            if args.debug:
                print("CPU CurMat matrix:")
                print(CurMat_cpu)
                if CurMat_gpu1 is not None:
                    print("GPU CurMat matrix:")
                    print(CurMat_gpu1)

                if K_gpu1 is not None:
                    print("CPU K(rate,norm) matrix:")
                    print(Kcpu_rate)
                    print("GPU K(export) matrix:")
                    print(K_gpu1)


if __name__ == '__main__':
    main()
