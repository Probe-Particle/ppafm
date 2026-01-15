#!/usr/bin/python

import os
import sys
import numpy as np
import matplotlib; matplotlib.use('Qt5Agg')
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from GUITemplate import GUITemplate, PlotManager, PlotConfig

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)
from pyProbeParticle import pauli
from pyProbeParticle import pauli_ocl

import pauli_scan

kBoltz = 8.617333262e-5 # eV/K


def _safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except BrokenPipeError:
        # stdout closed (e.g. piped to head); avoid crashing the GUI
        pass


def _make_xy_grid(params):
    npix = int(params['npix'])
    L    = float(params['L'])
    zT   = float(params['z_tip']) + float(params['Rtip'])
    coords = np.linspace(-L, L, npix)
    xx, yy = np.meshgrid(coords, coords)
    pTips = np.zeros((npix*npix, 3), dtype=np.float64)
    pTips[:, 0] = xx.ravel()
    pTips[:, 1] = yy.ravel()
    pTips[:, 2] = zT
    extent = (-L, L, -L, L)
    return pTips, extent


def _make_line_tips(params):
    p1 = (float(params['p1_x']), float(params['p1_y']))
    p2 = (float(params['p2_x']), float(params['p2_y']))
    nx = int(params.get('nx', 100))
    zT = float(params['z_tip']) + float(params['Rtip'])
    pTips, _, dist = pauli_scan.make_pTips_line(p1, p2, nx, zT=zT)
    return pTips, dist


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


class ApplicationWindow(GUITemplate):
    def __init__(self):
        super().__init__("Pauli Fast GUI")

        self.nsite = 4

        self.param_specs = {
            'radius':        {'group': 'Geometry',     'widget': 'double', 'range': (1.0, 20.0),   'value': 5.2, 'step': 0.5},
            'phiRot':        {'group': 'Geometry',     'widget': 'double', 'range': (-10.0, 10.0), 'value': 1.3, 'step': 0.1},
            'phi0_ax':       {'group': 'Geometry',     'widget': 'double', 'range': (-3.14, 3.14), 'value': 0.2, 'step': 0.1},

            'VBias':         {'group': 'Electrostatic Field', 'widget': 'double', 'range': (-10.0, 10.0), 'value': 1.00, 'step': 0.02},
            'Rtip':          {'group': 'Electrostatic Field', 'widget': 'double', 'range': (0.5, 10.0),   'value': 3.0,  'step': 0.5},
            'z_tip':         {'group': 'Electrostatic Field', 'widget': 'double', 'range': (0.5, 20.0),   'value': 5.0,  'step': 0.5},
            'zV0':           {'group': 'Electrostatic Field', 'widget': 'double', 'range': (-10.0, 10.0), 'value': -1.0, 'step': 0.1},
            'zVd':           {'group': 'Electrostatic Field', 'widget': 'double', 'range': (-5.0, 50.0),  'value': 15.0, 'step': 0.1},
            'zQd':           {'group': 'Electrostatic Field', 'widget': 'double', 'range': (-5.0, 5.0),   'value': 0.0,  'step': 0.1},
            'Q0':            {'group': 'Electrostatic Field', 'widget': 'double', 'range': (-10.0, 10.0), 'value': 1.0,  'step': 0.1},
            'Qzz':           {'group': 'Electrostatic Field', 'widget': 'double', 'range': (-20.0, 20.0), 'value': 10.0, 'step': 0.5},

            'Esite':         {'group': 'Transport Solver',  'widget': 'double', 'range': (-1.0, 1.0),   'value': -0.090, 'step': 0.002, 'decimals': 3},
            'W':             {'group': 'Transport Solver',  'widget': 'double', 'range': (-100.0, 100.0), 'value': 0.02, 'step': 0.001, 'decimals': 3},
            'bWijDistance':  {'group': 'Transport Solver',  'widget': 'bool',   'value': False},
            'Temp':          {'group': 'Transport Solver',  'widget': 'double', 'range': (0.0, 100.0),  'value': 3.0, 'step': 0.05, 'decimals': 2},
            'decay':         {'group': 'Transport Solver',  'widget': 'double', 'range': (0.1, 2.0),    'value': 0.3, 'step': 0.1, 'decimals': 2},
            'GammaS':        {'group': 'Transport Solver',  'widget': 'double', 'range': (0.0, 1.0),    'value': 0.01, 'step': 0.001, 'decimals': 3, 'fidget': False},
            'GammaT':        {'group': 'Transport Solver',  'widget': 'double', 'range': (0.0, 1.0),    'value': 0.01, 'step': 0.001, 'decimals': 3, 'fidget': False},

            'Et0':           {'group': 'Barrier', 'widget': 'double', 'range': (0.0, 10.0),   'value': 0.2, 'step': 0.01},
            'wt':            {'group': 'Barrier', 'widget': 'double', 'range': (0.0, 20.0),   'value': 8.0, 'step': 0.1},
            'At':            {'group': 'Barrier', 'widget': 'double', 'range': (-10.0, 10.0), 'value': 0.0, 'step': 0.01},
            'c_orb':         {'group': 'Barrier', 'widget': 'double', 'range': (0.0, 1.0),    'value': 1.0, 'step': 0.0001, 'decimals': 4},
            'T0':            {'group': 'Barrier', 'widget': 'double', 'range': (0.0, 1000.0), 'value': 1.0, 'step': 0.0001, 'decimals': 4},

            'L':             {'group': 'Visualization', 'widget': 'double', 'range': (5.0, 50.0),   'value': 20.0, 'step': 1.0},
            'npix':          {'group': 'Visualization', 'widget': 'int',    'range': (50, 500),     'value': 200,  'step': 50},
            'dQ':            {'group': 'Visualization', 'widget': 'double', 'range': (0.001, 0.1),  'value': 0.02, 'step': 0.001, 'decimals': 3},
        }

        self.create_gui()

        # Geometry file (optional, but needed to match reference datasets)
        geom_layout = QtWidgets.QHBoxLayout()
        self.layout0.addLayout(geom_layout)
        geom_layout.addWidget(QtWidgets.QLabel("Geometry file:"))
        self.leGeometryFile = QtWidgets.QLineEdit()
        default_geom = os.path.join(_REPO_ROOT, 'tests', 'ChargeRings', 'Ruslan_kite.txt')
        self.leGeometryFile.setText(default_geom)
        geom_layout.addWidget(self.leGeometryFile)
        btnLoadGeom = QtWidgets.QPushButton("Load")
        btnLoadGeom.clicked.connect(self.load_geometry_file)
        geom_layout.addWidget(btnLoadGeom)
        self.geometry_file = None

        # fast-mode controls
        hb = QtWidgets.QHBoxLayout()
        self.layout0.addLayout(hb)

        self.cbUseOpenCL = QtWidgets.QCheckBox('OpenCL')
        self.cbUseOpenCL.setChecked(False)
        self.cbUseOpenCL.stateChanged.connect(self.on_parameter_change)
        hb.addWidget(self.cbUseOpenCL)

        self.cbShowdIdV = QtWidgets.QCheckBox('dI/dV')
        self.cbShowdIdV.setChecked(True)
        self.cbShowdIdV.stateChanged.connect(self.on_parameter_change)
        hb.addWidget(self.cbShowdIdV)

        # xV scan resolution (kept out of param_specs to avoid clutter)
        hb2 = QtWidgets.QHBoxLayout()
        self.layout0.addLayout(hb2)
        hb2.addWidget(QtWidgets.QLabel('nx:'))
        self.sbNx = QtWidgets.QSpinBox(); self.sbNx.setRange(20, 400); self.sbNx.setValue(120)
        self.sbNx.valueChanged.connect(self.on_parameter_change)
        hb2.addWidget(self.sbNx)
        hb2.addWidget(QtWidgets.QLabel('nV:'))
        self.sbNV = QtWidgets.QSpinBox(); self.sbNV.setRange(20, 400); self.sbNV.setValue(120)
        self.sbNV.valueChanged.connect(self.on_parameter_change)
        hb2.addWidget(self.sbNV)

        # Matplotlib figure
        self.fig = Figure(figsize=(10, 5))
        self.canvas = FigureCanvas(self.fig)
        self.main_widget.layout().insertWidget(0, self.canvas)

        self.ax_xy = self.fig.add_subplot(1, 2, 1)
        self.ax_xv = self.fig.add_subplot(1, 2, 2)

        self.pm = PlotManager(self.fig)
        self.pm.bUpdateLimits = False
        self.pm.bBlitIndividual = True
        self.pm.add_plot('xy', PlotConfig(ax=self.ax_xy, title='XY', xlabel='x [Å]', ylabel='y [Å]', cmap='bwr'))
        self.pm.add_plot('xv', PlotConfig(ax=self.ax_xv, title='xV', xlabel='x [Å]', ylabel='V [V]', cmap='bwr'))
        self.pm.initialize_plots()

        self.solver_cpu = pauli.PauliSolver(nSingle=self.nsite, nleads=2, verbosity=0)
        self.solver_cpu.set_check_prob_stop(bCheckProb=False, bCheckProbStop=False, CheckProbTol=1e-12)
        # Use Gauss solver (mode=1) with tolerances matching CombinedChargeRingsGUI_v5
        self.solver_cpu.setLinSolver(1, 50, 1e-12, 0)

        # Use the same default validity cuts as the reference C++ path.
        # These cuts affect whether a point is skipped (current forced to 0), so W must
        # influence them.
        pauli.set_valid_point_cuts(0.0, 2.0)
        pauli.bValidateProbabilities = False

        self.solver_ocl = None
        try:
            self.solver_ocl = pauli_ocl.PauliSolverCL(nSingle=self.nsite, nLeads=2, verbosity=0)
        except Exception as e:
            print('OpenCL init failed, falling back to CPU:', e)
            self.cbUseOpenCL.setChecked(False)

        self._last_xy = None
        self._last_xv = None

        self.run()

    def get_param_values(self):
        params = super().get_param_values()
        params['nsite'] = self.nsite
        params['bMirror'] = True
        params['bRamp'] = True
        params['nx'] = int(self.sbNx.value())
        params['nV'] = int(self.sbNV.value())
        # fixed line endpoints (defaults from old GUI)
        params.setdefault('p1_x', 9.72)
        params.setdefault('p1_y', -9.96)
        params.setdefault('p2_x', -11.0)
        params.setdefault('p2_y', 12.0)

        # If geometry loaded, use it (pauli_scan.make_site_geom will pick it up)
        gf = self.leGeometryFile.text().strip() if hasattr(self, 'leGeometryFile') else ''
        if gf:
            params['geometry_file'] = gf
        return params

    def load_geometry_file(self):
        path = self.leGeometryFile.text().strip()
        self.geometry_file = path if path else None
        _safe_print(f"[geom] geometry_file={self.geometry_file}")
        self.run()

    def _solve_xy_cpu(self, params, spos, rots, cs, return_ets=False):
        # Match CombinedChargeRingsGUI_v5 / pauli_scan.scan_xy_orb convention
        STM, Es, Ts, _Probs, _StateEs = pauli.run_pauli_scan_top(
            spos,
            rots,
            params,
            pauli_solver=self.solver_cpu,
            bOmp=False,
            cs=cs,
            Ts=None,
            state_order=None,
        )
        _safe_print(f"[XY cpu] Es(min,max)=({Es.min():.3e},{Es.max():.3e}) Ts(min,max)=({Ts.min():.3e},{Ts.max():.3e})")
        if return_ets:
            return STM, Es, Ts
        return STM

    def _solve_xy_ocl(self, params, pTips, Vtips, spos, rots, order, cs, Wij=None):
        cpp_params = pauli.make_cpp_params(params).astype(np.float32)
        cur, *_ = self.solver_ocl.scan_current_tip(
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
        return cur.reshape(int(params['npix']), int(params['npix']))

    def _solve_xv_cpu(self, params, pTips_line, Vbiases, spos, rots, order, cs):
        state_order = pauli.make_state_order(self.nsite)
        cpp_params = pauli.make_cpp_params(params)
        nV = len(Vbiases)
        nx = len(pTips_line)
        # Ordering must match Vtips: blocks of constant bias over all x positions
        # [ (x0..xN at V0), (x0..xN at V1), ... ]
        pTips = np.tile(pTips_line, (nV, 1))
        Vtips = np.repeat(Vbiases, nx).astype(np.float64)
        out = self.solver_cpu.scan_current_tip(
            pTips, Vtips, spos, cpp_params, int(order), cs, state_order, rots=rots,
            bOmp=False, bMakeArrays=False, Ts=None,
            return_probs=False, return_state_energies=False,
        )
        cur = out[0]
        return cur.reshape(nV, nx)

    def _solve_xv_ocl(self, params, pTips_line, Vbiases, spos, rots, order, cs, Wij=None):
        nV = len(Vbiases)
        nx = len(pTips_line)
        pTips = np.tile(pTips_line, (nV, 1))
        Vtips = np.repeat(Vbiases, nx).astype(np.float32)
        cpp_params = pauli.make_cpp_params(params).astype(np.float32)
        cur, *_ = self.solver_ocl.scan_current_tip(
            pTips.astype(np.float32),
            Vtips,
            spos.astype(np.float32),
            cpp_params,
            int(order),
            np.asarray(cs, dtype=np.float32),
            rots=rots.astype(np.float32) if rots is not None else None,
            return_probs=False,
            return_state_energies=False,
            Wij=Wij.astype(np.float32) if Wij is not None else None,
        )
        return cur.reshape(nV, nx)

    def run(self):
        params = self.get_param_values()

        # solver setup
        V0 = float(params['VBias'])
        use_ocl = bool(self.cbUseOpenCL.isChecked()) and (self.solver_ocl is not None)
        T_eV = float(params.get('Temp', 0.0)) * kBoltz
        solver_mode = int(params.get('solver_mode', 0))
        try:
            self.solver_cpu.set_lead(0, 0.0, T_eV); self.solver_cpu.set_lead(1, 0.0, T_eV)
            self.solver_cpu.set_check_prob_stop(bCheckProb=False, bCheckProbStop=False, CheckProbTol=1e-12)
            self.solver_cpu.setLinSolver(1, 50, 1e-12, solver_mode)
        except Exception:
            pass
        if self.solver_ocl is not None:
            try:
                self.solver_ocl.set_lead(0, 0.0, T_eV); self.solver_ocl.set_lead(1, 0.0, T_eV)
            except Exception:
                pass
        _safe_print(
            f"[run] use_ocl={use_ocl} V0={V0:.3f} Temp_eV={T_eV:.3e} npix={params['npix']} L={params['L']} "
            f"decay={params['decay']} GammaT={params['GammaT']} W={params['W']} Q0={params['Q0']} Qzz={params['Qzz']}"
        )

        self.solver_cpu.set_lead(0, 0.0, T_eV)
        self.solver_cpu.set_lead(1, 0.0, T_eV)

        if self.solver_ocl is not None:
            self.solver_ocl.set_lead(0, 0.0, T_eV)
            self.solver_ocl.set_lead(1, 0.0, T_eV)

        # geometry
        spos, rots, _angles = pauli_scan.make_site_geom(params)
        # Always propagate GUI Esite into the onsite-energy slot (w) used by C++/OpenCL.
        # This matters especially when geometry is loaded from file (it can contain its own E column).
        if spos.shape[1] >= 4:
            spos = np.array(spos, dtype=np.float64, copy=True)
            spos[:, 3] = float(params.get('Esite', 0.0))

        # Match CombinedChargeRingsGUI_v5 / pauli_scan pipeline: configure Wij from W
        # (constant/distance-based/file) on the solver instance.
        pauli_scan._apply_wij_config(self.solver_cpu, spos, params)
        Wij = _build_Wij_matrix(spos, params)
        cs, order = pauli.make_quadrupole_Coeffs(float(params['Q0']), float(params['Qzz']))

        # single-point sanity check (C++ scan_current_tip directly)
        try:
            state_order_dbg = pauli.make_state_order(self.nsite)
            cpp_params_dbg = pauli.make_cpp_params(params)
            pTip0 = np.array([[0.0, 0.0, float(params['z_tip']) + float(params['Rtip'])]], dtype=np.float64)
            Vtip0 = np.array([V0], dtype=np.float64)
            out0 = self.solver_cpu.scan_current_tip(
                pTip0,
                Vtip0,
                spos,
                cpp_params_dbg,
                int(order),
                cs,
                state_order_dbg,
                rots=rots,
                bOmp=False,
                bMakeArrays=True,
                Ts=None,
                return_probs=True,
                return_state_energies=True,
            )
            cur0 = np.asarray(out0[0])[0]
            Es0 = out0[1]
            Ts0 = out0[2]
            Probs0 = np.asarray(out0[3]).reshape(-1)
            StateEs0 = np.asarray(out0[4]).reshape(-1)
            imax = int(np.argmax(Probs0))
            _safe_print(f"[single tip] I={cur0:.6e} Es(min,max)=({Es0.min():.3e},{Es0.max():.3e}) Ts(min,max)=({Ts0.min():.3e},{Ts0.max():.3e})")
            _safe_print(f"[single tip] Probs: max={float(Probs0[imax]):.6e} at state={imax}  StateEs(min,max)=({StateEs0.min():.3e},{StateEs0.max():.3e})")
            if imax == 0 or imax == (2**self.nsite - 1):
                _safe_print("[single tip] WARNING: distribution collapsed to empty/full charge state; current can be ~0 for many params.")
        except Exception as e:
            _safe_print("[single tip] failed:", e)

        # XY (CPU path matches CombinedChargeRingsGUI_v5)
        if use_ocl:
            pTips, extent_xy = _make_xy_grid(params)
            Vtips = np.full((pTips.shape[0],), V0, dtype=np.float64)
            cur_xy = self._solve_xy_ocl(params, pTips, Vtips, spos, rots, order, cs, Wij=Wij)
            cur_xy = cur_xy.reshape(int(params['npix']), int(params['npix']))
        else:
            cur_xy, Es_cpu, Ts_cpu = self._solve_xy_cpu(params, spos, rots, cs, return_ets=True)
            extent_xy = (-float(params['L']), float(params['L']), -float(params['L']), float(params['L']))
            if self.solver_ocl is not None:
                pTips_dbg, _ = _make_xy_grid(params)
                Vtips_dbg = np.full((pTips_dbg.shape[0],), V0, dtype=np.float64)
                cur_xy_gpu = self._solve_xy_ocl(params, pTips_dbg, Vtips_dbg, spos, rots, order, cs, Wij=Wij)
                diff_xy = cur_xy_gpu - cur_xy
                mask_zero_cpu = cur_xy == 0.0
                mask_zero_gpu = cur_xy_gpu == 0.0
                n_zero_cpu = int(np.sum(mask_zero_cpu))
                n_zero_gpu = int(np.sum(mask_zero_gpu))
                n_zero_mismatch = int(np.sum(mask_zero_cpu != mask_zero_gpu))
                _safe_print(f"[XY diff] max_abs={np.max(np.abs(diff_xy)):.3e} mean_abs={np.mean(np.abs(diff_xy)):.3e} zero_cpu={n_zero_cpu} zero_gpu={n_zero_gpu} zero_mismatch={n_zero_mismatch}")
                if n_zero_mismatch > 0:
                    # compute CPU/GPU validity masks using the same cut as C++
                    gamma_amp = float(params['GammaT']) / np.pi
                    W_scalar = float(params['W'])
                    EW_cut = 2.0
                    Tmin_cut = 0.0
                    Es_cpu_sp = Es_cpu.reshape(int(params['npix']) * int(params['npix']), -1)
                    Ts_cpu_sp = Ts_cpu.reshape(int(params['npix']) * int(params['npix']), -1)
                    Emax_cpu = Es_cpu_sp.max(axis=1)
                    Tmax_cpu = np.max(np.abs(gamma_amp * Ts_cpu_sp), axis=1)
                    invalid_cpu = (Emax_cpu + W_scalar * EW_cut < 0.0) | (Tmax_cpu < Tmin_cut)
                    # GPU Es/Ts (float32) for debug
                    cur_xy_gpu_dbg, Es_gpu, Ts_gpu, *_ = self.solver_ocl.scan_current_tip(
                        pTips_dbg.astype(np.float32),
                        Vtips_dbg.astype(np.float32),
                        spos.astype(np.float32),
                        pauli.make_cpp_params(params).astype(np.float32),
                        int(order),
                        np.asarray(cs, dtype=np.float32),
                        rots=rots.astype(np.float32) if rots is not None else None,
                        return_probs=False,
                        return_state_energies=True,
                        Wij=Wij.astype(np.float32) if Wij is not None else None,
                    )
                    if Es_gpu is not None and Ts_gpu is not None:
                        Es_gpu_sp = Es_gpu.reshape(int(params['npix']) * int(params['npix']), -1)
                        Ts_gpu_sp = Ts_gpu.reshape(int(params['npix']) * int(params['npix']), -1)
                        Emax_gpu = Es_gpu_sp.max(axis=1)
                        Tmax_gpu = np.max(np.abs(gamma_amp * Ts_gpu_sp), axis=1)
                        invalid_gpu = (Emax_gpu + W_scalar * EW_cut < 0.0) | (Tmax_gpu < Tmin_cut)
                    else:
                        invalid_gpu = None
                    idxs = np.flatnonzero(mask_zero_cpu != mask_zero_gpu)
                    for k in idxs[:5]:
                        xk, yk = pTips_dbg[k, 0], pTips_dbg[k, 1]
                        if invalid_gpu is not None:
                            _safe_print(f"[XY diff] mismatch idx={k} x={xk:.3f} y={yk:.3f} I_cpu={cur_xy.flat[k]:.3e} I_gpu={cur_xy_gpu.flat[k]:.3e} valid_cpu={not invalid_cpu[k]} valid_gpu={not invalid_gpu[k]} Emax_cpu={Emax_cpu[k]:.3e} Tmax_cpu={Tmax_cpu[k]:.3e} Emax_gpu={Emax_gpu[k]:.3e} Tmax_gpu={Tmax_gpu[k]:.3e}")
                        else:
                            _safe_print(f"[XY diff] mismatch idx={k} x={xk:.3f} y={yk:.3f} I_cpu={cur_xy.flat[k]:.3e} I_gpu={cur_xy_gpu.flat[k]:.3e}")
                if np.any(~mask_zero_cpu & ~mask_zero_gpu):
                    diff_xy_nz = diff_xy[~mask_zero_cpu & ~mask_zero_gpu]
                    _safe_print(f"[XY diff nz] max_abs={np.max(np.abs(diff_xy_nz)):.3e} mean_abs={np.mean(np.abs(diff_xy_nz)):.3e}")
                _safe_print(f"[XY diff] params: V0={V0:.6f} dQ={params['dQ']} Temp={params['Temp']} GammaT={params['GammaT']} W={params['W']} Q0={params['Q0']} Qzz={params['Qzz']}")

        _safe_print(f"[XY] V={V0:.3f} min={cur_xy.min():.3e} max={cur_xy.max():.3e}")

        if self.cbShowdIdV.isChecked():
            dQ = float(params['dQ'])
            Vp = V0 + 0.5 * dQ
            Vm = V0 - 0.5 * dQ
            params_p = params.copy(); params_p['VBias'] = Vp
            params_m = params.copy(); params_m['VBias'] = Vm
            if use_ocl:
                pTips_p, _ = _make_xy_grid(params_p)
                Vtips_p = np.full((pTips_p.shape[0],), Vp, dtype=np.float64)
                cur_p = self._solve_xy_ocl(params_p, pTips_p, Vtips_p, spos, rots, order, cs, Wij=Wij).reshape(cur_xy.shape)
                pTips_m, _ = _make_xy_grid(params_m)
                Vtips_m = np.full((pTips_m.shape[0],), Vm, dtype=np.float64)
                cur_m = self._solve_xy_ocl(params_m, pTips_m, Vtips_m, spos, rots, order, cs, Wij=Wij).reshape(cur_xy.shape)
            else:
                cur_p = self._solve_xy_cpu(params_p, spos, rots, cs)
                cur_m = self._solve_xy_cpu(params_m, spos, rots, cs)
                if self.solver_ocl is not None:
                    pTips_p_dbg, _ = _make_xy_grid(params_p)
                    Vtips_p_dbg = np.full((pTips_p_dbg.shape[0],), Vp, dtype=np.float64)
                    cur_gpu_p = self._solve_xy_ocl(params_p, pTips_p_dbg, Vtips_p_dbg, spos, rots, order, cs, Wij=Wij).reshape(cur_xy.shape)
                    pTips_m_dbg, _ = _make_xy_grid(params_m)
                    Vtips_m_dbg = np.full((pTips_m_dbg.shape[0],), Vm, dtype=np.float64)
                    cur_gpu_m = self._solve_xy_ocl(params_m, pTips_m_dbg, Vtips_m_dbg, spos, rots, order, cs, Wij=Wij).reshape(cur_xy.shape)
                    img_xy_gpu = (cur_gpu_p - cur_gpu_m) / dQ
                    img_xy_cpu = (cur_p - cur_m) / dQ
                    diff_didv = img_xy_gpu - img_xy_cpu
                    _safe_print(f"[XY dIdV diff] max_abs={np.max(np.abs(diff_didv)):.3e} mean_abs={np.mean(np.abs(diff_didv)):.3e}")
            img_xy = (cur_p - cur_m) / dQ
            _safe_print(f"[XY dIdV] min={img_xy.min():.3e} max={img_xy.max():.3e}")
        else:
            img_xy = cur_xy

        # xV
        if use_ocl:
            pTips_line, dist = _make_line_tips(params)
            nV = int(params['nV'])
            Vbiases = np.linspace(0.0, float(params['VBias']), nV).astype(np.float64)
            stm_xv = self._solve_xv_ocl(params, pTips_line, Vbiases.astype(np.float32), spos, rots, order, cs, Wij=Wij)
        else:
            pTips_line, dist = _make_line_tips(params)
            nV = int(params['nV'])
            Vbiases = np.linspace(0.0, float(params['VBias']), nV).astype(np.float64)
            stm_xv = self._solve_xv_cpu(params, pTips_line, Vbiases, spos, rots, order, cs)
        _safe_print(f"[xV] min={np.min(stm_xv):.3e} max={np.max(stm_xv):.3e}")

        if self.cbShowdIdV.isChecked() and len(Vbiases) > 1:
            img_xv = np.gradient(stm_xv, Vbiases, axis=0)
            _safe_print(f"[xV dIdV] min={img_xv.min():.3e} max={img_xv.max():.3e}")
        else:
            img_xv = stm_xv

        extent_xv = (0.0, float(dist), float(Vbiases[0]), float(Vbiases[-1]))

        # fast redraw with blit
        self.pm.restore_backgrounds()

        # update data
        m_xy = float(np.max(np.abs(img_xy)))
        clim_xy = (-m_xy, m_xy) if m_xy > 0 else None
        self.pm.update_plot('xy', img_xy, extent=extent_xy, clim=clim_xy)

        m_xv = float(np.max(np.abs(img_xv)))
        clim_xv = (-m_xv, m_xv) if m_xv > 0 else None
        self.pm.update_plot('xv', img_xv, extent=extent_xv, clim=clim_xv)
        self.ax_xv.set_aspect('auto')

        self.pm.blit()


if __name__ == "__main__":
    qApp = QtWidgets.QApplication(sys.argv)
    aw = ApplicationWindow()
    aw.show()
    sys.exit(qApp.exec_())
