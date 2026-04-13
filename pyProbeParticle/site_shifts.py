import os
import json
import numpy as np
import configparser

from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates

BOHR_TO_ANG = 0.5291772109217
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG
HARTREE_TO_EV = 27.211396132


def rotmat_z(theta_rad):
    c = float(np.cos(theta_rad))
    s = float(np.sin(theta_rad))
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)


def smoothstep01(t):
    t = np.asarray(t, dtype=float)
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def load_cube_with_atoms(fname):
    f = open(fname, 'r')
    header1 = f.readline()
    header2 = f.readline()

    parts = f.readline().split()
    natoms_signed = int(parts[0])
    natoms = abs(natoms_signed)
    origin = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float)

    axes = np.zeros((3, 3), dtype=float)
    npts = np.zeros(3, dtype=int)
    for i in range(3):
        parts = f.readline().split()
        npts[i] = int(parts[0])
        axes[i] = [float(parts[1]), float(parts[2]), float(parts[3])]

    atom_Z = np.zeros(natoms, dtype=int)
    atom_pos = np.zeros((natoms, 3), dtype=float)
    for i in range(natoms):
        parts = f.readline().split()
        atom_Z[i] = int(parts[0])
        atom_pos[i] = [float(parts[2]), float(parts[3]), float(parts[4])]

    if natoms_signed < 0:
        f.readline()

    vals = []
    for line in f:
        vals.extend(line.split())
    f.close()

    density = np.array(vals, dtype=np.float64).reshape(int(npts[0]), int(npts[1]), int(npts[2]))

    v1 = axes[0]
    v2 = axes[1]
    v3 = axes[2]
    dV = abs(np.dot(v1, np.cross(v2, v3)))

    return {
        'density': density,
        'origin': origin,
        'axes': axes,
        'npts': npts,
        'atom_Z': atom_Z,
        'atom_pos': atom_pos,
        'dV': float(dV),
        'header1': header1.strip(),
        'header2': header2.strip(),
    }


def grid_positions_1d(origin_component, step, n):
    return float(origin_component) + np.arange(int(n), dtype=float) * float(step)


def make_grid_axes(cube):
    ox, oy, oz = cube['origin']
    dx, dy, dz = cube['axes'][0, 0], cube['axes'][1, 1], cube['axes'][2, 2]
    nx, ny, nz = cube['npts']
    xs = grid_positions_1d(ox, dx, nx)
    ys = grid_positions_1d(oy, dy, ny)
    zs = grid_positions_1d(oz, dz, nz)
    return xs, ys, zs


def read_xyzq(fname):
    pts = []
    qs = []
    fin = open(fname, 'r')
    for line in fin:
        s = line.strip()
        if s == '':
            continue
        if s[0] == '#':
            continue
        w = s.split()
        if len(w) < 4:
            continue
        pts.append([float(w[0]), float(w[1]), float(w[2])])
        qs.append(float(w[3]))
    fin.close()
    if len(pts) == 0:
        raise ValueError(f"No charges read from xyzq file: {fname}")
    pts = np.asarray(pts, dtype=float) * ANG_TO_BOHR
    qs = np.asarray(qs, dtype=float)
    return pts, qs


def enforce_zero_integral_lobe_scaling(density, dV):
    dens = np.asarray(density, dtype=float)
    dV = float(dV)
    pos_mask = dens > 0.0
    neg_mask = dens < 0.0
    if (not bool(np.any(pos_mask))) or (not bool(np.any(neg_mask))):
        return dens
    q_pos = float(np.sum(dens[pos_mask]) * dV)
    q_neg = float(np.sum(dens[neg_mask]) * dV)
    q_neg_abs = -q_neg
    if (q_pos <= 0.0) or (q_neg_abs <= 0.0):
        return dens
    q_avg = 0.5 * (q_pos + q_neg_abs)
    s_pos = q_avg / q_pos
    s_neg = q_avg / q_neg_abs
    out = dens.copy()
    out[pos_mask] = out[pos_mask] * float(s_pos)
    out[neg_mask] = out[neg_mask] * float(s_neg)
    return out


class ScalarCube:
    """Centered cube-file interpolator for any scalar field (potential, density, wavefunction).

    Loads a Gaussian cube file, re-centers coordinates so that the atom-mean
    is at the origin, and builds either a linear (RegularGridInterpolator) or
    cubic (map_coordinates) interpolator.  Points outside the cube return NaN.

    Parameters
    ----------
    cube_path : str
        Path to a Gaussian .cube file.
    interp_kind : str
        'linear' or 'cubic' (default).
    """

    def __init__(self, cube_path, interp_kind='cubic'):
        self.cube_path = str(cube_path)
        self.interp_kind = str(interp_kind)
        if self.interp_kind not in ['linear', 'cubic']:
            raise ValueError(f"interp_kind must be 'linear' or 'cubic', got: {self.interp_kind}")

        cube = load_cube_with_atoms(self.cube_path)
        center = np.mean(cube['atom_pos'], axis=0)
        self.center_bohr = np.asarray(center, dtype=float)
        cube['origin'] = cube['origin'] - self.center_bohr
        cube['atom_pos'] = cube['atom_pos'] - self.center_bohr[None, :]

        self.cube = cube
        xs, ys, zs = make_grid_axes(cube)
        self.xs = xs
        self.ys = ys
        self.zs = zs

        self.bounds_min = np.array([float(xs[0]), float(ys[0]), float(zs[0])], dtype=float)
        self.bounds_max = np.array([float(xs[-1]), float(ys[-1]), float(zs[-1])], dtype=float)

        V = np.asarray(cube['density'], dtype=float)
        if self.interp_kind == 'linear':
            self._interp = RegularGridInterpolator(
                (xs, ys, zs), V,
                method='linear',
                bounds_error=False,
                fill_value=np.nan,
            )
            self._grid = None
            self._x0 = None
            self._y0 = None
            self._z0 = None
            self._dx = None
            self._dy = None
            self._dz = None
        else:
            self._interp = None
            self._grid = V
            self._x0 = float(xs[0])
            self._y0 = float(ys[0])
            self._z0 = float(zs[0])
            self._dx = float(xs[1] - xs[0])
            self._dy = float(ys[1] - ys[0])
            self._dz = float(zs[1] - zs[0])

    def eval(self, points_bohr):
        p = np.asarray(points_bohr, dtype=float)
        if self._interp is not None:
            return self._interp(p)
        ix = (p[:, 0] - self._x0) / self._dx
        iy = (p[:, 1] - self._y0) / self._dy
        iz = (p[:, 2] - self._z0) / self._dz
        coords = np.vstack([ix, iy, iz])
        V = map_coordinates(
            self._grid,
            coords,
            order=3,
            mode='constant',
            cval=np.nan,
            prefilter=True,
        )
        return V


PotentialCube = ScalarCube


class PointCharges:
    def __init__(self, positions_bohr, charges_e):
        self.pos = np.asarray(positions_bohr, dtype=float)
        self.q = np.asarray(charges_e, dtype=float)
        if self.pos.ndim != 2 or self.pos.shape[1] != 3:
            raise ValueError("positions_bohr must be (N,3)")
        if self.q.shape[0] != self.pos.shape[0]:
            raise ValueError("charges size mismatch")

    def eval(self, points_bohr, chunk=4096):
        p = np.asarray(points_bohr, dtype=float)
        out = np.zeros(p.shape[0], dtype=float)
        n = int(p.shape[0])
        for i0 in range(0, n, int(chunk)):
            i1 = min(i0 + int(chunk), n)
            dp = p[i0:i1, None, :] - self.pos[None, :, :]
            r2 = np.sum(dp * dp, axis=2)
            r = np.sqrt(np.maximum(r2, 1e-24))
            out[i0:i1] = np.sum(self.q[None, :] / r, axis=1)
        return out


class HybridPotential:
    def __init__(
        self,
        pot_cube,
        resp_charges,
        switch_width_ang=0.5,
        cube_margin_ang=0.0,
        beta_match_shell_ang=0.5,
        beta_max_points=2000,
    ):
        self.pot_cube = pot_cube
        self.resp = resp_charges
        self.switch_width_bohr = float(switch_width_ang) * ANG_TO_BOHR
        self.cube_margin_bohr = float(cube_margin_ang) * ANG_TO_BOHR
        self.beta_match_shell_bohr = float(beta_match_shell_ang) * ANG_TO_BOHR
        self.beta_max_points = int(beta_max_points)

        bmin = np.asarray(self.pot_cube.bounds_min, dtype=float) + self.cube_margin_bohr
        bmax = np.asarray(self.pot_cube.bounds_max, dtype=float) - self.cube_margin_bohr
        self.bounds_min = bmin
        self.bounds_max = bmax

        self.beta = float(self._compute_beta_match())

    def _distance_inside_bounds(self, points_bohr):
        p = np.asarray(points_bohr, dtype=float)
        dmin = p - self.bounds_min[None, :]
        dmax = self.bounds_max[None, :] - p
        d = np.minimum(dmin, dmax)
        return np.min(d, axis=1)

    def _compute_beta_match(self):
        xs = self.pot_cube.xs
        ys = self.pot_cube.ys
        zs = self.pot_cube.zs

        Xs, Ys, Zs = np.meshgrid(xs, ys, zs, indexing='ij')
        pts = np.stack([Xs.ravel(), Ys.ravel(), Zs.ravel()], axis=-1)

        dins = self._distance_inside_bounds(pts)
        shell = (dins >= 0.0) & (dins <= self.beta_match_shell_bohr)
        idx = np.where(shell)[0]
        if idx.size == 0:
            return 0.0

        stride = int(np.ceil(float(idx.size) / float(max(self.beta_max_points, 1))))
        stride = max(stride, 1)
        idx = idx[::stride]
        pts = pts[idx]

        Vc = self.pot_cube.eval(pts)
        good = np.isfinite(Vc)
        if not bool(np.any(good)):
            return 0.0
        pts = pts[good]
        Vc = Vc[good]
        Vr = self.resp.eval(pts)
        return float(np.mean(Vc - Vr))

    def eval(self, points_bohr):
        p = np.asarray(points_bohr, dtype=float)
        dins = self._distance_inside_bounds(p)
        w = smoothstep01(dins / max(self.switch_width_bohr, 1e-12))
        w = np.where(dins <= 0.0, 0.0, w)
        w = np.where(dins >= self.switch_width_bohr, 1.0, w)

        V = np.zeros(p.shape[0], dtype=float)

        m_resp = w < 1.0
        if bool(np.any(m_resp)):
            idx = np.where(m_resp)[0]
            V_resp = self.resp.eval(p[idx]) + self.beta
            V[idx] = V[idx] + (1.0 - w[idx]) * V_resp

        m_cube = w > 0.0
        if bool(np.any(m_cube)):
            idx = np.where(m_cube)[0]
            V_cube = self.pot_cube.eval(p[idx])
            good = np.isfinite(V_cube)
            if bool(np.any(good)):
                idx_good = idx[good]
                V[idx_good] = V[idx_good] + w[idx_good] * V_cube[good]
        return V


def _resolve_path(base_dir, p):
    p = str(p)
    if os.path.isabs(p):
        return p
    return os.path.join(str(base_dir), p)


def load_siteshift_cubes(fname, base_dir):
    """Load siteshift_cubes.ini returning per-row mol_ids, diff paths, gs paths, and optional orbital paths.

    Columns: mol_id  diff_density_cube  gs_density_cube  [virtual_orbital_cube]
    Column 4 (virtual orbital cube) is optional; missing entries are stored as None.
    """
    mol_ids = []
    diff_paths = []
    gs_paths = []
    orbital_paths = []
    fin = open(fname, 'r')
    for line in fin:
        s = line.strip()
        if s == '':
            continue
        if s[0] == '#':
            continue
        w = s.split()
        if len(w) < 3:
            raise ValueError(f"Expected at least 3 columns (mol_id diff_cube gs_cube) in {fname}, got line: {line}")
        mol_ids.append(int(w[0]))
        diff_paths.append(_resolve_path(base_dir, w[1]))
        gs_paths.append(_resolve_path(base_dir, w[2]))
        if len(w) >= 4:
            orbital_paths.append(_resolve_path(base_dir, w[3]))
        else:
            orbital_paths.append(None)
    fin.close()
    return mol_ids, diff_paths, gs_paths, orbital_paths


def load_electrostatics_ini(fname, base_dir):
    cfg = configparser.ConfigParser()
    cfg.read(fname)
    out = {}
    for sec in cfg.sections():
        key = sec.strip()
        mol_id = None
        if key.isdigit():
            mol_id = int(key)
        if mol_id is None and key.lower().startswith('mol') and key[3:].isdigit():
            mol_id = int(key[3:])
        if mol_id is None:
            continue
        d = {}
        if cfg.has_option(sec, 'potential_cube'):
            d['potential_cube'] = _resolve_path(base_dir, cfg.get(sec, 'potential_cube'))
        if cfg.has_option(sec, 'resp_charges'):
            d['resp_charges'] = _resolve_path(base_dir, cfg.get(sec, 'resp_charges'))
        if cfg.has_option(sec, 'switch_width_ang'):
            d['switch_width_ang'] = float(cfg.get(sec, 'switch_width_ang'))
        if cfg.has_option(sec, 'cube_margin_ang'):
            d['cube_margin_ang'] = float(cfg.get(sec, 'cube_margin_ang'))
        if cfg.has_option(sec, 'beta_match_shell_ang'):
            d['beta_match_shell_ang'] = float(cfg.get(sec, 'beta_match_shell_ang'))
        if cfg.has_option(sec, 'resp_center'):
            d['resp_center'] = str(cfg.get(sec, 'resp_center')).strip()
        if cfg.has_option(sec, 'gs_density_cube'):
            d['gs_density_cube'] = _resolve_path(base_dir, cfg.get(sec, 'gs_density_cube'))
        if cfg.has_option(sec, 'polarizability_tensor'):
            d['polarizability_tensor'] = _resolve_path(base_dir, cfg.get(sec, 'polarizability_tensor'))
        if cfg.has_option(sec, 'alpha_units'):
            d['alpha_units'] = str(cfg.get(sec, 'alpha_units')).strip().lower()
        if cfg.has_option(sec, 'alpha_frame'):
            d['alpha_frame'] = str(cfg.get(sec, 'alpha_frame')).strip().lower()
        if cfg.has_option(sec, 'homo_cube'):
            d['homo_cube'] = _resolve_path(base_dir, cfg.get(sec, 'homo_cube'))
        out[int(mol_id)] = d
    return out


def build_hybrid_potentials(electrostatics_cfg, interp_kind='cubic'):
    pots = {}
    for mol_id, d in electrostatics_cfg.items():
        if 'potential_cube' not in d:
            raise ValueError(f"Missing potential_cube for mol_id={mol_id} in electrostatics.ini")
        if 'resp_charges' not in d:
            raise ValueError(f"Missing resp_charges for mol_id={mol_id} in electrostatics.ini")
        pot_cube = PotentialCube(d['potential_cube'], interp_kind=interp_kind)
        qpos, q = read_xyzq(d['resp_charges'])
        resp_center = str(d.get('resp_center', 'cube_atom_mean')).strip().lower()
        if resp_center == 'cube_atom_mean':
            qpos = qpos - np.asarray(pot_cube.center_bohr, dtype=float)[None, :]
        elif resp_center == 'none':
            qpos = qpos
        else:
            raise ValueError(f"resp_center must be 'cube_atom_mean' or 'none', got: {resp_center}")
        resp = PointCharges(qpos, q)
        pots[int(mol_id)] = HybridPotential(
            pot_cube,
            resp,
            switch_width_ang=float(d.get('switch_width_ang', 0.5)),
            cube_margin_ang=float(d.get('cube_margin_ang', 0.0)),
            beta_match_shell_ang=float(d.get('beta_match_shell_ang', 0.5)),
        )
    return pots


# ---------------------------------------------------------------------------
#  Polarizability tensor I/O
# ---------------------------------------------------------------------------

BOHR3_TO_ANG3 = BOHR_TO_ANG ** 3
ANG3_TO_BOHR3 = ANG_TO_BOHR ** 3


def load_polarizability_tensor(fname, units='bohr3'):
    """Load a 3x3 polarizability tensor from a text file.

    Expected format: 3 data lines of 3 floats each (comment lines starting
    with '#' are skipped).  The tensor must be symmetric (alpha_ij == alpha_ji).

    Parameters
    ----------
    fname : str
        Path to the polarizability file.
    units : str
        'bohr3' (PySCF native, default) or 'ang3'.

    Returns
    -------
    alpha : np.ndarray, shape (3,3)
        Polarizability tensor in **Bohr^3** (atomic units).
    """
    rows = []
    fin = open(fname, 'r')
    for line in fin:
        s = line.strip()
        if s == '' or s[0] == '#':
            continue
        w = s.split()
        if len(w) < 3:
            continue
        rows.append([float(w[0]), float(w[1]), float(w[2])])
        if len(rows) == 3:
            break
    fin.close()
    if len(rows) != 3:
        raise ValueError(f"Expected 3 data rows in polarizability file {fname}, got {len(rows)}")
    alpha = np.array(rows, dtype=float)
    units = str(units).strip().lower()
    if units == 'ang3':
        alpha = alpha * ANG3_TO_BOHR3
    elif units != 'bohr3':
        raise ValueError(f"alpha_units must be 'bohr3' or 'ang3', got: {units}")
    sym_err = float(np.max(np.abs(alpha - alpha.T)))
    if sym_err > 1e-6 * float(np.max(np.abs(alpha)) + 1e-30):
        raise ValueError(f"Polarizability tensor in {fname} is not symmetric (max asymmetry={sym_err:.3e})")
    alpha = 0.5 * (alpha + alpha.T)
    return alpha


# ---------------------------------------------------------------------------
#  site_shift_terms.ini parser
# ---------------------------------------------------------------------------

def load_site_shift_terms_ini(fname):
    """Parse site_shift_terms.ini and return a dict of settings per section.

    Returns
    -------
    cfg : dict
        Keys: 'enabled_terms' (list of str), 'exchange' (dict), 'polarization' (dict), 'ct' (dict).
    """
    cp = configparser.ConfigParser()
    cp.read(fname)
    cfg = {}

    enabled = ['coulomb']
    if cp.has_section('terms') and cp.has_option('terms', 'enable'):
        enabled = [s.strip().lower() for s in cp.get('terms', 'enable').split(',') if s.strip()]
    cfg['enabled_terms'] = enabled

    exch = {}
    if cp.has_section('exchange'):
        exch['k_exch_eV_bohr3'] = float(cp.get('exchange', 'k_exch_eV_bohr3', fallback='7.0'))
    cfg['exchange'] = exch

    pol = {}
    if cp.has_section('polarization'):
        pol['field_method'] = str(cp.get('polarization', 'field_method', fallback='direct')).strip().lower()
        pol['fd_step_bohr'] = float(cp.get('polarization', 'fd_step_bohr', fallback='0.2'))
        pol['r_min_bohr'] = float(cp.get('polarization', 'r_min_bohr', fallback='0.5'))
    cfg['polarization'] = pol

    ct = {}
    if cp.has_section('ct'):
        ct['model'] = str(cp.get('ct', 'model', fallback='off')).strip().lower()
        ct['k_ct_eV'] = float(cp.get('ct', 'k_ct_eV', fallback='1.0'))
        ct['E_CT_model'] = str(cp.get('ct', 'E_CT_model', fallback='fixed')).strip().lower()
        if ct['E_CT_model'] == 'fixed':
            ct['E_CT_eV'] = float(cp.get('ct', 'E_CT_eV', fallback='0.0'))
        ct['epsilon_r'] = float(cp.get('ct', 'epsilon_r', fallback='1.0'))
        ct['V_coul_model'] = str(cp.get('ct', 'V_coul_model', fallback='point_charge')).strip().lower()
        if ct['V_coul_model'] == 'fixed':
            ct['V_coul_fixed_eV'] = float(cp.get('ct', 'V_coul_fixed_eV', fallback='0.0'))
        for key in ['A_e_eV', 'beta_e_inv_ang', 'A_h_eV', 'beta_h_inv_ang']:
            if cp.has_option('ct', key):
                ct[key] = float(cp.get('ct', key))
        for sec_key in cp.options('ct'):
            if sec_key.startswith('ip_mol') and sec_key.endswith('_ev'):
                ct[sec_key] = float(cp.get('ct', sec_key))
            if sec_key.startswith('ea_mol') and sec_key.endswith('_ev'):
                ct[sec_key] = float(cp.get('ct', sec_key))
    cfg['ct'] = ct

    return cfg


# ---------------------------------------------------------------------------
#  Per-mol_id GS density cache builder
# ---------------------------------------------------------------------------

def _build_gs_density_cache(mol_ids, gs_paths, electro_cfg, interp_kind='cubic'):
    """Build a dict mapping mol_id -> ScalarCube for ground-state density.

    Priority: electrostatics.ini 'gs_density_cube' > first gs_path in siteshift_cubes.ini per mol_id.
    """
    cache = {}
    gs_by_mol = {}
    for mi, gp in zip(mol_ids, gs_paths):
        mid = int(mi)
        if mid not in gs_by_mol:
            gs_by_mol[mid] = str(gp)

    for mid in sorted(set(int(x) for x in mol_ids)):
        path = None
        if mid in electro_cfg and 'gs_density_cube' in electro_cfg[mid]:
            path = electro_cfg[mid]['gs_density_cube']
        elif mid in gs_by_mol:
            path = gs_by_mol[mid]
        if path is not None and os.path.isfile(path):
            cache[mid] = ScalarCube(path, interp_kind=interp_kind)
    return cache


# ---------------------------------------------------------------------------
#  Per-mol_id polarizability tensor loader
# ---------------------------------------------------------------------------

def _build_alpha_cache(mol_ids, electro_cfg):
    """Build a dict mapping mol_id -> alpha_tensor (3x3, Bohr^3) from electrostatics.ini."""
    cache = {}
    for mid in sorted(set(int(x) for x in mol_ids)):
        if mid not in electro_cfg:
            continue
        d = electro_cfg[mid]
        if 'polarizability_tensor' not in d:
            continue
        path = d['polarizability_tensor']
        if not os.path.isfile(path):
            continue
        units = d.get('alpha_units', 'bohr3')
        cache[mid] = load_polarizability_tensor(path, units=units)
    return cache


# ---------------------------------------------------------------------------
#  Common: build mol_id -> pose dict
# ---------------------------------------------------------------------------

def _build_mol_id_to_pose(mol_ids, poss_bohr, rots_rad):
    """Build mapping from mol_id to position/rotation/rotation-matrix (from first row per mol_id)."""
    mol_id_to_pose = {}
    for mid_val in sorted(set(int(x) for x in mol_ids)):
        idxs = [i for i, mi in enumerate(mol_ids) if int(mi) == int(mid_val)]
        if len(idxs) == 0:
            continue
        i0 = int(idxs[0])
        mol_id_to_pose[int(mid_val)] = {
            'pos_bohr': poss_bohr[i0],
            'rot_rad': float(rots_rad[i0]),
            'R': rotmat_z(float(rots_rad[i0])),
        }
    return mol_id_to_pose


# ---------------------------------------------------------------------------
#  Common: prepare Δρ grid for a single site-state
# ---------------------------------------------------------------------------

def _prepare_diff_density_grid(diff_path, pose_i):
    """Load a difference density cube, center it, enforce zero integral, build local grid and global points.

    Returns (dn_flat, dV, pts_global, pts_i_local).
    """
    cube_dn = load_cube_with_atoms(diff_path)
    center_i = np.mean(cube_dn['atom_pos'], axis=0)
    cube_dn['origin'] = cube_dn['origin'] - center_i
    cube_dn['atom_pos'] = cube_dn['atom_pos'] - center_i[None, :]
    cube_dn['density'] = enforce_zero_integral_lobe_scaling(cube_dn['density'], cube_dn['dV'])

    xs, ys, zs = make_grid_axes(cube_dn)
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing='ij')
    pts_i_local = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=-1)
    dn_flat = np.asarray(cube_dn['density'], dtype=float).ravel()
    dV = float(cube_dn['dV'])

    Ri = pose_i['R']
    pos_i = pose_i['pos_bohr']
    pts_global = pts_i_local @ Ri + pos_i[None, :]

    return dn_flat, dV, pts_global, pts_i_local


# ---------------------------------------------------------------------------
#  Term B: Pauli Exchange / Steric Overlap
#  Δω_Exch[i] = Σ_{B≠A} K_exch · ∫ Δρ_A[i](r) · ρ^GS_B(r) dr
#  Ref: Amovilli & Mennucci, J. Phys. Chem. A 1997, 101, 2616-2622
# ---------------------------------------------------------------------------

def _compute_exchange_shifts(
    mol_ids, diff_paths, mol_id_to_pose, gs_density_cache,
    poss_bohr, rots_rad,
    k_exch_eV_bohr3=7.0,
    chunk=100000,
):
    """Compute Pauli exchange shift for each site-state.

    Returns array of shape (N_sites,) in eV, and per-site breakdown list.
    """
    n_sites = len(mol_ids)
    shifts_eV = np.zeros(n_sites, dtype=float)
    breakdown = []

    for i_site in range(n_sites):
        mi = int(mol_ids[i_site])
        pose_i = mol_id_to_pose[mi]
        dn_flat, dV, pts_global, _ = _prepare_diff_density_grid(diff_paths[i_site], pose_i)
        npts = pts_global.shape[0]

        neighbors = []
        for mj, gs_cube_j in gs_density_cache.items():
            if int(mj) == int(mi):
                continue
            pose_j = mol_id_to_pose.get(int(mj))
            if pose_j is None:
                continue
            Rj = pose_j['R']
            pos_j = pose_j['pos_bohr']

            rho_gs = np.zeros(npts, dtype=float)
            for i0 in range(0, npts, int(chunk)):
                i1 = min(i0 + int(chunk), npts)
                p_loc_j = (pts_global[i0:i1] - pos_j[None, :]) @ Rj.T
                vals = gs_cube_j.eval(p_loc_j)
                good = np.isfinite(vals)
                rho_gs[i0:i1] = np.where(good, vals, 0.0)

            overlap = float(np.sum(dn_flat * rho_gs) * dV)
            dw_pair = k_exch_eV_bohr3 * overlap
            shifts_eV[i_site] += dw_pair
            neighbors.append({'env_mol_id': int(mj), 'exchange_eV': float(dw_pair), 'overlap_bohr3': float(overlap)})

        breakdown.append({'site': int(i_site), 'mol_id': int(mi), 'neighbors': neighbors})

    return shifts_eV, breakdown


# ---------------------------------------------------------------------------
#  Term C: Mutual Polarization / Induction
#  Δω_Pol[i] = Σ_{B≠A} [-F_B^(A0)·α_B·ΔF_B[i] - ½ ΔF_B[i]·α_B·ΔF_B[i]]
#  Ref: Stone, "The Theory of Intermolecular Forces", Oxford 2013, Ch.5
# ---------------------------------------------------------------------------

def _electric_field_from_density_grid(dn_flat, dV, pts_global, target_bohr, r_min_bohr=0.5):
    """Compute electric field at a single target point from a charge density grid.

    The density is treated as electron density (negative charge), so the charge
    density is -dn_flat.  Field E = Σ_r (-dn[r]) * (target - r) / max(|target-r|^3, r_min^3) * dV.

    Returns 3-vector field in atomic units (Hartree / (e * Bohr)).
    """
    dr = target_bohr[None, :] - pts_global
    r2 = np.sum(dr * dr, axis=1)
    r = np.sqrt(np.maximum(r2, 1e-30))
    r3 = np.maximum(r ** 3, r_min_bohr ** 3)
    charge_density = -dn_flat
    Fx = float(np.sum(charge_density * dr[:, 0] / r3) * dV)
    Fy = float(np.sum(charge_density * dr[:, 1] / r3) * dV)
    Fz = float(np.sum(charge_density * dr[:, 2] / r3) * dV)
    return np.array([Fx, Fy, Fz], dtype=float)


def _gs_field_at_point_fd(hybrid_pot, point_bohr_local, fd_step=0.2):
    """Compute ground-state electric field at a point via finite differences of the hybrid potential.

    F = -grad(V).  Uses central differences with step fd_step (Bohr).
    """
    F = np.zeros(3, dtype=float)
    for dim in range(3):
        p_plus = point_bohr_local.copy()
        p_minus = point_bohr_local.copy()
        p_plus[dim] += fd_step
        p_minus[dim] -= fd_step
        Vp = hybrid_pot.eval(p_plus.reshape(1, 3))[0]
        Vm = hybrid_pot.eval(p_minus.reshape(1, 3))[0]
        if np.isfinite(Vp) and np.isfinite(Vm):
            F[dim] = -(Vp - Vm) / (2.0 * fd_step)
    return F


def _compute_polarization_shifts(
    mol_ids, diff_paths, mol_id_to_pose, alpha_cache, pots,
    electro_cfg,
    field_method='direct',
    fd_step_bohr=0.2,
    r_min_bohr=0.5,
    chunk=100000,
):
    """Compute mutual polarization shift for each site-state.

    Returns array of shape (N_sites,) in eV, and per-site breakdown list.
    """
    n_sites = len(mol_ids)
    shifts_eV = np.zeros(n_sites, dtype=float)
    breakdown = []

    gs_field_cache = {}

    for i_site in range(n_sites):
        mi = int(mol_ids[i_site])
        pose_i = mol_id_to_pose[mi]
        dn_flat, dV, pts_global, _ = _prepare_diff_density_grid(diff_paths[i_site], pose_i)

        neighbors = []
        for mj in alpha_cache:
            if int(mj) == int(mi):
                continue
            pose_j = mol_id_to_pose.get(int(mj))
            if pose_j is None:
                continue
            alpha_j_local = alpha_cache[int(mj)]
            alpha_frame = 'local'
            if int(mj) in electro_cfg:
                alpha_frame = electro_cfg[int(mj)].get('alpha_frame', 'local')

            Rj = pose_j['R']
            if alpha_frame == 'local':
                alpha_j = Rj @ alpha_j_local @ Rj.T
            else:
                alpha_j = alpha_j_local.copy()

            center_j_global = pose_j['pos_bohr']

            delta_F = _electric_field_from_density_grid(dn_flat, dV, pts_global, center_j_global, r_min_bohr=r_min_bohr)

            cache_key = (int(mi), int(mj))
            if cache_key not in gs_field_cache:
                if int(mi) in pots:
                    pot_i = pots[int(mi)]
                    pos_i = pose_i['pos_bohr']
                    Ri = pose_i['R']
                    target_local_i = (center_j_global - pos_i) @ Ri.T
                    F_gs = _gs_field_at_point_fd(pot_i, target_local_i, fd_step=fd_step_bohr)
                    F_gs_global = F_gs @ Ri
                    gs_field_cache[cache_key] = F_gs_global
                else:
                    gs_field_cache[cache_key] = np.zeros(3, dtype=float)
            F_gs = gs_field_cache[cache_key]

            cross_term = -float(F_gs @ alpha_j @ delta_F)
            self_term = -0.5 * float(delta_F @ alpha_j @ delta_F)
            dw_pair_h = cross_term + self_term
            dw_pair_eV = dw_pair_h * HARTREE_TO_EV

            shifts_eV[i_site] += dw_pair_eV
            neighbors.append({
                'env_mol_id': int(mj),
                'polarization_eV': float(dw_pair_eV),
                'cross_term_eV': float(cross_term * HARTREE_TO_EV),
                'self_term_eV': float(self_term * HARTREE_TO_EV),
                'delta_F_au': delta_F.tolist(),
                'F_gs_au': F_gs.tolist(),
            })

        breakdown.append({'site': int(i_site), 'mol_id': int(mi), 'neighbors': neighbors})

    return shifts_eV, breakdown


# ---------------------------------------------------------------------------
#  Term D: Charge-Transfer Mixing / Superexchange
#  Δω_CT[i] = Σ_{B≠A} [-|t_e|²/(E_CT-E_S1[i]) - |t_h|²/(E_CT-E_S1[i])]
#  Ref: Spano, Acc. Chem. Res. 2010, 43, 429-439
# ---------------------------------------------------------------------------

def _orbital_overlap(cube_A_path, cube_B_path, pose_A, pose_B, interp_kind='cubic', chunk=100000):
    """Compute spatial overlap ∫ ψ_A(r) ψ_B(r) dr between two orbital cubes.

    Each cube is centered on its atom-mean.  The integral is evaluated on A's grid,
    with B's orbital interpolated via ScalarCube.

    Returns the overlap integral (dimensionless, in Bohr^3 units absorbed by dV).
    """
    cube_a = load_cube_with_atoms(cube_A_path)
    center_a = np.mean(cube_a['atom_pos'], axis=0)
    cube_a['origin'] = cube_a['origin'] - center_a
    cube_a['atom_pos'] = cube_a['atom_pos'] - center_a[None, :]

    xs, ys, zs = make_grid_axes(cube_a)
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing='ij')
    pts_a_local = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=-1)
    psi_a = np.asarray(cube_a['density'], dtype=float).ravel()
    dV = float(cube_a['dV'])

    Ra = pose_A['R']
    pos_a = pose_A['pos_bohr']
    pts_global = pts_a_local @ Ra + pos_a[None, :]

    scb = ScalarCube(cube_B_path, interp_kind=interp_kind)
    Rb = pose_B['R']
    pos_b = pose_B['pos_bohr']

    npts = pts_global.shape[0]
    psi_b = np.zeros(npts, dtype=float)
    for i0 in range(0, npts, int(chunk)):
        i1 = min(i0 + int(chunk), npts)
        p_loc_b = (pts_global[i0:i1] - pos_b[None, :]) @ Rb.T
        vals = scb.eval(p_loc_b)
        good = np.isfinite(vals)
        psi_b[i0:i1] = np.where(good, vals, 0.0)

    overlap = float(np.sum(psi_a * psi_b) * dV)
    return overlap


def _compute_ct_shifts(
    mol_ids, diff_paths, orbital_paths, mol_id_to_pose,
    Ediags_eV, ct_cfg, electro_cfg,
    interp_kind='cubic',
    chunk=100000,
):
    """Compute CT mixing shift for each site-state.

    Returns array of shape (N_sites,) in eV, and per-site breakdown list.
    """
    n_sites = len(mol_ids)
    shifts_eV = np.zeros(n_sites, dtype=float)
    breakdown = []
    model = ct_cfg.get('model', 'off')
    if model == 'off':
        for i_site in range(n_sites):
            breakdown.append({'site': int(i_site), 'mol_id': int(mol_ids[i_site]), 'neighbors': []})
        return shifts_eV, breakdown

    k_ct = ct_cfg.get('k_ct_eV', 1.0)

    homo_cubes = {}
    for mid in sorted(set(int(x) for x in mol_ids)):
        if mid in electro_cfg and 'homo_cube' in electro_cfg[mid]:
            path = electro_cfg[mid]['homo_cube']
            if os.path.isfile(path):
                homo_cubes[mid] = path

    uniq_mols = sorted(set(int(x) for x in mol_ids))

    overlap_cache = {}

    for i_site in range(n_sites):
        mi = int(mol_ids[i_site])
        E_S1 = float(Ediags_eV[i_site])
        pose_i = mol_id_to_pose[mi]

        neighbors = []
        for mj in uniq_mols:
            if int(mj) == int(mi):
                continue
            pose_j = mol_id_to_pose.get(int(mj))
            if pose_j is None:
                continue

            R_vec = pose_j['pos_bohr'] - pose_i['pos_bohr']
            R_dist_bohr = float(np.sqrt(np.sum(R_vec ** 2)))
            R_dist_ang = R_dist_bohr * BOHR_TO_ANG

            E_CT = _compute_E_CT(mi, mj, R_dist_ang, ct_cfg)
            if E_CT is None:
                continue

            denom = E_CT - E_S1
            if denom <= 0.0:
                print(f"WARNING [CT]: E_CT ({E_CT:.4f} eV) <= E_S1 ({E_S1:.4f} eV) for site {i_site} "
                      f"(mol_id={mi}) -> env mol_id={mj}. Skipping CT for this pair.")
                continue

            t_e_sq = 0.0
            t_h_sq = 0.0

            if model == 'orbital_overlap':
                virt_path_i = orbital_paths[i_site] if i_site < len(orbital_paths) else None
                if virt_path_i is not None and os.path.isfile(virt_path_i):
                    first_j_virt = None
                    for k, mk in enumerate(mol_ids):
                        if int(mk) == int(mj) and k < len(orbital_paths) and orbital_paths[k] is not None:
                            first_j_virt = orbital_paths[k]
                            break
                    if first_j_virt is not None and os.path.isfile(first_j_virt):
                        cache_key_e = (virt_path_i, first_j_virt, mi, mj)
                        if cache_key_e not in overlap_cache:
                            overlap_cache[cache_key_e] = _orbital_overlap(
                                virt_path_i, first_j_virt, pose_i, pose_j,
                                interp_kind=interp_kind, chunk=chunk)
                        S_e = overlap_cache[cache_key_e]
                        t_e_sq = (k_ct * abs(S_e)) ** 2

                if mi in homo_cubes and mj in homo_cubes:
                    cache_key_h = (homo_cubes[mi], homo_cubes[mj], mi, mj)
                    if cache_key_h not in overlap_cache:
                        overlap_cache[cache_key_h] = _orbital_overlap(
                            homo_cubes[mi], homo_cubes[mj], pose_i, pose_j,
                            interp_kind=interp_kind, chunk=chunk)
                    S_h = overlap_cache[cache_key_h]
                    t_h_sq = (k_ct * abs(S_h)) ** 2

            elif model == 'exp':
                A_e = ct_cfg.get('A_e_eV', 0.0)
                beta_e = ct_cfg.get('beta_e_inv_ang', 0.0)
                A_h = ct_cfg.get('A_h_eV', 0.0)
                beta_h = ct_cfg.get('beta_h_inv_ang', 0.0)
                t_e_sq = (A_e * np.exp(-beta_e * R_dist_ang)) ** 2
                t_h_sq = (A_h * np.exp(-beta_h * R_dist_ang)) ** 2

            dw_ct = -(t_e_sq + t_h_sq) / denom
            shifts_eV[i_site] += dw_ct
            neighbors.append({
                'env_mol_id': int(mj),
                'ct_eV': float(dw_ct),
                't_e_sq_eV2': float(t_e_sq),
                't_h_sq_eV2': float(t_h_sq),
                'E_CT_eV': float(E_CT),
                'E_S1_eV': float(E_S1),
                'R_ang': float(R_dist_ang),
            })

        breakdown.append({'site': int(i_site), 'mol_id': int(mi), 'neighbors': neighbors})

    return shifts_eV, breakdown


def _compute_E_CT(mol_id_A, mol_id_B, R_ang, ct_cfg):
    """Compute the CT state energy E_CT for a given pair.

    Returns E_CT in eV, or None if insufficient parameters.
    """
    model = ct_cfg.get('E_CT_model', 'fixed')
    if model == 'fixed':
        val = ct_cfg.get('E_CT_eV', None)
        if val is None or float(val) == 0.0:
            return None
        return float(val)

    if model == 'ip_ea':
        ip_key = f'ip_mol{mol_id_A}_ev'
        ea_key = f'ea_mol{mol_id_B}_ev'
        if ip_key not in ct_cfg or ea_key not in ct_cfg:
            return None
        IP_A = float(ct_cfg[ip_key])
        EA_B = float(ct_cfg[ea_key])
        v_model = ct_cfg.get('V_coul_model', 'point_charge')
        if v_model == 'fixed':
            V_coul = ct_cfg.get('V_coul_fixed_eV', 0.0)
        else:
            eps_r = max(ct_cfg.get('epsilon_r', 1.0), 1e-6)
            R_bohr = max(R_ang * ANG_TO_BOHR, 1e-6)
            V_coul = HARTREE_TO_EV / (eps_r * R_bohr)
        return IP_A - EA_B + V_coul

    return None


# ---------------------------------------------------------------------------
#  Orchestrator: compute all enabled terms
# ---------------------------------------------------------------------------

def compute_site_shift_terms(
    system,
    siteshift_cubes_path,
    electrostatics_ini_path,
    terms_ini_path=None,
    base_dir='.',
    interp_kind='cubic',
    chunk=100000,
    save_json_path=None,
):
    """Compute per-term site-energy shifts for all site-states.

    Parameters
    ----------
    system : ExcitonSystem
        Must have .poss (Ang), .rots (rad), .Ediags (eV).
    siteshift_cubes_path : str
        Path to siteshift_cubes.ini.
    electrostatics_ini_path : str
        Path to electrostatics.ini.
    terms_ini_path : str or None
        Path to site_shift_terms.ini.  If None, only Coulomb is computed.
    base_dir : str
        Base directory for resolving relative paths.
    interp_kind : str
        Interpolation kind for ScalarCube ('linear' or 'cubic').
    chunk : int
        Chunk size for batched evaluation.
    save_json_path : str or None
        If not None, save per-term JSON report to this path.

    Returns
    -------
    result : dict
        Keys: 'coulomb_eV', 'exchange_eV', 'polarization_eV', 'ct_eV', 'total_eV',
        plus 'pair_breakdown' with per-site per-neighbor details.
    """
    mol_ids, diff_paths, gs_paths, orbital_paths = load_siteshift_cubes(siteshift_cubes_path, base_dir)
    n_sites = len(mol_ids)
    if n_sites != len(system.Ediags):
        raise ValueError(
            f"siteshift_cubes.ini row count ({n_sites}) != molecules.ini row count ({len(system.Ediags)})"
        )

    electro_cfg = load_electrostatics_ini(electrostatics_ini_path, base_dir)
    pots = build_hybrid_potentials(electro_cfg, interp_kind=interp_kind)

    poss_ang = np.asarray(system.poss, dtype=float)
    rots_rad = np.asarray(system.rots, dtype=float)
    poss_bohr = poss_ang * ANG_TO_BOHR

    mol_id_to_pose = _build_mol_id_to_pose(mol_ids, poss_bohr, rots_rad)

    terms_cfg = {'enabled_terms': ['coulomb'], 'exchange': {}, 'polarization': {}, 'ct': {}}
    if terms_ini_path is not None:
        terms_cfg = load_site_shift_terms_ini(terms_ini_path)
    enabled = terms_cfg['enabled_terms']

    result = {}
    pair_breakdown = [{} for _ in range(n_sites)]

    if 'coulomb' in enabled:
        coul_eV = np.zeros(n_sites, dtype=float)
        coul_bd = []
        for i_site in range(n_sites):
            mi = int(mol_ids[i_site])
            pose_i = mol_id_to_pose[mi]
            dn_flat, dV, pts_global, _ = _prepare_diff_density_grid(diff_paths[i_site], pose_i)
            npts = pts_global.shape[0]
            V_env = np.zeros(npts, dtype=float)
            nb = []
            for mj, potj in pots.items():
                if int(mj) == int(mi):
                    continue
                pose_j = mol_id_to_pose.get(int(mj))
                if pose_j is None:
                    raise ValueError(f"No pose for mol_id={mj}")
                Rj = pose_j['R']
                pos_j = pose_j['pos_bohr']
                V_part = np.zeros(npts, dtype=float)
                for i0 in range(0, npts, int(chunk)):
                    i1 = min(i0 + int(chunk), npts)
                    p_loc_j = (pts_global[i0:i1] - pos_j[None, :]) @ Rj.T
                    V_part[i0:i1] += potj.eval(p_loc_j)
                V_env += V_part
                shift_pair = -float(np.sum(dn_flat * V_part) * dV) * HARTREE_TO_EV
                nb.append({'env_mol_id': int(mj), 'coulomb_eV': float(shift_pair)})
            shift_h = -float(np.sum(dn_flat * V_env) * dV)
            coul_eV[i_site] = shift_h * HARTREE_TO_EV
            coul_bd.append({'site': int(i_site), 'mol_id': int(mi), 'neighbors': nb})
        result['coulomb_eV'] = coul_eV
        for i_site in range(n_sites):
            pair_breakdown[i_site]['coulomb'] = coul_bd[i_site]['neighbors']

    if 'exchange' in enabled:
        gs_cache = _build_gs_density_cache(mol_ids, gs_paths, electro_cfg, interp_kind=interp_kind)
        k_exch = terms_cfg['exchange'].get('k_exch_eV_bohr3', 7.0)
        exch_eV, exch_bd = _compute_exchange_shifts(
            mol_ids, diff_paths, mol_id_to_pose, gs_cache,
            poss_bohr, rots_rad,
            k_exch_eV_bohr3=k_exch, chunk=chunk)
        result['exchange_eV'] = exch_eV
        for i_site in range(n_sites):
            pair_breakdown[i_site]['exchange'] = exch_bd[i_site]['neighbors']

    if 'polarization' in enabled:
        alpha_cache = _build_alpha_cache(mol_ids, electro_cfg)
        if len(alpha_cache) > 0:
            pol_cfg = terms_cfg['polarization']
            pol_eV, pol_bd = _compute_polarization_shifts(
                mol_ids, diff_paths, mol_id_to_pose, alpha_cache, pots,
                electro_cfg,
                field_method=pol_cfg.get('field_method', 'direct'),
                fd_step_bohr=pol_cfg.get('fd_step_bohr', 0.2),
                r_min_bohr=pol_cfg.get('r_min_bohr', 0.5),
                chunk=chunk)
            result['polarization_eV'] = pol_eV
            for i_site in range(n_sites):
                pair_breakdown[i_site]['polarization'] = pol_bd[i_site]['neighbors']
        else:
            print("INFO [polarization]: No polarizability tensors found in electrostatics.ini; skipping.")
            result['polarization_eV'] = np.zeros(n_sites, dtype=float)

    if 'ct' in enabled:
        ct_eV, ct_bd = _compute_ct_shifts(
            mol_ids, diff_paths, orbital_paths, mol_id_to_pose,
            np.asarray(system.Ediags, dtype=float),
            terms_cfg['ct'], electro_cfg,
            interp_kind=interp_kind, chunk=chunk)
        result['ct_eV'] = ct_eV
        for i_site in range(n_sites):
            pair_breakdown[i_site]['ct'] = ct_bd[i_site]['neighbors']

    total = np.zeros(n_sites, dtype=float)
    for key in ['coulomb_eV', 'exchange_eV', 'polarization_eV', 'ct_eV']:
        if key in result:
            total += result[key]
    result['total_eV'] = total
    result['pair_breakdown'] = pair_breakdown

    if save_json_path is not None:
        report = []
        for i_site in range(n_sites):
            entry = {
                'site_index': int(i_site),
                'mol_id': int(mol_ids[i_site]),
                'E_S1_eV': float(system.Ediags[i_site]),
                'total_eV': float(total[i_site]),
            }
            for key in ['coulomb_eV', 'exchange_eV', 'polarization_eV', 'ct_eV']:
                if key in result:
                    entry[key] = float(result[key][i_site])
            entry['neighbors'] = pair_breakdown[i_site]
            report.append(entry)
        out_json = {
            'site_shifts': report,
            'settings': {
                'terms_enabled': enabled,
                'terms_ini': str(terms_ini_path) if terms_ini_path else None,
            },
        }
        fout = open(save_json_path, 'w')
        json.dump(out_json, fout, indent=2, sort_keys=False)
        fout.close()

    _print_sanity_checks(result, system.Ediags, enabled)

    return result


def _print_sanity_checks(result, Ediags, enabled):
    """Print brief sanity-check diagnostics for each enabled term."""
    n = len(Ediags)
    for term, expected_sign, label in [
        ('exchange_eV', '+', 'Exchange (blue shift expected)'),
        ('polarization_eV', '-', 'Polarization (red shift expected)'),
        ('ct_eV', '-', 'CT mixing (red shift expected)'),
    ]:
        if term not in result:
            continue
        arr = result[term]
        mn, mx = float(np.min(arr)), float(np.max(arr))
        sign_ok = True
        if expected_sign == '+' and mn < -1e-6:
            sign_ok = False
        if expected_sign == '-' and mx > 1e-6:
            sign_ok = False
        tag = 'OK' if sign_ok else 'UNEXPECTED SIGN'
        print(f"  [{label}] min={mn:.6f} eV, max={mx:.6f} eV  ({tag})")

    if 'total_eV' in result:
        total = result['total_eV']
        for i_site in range(n):
            if abs(total[i_site]) > abs(Ediags[i_site]):
                print(f"  WARNING: |Δω_total| ({abs(total[i_site]):.4f} eV) > |E_S1| "
                      f"({abs(Ediags[i_site]):.4f} eV) at site {i_site}")


def compute_site_shifts_for_system(
    system,
    siteshift_cubes_path,
    electrostatics_ini_path,
    base_dir='.',
    interp_kind='cubic',
    chunk=100000,
    save_json_path=None,
):
    mol_ids, diff_paths, gs_paths, orbital_paths = load_siteshift_cubes(siteshift_cubes_path, base_dir)
    if len(mol_ids) != len(system.Ediags):
        raise ValueError(
            f"siteshift_cubes.ini row count ({len(mol_ids)}) does not match molecules.ini row count ({len(system.Ediags)})"
        )

    electro_cfg = load_electrostatics_ini(electrostatics_ini_path, base_dir)
    pots = build_hybrid_potentials(electro_cfg, interp_kind=interp_kind)

    poss_ang = np.asarray(system.poss, dtype=float)
    rots_rad = np.asarray(system.rots, dtype=float)
    poss_bohr = poss_ang * ANG_TO_BOHR

    uniq_mol_ids = sorted(set(int(x) for x in mol_ids))
    mol_id_to_pose = {}
    for mid in uniq_mol_ids:
        idxs = [i for i, mi in enumerate(mol_ids) if int(mi) == int(mid)]
        if len(idxs) == 0:
            continue
        i0 = int(idxs[0])
        mol_id_to_pose[int(mid)] = {
            'pos_bohr': poss_bohr[i0],
            'rot_rad': float(rots_rad[i0]),
            'R': rotmat_z(float(rots_rad[i0])),
        }

    shifts_eV = np.zeros(len(mol_ids), dtype=float)
    report = []

    for i_site, diff_path in enumerate(diff_paths):
        mi = int(mol_ids[i_site])
        cube_dn = load_cube_with_atoms(diff_path)
        center_i = np.mean(cube_dn['atom_pos'], axis=0)
        cube_dn['origin'] = cube_dn['origin'] - center_i
        cube_dn['atom_pos'] = cube_dn['atom_pos'] - center_i[None, :]

        cube_dn['density'] = enforce_zero_integral_lobe_scaling(cube_dn['density'], cube_dn['dV'])

        xs, ys, zs = make_grid_axes(cube_dn)
        XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing='ij')
        pts_i_local = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=-1)
        dn_flat = np.asarray(cube_dn['density'], dtype=float).ravel()
        dV = float(cube_dn['dV'])

        pose_i = mol_id_to_pose.get(mi, None)
        if pose_i is None:
            raise ValueError(f"No pose found for mol_id={mi} (needed for site-state {i_site})")
        Ri = pose_i['R']
        pos_i = pose_i['pos_bohr']

        npts = int(pts_i_local.shape[0])
        V_env = np.zeros(npts, dtype=float)

        pts_global = pts_i_local @ Ri
        pts_global = pts_global + pos_i[None, :]

        for mj, potj in pots.items():
            if int(mj) == int(mi):
                continue
            pose_j = mol_id_to_pose.get(int(mj), None)
            if pose_j is None:
                raise ValueError(f"No pose found for mol_id={mj} (required by electrostatics.ini)")
            Rj = pose_j['R']
            pos_j = pose_j['pos_bohr']

            for i0 in range(0, npts, int(chunk)):
                i1 = min(i0 + int(chunk), npts)
                p_loc_j = (pts_global[i0:i1] - pos_j[None, :]) @ Rj.T
                V_env[i0:i1] = V_env[i0:i1] + potj.eval(p_loc_j)

        shift_h = -float(np.sum(dn_flat * V_env) * dV)
        shift_eV = shift_h * HARTREE_TO_EV
        shifts_eV[i_site] = shift_eV

        report.append({
            'site_index': int(i_site),
            'mol_id': int(mi),
            'diff_density_cube': str(diff_path),
            'delta_omega_eV': float(shift_eV),
            'E0_eV': float(system.Ediags[i_site]),
            'Eshifted_eV': float(system.Ediags[i_site] + shift_eV),
        })

    if save_json_path is not None:
        out = {
            'site_shifts': report,
        }
        fout = open(save_json_path, 'w')
        json.dump(out, fout, indent=2, sort_keys=True)
        fout.close()

    return shifts_eV


def apply_site_shifts_to_system(
    system,
    siteshift_cubes_path,
    electrostatics_ini_path,
    base_dir='.',
    interp_kind='cubic',
    chunk=100000,
    save_json_path=None,
):
    shifts = compute_site_shifts_for_system(
        system,
        siteshift_cubes_path,
        electrostatics_ini_path,
        base_dir=base_dir,
        interp_kind=interp_kind,
        chunk=chunk,
        save_json_path=save_json_path,
    )
    system.Ediags = np.asarray(system.Ediags, dtype=float) + np.asarray(shifts, dtype=float)
    return shifts
