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


class PotentialCube:
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
    mol_ids = []
    diff_paths = []
    gs_paths = []
    fin = open(fname, 'r')
    for line in fin:
        s = line.strip()
        if s == '':
            continue
        if s[0] == '#':
            continue
        w = s.split()
        if len(w) < 3:
            raise ValueError(f"Expected 3 columns (mol_id diff_cube gs_cube) in {fname}, got line: {line}")
        mol_ids.append(int(w[0]))
        diff_paths.append(_resolve_path(base_dir, w[1]))
        gs_paths.append(_resolve_path(base_dir, w[2]))
    fin.close()
    return mol_ids, diff_paths, gs_paths


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


def compute_site_shifts_for_system(
    system,
    siteshift_cubes_path,
    electrostatics_ini_path,
    base_dir='.',
    interp_kind='cubic',
    chunk=100000,
    save_json_path=None,
):
    mol_ids, diff_paths, _ = load_siteshift_cubes(siteshift_cubes_path, base_dir)
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
