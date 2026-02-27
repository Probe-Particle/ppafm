#!/usr/bin/env python3

import os
import sys
import argparse
import re
import numpy as np
from scipy.interpolate import RegularGridInterpolator

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from site_energy_shift import (
    load_cube_with_atoms,
    make_grid_axes,
    BOHR_TO_ANG,
    ANG_TO_BOHR,
    extract_multipoles_about,
    nuclear_multipoles_about,
    multipole_potential,
)


def classify_cube(cube, integral_e=None):
    h = (str(cube.get('header1', '')) + " " + str(cube.get('header2', ''))).lower()
    if ('potential' in h) or ('esp' in h) or ('electrostatic' in h) or ('mep' in h):
        return 'potential'
    if integral_e is None:
        integral_e = float(np.sum(cube['density']) * cube['dV'])
    if ('electron density' in h) or ('density' in h):
        if abs(float(integral_e)) < 1e-2:
            return 'signed_density'
        return 'density'
    if abs(float(integral_e)) < 1e-2:
        return 'signed_density'
    return 'unknown'


def cube_integral(cube):
    return float(np.sum(cube["density"]) * cube["dV"])


def cube_stats(arr):
    arr = np.asarray(arr)
    mn = float(arr.min())
    mx = float(arr.max())
    mean = float(arr.mean())
    rms = float(np.sqrt(np.mean(arr * arr)))
    frac_pos = float(np.mean(arr > 0.0))
    frac_neg = float(np.mean(arr < 0.0))
    return mn, mx, mean, rms, frac_pos, frac_neg


def format_stats(st):
    mn, mx, mean, rms, fp, fn = st
    return (
        f"min {mn:+.3e}  max {mx:+.3e}  mean {mean:+.3e}  "
        f"rms {rms:.3e}  frac(+) {fp:.3f} frac(-) {fn:.3f}"
    )


def cube_lengths_ang(cube):
    nx, ny, nz = (int(cube["npts"][0]), int(cube["npts"][1]), int(cube["npts"][2]))
    dx = float(abs(cube["axes"][0, 0]))
    dy = float(abs(cube["axes"][1, 1]))
    dz = float(abs(cube["axes"][2, 2]))
    Lx = (nx - 1) * dx * BOHR_TO_ANG
    Ly = (ny - 1) * dy * BOHR_TO_ANG
    Lz = (nz - 1) * dz * BOHR_TO_ANG
    return Lx, Ly, Lz


def corr(a, b, mask=None):
    a = np.asarray(a)
    b = np.asarray(b)
    if mask is not None:
        a = a[mask]
        b = b[mask]
    a = a.ravel()
    b = b.ravel()
    a0 = a - a.mean()
    b0 = b - b.mean()
    denom = float(np.sqrt(np.mean(a0 * a0) * np.mean(b0 * b0)))
    return float(np.mean(a0 * b0) / denom) if denom > 0.0 else float("nan")


def rms(x, mask=None):
    x = np.asarray(x)
    if mask is not None:
        x = x[mask]
    return float(np.sqrt(np.mean(x * x)))


def resample_density_to_target(source_cube, target_cube):
    if (not is_orthogonal_axes(source_cube)) or (not is_orthogonal_axes(target_cube)):
        raise ValueError('resample_density_to_target requires orthogonal cube axes (diagonal axes matrix)')
    xs, ys, zs = make_grid_axes(source_cube)
    xt, yt, zt = make_grid_axes(target_cube)
    itp = RegularGridInterpolator(
        (xs, ys, zs), source_cube["density"], bounds_error=False, fill_value=0.0
    )
    XX, YY, ZZ = np.meshgrid(xt, yt, zt, indexing="ij")
    pts = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=-1)
    out = itp(pts).reshape(target_cube["density"].shape)
    return out


def is_orthogonal_axes(cube, tol=1e-12):
    a = np.asarray(cube['axes'], dtype=float)
    off = a.copy()
    off[0, 0] = 0.0
    off[1, 1] = 0.0
    off[2, 2] = 0.0
    return bool(np.max(np.abs(off)) < float(tol))


def _cube_bounds_bohr(cube):
    origin = np.asarray(cube['origin'], dtype=float)
    npts = np.asarray(cube['npts'], dtype=int)
    axes = np.asarray(cube['axes'], dtype=float)

    nx, ny, nz = int(npts[0]), int(npts[1]), int(npts[2])
    ax = axes[0, :]
    ay = axes[1, :]
    az = axes[2, :]
    corners = []
    for ix in [0, nx - 1]:
        for iy in [0, ny - 1]:
            for iz in [0, nz - 1]:
                corners.append(origin + ix * ax + iy * ay + iz * az)
    corners = np.asarray(corners, dtype=float)
    mins = np.min(corners, axis=0)
    maxs = np.max(corners, axis=0)
    return mins, maxs


def _fit_loglog_exponent(R, V):
    R = np.asarray(R, dtype=float)
    V = np.asarray(V, dtype=float)
    mask = (R > 1e-12) & (np.abs(V) > 0.0)
    if int(np.sum(mask)) < 3:
        return float('nan')
    x = np.log(R[mask])
    y = np.log(np.abs(V[mask]))
    a, _b = np.polyfit(x, y, 1)
    return float(-a)


def _fit_beta_q_over_R(R, V):
    R = np.asarray(R, dtype=float)
    V = np.asarray(V, dtype=float)
    A = np.vstack([np.ones_like(R), 1.0 / R]).T
    x, _res, _rank, _s = np.linalg.lstsq(A, V, rcond=None)
    beta = float(x[0])
    q = float(x[1])
    return beta, q


def _random_unit_vectors(n, seed=0):
    rng = np.random.default_rng(int(seed))
    v = rng.normal(size=(int(n), 3))
    nrm = np.linalg.norm(v, axis=1)
    nrm = np.maximum(nrm, 1e-12)
    return v / nrm[:, None]


def _ray_box_tmax(center, direction, box_min, box_max):
    tmax = np.inf
    for i in range(3):
        ui = float(direction[i])
        if abs(ui) < 1e-14:
            continue
        if ui > 0.0:
            ti = (float(box_max[i]) - float(center[i])) / ui
        else:
            ti = (float(box_min[i]) - float(center[i])) / ui
        if ti < tmax:
            tmax = ti
    return float(tmax)


def _min_dist_to_atoms(points, atom_pos):
    p = np.asarray(points, dtype=float)
    a = np.asarray(atom_pos, dtype=float)
    d = p[:, None, :] - a[None, :, :]
    r2 = np.sum(d * d, axis=2)
    return np.sqrt(np.min(r2, axis=1))


def _fit_multipoles_from_potential(points, values, center):
    pts = np.asarray(points, dtype=float)
    V = np.asarray(values, dtype=float)
    Rvec = pts - np.asarray(center, dtype=float)[None, :]
    R = np.linalg.norm(Rvec, axis=1)
    R = np.maximum(R, 1e-12)
    rhat = Rvec / R[:, None]

    Rx, Ry, Rz = Rvec[:, 0], Rvec[:, 1], Rvec[:, 2]
    R5 = R**5

    cols = []
    cols.append(np.ones_like(R))
    cols.append(1.0 / R)
    cols.append(-rhat[:, 0] / (R**2))
    cols.append(-rhat[:, 1] / (R**2))
    cols.append(-rhat[:, 2] / (R**2))
    cols.append(0.5 * (Rx * Rx) / R5)
    cols.append((Rx * Ry) / R5)
    cols.append((Rx * Rz) / R5)
    cols.append(0.5 * (Ry * Ry) / R5)
    cols.append((Ry * Rz) / R5)
    cols.append(0.5 * (Rz * Rz) / R5)
    A = np.vstack(cols).T

    x, _res, _rank, _s = np.linalg.lstsq(A, V, rcond=None)
    beta = float(x[0])
    q = float(x[1])
    mu = np.array([float(x[2]), float(x[3]), float(x[4])], dtype=float)
    Q = np.array(
        [
            [float(x[5]), float(x[6]), float(x[7])],
            [float(x[6]), float(x[8]), float(x[9])],
            [float(x[7]), float(x[9]), float(x[10])],
        ],
        dtype=float,
    )
    V_fit = A @ x
    err = V - V_fit
    return beta, q, mu, Q, float(np.sqrt(np.mean(err * err))), float(np.max(np.abs(err)))


def audit_poisson_equation(
    density_cube_path,
    potential_cube_path,
    exclude_nuc_radius_ang=0.5,
    edge_pad=1,
):
    dens = load_cube_with_atoms(density_cube_path)
    pot = load_cube_with_atoms(potential_cube_path)

    if (not is_orthogonal_axes(pot)):
        raise ValueError('Potential-cube audit currently requires orthogonal axes in the potential cube')

    dens_I = float(np.sum(dens['density']) * dens['dV'])
    dens_type = classify_cube(dens, integral_e=dens_I)
    if dens_type != 'density':
        print("\nWARNING: Poisson audit expects an electron density cube (integral ~ N_e).")
        print(f"Detected density cube type: {dens_type}")

    if not (
        np.all(pot['npts'] == dens['npts'])
        and np.allclose(pot['axes'], dens['axes'])
        and np.allclose(pot['origin'], dens['origin'])
    ):
        dens_grid = {
            'density': resample_density_to_target(dens, pot),
            'atom_pos': dens['atom_pos'],
            'atom_Z': dens['atom_Z'],
        }
    else:
        dens_grid = {
            'density': np.asarray(dens['density'], dtype=float),
            'atom_pos': dens['atom_pos'],
            'atom_Z': dens['atom_Z'],
        }

    axes = np.asarray(pot['axes'], dtype=float)
    ortho = (
        abs(float(axes[0, 1])) < 1e-12 and abs(float(axes[0, 2])) < 1e-12
        and abs(float(axes[1, 0])) < 1e-12 and abs(float(axes[1, 2])) < 1e-12
        and abs(float(axes[2, 0])) < 1e-12 and abs(float(axes[2, 1])) < 1e-12
    )
    if not ortho:
        raise ValueError('Poisson audit currently requires orthogonal cube axes (diagonal axes matrix)')

    dx = float(abs(axes[0, 0]))
    dy = float(abs(axes[1, 1]))
    dz = float(abs(axes[2, 2]))
    V = np.asarray(pot['density'], dtype=float)
    n = np.asarray(dens_grid['density'], dtype=float)

    nx, ny, nz = (int(pot['npts'][0]), int(pot['npts'][1]), int(pot['npts'][2]))
    pad = int(edge_pad)
    if (nx <= 2 * pad + 2) or (ny <= 2 * pad + 2) or (nz <= 2 * pad + 2):
        raise ValueError('Cube is too small for Poisson audit with the requested edge_pad')

    lap = (
        (V[2:, 1:-1, 1:-1] - 2.0 * V[1:-1, 1:-1, 1:-1] + V[:-2, 1:-1, 1:-1]) / (dx * dx)
        + (V[1:-1, 2:, 1:-1] - 2.0 * V[1:-1, 1:-1, 1:-1] + V[1:-1, :-2, 1:-1]) / (dy * dy)
        + (V[1:-1, 1:-1, 2:] - 2.0 * V[1:-1, 1:-1, 1:-1] + V[1:-1, 1:-1, :-2]) / (dz * dz)
    )
    n_mid = n[1:-1, 1:-1, 1:-1]

    origin = np.asarray(pot['origin'], dtype=float)
    xs = origin[0] + np.arange(1, nx - 1) * axes[0, 0]
    ys = origin[1] + np.arange(1, ny - 1) * axes[1, 1]
    zs = origin[2] + np.arange(1, nz - 1) * axes[2, 2]
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing='ij')
    pts = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=-1)

    rmin_atoms = _min_dist_to_atoms(pts, np.asarray(dens_grid['atom_pos'], dtype=float))
    r_excl = float(exclude_nuc_radius_ang) * ANG_TO_BOHR
    mask = rmin_atoms > r_excl
    mask3 = mask.reshape(lap.shape)

    resid_plus = lap - (4.0 * np.pi * n_mid)
    resid_minus = lap + (4.0 * np.pi * n_mid)
    resid_plus_m = resid_plus[mask3]
    resid_minus_m = resid_minus[mask3]
    if resid_plus_m.size < 10:
        raise ValueError('Poisson audit mask removed too many points; decrease exclude radius or increase cube size')

    rms_plus = float(np.sqrt(np.mean(resid_plus_m * resid_plus_m)))
    rms_minus = float(np.sqrt(np.mean(resid_minus_m * resid_minus_m)))
    if rms_plus <= rms_minus:
        resid_masked = resid_plus_m
        sign = '+4πn'
        rms_best = rms_plus
    else:
        resid_masked = resid_minus_m
        sign = '-4πn'
        rms_best = rms_minus

    out = {
        'exclude_nuc_radius_ang': float(exclude_nuc_radius_ang),
        'n_points': int(resid_masked.size),
        'best_sign': str(sign),
        'rms_resid': float(rms_best),
        'max_abs_resid': float(np.max(np.abs(resid_masked))),
        'rms_lap': float(np.sqrt(np.mean(lap[mask3] * lap[mask3]))),
        'rms_4pi_n': float(np.sqrt(np.mean((4.0 * np.pi * n_mid[mask3]) ** 2))),
        'rms_resid_plus': float(rms_plus),
        'rms_resid_minus': float(rms_minus),
    }

    print("\n==================== POISSON CONSISTENCY AUDIT ====================")
    print(f"density cube   : {density_cube_path}")
    print(f"potential cube : {potential_cube_path}")
    print("Tested relation (away from nuclei):  ∇² V_total(r) ≈ ±4π n(r)")
    print(f"exclude radius : {float(exclude_nuc_radius_ang):.3f} Å")
    print(f"points used    : {out['n_points']}")
    print(f"best sign      : {out['best_sign']}")
    print(f"RMS(best resid): {out['rms_resid']:.6e} Ha/Bohr^2")
    print(f"RMS(resid for +4πn): {out['rms_resid_plus']:.6e} Ha/Bohr^2")
    print(f"RMS(resid for -4πn): {out['rms_resid_minus']:.6e} Ha/Bohr^2")
    print(f"max|resid|     : {out['max_abs_resid']:.6e} Ha/Bohr^2")
    print(f"RMS(∇²V)       : {out['rms_lap']:.6e} Ha/Bohr^2")
    print(f"RMS(4πn)       : {out['rms_4pi_n']:.6e} Ha/Bohr^2")

    return out


def parse_qc_log_text(text):
    out = {}
    patterns = {
        'charge': r'^\s*Charge:\s*([-+]?\d+)',
        'molecular_charge': r'^\s*Molecular charge:\s*([-+]?\d+)',
        'n_electrons': r'^\s*Number of electrons:\s*(\d+)',
        'nelec_numeric': r'^\s*nelec by numeric integration\s*=\s*\[([^\]]+)\]',
        'expected_electrons': r'^\s*Expected electrons:\s*([0-9\.]+)',
        'total_electrons': r'^\s*Total electrons:\s*([0-9\.]+)',
    }
    for line in str(text).splitlines():
        for k, pat in patterns.items():
            m = re.match(pat, line)
            if m:
                out[k] = m.group(1).strip()

    if 'nelec_numeric' in out:
        num_re = re.compile(r'^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$')
        toks = [t for t in re.split(r'[\s,]+', out['nelec_numeric'].strip()) if t]
        parts = [float(t) for t in toks if num_re.match(t)]
        if len(parts) >= 2:
            out['nelec_numeric_total'] = f"{(parts[0] + parts[1]):.8f}"
    return out


def audit_potential_cube(
    density_cube_path,
    potential_cube_path,
    expected_charge=None,
    axis='z',
    rmin_ang=8.0,
    nsamples=25,
    fit_samples=400,
    fit_min_atom_dist_ang=2.0,
):
    dens = load_cube_with_atoms(density_cube_path)
    pot = load_cube_with_atoms(potential_cube_path)

    if not is_orthogonal_axes(pot):
        raise ValueError('Potential-cube audit currently requires orthogonal axes in the potential cube')

    about = np.mean(dens['atom_pos'], axis=0)
    Z_total = float(np.sum(dens['atom_Z']))
    Ne_box = float(np.sum(dens['density']) * dens['dV'])
    q_from_box = float(Z_total - Ne_box)

    print("\n==================== POTENTIAL-CUBE AUDIT ====================")
    print(f"density cube   : {density_cube_path}")
    print(f"potential cube : {potential_cube_path}")
    print(f"Z_total        : {Z_total:.0f}")
    print(f"∫n dV (box)    : {Ne_box:.8f} e")
    print(f"q (Z-∫n)       : {q_from_box:+.8f} e")
    if expected_charge is not None:
        expected_charge = float(expected_charge)
        print(f"q_expected     : {expected_charge:+.8f} e")
        print(f"Δq_box         : {q_from_box - expected_charge:+.8f} e")

    xs, ys, zs = make_grid_axes(pot)
    Vgrid = np.asarray(pot['density'], dtype=float)
    itp = RegularGridInterpolator((xs, ys, zs), Vgrid, method='linear', bounds_error=False, fill_value=0.0)

    axis = axis.lower()
    axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
    rmin_bohr = float(rmin_ang) * ANG_TO_BOHR

    pot_min, pot_max = _cube_bounds_bohr(pot)
    center = np.asarray(about, dtype=float)
    r_plus = float(pot_max[axis_idx] - center[axis_idx])
    r_minus = float(center[axis_idx] - pot_min[axis_idx])
    r_max = float(max(0.0, min(r_plus, r_minus)))
    if r_max <= 1e-12:
        raise ValueError(f"Potential cube has effectively zero extent along {axis}")
    if r_max <= rmin_bohr:
        rmin_new = 0.35 * r_max
        print(
            f"\nNOTE: potential cube is too small for requested --audit-rmin-ang={float(rmin_ang):.2f} Å along {axis}. "
            f"Using rmin={rmin_new * BOHR_TO_ANG:.2f} Å instead (cube max reachable is {r_max * BOHR_TO_ANG:.2f} Å)."
        )
        rmin_bohr = float(rmin_new)

    Rs = np.linspace(rmin_bohr, 0.98 * r_max, int(nsamples))
    pts_plus = np.repeat(center[None, :], Rs.size, axis=0)
    pts_minus = np.repeat(center[None, :], Rs.size, axis=0)
    pts_plus[:, axis_idx] += Rs
    pts_minus[:, axis_idx] -= Rs

    V_plus = np.asarray(itp(pts_plus), dtype=float)
    V_minus = np.asarray(itp(pts_minus), dtype=float)
    V_axis = 0.5 * (V_plus + V_minus)

    n_raw = _fit_loglog_exponent(Rs, V_axis)

    n_tail = int(max(6, Rs.size // 3))
    R_tail = Rs[-n_tail:]
    V_tail = V_axis[-n_tail:]
    beta_fit, q_fit = _fit_beta_q_over_R(R_tail, V_tail)
    V_model = beta_fit + q_fit / Rs
    V_res = V_axis - V_model
    n_res = _fit_loglog_exponent(Rs, V_res)

    print("\n-- Far-field behaviour along axis --")
    print(f"axis           : {axis}")
    print(f"R range        : {Rs[0] * BOHR_TO_ANG:.2f} .. {Rs[-1] * BOHR_TO_ANG:.2f} Å")
    print(f"tail fit model : V(R) = beta + q/R  using last {n_tail} samples")
    print(f"beta_fit       : {beta_fit:+.6e} Ha")
    print(f"q_fit          : {q_fit:+.6e} e")
    if expected_charge is not None:
        print(f"q_expected     : {float(expected_charge):+.6e} e")
        print(f"q_fit-q_expected: {q_fit - float(expected_charge):+.6e} e")
    print(f"log-log exponent n_raw for |V| ~ 1/R^n          : {n_raw:.3f}")
    print(f"log-log exponent n_res for |V-(beta+q/R)| ~ 1/R^n: {n_res:.3f}")

    fit_min_atom_dist_bohr = float(fit_min_atom_dist_ang) * ANG_TO_BOHR
    fit_dirs = _random_unit_vectors(int(max(50, fit_samples * 2)), seed=0)
    pts_fit = []
    for u in fit_dirs:
        tmax = _ray_box_tmax(center, u, pot_min, pot_max)
        if not np.isfinite(tmax):
            continue
        p = center + (0.98 * float(tmax)) * u
        if float(np.linalg.norm(p - center)) <= rmin_bohr:
            continue
        pts_fit.append(p)
        if len(pts_fit) >= int(fit_samples):
            break
    pts_fit = np.asarray(pts_fit, dtype=float)
    if pts_fit.shape[0] < 20:
        raise ValueError(f"Not enough fit points inside cube for multipole fit: {pts_fit.shape[0]}")

    rmin_atoms = _min_dist_to_atoms(pts_fit, dens['atom_pos'])
    keep = rmin_atoms > fit_min_atom_dist_bohr
    pts_fit = pts_fit[keep]
    if pts_fit.shape[0] < 20:
        raise ValueError(
            f"Not enough fit points after atom-distance filtering: kept {pts_fit.shape[0]} (min dist {fit_min_atom_dist_ang:.2f} Å)"
        )
    V_fit_pts = np.asarray(itp(pts_fit), dtype=float)
    beta_mp, q_mp, mu_mp, Q_mp, rms_mp, max_mp = _fit_multipoles_from_potential(pts_fit, V_fit_pts, center)

    print("\n-- Potential-only multipole fit (from electrostatic_potential.cube) --")
    print(f"fit points used: {int(pts_fit.shape[0])}")
    print(f"atom-distance filter min: {fit_min_atom_dist_ang:.2f} Å")
    print(f"beta_fit(pot): {beta_mp:+.6e} Ha")
    print(f"q_fit(pot)   : {q_mp:+.6e} e")
    print(f"|mu_fit(pot)|: {float(np.linalg.norm(mu_mp)):.6e} e·Bohr")
    print(f"tr(Q_fit)    : {float(np.trace(Q_mp)):+.6e} e·Bohr^2")
    if expected_charge is not None:
        print(f"q_fit(pot)-q_expected: {q_mp - float(expected_charge):+.6e} e")
        print(f"q_fit(pot)-q_from_box : {q_mp - float(q_from_box):+.6e} e")
    print(f"RMS fit error : {rms_mp:.6e} Ha")
    print(f"max|error|    : {max_mp:.6e} Ha")

    mom_nuc = nuclear_multipoles_about(dens['atom_pos'], dens['atom_Z'], about)
    mom_e = extract_multipoles_about(dens['density'], dens['origin'], dens['axes'], dens['npts'], about)
    moments_net = {
        'monopole': float(mom_nuc['monopole'] - mom_e['monopole']),
        'dipole': mom_nuc['dipole'] - mom_e['dipole'],
        'quadrupole': mom_nuc['quadrupole'] - mom_e['quadrupole'],
        'center': np.asarray(about, dtype=float),
    }

    print("\n-- Net multipoles from density cube (box-integrated) --")
    mu = np.asarray(moments_net['dipole'], dtype=float)
    print(f"q_net(box)     : {moments_net['monopole']:+.8f} e")
    print(f"|mu_net|       : {float(np.linalg.norm(mu)):.6e} e·Bohr")

    V_mp = np.zeros_like(V_axis)
    for i, R in enumerate(Rs):
        Rvec = np.zeros(3, dtype=float)
        Rvec[axis_idx] = float(R)
        V_mp[i] = float(multipole_potential(moments_net, Rvec))

    beta = float(V_axis[-1] - V_mp[-1])
    err = V_axis - (V_mp + beta)
    print("\n-- Multipole-vs-cube comparison (offset-aligned at largest R) --")
    print(f"beta (V_cube - V_mp at largest R): {beta:+.6e} Ha")
    print(f"RMS(V_cube - (V_mp+beta)) over samples: {float(np.sqrt(np.mean(err * err))):.6e} Ha")
    print(f"max|error| over samples               : {float(np.max(np.abs(err))):.6e} Ha")

    i0 = 0
    i1 = int(Rs.size // 2)
    i2 = int(Rs.size - 1)
    for i in [i0, i1, i2]:
        print(
            f"  R={Rs[i] * BOHR_TO_ANG:6.2f} Å  V_cube={V_axis[i]:+.6e}  V_mp+beta={V_mp[i] + beta:+.6e}  err={err[i]:+.3e}"
        )


def report_cube(path, expected_charge=None, label=None):
    cube = load_cube_with_atoms(path)

    if label is None:
        label = os.path.basename(path)

    npts = tuple(int(x) for x in cube["npts"])
    origin = np.array(cube["origin"], dtype=float)
    axes = np.array(cube["axes"], dtype=float)
    dV = float(cube["dV"])
    I = cube_integral(cube)
    cube_type = classify_cube(cube, integral_e=I)
    Lx, Ly, Lz = cube_lengths_ang(cube)

    print(f"\n=== {label} ===")
    print(f"path   : {path}")
    print(f"header : {cube['header1']} | {cube['header2']}")
    print(f"npts   : {npts}")
    print(f"origin : {origin}")
    print(
        "steps  : diag(" + ", ".join(f"{axes[i, i]:.6e}" for i in range(3)) + ") Bohr"
    )
    print(f"dV     : {dV:.6e} Bohr^3")
    print(f"box L  : ({Lx:.2f}, {Ly:.2f}, {Lz:.2f}) Ang")
    if cube_type == 'density':
        print(f"∫dens dV: {I:+.8f} e")
    elif cube_type == 'signed_density':
        print(f"∫dens dV: {I:+.8f} e  (signed density; often expected ~0)")
    else:
        print(f"∫grid dV: {I:+.8f}  (not an electron count)")
    print(f"stats  : {format_stats(cube_stats(cube['density']))}")
    print(f"type   : {cube_type}")

    if np.any(~np.isfinite(np.asarray(cube['density'], dtype=float))):
        print("WARNING: cube contains NaN/Inf values")

    mins, maxs = _cube_bounds_bohr(cube)
    apos = np.asarray(cube['atom_pos'], dtype=float)
    inside = np.all((apos >= mins[None, :]) & (apos <= maxs[None, :]), axis=1)
    if not bool(np.all(inside)):
        print(f"WARNING: {int(np.sum(~inside))} atoms lie outside the cube bounds")
    else:
        margin = np.min(np.vstack([apos - mins[None, :], maxs[None, :] - apos]), axis=0)
        margin_min = float(np.min(margin))
        print(f"min atom-to-box margin: {margin_min * BOHR_TO_ANG:.3f} Å")

    if cube_type == 'density':
        mn, _mx, _mean, _rms, _fp, fn = cube_stats(cube['density'])
        if (mn < -1e-10) and (fn > 1e-6):
            print("WARNING: density cube has significant negative values; check conventions/units")

    if cube_type == 'signed_density':
        if abs(float(I)) > 1e-2:
            print("WARNING: signed-density cube has non-negligible integral; for TDM/Δn this is often expected to be ~0")

    if expected_charge is not None:
        Z_total = float(np.sum(cube["atom_Z"]))
        N_target = Z_total - float(expected_charge)
        if cube_type == 'density':
            print(f"Z_total                 : {Z_total:.0f}")
            print(f"expected charge         : {float(expected_charge):+.6f}")
            print(f"target ∫n dV            : {N_target:.8f} e")
            print(f"error (∫dens dV - target): {I - N_target:+.8f} e")
        else:
            print("NOTE: expected-charge check is only meaningful for true electron density cubes")

    return cube


def dir_summary(dir_path, pattern='*.cube'):
    from pathlib import Path

    print("\n==================== DIRECTORY SUMMARY ====================")
    print(f"dir     : {dir_path}")
    print(f"pattern : {pattern}")
    paths = sorted(str(pp) for pp in Path(dir_path).glob(pattern))
    print(f"n_files : {len(paths)}")
    if len(paths) == 0:
        return
    print("\nfile\ttype\tnpts\tintegral\tmin\tmax\trms")
    for path in paths:
        c = load_cube_with_atoms(path)
        I = float(np.sum(c['density']) * c['dV'])
        ctype = classify_cube(c, integral_e=I)
        mn, mx, _mean, rr, _fp, _fn = cube_stats(c['density'])
        npts = tuple(int(x) for x in c['npts'])
        print(
            f"{os.path.basename(path)}\t{ctype}\t{npts[0]}x{npts[1]}x{npts[2]}\t{I:+.4e}\t{mn:+.3e}\t{mx:+.3e}\t{rr:.3e}"
        )


def compare_es_gs_diff(gs_path, es_path, diff_path=None, mask_gs_below=None):
    print("\n==================== ES/GS/DIFF CONSISTENCY ====================")

    gs = load_cube_with_atoms(gs_path)
    es = load_cube_with_atoms(es_path)

    gs_on_es = resample_density_to_target(gs, es)
    comp = es["density"] - gs_on_es

    print("\n-- Integrals on ES grid --")
    print(f"∫GS_resampled dV      : {float(np.sum(gs_on_es) * es['dV']):+.8f} e")
    print(f"∫ES dV                : {cube_integral(es):+.8f} e")
    print(f"∫(ES - GS_resamp) dV  : {float(np.sum(comp) * es['dV']):+.8f} e")
    print(f"stats(ES - GS_resamp) : {format_stats(cube_stats(comp))}")

    if diff_path is None:
        return

    diff = load_cube_with_atoms(diff_path)
    diff_on_es = diff["density"]
    if diff_on_es.shape != es["density"].shape:
        diff_on_es = resample_density_to_target(diff, es)

    print("\n-- DIFF file on ES grid --")
    print(f"∫diff_file dV         : {float(np.sum(diff_on_es) * es['dV']):+.8f} e")
    print(f"stats(diff_file)      : {format_stats(cube_stats(diff_on_es))}")

    err_minus = diff_on_es - comp
    err_plus = diff_on_es + comp

    print("\n-- Pointwise match --")
    print(f"corr(diff_file, (ES-GS_resamp))          : {corr(diff_on_es, comp):+.6f}")
    print(f"rms(ES-GS_resamp)                        : {rms(comp):.6e}")
    print(f"rms(diff_file - (ES-GS_resamp))          : {rms(err_minus):.6e}")
    print(f"rms(diff_file + (ES-GS_resamp))          : {rms(err_plus):.6e}")

    es_pred = gs_on_es + diff_on_es
    err_recon = es["density"] - es_pred
    print("\n-- Reconstruction check: ES ?= GS + diff_file --")
    print(f"rms(ES - (GS_resamp + diff_file))        : {rms(err_recon):.6e}")

    if mask_gs_below is not None:
        for thr in mask_gs_below:
            thr = float(thr)
            mask = gs_on_es < thr
            frac = float(np.mean(mask))
            print(f"\nMask: GS_resamp < {thr:.6g}  (vox frac {frac:.3f})")
            print(f"  corr(diff_file, ES-GS_resamp) masked   : {corr(diff_on_es, comp, mask):+.6f}")
            print(f"  rms(diff_file - (ES-GS_resamp)) masked : {rms(err_minus, mask):.6e}")
            print(f"  rms(ES - (GS + diff)) masked           : {rms(err_recon, mask):.6e}")


def main():
    p = argparse.ArgumentParser(
        description=(
            "Audit Gaussian cube files using the existing site_energy_shift.py parser. "
            "Print integrals, basic stats, and optionally compare DIFF vs (ES-GS) with resampling."
        )
    )

    p.add_argument("--cube", action="append", default=[], help="Path to cube file (repeatable)")
    p.add_argument("--dir", type=str, default=None, help="Directory to scan for cube files")
    p.add_argument("--pattern", type=str, default="*.cube", help="Glob pattern used with --dir")
    p.add_argument("--no-dir-scan", action="store_true", help="If set, do not list/report all cubes in --dir (useful with --audit-potential)",)
    p.add_argument("--dir-summary", action="store_true", help="Print one-line summary per cube in --dir (recommended for large directories)",)
    p.add_argument("--expected-charge", type=float, default=None, help="If set, print electron-count error vs Z_total - charge",)
    p.add_argument("--gs", type=str, default=None, help="Ground-state density cube (for ES/GS/DIFF compare)")
    p.add_argument("--es", type=str, default=None, help="Excited-state density cube (for ES/GS/DIFF compare)")
    p.add_argument("--diff", type=str, default=None, help="Difference density cube (for ES/GS/DIFF compare)")
    p.add_argument("--mask-gs-below", type=float, nargs="*", default=None, help="If provided (one or more thresholds), also report correlations/errors only in regions where GS_resampled < threshold.")
    p.add_argument("--audit-potential", action="store_true", help="Audit electrostatic potential cube vs density cube")
    p.add_argument("--density", type=str, default=None, help="Electron density cube for potential audit")
    p.add_argument("--potential", type=str, default=None, help="Electrostatic potential cube for potential audit")
    p.add_argument("--audit-axis", type=str, default="z", choices=["x", "y", "z"], help="Axis used for far-field sampling",)
    p.add_argument("--audit-rmin-ang", type=float, default=8.0, help="Minimum radius (Å) for far-field sampling",)
    p.add_argument("--audit-nsamples", type=int, default=25, help="Number of samples along the axis",)
    p.add_argument("--audit-fit-samples", type=int, default=400, help="Number of random boundary points used to fit multipoles from potential cube",)
    p.add_argument("--audit-fit-min-atom-dist-ang", type=float, default=2.0, help="Reject fit points closer than this to any nucleus (Å)",)
    p.add_argument("--audit-poisson", action="store_true", help="Check Poisson consistency between potential cube and density cube away from nuclei",)
    p.add_argument("--poisson-exclude-nuc-radius-ang", type=float, default=0.5, help="Exclude points closer than this to any nucleus (Å)",)
    p.add_argument("--poisson-edge-pad", type=int, default=1, help="Number of grid layers excluded at boundaries for finite differences",)
    p.add_argument("--parse-log", action="store_true", help="Parse a QC log file for charge/electron counts (or auto-pick newest *.log in --dir)",)
    p.add_argument("--log", type=str, default=None, help="Path to QC log file (optional if using --dir with --parse-log)",)

    args = p.parse_args()

    if args.dir_summary:
        if args.dir is None:
            raise ValueError('For --dir-summary you must provide --dir')
        dir_summary(args.dir, pattern=args.pattern)

    if args.parse_log:
        log_path = args.log
        if (log_path is None) and (args.dir is not None):
            logs = [
                os.path.join(args.dir, fn)
                for fn in os.listdir(args.dir)
                if fn.lower().endswith('.log')
            ]
            if len(logs) > 0:
                logs = sorted(logs, key=lambda p: os.path.getmtime(p))
                log_path = logs[-1]
        if log_path is None:
            raise ValueError('No log file provided/found. Use --log or provide --dir containing *.log')
        if (args.dir is not None) and (not os.path.isabs(log_path)):
            log_path = os.path.join(args.dir, log_path)
        with open(log_path, 'r') as f:
            txt = f.read()
        info = parse_qc_log_text(txt)
        print("\n==================== LOG PARSE ====================")
        print(f"log: {log_path}")
        for k in sorted(info.keys()):
            print(f"{k:20s}: {info[k]}")

    if args.audit_potential:
        if args.density is None or args.potential is None:
            raise ValueError("For --audit-potential you must provide --density and --potential")
        dens_path = args.density
        pot_path = args.potential
        if args.dir is not None:
            if not os.path.isabs(dens_path):
                dens_path = os.path.join(args.dir, dens_path)
            if not os.path.isabs(pot_path):
                pot_path = os.path.join(args.dir, pot_path)
        audit_potential_cube(
            density_cube_path=dens_path,
            potential_cube_path=pot_path,
            expected_charge=args.expected_charge,
            axis=args.audit_axis,
            rmin_ang=args.audit_rmin_ang,
            nsamples=args.audit_nsamples,
            fit_samples=args.audit_fit_samples,
            fit_min_atom_dist_ang=args.audit_fit_min_atom_dist_ang,
        )

    if args.audit_poisson:
        if args.density is None or args.potential is None:
            raise ValueError('For --audit-poisson you must provide --density and --potential')
        dens_path = args.density
        pot_path = args.potential
        if args.dir is not None:
            if not os.path.isabs(dens_path):
                dens_path = os.path.join(args.dir, dens_path)
            if not os.path.isabs(pot_path):
                pot_path = os.path.join(args.dir, pot_path)
        audit_poisson_equation(
            density_cube_path=dens_path,
            potential_cube_path=pot_path,
            exclude_nuc_radius_ang=args.poisson_exclude_nuc_radius_ang,
            edge_pad=args.poisson_edge_pad,
        )

    if (args.dir is not None) and (not args.no_dir_scan):
        paths = sorted(str(pp) for pp in Path(args.dir).glob(args.pattern))
        for path in paths:
            report_cube(path, expected_charge=args.expected_charge)

    for path in args.cube:
        report_cube(path, expected_charge=args.expected_charge)

    if (args.gs is not None) or (args.es is not None) or (args.diff is not None):
        if args.gs is None or args.es is None:
            raise ValueError("For ES/GS/DIFF comparison you must provide at least --gs and --es")
        compare_es_gs_diff(
            gs_path=args.gs,
            es_path=args.es,
            diff_path=args.diff,
            mask_gs_below=args.mask_gs_below,
        )


if __name__ == "__main__":
    from pathlib import Path

    main()



'''
How to use it
1) Audit all .cube files in a directory

python3 /home/indranil/git/ppafm/tests/PhotonMap/test_indranil/interaction_framework/cube_audit.py \
  --dir /home/indranil/Documents/Secondment/test/PTCDA/anion_cal/opt/charge0/wb97x_d3bj

Optionally include expected molecular charge to print the electron-count error (∫n dV - (Z-charge)):

python3 .../cube_audit.py   --dir /path/to/wb97x_d3bj   --expected-charge 0

For an anion:

python3 .../cube_audit.py   --dir /path/to/charge-1/wb97x_d3bj   --expected-charge -1

2) Audit a few specific cube files

python3 .../cube_audit.py   --cube ground_state_density.cube   --cube density_difference_state1.cube

3) ES/GS/DIFF consistency check (with resampling)
This reproduces the “literal difference” check correctly (GS interpolated onto ES grid):

python3 .../cube_audit.py --gs /path/to/ground_state_density.cube --es /path/to/excited_state_density_state1.cube --diff /path/to/density_difference_state1.cube


4) Same consistency check, but also report results in low-density regions
This is useful because the core region subtraction can be numerically unstable when grids differ.

Example:
python3 .../cube_audit.py --gs .../ground_state_density.cube --es .../excited_state_density_state1.cube --diff .../density_difference_state1.cube --mask-gs-below 0.05 0.1 0.2


python3 cube_audit.py --gs /home/indranil/Documents/Secondment/test/PTCDA/anion_cal/opt/charge-1/wb97x_d3bj/ground_state_density.cube --es /home/indranil/Documents/Secondment/test/PTCDA/anion_cal/opt/charge-1/wb97x_d3bj/excited_state_density_state1.cube --diff /home/indranil/Documents/Secondment/test/PTCDA/anion_cal/opt/charge-1/wb97x_d3bj/density_difference_state1.cube --mask-gs-below 0.05 0.1 0.2


5) Potential-vs-density audits (generic; works for any molecule/code if cubes are consistent)

This checks:
  - far-field monopole behaviour from the potential cube
  - multipole fit directly from the potential cube
  - Poisson consistency: ∇²V ≈ ±4π n away from nuclei

python3 cube_audit.py \
  --dir /path/to/cubes \
  --no-dir-scan \
  --parse-log \
  --audit-potential --audit-poisson \
  --density ground_state_density.cube \
  --potential electrostatic_potential.cube \
  --expected-charge -1 \
  --audit-axis x --audit-rmin-ang 6.0 --audit-nsamples 30 \
  --audit-fit-samples 800 --audit-fit-min-atom-dist-ang 2.0 \
  --poisson-exclude-nuc-radius-ang 0.8


  python cube_audit.py \
  --dir /home/indranil/git/ppafm/tests/PhotonMap/test_indranil/interaction_framework/M4_S0.2_optimised_structure_wb97x_d3bj_6_31g__gpu_charge-1 \
  --no-dir-scan \
  --parse-log \
  --audit-potential --audit-poisson \
  --density ground_state_density.cube \
  --potential electrostatic_potential.cube \
  --expected-charge -1 \
  --audit-axis x --audit-rmin-ang 6.0 --audit-nsamples 30 \
  --audit-fit-samples 800 --audit-fit-min-atom-dist-ang 2.0 \
  --poisson-exclude-nuc-radius-ang 0.8 --poisson-edge-pad 1
'''