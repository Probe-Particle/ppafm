"""
Static Site-Energy Shift Framework
====================================

Computes the electrostatic site-energy shift Δω₁ of molecule 1's excitation
energy caused by the static charge distribution of molecule 2:

    Δω₁ = ∫ Δρ₁(r) · V₂^total(r) dr

where:
    Δρ₁(r) = -(n₁^ES - n₁^GS)  = charge redistribution upon excitation
    V₂^total(r) = V₂^nuc(r) + V₂^elec(r)
                 = Σ_A Z_A/|r-R_A| - ∫ n₂^GS(r')/|r-r'| dr'

All internal computations in atomic units (Bohr, Hartree).
Output energies converted to eV.

Units convention for cube files (Gaussian format from PySCF):
  - Coordinates in Bohr
  - Electron number densities in e/Bohr³  (positive = electrons present)
  - Charge density Δρ = -Δn  (electrons carry negative charge)
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

# ======================== Constants ========================
BOHR_TO_ANG = 0.5291772109217
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG
HARTREE_TO_EV = 27.211396132


# ======================== Cube File I/O ========================

def load_cube_with_atoms(fname):
    """Parse a Gaussian cube file, returning raw data in atomic units (Bohr).

    Returns
    -------
    data : dict with keys:
        'density'  : ndarray (nx, ny, nz) — volumetric data in native units (e/Bohr³)
        'origin'   : ndarray (3,) — grid origin in Bohr
        'axes'     : ndarray (3, 3) — step vectors [dx, dy, dz] rows, Bohr
        'npts'     : ndarray (3,) int — number of grid points (nx, ny, nz)
        'atom_Z'   : ndarray (natom,) — atomic numbers
        'atom_pos' : ndarray (natom, 3) — atom positions in Bohr
        'dV'       : float — voxel volume in Bohr³
    """
    f = open(fname, 'r')
    header1 = f.readline()
    header2 = f.readline()

    parts = f.readline().split()
    natoms_signed = int(parts[0])
    natoms = abs(natoms_signed)
    origin = np.array([float(parts[1]), float(parts[2]), float(parts[3])])

    axes = np.zeros((3, 3))
    npts = np.zeros(3, dtype=int)
    for i in range(3):
        parts = f.readline().split()
        npts[i] = int(parts[0])
        axes[i] = [float(parts[1]), float(parts[2]), float(parts[3])]

    atom_Z = np.zeros(natoms, dtype=int)
    atom_pos = np.zeros((natoms, 3))
    for i in range(natoms):
        parts = f.readline().split()
        atom_Z[i] = int(parts[0])
        atom_pos[i] = [float(parts[2]), float(parts[3]), float(parts[4])]

    if natoms_signed < 0:
        f.readline()  # skip extra line for MO cube files

    vals = []
    for line in f:
        vals.extend(line.split())
    f.close()

    density = np.array(vals, dtype=np.float64).reshape(npts[0], npts[1], npts[2])

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
        'dV': dV,
        'header1': header1.strip(),
        'header2': header2.strip(),
    }


# ======================== Grid Utilities ========================

def grid_positions_1d(origin_component, step, n):
    """1D array of grid NODE coordinates along one axis.

    Gaussian cube convention: position[k] = origin + k * step.
    Grid nodes, NOT cell-centered.
    """
    return origin_component + np.arange(n) * step


def make_grid_axes(cube):
    """Return 1D coordinate arrays (x, y, z) for an orthogonal cube grid.

    Assumes orthogonal axes (off-diagonal elements zero).
    """
    ox, oy, oz = cube['origin']
    dx, dy, dz = cube['axes'][0, 0], cube['axes'][1, 1], cube['axes'][2, 2]
    nx, ny, nz = cube['npts']
    xs = grid_positions_1d(ox, dx, nx)
    ys = grid_positions_1d(oy, dy, ny)
    zs = grid_positions_1d(oz, dz, nz)
    return xs, ys, zs


def make_grid_points(cube):
    """Return (N, 3) array of all grid point positions in Bohr."""
    xs, ys, zs = make_grid_axes(cube)
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing='ij')
    return np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=-1)


# ======================== FFT Poisson Solver ========================

def fft_poisson_potential(rho, origin, axes, pad_sizes=None):
    """Solve Poisson equation via FFT: V(r) = ∫ n(r')/|r-r'| dr'.

    Given electron number density n(r) on an orthogonal grid, returns the
    positive-definite potential V = ∫ n/|r-r'| dr' (Hartree).

    The density is placed at corner [0:nx, 0:ny, 0:nz] of the padded grid.
    After FFT convolution, the result is fftshift-ed so that the output
    covers both positive and negative displacements from the density.

    Parameters
    ----------
    rho : ndarray (nx, ny, nz)
        Electron number density n(r) in e/Bohr³
    origin : ndarray (3,) — grid origin in Bohr
    axes : ndarray (3, 3) — step vectors (diagonal assumed)
    pad_sizes : tuple (pnx, pny, pnz) or None
        Padded grid dimensions. If None, uses 3× original in each direction.

    Returns
    -------
    V_shifted : ndarray (pnx, pny, pnz) — potential (Hartree), fftshift-ed
    xs, ys, zs : 1D coordinate arrays for the shifted grid (Bohr)
    """
    nx, ny, nz = rho.shape
    dx = abs(axes[0, 0])
    dy = abs(axes[1, 1])
    dz = abs(axes[2, 2])
    dV = dx * dy * dz

    if pad_sizes is None:
        pad_sizes = (3 * nx, 3 * ny, 3 * nz)
    pnx, pny, pnz = pad_sizes

    rho_padded = np.zeros((pnx, pny, pnz))
    rho_padded[:nx, :ny, :nz] = rho

    ix = np.arange(pnx)
    iy = np.arange(pny)
    iz = np.arange(pnz)
    ix = np.where(ix <= pnx // 2, ix, ix - pnx) * dx
    iy = np.where(iy <= pny // 2, iy, iy - pny) * dy
    iz = np.where(iz <= pnz // 2, iz, iz - pnz) * dz
    X, Y, Z = np.meshgrid(ix, iy, iz, indexing='ij')
    R = np.sqrt(X * X + Y * Y + Z * Z)
    kernel = np.zeros_like(R)
    mask = R > 0
    kernel[mask] = 1.0 / R[mask]

    rho_k = np.fft.fftn(rho_padded)
    ker_k = np.fft.fftn(kernel)
    V = np.fft.ifftn(rho_k * ker_k).real * dV

    # fftshift: moves wraparound (negative-displacement) indices to front,
    # producing a contiguous spatial grid from most-negative to most-positive.
    V_shifted = np.fft.fftshift(V)

    # After fftshift, index j corresponds to:
    #   position = origin + (j - pn//2) * step
    # This covers negative displacements (j < pn//2) and positive (j >= pn//2).
    xs = origin[0] + (np.arange(pnx) - pnx // 2) * dx
    ys = origin[1] + (np.arange(pny) - pny // 2) * dy
    zs = origin[2] + (np.arange(pnz) - pnz // 2) * dz

    return V_shifted, xs, ys, zs


# ======================== Nuclear Charge on Grid ========================

def smear_nuclear_charges(atom_pos, atom_Z, origin, axes, npts):
    """Smear nuclear point charges onto a grid using trilinear interpolation.

    Each nuclear charge Z_A is distributed to the 8 nearest grid voxels
    with weights proportional to the overlap volume (trilinear).

    Parameters
    ----------
    atom_pos : ndarray (natom, 3) — nuclear positions in Bohr
    atom_Z : ndarray (natom,) — atomic numbers
    origin : ndarray (3,) — grid origin in Bohr
    axes : ndarray (3, 3) — step vectors (diagonal assumed)
    npts : ndarray (3,) int — grid dimensions

    Returns
    -------
    rho_nuc : ndarray (nx, ny, nz) — nuclear charge density in e/Bohr³
        (positive values = positive charge)
    """
    dx = axes[0, 0]
    dy = axes[1, 1]
    dz = axes[2, 2]
    dV = abs(dx * dy * dz)
    nx, ny, nz = int(npts[0]), int(npts[1]), int(npts[2])
    rho_nuc = np.zeros((nx, ny, nz))

    for iatom in range(len(atom_Z)):
        Z = float(atom_Z[iatom])
        pos = atom_pos[iatom]

        # Fractional grid coordinates
        fx = (pos[0] - origin[0]) / dx
        fy = (pos[1] - origin[1]) / dy
        fz = (pos[2] - origin[2]) / dz

        # Lower-left grid indices
        ix0 = int(np.floor(fx))
        iy0 = int(np.floor(fy))
        iz0 = int(np.floor(fz))

        # Trilinear weights
        wx1 = fx - ix0
        wy1 = fy - iy0
        wz1 = fz - iz0
        wx0 = 1.0 - wx1
        wy0 = 1.0 - wy1
        wz0 = 1.0 - wz1

        # Distribute charge to 8 neighbors (charge per volume = Z / dV * weight)
        for dix, wx in [(0, wx0), (1, wx1)]:
            for diy, wy in [(0, wy0), (1, wy1)]:
                for diz, wz in [(0, wz0), (1, wz1)]:
                    jx = ix0 + dix
                    jy = iy0 + diy
                    jz = iz0 + diz
                    if 0 <= jx < nx and 0 <= jy < ny and 0 <= jz < nz:
                        rho_nuc[jx, jy, jz] += Z * wx * wy * wz / dV

    return rho_nuc


def compute_total_potential_fft(cube_gs, pad_sizes=None):
    """Compute the total electrostatic potential V₂^total of a molecule via FFT.

    V₂^total(r) = V_nuc(r) - V_elec(r) = Σ Z_A/|r-R_A| - ∫ n_GS(r')/|r-r'| dr'

    Instead of computing V_nuc and V_elec separately (catastrophic cancellation),
    we compute the potential of the NET charge density:
        ρ_net(r) = Σ Z_A δ(r-R_A) - n_GS(r)

    For a neutral molecule, ∫ ρ_net dV ≈ 0, so G(k=0)=0 is physically correct.
    The FFT directly gives the multipole-level potential.

    Parameters
    ----------
    cube_gs : dict — loaded cube file data for GS density
    pad_sizes : tuple or None — padded grid dimensions

    Returns
    -------
    V_total : ndarray — total electrostatic potential (Hartree), fftshift-ed
    xs, ys, zs : 1D coordinate arrays (Bohr)
    """
    n_gs = cube_gs['density']  # electron number density (e/Bohr³)
    origin = cube_gs['origin']
    axes = cube_gs['axes']
    npts = cube_gs['npts']

    # Smear nuclear charges onto the same grid
    rho_nuc = smear_nuclear_charges(
        cube_gs['atom_pos'], cube_gs['atom_Z'], origin, axes, npts
    )

    # Net charge density: nuclear (positive) minus electronic (positive number density)
    # ρ_net = ρ_nuc - n_GS  (in units of charge/Bohr³)
    rho_net = rho_nuc - n_gs

    # Check neutrality
    dV = cube_gs['dV']
    q_net = np.sum(rho_net) * dV
    q_nuc = np.sum(rho_nuc) * dV
    q_elec = np.sum(n_gs) * dV
    print(f"    Nuclear charge on grid: {q_nuc:.4f} e")
    print(f"    Electronic charge: {q_elec:.4f} e")
    print(f"    Net charge (should be ~0): {q_net:.6e} e")

    # FFT Poisson of the net charge density
    # V_total = ∫ ρ_net(r')/|r-r'| dr'  (positive for positive charge)
    V_total, xs, ys, zs = fft_poisson_potential(rho_net, origin, axes, pad_sizes=pad_sizes)

    return V_total, xs, ys, zs


def multipole_potential(moments, R_vec):
    R = np.linalg.norm(R_vec)
    if R < 1e-12:
        return 0.0
    q = moments['monopole']
    mu = moments['dipole']
    Q = moments['quadrupole']
    R2 = R * R
    R3 = R2 * R
    R5 = R3 * R2
    V_q = q / R
    V_mu = np.dot(mu, R_vec) / R3
    V_Q = 0.5 * float(R_vec @ Q @ R_vec) / R5
    return V_q + V_mu + V_Q


def multipole_potential_points(moments, R_vecs):
    q = moments['monopole']
    mu = moments['dipole']
    Q = moments['quadrupole']
    R2 = np.sum(R_vecs * R_vecs, axis=1)
    R = np.sqrt(R2)
    R = np.maximum(R, 1e-12)
    invR = 1.0 / R
    invR3 = invR / R2
    invR5 = invR3 / R2
    V_q = q * invR
    V_mu = (R_vecs @ mu) * invR3
    QR = R_vecs @ Q
    V_Q = 0.5 * np.sum(QR * R_vecs, axis=1) * invR5
    return V_q + V_mu + V_Q


def multipole_field_points(moments, R_vecs):
    q = moments['monopole']
    mu = moments['dipole']
    Q = moments['quadrupole']
    R_vecs = np.asarray(R_vecs, dtype=float)

    R2 = np.sum(R_vecs * R_vecs, axis=1)
    R = np.sqrt(R2)
    R = np.maximum(R, 1e-12)
    invR = 1.0 / R
    invR2 = invR * invR
    invR3 = invR2 * invR
    invR5 = invR3 * invR2
    invR7 = invR5 * invR2

    E_q = q * R_vecs * invR3[:, None]

    mudotr = np.sum(R_vecs * mu[None, :], axis=1)
    E_mu = (-mu[None, :] * invR3[:, None]) + (3.0 * R_vecs * (mudotr * invR5)[:, None])

    Qr = R_vecs @ Q
    A = 0.5 * np.sum(Qr * R_vecs, axis=1)
    gradV_Q = (Qr * invR5[:, None]) - (5.0 * R_vecs * (A * invR7)[:, None])
    E_Q = -gradV_Q

    return E_q + E_mu + E_Q


def fft_field_at_points(V_interp, points, h=0.5):
    p = np.asarray(points, dtype=float)
    E = np.zeros_like(p)
    for i in range(3):
        dp = np.zeros(3)
        dp[i] = h
        Vp = V_interp(p + dp[None, :])
        Vm = V_interp(p - dp[None, :])
        E[:, i] = -(Vp - Vm) / (2.0 * h)
    return E


def resample_orthogonal_grid(density, origin, axes, npts_new):
    from scipy.interpolate import RegularGridInterpolator
    nx, ny, nz = density.shape
    dx, dy, dz = float(axes[0, 0]), float(axes[1, 1]), float(axes[2, 2])
    xs = origin[0] + np.arange(nx) * dx
    ys = origin[1] + np.arange(ny) * dy
    zs = origin[2] + np.arange(nz) * dz
    interp = RegularGridInterpolator((xs, ys, zs), density, method='linear', bounds_error=False, fill_value=0.0)
    nxs, nys, nzs = int(npts_new[0]), int(npts_new[1]), int(npts_new[2])
    xs2 = np.linspace(xs[0], xs[-1], nxs)
    ys2 = np.linspace(ys[0], ys[-1], nys)
    zs2 = np.linspace(zs[0], zs[-1], nzs)
    XX, YY, ZZ = np.meshgrid(xs2, ys2, zs2, indexing='ij')
    pts = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=-1)
    dens2 = interp(pts).reshape((nxs, nys, nzs))
    ax2 = np.zeros((3, 3), dtype=float)
    ax2[0, 0] = (xs2[-1] - xs2[0]) / (nxs - 1) if nxs > 1 else dx
    ax2[1, 1] = (ys2[-1] - ys2[0]) / (nys - 1) if nys > 1 else dy
    ax2[2, 2] = (zs2[-1] - zs2[0]) / (nzs - 1) if nzs > 1 else dz
    dV2 = abs(ax2[0, 0] * ax2[1, 1] * ax2[2, 2])
    return dens2, np.array([xs2[0], ys2[0], zs2[0]]), ax2, np.array([nxs, nys, nzs], dtype=int), dV2


def coulomb_energy_bruteforce(q1, pos1, q2, pos2, chunk=512, r_min=1e-6):
    e = 0.0
    n1 = q1.shape[0]
    for i0 in range(0, n1, chunk):
        i1 = min(i0 + chunk, n1)
        dp = pos1[i0:i1, None, :] - pos2[None, :, :]
        r = np.sqrt(np.sum(dp * dp, axis=2))
        r = np.maximum(r, float(r_min))
        e += np.sum((q1[i0:i1, None] * q2[None, :]) / r)
    return e


def nuclear_potential_at_points(atom_pos, atom_Z, eval_points):
    """Compute nuclear electrostatic potential at evaluation points.

    V_nuc(r) = Σ_A Z_A / |r - R_A|

    Parameters
    ----------
    atom_pos : ndarray (natom, 3) — nuclear positions in Bohr
    atom_Z : ndarray (natom,) int — atomic numbers
    eval_points : ndarray (N, 3) — evaluation points in Bohr

    Returns
    -------
    V : ndarray (N,) — potential in Hartree
    """
    V = np.zeros(eval_points.shape[0])
    for i in range(len(atom_Z)):
        dr = eval_points - atom_pos[i]
        r = np.sqrt(np.sum(dr**2, axis=1))
        r = np.maximum(r, 1e-10)
        V += float(atom_Z[i]) / r
    return V


# ======================== Multipole Analysis ========================

def extract_multipoles(density, origin, axes, npts):
    """Extract multipole moments of a density distribution.

    Parameters
    ----------
    density : ndarray (nx, ny, nz) — number density n(r) in e/Bohr³
    origin : ndarray (3,)
    axes : ndarray (3, 3) — step vectors
    npts : ndarray (3,) int

    Returns
    -------
    dict with:
        'monopole' : float (total charge = -∫n dV for electrons, but we return ∫n dV)
        'dipole'   : ndarray (3,) in e·Bohr
        'quadrupole' : ndarray (3, 3) traceless quadrupole tensor in e·Bohr²
        'center'   : ndarray (3,) center of density
    """
    dx = axes[0, 0]
    dy = axes[1, 1]
    dz = axes[2, 2]
    dV = abs(dx * dy * dz)

    xs = origin[0] + np.arange(npts[0]) * dx
    ys = origin[1] + np.arange(npts[1]) * dy
    zs = origin[2] + np.arange(npts[2]) * dz

    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing='ij')

    # Monopole
    q = np.sum(density) * dV

    # Dipole
    mu_x = np.sum(density * XX) * dV
    mu_y = np.sum(density * YY) * dV
    mu_z = np.sum(density * ZZ) * dV
    mu = np.array([mu_x, mu_y, mu_z])

    # Center of density
    if abs(q) > 1e-15:
        center = mu / q
    else:
        center = np.array([0.0, 0.0, 0.0])

    # Quadrupole tensor (traceless): Q_ij = ∫ (3 r_i r_j - r² δ_ij) n(r) dV
    R2 = XX**2 + YY**2 + ZZ**2
    coords = [XX, YY, ZZ]
    Q = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            integrand = (3.0 * coords[i] * coords[j]) * density
            if i == j:
                integrand -= R2 * density
            Q[i, j] = np.sum(integrand) * dV

    return {
        'monopole': q,
        'dipole': mu,
        'quadrupole': Q,
        'center': center,
    }


def extract_multipoles_about(density, origin, axes, npts, about):
    dx = axes[0, 0]
    dy = axes[1, 1]
    dz = axes[2, 2]
    dV = abs(dx * dy * dz)

    xs = origin[0] + np.arange(npts[0]) * dx - about[0]
    ys = origin[1] + np.arange(npts[1]) * dy - about[1]
    zs = origin[2] + np.arange(npts[2]) * dz - about[2]

    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing='ij')

    q = np.sum(density) * dV
    mu_x = np.sum(density * XX) * dV
    mu_y = np.sum(density * YY) * dV
    mu_z = np.sum(density * ZZ) * dV
    mu = np.array([mu_x, mu_y, mu_z])

    if abs(q) > 1e-15:
        center = mu / q
    else:
        center = np.array([0.0, 0.0, 0.0])

    R2 = XX**2 + YY**2 + ZZ**2
    coords = [XX, YY, ZZ]
    Q = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            integrand = (3.0 * coords[i] * coords[j]) * density
            if i == j:
                integrand -= R2 * density
            Q[i, j] = np.sum(integrand) * dV

    return {
        'monopole': q,
        'dipole': mu,
        'quadrupole': Q,
        'center': center,
    }


def nuclear_multipoles_about(atom_pos, atom_Z, about):
    rel = atom_pos - about[None, :]
    Z = atom_Z.astype(float)
    q = float(np.sum(Z))
    mu = np.sum(Z[:, None] * rel, axis=0)
    Q = np.zeros((3, 3))
    for i in range(rel.shape[0]):
        r = rel[i]
        r2 = float(np.dot(r, r))
        for a in range(3):
            for b in range(3):
                val = 3.0 * r[a] * r[b]
                if a == b:
                    val -= r2
                Q[a, b] += Z[i] * val
    return {
        'monopole': q,
        'dipole': mu,
        'quadrupole': Q,
        'center': np.array([0.0, 0.0, 0.0]),
    }


def multipole_shift_estimate(moments_delta_n, moments_gs, R_vec):
    """Estimate Δω from point-multipole expansion.

    For the site-energy shift, we need:
        Δω = -∫ Δn₁(r) V₂^total(r) dr

    At large R, V₂^total ≈ multipole expansion of the total charge of Mol2
    (nuclei minus electrons). For a neutral molecule, the leading surviving
    term depends on the permanent multipole moments.

    For PTCDA (D₂ₕ): dipole = 0, so leading term is quadrupole-quadrupole ~ 1/R⁵.

    This function computes the monopole-monopole and dipole-dipole contributions.

    Parameters
    ----------
    moments_delta_n : dict — multipole moments of Δn₁
    moments_gs : dict — multipole moments of total charge of Mol2 (Z_nuc - n_GS)
    R_vec : ndarray (3,) — displacement vector from Mol1 center to Mol2 center (Bohr)

    Returns
    -------
    dict with 'monopole_monopole', 'dipole_dipole', 'total' in Hartree
    """
    R = np.linalg.norm(R_vec)
    R_hat = R_vec / R

    q1 = moments_delta_n['monopole']   # Should be ~0 (electron conservation)
    q2 = moments_gs['monopole']         # Should be ~0 (neutral molecule)
    mu1 = moments_delta_n['dipole']
    mu2 = moments_gs['dipole']

    # Monopole-monopole: -q1*q2/R  (the minus sign is from Δρ = -Δn)
    E_qq = -q1 * q2 / R if R > 1e-10 else 0.0

    # Dipole-dipole: [μ1·μ2 - 3(μ1·R̂)(μ2·R̂)] / R³
    # But with sign: Δρ₁ = -Δn₁, so we negate
    E_dd = 0.0
    if R > 1e-10:
        E_dd = -(np.dot(mu1, mu2) - 3.0 * np.dot(mu1, R_hat) * np.dot(mu2, R_hat)) / R**3

    return {
        'monopole_monopole': E_qq,
        'dipole_dipole': E_dd,
        'total': E_qq + E_dd,
    }


# ======================== Main Shift Computation ========================

class SiteEnergyShiftCalculator:
    """Compute static site-energy shift Δω₁ as a function of intermolecular distance.

    Usage:
        calc = SiteEnergyShiftCalculator(
            diff_density_cube_path,
            gs_density_cube_path,
        )
        calc.precompute(max_distance_ang=40.0, scan_axis='y')
        shifts = calc.distance_scan(distances_ang, axis='y')
    """

    def __init__(self, diff_density_path, gs_density_path, mol2_charge=0.0, enforce_dn_zero_integral=True):
        """
        Parameters
        ----------
        diff_density_path : str
            Path to density_difference cube file (Δn = n_ES - n_GS)
        gs_density_path : str
            Path to ground-state electron density cube file (n_GS)
        """
        self.diff_density_path = diff_density_path
        self.gs_density_path = gs_density_path
        self.mol2_charge = float(mol2_charge)
        self.enforce_dn_zero_integral = bool(enforce_dn_zero_integral)

        print(f"Loading difference density: {diff_density_path}")
        self.cube_dn = load_cube_with_atoms(diff_density_path)
        print(f"  Grid: {self.cube_dn['npts']}, dV = {self.cube_dn['dV']:.6e} Bohr³")

        print(f"Loading GS density: {gs_density_path}")
        self.cube_gs = load_cube_with_atoms(gs_density_path)
        print(f"  Grid: {self.cube_gs['npts']}, dV = {self.cube_gs['dV']:.6e} Bohr³")

        self.V_total_interp = None
        self.mol1_center = np.mean(self.cube_dn['atom_pos'], axis=0)
        self.mol2_center = np.mean(self.cube_gs['atom_pos'], axis=0)

        self.cube_dn['origin'] = self.cube_dn['origin'] - self.mol1_center
        self.cube_dn['atom_pos'] = self.cube_dn['atom_pos'] - self.mol1_center[None, :]
        self.cube_gs['origin'] = self.cube_gs['origin'] - self.mol2_center
        self.cube_gs['atom_pos'] = self.cube_gs['atom_pos'] - self.mol2_center[None, :]

        self.mol1_center = np.array([0.0, 0.0, 0.0])
        self.mol2_center = np.array([0.0, 0.0, 0.0])
        self.mol_center = self.mol2_center
        self.moments_net = None

        self.points_mol1 = None
        self.delta_n_flat = None
        self.dV_dn = None

    def _sanity_checks_and_normalize(self):
        """Run sanity checks and enforce charge neutrality.

        Finite-grid cube files typically miss electron density in the tails,
        creating an artificial net charge. We rescale densities to fix this:
          - GS density: rescale so ∫n_GS dV = Z_total (neutral molecule)
          - Diff density: subtract uniform offset so ∫Δn dV = 0 (electron conservation)
        """
        dn = self.cube_dn
        gs = self.cube_gs

        integral_dn_raw = np.sum(dn['density']) * dn['dV']
        integral_gs_raw = np.sum(gs['density']) * gs['dV']
        Z_total = float(np.sum(gs['atom_Z']))
        N_e_target = Z_total - self.mol2_charge

        print(f"\nSanity checks (before normalization):")
        print(f"  ∫Δn₁ dV = {integral_dn_raw:.6e} e  (should be ≈ 0)")
        print(f"  ∫n_GS dV = {integral_gs_raw:.4f} e  (target N_e = {N_e_target:.0f}, Z_total = {Z_total:.0f})")
        print(f"  Net charge (raw) = {Z_total - integral_gs_raw:.4f} e  (target charge = {self.mol2_charge:.4f} e)")

        dn_min = float(np.min(dn['density']))
        dn_max = float(np.max(dn['density']))
        frac_pos = float(np.mean(dn['density'] > 0.0))
        frac_neg = float(np.mean(dn['density'] < 0.0))
        print(f"  Δn range: [{dn_min:+.6e}, {dn_max:+.6e}] e/Bohr³   frac(+): {frac_pos:.3f}  frac(-): {frac_neg:.3f}")
        if self.enforce_dn_zero_integral:
            if abs(integral_dn_raw) > 1.0:
                raise ValueError(f"Δn cube integral is too large for a density difference: ∫Δn dV = {integral_dn_raw:.6e} e")
            if (frac_pos < 1e-6) or (frac_neg < 1e-6):
                raise ValueError("Δn cube appears single-signed; this is not a valid density difference cube")

        # Rescale GS density: n_GS *= N_e_target / integral
        if abs(integral_gs_raw) > 1e-10:
            scale_gs = N_e_target / integral_gs_raw
            gs['density'] = gs['density'] * scale_gs
            integral_gs_new = np.sum(gs['density']) * gs['dV']
            print(f"\n  Rescaled GS density by {scale_gs:.6f}")
            print(f"  ∫n_GS dV (after) = {integral_gs_new:.6f} e")

        if self.enforce_dn_zero_integral:
            n_voxels = np.prod(dn['npts'])
            offset = integral_dn_raw / (n_voxels * dn['dV'])
            dn['density'] = dn['density'] - offset
            integral_dn_new = np.sum(dn['density']) * dn['dV']
            print(f"  Corrected Δn offset by {offset:.6e} e/Bohr³")
            print(f"  ∫Δn₁ dV (after) = {integral_dn_new:.6e} e")

        return integral_dn_raw, integral_gs_raw, Z_total

    def _compute_pad_sizes(self, max_disp_bohr_vec, rot_mol2=None):
        """Compute padded grid dimensions to cover the required spatial range.

        The V_elec grid must cover all positions where Mol1's grid points
        can fall when transformed into Mol2's reference frame.

        For a displacement d along scan_axis, Mol1's points in Mol2's frame
        are at (r_mol1 - d). The range of these positions along the scan axis:
          min: mol1_grid_min - d_max
          max: mol1_grid_max - d_min  (d_min > 0, so max ≈ mol1_grid_max)

        The padded V_elec grid must extend from (origin - pn//2 * step) to
        (origin + (pn - pn//2 - 1) * step).
        """
        gs = self.cube_gs
        dn = self.cube_dn
        dx = abs(gs['axes'][0, 0])
        dy = abs(gs['axes'][1, 1])
        dz = abs(gs['axes'][2, 2])
        nx, ny, nz = gs['npts']
        steps = [dx, dy, dz]
        ns_gs = [nx, ny, nz]

        # Mol1 grid extents in absolute coordinates
        dn_origin = dn['origin']
        dn_steps = [abs(dn['axes'][i, i]) for i in range(3)]
        dn_npts = dn['npts']
        mol1_min = np.array([dn_origin[i] for i in range(3)])
        mol1_max = np.array([dn_origin[i] + (dn_npts[i] - 1) * dn_steps[i] for i in range(3)])

        corners = np.array([
            [mol1_min[0], mol1_min[1], mol1_min[2]],
            [mol1_min[0], mol1_min[1], mol1_max[2]],
            [mol1_min[0], mol1_max[1], mol1_min[2]],
            [mol1_min[0], mol1_max[1], mol1_max[2]],
            [mol1_max[0], mol1_min[1], mol1_min[2]],
            [mol1_max[0], mol1_min[1], mol1_max[2]],
            [mol1_max[0], mol1_max[1], mol1_min[2]],
            [mol1_max[0], mol1_max[1], mol1_max[2]],
        ], dtype=float)
        if rot_mol2 is not None:
            corners = corners @ np.asarray(rot_mol2, dtype=float)
        mol1_min = np.min(corners, axis=0)
        mol1_max = np.max(corners, axis=0)

        # V_elec is in Mol2's native frame (centered at its cube origin).
        # Points queried: r_mol1 - d, where d is along scan_axis.
        # Along scan axis: query range = [mol1_min[ax] - d_max, mol1_max[ax]]
        # Along other axes: query range = [mol1_min[i], mol1_max[i]]

        # The V_elec grid after fftshift covers:
        #   [origin[i] - pn[i]//2 * step[i],  origin[i] + (pn[i] - pn[i]//2 - 1) * step[i]]
        # We need: origin[i] - pn[i]//2 * step[i]  <=  query_min[i]
        #          origin[i] + (pn[i] - pn[i]//2 - 1) * step[i]  >=  query_max[i]

        max_disp_bohr_vec = np.asarray(max_disp_bohr_vec, dtype=float)
        pad_sizes = []
        for i in range(3):
            q_min = mol1_min[i]
            q_max = mol1_max[i]
            q_min -= max_disp_bohr_vec[i]
            q_max += max_disp_bohr_vec[i]

            margin = 4.0 * steps[i]
            if max_disp_bohr_vec[i] > 1e-12:
                margin += (ns_gs[i] - 1) * steps[i]
            q_min -= margin
            q_max += margin

            # Required range from origin:
            range_neg = gs['origin'][i] - q_min  # how far below origin we need
            range_pos = q_max - gs['origin'][i]  # how far above origin we need

            # Convert to grid points
            pts_neg = int(np.ceil(range_neg / steps[i])) + 2  # +2 safety margin
            pts_pos = int(np.ceil(range_pos / steps[i])) + 2

            # pn//2 >= pts_neg and pn - pn//2 >= pts_pos + ns_gs[i]
            # For even pn: pn//2 = pn/2, so pn >= 2*max(pts_neg, pts_pos + ns_gs[i])
            pn = max(2 * pts_neg, 2 * (pts_pos + ns_gs[i]), 2 * ns_gs[i])
            # Round up to even number for FFT efficiency
            pn = pn + (pn % 2)
            pad_sizes.append(pn)

        return tuple(pad_sizes)

    def precompute(self, max_distance_ang=40.0, scan_axis='y', rot_mol2=None, max_disp_vec_ang=None):
        """Pre-compute the total electrostatic potential of Mol2 via FFT Poisson.

        Uses the NET charge density (ρ_nuc - n_GS) in a single FFT solve,
        avoiding catastrophic cancellation between V_nuc (~Z/R) and V_elec (~Z/R).

        Parameters
        ----------
        max_distance_ang : float
            Maximum scan distance in Å. Determines padding size.
        scan_axis : str
            Axis along which molecules will be displaced ('x', 'y', or 'z').
        """
        self._sanity_checks_and_normalize()

        gs = self.cube_gs
        scan_axis_idx = {'x': 0, 'y': 1, 'z': 2}[scan_axis.lower()]
        if max_disp_vec_ang is None:
            max_disp_bohr_vec = np.zeros(3, dtype=float)
            max_disp_bohr_vec[scan_axis_idx] = float(max_distance_ang) * ANG_TO_BOHR
        else:
            max_disp_vec_ang = np.asarray(max_disp_vec_ang, dtype=float)
            max_disp_bohr_vec = np.abs(max_disp_vec_ang) * ANG_TO_BOHR

        self.mol1_center = np.array([0.0, 0.0, 0.0])
        self.mol2_center = np.array([0.0, 0.0, 0.0])
        self.mol_center = self.mol2_center
        print(f"\n  Molecular center (Mol1): ({self.mol1_center[0]:.3f}, "
              f"{self.mol1_center[1]:.3f}, {self.mol1_center[2]:.3f}) Bohr")
        print(f"  Molecular center (Mol2): ({self.mol2_center[0]:.3f}, "
              f"{self.mol2_center[1]:.3f}, {self.mol2_center[2]:.3f}) Bohr")

        pad_sizes = self._compute_pad_sizes(max_disp_bohr_vec, rot_mol2=rot_mol2)
        print(f"\n  Computing total electrostatic potential (net charge FFT)...")
        print(f"  Padded grid: {pad_sizes} (for max R = {max_distance_ang:.1f} Å along {scan_axis})")
        mem_mb = np.prod(pad_sizes) * 8 / 1e6
        print(f"  Memory: {mem_mb:.1f} MB")

        V_total, xs, ys, zs = compute_total_potential_fft(gs, pad_sizes=pad_sizes)

        self.V_total_interp = RegularGridInterpolator(
            (xs, ys, zs), V_total,
            method='linear',
            bounds_error=False,
            fill_value=0.0,
        )
        print(f"  V_total grid bounds (Bohr):")
        print(f"    X: [{xs[0]:.2f}, {xs[-1]:.2f}]")
        print(f"    Y: [{ys[0]:.2f}, {ys[-1]:.2f}]")
        print(f"    Z: [{zs[0]:.2f}, {zs[-1]:.2f}]")

        about = self.mol_center
        mom_nuc = nuclear_multipoles_about(gs['atom_pos'], gs['atom_Z'], about)
        mom_e = extract_multipoles_about(gs['density'], gs['origin'], gs['axes'], gs['npts'], about)
        self.moments_net = {
            'monopole': mom_nuc['monopole'] - mom_e['monopole'],
            'dipole': mom_nuc['dipole'] - mom_e['dipole'],
            'quadrupole': mom_nuc['quadrupole'] - mom_e['quadrupole'],
            'center': np.array([0.0, 0.0, 0.0]),
        }

        print(f"\n  Net-charge multipoles about molecular center:")
        print(f"    q_net: {self.moments_net['monopole']:.6e} e")
        print(f"    mu_net: ({self.moments_net['dipole'][0]:.4e}, {self.moments_net['dipole'][1]:.4e}, {self.moments_net['dipole'][2]:.4e}) e·Bohr")

        
        test_Rs = [30.0, 50.0, 60.0]
        vals = []
        for Rtest in test_Rs:
            Rvec = np.zeros(3)
            Rvec[scan_axis_idx] = -Rtest
            p = about + Rvec
            V_fft = float(self.V_total_interp(p[None, :]))
            V_mp = multipole_potential(self.moments_net, Rvec)
            vals.append((Rtest, V_fft, V_mp))
        if len(vals) >= 2:
            C = vals[-1][1] - vals[-1][2]
            for (Rtest, V_fft, V_mp) in vals:
                denom = max(abs(V_fft - C), 1e-16)
                rel = ((V_fft - C) - V_mp) / denom
                print(f"    V_total check @ R={Rtest:6.1f} Bohr along {scan_axis}:  V_fft={V_fft:+.6e}  V_mp={V_mp:+.6e}  (V_fft-C)={V_fft-C:+.6e}  rel.err={rel:+.3e}")

        for Rtest in test_Rs:
            Rvec = np.zeros(3)
            Rvec[scan_axis_idx] = -Rtest
            p = about + Rvec
            E_fft = fft_field_at_points(self.V_total_interp, p[None, :], h=0.5)[0]
            E_mp = multipole_field_points(self.moments_net, Rvec[None, :])[0]
            denom = max(float(np.linalg.norm(E_fft)), 1e-16)
            rel = float(np.linalg.norm(E_fft - E_mp)) / denom
            print(f"    E_total check @ R={Rtest:6.1f} Bohr along {scan_axis}:  |E_fft|={np.linalg.norm(E_fft):.6e}  |E_mp|={np.linalg.norm(E_mp):.6e}  rel.err={rel:+.3e}")

        # Extract multipole moments for validation
        print(f"\n  Extracting multipoles of Δn₁...")
        self.moments_dn = extract_multipoles(
            self.cube_dn['density'], self.cube_dn['origin'],
            self.cube_dn['axes'], self.cube_dn['npts']
        )
        print(f"    Δn monopole: {self.moments_dn['monopole']:.6e}")
        print(f"    Δn dipole: ({self.moments_dn['dipole'][0]:.4e}, "
              f"{self.moments_dn['dipole'][1]:.4e}, {self.moments_dn['dipole'][2]:.4e})")

        # Total molecule multipoles (nuclear - electronic) for multipole estimate
        moments_gs_elec = extract_multipoles(
            self.cube_gs['density'], self.cube_gs['origin'],
            self.cube_gs['axes'], self.cube_gs['npts']
        )
        Z_nuc = float(np.sum(self.cube_gs['atom_Z']))
        mu_nuc = np.sum(self.cube_gs['atom_Z'][:, None].astype(float) * self.cube_gs['atom_pos'], axis=0)

        self.moments_total_mol = {
            'monopole': Z_nuc - moments_gs_elec['monopole'],
            'dipole': mu_nuc - moments_gs_elec['dipole'],
        }
        print(f"\n  Total molecule multipoles:")
        print(f"    Monopole (Z-N_e): {self.moments_total_mol['monopole']:.6e}")
        print(f"    Dipole: ({self.moments_total_mol['dipole'][0]:.4e}, "
              f"{self.moments_total_mol['dipole'][1]:.4e}, {self.moments_total_mol['dipole'][2]:.4e})")

        q2 = float(self.moments_net['monopole'])
        mu1 = np.asarray(self.moments_dn['dipole'], dtype=float)
        mu2 = np.asarray(self.moments_net['dipole'], dtype=float)
        Q2 = np.asarray(self.moments_net['quadrupole'], dtype=float)
        q2_ok = abs(q2) > 1e-6
        mu1_ok = float(np.linalg.norm(mu1)) > 1e-6
        mu2_ok = float(np.linalg.norm(mu2)) > 1e-6
        Q2_ok = float(np.linalg.norm(Q2)) > 1e-6
        if q2_ok and mu1_ok:
            print("\n  Leading-order expectation: monopole(Mol2) × dipole(Δn1) ⇒ |Δω| ~ R^(-2)")
        elif (not q2_ok) and mu1_ok and mu2_ok:
            print("\n  Leading-order expectation: dipole(Δn1) × dipole(Mol2) ⇒ |Δω| ~ R^(-3)")
        elif (not q2_ok) and mu1_ok and Q2_ok:
            print("\n  Leading-order expectation: dipole(Δn1) × quadrupole(Mol2) ⇒ |Δω| ~ R^(-4)")
        else:
            print("\n  Leading-order expectation: higher multipoles dominate (often ~R^(-5) or faster)")

        print("\n  Precomputation complete.")

        dn = self.cube_dn
        xs_mol1, ys_mol1, zs_mol1 = make_grid_axes(dn)
        XX, YY, ZZ = np.meshgrid(xs_mol1, ys_mol1, zs_mol1, indexing='ij')
        self.points_mol1 = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=-1)
        self.delta_n_flat = dn['density'].ravel()
        self.dV_dn = dn['dV']

    def displacement_from_scan_value(self, value_ang, axis='y', scan_kind='displacement'):
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis.lower()]
        value_bohr = float(value_ang) * ANG_TO_BOHR
        d = np.zeros(3, dtype=float)
        if scan_kind == 'displacement':
            d[axis_idx] = value_bohr
            return d
        if scan_kind == 'center':
            base = self.mol2_center - self.mol1_center
            d[axis_idx] = value_bohr - base[axis_idx]
            return d
        if scan_kind == 'proj_gap':
            proj1_max = float(np.max(self.cube_dn['atom_pos'][:, axis_idx]))
            proj2_min = float(np.min(self.cube_gs['atom_pos'][:, axis_idx]))
            base_gap = proj2_min - proj1_max
            d[axis_idx] = value_bohr - base_gap
            return d
        if scan_kind == 'gap':
            d[axis_idx] = self._solve_disp_for_min_atom_dist(value_bohr, axis_idx)
            return d
        raise ValueError(f"Unknown scan_kind: {scan_kind}")

    def _min_atom_dist_for_axis_disp_bohr(self, disp_axis_bohr, axis_idx):
        d = np.zeros(3, dtype=float)
        d[axis_idx] = float(disp_axis_bohr)
        return self.min_atom_distance_bohr(d)

    def _solve_disp_for_min_atom_dist(self, target_dmin_bohr, axis_idx, max_iter=64):
        target_dmin_bohr = float(target_dmin_bohr)
        proj1_max = float(np.max(self.cube_dn['atom_pos'][:, axis_idx]))
        proj2_min = float(np.min(self.cube_gs['atom_pos'][:, axis_idx]))
        base_disp = -(proj2_min - proj1_max)

        t_lo = base_disp
        f_lo = self._min_atom_dist_for_axis_disp_bohr(t_lo, axis_idx) - target_dmin_bohr

        step = 2.0 * ANG_TO_BOHR
        t_hi = t_lo
        f_hi = f_lo
        for _ in range(200):
            if f_hi >= 0.0:
                break
            t_hi = t_hi + step
            f_hi = self._min_atom_dist_for_axis_disp_bohr(t_hi, axis_idx) - target_dmin_bohr
            step = step * 1.2

        if f_hi < 0.0:
            return float(t_hi)

        for _ in range(int(max_iter)):
            t_mid = 0.5 * (t_lo + t_hi)
            f_mid = self._min_atom_dist_for_axis_disp_bohr(t_mid, axis_idx) - target_dmin_bohr
            if f_mid >= 0.0:
                t_hi = t_mid
                f_hi = f_mid
            else:
                t_lo = t_mid
                f_lo = f_mid
        return float(t_hi)

    def axis_gap_bohr(self, displacement_bohr, axis='y'):
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis.lower()]
        d = np.asarray(displacement_bohr, dtype=float)
        pos1 = self.cube_dn['atom_pos']
        pos2 = self.cube_gs['atom_pos'] + d[None, :]
        p1_min = float(np.min(pos1[:, axis_idx]))
        p1_max = float(np.max(pos1[:, axis_idx]))
        p2_min = float(np.min(pos2[:, axis_idx]))
        p2_max = float(np.max(pos2[:, axis_idx]))
        if d[axis_idx] >= 0:
            return p2_min - p1_max
        return p1_min - p2_max

    def min_atom_distance_bohr(self, displacement_bohr):
        d = np.asarray(displacement_bohr, dtype=float)
        pos1 = self.cube_dn['atom_pos']
        pos2 = self.cube_gs['atom_pos'] + d[None, :]
        diff = pos1[:, None, :] - pos2[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        return float(np.sqrt(np.min(dist2)))

    def compute_shift(self, displacement_bohr, rot_mol2=None):
        """Compute Δω₁ for Mol2 displaced by the given vector (Bohr).

        Mol1 stays at its native position (as in the cube file).
        Mol2's density and atoms are shifted by displacement_bohr.

        The total potential V₂^total was pre-computed from the net charge density
        (nuclear - electronic) in a single FFT solve (no cancellation issues).

        Returns
        -------
        dict with 'shift_eV', 'shift_hartree'
        """
        d = np.asarray(displacement_bohr, dtype=float)

        dn = self.cube_dn
        points_mol1 = self.points_mol1
        if points_mol1 is None:
            xs_mol1, ys_mol1, zs_mol1 = make_grid_axes(dn)
            XX, YY, ZZ = np.meshgrid(xs_mol1, ys_mol1, zs_mol1, indexing='ij')
            points_mol1 = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=-1)
        dV_dn = dn['dV'] if self.dV_dn is None else self.dV_dn

        # V_total was computed for Mol2 at its native position.
        # To evaluate when Mol2 is shifted by d: query at (r_mol1 - d) in Mol2's frame.
        points_in_mol2_frame = points_mol1 - d
        if rot_mol2 is not None:
            points_in_mol2_frame = points_in_mol2_frame @ np.asarray(rot_mol2, dtype=float)
        V_total = self.V_total_interp(points_in_mol2_frame)

        # Δω₁ = ∫ Δρ₁ · V₂ = -∫ Δn₁ · V₂  (Δρ = -Δn for electrons)
        delta_n_flat = dn['density'].ravel() if self.delta_n_flat is None else self.delta_n_flat
        shift_hartree = -np.sum(delta_n_flat * V_total) * dV_dn

        return {
            'shift_eV': shift_hartree * HARTREE_TO_EV,
            'shift_hartree': shift_hartree,
        }

    def compute_shift_multipole(self, displacement_bohr, rot_mol2=None):
        d = np.asarray(displacement_bohr, dtype=float)
        dn = self.cube_dn
        points_mol1 = self.points_mol1
        if points_mol1 is None:
            xs_mol1, ys_mol1, zs_mol1 = make_grid_axes(dn)
            XX, YY, ZZ = np.meshgrid(xs_mol1, ys_mol1, zs_mol1, indexing='ij')
            points_mol1 = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=-1)
        dV_dn = dn['dV'] if self.dV_dn is None else self.dV_dn
        points_in_mol2_frame = points_mol1 - d
        if rot_mol2 is not None:
            points_in_mol2_frame = points_in_mol2_frame @ np.asarray(rot_mol2, dtype=float)
        R_vecs = points_in_mol2_frame - self.mol_center[None, :]
        V_mp = multipole_potential_points(self.moments_net, R_vecs)
        delta_n_flat = dn['density'].ravel() if self.delta_n_flat is None else self.delta_n_flat
        shift_hartree = -np.sum(delta_n_flat * V_mp) * dV_dn
        return shift_hartree

    def compute_shift_bruteforce(self, displacement_bohr, rot_mol2=None, npts_dn=(18, 14, 10), npts_gs=(18, 18, 18), chunk=384, rho_cut=0.0, r_min=None):
        d = np.asarray(displacement_bohr, dtype=float)

        dn = self.cube_dn
        gs = self.cube_gs

        rho1, o1, a1, n1, dV1 = resample_orthogonal_grid(-dn['density'], dn['origin'], dn['axes'], npts_dn)
        mask1 = np.abs(rho1) > float(rho_cut)
        nsel1 = int(np.sum(mask1))
        if self.enforce_dn_zero_integral and (nsel1 > 0):
            q1_sel = float(np.sum(rho1[mask1]) * dV1)
            rho1[mask1] = rho1[mask1] - (q1_sel / (nsel1 * dV1))

        rho_nuc = smear_nuclear_charges(gs['atom_pos'], gs['atom_Z'], gs['origin'], gs['axes'], gs['npts'])
        rho_net = rho_nuc - gs['density']
        rho2, o2, a2, n2, dV2 = resample_orthogonal_grid(rho_net, gs['origin'], gs['axes'], npts_gs)
        mask2 = np.abs(rho2) > float(rho_cut)
        nsel2 = int(np.sum(mask2))
        if nsel2 > 0:
            q2_sel = float(np.sum(rho2[mask2]) * dV2)
            target_q2 = float(self.mol2_charge)
            rho2[mask2] = rho2[mask2] - ((q2_sel - target_q2) / (nsel2 * dV2))

        if nsel1 > 0:
            q1 = (rho1[mask1] * dV1).ravel()
        else:
            q1 = (rho1 * dV1).ravel()
        if nsel2 > 0:
            q2 = (rho2[mask2] * dV2).ravel()
        else:
            q2 = (rho2 * dV2).ravel()

        xs1 = o1[0] + np.arange(n1[0]) * a1[0, 0]
        ys1 = o1[1] + np.arange(n1[1]) * a1[1, 1]
        zs1 = o1[2] + np.arange(n1[2]) * a1[2, 2]
        X1, Y1, Z1 = np.meshgrid(xs1, ys1, zs1, indexing='ij')
        pos1 = np.stack([X1.ravel(), Y1.ravel(), Z1.ravel()], axis=-1)
        if nsel1 > 0:
            pos1 = pos1[mask1.ravel()]

        xs2 = o2[0] + np.arange(n2[0]) * a2[0, 0]
        ys2 = o2[1] + np.arange(n2[1]) * a2[1, 1]
        zs2 = o2[2] + np.arange(n2[2]) * a2[2, 2]
        X2, Y2, Z2 = np.meshgrid(xs2, ys2, zs2, indexing='ij')
        pos2 = np.stack([X2.ravel(), Y2.ravel(), Z2.ravel()], axis=-1)
        if nsel2 > 0:
            pos2 = pos2[mask2.ravel()]
        pos2 = pos2 + d[None, :]
        if rot_mol2 is not None:
            pos2 = pos2 @ np.asarray(rot_mol2, dtype=float)

        if r_min is None:
            h1 = min(abs(float(a1[0, 0])), abs(float(a1[1, 1])), abs(float(a1[2, 2])))
            h2 = min(abs(float(a2[0, 0])), abs(float(a2[1, 1])), abs(float(a2[2, 2])))
            r_min = 0.5 * min(h1, h2)

        e = coulomb_energy_bruteforce(q1, pos1, q2, pos2, chunk=chunk, r_min=float(r_min))
        return e

    def distance_scan(self, distances_ang, axis='y', scan_kind='displacement', rot_mol2=None):
        """Compute Δω₁ for a range of distances along the specified axis.

        Parameters
        ----------
        distances_ang : array-like — center-to-center distances in Ångström
        axis : str — 'x', 'y', or 'z'

        Returns
        -------
        results : dict with arrays keyed by 'R_ang', 'R_bohr', 'shift_eV',
            'shift_hartree', 'multipole_eV'
        """
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis.lower()]
        distances_ang = np.asarray(distances_ang, dtype=float)
        distances_bohr = distances_ang * ANG_TO_BOHR

        n = len(distances_ang)
        results = {
            'scan_ang': distances_ang,
            'scan_kind': scan_kind,
            'axis': axis,
            'R_ang': np.zeros(n),
            'R_bohr': np.zeros(n),
            'gap_ang': np.zeros(n),
            'gap_bohr': np.zeros(n),
            'proj_gap_ang': np.zeros(n),
            'proj_gap_bohr': np.zeros(n),
            'min_atom_dist_ang': np.zeros(n),
            'disp_axis_ang': np.zeros(n),
            'shift_eV': np.zeros(n),
            'shift_hartree': np.zeros(n),
            'multipole_eV': np.zeros(n),
        }

        scan_label = {
            'displacement': 'disp',
            'center': 'center',
            'gap': 'gap',
            'proj_gap': 'proj_gap',
        }.get(scan_kind, 'scan')

        for i, x_ang in enumerate(distances_ang):
            displacement = self.displacement_from_scan_value(x_ang, axis=axis, scan_kind=scan_kind)
            disp_axis_ang = displacement[axis_idx] * BOHR_TO_ANG

            center_vec = (self.mol2_center + displacement) - self.mol1_center
            R_bohr = float(np.linalg.norm(center_vec))
            R_ang = R_bohr * BOHR_TO_ANG

            dmin_ang = self.min_atom_distance_bohr(displacement) * BOHR_TO_ANG

            proj_gap_bohr = self.axis_gap_bohr(displacement, axis=axis)
            proj_gap_ang = proj_gap_bohr * BOHR_TO_ANG
            if scan_kind == 'gap':
                gap_ang = float(dmin_ang)
                gap_bohr = gap_ang * ANG_TO_BOHR
            else:
                gap_bohr = proj_gap_bohr
                gap_ang = proj_gap_ang

            results['R_ang'][i] = R_ang
            results['R_bohr'][i] = R_bohr
            results['gap_ang'][i] = gap_ang
            results['gap_bohr'][i] = gap_bohr
            results['proj_gap_ang'][i] = proj_gap_ang
            results['proj_gap_bohr'][i] = proj_gap_bohr
            results['min_atom_dist_ang'][i] = dmin_ang
            results['disp_axis_ang'][i] = disp_axis_ang

            print(f"  {scan_label} = {x_ang:8.2f} Å  R = {R_ang:8.2f} Å  gap = {gap_ang:8.2f} Å  d_min = {dmin_ang:7.3f} Å ...", end=' ', flush=True)

            res = self.compute_shift(displacement, rot_mol2=rot_mol2)
            results['shift_eV'][i] = res['shift_eV']
            results['shift_hartree'][i] = res['shift_hartree']

            mp_ha = self.compute_shift_multipole(displacement, rot_mol2=rot_mol2)
            results['multipole_eV'][i] = mp_ha * HARTREE_TO_EV

            print(f"Δω = {res['shift_eV']:+12.6e} eV  ({res['shift_hartree']:+.4e} Ha)")

        return results


# ======================== Plotting ========================

def plot_results(results, title="PTCDA Homodimer — Static Site-Energy Shift", save_prefix=None, show=True):
    """Generate validation plots from distance scan results."""
    if not show:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    x = results.get('scan_ang', results['R_ang'])
    R = results['R_ang']
    dw = results['shift_eV']
    dw_mp = results['multipole_eV']
    scan_kind = results.get('scan_kind', 'displacement')
    if scan_kind == 'gap':
        x = results.get('gap_ang', x)
    if scan_kind == 'proj_gap':
        x = results.get('proj_gap_ang', x)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Plot 1: Δω(R) linear scale ---
    ax = axes[0]
    ax.plot(x, dw * 1000, 'o-', label='Exact (FFT Poisson)', color='C0', markersize=4)
    ax.plot(x, dw_mp * 1000, 's--', label='Multipole', color='C1', markersize=4)
    xlabel0 = {'gap': 'd_min (Å)', 'proj_gap': 'proj_gap (Å)', 'center': 'R (Å)', 'displacement': 'disp (Å)'}.get(scan_kind, 'scan (Å)')
    ax.set_xlabel(xlabel0)
    ax.set_ylabel('Δω (meV)')
    ax.set_title('Site-Energy Shift vs Distance')
    ax.legend()
    ax.axhline(0, color='gray', lw=0.5)
    ax.grid(True, alpha=0.3)

    # --- Plot 2: log-log for power-law analysis ---
    ax = axes[1]
    mask = np.abs(dw) > 0
    if np.any(mask):
        ax.loglog(R[mask], np.abs(dw[mask]), 'o-', label='|Δω| exact', color='C0', markersize=4)
    mask_mp = np.abs(dw_mp) > 0
    if np.any(mask_mp):
        ax.loglog(R[mask_mp], np.abs(dw_mp[mask_mp]), 's--', label='|Δω| multipole', color='C1', markersize=4)
    R_ref = np.linspace(R.min(), R.max(), 100)
    for n, ls in [(3, ':'), (5, '-.'), (7, '--')]:
        scale = np.abs(dw[len(dw)//3]) * R[len(dw)//3]**n if len(dw) > 3 and np.abs(dw[len(dw)//3]) > 0 else 1.0
        ax.loglog(R_ref, scale / R_ref**n, ls, color='gray', alpha=0.5, label=f'1/R^{n}')
    ax.set_xlabel('R (Å)')
    ax.set_ylabel('|Δω| (eV)')
    ax.set_title('Power-Law Decay Analysis')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    # --- Plot 3: Δω × R^n diagnostic (should plateau for correct exponent) ---
    ax = axes[2]
    mask = np.abs(dw) > 0
    if np.any(mask):
        for n, color in [(3, 'C2'), (5, 'C3'), (7, 'C4')]:
            ax.plot(R[mask], np.abs(dw[mask]) * R[mask]**n, 'o-',
                    label=f'|Δω|×R^{n}', color=color, markersize=4)
    ax.set_xlabel('R (Å)')
    ax.set_ylabel('|Δω| × R^n (eV·Å^n)')
    ax.set_title('Compensated Plot (plateau = correct exponent)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_prefix:
        fig.savefig(f"{save_prefix}_shift.png", dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_prefix}_shift.png")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
