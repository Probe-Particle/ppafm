import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

# Make sure we can import pyProbeParticle from the repo root when run from tests/Interpolation
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from pyProbeParticle.InterpolatorRBF import InterpolatorRBF
from pyProbeParticle.InterpolatorKriging import InterpolatorKriging


def load_clean_points(fname):
    """Load points from file with header 'type x y' or 'index type x y'."""
    types = []
    xs = []
    ys = []
    with open(fname, 'r') as f:
        header = f.readline().strip().split()
        # Determine column indices
        if header[0] == 'type':
            col_type, col_x, col_y = 0, 1, 2
        elif header[0] == 'index' and len(header) >= 4:
            col_type, col_x, col_y = 1, 2, 3
        else:
            # Fallback: assume 'type x y' without header
            f.seek(0)
            col_type, col_x, col_y = 0, 1, 2

        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) <= max(col_type, col_x, col_y):
                continue
            types.append(parts[col_type])
            xs.append(float(parts[col_x]))
            ys.append(float(parts[col_y]))

    pts = np.stack([xs, ys], axis=1)
    return types, pts


def load_zscan(fname):
    """Load z-scan data with lines of the form 'p000 z00 value'.

    Returns
    -------
    values : ndarray, shape (n_points, n_z)
    """
    data = {}
    with open(fname, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            p_label, z_label, val_str = parts
            try:
                val = float(val_str)
            except ValueError:
                continue

            if not p_label.startswith('p') or not z_label.startswith('z'):
                continue

            try:
                i_point = int(p_label[1:])
                i_z = int(z_label[1:])
            except ValueError:
                continue

            data.setdefault(i_point, {})[i_z] = val

    if not data:
        raise RuntimeError(f"No z-scan data parsed from {fname}")

    point_indices = sorted(data.keys())
    z_indices = sorted(next(iter(data.values())).keys())

    n_points = len(point_indices)
    n_z = len(z_indices)
    values = np.zeros((n_points, n_z), dtype=float)

    for ip, p in enumerate(point_indices):
        row = data[p]
        for iz, z in enumerate(z_indices):
            try:
                values[ip, iz] = row[z]
            except KeyError:
                raise RuntimeError(f"Missing value for point p{p:03d} z{z:02d}")

    return values


def build_grid(points_xy, nx, ny, nz, z0, dz, pad=0.1):
    xmin = points_xy[:, 0].min()
    xmax = points_xy[:, 0].max()
    ymin = points_xy[:, 1].min()
    ymax = points_xy[:, 1].max()

    # Simple padding around the convex hull
    Lx = xmax - xmin
    Ly = ymax - ymin
    xmin -= pad * Lx
    xmax += pad * Lx
    ymin -= pad * Ly
    ymax += pad * Ly

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    zs = z0 + dz * np.arange(nz)

    Xs, Ys = np.meshgrid(xs, ys)
    grid_points = np.stack([Xs.ravel(), Ys.ravel()], axis=1)

    return xs, ys, zs, grid_points


def make_interpolator(kind, points_xy, R_basis):
    if kind == 'rbf':
        return InterpolatorRBF(points_xy, R_basis)
    elif kind == 'kriging':
        return InterpolatorKriging(points_xy, R_basis)
    else:
        raise ValueError(f"Unknown interpolator kind '{kind}', expected 'rbf' or 'kriging'")


def interpolate_volume(points_xy, zscan_vals, nx, ny, nz, z0, dz, R_basis, kind='rbf'):
    types = None  # not needed here
    xs, ys, zs, grid_points = build_grid(points_xy, nx, ny, nz, z0, dz)

    interp = make_interpolator(kind, points_xy, R_basis)

    vol = np.zeros((nz, ny, nx), dtype=float)

    for iz in range(nz):
        data_vals = zscan_vals[:, iz]
        if not interp.update_weights(data_vals):
            raise RuntimeError(f"Failed to update weights for z-index {iz}")
        vals_xy = interp.evaluate(grid_points)
        if vals_xy is None:
            raise RuntimeError(f"Interpolation failed at z-index {iz}")
        vol[iz, :, :] = vals_xy.reshape((ny, nx))

    return xs, ys, zs, vol


def plot_z_sequence(xs, ys, zs, vol, zmin, zmax, zstep, save_prefix=None):
    """Plot a sequence of heights using linear interpolation along z.

    Parameters
    ----------
    xs, ys, zs : 1D arrays defining the regular grid
    vol        : 3D array (nz, ny, nx)
    zmin, zmax, zstep : float
        Range of z values to visualize.
    save_prefix : str or None
        If not None, treat this as output directory and save each slice there.
    """
    nz = vol.shape[0]
    extent = (xs[0], xs[-1], ys[0], ys[-1])

    print("[DEBUG] plot_z_sequence: zmin=", zmin, "zmax=", zmax, "zstep=", zstep, "save_prefix=", save_prefix)

    out_dir = None
    if save_prefix is not None:
        out_dir = save_prefix
        os.makedirs(out_dir, exist_ok=True)

    z_plots = np.arange(zmin, zmax + 0.5 * zstep, zstep)
    z_grid = zs

    if len(z_grid) < 2:
        # Not enough slices for interpolation; just pick nearest existing ones
        for z_plot in z_plots:
            if z_plot < z_grid[0] or z_plot > z_grid[-1]:
                continue
            iz = int(np.round((z_plot - z_grid[0]) / (z_grid[1] - z_grid[0])))
            iz = max(0, min(nz - 1, iz))
            slice2d = vol[iz, :, :]
            fig = plt.figure()
            plt.imshow(slice2d, origin='lower', extent=extent, aspect='auto')
            plt.colorbar(label='Interpolated value')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'z ~ {z_plot:.3f} (nearest slice)')
            if out_dir is not None:
                fname = os.path.join(out_dir, f"slice_z{z_plot:.3f}.png")
                fig.savefig(fname, dpi=150)
            plt.close(fig)
        return

    for z_plot in z_plots:
        # Skip values outside available z range
        if z_plot < z_grid[0] or z_plot > z_grid[-1]:
            continue

        # Find neighboring indices for linear interpolation (uniform spacing assumed)
        t = (z_plot - z_grid[0]) / (z_grid[1] - z_grid[0])
        iz0 = int(np.floor(t))
        iz1 = min(iz0 + 1, nz - 1)
        if iz0 == iz1:
            slice2d = vol[iz0, :, :]
        else:
            alpha = (z_plot - z_grid[iz0]) / (z_grid[iz1] - z_grid[iz0])
            slice2d = (1.0 - alpha) * vol[iz0, :, :] + alpha * vol[iz1, :, :]

        fig = plt.figure()
        plt.imshow(slice2d, origin='lower', extent=extent, aspect='auto')
        plt.colorbar(label='Interpolated value')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'z ~ {z_plot:.3f} (linear interp)')
        if out_dir is not None:
            fname = os.path.join(out_dir, f"slice_z{z_plot:.3f}.png")
            fig.savefig(fname, dpi=150)
        plt.close(fig)


def plot_single_slice(xs, ys, zs, vol, plot_slice_z=None, show=True, save_prefix=None):
    """Plot a single z slice.

    If plot_slice_z is None, take the middle slice.
    Otherwise, choose the nearest stored z in zs.
    """
    nz = vol.shape[0]
    extent = (xs[0], xs[-1], ys[0], ys[-1])

    if plot_slice_z is None:
        iz = nz // 2
    else:
        dz = zs[1] - zs[0] if len(zs) > 1 else 1.0
        iz = int(round((plot_slice_z - zs[0]) / dz))
        iz = max(0, min(nz - 1, iz))

    fig = plt.figure()
    plt.imshow(vol[iz, :, :], origin='lower', extent=extent, aspect='auto')
    plt.colorbar(label='Interpolated value')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'z-slice index {iz}, z ~ {zs[iz]:.3f}')
    if save_prefix is not None:
        os.makedirs(save_prefix, exist_ok=True)
        fname = os.path.join(save_prefix, f"slice_z{zs[iz]:.3f}.png")
        fig.savefig(fname, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)





'''

python interp_zscan_to_grid.py \
    --points data_Mithun/points/OHO-h_1_points_clean.txt \
    --zscan  data_Mithun/scans/OHO-h_1-CO_O.dat \
    --out-npy data_Mithun/OHO-h_1-CO_O_volume_rbf.npy \
    --nx 50 --ny 50 \
    --kind rbf --R-basis 1.2 \
    --z0 1.6 --dz 0.1 \
    --zmin 2.0 --zmax 4.0 --zstep 0.1 \
    --save-prefix data_Mithun/OHO-h_1-CO_O_slice

python interp_zscan_to_grid.py \
    --points data_Mithun/points/OHO-h_1_points_clean.txt \
    --zscan  data_Mithun/scans/OHO-h_1-CO_O.dat \
    --out-npy data_Mithun/OHO-h_1-CO_O_volume_rbf.npy \
    --save-prefix data_Mithun/OHO-h_1-CO_O_slice \
    --nx 50 --ny 50 \
    --kind rbf --R-basis 2.0 \
    --z0 1.6 --dz 0.1 \
    --zmin 2.0 --zmax 4.0 --zstep 0.1 

'''


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Interpolate DFT z-scan data from irregular points to regular 3D grid")
    p.add_argument("-k","--kind",           type=str,   choices=["rbf","kriging"], default="rbf",                                    help="Interpolator type")
    p.add_argument("-p","--points",         type=str,   default="data_Mithun/points/OHO-h_1_points_clean.txt",                       help="Clean points file (type x y or index type x y)")
    p.add_argument("-z","--zscan",          type=str,   default="data_Mithun/scans/OHO-h_1-CO_O.dat",                                help="Z-scan data file (pNNN zMM value)")
    p.add_argument("-o","--out-npy",        type=str,   default="data_Mithun/OHO-h_1-CO_O_volume_rbf.npy",                           help="Output .npy file for 3D volume (nz, ny, nx)")
    p.add_argument("-x","--nx",             type=int,   default=50,                                                                  help="Number of grid points in x")
    p.add_argument("-y","--ny",             type=int,   default=50,                                                                  help="Number of grid points in y")
    p.add_argument("-n","--nz",             type=int,   default=None,                                                                help="Number of grid points in z (default: use all from input)")
    p.add_argument("-s","--z0",             type=float, default=1.6,                                                                 help="Starting z value (offset)")
    p.add_argument("-d","--dz",             type=float, default=0.1,                                                                 help="Grid spacing in z")
    p.add_argument("-r","--R-basis",        type=float, default=1.2,                                                                 help="Support radius for RBF/Kriging kernel")
    p.add_argument("-c","--plot-slice-z",   type=float, default=None,                                                                help="Optional single z-value to plot nearest slice with imshow")
    p.add_argument("-a","--zmin",           type=float, default=None,                                                                help="Optional minimum z for sequence of slices (linear interp in z)")
    p.add_argument("-b","--zmax",           type=float, default=None,                                                                help="Optional maximum z for sequence of slices (linear interp in z)")
    p.add_argument("-t","--zstep",          type=float, default=None,                                                                help="Optional z step for sequence of slices (linear interp in z)")
    p.add_argument("-w","--show",           type=int,   default=1,                                                                   help="Show matplotlib plot for selected slice(s)")
    p.add_argument("-f","--save-prefix",    type=str,   default=None,                                                                help="If given, save plotted slice(s) as PNG with this prefix")
    args = p.parse_args()

    _, points_xy = load_clean_points(args.points)
    zscan_vals = load_zscan(args.zscan)

    n_points, n_z_input = zscan_vals.shape
    if args.nz is None:
        nz = n_z_input
    else:
        nz = min(args.nz, n_z_input)

    xs, ys, zs, vol = interpolate_volume(
        points_xy,
        zscan_vals[:, :nz],
        nx=args.nx,
        ny=args.ny,
        nz=nz,
        z0=args.z0,
        dz=args.dz,
        R_basis=args.R_basis,
        kind=args.kind,
    )

    np.save(args.out_npy, vol)

    # --- Determine whether to plot a sequence of heights or a single slice ---
    print("[DEBUG] main: zmin=", args.zmin, "zmax=", args.zmax, "zstep=", args.zstep,
          "plot_slice_z=", args.plot_slice_z, "show=", args.show, "save_prefix=", args.save_prefix)

    if (args.zmin is not None) and (args.zmax is not None) and (args.zstep is not None):
        print("[DEBUG] main: calling plot_z_sequence()")
        plot_z_sequence(xs, ys, zs, vol, args.zmin, args.zmax, args.zstep, save_prefix=args.save_prefix)
    elif args.show or args.plot_slice_z is not None:
        print("[DEBUG] main: calling plot_single_slice()")
        plot_single_slice(xs, ys, zs, vol, plot_slice_z=args.plot_slice_z, show=args.show, save_prefix=args.save_prefix)
    else:
        print("[DEBUG] main: no plotting requested (neither z-range nor single slice)")
