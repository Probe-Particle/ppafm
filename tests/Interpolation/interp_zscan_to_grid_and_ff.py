import argparse, os, numpy as np, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from pyProbeParticle.GridUtils import save_vec_field, save_scal_field
from pyProbeParticle import InterpolatorRBF, InterpolatorKriging
from interp_zscan_to_grid import (load_clean_points, load_zscan, build_grid, make_interpolator,
                                   auto_support_radii, plot_z_sequence, plot_single_slice)

def interpolate_volume_and_forces(points_xy, zscan_vals, nx, ny, nz, z0, dz, R_basis, kind='rbf',
                                  dx=None, dy=None, kriging_nugget=0.0, kriging_global_eval=False,
                                  rbf_normalized=False, rbf_eps_norm=0.0):
    xs, ys, zs, grid_points = build_grid(points_xy, nx, ny, nz, z0, dz, dx=dx, dy=dy)
    nx_eff, ny_eff = len(xs), len(ys)
    interp = make_interpolator(kind, points_xy, R_basis, kriging_nugget=kriging_nugget,
                               kriging_global_eval=kriging_global_eval, rbf_normalized=rbf_normalized,
                               rbf_eps_norm=rbf_eps_norm)
    vol = np.zeros((nz, ny_eff, nx_eff), dtype=float)
    vol_Fxy = np.zeros((nz, ny_eff, nx_eff, 2), dtype=float)
    for iz in range(nz):
        if not interp.update_weights(zscan_vals[:, iz]):
            raise RuntimeError(f"Failed weights at z-index {iz}")
        vals = interp.evaluate(grid_points)
        if vals is None:
            raise RuntimeError(f"Interp failed at z-index {iz}")
        vol[iz, :, :] = vals.reshape((ny_eff, nx_eff))
        grads = interp.evaluate_gradient(grid_points)
        if grads is None:
            raise RuntimeError(f"Grad failed at z-index {iz}")
        vol_Fxy[iz, :, :, 0] = -grads[:, 0].reshape((ny_eff, nx_eff))
        vol_Fxy[iz, :, :, 1] = -grads[:, 1].reshape((ny_eff, nx_eff))
    vol_Fz = np.zeros((nz, ny_eff, nx_eff), dtype=float)
    if nz >= 2:
        for iz in range(1, nz - 1):
            vol_Fz[iz, :, :] = -(vol[iz+1, :, :] - vol[iz-1, :, :]) / (2.0 * dz)
        vol_Fz[0, :, :] = -(vol[1, :, :] - vol[0, :, :]) / dz
        vol_Fz[nz-1, :, :] = -(vol[nz-1, :, :] - vol[nz-2, :, :]) / dz
    # Unit conversion: input data is in kcal/mol, PPAFM expects eV (energy) and eV/Å (force)
    kcal_to_eV = 0.043364115
    gridFF = np.zeros((nz, ny_eff, nx_eff, 4), dtype=float)
    gridFF[:, :, :, 0] = vol_Fxy[:, :, :, 0] * kcal_to_eV
    gridFF[:, :, :, 1] = vol_Fxy[:, :, :, 1] * kcal_to_eV
    gridFF[:, :, :, 2] = vol_Fz * kcal_to_eV
    gridFF[:, :, :, 3] = vol * kcal_to_eV
    gridFF = np.ascontiguousarray(gridFF)
    lvec = np.array([[0.,0.,0.], [xs[-1]-xs[0],0.,0.], [0.,ys[-1]-ys[0],0.], [0.,0.,zs[-1]-zs[0]]], dtype=float)
    return xs, ys, zs, gridFF, lvec

def save_gridff_ppafm(prefix, gridFF, lvec, fmt="npy"):
    FF = gridFF[:, :, :, :3]
    E = gridFF[:, :, :, 3]
    save_vec_field(prefix + "_FF", FF, lvec, data_format=fmt)
    save_scal_field(prefix + "_E", E, lvec, data_format=fmt)
    print(f"[GridFF] Saved PPAFM force field: {prefix}_FF_{{x,y,z}}.{fmt}")
    print(f"[GridFF] Saved PPAFM energy field: {prefix}_E.{fmt}")

def main():
    p = argparse.ArgumentParser(description="Interpolate DFT z-scan to 3D GridFF with forces for PPAFM")
    p.add_argument("-k", "--kind",           type=str,   choices=["rbf","kriging"], default="rbf")
    p.add_argument("-p", "--points",         type=str,   default="data_Mithun_new/points/OHO-h_1_points_clean.txt")
    p.add_argument("-z", "--zscan",          type=str,   default="data_Mithun_new/scans/OHO-h_1-CO_O.dat")
    p.add_argument("-o", "--out-npy",        type=str,   default=None, help="Output .npy for GridFF [nx,ny,nz,4]")
    p.add_argument("--out-ppafm-prefix",     type=str,   default=None, help="Save FF/E in PPAFM npy format with this prefix")
    p.add_argument("-x", "--nx",             type=int,   default=50)
    p.add_argument("-y", "--ny",             type=int,   default=50)
    p.add_argument("--dx",                    type=float, default=None)
    p.add_argument("--dy",                    type=float, default=None)
    p.add_argument("-n", "--nz",             type=int,   default=None)
    p.add_argument("--iz0",                   type=int,   default=None)
    p.add_argument("--iz1",                   type=int,   default=None)
    p.add_argument("-s", "--z0",             type=float, default=1.6)
    p.add_argument("-d", "--dz",             type=float, default=0.1)
    p.add_argument("--dz-grid",               type=float, default=None)
    p.add_argument("-r", "--R-basis",        type=float, default=1.2)
    p.add_argument("--kriging-nugget",        type=float, default=0.0)
    p.add_argument("--kriging-global",        type=int,   default=0)
    p.add_argument("--rbf-normalized",        type=int,   default=0)
    p.add_argument("--rbf-eps-norm",          type=float, default=0.0)
    p.add_argument("--autoR-k",               type=int,   default=0)
    p.add_argument("--autoR-scale",           type=float, default=1.3)
    p.add_argument("--autoR-rmin",            type=float, default=0.5)
    p.add_argument("--autoR-rmax",            type=float, default=1e9)
    p.add_argument("--autoR-percentile",      type=float, default=-1.0)
    p.add_argument("-c", "--plot-slice-z",   type=float, default=None)
    p.add_argument("-a", "--zmin",           type=float, default=None)
    p.add_argument("-b", "--zmax",           type=float, default=None)
    p.add_argument("-t", "--zstep",          type=float, default=None)
    p.add_argument("-w", "--show",           type=int,   default=1)
    p.add_argument("-f", "--save-prefix",    type=str,   default=None)
    p.add_argument("--scatter-overlay",       type=int,   default=0)
    p.add_argument("--scatter-size",          type=float, default=8.0)
    p.add_argument("--scatter-alpha",         type=float, default=1.0)
    p.add_argument("--scatter-skip",          type=int,   default=1)
    args = p.parse_args()

    _, points_xy = load_clean_points(args.points)
    zscan_vals = load_zscan(args.zscan)
    R_basis = args.R_basis
    if args.autoR_k and args.autoR_k > 0:
        perc = None if args.autoR_percentile < 0 else float(args.autoR_percentile)
        R_basis = auto_support_radii(points_xy, k=args.autoR_k, scale=args.autoR_scale,
                                     rmin=args.autoR_rmin, rmax=args.autoR_rmax, percentile=perc)
    n_points, n_z_input = zscan_vals.shape
    dz_eff = args.dz if args.dz_grid is None else args.dz_grid
    if args.iz0 is not None or args.iz1 is not None:
        iz0 = 0 if args.iz0 is None else int(args.iz0)
        iz1 = n_z_input if args.iz1 is None else int(args.iz1)
        zscan_vals = zscan_vals[:, iz0:iz1]
        n_z_input = zscan_vals.shape[1]
        args.z0 += dz_eff * iz0
    nz = n_z_input if args.nz is None else min(args.nz, n_z_input)

    xs, ys, zs, gridFF, lvec = interpolate_volume_and_forces(
        points_xy, zscan_vals[:, :nz], nx=args.nx, ny=args.ny, nz=nz, z0=args.z0, dz=dz_eff,
        R_basis=R_basis, kind=args.kind, dx=args.dx, dy=args.dy,
        kriging_nugget=args.kriging_nugget, kriging_global_eval=bool(args.kriging_global),
        rbf_normalized=bool(args.rbf_normalized), rbf_eps_norm=args.rbf_eps_norm)

    if args.out_npy:
        out_dir = os.path.dirname(os.path.abspath(args.out_npy))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        np.save(args.out_npy, gridFF)
        print(f"[GridFF] Saved {args.out_npy}")
    if args.out_ppafm_prefix:
        save_gridff_ppafm(args.out_ppafm_prefix, gridFF, lvec, fmt="npy")

    if args.zmin is not None and args.zmax is not None and args.zstep is not None:
        if args.scatter_overlay:
            plot_z_sequence(xs, ys, zs, gridFF[:, :, :, 3], args.zmin, args.zmax, args.zstep,
                           save_prefix=args.save_prefix, scatter_xy=points_xy,
                           scatter_vals=zscan_vals[:, :nz], scatter_z0=args.z0, scatter_dz=dz_eff,
                           scatter_size=args.scatter_size, scatter_alpha=args.scatter_alpha,
                           scatter_skip=args.scatter_skip)
        else:
            plot_z_sequence(xs, ys, zs, gridFF[:, :, :, 3], args.zmin, args.zmax, args.zstep,
                           save_prefix=args.save_prefix)
    elif args.show or args.plot_slice_z is not None:
        plot_single_slice(xs, ys, zs, gridFF[:, :, :, 3], plot_slice_z=args.plot_slice_z,
                         show=args.show, save_prefix=args.save_prefix)

if __name__ == "__main__":
    main()
