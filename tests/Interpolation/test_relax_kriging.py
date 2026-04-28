#!/usr/bin/env python
"""Test script: Generate GridFF from Kriging interpolation and run PPAFM relaxation."""
import os, sys, numpy as np, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import pyProbeParticle                as PPU
import pyProbeParticle.GridUtils      as GU
import pyProbeParticle.HighLevel      as PPH
from interp_zscan_to_grid_and_ff import interpolate_volume_and_forces, save_gridff_ppafm
from interp_zscan_to_grid import load_clean_points, load_zscan, load_point_info, auto_support_radii

parser = argparse.ArgumentParser()
parser.add_argument('--points_file', type=str, default=None, help='Path to points file')
parser.add_argument('--zscan_file', type=str, default=None, help='Path to z-scan file')
parser.add_argument('--out_dir', type=str, default=None, help='Output directory')
parser.add_argument('--save_outputs', type=int, default=0, help='Save auxiliary .npy files (0=no, 1=yes)')
parser.add_argument('--plot_outfz', type=int, default=1, help='Plot OutFz_all_slices.png (0=no, 1=yes)')
parser.add_argument('--outfz_cmap', type=str, default='afmhot', help='Colormap for OutFz_all_slices.png (afmhot or gray)')
parser.add_argument('--plot_pppos', type=int, default=1, help='Plot PPpos_top_slice.png (0=no, 1=yes)')
parser.add_argument('--plot_comparison', type=int, default=1, help='Plot GridFF_vs_OutFz.png (0=no, 1=yes)')
parser.add_argument('--plot_gridff', type=int, default=1, help='Plot GridFF_approach_slices.png (0=no, 1=yes)')
args = parser.parse_args()

DATA_DIR = "data_Mithun_new"
OUT_DIR = args.out_dir if args.out_dir else "data_Mithun_new/relax_test"
os.makedirs(OUT_DIR, exist_ok=True)

# --- 1. Load data ---
if args.points_file and args.zscan_file:
    points_file = args.points_file
    zscan_file = args.zscan_file
else:
    points_file = os.path.join(DATA_DIR, "points_clean", "OHO-h_1_points_clean.txt")
    zscan_file = os.path.join(DATA_DIR, "results", "OHO-h_1-CO_O.dat")
print("[test] Loading points:", points_file)
# Detect file format and use appropriate loader
if "_point_info.txt" in points_file:
    _, points_xy = load_point_info(points_file)
else:
    _, points_xy = load_clean_points(points_file)
print("[test] Loading z-scan:", zscan_file)
zscan_vals = load_zscan(zscan_file)
print(f"[test] Points: {points_xy.shape}, z-scan shape: {zscan_vals.shape}")

# --- 2. Interpolate to GridFF ---
# Check if precomputed GridFF exists (single file)
gridFF_file = os.path.join(OUT_DIR, "gridFF.npy")
lvec_file = os.path.join(OUT_DIR, "lvec.npy")
if os.path.exists(gridFF_file) and os.path.exists(lvec_file):
    print(f"[test] Loading precomputed GridFF from {gridFF_file}")
    gridFF = np.load(gridFF_file)
    lvec = np.load(lvec_file)
    # Reconstruct xs, ys, zs from lvec for plotting
    xs = np.linspace(0, lvec[1,0], gridFF.shape[2])
    ys = np.linspace(0, lvec[2,1], gridFF.shape[1])
    zs = np.linspace(0, lvec[3,2], gridFF.shape[0])
    # Shift to match original z0, dz
    zs = zs + 1.6
    dz = 0.1
else:
    R_basis = 8.0
    nx, ny = 100, 100
    nz = zscan_vals.shape[1]
    z0, dz = 1.6, 0.1
    print(f"[test] Interpolating to GridFF: nx={nx} ny={ny} nz={nz} R_basis={R_basis} kind=kriging")
    xs, ys, zs, gridFF, lvec = interpolate_volume_and_forces(
        points_xy, zscan_vals,
        nx=nx, ny=ny, nz=nz, z0=z0, dz=dz,
        R_basis=R_basis, kind='kriging'
    )
    print(f"[test] GridFF shape: {gridFF.shape}")
    # Save GridFF for future runs (single file)
    np.save(gridFF_file, gridFF)
    np.save(lvec_file, lvec)

# --- 3. Run PPAFM relaxation ---
# Set PPU params directly (skip params.ini)
PPU.params['klat']      = 0.5
PPU.params['charge']    = 0.0
PPU.params['Amplitude'] = 1.0
PPU.params['scanStep']  = np.array([0.1, 0.1, 0.1])
# Probe hangs lRadial=4.0 BELOW tip (rProbe.z = rTip.z - 4.0). Grid z = 1.6 to 6.0.
# zTip for OutFz: 6.6-12.0 Å (as specified). GridFF stays at 1.6-6.0 Å.
PPU.params['scanMin']   = np.array([0.0, 0.0, 6.6])
PPU.params['scanMax']   = np.array([xs[-1]-xs[0], ys[-1]-ys[0], 12.0])
PPU.params['tilt']      = np.array([0.0, 0.0])
PPU.params['flexible']  = True
PPU.params['stiffness'] = np.array([0.5, 0.5, 0.0])
# kCantilever etc. may be required
PPU.lvec2params(lvec)

FF = gridFF[:, :, :, :3]

print("[test] Running PPAFM relaxation...")
fzs, PPpos, PPdisp, lvecScan = PPH.perform_relaxation(lvec, FF, FFel=None, FFpauli=None, FFboltz=None, tipspline=None, bPPdisp=True, bFFtotDebug=False)
print(f"[test] fzs shape: {fzs.shape}, PPpos shape: {PPpos.shape}")

# --- 4. Save outputs ---
if args.save_outputs:
    GU.save_scal_field(os.path.join(OUT_DIR, "OutFz"), fzs, lvecScan, data_format="npy")
    GU.save_vec_field(os.path.join(OUT_DIR, "PPpos"), PPpos, lvecScan, data_format="npy")

# --- 5. Plot results ---
try:
    import matplotlib.pyplot as plt
    extent = (lvecScan[0, 0], lvecScan[0, 0] + lvecScan[1, 0],
              lvecScan[0, 1], lvecScan[0, 1] + lvecScan[2, 1])
    # --- Compute zTips and probe-equilibrium z for mapping to GridFF ---
    xTips, yTips, zTips, _ = PPU.prepareScanGrids()
    zProbe_eq = zTips - PPU.params['r0Probe'][2]
    # Filter: only show slices where probe z >= 5.0A (common for all plots)
    zProbe_all = zTips - PPU.params['r0Probe'][2]
    iz_start = np.argmax(zProbe_all >= 5.0)
    nz_out = fzs.shape[0] - iz_start
    zProbe_filtered = zProbe_all[iz_start:]
    E = gridFF[:, :, :, 3]
    Fz_grid = gridFF[:, :, :, 2]

    # --- Plot ALL OutFz z-slices (approach sequence) ---
    if args.plot_outfz:
        ncols = 9
        nrows = (nz_out + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.0))
        if nrows == 1:
            axes = axes.reshape(1, -1)
        for i, iz in enumerate(range(iz_start, fzs.shape[0])):
            ax = axes[i // ncols, i % ncols]
            z_tip = zTips[iz]
            z_probe = zProbe_all[iz]
            im = ax.imshow(fzs[iz, :, :], origin='lower', extent=extent, cmap=args.outfz_cmap)
            ax.set_title(f"Fz z={z_probe:.1f}A")
            ax.set_xticks([]); ax.set_yticks([])
        for j in range(nz_out, nrows * ncols):
            axes[j // ncols, j % ncols].axis('off')
        plt.tight_layout()
        out_png = os.path.join(OUT_DIR, "OutFz_all_slices.png")
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"[test] Saved ALL Fz slices ({nz_out} panels): {out_png}")

    # --- Plot relative PP displacement dX,dY,dZ = PPpos - TipPos at top slice ---
    if args.plot_pppos:
        # PPpos shape: (nz, ny, nx, 3), xTips: (nx,), yTips: (ny,)
        dX = PPpos[-1, :, :, 0] - xTips[None, :]
        dY = PPpos[-1, :, :, 1] - yTips[:, None]
        dZ = PPpos[-1, :, :, 2] - zTips[-1] + 4.0  # add 4A to get values around 0
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for i, (title, arr) in enumerate([('dX = PPx-TipX', dX), ('dY = PPy-TipY', dY), ('dZ = PPz-TipZ', dZ)]):
            ax = axes[i]
            vlim = max(abs(arr.min()), abs(arr.max()))
            im = ax.imshow(arr, origin='lower', extent=extent, cmap='bwr', vmin=-vlim, vmax=vlim)
            ax.set_title(title)
            plt.colorbar(im, ax=ax)
        plt.tight_layout()
        out_png2 = os.path.join(OUT_DIR, "PPpos_top_slice.png")
        fig.savefig(out_png2, dpi=150)
        plt.close(fig)
        print(f"[test] Saved PPpos top slice: {out_png2}")

    # --- Side-by-side: GridFF E / GridFF Fz / Relaxed OutFz / PP displacement at matching probe z ---
    if args.plot_comparison:
        # GridFF shifted by 5A relative to OutFz coordinate system
        # OutFz probe z >= 5.0A, GridFF z = probe z - 5.0 (maps to valid grid range 0.1-3.0A)
        # With tip z 6.6-12.0 and lRadial=4.0, probe z = 2.6-8.0. Filter to z >= 5.0A.
        # GridFF index: start at probe z - 5.0, increment by dz
        iz_grid0 = int(round((zProbe_filtered[0] - 5.0 - zs[0]) / dz))
        iz_grid0 = max(0, min(iz_grid0, len(zs) - nz_out))  # clamp to valid range
        print(f"[debug] OutFz start z={zProbe_filtered[0]:.1f}A, GridFF z={zProbe_filtered[0]-5.0:.1f}A, iz_grid0={iz_grid0}")
        fig, axes = plt.subplots(nz_out, 6, figsize=(18, nz_out * 1.8))
        if nz_out == 1:
            axes = axes.reshape(1, -1)
        for i_scan in range(nz_out):
            iz_scan = iz_start + i_scan
            iz_grid = iz_grid0 + i_scan
            z_tip = zTips[iz_scan]
            z_probe = zProbe_all[iz_scan]
            z_grid = zs[iz_grid]
            # Compute PP displacement for this slice
            dX_slice = PPpos[iz_scan, :, :, 0] - xTips[None, :]
            dY_slice = PPpos[iz_scan, :, :, 1] - yTips[:, None]
            dZ_slice = PPpos[iz_scan, :, :, 2] - zTips[iz_scan] + 4.0
            ax_row = axes[i_scan]
            for ax, arr, title, cmap in zip(ax_row,
                [E[iz_grid, :, :], Fz_grid[iz_grid, :, :], fzs[iz_scan, :, :],
                 dX_slice, dY_slice, dZ_slice],
                [f"E z={z_grid:.1f}A", f"Fz_grid z={z_grid:.1f}A", f"OutFz z={z_probe:.1f}A",
                 "dX", "dY", "dZ"],
                ['RdBu_r', 'RdBu_r', 'RdBu_r', 'bwr', 'bwr', 'bwr']):
                im = ax.imshow(arr, origin='lower', extent=extent if 'OutFz' in title else (xs[0], xs[-1], ys[0], ys[-1]), cmap=cmap)
                if cmap == 'bwr':
                    vlim = max(abs(arr.min()), abs(arr.max()))
                    im.set_clim(-vlim, vlim)
                ax.set_title(title, fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
                plt.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        out_png3 = os.path.join(OUT_DIR, "GridFF_vs_OutFz.png")
        fig.savefig(out_png3, dpi=150)
        plt.close(fig)
        print(f"[test] Saved side-by-side comparison ({nz_out} rows): {out_png3}")

    # --- Plot GridFF E,Fx,Fy,Fz at multiple z levels (raw) ---
    if args.plot_gridff:
        Fx = gridFF[:, :, :, 0]
        Fy = gridFF[:, :, :, 1]
        n_grid_show = min(9, len(zs))
        fig, axes = plt.subplots(n_grid_show, 4, figsize=(16, n_grid_show * 2.2))
        if n_grid_show == 1:
            axes = axes.reshape(1, -1)
        for i in range(n_grid_show):
            iz = int((i / max(n_grid_show - 1, 1)) * (len(zs) - 1))
            zval = zs[iz]
            for ax, arr, title in zip(axes[i], [E[iz, :, :], Fx[iz, :, :], Fy[iz, :, :], Fz_grid[iz, :, :]], ['E', 'Fx', 'Fy', 'Fz']):
                im = ax.imshow(arr, origin='lower', extent=(xs[0], xs[-1], ys[0], ys[-1]), cmap='RdBu_r')
                ax.set_title(f"{title} z={zval:.2f}A")
                ax.set_xticks([]); ax.set_yticks([])
                plt.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        out_png4 = os.path.join(OUT_DIR, "GridFF_approach_slices.png")
        fig.savefig(out_png4, dpi=150)
        plt.close(fig)
        print(f"[test] Saved GridFF approach sequence: {out_png4}")
except Exception as e:
    print(f"[test] Plotting skipped: {e}")

print("[test] Done. Outputs in:", OUT_DIR)
