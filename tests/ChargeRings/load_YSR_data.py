"""
Script to load 2D grid spectroscopy data into a 3D numpy array (x, y, energy).
Files expected: Grid_Spectroscopy*-PointSpec#####_deconv.dat
Columns: Energy, DOS, DOStip, dIdV, dIdVrec.

Usage example:
    python load_grid_data.py --folder . --nx 45 --ny 45 --quantity dIdVrec
"""
import os
import glob
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt

COLUMN_MAP = {
    'Energy': 0,
    'DOS': 1,
    'DOStip': 2,
    'dIdV': 3,
    'dIdVrec': 4,
}


def load_grid_data(folder, nx=None, ny=None, quantity='dIdVrec'):
    files = sorted(glob.glob(os.path.join(folder, '*_deconv.dat')))
    if not files:
        raise FileNotFoundError(f"No data files found in {folder}")
    # extract point IDs
    pattern = re.compile(r"PointSpec(\d+)_deconv\.dat$")
    pts = []
    for f in files:
        m = pattern.search(os.path.basename(f))
        if not m:
            continue
        pts.append((int(m.group(1)), f))
    pts.sort(key=lambda x: x[0])
    N = len(pts)
    # infer grid size if not provided
    if nx is None or ny is None:
        side = int(np.sqrt(N))
        if side*side != N:
            raise ValueError(f"Cannot infer square grid from {N} files")
        nx = ny = side
    # load first file to get energy axis
    data0 = np.loadtxt(pts[0][1], comments='#')
    energies = data0[:, 0]
    nE = energies.size
    # prepare volume
    vol = np.zeros((ny, nx, nE))
    # fill data
    col_idx = COLUMN_MAP.get(quantity)
    if col_idx is None:
        raise ValueError(f"Unknown quantity '{quantity}', choose from {list(COLUMN_MAP)}")
    for idx, fpath in pts:
        i = idx - 1
        x = i % nx
        y = i // nx
        data = np.loadtxt(fpath, comments='#')
        if data.shape[1] <= col_idx:
            raise ValueError(f"File {fpath} has no column index {col_idx}")
        vol[y, x, :] = data[:, col_idx]

    vol[18,32, :] = np.nan
    
    return energies, vol


def main():
    parser = argparse.ArgumentParser(description='Load grid spectroscopy data into 3D array')
    parser.add_argument('--folder', '-f', default='./YSR', help='Folder containing _deconv.dat files')
    parser.add_argument('--nx', type=int, default=45, help='Grid size in x (infer if omitted)')
    parser.add_argument('--ny', type=int, default=45, help='Grid size in y (infer if omitted)')
    parser.add_argument('--quantity', '-q', default='DOS', choices=list(COLUMN_MAP), help='Which column to load into volume')
    parser.add_argument('--plot', action='store_false', help='Show demo plots')
    parser.add_argument('--axis', '-a', choices=['x','y','E'], default='x', help='Axis to slice: E,x,y')
    parser.add_argument('--slice-index', '-s', type=int, default=None, help='Index along axis for the slice; default center')
    parser.add_argument('--energy', '-E', type=float, default=None, help='Energy value for E-slice; overrides slice-index')
    args = parser.parse_args()

    energies, vol = load_grid_data(args.folder, args.nx, args.ny, args.quantity)
    print(f"Loaded volume: {vol.shape}, energy axis length: {energies.size}")

    if args.plot:
        # determine slice and prepare data2d
        if args.axis == 'E':
            if args.energy is not None:
                i1 = np.searchsorted(energies, args.energy)
                if i1 == 0 or i1 >= energies.size:
                    raise ValueError(f"Energy {args.energy} out of range [{energies[0]}, {energies[-1]}]")
                i0 = i1 - 1
                t = (args.energy - energies[i0]) / (energies[i1] - energies[i0])
                data2d = (1-t) * vol[:, :, i0] + t * vol[:, :, i1]
                slice_label = f"{args.energy:.3f}"
            else:
                idx = args.slice_index if args.slice_index is not None else energies.size // 2
                data2d = vol[:, :, idx]
                slice_label = f"index {idx}"
            xlabel, ylabel = 'x index', 'y index'
            extent = [0, args.nx-1, 0, args.ny-1]
        elif args.axis == 'x':
            idx = args.slice_index if args.slice_index is not None else args.nx // 2
            data2d = vol[:, idx, :]
            slice_label = f"index {idx}"
            xlabel, ylabel = 'Energy', 'y index'
            extent = [energies[0], energies[-1], 0, args.ny-1]
        else:  # args.axis == 'y'
            idx = args.slice_index if args.slice_index is not None else args.ny // 2
            data2d = vol[idx, :, :]
            slice_label = f"index {idx}"
            xlabel, ylabel = 'Energy', 'x index'
            extent = [energies[0], energies[-1], 0, args.nx-1]

        # plot slice
        plt.figure(figsize=(8,8))
        plt.title(f"{args.quantity} slice along {args.axis} at {slice_label}")
        plt.imshow(data2d, origin='lower', aspect='auto', extent=extent)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.colorbar(label=args.quantity)
        plt.tight_layout()
        if args.axis == 'E':
            plt.axis('equal')
        plt.show()

if __name__ == '__main__':
    main()
