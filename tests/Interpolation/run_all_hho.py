#!/usr/bin/env python
"""Run test_relax_kriging.py for all HHO-h-* datasets."""
import os, glob, subprocess, sys, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default="data_Mithun_new", help='Input data directory')
parser.add_argument('--out_dir', type=str, default=None, help='Output directory (default: {data_dir}/relax_test_hho)')
args = parser.parse_args()

DATA_DIR = args.data_dir
BASE_OUT_DIR = args.out_dir if args.out_dir else os.path.join(DATA_DIR, "relax_test_hho")

# Try different directory structures for points files
# Option 1: points_clean/ with _points_clean.txt suffix
points_pattern1 = os.path.join(DATA_DIR, "points_clean", "*-h*_points_clean.txt")
points_files = glob.glob(points_pattern1)

# Option 2: endgroup_points/ with _point_info.txt suffix
if not points_files:
    points_pattern2 = os.path.join(DATA_DIR, "endgroup_points", "*-h*_point_info.txt")
    points_files = glob.glob(points_pattern2)
    points_suffix = "_point_info.txt"
else:
    points_suffix = "_points_clean.txt"

if not points_files:
    print(f"[run_all] No *-h* files found in either:")
    print(f"  - {points_pattern1}")
    print(f"  - {points_pattern2}")
    sys.exit(1)

print(f"[run_all] Found {len(points_files)} *-h* datasets in {points_pattern1 if os.path.exists(os.path.dirname(points_pattern1)) else points_pattern2}")

for points_file in sorted(points_files):
    # Extract basename (e.g., HHO-h-p_1 from HHO-h-p_1_points_clean.txt or HHO-h-p_1_point_info.txt)
    basename = os.path.basename(points_file).replace(points_suffix, "")
    
    # Find all zscan files for this molecule (different tips)
    zscan_pattern = os.path.join(DATA_DIR, "results", f"{basename}-*.dat")
    zscan_files = glob.glob(zscan_pattern)
    
    if not zscan_files:
        print(f"[run_all] Skipping {basename}: no zscan files found matching: {zscan_pattern}")
        continue
    
    print(f"[run_all] Found {len(zscan_files)} tip variants for {basename}")
    
    for zscan_file in sorted(zscan_files):
        # Extract tip variant name (e.g., CO_O from HHO-h-p_1-CO_O.dat)
        tip_variant = os.path.basename(zscan_file).replace(f"{basename}-", "").replace(".dat", "")
        
        # Create output directory for this tip variant
        out_dir = os.path.join(BASE_OUT_DIR, f"{basename}_{tip_variant}")
        os.makedirs(out_dir, exist_ok=True)
        
        # Build command
        cmd = [
            "python", "test_relax_kriging.py",
            "--points_file", points_file,
            "--zscan_file", zscan_file,
            "--out_dir", out_dir,
            "--save_outputs", "0",
            "--plot_comparison", "1",
            "--plot_pppos", "0",
            "--plot_gridff", "0",
            "--outfz_cmap", "gray"
        ]
        
        print(f"[run_all] Running for {basename} tip={tip_variant}...")
        print(f"[run_all] Output: {out_dir}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            print(f"[run_all] Completed {basename} tip={tip_variant}")
        except subprocess.CalledProcessError as e:
            print(f"[run_all] FAILED for {basename} tip={tip_variant}: {e}")
            continue

print(f"[run_all] Done. Results in: {BASE_OUT_DIR}")
