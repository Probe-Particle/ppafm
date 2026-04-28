#!/usr/bin/env python
"""Run test_relax_kriging.py for all HHO-h-* datasets."""
import os, glob, subprocess, sys

DATA_DIR = "data_Mithun_new"
BASE_OUT_DIR = "data_Mithun_new/relax_test_hho"

# Find all *-h* points files (matches HHO-h-p_1, HN-hh, OHO-h_1, etc.)
points_pattern = os.path.join(DATA_DIR, "points_clean", "*-h*_points_clean.txt")
points_files = glob.glob(points_pattern)

if not points_files:
    print(f"[run_all] No *-h* files found matching: {points_pattern}")
    sys.exit(1)

print(f"[run_all] Found {len(points_files)} *-h* datasets")

for points_file in sorted(points_files):
    # Extract basename (e.g., HHO-h-p_1 from HHO-h-p_1_points_clean.txt)
    basename = os.path.basename(points_file).replace("_points_clean.txt", "")
    
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
