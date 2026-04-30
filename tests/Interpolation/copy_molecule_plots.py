#!/usr/bin/env python
"""Plot molecular geometries and copy to tip variant directories."""
import os, glob, shutil, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OPT_GEOM_DIR = os.path.join(SCRIPT_DIR, "data_Mithun_flat/opt-geom")
RELAX_TEST_DIR = os.path.join(SCRIPT_DIR, "data_Mithun_flat/relax_test_hho")
PLOT_SCRIPT = os.path.join(os.path.dirname(SCRIPT_DIR), "plot_molecule.py")

# Get all molecule xyz files
mol_files = sorted(glob.glob(os.path.join(OPT_GEOM_DIR, "*.xyz")))
print(f"[copy_molecule_plots] Found {len(mol_files)} molecule files in {OPT_GEOM_DIR}")

# Get all tip variant directories
tip_dirs = sorted(glob.glob(os.path.join(RELAX_TEST_DIR, "*")))
print(f"[copy_molecule_plots] Found {len(tip_dirs)} tip variant directories in {RELAX_TEST_DIR}")

# Create temporary directory for plots
temp_plot_dir = "temp_molecule_plots"
os.makedirs(temp_plot_dir, exist_ok=True)

for mol_file in mol_files:
    mol_name = os.path.basename(mol_file).replace(".xyz", "")
    plot_file = os.path.join(temp_plot_dir, f"{mol_name}.png")
    
    # Plot the molecule
    print(f"[copy_molecule_plots] Plotting {mol_name}...")
    cmd = [
        "python", PLOT_SCRIPT,
        "--input", mol_file,
        "--output", plot_file,
        "--axes", "0,1",
        "--size", "80",
        "--bonds", "1",
        "--labels", "1"
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"[copy_molecule_plots] Created {plot_file}")
    except subprocess.CalledProcessError as e:
        print(f"[copy_molecule_plots] ERROR plotting {mol_name}: {e}")
        continue
    
    # Copy to all matching tip directories
    matching_dirs = [d for d in tip_dirs if os.path.basename(d).startswith(mol_name)]
    print(f"[copy_molecule_plots] Found {len(matching_dirs)} directories for {mol_name}")
    
    for tip_dir in matching_dirs:
        dest_file = os.path.join(tip_dir, f"{mol_name}.png")
        # Check if file exists to avoid overwriting
        if os.path.exists(dest_file):
            print(f"[copy_molecule_plots] Skipping {dest_file} (already exists)")
            continue
        # Check if it's GridFF.npy or other critical file
        if "gridFF" in dest_file.lower() or "lvec" in dest_file.lower():
            print(f"[copy_molecule_plots] SKIPPING critical file: {dest_file}")
            continue
        shutil.copy(plot_file, dest_file)
        print(f"[copy_molecule_plots] Copied to {dest_file}")

print(f"[copy_molecule_plots] Done. Temporary plots in {temp_plot_dir}")
