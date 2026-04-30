#!/usr/bin/env python
"""Plot sampling points on top of molecular geometry."""
import os, glob, shutil, subprocess, sys, numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from pyProbeParticle.AtomicSystem import AtomicSystem
from pyProbeParticle import plotUtils as PU
from interp_zscan_to_grid import load_point_info

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OPT_GEOM_DIR = os.path.join(SCRIPT_DIR, "data_Mithun_flat/opt-geom")
POINTS_DIR = os.path.join(SCRIPT_DIR, "data_Mithun_flat/endgroup_points")
RELAX_TEST_DIR = os.path.join(SCRIPT_DIR, "data_Mithun_flat/relax_test_hho")
PLOT_SCRIPT = os.path.join(os.path.dirname(SCRIPT_DIR), "plot_molecule.py")

# Color map for point types
POINT_COLORS = {
    'N': 'blue',
    'C': 'black',
    'H': 'gray',
    'O': 'red',
    'kink': 'purple',
    'center': 'green',
    'bond': 'orange',
    'cp': 'cyan',
    'grid': 'lightblue',
}

def plot_points_on_molecule(mol_file, points_file, output_file):
    """Plot molecular geometry with sampling points overlaid."""
    # Load molecular geometry
    sys = AtomicSystem(fname=mol_file)
    
    # Load sampling points
    point_types, point_coords = load_point_info(points_file)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot bonds
    if sys.bonds is None:
        sys.findBonds(Rcut=3.0, RvdwCut=0.5)
    PU.plotBonds(links=sys.bonds, ps=sys.apos, axes=(0,1))
    
    # Plot atoms with element colors
    enames = [e.split('_')[0] for e in sys.enames]
    colors = []
    for i, ename in enumerate(enames):
        elem_data = PU.elements.ELEMENT_DICT.get(ename, PU.elements.ELEMENTS[0])
        clr = elem_data[8]
        if isinstance(clr, (list, tuple, np.ndarray)):
            if len(clr) >= 3:
                if clr[0] > 1 or clr[1] > 1 or clr[2] > 1:
                    clr = [c/255.0 for c in clr[:3]]
                colors.append(clr)
            else:
                colors.append([0.5, 0.5, 0.5])
        else:
            colors.append([0.5, 0.5, 0.5])
    sizes = [PU.elements.ELEMENT_DICT.get(ename, PU.elements.ELEMENTS[0])[6] * 80 for ename in enames]
    PU.plotAtoms(apos=sys.apos, es=sys.enames, sizes=sizes, colors=colors, marker='o', axes=(0,1), labels=None)
    
    # Plot sampling points with different colors per type
    unique_types = sorted(set(point_types))
    for ptype in unique_types:
        mask = [t == ptype for t in point_types]
        coords = point_coords[mask]
        color = POINT_COLORS.get(ptype, 'magenta')
        ax.scatter(coords[:, 0], coords[:, 1], c=color, s=30, marker='s', alpha=0.7, label=ptype, zorder=3)
    
    # Number all points
    for i, (x, y) in enumerate(point_coords):
        ax.annotate(str(i), (x, y), fontsize=6, ha='center', va='center', color='white', 
                   bbox=dict(boxstyle='circle,pad=0.1', fc='black', alpha=0.5), zorder=4)
    
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_title(f'{os.path.basename(mol_file)} with {len(point_coords)} sampling points')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)
    
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"[plot_points] Created {output_file}")

# Get all point_info files
point_files = sorted(glob.glob(os.path.join(POINTS_DIR, "*_point_info.txt")))
print(f"[plot_points] Found {len(point_files)} point_info files")

# Create temporary directory for plots
temp_plot_dir = "temp_points_plots"
os.makedirs(temp_plot_dir, exist_ok=True)

# Get all tip variant directories
tip_dirs = sorted(glob.glob(os.path.join(RELAX_TEST_DIR, "*")))
print(f"[plot_points] Found {len(tip_dirs)} tip variant directories")

for points_file in point_files:
    mol_name = os.path.basename(points_file).replace("_point_info.txt", "")
    mol_file = os.path.join(OPT_GEOM_DIR, f"{mol_name}.xyz")
    
    if not os.path.exists(mol_file):
        print(f"[plot_points] WARNING: No geometry file for {mol_name}, skipping")
        continue
    
    plot_file = os.path.join(temp_plot_dir, f"{mol_name}_points.png")
    
    # Plot points on molecule
    try:
        plot_points_on_molecule(mol_file, points_file, plot_file)
    except Exception as e:
        print(f"[plot_points] ERROR plotting {mol_name}: {e}")
        continue
    
    # Copy to all matching tip directories
    matching_dirs = [d for d in tip_dirs if os.path.basename(d).startswith(mol_name)]
    print(f"[plot_points] Found {len(matching_dirs)} directories for {mol_name}")
    
    for tip_dir in matching_dirs:
        dest_file = os.path.join(tip_dir, f"{mol_name}_points.png")
        # Check if file exists to avoid overwriting
        if os.path.exists(dest_file):
            print(f"[plot_points] Skipping {dest_file} (already exists)")
            continue
        # Check if it's GridFF.npy or other critical file
        if "gridFF" in dest_file.lower() or "lvec" in dest_file.lower():
            print(f"[plot_points] SKIPPING critical file: {dest_file}")
            continue
        shutil.copy(plot_file, dest_file)
        print(f"[plot_points] Copied to {dest_file}")

print(f"[plot_points] Done. Temporary plots in {temp_plot_dir}")
