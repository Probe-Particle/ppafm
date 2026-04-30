#!/usr/bin/env python
"""Plot molecular geometry with bonds and colored atoms.

Usage:
    python plot_molecule.py --input molecule.xyz --output molecule.png
    python plot_molecule.py --input molecule.mol2 --axes 0 2  # XY vs XZ view
"""
import os, sys, argparse, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pyProbeParticle.AtomicSystem import AtomicSystem
from pyProbeParticle import plotUtils as PU
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Plot molecular geometry with bonds and colored atoms')
parser.add_argument('--input', type=str, required=True, help='Input molecular structure file (xyz, mol, mol2)')
parser.add_argument('--output', type=str, default=None, help='Output image file (default: show interactively)')
parser.add_argument('--axes', type=str, default='0,1', help='Axes to plot (comma-separated, e.g., "0,1" for XY, "0,2" for XZ, "1,2" for YZ)')
parser.add_argument('--size', type=float, default=50.0, help='Atom size scaling factor')
parser.add_argument('--bonds', type=int, default=1, help='Plot bonds (0=no, 1=yes)')
parser.add_argument('--labels', type=int, default=1, help='Show atom labels (0=no, 1=yes)')
parser.add_argument('--RvdwCut', type=float, default=0.5, help='vdW radius cutoff for bond detection')
parser.add_argument('--extent', type=str, default=None, help='Plot extent as "xmin,xmax,ymin,ymax"')
parser.add_argument('--figsize', type=str, default='8,8', help='Figure size as "width,height"')
args = parser.parse_args()

# Parse axes
axes = tuple(map(int, args.axes.split(',')))
if len(axes) != 2 or any(ax not in (0,1,2) for ax in axes):
    raise ValueError('--axes must be two comma-separated integers from {0,1,2}')

# Parse extent
extent = None
if args.extent:
    extent = tuple(map(float, args.extent.split(',')))
    if len(extent) != 4:
        raise ValueError('--extent must be four comma-separated floats: xmin,xmax,ymin,ymax')

# Parse figure size
figsize = tuple(map(float, args.figsize.split(',')))

# Load molecular structure
print(f"[plot_molecule] Loading structure from: {args.input}")
sys = AtomicSystem(fname=args.input)
print(f"[plot_molecule] Loaded {sys.natoms} atoms")

# Find bonds if needed and not already present
if args.bonds and sys.bonds is None:
    print("[plot_molecule] Finding bonds...")
    sys.findBonds(Rcut=3.0, RvdwCut=args.RvdwCut)
    print(f"[plot_molecule] Found {len(sys.bonds)} bonds")

# Get element colors and convert from 0-255 to 0-1 range
enames = [e.split('_')[0] for e in sys.enames]
colors_raw = [sys.atypes[i] if sys.atypes is not None else 1 for i in range(len(sys.enames))]
colors = []
for i, ename in enumerate(enames):
    elem_data = PU.elements.ELEMENT_DICT.get(ename, PU.elements.ELEMENTS[0])
    clr = elem_data[8]  # RGB color
    if isinstance(clr, (list, tuple, np.ndarray)):
        if len(clr) >= 3:
            # Convert from 0-255 to 0-1 if needed
            if clr[0] > 1 or clr[1] > 1 or clr[2] > 1:
                clr = [c/255.0 for c in clr[:3]]
            colors.append(clr)
        else:
            colors.append([0.5, 0.5, 0.5])
    else:
        colors.append([0.5, 0.5, 0.5])

sizes = [PU.elements.ELEMENT_DICT.get(ename, PU.elements.ELEMENTS[0])[6] * args.size for ename in enames]

# Create figure
fig, ax = plt.subplots(figsize=figsize)

# Plot bonds
if args.bonds and sys.bonds is not None:
    PU.plotBonds(links=sys.bonds, ps=sys.apos, axes=axes)

# Plot atoms
PU.plotAtoms(apos=sys.apos, es=sys.enames, sizes=sizes, colors=colors, marker='o', axes=axes, labels=[f"{e}{i}" for i,e in enumerate(sys.enames)] if args.labels else None)

# Set axis labels based on axes
axis_names = ['X', 'Y', 'Z']
ax.set_xlabel(f'{axis_names[axes[0]]} (Å)')
ax.set_ylabel(f'{axis_names[axes[1]]} (Å)')
ax.set_title(f'{os.path.basename(args.input)} - {sys.natoms} atoms')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# Save or show
if args.output:
    plt.savefig(args.output, bbox_inches='tight', dpi=150)
    print(f"[plot_molecule] Saved to: {args.output}")
    plt.close(fig)
else:
    plt.show()
