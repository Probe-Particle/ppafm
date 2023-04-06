#!/usr/bin/env python3

from ppafm.io import loadXYZ
from ppafm.ocl.AFMulator import AFMulator

# Load sample coordinates (xyzs), atomic numbers (Zs), and charges (qs)
xyzs, Zs, qs, _ = loadXYZ('./Gr6x6N3hole.xyz')

# Create simulator and load parameters
afmulator = AFMulator.from_params('./params.ini')

# Run simulation and plot
afmulator(xyzs, Zs, qs, plot_to_dir='./afm_ocl')
