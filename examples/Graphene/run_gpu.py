#!/usr/bin/env python3

from ppafm.io import loadXYZ
from ppafm.ocl.AFMulator import AFMulator

# Load sample coordinates (xyzs), atomic numbers (Zs), and charges (qs)
xyzs, Zs, qs, _ = loadXYZ('./Gr6x6N3hole.xyz')

# Create simulator and load parameters
afmulator = AFMulator.from_params('./params.ini')

# This is bit of a hack: the afmulator does not fully support non-rectangular cells so
# we first grab the sample lvec we read from params.ini and then reset the afmulator
# force field lvec to rectangular.
pbc_lvec = afmulator.lvec
afmulator.setLvec()

# Run simulation and plot
afmulator(xyzs, Zs, qs, pbc_lvec=pbc_lvec, plot_to_dir='./afm_ocl')
