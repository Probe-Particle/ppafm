#!/usr/bin/env python3

from ppafm.data import download_dataset
from ppafm.ocl.AFMulator import AFMulator
from ppafm.ocl.field import HartreePotential

download_dataset("PTCDA-Ag", "sample")

afmulator = AFMulator(
    scan_dim=(200, 200, 60),
    scan_window=((0.0, 0.0, 16.0), (19.875, 19.875, 22.0)),
    iZPP=8,
    df_steps=10,  # Amplitude = 10 * (22.0Å - 16.0Å) / 60 = 1.0Å
    tipStiffness=(0.37, 0.37, 0.0, 20.0),
    rho={"dz2": -0.05},
    tipR0=[0.0, 0.0, 4.0],
    npbc=(1, 1, 0),
)

print("Loading potential")
pot, xyzs, Zs = HartreePotential.from_file("./sample/LOCPOT.xsf", scale=-1.0)  # scale=-1.0 for correct units of potential (V) instead of energy (eV)

print("Running simulation")
X = afmulator(xyzs, Zs, pot, plot_to_dir="./PTCDA_CO_dz2-0.05_K0.37_A1.0")
