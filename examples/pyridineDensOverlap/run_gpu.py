#!/usr/bin/env python3

"""
An example of using the full-density based model (FDBM) for doing an AFM simulation.
To run this script, download the input files first as instructed in the run.sh file
in the same folder.
"""

import ppafm.ocl.field as FFcl
import ppafm.ocl.oclUtils as oclu
from ppafm.ocl.AFMulator import AFMulator

# Initialize an OpenCL environment. You can change i_platform to select the device to use
oclu.init_env(i_platform=0)
FFcl.bRuntime = True  # Print timings

# Load all input files
rho_tip, xyzs_tip, Zs_tip = FFcl.TipDensity.from_file("tip/CHGCAR.xsf")
pot, xyzs, Zs = FFcl.HartreePotential.from_file("sample/LOCPOT.xsf", scale=-1.0)  # Scale=-1.0 for correct units of potential (V) instead of energy (eV)
rho_sample, _, _ = FFcl.ElectronDensity.from_file("sample/CHGCAR.xsf")

# Get tip delta density by subtracting the valence electrons
rho_tip_delta = rho_tip.subCores(xyzs_tip, Zs_tip, Rcore=0.7)

# Construct the simulator
afmulator = AFMulator(
    pixPerAngstrome=10,  # Force field grid density
    scan_dim=(201, 201, 50),  # Output scan dimensions (z dimension gets reduced by df_steps - 1)
    scan_window=((0, 0, 5), (20, 20, 10)),  # Physical limits of the scan
    rho=rho_tip,  # Tip charge density (used for calculating Pauli repulsion)
    rho_delta=rho_tip_delta,  # Tip delta charge density (used for calculating electrostatic force)
    A_pauli=18.0,  # Prefactor for Pauli repulsion
    B_pauli=1.0,  # Exponent in overlap integral for Pauli repulsion
    fdbm_vdw_type="D3",  # Use DFT-D3 for vdw calculation.
    d3_params="PBE",  # DFT-D3 functional-specific parameters. Default parameters for various functionals are available.
    tipR0=[0, 0, 4],  # Tip equilibrium position (x, y, R)
    tipStiffness=[0.25, 0.25, 0.0, 20.0],  # Tip spring constants (x, y, z, R)
    npbc=(1, 1, 1),  # Periodic images of atoms in (x, y, z) directions for vdW calculation
    df_steps=20,  # Oscillation amplitude in number of scan steps
)

# Run simulation and plot images
X = afmulator(xyzs, Zs, pot, rho_sample, plot_to_dir="afm_sims_ocl")
