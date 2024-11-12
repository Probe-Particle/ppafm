#!/usr/bin/env python3

"""
An example of using the full-density based model (FDBM) with an sample electron density obtained with FHI-aims.
The elecron density in aims is all-electron, so the density at the nuclei positions can be extremely high.
This sometimes causes artifacts to appear in the resuting simulation images. This can be mitigated by cutting
out the high electron density values, which is demonstrated in this example.

Simulations are printed both with and without the cutoff into two separate directories, `sims_cutoff100` and `sims_no_cutoff`.
Compare the simulation image and observe the unphysical artifacts around the bromine atom at the top-right without the cutoff
and how the artifacts disappear with the cutoff.
"""

from pathlib import Path

import ppafm.ocl.field as FFcl
import ppafm.ocl.oclUtils as oclu
from ppafm.data import download_dataset
from ppafm.ocl.AFMulator import AFMulator

if __name__ == "__main__":

    # Initialize an OpenCL environment. You can change i_platform to select the device to use
    oclu.init_env(i_platform=0)

    # Download input files (if not already downloaded)
    tip_dir = Path("tip")
    sample_dir = Path("sample")
    download_dataset("CO-tip-densities", target_dir=tip_dir)
    download_dataset("BrClPyridine-hartree-density", target_dir=sample_dir)

    # Load all input files
    rho_tip, xyzs_tip, Zs_tip = FFcl.TipDensity.from_file(tip_dir / "density_CO.xsf")
    rho_tip_delta, _, _ = FFcl.TipDensity.from_file(tip_dir / "CO_delta_density_aims.xsf")
    pot, xyzs, Zs = FFcl.HartreePotential.from_file(sample_dir / "hartree.xsf", scale=-1.0)  # Scale=-1.0 for correct units of potential (V) instead of energy (eV)
    rho_sample, _, _ = FFcl.ElectronDensity.from_file(sample_dir / "density.xsf")

    # Construct the simulator
    afmulator = AFMulator(
        pixPerAngstrome=10,
        scan_dim=(160, 160, 40),
        scan_window=((2, 2, 6), (18, 18, 10)),
        rho=rho_tip,
        rho_delta=rho_tip_delta,
        A_pauli=25.0,  # Arbitrarily chosen values, not fitted to anything.
        B_pauli=1.1,  # Higher values for the A and B parameters result in more severe artifacts.
        fdbm_vdw_type="D3",
        d3_params="PBE",
        tipR0=[0, 0, 3],
        tipStiffness=[0.25, 0.25, 0.0, 20.0],
        npbc=(1, 1, 0),
        df_steps=10,
    )

    # Let's run the simulation both with and without the cutoff
    for cutoff in [None, 100]:

        # Apply cutoff to sample electron density
        rho_sample_cutoff = rho_sample.clamp(maximum=cutoff, in_place=False)
        plot_dir = f"sims_cutoff{cutoff}" if cutoff else "sims_no_cutoff"

        # Run simulation and plot images
        X = afmulator(xyzs, Zs, pot, rho_sample_cutoff, plot_to_dir=plot_dir)
