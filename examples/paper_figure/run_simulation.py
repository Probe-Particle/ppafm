#!/usr/bin/env python3

"""
Example that reproduces the main simulations figure in the paper: [Niko Oinonen, Aliaksandr V. Yakutovich, Aurelio Gallardo, Martin Ondracek, Prokop Hapala, Ondrej Krejci, Advancing Scanning Probe Microscopy Simulations: A Decade of Development in Probe-Particle Models, Comput. Phys. Commun. **issue to be updated**, 109341 - Available online 10 August 2024](https://doi.org/10.1016/j.cpc.2024.109341).

The simulation is run for 6 different molecules:
    - C60 fullerene
    - Formic acid dimer (FAD)
    - 4-(4-(2,3,4,5,6-pentafluorophenylethynyl)-2,3,5,6-tetrafluorophenylethynyl) phenylethynylbenzene (FFPB),
    - Pentacene
    - Phtalocyanine
    - Perylene carboxylic anhydride (PTCDA)

Every simulation is run using three different force field models: Lennard-Jones + point-charge electrostatics, Lennard-Jones + hartree
potential electrostatics, and the full-density based model. Additionally, comparison data from DFT AFM simulation is plotted.

The used dataset is available at https://doi.org/10.5281/zenodo.10418629. It is downloaded automatically when running this script.
(~660MB download -> ~5GB uncompressed on disk).

Note: This script uses the OpenCL version of PPAFM. Additionally, in order to render the molecule geometries, povray needs to be
installed on the system (optional).
"""

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import ppafm.ocl.field as FFcl
import ppafm.ocl.oclUtils as oclu
from ppafm.common import Fz2df
from ppafm.data import download_dataset
from ppafm.io import DEFAULT_POV_HEAD_NO_CAM, loadXYZ, makePovCam, writePov
from ppafm.ocl.AFMulator import AFMulator

MM_TO_INCH = 1 / 25.4
POVRAY_AVAILABLE = shutil.which("povray") is not None

# Set matplotlib font rendering to use LaTex
# plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.serif": ["Computer Modern Roman"]})


def get_dft_afm(data_dir, sample_name, amp=0.2):
    # Load data
    data = np.load(data_dir / f"{sample_name}.npz")
    force = data["force"]
    scan_window = data["scan_window"]

    # Convert force in z-direction into df
    force_z = force[..., 2]
    dz = (scan_window[1, 2] - scan_window[0, 2]) / force.shape[2]
    df = Fz2df(force_z.transpose(2, 1, 0), dz=dz, amplitude=amp)
    df = df.transpose(2, 1, 0)  # Transpose because Fz2df assumes zyx index order, but later we use xyz

    # The image is not complete because only one part of the symmetric structure is calculated.
    # Here we complete the image by taking the symmetry into account
    df, scan_window = complete_dft_afm(sample_name, df, scan_window)

    return df, scan_window


def complete_dft_afm(sample_name, dft_afm, scan_window):
    if sample_name == "FAD":
        mirror = np.flip(dft_afm, axis=(0, 1))
        dft_afm = np.concatenate([dft_afm, mirror], axis=0)
        scan_window[1, 0] += scan_window[1, 0] - scan_window[0, 0]
    elif sample_name == "FFPB":
        mirror = np.flip(dft_afm, axis=1)
        dft_afm = np.concatenate([dft_afm, mirror], axis=1)
        scan_window[1, 1] += scan_window[1, 1] - scan_window[0, 1]
    elif sample_name in ["Pentacene", "Phtalocyanine", "PTCDA"]:
        mirror = np.flip(dft_afm, axis=0)
        dft_afm = np.concatenate([dft_afm, mirror], axis=0)
        mirror = np.flip(dft_afm, axis=1)
        dft_afm = np.concatenate([dft_afm, mirror], axis=1)
        scan_window[1, 0] += scan_window[1, 0] - scan_window[0, 0]
        scan_window[1, 1] += scan_window[1, 1] - scan_window[0, 1]
    return dft_afm, scan_window


def get_sims(sample_dir, co_dir, scan_window, xy_shape, amp=0.2, A_pauli=18.0, B_pauli=1.0):
    # Load geometry with point-charge electrostatics
    xyzs_pt, Zs_pt, qs, _ = loadXYZ(sample_dir / "mol.xyz")

    # Load sample Hartree potential and electron density
    pot, xyzs, Zs = FFcl.HartreePotential.from_file(sample_dir / "LOCPOT.xsf", scale=-1)
    rho_sample, _, _ = FFcl.ElectronDensity.from_file(sample_dir / "CHGCAR.xsf")

    # Load tip densities: total density and delta density
    rho_tip, _, _ = FFcl.TipDensity.from_file(co_dir / "density_CO.xsf")
    rho_tip_delta, _, _ = FFcl.TipDensity.from_file(co_dir / "CO_delta_density_aims.xsf")

    # Scan dimension
    d = 0.1
    df_steps = int(amp / d)
    z_dim = int((scan_window[1, 2] - scan_window[0, 2]) / d)
    scan_dim = (xy_shape[0], xy_shape[1], z_dim)

    # Define simulator
    afmulator = AFMulator(
        pixPerAngstrome=14,
        scan_window=scan_window,
        scan_dim=scan_dim,
        Qs=[-10, 20, -10, 0],
        QZs=[0.07, 0.0, -0.07, 0],
        rho={"dz2": -0.1},
        fdbm_vdw_type="D3",
        d3_params="PBE",
        tipR0=[0, 0, 3],
        tipStiffness=[0.25, 0.25, 0.0, 30.0],
        df_steps=df_steps,
    )

    print(afmulator.dz, d)
    assert np.allclose(afmulator.dz, d)

    # Run simulation with point-charge electrostatics
    X_pt = afmulator(xyzs_pt, Zs_pt, qs, sample_lvec=pot.lvec[1:])

    # Run simulation with Hartree potential electrostatics
    X_hartree = afmulator(xyzs, Zs, pot)

    # Run simulation with full-density based model
    afmulator.setRho(rho_tip, B_pauli=B_pauli)
    afmulator.setRhoDelta(rho_tip_delta)
    afmulator.A_pauli = A_pauli
    X_fdbm = afmulator(xyzs, Zs, pot, rho_sample)

    # Render molecule geometry
    if POVRAY_AVAILABLE:
        mol_img = get_mol_img(xyzs_pt, Zs_pt, scan_window)
    else:
        print("POV-RAY not available. Skipping rendering geometry.")
        mol_img = np.ones((xy_shape[1], xy_shape[0], 3))

    X_pt = X_pt[:, :, :-1]
    X_hartree = X_hartree[:, :, :-1]
    X_fdbm = X_fdbm[:, :, :-1]

    return (X_pt, X_hartree, X_fdbm), mol_img


def get_mol_img(xyz, Zs, scan_window, res=100):
    x_size = scan_window[1][0] - scan_window[0][0]
    y_size = scan_window[1][1] - scan_window[0][1]
    x_res = x_size * res
    y_res = y_size * res
    pos = [(scan_window[1][0] + scan_window[0][0]) / 2, (scan_window[1][1] + scan_window[0][1]) / 2, 0]
    dims = [x_size / np.sqrt(3), y_size / np.sqrt(3)]

    xyz = xyz[:, :3]
    xyz -= (xyz.max(axis=0) + xyz.min(axis=0)) / 2

    temp_name = "mol_temp"
    povray_fname = f"{temp_name}.pov"
    img_fname = f"{temp_name}.png"

    cam = makePovCam(pos, fw=[0, 0, -100], lpos=[0, 0, 100], W=dims[0], H=dims[1])
    cam += DEFAULT_POV_HEAD_NO_CAM
    writePov(povray_fname, xyz, Zs, HEAD=cam, spherescale=0.5)
    os.system(f"povray Width={x_res} Height={y_res} Antialias=On Antialias_Threshold=0.3 Display=off Output_Alpha=true {povray_fname}")

    mol_img = np.array(Image.open(img_fname))

    os.remove(povray_fname)
    os.remove(img_fname)

    return mol_img


def init_fig(img_shapes, n_cols=4, ax_width=30, left_pad=0.2, right_pad=0.2, bottom_pad=0.2, top_pad=0.2, h_pad=0.5, v_pad=0.5):
    ax_width *= MM_TO_INCH
    left_pad *= MM_TO_INCH
    right_pad *= MM_TO_INCH
    bottom_pad *= MM_TO_INCH
    top_pad *= MM_TO_INCH
    h_pad *= MM_TO_INCH
    v_pad *= MM_TO_INCH

    n_rows = len(img_shapes)
    ax_heights = [s[1] / s[0] * ax_width for s in img_shapes]

    width = n_cols * ax_width + left_pad + (n_cols - 1) * h_pad + right_pad
    height = sum(ax_heights) + top_pad + (n_rows - 1) * v_pad + bottom_pad
    fig = plt.figure(figsize=(width, height))

    print("Image size:", width / MM_TO_INCH, height / MM_TO_INCH)

    axes = []
    y = height - top_pad
    for ax_height in ax_heights:
        y -= ax_height
        x = left_pad
        axes_ = []
        for _ in range(n_cols):
            rect = [x / width, y / height, ax_width / width, ax_height / height]
            ax = fig.add_axes(rect)
            for axis in ["top", "bottom", "left", "right"]:
                ax.spines[axis].set_linewidth(0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            axes_.append(ax)
            x += ax_width + h_pad
        y -= v_pad
        axes.append(axes_)

    axes = np.array(axes)

    return fig, axes


if __name__ == "__main__":
    # Choose OpenCL device to run on
    # (List available devices by running "ppafm-gui -l" on the command line)
    oclu.init_env(i_platform=0)

    # Paths to data folders
    dft_afm_dir = Path("dft-afm")
    hartree_density_dir = Path("hartree-density")
    co_density_dir = Path("CO-densities")

    # Oscillation amplitude
    amplitude = 0.5

    # Full-density based model parameters
    A_pauli = 12.0
    B_pauli = 1.20

    # Additional offset in z-direction to get a matching contrast between DFT-AFM and PP-AFM
    z_offset = -0.2

    # Choose z-slices to plot
    z_inds = {
        "C60": 1,
        "FAD": 0,
        "FFPB": 1,
        "Pentacene": 2,
        "Phtalocyanine": 1,
        "PTCDA": 1,
    }

    # Download data
    download_dataset("dft-afm", dft_afm_dir)
    download_dataset("hartree-density", hartree_density_dir)
    download_dataset("CO-tip-densities", co_density_dir)

    # Run simulations for each molecule
    data = []
    for i_s, (sample_name, z_ind) in enumerate(z_inds.items()):
        print(f"Sample: {sample_name}")

        # Get DFT-AFM data
        afm_dft, scan_window = get_dft_afm(dft_afm_dir, sample_name, amp=amplitude)

        # Run PP-AFM simulations
        scan_window[:, 2] += z_offset
        xy_shape = (3 * afm_dft.shape[0], 3 * afm_dft.shape[1])  # Use 3x resolution in PP-AFM compared to DFT
        sims, mol_img = get_sims(hartree_density_dir / sample_name, co_density_dir, scan_window, xy_shape=xy_shape, amp=amplitude, A_pauli=A_pauli, B_pauli=B_pauli)

        data.append((sample_name, z_ind, afm_dft, sims, mol_img))

    # Initialize figure
    img_shapes = [afm_dft.shape[:2] for _, _, afm_dft, _, _ in data]
    fig, axes = init_fig(img_shapes, n_cols=5, top_pad=4, left_pad=4, v_pad=2, right_pad=0.5, bottom_pad=0.5)

    # Plot the data, looping over the molecules
    for i_s, (sample_name, z_ind, afm_dft, sims, mol_img) in enumerate(data):
        # Molecule geometry
        axes[i_s, 0].imshow(mol_img)
        axes[i_s, 0].set_axis_off()

        # PP-AFM
        for j, afm in enumerate(sims):
            afm = afm[:, :, ::-1]
            axes[i_s, j + 1].imshow(afm[:, :, z_ind].T, origin="lower", cmap="gray")

        # DFT-AFM
        axes[i_s, -1].imshow(afm_dft[:, :, z_ind].T, origin="lower", cmap="gray")

        # Left side label
        axes[i_s, 0].text(-0.07, 0.5, sample_name, ha="center", va="center", transform=axes[i_s, 0].transAxes, rotation="vertical", fontsize=8)

    # Set titles at the top
    for ax, title in zip(axes[0], ["Geometry", "LJ + PC", "LJ + Hartree", "FDBM", "DFT"]):
        ax.set_title(title, fontsize=8, y=1.05, pad=0)

    # Save figure
    fig.savefig("sims_comparison.png", dpi=300)
