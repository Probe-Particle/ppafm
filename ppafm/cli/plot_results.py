#!/usr/bin/python -u

import itertools as it
import os

import matplotlib as mpl
import numpy as np

from .. import PPPlot, atomicUtils, common, io
from ..HighLevel import symGauss

mpl.use("Agg")
print("plot WITHOUT Xserver")
# this makes it run without Xserver (e.g. on supercomputer) # see http://stackoverflow.com/questions/4931376/generating-matplotlib-graphs-without-a-running-x-server


atom_size = 0.15


def main(argv=None):
    # fmt: off
    parser = common.CLIParser( description="Plot results for a scan with a specified charge, amplitude, and spring constant.Images are saved in folder Q{charge}K{klat}/Amp{Amplitude}." )
    parser.add_arguments(["output_format","Amplitude","arange","klat","krange","charge", "qrange", "Vbias", "Vrange", "noPBC", ])
    parser.add_argument( "--iets",      action="store",      type=float,               help="Mass [a.u.]; Bias offset [eV]; Peak width [eV] ",   nargs=3,  )
    parser.add_argument( "--LCPD_maps", action="store_true",                           help="Print LCPD maps")
    parser.add_argument( "--z0",        action="store",      type=float,  default=0.0, help="Height of the topmost layer of metallic substrate for E to V conversion (Ang)",    )
    parser.add_argument( "--V0",        action="store",      type=float,  default=0.0, help="Empirical LCPD maxima shift due to mesoscopic workfunction diference",    )
    parser.add_argument( "--df",        action="store_true",                           help="Plot images for dfz ")
    parser.add_argument( "--save_df",   action="store_true",                           help="Save frequency shift as df.xsf "    )
    parser.add_argument( "--Laplace",   action="store_true",                           help="Plot Laplace-filtered images and save them ",    )
    parser.add_argument( "--pos",       action="store_true",                           help="Save probe particle positions"    )
    parser.add_argument( "--atoms",     action="store_true",                           help="Plot atoms to images")
    parser.add_argument( "--bonds",     action="store_true",                           help="Plot bonds to images")
    parser.add_argument( "--cbar",      action="store_true",                           help="Plot colorbars to images")
    parser.add_argument( "--WSxM",      action="store_true",                           help="Save frequency shift into WsXM *.dat files"    )
    parser.add_argument( "--bI",        action="store_true",                           help="Plot images for Boltzmann current"    )
    # fmt: on

    args = parser.parse_args(argv)
    opt_dict = vars(args)

    common.loadParams("params.ini")
    common.apply_options(opt_dict)

    if opt_dict["Laplace"]:
        from scipy.ndimage import laplace

    print(" >> OVEWRITING SETTINGS by command line arguments  ")

    # Spring constants.
    if opt_dict["krange"] is not None:
        k_constants = np.linspace(opt_dict["krange"][0], opt_dict["krange"][1], int(opt_dict["krange"][2]))
    elif opt_dict["klat"] is not None:
        k_constants = [opt_dict["klat"]]
    elif common.params["stiffness"][0] > 0.0:
        k_constants = [common.params["stiffness"][0]]
    else:
        k_constants = [common.params["klat"]]

    # Charges.
    if opt_dict["qrange"] is not None:
        charges = np.linspace(opt_dict["qrange"][0], opt_dict["qrange"][1], int(opt_dict["qrange"][2]))
    elif opt_dict["charge"] is not None:
        charges = [opt_dict["charge"]]
    else:
        charges = [common.params["charge"]]

    # Amplidudes.
    if opt_dict["arange"] is not None:
        amplitudes = np.linspace(opt_dict["arange"][0], opt_dict["arange"][1], int(opt_dict["arange"][2]))
    elif opt_dict["Amplitude"] is not None:
        amplitudes = [opt_dict["Amplitude"]]
    else:
        amplitudes = [common.params["Amplitude"]]

    # Applied biases.
    applied_bias = False
    if opt_dict["Vrange"] is not None:
        bias_voltages = np.linspace(opt_dict["Vrange"][0], opt_dict["Vrange"][1], int(opt_dict["Vrange"][2]))
    elif opt_dict["Vbias"] is not None:
        bias_voltages = [opt_dict["Vbias"]]
    else:
        bias_voltages = [0.0]
    for voltage in bias_voltages:
        if abs(voltage) > 1e-7:
            applied_bias = True

    if applied_bias:
        print("Vs   =", bias_voltages)
    print("Ks   =", k_constants)
    print("Qs   =", charges)
    print("Amps =", amplitudes)
    print(" ============= RUN  ")

    dz = common.params["scanStep"][2]
    tip_positions_x, tip_positions_y, tip_positions_z, _ = common.prepareScanGrids()
    extent = (tip_positions_x[0], tip_positions_x[-1], tip_positions_y[0], tip_positions_y[-1])

    atoms_str = ""
    atoms = None
    bonds = None
    ff_parameters = None
    if opt_dict["atoms"] or opt_dict["bonds"]:
        species_file = None
        if os.path.isfile("atomtypes.ini"):
            species_file = "atomtypes.ini"
        ff_parameters = common.loadSpecies(species_file)
        atoms_str = "_atoms"
        xyzs, z_s, qs, _ = io.loadXYZ("input_plot.xyz")
        atoms = [
            list(z_s),
            list(xyzs[:, 0]),
            list(xyzs[:, 1]),
            list(xyzs[:, 2]),
            list(qs),
        ]
        ff_parameters = common.loadSpecies()
        elem_dict = common.getFFdict(ff_parameters)
        i_zs, r_s, _ = common.parseAtoms(atoms, elem_dict, autogeom=False, PBC=common.params["PBC"])
        atom_colors = atomicUtils.getAtomColors(i_zs, FFparams=ff_parameters)
        r_s = r_s.transpose().copy()
        atoms = [i_zs, r_s[0], r_s[1], r_s[2], atom_colors]
    if opt_dict["bonds"]:
        bonds = atomicUtils.findBonds(atoms, i_zs, 1.0, FFparams=ff_parameters)

    cbar_str = ""
    if opt_dict["cbar"]:
        cbar_str = "_cbar"

    for charge, stiffness in it.product(charges, k_constants):
        dirname = f"Q{charge:1.2f}K{stiffness:1.2f}"
        for iv, voltage in enumerate(bias_voltages):
            if applied_bias:
                dirname = f"Q{charge:1.2f}K{stiffness:1.2f}V{voltage:1.2f}"
            if opt_dict["pos"]:
                pp_positions, lvec, _, atomic_info_or_head = io.load_vec_field(dirname + "/PPpos", data_format=args.output_format)
                print("Plotting PPpos: ")
                PPPlot.plotDistortions(
                    dirname + "/xy" + atoms_str + cbar_str,
                    pp_positions[:, :, :, 0],
                    pp_positions[:, :, :, 1],
                    slices=list(range(0, len(pp_positions))),
                    BG=pp_positions[:, :, :, 2],
                    extent=extent,
                    atoms=atoms,
                    bonds=bonds,
                    atomSize=atom_size,
                    markersize=2.0,
                    cbar=opt_dict["cbar"],
                )

            if opt_dict["iets"] is not None:
                eigenvalue_k, lvec, _, atomic_info_or_head = io.load_vec_field(dirname + "/eigvalKs", data_format=args.output_format)
                iets_m = opt_dict["iets"][0]
                iets_e = opt_dict["iets"][1]
                iets_w = opt_dict["iets"][2]
                print(f"Plotting IETS M={iets_m:f} V={iets_e:f} w={iets_w:f}")
                e_vib = common.HBAR * np.sqrt((common.eVA_Nm * eigenvalue_k) / (iets_m * common.AUMASS))
                iets = symGauss(e_vib[:, :, :, 0], iets_e, iets_w) + symGauss(e_vib[:, :, :, 1], iets_e, iets_w) + symGauss(e_vib[:, :, :, 2], iets_e, iets_w)
                PPPlot.plotImages(
                    dirname + "/IETS" + atoms_str + cbar_str,
                    iets,
                    slices=list(range(0, len(iets))),
                    zs=tip_positions_z,
                    extent=extent,
                    atoms=atoms,
                    bonds=bonds,
                    atomSize=atom_size,
                    cbar=opt_dict["cbar"],
                )
                PPPlot.plotImages(
                    dirname + "/Evib" + atoms_str + cbar_str,
                    e_vib[:, :, :, 0],
                    slices=list(range(0, len(iets))),
                    zs=tip_positions_z,
                    extent=extent,
                    atoms=atoms,
                    bonds=bonds,
                    atomSize=atom_size,
                    cbar=opt_dict["cbar"],
                )
                PPPlot.plotImages(
                    dirname + "/Kvib" + atoms_str + cbar_str,
                    16.0217662 * eigenvalue_k[:, :, :, 0],
                    slices=list(range(0, len(iets))),
                    zs=tip_positions_z,
                    extent=extent,
                    atoms=atoms,
                    bonds=bonds,
                    atomSize=atom_size,
                    cbar=opt_dict["cbar"],
                )

            if opt_dict["df"] or opt_dict["save_df"] or opt_dict["WSxM"] or opt_dict["LCPD_maps"]:
                for amplitude in amplitudes:
                    common.params["Amplitude"] = amplitude
                    amp_string = f"/Amp{amplitude:2.2f}"
                    print("Amplitude= ", amp_string)
                    dir_name_amplitude = dirname + amp_string
                    if not os.path.exists(dir_name_amplitude):
                        os.makedirs(dir_name_amplitude)

                    if common.params["tiltedScan"]:
                        (
                            f_out,
                            lvec,
                            _,
                            atomic_info_or_head,
                        ) = io.load_vec_field(dirname + "/OutF", data_format=args.output_format)
                        dfs = common.Fz2df_tilt(
                            f_out,
                            common.params["scanTilt"],
                            k0=common.params["kCantilever"],
                            f0=common.params["f0Cantilever"],
                            amplitude=amplitude,
                        )
                        lvec_df = np.array(lvec.copy())
                        lvec_3_norm = np.linalg.norm(lvec[3])
                        lvec_df[0] = lvec_df[0] + lvec_df[3] / lvec_3_norm * amplitude / 2
                        lvec_df[3] = lvec_df[3] / lvec_3_norm * (lvec_3_norm - amplitude)
                    else:
                        (
                            fzs,
                            lvec,
                            _,
                            atomic_info_or_head,
                        ) = io.load_scal_field(dirname + "/OutFz", data_format=args.output_format)
                        if applied_bias:
                            r_tip = common.params["Rtip"]
                            for iz, z in enumerate(tip_positions_z):
                                fzs[iz, :, :] = fzs[iz, :, :] - np.pi * common.params["permit"] * ((r_tip * r_tip) / ((z - args.z0) * (z + r_tip))) * (voltage - args.V0) * (
                                    voltage - args.V0
                                )
                        dfs = common.Fz2df(
                            fzs,
                            dz=dz,
                            k0=common.params["kCantilever"],
                            f0=common.params["f0Cantilever"],
                            amplitude=amplitude,
                        )
                        lvec_df = np.array(lvec.copy())
                        lvec_df[0][2] += amplitude / 2
                        lvec_df[3][2] -= amplitude

                    if opt_dict["save_df"]:
                        io.save_scal_field(
                            dir_name_amplitude + "/df",
                            dfs,
                            lvec_df,
                            data_format=args.output_format,
                            head=atomic_info_or_head,
                            atomic_info=atomic_info_or_head,
                        )
                    if opt_dict["df"]:
                        print(" plotting df : ")
                        PPPlot.plotImages(
                            dir_name_amplitude + "/df" + atoms_str + cbar_str,
                            dfs,
                            slices=list(range(0, len(dfs))),
                            zs=tip_positions_z + common.params["Amplitude"] / 2.0,
                            extent=extent,
                            cmap=common.params["colorscale"],
                            atoms=atoms,
                            bonds=bonds,
                            atomSize=atom_size,
                            cbar=opt_dict["cbar"],
                            cbar_label="df [Hz]",
                        )
                    if opt_dict["Laplace"]:
                        print("plotting Laplace-filtered df : ")
                        df_laplace_filtered = dfs.copy()
                        laplace(dfs, output=df_laplace_filtered)
                        io.save_scal_field(
                            dir_name_amplitude + "/df_laplace",
                            df_laplace_filtered,
                            lvec,
                            data_format=args.output_format,
                            head=atomic_info_or_head,
                            atomic_info=atomic_info_or_head,
                        )
                        PPPlot.plotImages(
                            dir_name_amplitude + "/df_laplace" + atoms_str + cbar_str,
                            df_laplace_filtered,
                            slices=list(range(0, len(dfs))),
                            zs=tip_positions_z + common.params["Amplitude"] / 2.0,
                            extent=extent,
                            cmap=common.params["colorscale"],
                            atoms=atoms,
                            bonds=bonds,
                            atomSize=atom_size,
                            cbar=opt_dict["cbar"],
                        )
                    if opt_dict["WSxM"]:
                        print(" printing df into WSxM files :")
                        io.saveWSxM_3D(dir_name_amplitude + "/df", dfs, extent, slices=None)
                    if opt_dict["LCPD_maps"]:
                        if iv == 0:
                            lcpd_b = -dfs
                        if iv == (bias_voltages.shape[0] - 1):
                            lcpd_b = (lcpd_b + dfs) / (2 * voltage)

                        if iv == 0:
                            lcpd_a = dfs
                        if voltage == 0.0:
                            lcpd_a = lcpd_a - 2 * dfs
                        if iv == (bias_voltages.shape[0] - 1):
                            lcpd_a = (lcpd_a + dfs) / (2 * voltage**2)

            if opt_dict["bI"]:
                current, lvec, _, atomic_info_or_head = io.load_scal_field(dirname + "/OutI_boltzmann", data_format=args.output_format)
                print(" plotting Boltzmann current: ")
                PPPlot.plotImages(
                    dirname + "/OutI" + atoms_str + cbar_str,
                    current,
                    slices=list(range(0, len(current))),
                    zs=tip_positions_z,
                    extent=extent,
                    atoms=atoms,
                    bonds=bonds,
                    atomSize=atom_size,
                    cbar=opt_dict["cbar"],
                )

        if opt_dict["LCPD_maps"]:
            lcpd = -lcpd_b / (2 * lcpd_a)
            PPPlot.plotImages(
                "./LCPD" + atoms_str + cbar_str,
                lcpd,
                slices=list(range(0, len(lcpd))),
                zs=tip_positions_z + common.params["Amplitude"] / 2.0,
                extent=extent,
                cmap=common.params["colorscale_kpfm"],
                atoms=atoms,
                bonds=bonds,
                atomSize=atom_size,
                cbar=opt_dict["cbar"],
                symmetric_map=True,
                V0=args.V0,
                cbar_label="V_LCPD [V]",
            )
            PPPlot.plotImages(
                "./_Asym-LCPD" + atoms_str + cbar_str,
                lcpd,
                slices=list(range(0, len(lcpd))),
                zs=tip_positions_z + common.params["Amplitude"] / 2.0,
                extent=extent,
                cmap=common.params["colorscale_kpfm"],
                atoms=atoms,
                bonds=bonds,
                atomSize=atom_size,
                cbar=opt_dict["cbar"],
                symmetric_map=False,
                cbar_label="V_LCPD [V]",
            )
            io.save_scal_field(
                "./LCPD",
                lcpd,
                lvec_df,
                data_format=args.output_format,
                head=atomic_info_or_head,
                atomic_info=atomic_info_or_head,
            )
            if opt_dict["WSxM"]:
                print("Saving LCPD_b into WSxM files :")
                io.saveWSxM_3D("./LCPD" + atoms_str, lcpd, extent, slices=None)

    print(" ***** ALL DONE ***** ")


if __name__ == "__main__":
    main()
