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
                print()

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
                print()

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
                print()

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
                print()

            if opt_dict["bI"]:
                current, lvec, _, atomic_info_or_head = io.load_scal_field(dirname + "/OutI_boltzmann", data_format=args.output_format)
                print("Plotting Boltzmann current: ")
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
                print()

        if opt_dict["LCPD_maps"]:
            if len(bias_voltages) < 3:
                print("At last three different values of volage needed to evaluate LCPD!")
                print("LCPD will not be calculated here.")
                opt_dict["LCPD_maps"] = False
            else:
                # Prepare to calculate KPFM/LCPD
                # For each pixel, find a,b,c such that df = aV^2 + bV + c
                # This is done as polynomial (2nd order) regression, the above equality need not be exact
                # but the least-square criterion will be used.
                # The coefficients a,b,c are going to be determined as linar combinations of df(V) at different biases:
                #   a = Sum_i w_KPFM_a (V_i)
                #   b = Sum_i w_KPFM_b (V_i)
                #   c = Sum_i w_KPFM_c (V_i)
                # Now, the coefficients (weights) w_KPFM have to be determined.
                # This will be done with the help of Gram-sSchmidt ortogonalization:
                # Create vectors v0, v1, v2,
                #   v0 = [1]_(i=1..N),
                #   v1 = [V_i]_(i=1..N,)
                #   v2 = [V^2_i]_(i=1..N),
                # orthogonalize them, find an expansion of [df(V_i)] into the orthogonalized v0', v1', v2',
                # then into the original v0, v1, v2 as defined above.
                # The following notation is going to be used
                # in order to desribe the relation between the original and the orthogonal vectors:
                #   v0' = v0
                #   v1' = v1 + p10 v0
                #   v2' = v2 + p21 v1 + p20 v0
                # The coefficients p10, p21, and p20 can be found as follows:
                #   p10 = -(v1.v0) / N (where N = |v0|^2),
                #   p21 = -(v2.v1') / |v1'|^2,
                #   p20 = -(v2.v0) / N + p21 p10.
                # We have the following expansion of df:
                #   df = (df.v0')/nv0 v0' + (df.v1')/nv1 v1' + (df.v2')/nv2 v2'
                #      = (df.v0')/nv0 v0 + (df.v1')/nv1 (v1 + p10 v0) + (df.v2')/nv2 (v2 + p21 v1 + p20 v0),
                # which gives, for vectors of the desired coefficients w_KPFM
                #   w_KPFM_a = v2'/nv2
                #   w_KPFM_b = v1'/nv1 + p21/nv2 v2'
                #   w_KPFM_c = v0'/nv0 + p10/nv1 v1' + p20/nv2 v2'
                # where nv0 = N = |v0'|^2, nv1 = |v1'|^2, nv2 = |v2'|^2.
                nv0 = len(bias_voltages)
                v0 = np.ones(nv0)
                v1 = np.copy(bias_voltages)
                v2 = v1 * v1
                p10 = -np.dot(v1, v0) / nv0
                v1 += p10 * v0
                nv1 = np.dot(v1, v1)
                p20 = -np.dot(v2, v0) / nv0
                v2 += p20 * v0
                p21 = -np.dot(v2, v1) / nv1
                v2 += p21 * v1
                p20 += p21 * p10
                nv2 = np.dot(v2, v2)
                w_kpfm_a = v2 / nv2
                w_kpfm_b = v1 / nv1 + v2 * p21 / nv2
                w_kpfm_c = v0 / nv0 + v1 * p10 / nv1 + v2 * p20 / nv2

        if opt_dict["df"] or opt_dict["save_df"] or opt_dict["WSxM"] or opt_dict["LCPD_maps"]:
            for amplitude in amplitudes:
                for iv, voltage in enumerate(bias_voltages):
                    common.params["Amplitude"] = amplitude
                    amp_string = f"/Amp{amplitude:2.2f}"
                    print("Amplitude= ", amp_string)
                    dirname0 = f"Q{charge:1.2f}K{stiffness:1.2f}"
                    if applied_bias:
                        dirname = dirname0 + f"V{voltage:1.2f}"
                    else:
                        dirname = dirname0
                    dir_name_amplitude = dirname + amp_string
                    dir_name_lcpd = dirname0 + amp_string
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
                        print("Plotting df: ")
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
                        print("")

                    if opt_dict["Laplace"]:
                        print("Plotting Laplace-filtered df: ")
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
                        print()

                    if opt_dict["WSxM"]:
                        print(" printing df into WSxM files :")
                        io.saveWSxM_3D(dir_name_amplitude + "/df", dfs, extent, slices=None)

                    if opt_dict["LCPD_maps"]:
                        if iv == 0:
                            kpfm_a = w_kpfm_a[0] * dfs
                            kpfm_b = w_kpfm_b[0] * dfs
                            kpfm_c = w_kpfm_c[0] * dfs
                        else:
                            kpfm_a += w_kpfm_a[iv] * dfs
                            kpfm_b += w_kpfm_b[iv] * dfs
                            kpfm_c += w_kpfm_c[iv] * dfs

                if opt_dict["LCPD_maps"]:
                    lcpd = -kpfm_b / (2 * kpfm_a)

                    print("Plotting LCPD: ")
                    if not os.path.exists(dir_name_lcpd):
                        os.makedirs(dir_name_lcpd)
                    PPPlot.plotImages(
                        dir_name_lcpd + "/LCPD" + atoms_str + cbar_str,
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
                    print()

                    PPPlot.plotImages(
                        dir_name_lcpd + "/_Asym-LCPD" + atoms_str + cbar_str,
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
                    print()

                    io.save_scal_field(
                        dir_name_lcpd + "/LCPD",
                        lcpd,
                        lvec_df,
                        data_format=args.output_format,
                        head=atomic_info_or_head,
                        atomic_info=atomic_info_or_head,
                    )

                    if opt_dict["WSxM"]:
                        print("Saving LCPD into WSxM files :")
                        io.saveWSxM_3D(dir_name_lcpd + "/LCPD" + atoms_str, lcpd, extent, slices=None)

    print(" ***** ALL DONE ***** ")


if __name__ == "__main__":
    main()
