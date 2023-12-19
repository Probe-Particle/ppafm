#!/usr/bin/python -u

import itertools as it
import os

import numpy as np

from .. import common, core, io
from ..GridUtils import interpolate_cartesian
from ..HighLevel import perform_relaxation


def rotate_vector(v, a):
    ca = np.cos(a)
    sa = np.sin(a)
    x = ca * v[0] - sa * v[1]
    y = sa * v[0] + ca * v[1]
    v[0] = x
    v[1] = y
    return v


def rotate_ff(fx, fy, a):
    ca = np.cos(a)
    sa = np.sin(a)
    new_fx = ca * fx - sa * fy
    new_fy = sa * fx + ca * fy
    return new_fx, new_fy


def main(argv=None):
    parser = common.CLIParser(
        description="Perform a scan, relaxing the probe particle in a precalculated force field. The generated force field is saved to Q{charge}K{klat}/OutFz.xsf."
    )
    # fmt: off
    parser.add_arguments(['klat', 'krange', 'charge', 'qrange', 'Vbias', 'Vrange', 'Apauli', 'output_format'])
    parser.add_argument("--noLJ",           action="store_true",                          help="Load Pauli and vdW force fields from separate files")
    parser.add_argument("-b","--boltzmann", action="store_true",                          help="Calculate forces with boltzmann particle")
    parser.add_argument("--bI",             action="store_true",                          help="Calculate current between boltzmann particle and tip")
    parser.add_argument("--pos",            action="store_true",                          help="Save probe particle positions")
    parser.add_argument("--disp",           action="store_true",                          help="Save probe particle displacements")
    parser.add_argument("--bDebugFFtot",    action="store_true",                          help="Store total force field for debugging")
    parser.add_argument("--vib",            action="store",      type=int,   default=-1,  help="Map PP vibration eigenmodes; 0-just eigenvals; 1-3 eigenvecs")
    parser.add_argument("--tipspline",      action="store",      type=str,                help="File where spline is stored")
    parser.add_argument("--rotate",         action="store",      type=float, default=0.0, help="Rotates sampling in xy-plane")
    parser.add_argument("--pol_t",          action="store",      type=float, default=1.0, help="Scaling factor for tip polarization")
    parser.add_argument("--pol_s",          action="store",      type=float, default=1.0, help="Scaling factor for sample polarization")
    # fmt: on
    args = parser.parse_args(argv)

    opt_dict = vars(args)
    common.loadParams("params.ini")
    common.apply_options(opt_dict)

    # =============== Setup

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
    charged_system = False
    if opt_dict["qrange"] is not None:
        charges = np.linspace(opt_dict["qrange"][0], opt_dict["qrange"][1], int(opt_dict["qrange"][2]))
    elif opt_dict["charge"] is not None:
        charges = [opt_dict["charge"]]
    else:
        charges = [common.params["charge"]]
    for charge in charges:
        if abs(charge) > 1e-7:
            charged_system = True

    # Vkpfm
    applied_bias = False
    if opt_dict["Vrange"] is not None:
        voltages = np.linspace(opt_dict["Vrange"][0], opt_dict["Vrange"][1], int(opt_dict["Vrange"][2]))
    elif opt_dict["Vbias"] is not None:
        voltages = [opt_dict["Vbias"]]
    else:
        voltages = [0.0]
    for voltage in voltages:
        if abs(voltage) > 1e-7:
            applied_bias = True

    if applied_bias:
        print("Vs   =", voltages)
    print("Ks   =", k_constants)
    print("Qs   =", charges)

    print(" ============= RUN  ")

    ff_vdw = ff_pauli = ff_electrostatics = ff_boltzman = ff_kpfm_t0sv = ff_kpfm_tvs0 = None

    if args.noLJ:
        print("Apauli", common.params["Apauli"])

        print("Loading Pauli force field from FFpauli_{x,y,z}")
        ff_pauli, lvec, _, atomic_info_or_head = io.load_vec_field("FFpauli", data_format=args.output_format)
        ff_pauli[0, :, :, :], ff_pauli[1, :, :, :] = rotate_ff(ff_pauli[0, :, :, :], ff_pauli[1, :, :, :], opt_dict["rotate"])

        print("Loading vdW force field from FFvdW_{x,y,z}")
        ff_vdw, lvec, _, atomic_info_or_head = io.load_vec_field("FFvdW", data_format=args.output_format)
        ff_vdw[0, :, :, :], ff_vdw[1, :, :, :] = rotate_ff(ff_vdw[0, :, :, :], ff_vdw[1, :, :, :], opt_dict["rotate"])

    else:
        print("Loading Lennard-Jones force field from FFLJ_{x,y,z}")
        ff_vdw, lvec, _, atomic_info_or_head = io.load_vec_field("FFLJ", data_format=args.output_format)
        ff_vdw[0, :, :, :], ff_vdw[1, :, :, :] = rotate_ff(ff_vdw[0, :, :, :], ff_vdw[1, :, :, :], opt_dict["rotate"])

    if charged_system:
        print("Loading electrostatic force field from FFel_{x,y,z}")
        ff_electrostatics, lvec, _, atomic_info_or_head = io.load_vec_field("FFel", data_format=args.output_format)
        ff_electrostatics[0, :, :, :], ff_electrostatics[1, :, :, :] = rotate_ff(ff_electrostatics[0, :, :, :], ff_electrostatics[1, :, :, :], opt_dict["rotate"])

    if args.boltzmann or args.bI:
        print("Loading Boltzmann force field from FFboltz_{x,y,z}")
        ff_boltzman, lvec, _, atomic_info_or_head = io.load_vec_field("FFboltz", data_format=args.output_format)
        ff_boltzman[0, :, :, :], ff_boltzman[1, :, :, :] = rotate_ff(ff_boltzman[0, :, :, :], ff_boltzman[1, :, :, :], opt_dict["rotate"])

    if applied_bias:
        print("Loading electrostatic contribution from applied bias from FFkpfm_t0sV_{x,y,z} and FFkpfm_tVs0_{x,y,z}")
        ff_kpfm_t0sv, lvec, _, atomic_info_or_head = io.load_vec_field("FFkpfm_t0sV", data_format=args.output_format)
        ff_kpfm_tvs0, lvec, _, atomic_info_or_head = io.load_vec_field("FFkpfm_tVs0", data_format=args.output_format)

        ff_kpfm_t0sv[0, :, :, :], ff_kpfm_t0sv[1, :, :, :] = rotate_ff(ff_kpfm_t0sv[0, :, :, :], ff_kpfm_t0sv[1, :, :, :], opt_dict["rotate"])
        ff_kpfm_tvs0[0, :, :, :], ff_kpfm_tvs0[1, :, :, :] = rotate_ff(ff_kpfm_tvs0[0, :, :, :], ff_kpfm_tvs0[1, :, :, :], opt_dict["rotate"])

        ff_kpfm_t0sv = ff_kpfm_t0sv * opt_dict["pol_s"]
        ff_kpfm_tvs0 = ff_kpfm_tvs0 * opt_dict["pol_t"]

    lvec[1, :] = rotate_vector(lvec[1, :], opt_dict["rotate"])
    lvec[2, :] = rotate_vector(lvec[2, :], opt_dict["rotate"])
    common.lvec2params(lvec)

    for charge, stiffness, voltage in it.product(charges, k_constants, voltages):
        common.params["charge"] = charge
        common.params["klat"] = stiffness
        common.params["Vbias"] = voltage

        dirname = f"Q{charge:1.2f}K{stiffness:1.2f}"
        if applied_bias:
            dirname += f"V{voltage:1.2f}"
        print(" Relaxed_scan for ", dirname)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        # Run relaxation
        fzs, pp_positions, pp_displacements, lvec_scan = perform_relaxation(
            lvec,
            ff_vdw,
            FFel=ff_electrostatics,
            FFpauli=ff_pauli,
            FFboltz=ff_boltzman,
            FFkpfm_t0sV=ff_kpfm_t0sv,
            FFkpfm_tVs0=ff_kpfm_tvs0,
            tipspline=args.tipspline,
            bFFtotDebug=args.bDebugFFtot,
        )

        data_info = {"lvec": lvec_scan, "data_format": args.output_format, "head": atomic_info_or_head, "atomic_info": atomic_info_or_head}
        if common.params["tiltedScan"]:
            io.save_vec_field(dirname + "/OutF", fzs, **data_info)
        else:
            io.save_scal_field(dirname + "/OutFz", fzs, **data_info)

        if opt_dict["vib"] >= 0:
            which = opt_dict["vib"]
            print(f" === Computing eigenvectors of dynamical matrix: which={which} ddisp={common.params['ddisp']}")
            tip_positions_x, tip_positions_y, tip_positions_z, lvec_scan = common.prepareScanGrids()
            r_tips = np.array(np.meshgrid(tip_positions_x, tip_positions_y, tip_positions_z)).transpose(3, 1, 2, 0).copy()
            evals, evecs = core.stiffnessMatrix(r_tips.reshape((-1, 3)), pp_positions.reshape((-1, 3)), which=which, ddisp=common.params["ddisp"])
            print("vib eigenval 1 min..max : ", np.min(evals[:, 0]), np.max(evals[:, 0]))
            print("vib eigenval 2 min..max : ", np.min(evals[:, 1]), np.max(evals[:, 1]))
            print("vib eigenval 3 min..max : ", np.min(evals[:, 2]), np.max(evals[:, 2]))
            io.save_vec_field(dirname + "/eigvalKs", evals.reshape(r_tips.shape), **data_info)
            if which > 0:
                io.save_vec_field(dirname + "/eigvecK1", evecs[0].reshape(r_tips.shape), **data_info)
            if which > 1:
                io.save_vec_field(dirname + "/eigvecK2", evecs[1].reshape(r_tips.shape), **data_info)
            if which > 2:
                io.save_vec_field(dirname + "/eigvecK3", evecs[2].reshape(r_tips.shape), **data_info)

        if opt_dict["disp"]:
            io.save_vec_field(dirname + "/PPdisp", pp_displacements, **data_info)

        if opt_dict["pos"]:
            io.save_vec_field(dirname + "/PPpos", pp_positions, **data_info)

        if args.bI:
            print("Calculating current from tip to the Boltzmann particle:")
            current_in, lvec, _, atomic_info_or_head = io.load_scal_field("I_boltzmann", data_format=args.output_format)
            current_out = interpolate_cartesian(current_in, pp_positions, cell=lvec[1:, :], result=None)
            io.save_scal_field(dirname + "/OutI_boltzmann", current_out, **data_info)


if __name__ == "__main__":
    main()
