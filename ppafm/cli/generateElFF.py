#!/usr/bin/python
import sys
from pathlib import Path

import numpy as np

from .. import common, cpp_utils, io
from ..HighLevel import (
    computeElFF,
    getAtomsWhichTouchPBCcell,
    loadValenceElectronDict,
    subtractCoreDensities,
)


def main(argv=None):
    parser = common.CLIParser(
        description="Generate electrostatic force field by cross-correlation of sample Hartree potential with tip charge density. "
        "The generated force field is saved to FFel_{x,y,z}.[ext]."
    )

    # fmt: off
    parser.add_arguments(['input', 'input_format', 'output_format', 'tip', 'sigma', 'Rcore', 'energy', 'noPBC'])
    parser.add_argument("--tip_dens",   action="store", type=str,   default=None,  help="Use tip density from a file (.xsf or .cube). Overrides --tip.")
    parser.add_argument("--doDensity",  action="store_true",                       help="Do density overlap")
    parser.add_argument( "--tilt",      action="store", type=float, default=0,     help="Tilt of tip electrostatic field (radians)")
    parser.add_argument("--KPFM_tip",   action="store", type=str,   default='Fit', help="Read tip density under bias")
    parser.add_argument("--KPFM_sample",action="store", type=str,                  help="Read sample hartree under bias")
    parser.add_argument("--Vref",       action="store", type=float,                help="Field under the KPFM dens. and Vh was calculated in V/Ang")
    parser.add_argument("--z0",         action="store", type=float, default=0.0,   help="Heigth of the topmost layer of metallic substrate for E to V conversion (Ang)")
    # fmt: on
    args = parser.parse_args(argv)

    # Load parameters.
    params_path = Path("params.ini") if Path("params.ini").is_file() else cpp_utils.PACKAGE_PATH / "defaults" / "params.ini"
    common.loadParams(params_path)
    common.apply_options(vars(args))

    # Load species.
    species_path = Path("atomtypes.ini") if Path("atomtypes.ini").is_file() else cpp_utils.PACKAGE_PATH / "defaults" / "atomtypes.ini"
    common.loadSpecies(species_path)

    subtract_core_densities = (args.doDensity) and (args.Rcore > 0.0) and (args.tip_dens is not None)
    if subtract_core_densities:  # We do it here, in case it crash we don't want to wait for all the huge density files to load
        if args.tip_dens is None:
            raise Exception("Rcore>0 but no tip density provided!")
        valence_electrons_dictionary = loadValenceElectronDict()
        rs_tip, elems_tip = getAtomsWhichTouchPBCcell(args.tip_dens, Rcut=args.Rcore)

    atoms_samp, _, lvec_samp = io.loadGeometry(args.input, format=args.input_format, params=common.params)
    head_samp = io.primcoords2Xsf(atoms_samp[0], [atoms_samp[1], atoms_samp[2], atoms_samp[3]], lvec_samp)

    # Load electrostatic potential.
    loaders = {
        "xsf": io.loadXSF,
        "cube": io.loadCUBE,
    }
    input_format = args.input_format
    try:
        electrostatic_potential, lvec, n_dim, _ = loaders[input_format](args.input)
    except KeyError:
        input_format = args.input.split(".")[-1]
        electrostatic_potential, lvec, n_dim, _ = loaders[input_format](args.input)

    electrostatic_potential *= -1  # Unit conversion, energy to potential (eV -> V)

    # To fix.
    # if common.params["tip"] == ".py":
    #     # import tip
    #     exec(compile(open("tip.py", "rb").read(), "tip.py", "exec"))
    #     print(tipMultipole)
    #     common.params["tip"] = tipMultipole
    #     print("params['tip'] ", common.params["tip"])

    if args.tip_dens is not None:
        #  No need to renormalize: fieldFFT already works with density
        print(">>> Loading tip density from ", args.tip_dens, "...")
        if args.tip_dens.lower().endswith("xsf"):
            rho_tip, lvec_tip, _, head_tip = io.loadXSF(args.tip_dens)
        else:
            print(f'ERROR!!! Unknown or unsupported format of the tip density file "{args.tip_dens}"\n', file=sys.stderr)
            sys.exit(1)
        if subtract_core_densities:
            print(">>> subtracting core densities from rho_tip ... ")
            subtractCoreDensities(rho_tip, lvec_tip, elems=elems_tip, Rs=rs_tip, valElDict=valence_electrons_dictionary, Rcore=args.Rcore, head=head_tip)

        common.params["tip"] = -rho_tip  # Negative sign, because the electron density needs to be negative but the input density is positive

    if args.KPFM_sample is not None:
        sigma = common.params["sigma"]
        print(common.params["sigma"])
        if input_format == "xsf" and args.KPFM_sample.lower().endswith(".xsf"):
            v_ref_s = args.Vref
            print(">>> Loading Hartree potential under bias from ", args.KPFM_sample, "...")
            print("Use loadXSF")
            v_kpfm, lvec, n_dim, head = io.loadXSF(args.KPFM_sample)

        elif input_format == "cube" and args.KPFM_sample.lower().endswith(".cube"):
            v_ref_s = args.Vref
            print(">>> Loading Hartree potential under bias from ", args.KPFM_sample, "...")
            print("Use loadCUBE")
            v_kpfm, lvec, n_dim, head = io.loadCUBE(args.KPFM_sample)

        else:
            print(
                f'ERROR!!! Format of the "{args.KPFM_sample}" file with Hartree potential under bias is unknown or incompatible with the main input format, which is "{input_format}".\n',
                file=sys.stderr,
            )
            sys.exit(1)
        v_kpfm *= -1  # Unit conversion, energy to potential (eV -> V)
        dv_kpfm = v_kpfm - electrostatic_potential

        print(">>> Loading tip density under bias from ", args.KPFM_tip, "...")
        if input_format == "xsf" and args.KPFM_tip.lower().endswith(".xsf"):
            v_ref_t = args.Vref
            rho_tip_kpfm, lvec_tip, _, head_tip = io.loadXSF(args.KPFM_tip)
            drho_kpfm = rho_tip - rho_tip_kpfm  # Order of terms in the difference is swapped,
            # because the sign of rho_tip is as in *electron* density while drho_kpfm should be a difference of *charge* densities.
        elif input_format == "cube" and args.KPFM_tip.lower().endswith(".cube"):
            v_ref_t = args.Vref
            rho_tip_kpfm, lvec_tip, _, head_tip = io.loadCUBE(args.KPFM_tip, hartree=False, borh=args.borh)
            drho_kpfm = rho_tip - rho_tip_kpfm  # Order of terms in the difference is swapped,
            # because the sign of rho_tip is as in *electron* density while drho_kpfm should be a difference of *charge* densities.
        elif args.KPFM_tip in {"Fit", "fit", "dipole", "pz"}:  # To be put on a library in the near future...
            v_ref_t = -0.1
            if common.params["probeType"] == "8":
                drho_kpfm = {"pz": 0.045}
                sigma = 0.48
                print(" Select CO-tip polarization ")
            if common.params["probeType"] == "47":
                drho_kpfm = {"pz": 0.21875}
                sigma = 0.7
                print(" Select Ag polarization with decay sigma", sigma)
            if common.params["probeType"] == "54":
                drho_kpfm = {"pz": 0.250}
                sigma = 0.67
                print(" Select Xe-tip polarization")
        else:
            raise ValueError(
                'ERROR!!! Neither is "'
                + args.KPFM_sample
                + '" a density file with an appropriate ("'
                + input_format
                + '") format\nnor is it a valid name of a tip polarizability model.\n'
            )

        ff_kpfm_t0sv, _ = computeElFF(dv_kpfm, lvec, n_dim, common.params["tip"], computeVpot=args.energy, tilt=args.tilt)
        ff_kpfm_tvs0, _ = computeElFF(electrostatic_potential, lvec, n_dim, drho_kpfm, computeVpot=args.energy, tilt=args.tilt, sigma=sigma, deleteV=False)

        print("Linear E to V")
        zpos = np.linspace(lvec[0, 2] - args.z0, lvec[0, 2] + lvec[3, 2] - args.z0, n_dim[0])
        for i in range(n_dim[0]):
            # z position of the KPFM tip with respect to the sample must not be zero or negative
            # Should that happen, use periodicity in z to get zpos>0
            zpos[i] %= lvec[3, 2]

            ff_kpfm_t0sv[i, :, :] = ff_kpfm_t0sv[i, :, :] / ((v_ref_s) * (zpos[i] + 0.1))
            ff_kpfm_tvs0[i, :, :] = ff_kpfm_tvs0[i, :, :] / ((v_ref_t) * (zpos[i] + 0.1))

        print(">>> Saving electrostatic forcefield ... ")
        io.save_vec_field("FFkpfm_t0sV", ff_kpfm_t0sv, lvec_samp, data_format=args.output_format, head=head_samp)
        io.save_vec_field("FFkpfm_tVs0", ff_kpfm_tvs0, lvec_samp, data_format=args.output_format, head=head_samp)

    print(">>> Calculating electrostatic forcefield with FFT convolution as Eel(R) = Integral( rho_tip(r-R) V_sample(r) ) ... ")
    ff_electrostatic, e_electrostatic = computeElFF(electrostatic_potential, lvec, n_dim, common.params["tip"], computeVpot=args.energy, tilt=args.tilt)

    print(">>> Saving electrostatic forcefield ... ")

    io.save_vec_field("FFel", ff_electrostatic, lvec_samp, data_format=args.output_format, head=head_samp, atomic_info=(atoms_samp[:4], lvec_samp))
    if args.energy:
        io.save_scal_field("Eel", e_electrostatic, lvec_samp, data_format=args.output_format, head=head_samp, atomic_info=(atoms_samp[:4], lvec_samp))


if __name__ == "__main__":
    main()
