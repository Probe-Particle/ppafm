#!/usr/bin/env python3

import sys

from ppafm.defaults import d3

from .. import common
from ..HighLevel import computeDFTD3


def main():
    parser = common.CLIParser(
        description="Generate Grimme DFT-D3 vdW force field using the Becke-Johnson damping function. The generated force field is saved to FFvdW_{x,y,z}.[ext]."
    )

    parser.add_arguments(["input", "input_format", "output_format", "noPBC", "energy"])
    parser.add_argument(
        "--df_name",
        action="store",
        default="PBE",
        help="Which density functional-specific scaling parameters (s6, s8, a1, a2) to use. Give the name of the functional. "
        "The following functionals are available: PBE, B1B95, B2GPPLYP, B3PW91, BHLYP, BMK, BOP, BPBE, CAMB3LYP, LCwPBE, "
        "MPW1B95, MPWB1K, mPWLYP, OLYP, OPBE, oTPSS, PBE38, PBEsol, PTPSS, PWB6K, revSSB, SSB, TPSSh, HCTH120, B2PLYP, "
        "B3LYP, B97D, BLYP, BP86, DSDBLYP, PBE0, PBE, PW6B95, PWPB95, revPBE0, revPBE38, revPBE, rPW86PBE, TPSS0, TPSS.",
    )
    parser.add_argument(
        "--df_params",
        action="store",
        default=None,
        nargs=4,
        type=float,
        metavar=("s6", "s8", "a1", "a2"),
        help="Manually specify scaling parameters s6, s8, a1, a2. Overwrites --df_name.",
    )
    args = parser.parse_args()

    try:
        # Try overwriting global parameters with params.ini file.
        common.loadParams("params.ini")
    except Exception:
        print("No params.ini provided => using default parameters.")

    # Overwrite global parameters with command line arguments.
    common.apply_options(vars(args))

    if args.df_params is not None:
        p = args.df_params
        df_params = {"s6": p[0], "s8": p[1], "a1": p[2], "a2": p[3]}
    else:
        if args.df_name not in d3.DF_DEFAULT_PARAMS:
            print(f"Unknown functional name `{args.df_name}`!")
            sys.exit(1)
        df_params = args.df_name

    computeDFTD3(args.input, df_params=df_params, geometry_format=args.input_format, save_format=args.output_format, compute_energy=args.energy)


if __name__ == "__main__":
    main()
