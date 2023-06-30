#!/usr/bin/env python3

import sys
from argparse import ArgumentParser

import numpy as np

import ppafm as PPU
import ppafm.fieldFFT as fFFT
import ppafm.HighLevel as PPH
from ppafm import elements
from ppafm.defaults import d3


def main():

    parser = ArgumentParser(
        prog='generateDFTD3',
        description='Generate Grimme DFT-D3 vdW force field using the Becke-Johnson damping function. '
            'The generated force field is saved to FFvdW_{x,y,z}.[ext].'
    )

    parser.add_argument('-i', '--input',       action='store',
        help='Input file. Mandatory. Supported formats are: .xyz, .cube, .xsf.')
    parser.add_argument("--format", action="store", type="string", help="Format of the input geometry file (overrides format concluded from the file name extension)", default=None)
    parser.add_argument('-f', '--data_format', action='store', default='xsf',
        help='Specify the output format of the force field. Supported formats are: xsf, npy')
    parser.add_argument(      '--noPBC',       action='store_false', dest='PBC', default=None,
        help='No periodic boundary conditions.')
    parser.add_argument('-E', '--energy',      action='store_true', default=False,
        help='Compute potential energy in addition to force.')
    parser.add_argument(      '--df_name',     action='store', default='PBE',
        help='Which density functional-specific scaling parameters (s6, s8, a1, a2) to use. Give the name of the functional. '
            'The following functionals are available: PBE, B1B95, B2GPPLYP, B3PW91, BHLYP, BMK, BOP, BPBE, CAMB3LYP, LCwPBE, '
            'MPW1B95, MPWB1K, mPWLYP, OLYP, OPBE, oTPSS, PBE38, PBEsol, PTPSS, PWB6K, revSSB, SSB, TPSSh, HCTH120, B2PLYP, '
            'B3LYP, B97D, BLYP, BP86, DSDBLYP, PBE0, PBE, PW6B95, PWPB95, revPBE0, revPBE38, revPBE, rPW86PBE, TPSS0, TPSS.')
    parser.add_argument(      '--df_params',   action='store', default=None, nargs=4, type=float, metavar=('s6', 's8', 'a1', 'a2'),
        help='Manually specify scaling parameters s6, s8, a1, a2. Overwrites --df_name.')
    args = parser.parse_args()

    if args.input is None:
        parser.print_help()
        print('\nMissing input file (-i, --input)!\n')
        sys.exit(1)

    try:
        # Try overwriting global parameters with params.ini file
        PPU.loadParams('params.ini')
    except:
        print("No params.ini provided => using default parameters.")

    # Overwrite global parameters with command line arguments
    PPU.apply_options(vars(args))

    if args.df_params is not None:
        p = args.df_params
        df_params = {'s6': p[0], 's8': p[1], 'a1': p[2], 'a2': p[3]}
    else:
        if args.df_name not in d3.DF_DEFAULT_PARAMS:
            print(f'Unknown functional name `{args.df_name}`!')
            sys.exit(1)
        df_params = args.df_name

    PPH.computeDFTD3(args.input, df_params=df_params, geometry_format=args.format, save_format=args.data_format, compute_energy=args.energy)


if __name__ == "__main__":
    main()
