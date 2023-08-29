#!/usr/bin/python

import numpy as np

import ppafm as PPU
import ppafm.cpp_utils as cpp_utils
import ppafm.fieldFFT as fFFT
import ppafm.HighLevel as PPH
from ppafm import elements


def main():

    parser = PPU.CLIParser(
        description='Generate electrostatic force field by Coulomb interaction of point charges. '
            'The generated force field is saved to FFel_{x,y,z}.[ext].'
    )
    parser.add_arguments(['input', 'input_format', 'output_format', 'tip', 'energy', 'noPBC'])
    args = parser.parse_args()

    PPU.loadParams( 'params.ini' )
    PPU.apply_options(vars(args))

    PPH.computeELFF_pointCharge( args.input, geometry_format=args.input_format, tip=args.tip, save_format=args.output_format, computeVpot=args.energy )


if __name__ == "__main__":
    main()
