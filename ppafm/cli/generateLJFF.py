#!/usr/bin/python
import os

import numpy as np

import ppafm as PPU
import ppafm.fieldFFT as fFFT
import ppafm.HighLevel as PPH
from ppafm import elements


def main():

    parser = PPU.CLIParser(
        description='Generate a Lennard-Jones, Morse, or vdW force field. '
            'The generated force field is saved to FFLJ_{x,y,z}.[ext].'
    )

    parser.add_arguments(['input', 'input_format', 'output_format', 'ffModel', 'energy', 'noPBC'])
    args = parser.parse_args()

    try:
        PPU.loadParams( 'params.ini' )
    except ValueError as e:
        print(e)
        print("no params.ini provided => using default params ")

    PPU.apply_options(vars(args))

    speciesFile = 'atomtypes.ini' if os.path.isfile('atomtypes.ini') else None

    PPH.computeLJ( args.input, geometry_format=args.input_format, speciesFile=speciesFile, save_format=args.output_format, computeVpot=args.energy, ffModel=args.ffModel )


if __name__ == "__main__":
    main()
