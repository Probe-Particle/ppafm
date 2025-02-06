#!/usr/bin/python

import gc

from .. import common
from ..HighLevel import computeELFF_pointCharge


def main(argv=None):
    parser = common.CLIParser(description="Generate electrostatic force field by Coulomb interaction of point charges. The generated force field is saved to FFel_{x,y,z}.[ext].")
    parser.add_arguments(["input", "input_format", "output_format", "tip", "energy", "noPBC"])
    args = parser.parse_args(argv)
    parameters = common.PpafmParameters.from_file("params.ini")
    parameters.apply_options(vars(args))

    computeELFF_pointCharge(args.input, geometry_format=args.input_format, tip=args.tip, save_format=args.output_format, computeVpot=args.energy, parameters=parameters)

    # Make sure that the energy and force field pointers are deleted so that they don't interfere if any other force fields are computed after this.
    gc.collect()


if __name__ == "__main__":
    main()
