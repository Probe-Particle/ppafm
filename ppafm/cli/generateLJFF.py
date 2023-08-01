#!/usr/bin/python
import os
import sys

import numpy as np

import ppafm as PPU
import ppafm.fieldFFT as fFFT
import ppafm.HighLevel as PPH
from ppafm import elements

HELP_MESSAGE=f"""Use this program in the following way:
ppafm-generate-ljff -i <filename>

Supported file fromats are:
    * xyz
"""

def main():

    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-i", "--input",      action="store", type="string",  help="Input file, supported formats are:\n.xyz\n.cube,.xsf")
    parser.add_option("-F", "--input_format", action="store", type="string", help="Format of the input geometry file (overrides format concluded from the file name extension)", default=None)
    parser.add_option("-f","--data_format", action="store" , type="string", help="Specify the output format of the vector and scalar field. Supported formats are: xsf,npy", default="xsf")
    parser.add_option("--noPBC",            action="store_false",           help="pbc False", dest="PBC", default=None)
    parser.add_option("-E", "--energy",     action="store_true",            help="Compue potential energ y(not just Force)", default=False)
    parser.add_option("--ffModel",          action="store",                 help="kind of potential 'LJ','Morse','vdW' ", default='LJ')
    (options, args) = parser.parse_args()
    if options.input==None:
        sys.exit("ERROR!!! Please, specify the input file with the '-i' option \n\n"+HELP_MESSAGE)
    opt_dict = vars(options)
    try:
        PPU.loadParams( 'params.ini' )
    except Exception as e:
        print (e)
        print("no params.ini provided => using default params ")
    PPU.apply_options(opt_dict)
    speciesFile = None
    if os.path.isfile( 'atomtypes.ini' ):
        speciesFile='atomtypes.ini'
    PPH.computeLJ( options.input, geometry_format=opt_dict["input_format"],speciesFile=speciesFile, save_format=options.data_format, computeVpot=options.energy, ffModel=options.ffModel )


if __name__ == "__main__":
    main()
