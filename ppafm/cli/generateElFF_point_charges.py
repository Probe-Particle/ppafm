#!/usr/bin/python
import sys
from optparse import OptionParser

import numpy as np

import ppafm as PPU
import ppafm.cpp_utils as cpp_utils
import ppafm.fieldFFT as fFFT
import ppafm.HighLevel as PPH
from ppafm import elements

HELP_MESSAGE = """Use this program in the following way:
ppafm-generate-elff-point-charges -i <filename> [ --sigma <value> ]
Supported file fromats are:
    * cube
    * xsf
"""

def main():
    parser = OptionParser()
    parser.add_option( "-i", "--input"      , action="store", type="string", help="format of input file")
    parser.add_option("-F", "--input_format", action="store", type="string", help="Format of the input geometry file (overrides format concluded from the file name extension)", default=None)
    parser.add_option( "-t", "--tip"        , action="store", type="string", help="tip model (multipole)", default='s')
    parser.add_option( "-E", "--energy"     , action="store_true",           help="pbc False",             default=False)
    parser.add_option( "--noPBC"            , action="store_false",          help="pbc False", dest="PBC", default=None)
    parser.add_option( "-f","--data_format" , action="store", type="string", help="Specify the output format of the vector and scalar field. Supported formats are: xsf,npy", default="xsf")
    (options, args) = parser.parse_args()
    if options.input==None:
        sys.exit("ERROR!!! Please, specify the input file with the '-i' option \n\n"+HELP_MESSAGE)
    opt_dict = vars(options)
    PPU.loadParams( 'params.ini' )
    PPU.apply_options(opt_dict)
    PPH.computeELFF_pointCharge( options.input, geometry_format=options.input_format, tip=options.tip, save_format=options.data_format, computeVpot=options.energy )

if __name__ == "__main__":
    main()
