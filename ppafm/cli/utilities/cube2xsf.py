#!/usr/bin/python
import os
import sys

import __main__ as main
import numpy as np

import ppafm as PPU
import ppafm.cpp_utils as cpp_utils
import ppafm.HighLevel as PPH
from ppafm import io

if __name__=="__main__":
    HELP_MSG="""Use this program in the following way:
    """+os.path.basename(main.__file__) +""" -i <filename> [ --sigma <value> ]
    Supported file fromats are:
       * cube
       * xsf """
    from optparse import OptionParser
    parser = OptionParser()
    # fmt: off
    parser.add_option( "-i", "--input",  action="store", type="string", help="input file")
    parser.add_option( "-o", "--output", action="store", type="string", help="output file")
    # fmt: on
    (options, args) = parser.parse_args()
    if options.input==None:
        sys.exit("ERROR!!! Please, specify the input file with the '-i' option \n\n"+HELP_MSG)


    FFparams            = PPU.loadSpecies()
    elem_dict           = PPU.getFFdict(FFparams); # print elem_dict

    atoms,nDim,lvec     = io .loadGeometry( options.input, params=PPU.params )

    data, lvec, nDim, head = io.loadCUBE(options.input)
    lvec[0] = [0.0, 0.0, 0.0]
    lvec = np.array( [lvec[0], lvec[3], lvec[2], lvec[1]]  )
    data = np.transpose(  data, (2,1,0) )

    atomstring          = io.primcoords2Xsf( PPU.atoms2iZs( atoms[0],elem_dict ), [atoms[1],atoms[2],atoms[3]], lvec );

    io.saveXSF( options.output, data, lvec, head=atomstring  )
