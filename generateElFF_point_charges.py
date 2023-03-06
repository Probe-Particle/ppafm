#!/usr/bin/python
import os
import sys

import __main__ as main
import numpy as np

import ppafm as PPU
import ppafm.cpp_utils as cpp_utils
import ppafm.fieldFFT as fFFT
import ppafm.HighLevel as PPH
from ppafm import elements, io

if __name__=="__main__":
    HELP_MSG="""Use this program in the following way:
    """+os.path.basename(main.__file__) +""" -i <filename> [ --sigma <value> ]
    Supported file fromats are:
       * cube
       * xsf """
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option( "-i", "--input"      , action="store", type="string", help="format of input file")
    parser.add_option( "-t", "--tip"        , action="store", type="string", help="tip model (multipole)", default='s')
    parser.add_option( "-E", "--energy"     , action="store_true",           help="pbc False",             default=False)
    parser.add_option( "--noPBC"            , action="store_false",          help="pbc False", dest="PBC", default=None)
    parser.add_option( "-f","--data_format" , action="store", type="string", help="Specify the output format of the vector and scalar field. Supported formats are: xsf,npy", default="xsf")
    (options, args) = parser.parse_args()
    if options.input==None:
        sys.exit("ERROR!!! Please, specify the input file with the '-i' option \n\n"+HELP_MSG)
    opt_dict = vars(options)
    PPU.loadParams( 'params.ini' )
    PPU.apply_options(opt_dict)
    PPH.computeELFF_pointCharge( options.input, tip=options.tip, save_format=options.data_format, computeVpot=options.energy )
    '''
    iZs,Rs,Qs=None,None,None
    V=None
    atoms,nDim,lvec=io.loadGeometry(options.input, params=PPU.params)
    if os.path.isfile( 'atomtypes.ini' ):
        print ">> LOADING LOCAL atomtypes.ini"
        FFparams=PPU.loadSpecies( 'atomtypes.ini' )
    else:
        FFparams = PPU.loadSpecies( cpp_utils.PACKAGE_PATH+'/defaults/atomtypes.ini' )
    iZs,Rs,Qs=PPH.parseAtoms(atoms, autogeom = False, PBC =PPU.params['PBC'],FFparams=FFparams )
    FFel,V=PPH.computeELFF_pch(iZs,Rs,Qs,False, tip=options.tip)
    print " saving electrostatic forcefiled "
    io.save_vec_field('FFel',FFel,lvec,data_format=options.data_format)
    if options.energy :
        io.save_scal_field( 'Vel', V, lvec, data_format=options.data_format)
    del FFel,V;
    '''
