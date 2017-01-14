#!/usr/bin/python
import sys
import numpy as np
import os
import __main__ as main


import pyProbeParticle                as PPU     
from   pyProbeParticle            import basUtils
from   pyProbeParticle            import elements   
import pyProbeParticle.GridUtils      as GU
import pyProbeParticle.HighLevel      as PPH
import pyProbeParticle.cpp_utils      as cpp_utils


if __name__=="__main__":
    HELP_MSG="""Use this program in the following way:
    """+os.path.basename(main.__file__) +""" -i <filename> [ --sigma <value> ]
    Supported file fromats are:
       * cube
       * xsf """
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option( "-i", "--input", action="store", type="string", help="format of input file")
    parser.add_option( "-t", "--tip", action="store", type="string", help="tip "
                      "model (multipole)", default=None)
    parser.add_option( "-E", "--energy", action="store_true",  help="pbc False", default=False)
    parser.add_option("--noPBC", action="store_false",  help="pbc False",
    dest="PBC", default=None)
    parser.add_option( "-w", "--sigma", action="store", type="float",
                      help="gaussian width for convolution in Electrostatics "
                      "[Angstroem]", default=None)
    parser.add_option("-f","--data_format" , action="store" , type="string",
                      help="Specify the output format of the vector and scalar "
                      "field. Supported formats are: xsf,npy", default="xsf")
    (options, args) = parser.parse_args()
    print options
    opt_dict = vars(options)
    
    if options.input==None:
        sys.exit("ERROR!!! Please, specify the input file with the '-i' option \n\n"+HELP_MSG)
    print " >> OVEWRITING SETTINGS by params.ini  "

    if os.path.isfile( 'atomtypes.ini' ):
        print ">> LOADING LOCAL atomtypes.ini"  
        FFparams=PPU.loadSpecies( 'atomtypes.ini' ) 
    else:
        FFparams = PPU.loadSpecies( cpp_utils.PACKAGE_PATH+'/defaults/atomtypes.ini' )

    PPU.loadParams( 'params.ini', FFparams=FFparams)
    PPU.apply_options(opt_dict)
    iZs,Rs,Qs=None,None,None
    V=None
    if(options.input.lower().endswith(".xsf") ):
        print " loading Hartree potential from disk "
        print "Use loadXSF"
        V, lvec, nDim, head = GU.loadXSF(options.input)
    elif(options.input.lower().endswith(".cube") ):
        print " loading Hartree potential from disk "
        print "Use loadCUBE"
        V, lvec, nDim, head = GU.loadCUBE(options.input)
    elif(options.input.lower().endswith(".xyz") ):
        atoms,nDim,lvec=basUtils.loadGeometry(options.input, params=PPU.params)
        iZs,Rs,Qs=PPH.parseAtoms(atoms, autogeom = False, PBC =PPU.params['PBC'],
                                 FFparams=FFparams )
    else:
        sys.exit("ERROR!!! Unknown format of the input file\n\n"+HELP_MSG)
    
    FFel=PPH.computeElFF(V,lvec,nDim,PPU.params['tip'],Fmax=10.0,computeVpot=options.energy,Vmax=10)
   
    print " saving electrostatic forcefiled "
    GU.save_vec_field('FFel',FFel,lvec,data_format=options.data_format)
    if options.energy :
        GU.save_scal_field( 'Vel', V, lvec, data_format=options.data_format)
    del FFel,V;
