#!/usr/bin/python
import sys
import numpy as np
import os
import __main__ as main


from   pyProbeParticle            import basUtils
from   pyProbeParticle            import elements   
import pyProbeParticle.HighLevel      as PPH
import pyProbeParticle.fieldFFT       as fFFT
import pyProbeParticle.GridUtils as GU
import pyProbeParticle  as PPU     

if __name__=="__main__":
    HELP_MSG="""Use this program in the following way:
    %s -i <filename> 
    
    Supported file fromats are:
       * xyz 
    """ %os.path.basename(main.__file__)
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option( "-i", "--input", action="store", type="string",
    help="Input file, supported formats are:\n.xyz\n.cube,.xsf")
    parser.add_option( "--noPBC", action="store_false",  help="pbc False", dest="PBC", default=None)
    parser.add_option( "-E", "--energy", action="store_true",  help="pbc False", default=False)
    parser.add_option("-f","--data_format" , action="store" , type="string",
                      help="Specify the output format of the vector and scalar "
                      "field. Supported formats are: xsf,npy", default="xsf")
    (options, args) = parser.parse_args()
    opt_dict = vars(options)
    print options
    if options.input==None:
        sys.exit("ERROR!!! Please, specify the input file with the '-i' option \n\n"+HELP_MSG)
    FFparams=None
    if os.path.isfile( 'atomtypes.ini' ):
    	print ">> LOADING LOCAL atomtypes.ini"  
    	FFparams=PPU.loadSpecies( 'atomtypes.ini' ) 
    else:
        import pyProbeParticle.cpp_utils as cpp_utils
    	FFparams = PPU.loadSpecies( cpp_utils.PACKAGE_PATH+'/defaults/atomtypes.ini' )
    print " >> OVEWRITING SETTINGS by params.ini  "
    PPU.loadParams( 'params.ini',FFparams=FFparams )
    print " >> APPLYING options to the SETTINGS"
    # TODO: introduce a class "Parameters", add a new function
    PPU.apply_options(opt_dict)
    atoms,nDim,lvec=basUtils.loadGeometry(options.input, params=PPU.params)
    # The function automatically load the geometry from the file of any
    # supported format. The desigion about the file format is based on the
    # filename extension
    PPU.params['gridN'] = nDim
    PPU.params['gridA'] = lvec[1]
    PPU.params['gridB'] = lvec[2]
    PPU.params['gridC'] = lvec[3]
    iZs,Rs,Qs=PPH.parseAtoms(atoms, autogeom = False, PBC = PPU.params['PBC'],
                             FFparams=FFparams )
    # The function returns the following information:
    # iZs - 1D array, containing the numbers of the elements, which corresponds to
    # their position in the atomtypes.ini file (Number of line - 1)
    # Rs  - 2D array, containing the coordinates of the atoms:
    #       [ [x1,y1,z1],
    #         [x2,y2,z2],
    #          ... 
    #         [xn,yn,zn]]
    # Qs  - 1D array, containing the atomic charges
    #FFLJ,VLJ=computeLJFF(iZs,Rs,FFparams,Fmax=10.0,computeVpot=options.energy,Vmax=10.0)
    FFLJ,VLJ=PPH.computeLJFF(iZs,Rs,FFparams)
    print "--- Save  ---"
    GU.save_vec_field( 'FFLJ', FFLJ, lvec,data_format=options.data_format)
    if options.energy :
        GU.save_scal_field( 'VLJ', VLJ, lvec,data_format=options.data_format)
